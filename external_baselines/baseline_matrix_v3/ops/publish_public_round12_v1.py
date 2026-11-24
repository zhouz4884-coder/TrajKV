"""Incremental Round12 delivery, with a local-only default and remote replay.

The Root-sealed explicit inventory is the only publication input. Credentials
are obtained by the existing noninteractive helper in process memory only.
This module never discovers extra project files to publish.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

from round12_public_core_v1 import NAMESPACE, require, sha
from verify_public_round12_v1 import lf_identity, verify

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('${EVIDENCE_ROOT}').resolve()
REPOSITORY = 'https://github.com/zhouz4884-coder/TrajKV'
API = 'https://api.github.com/repos/zhouz4884-coder/TrajKV'
ROUND11_ANCESTOR = '28eb91cdb0d8f9de16946b33529da660c02c8988'
REVIEWED_FILES = ('ops/round12_public_core_v1.py', 'ops/verify_public_round12_v1.py',
                  'ops/round12_public_security_v1.py', 'ops/round12_public_legacy_v1.py',
                  'ops/export_public.py')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def write_journal(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def longpath(path):
    path = Path(path).resolve()
    prefix = chr(92) * 2 + '?' + chr(92)
    return Path(prefix + str(path)) if os.name == 'nt' and not str(path).startswith(prefix) else path


def browser_entries(index, index_path):
    entries = {NAMESPACE + '/' + rel: entry for rel, entry in index['files'].items()
               if entry['expected_in_git']}
    entries[NAMESPACE + '/PUBLIC_CONTENT_REFS.json'] = {'git_lf': lf_identity(index_path)}
    require(set(entries) == set(index['delivery_channels']['browser_paths']),
            'actual browser paths differ from the sealed channel inventory')
    return entries


def preserved_tree(prior, current):
    """Every old path survives; frozen results/configs retain exact blob OIDs."""
    require(set(prior) <= set(current), 'An earlier public tree path disappeared')
    protected = (NAMESPACE + '/results/', NAMESPACE + '/configs/')
    for path, oid in prior.items():
        if not path.startswith(NAMESPACE + '/') or path.startswith(protected):
            require(current[path] == oid, 'An earlier frozen public blob changed: ' + path)


def prepare(stage, expected_base):
    stage = Path(stage).resolve()
    require(stage.is_relative_to(EVIDENCE / 'public_staging'), 'stage must use the named evidence disk')
    require(re.fullmatch('[0-9a-f]{40}', expected_base) is not None, 'explicit reviewed base commit required')
    delivery = load(stage / 'DELIVERY.json')
    require(delivery.get('schema') == 'trajkv.round12.public_delivery.v1'
            and delivery.get('status') == 'PREPARED_NOT_PUBLISHED'
            and delivery.get('namespace') == NAMESPACE
            and delivery.get('publication_eligible') is True
            and delivery.get('fixture_only') is False
            and delivery.get('cold_verification_status') == 'PASS_PUBLIC_ROUND12_EXPORT_VERIFIED_V1'
            and delivery.get('preserve_prior_namespace_release_tag_history') is True,
            'nonfixture Root-sealed Round12 production export required')
    tag, archive_name = delivery['release_tag'], delivery['archive']
    require(re.fullmatch('[A-Za-z0-9_.-]+', tag) and Path(archive_name).name == archive_name,
            'unsafe fixed tag/archive name')
    archive = stage / archive_name
    public = longpath(stage / 'tree' / NAMESPACE)
    index_path = public / 'PUBLIC_CONTENT_REFS.json'
    require(sha(archive) == delivery['archive_sha256']
            and sha(index_path) == delivery['content_index_sha256'], 'prepared payload identity changed')
    index = load(index_path)
    require(index.get('schema') == 'trajkv.round12.public_content_refs.v1'
            and index.get('fixture_only') is False and index.get('publication_eligible') is True
            and index.get('release_tag') == tag, 'exact production index required')
    reviewed = {rel: sha(ROOT / rel) for rel in REVIEWED_FILES}
    require(delivery['core_verifier_sha256'] == reviewed[REVIEWED_FILES[0]]
            and delivery['cold_verifier_sha256'] == reviewed[REVIEWED_FILES[1]]
            and delivery.get('portable_verifier_sha256') == reviewed,
            'current reviewed verifier/core differs from export epoch')
    for rel, digest in reviewed.items():
        entry = index['files'][rel]
        require(entry['public_sha256'] == digest
                and sha(public / rel) == digest, 'packaged reviewed verifier/core differs: ' + rel)
    cold = verify(public, archive)
    require(cold.get('status') == 'PASS_PUBLIC_ROUND12_EXPORT_VERIFIED_V1'
            and cold.get('publication_eligible') is True and cold.get('archive_verified') is True
            and cold.get('actual_producer_binding_required') is True
            and cold.get('private_tensors_or_GPU_replayed') is False
            and cold.get('whole_external_library_completion_claimed') is False,
            'full actual producer/source/metric/archive cold verification required')
    require(reviewed == {rel: sha(ROOT / rel) for rel in REVIEWED_FILES},
            'reviewed source changed during cold verification')
    entries = browser_entries(index, index_path)
    require(len(entries) == delivery['expected_git_payload_files'], 'delivery browser count differs')
    return {'stage': stage, 'delivery': delivery, 'index': index, 'archive': archive,
            'public': public, 'index_path': index_path, 'entries': entries,
            'reviewed': reviewed, 'cold': cold, 'expected_base': expected_base}


class Git:
    def __init__(self, checkout):
        self.checkout = Path(checkout).resolve()
        self.env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')

    def __call__(self, *argv, binary=False, cwd=None):
        transient = any(v in argv for v in ('fetch', 'push', 'ls-remote'))
        for attempt in range(4 if transient else 1):
            command = [shutil.which('git'), '-c', 'http.version=HTTP/1.1', '-c', 'core.longpaths=true', *argv]
            try:
                run = subprocess.run(command, cwd=cwd or self.checkout, env=self.env,
                                     capture_output=True, timeout=600)
            except subprocess.TimeoutExpired:
                raise RuntimeError('This Git operation timed out; inspect its own preserved state before retry') from None
            if run.returncode == 0:
                return run.stdout if binary else run.stdout.decode('utf-8', errors='replace').strip()
            if attempt + 1 < (4 if transient else 1):
                time.sleep(min(attempt + 1, 5))
        # Error bodies are intentionally not echoed: they may contain helper diagnostics.
        raise RuntimeError('Git operation failed: ' + str(argv[0]) + ' (exit ' + str(run.returncode) + ')')

    def tree(self, revision=None):
        body = self('ls-tree', '-r', '-z', '--full-tree', revision, binary=True) if revision else self('ls-files', '--stage', '-z', binary=True)
        result = {}
        for row in body.split(b'\0'):
            if row:
                metadata, name = row.split(b'\t', 1)
                result[name.decode('utf-8')] = metadata.split()[2 if revision else 1].decode()
        return result

    def verify_bodies(self, entries, mapping):
        require(set(entries) <= set(mapping), 'actual Git tree omitted expected browser files')
        oids = list(dict.fromkeys(mapping[path] for path in entries))
        run = subprocess.run(['git', 'cat-file', '--batch'], cwd=self.checkout, env=self.env,
                             input=('\n'.join(oids) + '\n').encode(), capture_output=True, timeout=300)
        require(run.returncode == 0, 'Cannot read actual staged Git blob bodies')
        offset, bodies = 0, {}
        for oid in oids:
            stop = run.stdout.find(b'\n', offset)
            require(stop >= 0, 'Incomplete Git object header')
            header = run.stdout[offset:stop].split()
            require(len(header) == 3 and header[:2] == [oid.encode(), b'blob'], 'Unexpected Git object kind')
            size = int(header[2]); start = stop + 1
            body = run.stdout[start:start + size]
            require(len(body) == size and run.stdout[start + size:start + size + 1] == b'\n', 'Incomplete Git object body')
            bodies[oid] = hashlib.sha256(body).hexdigest(); offset = start + size + 1
        require(offset == len(run.stdout), 'Trailing Git object bytes')
        for path, entry in entries.items():
            identity = entry['git_lf']
            require(mapping[path] == identity['blob_oid_sha1']
                    and bodies[mapping[path]] == identity['sha256'], 'actual Git LF/body identity differs: ' + path)


class GitHub:
    def __init__(self):
        # Existing approved route: no token in command arguments or persistent files.
        location = ROOT.parent / 'TrajKV/code/check_github_access.py'
        spec = importlib.util.spec_from_file_location('round12_existing_github_route', location)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        self.client = helper.session()

    def request(self, method, url, **kwargs):
        timeout = kwargs.pop('timeout', (30, 120)); stream = kwargs.get('data')
        origin = stream.tell() if hasattr(stream, 'tell') else None
        for attempt in range(5):
            if origin is not None:
                stream.seek(origin)
            try:
                response = self.client.request(method, url, timeout=timeout, **kwargs)
            except Exception:
                if attempt == 4:
                    raise RuntimeError('GitHub transport failed after bounded retries') from None
                time.sleep(min(2 ** attempt, 8)); continue
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
            response.close(); time.sleep(min(2 ** attempt, 8))
        raise RuntimeError('GitHub server remained unavailable after bounded retries')

    def doc(self, method, url, codes=(200,), **kwargs):
        response = self.request(method, url, **kwargs)
        try:
            require(response.status_code in codes, 'GitHub status ' + str(response.status_code))
            return response.json()
        finally:
            response.close()

    def identity(self):
        user = self.doc('GET', 'https://api.github.com/user')
        repo = self.doc('GET', API)
        require(user['login'] == 'zhouz4884-coder' and repo['full_name'] == 'zhouz4884-coder/TrajKV'
                and repo['private'] is False and repo['permissions']['push'] is True,
                'approved public account/repository/push identity differs')
        return user

    def releases(self):
        values, page = [], 1
        while True:
            rows = self.doc('GET', API + '/releases', params={'per_page': 100, 'page': page})
            values.extend(rows)
            if len(rows) < 100:
                break
            page += 1
        return {str(row['id']): {'tag': row['tag_name'], 'target': row['target_commitish'],
                'assets': sorted([{'id': a['id'], 'name': a['name'], 'bytes': a['size'],
                                  'digest': a.get('digest')} for a in row['assets']], key=lambda a: a['id'])}
                for row in values}

    def tags(self):
        # Matching refs is paginated by some GitHub installations; use /tags for
        # pagination, then exact Git ref objects and peel annotated tags.
        values, page = {}, 1
        while True:
            rows = self.doc('GET', API + '/tags', params={'per_page': 100, 'page': page})
            for row in rows:
                name = row['name']
                from urllib.parse import quote
                obj = self.doc('GET', API + '/git/ref/tags/' + quote(name, safe=''))['object']
                raw = dict(obj); seen = set()
                while obj['type'] == 'tag':
                    require(obj['sha'] not in seen, 'cyclic annotated tag')
                    seen.add(obj['sha'])
                    obj = self.doc('GET', API + '/git/tags/' + obj['sha'])['object']
                values[name] = {'object': raw, 'peeled': obj}
            if len(rows) < 100:
                break
            page += 1
        return values

    def download(self, url, target, expected_sha=None, accept=None):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), 'new independent download path required')
        for attempt in range(4):
            temporary = target.with_name(target.name + '.partial_' + str(time.time_ns()))
            try:
                if url.startswith('https://codeload.github.com/'):
                    # Public source bodies need no account header on this host.
                    import requests
                    response = requests.get(url, stream=True, timeout=(30, 180))
                else:
                    response = self.request('GET', url, stream=True,
                                            headers={'Accept': accept} if accept else {}, timeout=(30, 180))
                with response:
                    require(response.status_code == 200, 'download status ' + str(response.status_code))
                    with temporary.open('xb') as out:
                        for chunk in response.iter_content(1024 ** 2):
                            if chunk:
                                out.write(chunk)
                if expected_sha:
                    require(sha(temporary) == expected_sha, 'downloaded body SHA differs')
                os.replace(temporary, target)
                return target
            except ValueError:
                # An identity/status mismatch is evidence, not a transient
                # transport failure to conceal by replacing the same file.
                raise
            except Exception:
                if attempt == 3:
                    raise RuntimeError('Independent download failed after bounded retries; partial files preserved') from None
                time.sleep(min(2 ** attempt, 8))


def verify_remote(ctx, journal, github, git):
    entries, stage = ctx['entries'], ctx['stage']
    commit, tree = journal['commit'], journal['tree']
    require(github.doc('GET', API + '/git/ref/heads/main')['object']['sha'] == commit,
            'remote main differs from intended stage commit')
    remote_commit = github.doc('GET', API + '/git/commits/' + commit)
    require(remote_commit['tree']['sha'] == tree and remote_commit['parents'][0]['sha'] == journal['prior_head'],
            'remote commit/tree/parent differs')
    remote_tree = github.doc('GET', API + '/git/trees/' + tree, params={'recursive': 1})
    require(not remote_tree.get('truncated'), 'remote tree truncated')
    mapping = {row['path']: row['sha'] for row in remote_tree['tree'] if row['type'] == 'blob'}
    preserved_tree(journal['prior_tree'], mapping)
    require(all(mapping.get(path) == info['git_lf']['blob_oid_sha1'] for path, info in entries.items()),
            'remote actual tree omitted/changed a browser file')
    verification_root = EVIDENCE / 'remote_verification' / ('round12_' + str(time.time_ns()))
    source_url = 'https://codeload.github.com/zhouz4884-coder/TrajKV/zip/' + commit
    source_zip = github.download(source_url, verification_root / 'fixed_commit.zip')
    with zipfile.ZipFile(source_zip) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'duplicate fixed-commit ZIP member')
        roots = {name.split('/', 1)[0] for name in names if '/' in name}
        require(len(roots) == 1, 'ambiguous fixed-commit ZIP root')
        prefix = next(iter(roots)) + '/'
        members = set(names)
        for path, info in entries.items():
            require(prefix + path in members, 'fixed-commit ZIP omitted body: ' + path)
            body = z.read(prefix + path)
            oid = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
            require(oid == mapping[path] == info['git_lf']['blob_oid_sha1']
                    and hashlib.sha256(body).hexdigest() == info['git_lf']['sha256'],
                    'independently refetched Git body differs: ' + path)
    tag = ctx['delivery']['release_tag']
    release = github.doc('GET', API + '/releases/tags/' + tag)
    tags = github.tags()
    require(tag in tags and tags[tag]['peeled']['sha'] == commit,
            'new release actual tag does not peel to intended commit')
    require(release['target_commitish'] == commit and release['draft'] is False,
            'new public release target differs')
    assets_by_name = {x['name']: x for x in release['assets']}
    require(len(assets_by_name) == len(release['assets']), 'ambiguous release asset names')
    assets = []
    for path in (ctx['archive'], ctx['index_path'], stage / 'DELIVERY.json'):
        require(path.name in assets_by_name, 'release asset missing: ' + path.name)
        item = assets_by_name[path.name]; digest = sha(path)
        require(item['size'] == path.stat().st_size, 'release asset size differs')
        if item.get('digest'):
            require(item['digest'] == 'sha256:' + digest, 'server release digest differs')
        downloaded = github.download(item['url'], verification_root / path.name, digest, 'application/octet-stream')
        assets.append({'name': path.name, 'bytes': downloaded.stat().st_size,
                       'sha256': digest, 'server_digest': item.get('digest'),
                       'url': item['browser_download_url']})
    # Full original ZIP is downloaded and checked, not only its metadata/index.
    downloaded_archive = verification_root / ctx['archive'].name
    require(sha(downloaded_archive) == ctx['delivery']['archive_sha256'], 'downloaded raw-evidence archive differs')
    with zipfile.ZipFile(downloaded_archive) as z:
        expected = set(ctx['index']['delivery_channels']['archive_paths'])
        require(len(z.namelist()) == len(expected) and set(z.namelist()) == expected,
                'downloaded archive actual member set differs')
        for rel, item in ctx['index']['files'].items():
            h = hashlib.sha256()
            with z.open(NAMESPACE + '/' + rel) as src:
                for part in iter(lambda: src.read(1024 ** 2), b''):
                    h.update(part)
            require(h.hexdigest() == item['public_sha256'], 'downloaded archive member differs: ' + rel)
    final_releases = github.releases()
    require(all(final_releases.get(key) == value for key, value in journal['prior_releases'].items()),
            'earlier release/assets state changed')
    require(all(tags.get(key) == value for key, value in journal['prior_tags'].items()),
            'earlier raw annotated tag/peeled target changed')
    git('merge-base', '--is-ancestor', journal['prior_head'], commit)
    require(github.doc('GET', API + '/git/ref/heads/main')['object']['sha'] == commit,
            'remote main changed during independent verification')
    return {'status': 'PASS_GITHUB_ROUND12_INCREMENT_DELIVERED_V1', 'repository': REPOSITORY,
            'head': commit, 'tree': tree, 'prior_head': journal['prior_head'],
            'namespace': NAMESPACE, 'release': release['html_url'], 'assets': assets,
            'actual_remote_tree_paths_verified': len(entries),
            'all_expected_blob_bodies_independently_refetched': True,
            'independent_refetch': {'url': source_url, 'source_archive_sha256': sha(source_zip),
                                  'verification_directory': str(verification_root),
                                  'bodies_verified': len(entries), 'local_staging_used_as_download_source': False},
            'prior_tree_paths_preserved': len(journal['prior_tree']),
            'prior_frozen_results_configs_and_other_namespaces_unchanged': True,
            'prior_releases_preserved': len(journal['prior_releases']),
            'prior_tag_objects_and_peeled_targets_preserved': len(journal['prior_tags']),
            'counts': ctx['cold']['counts'], 'local_cold_status': ctx['cold']['status'],
            'reviewed_verifier_sources': ctx['reviewed'], 'publisher_sha256': sha(__file__),
            'content_ref': {'commit': commit, 'index_blob_oid': entries[NAMESPACE + '/PUBLIC_CONTENT_REFS.json']['git_lf']['blob_oid_sha1'],
                            'index_public_sha256': ctx['delivery']['content_index_sha256'],
                            'archive_sha256': ctx['delivery']['archive_sha256']},
            'browser_and_archive_delivery_verified_separately': True,
            'completion_scope': 'root_sealed_research_stage_only',
            'whole_external_library_completion_claimed': False, 'SourceRun_admission_claimed': False,
            'private_tensors_or_GPU_replayed': False, 'verified_utc': datetime.now(timezone.utc).isoformat()}


def publish(ctx, checkout, receipt, *, verify_only=False):
    checkout = Path(checkout).resolve()
    require(checkout.is_relative_to(EVIDENCE), 'public checkout must use the named evidence disk')
    require((checkout / '.git').exists(), 'existing reviewed public checkout required; do not bootstrap blindly')
    require(not Path(receipt).exists(), 'new immutable final receipt required')
    git, github = Git(checkout), GitHub()
    try:
        user = github.identity()
        require(git('remote', 'get-url', 'origin') == REPOSITORY + '.git'
                and git('symbolic-ref', '--short', 'HEAD') == 'main', 'reviewed origin/main checkout required')
        stage, delivery = ctx['stage'], ctx['delivery']
        journal_path = stage / 'PUBLISH_ROUND12_JOURNAL_V1.json'
        journal = load(journal_path) if journal_path.exists() else None
        if journal is None:
            require(not verify_only, 'no existing journal/commit to verify')
            require(not git('status', '--porcelain'), 'public checkout contains unrelated changes')
            tag = delivery['release_tag']
            require(tag not in github.tags(), 'new tag already exists; never overwrite')
            check = github.request('GET', API + '/releases/tags/' + tag)
            require(check.status_code == 404, 'new release already exists; never overwrite'); check.close()
            git('fetch', 'origin', 'main'); git('merge', '--ff-only', 'origin/main')
            prior_head = git('rev-parse', 'HEAD')
            require(prior_head == ctx['expected_base'], 'reviewed expected base differs from current remote/local head')
            git('merge-base', '--is-ancestor', ROUND11_ANCESTOR, prior_head)
            releases = github.releases()
            require(len(releases) >= 17, 'all17 prior releases must be present')
            journal = {'schema': 'trajkv.round12.publish_journal.v1', 'prior_head': prior_head,
                       'prior_tree': git.tree(prior_head), 'prior_releases': releases, 'prior_tags': github.tags(),
                       'namespace': NAMESPACE, 'release_tag': tag, 'archive_sha256': delivery['archive_sha256'],
                       'index_sha256': delivery['content_index_sha256'], 'reviewed': ctx['reviewed'],
                       'publisher_sha256': sha(__file__), 'commit': None, 'tree': None}
            write_journal(journal_path, journal)
        require(journal['schema'] == 'trajkv.round12.publish_journal.v1'
                and journal['prior_head'] == ctx['expected_base'] and journal['namespace'] == NAMESPACE
                and journal['release_tag'] == delivery['release_tag'] and journal['reviewed'] == ctx['reviewed']
                and journal['publisher_sha256'] == sha(__file__)
                and journal['archive_sha256'] == delivery['archive_sha256']
                and journal['index_sha256'] == delivery['content_index_sha256'], 'journal source/payload/base epoch differs')
        if verify_only:
            require(journal['commit'] and journal['tree'], 'no committed stage to verify')
        elif journal['commit'] is None:
            require(git('rev-parse', 'HEAD') == journal['prior_head'], 'checkout moved before stage commit')
            for path in ctx['entries']:
                relative = path[len(NAMESPACE) + 1:]
                src, dst = ctx['public'] / relative, longpath(checkout / path)
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            pathspec = stage / 'GIT_PATHSPEC_ROUND12.nul'
            pathspec.write_bytes(b''.join(path.encode() + b'\0' for path in ctx['entries']))
            git('add', '-f', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul')
            changed = {p.decode() for p in git('diff', '--cached', '--name-only', '-z', binary=True).split(b'\0') if p}
            require(changed and changed <= set(ctx['entries']), 'unexpected or empty staged file set')
            staged = git.tree(); preserved_tree(journal['prior_tree'], staged)
            git.verify_bodies(ctx['entries'], staged)
            message = stage / 'ROUND12_COMMIT_MESSAGE.txt'
            message.write_text('Add Round12 frozen research and fixed500 comparison evidence\n\n'
                               'Keep development, fixed500 benchmark comparisons, oracle/QA0 and pilots separate.\n'
                               'Publish the Root-sealed explicit source, metrics, complete budgets and raw-evidence archive.\n'
                               'Preserve previous frozen results, configurations, releases and history.\n', encoding='utf-8', newline='\n')
            git('-c', 'user.name=zhouz4884-coder', '-c', 'user.email=269925009+zhouz4884-coder@users.noreply.github.com',
                'commit', '--file', str(message))
            journal.update(commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
            write_journal(journal_path, journal)
        require(git('rev-parse', 'HEAD') == journal['commit'], 'checkout diverged from journal commit')
        git.verify_bodies(ctx['entries'], git.tree(journal['commit']))
        preserved_tree(journal['prior_tree'], git.tree(journal['commit']))
        if not verify_only:
            remote_head = github.doc('GET', API + '/git/ref/heads/main')['object']['sha']
            require(remote_head in (journal['prior_head'], journal['commit']), 'remote changed concurrently')
            releases = github.releases()
            require(all(releases.get(k) == v for k, v in journal['prior_releases'].items()), 'prior releases changed before push')
            if remote_head != journal['commit']:
                git('push', 'origin', 'HEAD:main')
            require(github.doc('GET', API + '/git/ref/heads/main')['object']['sha'] == journal['commit'],
                    'remote main did not reach intended commit')
            response = github.request('GET', API + '/releases/tags/' + journal['release_tag'])
            if response.status_code == 404:
                response.close()
                counts = ctx['cold']['counts']
                release = github.doc('POST', API + '/releases', codes=(201,), json={
                    'tag_name': journal['release_tag'], 'target_commitish': journal['commit'],
                    'name': 'Round12 frozen development and fixed500 comparisons', 'draft': False, 'prerelease': False,
                    'body': 'Root-sealed research cutoff. Counts: ' + json.dumps(counts, sort_keys=True) + '\n\n'
                            'Development and previously observed fixed500 benchmark comparisons are distinguished. '
                            'Fresh terminal answers, reused references, pilots, native-reference oracles and QA0 are separate. '
                            'Includes reproducible text/scalar results, full physical-byte budgets, code, configs and raw-evidence archive. '
                            'No fresh independent benchmark, full external-library completion, private-tensor replay or controlled speed claim. '
                            'Earlier results, configurations, releases, assets, tags and commit history are preserved.'})
            else:
                require(response.status_code == 200, 'release lookup failed')
                release = response.json(); response.close()
            require(release['target_commitish'] == journal['commit'], 'release target differs; never replace')
            upload = release['upload_url'].split('{', 1)[0]
            existing = {a['name']: a for a in release['assets']}
            for path in (ctx['archive'], ctx['index_path'], stage / 'DELIVERY.json'):
                if path.name in existing:
                    require(existing[path.name]['size'] == path.stat().st_size,
                            'existing release asset differs; never overwrite')
                else:
                    require(github.doc('GET', API + '/git/ref/heads/main')['object']['sha'] == journal['commit'],
                            'remote head changed before asset upload')
                    with path.open('rb') as source:
                        github.doc('POST', upload, codes=(201,), params={'name': path.name}, data=source,
                                   headers={'Content-Type': 'application/octet-stream'}, timeout=(30, 600))
        result = verify_remote(ctx, journal, github, git)
        result['account'] = user['login']; result['visibility'] = 'public'
        write_new(receipt, result)
        journal['status'] = 'VERIFIED_COMPLETE'; write_journal(journal_path, journal)
        return result
    finally:
        github.client.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage-dir', required=True, type=Path)
    p.add_argument('--expected-base', required=True)
    p.add_argument('--checkout', type=Path, default=EVIDENCE / 'public_checkout')
    p.add_argument('--receipt', type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--verify-remote', action='store_true')
    a = p.parse_args(argv)
    ctx = prepare(a.stage_dir, a.expected_base)
    if not (a.execute or a.verify_remote):
        print(json.dumps({'status': 'READY_FOR_ROOT_ROUND12_EXECUTION',
                          'expected_git_payload_files': len(ctx['entries']),
                          'archive_sha256': ctx['delivery']['archive_sha256'],
                          'release_tag': ctx['delivery']['release_tag'],
                          'counts': ctx['cold']['counts'], 'authentication_accessed': False,
                          'network_calls': 0, 'GitHub_mutation': False}, ensure_ascii=False))
        return 0
    require(a.receipt is not None, 'explicit new final receipt path required')
    result = publish(ctx, a.checkout, a.receipt, verify_only=a.verify_remote)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
