"""Publish a verified Round7 increment only when root explicitly passes --execute.

Existing credentials are resolved by the approved local helper, never printed
or placed in arguments. Old commits, tags, releases and assets are preserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from round7_content_index import identities, file_digest
from verify_round7_increment import verify
from prepare_round7_delivery import Round7Exporter
from prepare_round5_delivery import secret_scan

REPOSITORY = 'https://github.com/zhouz4884-coder/TrajKV'
API = 'https://api.github.com/repos/zhouz4884-coder/TrajKV'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def rel(value):
    text = str(value).replace('\\', '/')
    return text.split('/TrajKV/', 1)[1] if '/TrajKV/' in text else text


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def release_identity(value):
    return dict(id=value['id'], tag_name=value['tag_name'], target_commitish=value['target_commitish'],
        assets=sorted([dict(id=x['id'], name=x['name'], size=x['size'], digest=x.get('digest'))
                       for x in value['assets']], key=lambda x: x['id']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--checkout', type=Path, default=ROOT.parent / 'TrajKV_public_20261114')
    p.add_argument('--receipt', type=Path, default=ROOT / 'audit/publication/GITHUB_ROUND7_DELIVERY.json')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(); assets, checkout = a.assets.resolve(), a.checkout.resolve()
    delivery = load(assets / 'DELIVERY.json'); validation = load(assets / 'VERIFICATION.json')
    require(validation.get('status') == 'PASS', 'Independently verified assets required')
    require(validation['archive_sha256'] == delivery['archive_sha256'], 'Verification receipt refers to another archive')
    current_validation = verify(assets)
    require(current_validation['status'] == 'PASS', 'Prepared public bytes/numerics changed')
    index = load(assets / 'ROUND7_SOURCE_INDEX.json'); content = load(assets / 'ROUND7_CONTENT_REFS.json')
    require(not index.get('CPU_fixture_only'), 'CPU acceptance fixtures are not authorized publication packages')
    plan = index['publication_plan']; tag = plan['tag_name']
    require(tag.startswith('research-round7-') and '/' not in tag and '..' not in tag, 'Use a new Round7 tag')
    if not a.execute:
        print(json.dumps(dict(status='READY_FOR_ROOT_EXECUTION', repository=REPOSITORY, tag=tag,
            files=sum(x['expected_in_git'] for x in content['files'].values()),
            force_add_metadata_paths=content['git_stage_plan']['force_paths'], mutation_performed=False)))
        return
    require(not a.receipt.exists(), 'Completed publication receipt already exists; inspect instead of overwriting')
    require((checkout / '.git').exists(), 'Use the existing public Git checkout')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    def git(*argv, binary=False):
        last = None
        for attempt in range(3):
            prefix = ['git'] if attempt == 0 else ['git', '-c', 'http.version=HTTP/1.1']
            result = subprocess.run([*prefix, *argv], cwd=checkout, env=env, capture_output=True, timeout=180)
            if result.returncode == 0:
                return result.stdout if binary else result.stdout.decode('utf-8', errors='replace').strip()
            last = result.stderr.decode('utf-8', errors='replace')[-1200:]
            if argv[0] not in ('push', 'ls-remote', 'fetch'):
                break
            time.sleep(1 + attempt)
        raise RuntimeError('Git operation failed: ' + last)
    def tree_map(revision=None):
        output = (git('ls-tree', '-r', '-z', '--full-tree', revision, binary=True) if revision
                  else git('ls-files', '--stage', '-z', binary=True))
        mapping = {}
        for row in output.split(b'\0'):
            if row:
                metadata, name = row.split(b'\t', 1)
                fields = metadata.split()
                mapping[name.decode('utf-8')] = fields[2 if revision else 1].decode()
        return mapping
    def verify_blob_bodies(expected_paths, expected_bodies):
        oids = list(dict.fromkeys(expected_paths.values()))
        result = subprocess.run(['git', 'cat-file', '--batch'], cwd=checkout, env=env,
            input=('\n'.join(oids) + '\n').encode(), capture_output=True, timeout=180)
        require(result.returncode == 0, 'Cannot read actual Git blob bodies')
        payload, offset, actual = result.stdout, 0, {}
        for oid in oids:
            end = payload.find(b'\n', offset); require(end >= 0, 'Incomplete Git batch object header')
            header = payload[offset:end].split(); require(len(header) == 3 and header[0].decode() == oid and header[1] == b'blob', 'Unexpected Git object identity/type')
            size = int(header[2]); start = end + 1; body = payload[start:start + size]
            require(len(body) == size and payload[start + size:start + size + 1] == b'\n', 'Incomplete Git blob body')
            actual[oid] = hashlib.sha256(body).hexdigest(); offset = start + size + 1
        require(all(actual[oid] == expected_bodies[name] for name, oid in expected_paths.items()),
                'Actual normalized Git body SHA256 differs from exact archive/Git identity contract')
    require(git('remote', 'get-url', 'origin') == REPOSITORY + '.git', 'Wrong publication repository')
    require(git('symbolic-ref', '--short', 'HEAD') == 'main', 'Use the intended main branch in the public checkout')
    require(git('rev-parse', '--show-object-format') == 'sha1', 'This content index pins SHA1 Git blob OIDs')
    journal_path = assets / 'PUBLISH_JOURNAL.json'
    journal = load(journal_path) if journal_path.exists() else None
    from check_github_access import session
    client = session()
    def request(method, url, **kwargs):
        import requests
        timeout = kwargs.pop('timeout', 45)
        data = kwargs.get('data'); origin = data.tell() if hasattr(data, 'tell') else None
        response = None
        for attempt in range(5):
            if origin is not None:
                data.seek(origin)
            try:
                response = client.request(method, url, timeout=timeout, **kwargs)
            except requests.RequestException:
                if attempt == 4:
                    raise RuntimeError('GitHub request transport failed after safe retries') from None
                time.sleep(min(2 ** attempt, 8)); continue
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(min(2 ** attempt, 8))
        return response
    def releases():
        values = {}
        for page in range(1, 100):
            response = request('GET', API + '/releases', params={'per_page': 100, 'page': page}); response.raise_for_status()
            batch = response.json()
            for value in batch:
                values[value['tag_name']] = release_identity(value)
            if len(batch) < 100:
                return values
        raise RuntimeError('Release pagination exceeded bound')
    try:
        user = request('GET', 'https://api.github.com/user'); user.raise_for_status()
        require(user.json()['login'] == 'zhouz4884-coder', 'GitHub authorization belongs to another account')
        repo = request('GET', API); repo.raise_for_status()
        require(repo.json()['private'] is False and repo.json()['permissions']['push'], 'Authorized public push repository required')
        head = git('rev-parse', 'HEAD')
        remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
        if journal is None:
            require(not git('status', '--porcelain'), 'Preserve existing checkout changes before publishing')
            require(head == remote, 'Remote moved; reconcile without overwriting divergent history')
            previous_releases = releases()
            require(tag not in previous_releases, 'Target tag/release already exists; never modify an old release')
            journal = dict(schema='trajkv.round7.publication-journal.v1', archive_sha256=delivery['archive_sha256'],
                prior_head=head, prior_tree=git('rev-parse', 'HEAD^{tree}'), prior_releases=previous_releases,
                prior_tags=git('ls-remote', '--tags', 'origin'), step='installing')
            write_json(journal_path, journal)
        else:
            require(journal['archive_sha256'] == delivery['archive_sha256'], 'Journal belongs to another content package')
            require(remote in (journal['prior_head'], journal.get('head')), 'Remote advanced outside this publication; reconcile safely')
        release_url = REPOSITORY + '/releases/tag/' + tag
        expected = {name: item['predicted_git_blob_oid_sha1'] for name, item in content['files'].items() if item['expected_in_git']}
        expected_bodies = {name: item['git_normalized_body_sha256'] for name, item in content['files'].items() if item['expected_in_git']}
        installed = list(expected)
        if journal['step'] == 'committing':
            current = git('rev-parse', 'HEAD')
            if current == journal['prior_head']:
                git('commit', '-m', 'research: publish Round7 frozen source confirmation and evidence')
            else:
                require(git('rev-parse', 'HEAD^') == journal['prior_head']
                        and git('rev-parse', 'HEAD^{tree}') == journal['staged_tree'],
                        'Interrupted commit differs from journal; preserve and inspect it')
            journal.update(step='committed', head=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
            write_json(journal_path, journal)
        if journal['step'] == 'installing':
            require(git('rev-parse', 'HEAD') == journal['prior_head'], 'Checkout HEAD changed during installation')
            for name in installed:
                source = assets / 'payload' / name
                require(file_digest(source) == content['files'][name]['archive_exact_sha256'], 'Payload changed')
                target = checkout / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            prior_readme = git('show', journal['prior_head'] + ':README.md', binary=True).decode('utf-8-sig')
            intro = (assets / 'payload' / rel(plan['readme_entry_file'])).read_text(encoding='utf-8-sig').strip()
            intro += ('\n\nStrict `encoder_book_sha256` checks use the exact parameter bytes from the '
                '[release archive](' + release_url + '). Git normalizes text to LF; the '
                '[fixed content index](evidence/ROUND7_CONTENT_REFS.json) lists archive exact SHA256, '
                'Git body SHA256 and Git blob OID separately. Do not replace a frozen expected SHA '
                'with the hash of Git-rendered text. The final release receipt binds this index to the verified commit.')
            (checkout / 'README.md').write_text(intro + '\n\n---\n\n' + prior_readme, encoding='utf-8', newline='\n')
            evidence = checkout / 'evidence'; evidence.mkdir(exist_ok=True)
            for name in ('ROUND7_SOURCE_INDEX.json', 'ROUND7_CONTENT_REFS.json'):
                shutil.copy2(assets / name, evidence / name)
            public_delivery = dict(status='CONTENT_PACKAGE_VERIFIED_LOCALLY', stage='Round7', research_execution_completed=True,
                github_sync_pending_at_preparation=True, execution_goal_completed_at_preparation=False,
                source_index_sha256=delivery['source_index_sha256'], content_index_sha256=delivery['content_index_sha256'],
                archive=dict(name=delivery['archive'], sha256=delivery['archive_sha256'], bytes=delivery['archive_bytes'], release=release_url),
                final_report=rel(plan['final_report']), final_remote_receipt='release asset round7_delivery_receipt.json',
                scope='Execution completion is separate from performance hypotheses; final report contains measured outcomes and limitations.')
            write_json(evidence / 'ROUND7_DELIVERY.json', public_delivery)
            try:
                ready = json.loads(git('show', journal['prior_head'] + ':PUBLICATION_READY.json', binary=True))
            except RuntimeError:
                ready = {}
            ready['round7'] = public_delivery; write_json(checkout / 'PUBLICATION_READY.json', ready)
            exported = json.loads(git('show', journal['prior_head'] + ':PUBLIC_EXPORT_MANIFEST.json', binary=True))
            for name in installed:
                exported['files'][name] = index['files'][name]
            generated = ['README.md', 'evidence/ROUND7_SOURCE_INDEX.json', 'evidence/ROUND7_CONTENT_REFS.json',
                         'evidence/ROUND7_DELIVERY.json', 'PUBLICATION_READY.json']
            for name in generated:
                path = checkout / name; data = path.read_bytes()
                exported['files'][name] = dict(source_sha256=hashlib.sha256(data).hexdigest(), public_sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data), role='generated Round7 public entry/content metadata')
            exported.setdefault('stage_updates', {})['round7'] = dict(source_index_sha256=delivery['source_index_sha256'],
                content_index_sha256=delivery['content_index_sha256'], release=release_url)
            exported.update(file_count=len(exported['files']), total_bytes=sum(item['bytes'] for item in exported['files'].values()))
            write_json(checkout / 'PUBLIC_EXPORT_MANIFEST.json', exported)
            generated.append('PUBLIC_EXPORT_MANIFEST.json')
            for name in generated:
                identity = identities((checkout / name).read_bytes(), name)
                expected[name] = identity['predicted_git_blob_oid_sha1']
                expected_bodies[name] = identity['git_normalized_body_sha256']
            normal = [name for name in expected if name not in content['git_stage_plan']['force_paths']]
            for start in range(0, len(normal), 50):
                git('add', '-f', '--', *normal[start:start + 50])
            for name in content['git_stage_plan']['force_paths']:
                git('add', '-f', '--', name)
            staged = tree_map()
            require(all(staged.get(name) == oid for name, oid in expected.items()), 'Actual staged Git tree omitted/changed expected paths, including ignored metadata')
            journal.update(step='committing', staged_tree=git('write-tree'), expected_git_paths=expected,
                expected_git_body_sha256=expected_bodies)
            write_json(journal_path, journal)
            git('commit', '-m', 'research: publish Round7 frozen source confirmation and evidence')
            journal.update(step='committed', head=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
            write_json(journal_path, journal)
        else:
            expected = journal['expected_git_paths']
            expected_bodies = journal['expected_git_body_sha256']
            require(git('rev-parse', 'HEAD') == journal['head'] and not git('status', '--porcelain'), 'Preserve unexpected post-commit checkout changes')
        local_tree = tree_map(journal['head'])
        require(all(local_tree.get(name) == oid for name, oid in expected.items()), 'Actual committed Git tree differs from expected content')
        verify_blob_bodies(expected, expected_bodies)
        require(set(tree_map(journal['prior_head'])) <= set(local_tree), 'Historical public files were deleted')
        git('merge-base', '--is-ancestor', journal['prior_head'], journal['head'])
        if remote != journal['head']:
            git('push', 'origin', 'main')
        journal['step'] = 'pushed'; write_json(journal_path, journal)
        response = request('GET', API + '/releases/tags/' + tag)
        if response.status_code == 404:
            response = request('POST', API + '/releases', json=dict(tag_name=tag, target_commitish=journal['head'],
                name=plan['release_title'], body=plan['release_body'], draft=False, prerelease=plan.get('prerelease', True)))
        response.raise_for_status(); release = response.json()
        def upload(path, name):
            asset = next((x for x in release['assets'] if x['name'] == name), None)
            if asset is None:
                with path.open('rb') as stream:
                    response = request('POST', release['upload_url'].split('{')[0], params={'name': name},
                        headers={'Content-Type': 'application/zip' if path.suffix == '.zip' else 'application/json'}, data=stream, timeout=300)
                response.raise_for_status(); asset = response.json()
            require(asset['size'] == path.stat().st_size, 'Existing/uploaded asset has different size; never overwrite it')
            expected_digest = 'sha256:' + file_digest(path)
            if asset.get('digest') != expected_digest:
                import requests
                h = hashlib.sha256()
                with requests.get(asset['browser_download_url'], stream=True, timeout=(30, 120)) as fetched:
                    require(fetched.status_code == 200, 'Anonymous public asset download failed HTTP ' + str(fetched.status_code))
                    for block in fetched.iter_content(8 * 1024 * 1024):
                        h.update(block)
                require('sha256:' + h.hexdigest() == expected_digest, 'Existing/uploaded remote asset differs; never overwrite it')
            return dict(name=name, bytes=asset['size'], sha256=file_digest(path), url=asset['browser_download_url'], server_digest=asset.get('digest'))
        uploaded = [upload(assets / delivery['archive'], delivery['archive']),
                    upload(assets / 'ROUND7_SOURCE_INDEX.json', 'round7_source_index.json'),
                    upload(assets / 'ROUND7_CONTENT_REFS.json', 'round7_content_refs.json')]
        remote_commit = request('GET', API + '/git/commits/' + journal['head']); remote_commit.raise_for_status()
        require(remote_commit.json()['tree']['sha'] == journal['tree'], 'Remote commit/tree differs')
        remote_head = request('GET', API + '/git/ref/heads/main'); remote_head.raise_for_status()
        require(remote_head.json()['object']['sha'] == journal['head'], 'Remote main differs')
        remote_tree = request('GET', API + '/git/trees/' + journal['tree'], params={'recursive': 1}); remote_tree.raise_for_status()
        require(not remote_tree.json().get('truncated'), 'Remote tree response truncated')
        remote_paths = {x['path']: x['sha'] for x in remote_tree.json()['tree'] if x['type'] == 'blob'}
        require(all(remote_paths.get(name) == oid for name, oid in expected.items()), 'Remote actual tree omitted/changed expected paths')
        old_releases = releases()
        require(all(old_releases.get(name) == value for name, value in journal['prior_releases'].items()), 'Historical release identity changed')
        old_tags = dict(line.split('\t')[::-1] for line in journal['prior_tags'].splitlines() if line)
        new_tags = dict(line.split('\t')[::-1] for line in git('ls-remote', '--tags', 'origin').splitlines() if line)
        require(all(new_tags.get(name) == value for name, value in old_tags.items()), 'Historical Git tag changed')
        target = new_tags.get('refs/tags/' + tag + '^{}', new_tags.get('refs/tags/' + tag))
        require(target == journal['head'], 'New release tag does not identify the verified commit')
        tree_path = assets / 'ACTUAL_GIT_TREE.json'; write_json(tree_path, remote_paths)
        final_validation = verify(assets, tree_path)
        require(not git('status', '--porcelain'), 'Publication checkout has uncommitted changes')
        import requests
        anonymous = requests.get(REPOSITORY, timeout=45)
        require(anonymous.status_code == 200, 'Anonymous repository access did not pass')
        raw_index_url = ('https://raw.githubusercontent.com/zhouz4884-coder/TrajKV/' + journal['head']
                         + '/evidence/ROUND7_CONTENT_REFS.json')
        public_index = requests.get(raw_index_url, timeout=45)
        require(public_index.status_code == 200
                and hashlib.sha256(public_index.content).hexdigest() == expected_bodies['evidence/ROUND7_CONTENT_REFS.json'],
                'Anonymous fixed-commit content index differs from actual Git body')
        receipt = dict(status='PASS', verified_at_utc=datetime.now(timezone.utc).isoformat(), repository=REPOSITORY,
            prior_head=journal['prior_head'], head=journal['head'], tree=journal['tree'], release=release['html_url'],
            assets=uploaded, actual_remote_tree_paths_verified=len(expected), configs_data_force_paths=content['git_stage_plan']['force_paths'],
            exact_archive_and_LF_Git_identities_verified=True, prior_releases_preserved=len(journal['prior_releases']),
            actual_committed_blob_body_SHA256_verified=True, remote_tree_points_to_verified_blob_oids=True,
            anonymous_repository_http_status=anonymous.status_code,
            anonymous_fixed_commit_content_index_body_sha256=hashlib.sha256(public_index.content).hexdigest(),
            research_execution_completed=True, github_sync_completed=True, global_goal_completion_left_to_root=True,
            committed_content_index=REPOSITORY + '/blob/' + journal['head'] + '/evidence/ROUND7_CONTENT_REFS.json',
            content_ref_binding=dict(commit=journal['head'], index_blob_oid=expected['evidence/ROUND7_CONTENT_REFS.json'],
                index_git_body_sha256=expected_bodies['evidence/ROUND7_CONTENT_REFS.json'], archive_sha256=delivery['archive_sha256']),
            prior_tags_preserved=True, validation=final_validation,
            final_report=REPOSITORY + '/blob/' + journal['head'] + '/' + rel(plan['final_report']))
        private_receipt = assets / 'FINAL_REMOTE_RECEIPT.private.json'
        if private_receipt.exists():
            previous = load(private_receipt)
            require(previous['head'] == receipt['head'] and previous['tree'] == receipt['tree']
                    and previous['validation']['archive_sha256'] == delivery['archive_sha256'],
                    'Existing final receipt describes different content')
            receipt = previous
        else:
            write_json(private_receipt, receipt)
        # The local source receipt remains untouched. Public receipt derivation
        # is explicit; metadata paths may change, execution/artifact hashes do not.
        exporter = Round7Exporter(checkout)
        public_receipt = exporter.clean_object(receipt)
        public_receipt['receipt_derivation'] = dict(source_receipt_sha256=file_digest(private_receipt),
            scope='derived public metadata with machine paths/host/auth fields redacted; artifact and codec expected hashes are not rewritten')
        prepared_receipt = assets / 'FINAL_REMOTE_RECEIPT.json'
        encoded = (json.dumps(public_receipt, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        secret_scan(encoded, 'round7_delivery_receipt.json')
        require(not any(marker in encoded.decode('utf-8') for marker in ('/media/user/', '/path/to/research', '/path/to/research', 'C:/Users/RESEARCHER')),
                'Private machine path survived public receipt derivation')
        if prepared_receipt.exists():
            require(prepared_receipt.read_bytes() == encoded, 'Previously fixed public receipt bytes differ')
        else:
            prepared_receipt.write_bytes(encoded)
        receipt_asset = upload(prepared_receipt, 'round7_delivery_receipt.json')
        receipt = dict(public_receipt, public_receipt_asset=receipt_asset,
            private_source_receipt_sha256=file_digest(private_receipt), public_receipt_sha256=file_digest(prepared_receipt))
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        with a.receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False)
        journal['step'] = 'COMPLETE'; write_json(journal_path, journal)
        print(json.dumps(dict(status='PASS', head=receipt['head'], tree=receipt['tree'], release=receipt['release'],
            assets=len(uploaded) + 1, actual_tree_paths_verified=len(expected), receipt=str(a.receipt))))
    finally:
        client.close()


if __name__ == '__main__':
    main()
