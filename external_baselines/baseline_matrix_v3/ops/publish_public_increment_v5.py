"""Publish one reviewed export using ordinary history-preserving Git operations.

Default is a local dry run. --execute uses the pre-existing credential manager
in memory; credentials never enter arguments, files, URLs, reports or stdout.
No old release/tag/asset is deleted or modified. Every resume uses a journal.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from verify_public_export_v3 import verify

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('${EVIDENCE_ROOT}').resolve()
REPOSITORY = 'https://github.com/zhouz4884-coder/TrajKV'
API = 'https://api.github.com/repos/zhouz4884-coder/TrajKV'
BASELINE = '5e9f4f19051b0fc2db5b1c55b19776d690b322d9'
P1_ANCHOR = '9171b357eb610387c09ac50014510a8b9349da8d'


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def io_path(value):
    path=Path(value).resolve();prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(path)) if os.name=='nt' and not str(path).startswith(prefix) else path
def require(condition, reason):
    if not condition: raise ValueError(reason)
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def session():
    location = ROOT.parent / 'TrajKV/code/check_github_access.py'
    spec = importlib.util.spec_from_file_location('matrix_existing_github_credential_route', location)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper.session()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, default=EVIDENCE / 'public_checkout')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--expected-base', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); stage = args.stage_dir.resolve(); checkout = args.checkout.resolve()
    require(stage.is_relative_to(EVIDENCE / 'public_staging'), 'Prepared stage must be on the named E evidence volume')
    require(checkout.is_relative_to(EVIDENCE), 'Public checkout must use the named E evidence volume')
    delivery = load(stage / 'DELIVERY.json'); namespace = delivery['namespace']
    public = io_path(stage / 'tree' / namespace); index_path = public / 'PUBLIC_CONTENT_REFS.json'; index = load(index_path)
    archive = stage / delivery['archive']; tag = delivery['release_tag']
    require(sha(archive) == delivery['archive_sha256'], 'Prepared archive identity changed')
    require(sha(index_path) == delivery['content_index_sha256'], 'Prepared index identity changed')
    require(delivery.get('schema') == 'baseline_matrix.public_delivery.v3' and
            delivery.get('publication_eligible') is True and
            delivery['phase']['status'] == 'PASS_DECLARED_MATRIX_COMPLETE',
            'Preview/incomplete declared phase cannot be published')
    require(delivery['completion_scope'] == 'declared_matrix_only', 'Publisher does not certify whole-library completion')
    label,scope=delivery['phase']['phase_label'],delivery['phase']['coverage_scope']
    if args.receipt is None:args.receipt=ROOT/'audit/publication'/('GITHUB_'+label+'_DELIVERY.json')
    result = verify(public, archive, recompute_scores=True)
    require(result['status'] == 'PASS_PUBLIC_EXPORT_VERIFIED_V3' and result['source_closure_verified'] is True and
            result['publication_eligible'] is True, 'Cold matrix and donor-closure verification required')
    entries = {f'{namespace}/{relative}': info for relative, info in index['files'].items() if info['expected_in_git']}
    index_raw = index_path.read_bytes().replace(b'\r\n', b'\n')
    entries[f'{namespace}/PUBLIC_CONTENT_REFS.json'] = {'git_lf': {
        'sha256': hashlib.sha256(index_raw).hexdigest(),
        'blob_oid_sha1': hashlib.sha1(f'blob {len(index_raw)}\0'.encode() + index_raw).hexdigest()}}
    if not args.execute:
        print(json.dumps({'status': 'READY_FOR_ROOT_EXECUTION', 'namespace': namespace, 'tag': tag,
                          'expected_git_files': len(entries), 'archive_sha256': delivery['archive_sha256'],
                          'content_index_sha256': delivery['content_index_sha256'], 'github_mutation_performed': False}))
        return
    require(not args.receipt.exists(), 'Final receipt already exists; inspect instead of publishing twice')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    def git(*argv, cwd=checkout, binary=False):
        error = None
        for attempt in range(4):
            command=[shutil.which('git'),'-c','http.version=HTTP/1.1','-c','core.longpaths=true',*argv]
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:stdout,stderr=process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                # Verify the exact parent argv and every descendant's exe,
                # full argv classification and ancestry BEFORE any tree kill.
                # A failed check leaves it for explicit inspection, never kills
                # an unrelated Git/SSH/GPU process.
                if os.name=='nt':
                    query=r'''$ids=@(ROOTPID);$rows=@();while($ids.Count){$next=@();foreach($p in $ids){$r=Get-CimInstance Win32_Process -Filter "ProcessId=$p";if($r){$rows+=$r};$c=Get-CimInstance Win32_Process -Filter "ParentProcessId=$p";$next+=@($c|Select-Object -ExpandProperty ProcessId)};$ids=$next};$rows|Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine|ConvertTo-Json -Compress'''.replace('ROOTPID',str(process.pid))
                    observed=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,text=True,timeout=30)
                    require(observed.returncode==0,'Cannot inspect timed-out Git process tree')
                    rows=json.loads(observed.stdout);rows=rows if isinstance(rows,list) else [rows]
                    root_row=next((r for r in rows if r['ProcessId']==process.pid),None)
                    require(root_row is not None and root_row['CommandLine']==subprocess.list2cmdline(command),
                            'Timed-out Git root argv changed; manual scoped inspection required')
                    git_root=Path(command[0]).resolve().parents[1]
                    for row in rows:
                        require(row['Name'].lower() in ('git.exe','git-remote-https.exe'), 'Unrecognized Git descendant; do not terminate')
                        require(Path(row['ExecutablePath']).resolve().is_relative_to(git_root),'Git descendant executable outside expected installation')
                        text=row['CommandLine'] or ''
                        require('authorization' not in text.lower() and 'http.extraheader' not in text.lower(),'Unexpected credential-bearing Git argv')
                        require(row['ProcessId']==process.pid or REPOSITORY in text or any(v in text for v in ('index-pack','fetch','pack-objects','unpack-objects','remote-https','rev-list')),
                                'Git descendant argv does not match this operation')
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=20,check=True)
                    write(stage/f'GIT_TIMEOUT_TREE_{process.pid}.json',{'operation':argv[0],'root_pid':process.pid,
                        'verified_and_stopped':[{'pid':r['ProcessId'],'parent':r['ParentProcessId'],'exe':r['Name'],
                            'full_argv_sha256':hashlib.sha256(r['CommandLine'].encode()).hexdigest()} for r in rows],
                        'partial_directories_preserved':True})
                else:process.kill()
                try:stdout,stderr=process.communicate(timeout=15)
                except subprocess.TimeoutExpired:raise RuntimeError('Git output pipe still held after scoped cleanup') from None
                error='Git operation timed out after scoped cleanup'
                if not any(v in argv for v in ('fetch','push','ls-remote')):break
                time.sleep(2+attempt);continue
            run=subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
            if run.returncode == 0: return run.stdout if binary else run.stdout.decode('utf-8', errors='replace').strip()
            error = run.stderr.decode('utf-8', errors='replace')[-1200:]
            # A failed clone may leave a useful partial object database. Never
            # blindly repeat clone into the same now-nonempty directory.
            if not any(command in argv for command in ('fetch', 'push', 'ls-remote')): break
            time.sleep(1 + attempt)
        raise RuntimeError('Git operation failed: ' + error)
    def tree_map(revision=None, cwd=checkout):
        body = git('ls-tree', '-r', '-z', '--full-tree', revision, cwd=cwd, binary=True) if revision else git('ls-files', '--stage', '-z', cwd=cwd, binary=True)
        result = {}
        for row in body.split(b'\0'):
            if row:
                metadata, name = row.split(b'\t', 1)
                result[name.decode()] = metadata.split()[2 if revision else 1].decode()
        return result
    def verify_bodies(mapping, cwd=checkout):
        oids = list(dict.fromkeys(mapping.values()))
        run = subprocess.run(['git', 'cat-file', '--batch'], cwd=cwd, env=env,
                             input=('\n'.join(oids) + '\n').encode(), capture_output=True, timeout=300)
        require(run.returncode == 0, 'Cannot read actual Git blob bodies')
        offset = 0; bodies = {}
        for oid in oids:
            stop = run.stdout.find(b'\n', offset); require(stop >= 0, 'Incomplete Git object header')
            header = run.stdout[offset:stop].split(); require(len(header) == 3 and header[:2] == [oid.encode(), b'blob'], 'Unexpected Git object')
            count = int(header[2]); start = stop + 1; body = run.stdout[start:start + count]
            require(len(body) == count and run.stdout[start + count:start + count + 1] == b'\n', 'Incomplete Git object body')
            bodies[oid] = hashlib.sha256(body).hexdigest(); offset = start + count + 1
        for path, oid in mapping.items():
            require(oid == entries[path]['git_lf']['blob_oid_sha1'], 'Git blob OID differs: ' + path)
            require(bodies[oid] == entries[path]['git_lf']['sha256'], 'Git blob body differs: ' + path)
    client = session()
    def request(method, url, **kwargs):
        timeout = kwargs.pop('timeout', 60); stream = kwargs.get('data')
        origin = stream.tell() if hasattr(stream, 'tell') else None
        for attempt in range(5):
            if origin is not None: stream.seek(origin)
            try: response = client.request(method, url, timeout=timeout, **kwargs)
            except Exception:
                if attempt == 4: raise RuntimeError('GitHub request transport failed after retries') from None
                time.sleep(min(2 ** attempt, 8)); continue
            if response.status_code not in (429, 500, 502, 503, 504): return response
            time.sleep(min(2 ** attempt, 8))
        return response
    def ok(response, codes=(200,)):
        require(response.status_code in codes, f'GitHub returned status {response.status_code}')
        return response.json()
    def release_snapshot():
        values = []; page = 1
        while True:
            rows = ok(request('GET', API + '/releases', params={'per_page': 100, 'page': page}))
            values.extend(rows)
            if len(rows) < 100: break
            page += 1
        return {str(row['id']): {'tag': row['tag_name'], 'target': row['target_commitish'],
                'assets': sorted([{'id': a['id'], 'name': a['name'], 'bytes': a['size'], 'digest': a.get('digest')} for a in row['assets']], key=lambda a: a['id'])}
                for row in values}
    try:
        user = ok(request('GET', 'https://api.github.com/user'))
        repo = ok(request('GET', API))
        require(user['login'] == 'zhouz4884-coder' and repo['full_name'] == 'zhouz4884-coder/TrajKV', 'Wrong GitHub account/repository')
        require(not repo['private'] and repo['permissions']['push'], 'Expected public push permission missing')
        if not (checkout / '.git').exists():
            require(not checkout.exists() or not any(checkout.iterdir()), 'Clone target is nonempty')
            checkout.parent.mkdir(parents=True, exist_ok=True)
            git('clone', '--filter=blob:none', '--no-checkout', REPOSITORY + '.git', str(checkout), cwd=checkout.parent)
            git('config', 'core.sparseCheckout', 'true')
            git('config', 'core.sparseCheckoutCone', 'false')
            sparse = checkout / '.git/info/sparse-checkout'
            sparse.parent.mkdir(parents=True, exist_ok=True)
            sparse.write_text('/' + namespace + '/\n', encoding='utf-8')
            git('read-tree', '-mu', 'HEAD')
        require(git('remote', 'get-url', 'origin') == REPOSITORY + '.git', 'Wrong Git origin')
        require(git('symbolic-ref', '--short', 'HEAD') == 'main', 'Expected main checkout')
        journal_path = stage / 'PUBLISH_JOURNAL_V5.json'; journal = load(journal_path) if journal_path.exists() else None
        predecessor=stage/'PUBLISH_JOURNAL_V4.json'
        if not predecessor.exists():predecessor=stage/'PUBLISH_JOURNAL.json'
        if journal is None and predecessor.exists():
            old=load(predecessor)
            require(old['archive_sha256']==delivery['archive_sha256'] and old['index_sha256']==delivery['content_index_sha256'] and old.get('commit'),
                    'Predecessor journal is not the reviewed already-committed payload')
            journal={**old,'transport_version':4,'predecessor_journal_sha256':sha(predecessor),
                     'predecessor_journal_left_unchanged':True}
            write(journal_path,journal)
        if journal is None:
            require(request('GET', API + '/releases/tags/' + tag).status_code == 404,
                    'Intended release tag already exists; use a new reviewed increment, never overwrite')
            require(not git('ls-remote', '--tags', 'origin', 'refs/tags/' + tag),
                    'Intended Git tag already exists; never replace it')
            require(not git('status', '--porcelain'), 'Public checkout has unrelated changes')
            git('fetch', 'origin', 'main'); git('merge', '--ff-only', 'origin/main')
            prior_head = git('rev-parse', 'HEAD')
            require(prior_head == args.expected_base, 'Remote head differs from reviewed expected base; inspect before retry')
            git('merge-base', '--is-ancestor', BASELINE, prior_head)
            git('merge-base', '--is-ancestor', P1_ANCHOR, prior_head)
            journal = {'schema': 'baseline_matrix.publish_journal.v1', 'prior_head': prior_head,
                       'archive_sha256': delivery['archive_sha256'], 'index_sha256': delivery['content_index_sha256'],
                       'prior_releases': release_snapshot(), 'prior_tags': git('ls-remote', '--tags', 'origin'),
                       'prior_tree': tree_map(prior_head), 'commit': None, 'namespace': namespace, 'release_tag': tag}
            write(journal_path, journal)
        require(journal['archive_sha256'] == delivery['archive_sha256'] and journal['index_sha256'] == delivery['content_index_sha256'], 'Journal belongs to another payload')
        if journal['commit'] is None:
            for path in entries:
                relative = path[len(namespace) + 1:]; source = public / relative; dest = io_path(checkout / path)
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
            pathspec = stage / 'GIT_PATHSPEC.nul'; pathspec.write_bytes(b''.join(path.encode('utf-8') + b'\0' for path in entries))
            git('add', '-f', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul')
            changed = [p.decode() for p in git('diff', '--cached', '--name-only', '-z', binary=True).split(b'\0') if p]
            require(changed and all(path in entries for path in changed), 'Staged unexpected or no files')
            staged = tree_map(); verify_bodies({path: staged[path] for path in entries})
            body = stage / 'COMMIT_MESSAGE.txt'
            body.write_text(f'Add external baseline {label} evidence\n\n'
                f'Declared scope: {scope}\n'
                f"Close {delivery['phase']['expected_cells']} declared cells and {delivery['phase']['expected_questions']} questions.\n"
                'Publish source-bound original/derived predictions, donor proofs and fixed-reference metrics.\n'
                'This increment certifies its declared matrix; whole-library completion is assessed separately.\n',
                encoding='utf-8',newline='\n')
            git('-c', 'user.name=zhouz4884-coder', '-c', 'user.email=269925009+zhouz4884-coder@users.noreply.github.com', 'commit', '--file', str(body))
            journal.update(commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}')); write(journal_path, journal)
        commit = journal['commit']; tree = journal['tree']
        require(git('rev-parse', 'HEAD') == commit, 'Checkout diverged from prepared commit')
        current_tree = tree_map(commit); verify_bodies({path: current_tree[path] for path in entries})
        require(all(current_tree.get(path) == oid for path, oid in journal['prior_tree'].items() if not path.startswith(namespace + '/')), 'Changed an earlier namespace/round')
        frozen_prefixes=(namespace+'/results/',namespace+'/configs/')
        require(all(current_tree.get(path)==oid for path,oid in journal['prior_tree'].items()
                    if path.startswith(frozen_prefixes)),
                'Frozen public result/config changed; introduce a new versioned path instead')
        git('merge-base', '--is-ancestor', journal['prior_head'], commit)
        remote_head = ok(request('GET', API + '/git/ref/heads/main'))['object']['sha']
        require(remote_head in (journal['prior_head'], commit), 'Remote changed concurrently; ordinary push would need reconciliation')
        if remote_head != commit: git('push', 'origin', 'HEAD:main')
        require(ok(request('GET', API + '/git/ref/heads/main'))['object']['sha'] == commit, 'Remote main did not reach intended commit')
        remote_commit = ok(request('GET', API + '/git/commits/' + commit))
        require(remote_commit['tree']['sha'] == tree and remote_commit['parents'][0]['sha'] == journal['prior_head'], 'Remote commit/tree/parent mismatch')
        remote_tree = ok(request('GET', API + '/git/trees/' + tree, params={'recursive': 1}))
        require(not remote_tree.get('truncated'), 'Remote tree truncated')
        remote_map = {row['path']: row['sha'] for row in remote_tree['tree'] if row['type'] == 'blob'}
        require(all(remote_map.get(path) == entries[path]['git_lf']['blob_oid_sha1'] for path in entries), 'Remote actual tree omitted or changed expected public files')
        # Independent server-fresh fixed-commit source archive. Every expected
        # member is checked as an actual Git blob body against BOTH the remote
        # tree OID and staged LF SHA256; no local staging body is reused here.
        # This avoids the observed stalled smart-HTTP shallow-pack transport.
        source_archive=EVIDENCE/'remote_verification'/(commit+'_source_archive_v3.zip')
        source_archive.parent.mkdir(parents=True,exist_ok=True)
        source_url='https://codeload.github.com/zhouz4884-coder/TrajKV/zip/'+commit
        if not source_archive.exists():
            import requests
            failure=None
            for attempt in range(4):
                temporary=source_archive.with_name(source_archive.name+'.partial_'+str(time.time_ns()))
                try:
                    with requests.get(source_url,stream=True,timeout=(30,120)) as response:
                        require(response.status_code==200,'Fixed-commit source archive HTTP '+str(response.status_code))
                        with temporary.open('xb') as stream:
                            for part in response.iter_content(1024**2):
                                if part:stream.write(part)
                    require(zipfile.is_zipfile(temporary),'Source archive response is not a complete ZIP')
                    os.replace(temporary,source_archive);failure=None;break
                except Exception as exc:
                    failure=type(exc).__name__+' during fixed-commit archive transport'
                    time.sleep(min(4*(attempt+1),20))
            require(failure is None,'Independent source archive transport failed: '+str(failure))
        with zipfile.ZipFile(source_archive) as server_archive:
            names=server_archive.namelist();require(len(names)==len(set(names)),'Duplicate server archive members')
            roots={name.split('/',1)[0] for name in names if '/' in name};require(len(roots)==1,'Ambiguous source archive root')
            prefix=next(iter(roots))+'/'
            for path,info in entries.items():
                member=prefix+path;require(member in names,'Remote source archive omitted expected body: '+path)
                body=server_archive.read(member)
                oid=hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()
                require(oid==remote_map[path]==info['git_lf']['blob_oid_sha1'],'Remote source body/tree Git OID differs: '+path)
                require(hashlib.sha256(body).hexdigest()==info['git_lf']['sha256'],'Remote source body LF SHA256 differs: '+path)
        independent={'status':'PASS_REMOTE_FIXED_COMMIT_BLOB_BODIES','commit':commit,'tree':tree,'source_url':source_url,
            'source_archive_sha256':sha(source_archive),'bodies_verified':len(entries),
            'body_checks':'server actual tree OID plus Git blob header/OID plus LF SHA256','local_staging_used_as_download_source':False}
        write(stage/'REMOTE_FIXED_COMMIT_BODIES_V5.json',independent)
        auxiliary_text=''.join(f"\n- Separate auxiliary {item['phase_label']}: {item['expected_cells']} views / {item['expected_questions']} view questions. {item['coverage_scope']}"
                               for item in delivery.get('auxiliary_phases',[]))
        support_text=''.join(f"\n- Supporting completed evidence {item['label']}: {item['scope']}"
                            for item in delivery.get('supporting_completed_evidence_groups',[]))
        response = request('GET', API + '/releases/tags/' + tag)
        if response.status_code == 404:
            release = ok(request('POST', API + '/releases', json={'tag_name': tag, 'target_commitish': commit,
                'name': 'External baseline '+label+' evidence', 'draft': False, 'prerelease': False,
                'body': f"Declared scope: {scope}\n\n{delivery['phase']['expected_cells']} cells / {delivery['phase']['expected_questions']} questions. "
                        'Includes original and derived quality records, complete donor/source proofs, actual visual bytes and recomputable metrics. '
                        'Derived views retain the original answers and instrumented timing; they are not new GPU runs. '
                        'Completion applies to this declared matrix, not automatically to the full external SOTA library. '
                        'See the fixed-commit reports and exact-byte archive. Auxiliary and control counts are separate; reused reference views are not new GPU inference.'+
                        auxiliary_text+support_text}), (201,))
        else: release = ok(response)
        require(release['target_commitish'] == commit, 'Release tag is bound to another target')
        upload = release['upload_url'].split('{', 1)[0]; assets = []
        existing = {item['name']: item for item in release['assets']}
        for path in (archive, index_path, stage / 'DELIVERY.json'):
            expected_sha = sha(path)
            if path.name in existing:
                asset = existing[path.name]; require(asset['size'] == path.stat().st_size, 'Existing asset size differs; never overwrite')
            else:
                with path.open('rb') as stream:
                    asset = ok(request('POST', upload, params={'name': path.name}, data=stream,
                                       headers={'Content-Type': 'application/octet-stream'}, timeout=300), (201,))
            if asset.get('digest'): require(asset['digest'] == 'sha256:' + expected_sha, 'Server asset digest differs')
            data = request('GET', asset['url'], headers={'Accept': 'application/octet-stream'}, timeout=300)
            require(data.status_code == 200 and hashlib.sha256(data.content).hexdigest() == expected_sha, 'Downloaded release asset differs')
            assets.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': expected_sha,
                           'server_digest': asset.get('digest'), 'url': asset['browser_download_url']})
        final_releases = release_snapshot()
        require(all(final_releases.get(key) == value for key, value in journal['prior_releases'].items()), 'Earlier release/assets changed')
        # Authenticated Git refs are the authoritative equivalent of ls-remote.
        # Preserve both raw annotated-tag objects and their peeled targets.
        refs=ok(request('GET',API+'/git/matching-refs/tags/'))
        require(isinstance(refs,list),'Unexpected tag-ref response')
        tag_objects={item['ref']:item['object'] for item in refs}
        tag_checks=[]
        for line in journal['prior_tags'].splitlines():
            prior_oid,ref=line.split('\t',1);peeled=ref.endswith('^{}');base_ref=ref[:-3] if peeled else ref
            require(base_ref in tag_objects,'An earlier tag disappeared')
            obj=tag_objects[base_ref]
            if peeled:
                seen=set()
                while obj['type']=='tag':
                    require(obj['sha'] not in seen,'Cyclic annotated tag');seen.add(obj['sha'])
                    obj=ok(request('GET',API+'/git/tags/'+obj['sha']))['object']
            require(obj['sha']==prior_oid,'An earlier tag object/peeled target changed')
            tag_checks.append({'ref':ref,'sha':prior_oid,'source':'authenticated Git refs API'})
        write(stage/'OLD_TAG_REFS_API_VERIFICATION_V5.json',{'status':'PASS_OLD_TAG_OBJECTS_AND_PEELED_TARGETS_PRESERVED',
            'checked':tag_checks,'new_tag':tag,'remote_tree':tree})
        index_entry = entries[f'{namespace}/PUBLIC_CONTENT_REFS.json']['git_lf']
        receipt = {'status': 'PASS_GITHUB_BASELINE_INCREMENT_DELIVERED', 'repository': REPOSITORY,
            'phase': delivery['phase'], 'auxiliary_phases': delivery.get('auxiliary_phases', []),
            'supporting_completed_evidence_groups': delivery.get('supporting_completed_evidence_groups', []),
            'phase_group_totals_are_separate': True,
            'prior_head': journal['prior_head'], 'head': commit, 'tree': tree,
            'namespace': namespace, 'release': release['html_url'], 'assets': assets,
            'actual_remote_tree_paths_verified': len(entries), 'all_expected_blob_bodies_independently_refetched': True,
            'independent_refetch_backend':independent,
            'old_namespace_tree_preserved': True, 'prior_releases_preserved': len(journal['prior_releases']),
            'prior_tags_preserved': True, 'account': user['login'], 'visibility': 'public',
            'content_ref': {'commit': commit, 'index_blob_oid': index_entry['blob_oid_sha1'],
                'index_git_body_sha256': index_entry['sha256'], 'archive_sha256': delivery['archive_sha256']},
            'local_cold_verification': {'status': result['status'], 'score_cells': len(result['score_recomputations']),
                'independent_phase_groups': result['independent_phases_verified'],
                'closure_nodes':result['raw_complete_runs_verified'],'original_runs':result['original_donors'],
                'derived_quality_views':result['derived_views'],'source_closure_verified':True},
            'completion_scope':'declared_matrix_only','whole_external_library_completion_assessed':False,
            'P1_commit_preserved_as_ancestor':True,'verified_utc': datetime.now(timezone.utc).isoformat()}
        require(not args.receipt.exists(), 'Receipt appeared concurrently')
        write(args.receipt, receipt); journal['status'] = 'VERIFIED_COMPLETE'; write(journal_path, journal)
        print(json.dumps(receipt))
    finally: client.close()


if __name__ == '__main__': main()
