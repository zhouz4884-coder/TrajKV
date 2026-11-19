"""Read-only R8 SSH source/receipt/final-log capture to the local data disk.

Uses the existing remote.connect route, /usr/bin/python3 -B and a read-only
standard-library remote scanner. No GPU, remote writes, native tensors or
original images are accessed. Exact existing local files are never replaced.
"""
import argparse
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import shutil
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from remote import connect,REMOTE_ROOT


REMOTE_SCAN=r'''
import base64,hashlib,json,os,re,sys
from pathlib import Path
root=Path(sys.argv[1]); known=set(json.loads(sys.argv[2])); emitted=set(known)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def send(kind,**data): print(json.dumps(dict(kind=kind,**data)),flush=True)
def rawfile(path,limit=2097152):
    before=path.stat()
    if before.st_size>limit: raise ValueError('Metadata/source size cap: '+str(path))
    raw=path.read_bytes(); after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('File changed during read: '+str(path))
    return raw
def artifact(path,relative,category,limit=2097152):
    raw=rawfile(path,limit)
    send('file',relative=relative,category=category,sha256=sha(raw),bytes=len(raw),data=base64.b64encode(raw).decode())
    return raw
def source(path,expected=None,origin=None):
    raw=rawfile(path); digest=sha(raw)
    if expected is not None and digest!=expected: raise ValueError('Immutable source SHA mismatch: '+str(path))
    if digest not in emitted:
        send('object',sha256=digest,bytes=len(raw),origin=str(path),data=base64.b64encode(raw).decode()); emitted.add(digest)
    return digest
def alive(launch,snapshot):
    pid=launch.get('pid')
    if not pid: return None,'launch_pid_missing'
    proc=Path('/proc')/str(int(pid))
    if not proc.exists(): return False,'pid_absent'
    try:
        state=(proc/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except FileNotFoundError: return False,'pid_exited_during_check'
    if state in ('Z','X'): return False,'pid_terminated_'+state
    if str(snapshot) not in command: return False,'pid_reused_or_job_no_longer_matches_snapshot'
    return True,'matching_snapshot_worker_alive'
def output_path(job,launch):
    argv=launch.get('argv_executed') or job.get('argv',[])
    return Path(argv[argv.index('--output')+1]) if '--output' in argv else None
jobs=[]; gaps=[]; snapshot_root=root/'jobs'
for snapshot in sorted(snapshot_root.glob('*.source')):
    if not snapshot.is_dir(): continue
    name=snapshot.name[:-7]; inventory_path=snapshot/'sha256.json'
    if not inventory_path.is_file():
        send('pending_snapshot',job=name,reason='source inventory not finalized'); continue
    inventory_raw=artifact(inventory_path,'snapshots/'+name+'/sha256.json','immutable_source_inventory')
    inventory=json.loads(inventory_raw)
    if not isinstance(inventory,dict) or not all(re.fullmatch('[0-9a-f]{64}',v) for v in inventory.values()): raise ValueError('Invalid inventory')
    objects={}
    for path in sorted(snapshot.rglob('*.py')):
        if '__pycache__' in path.parts: continue
        relative=path.relative_to(snapshot).as_posix()
        objects[relative]=source(path,inventory.get(relative))
    missing=[name for name in inventory if name.endswith('.py') and name not in objects]
    if missing: raise ValueError('Snapshot is missing original inventory Python files: '+str(missing))
    job={}; launch={}
    for suffix,target in [('.json','job'),('.launch.json','launch')]:
        path=snapshot_root/(name+suffix)
        if path.is_file():
            raw=artifact(path,'receipts/'+name+suffix,target+'_receipt')
            parsed=json.loads(raw)
            if target=='job': job=parsed
            else: launch=parsed
    running,reason=alive(launch,snapshot)
    out=output_path(job,launch); completion={}; narrow={}; manifests=[]
    if out is not None and running is False:
        for filename in ('complete.json','complete_round8.json','complete_books.json','round8_postcheck.json',
                'complete_candidate.json','candidate_subset_postcheck.json','complete_postres.json',
                'complete_final_ray.json','complete_mean.json','manifest.json'):
            path=out/filename
            if not path.is_file(): continue
            raw=artifact(path,'completions/'+name+'/'+filename,'ended_run_metadata')
            parsed=json.loads(raw); completion[filename]=dict(sha256=sha(raw),status=parsed.get('status'))
            if filename=='manifest.json': manifests.append((str(path),sha(raw),parsed))
    log=root/'logs'/(name+'.log'); final_log=None
    if running is False and log.is_file():
        raw=artifact(log,'logs/'+name+'.log','final_ended_worker_log',8388608)
        final_log=dict(sha256=sha(raw),bytes=len(raw),relative='logs/'+name+'.log')
    for manifest_path,manifest_sha,manifest in manifests:
        legacy=manifest.get('config',{}).get('arguments',{}).get('legacy_scripts')
        source_entries=manifest.get('sources',{})
        if not isinstance(source_entries,dict): continue  # native dump: sources is an integer count
        for filename,expected in source_entries.items():
            if not filename.endswith('.py'): continue
            narrow[filename]=expected
            if expected in emitted: continue
            fallback=Path(legacy)/Path(filename).name if legacy else None
            if fallback is not None and fallback.is_file():
                raw=rawfile(fallback)
                if sha(raw)==expected:
                    source(fallback,expected)
                    send('legacy_source_match',job=name,manifest=manifest_path,manifest_sha256=manifest_sha,
                        source_name=filename,source_sha256=expected,remote_candidate=str(fallback))
                    continue
            gaps.append(dict(job=name,manifest=manifest_path,source_name=filename,expected_sha256=expected,
                reason='Missing immutable object; same-name current legacy source absent or different, never substituted'))
    record=dict(job=name,source_snapshot=str(snapshot),source_inventory_sha256=sha(inventory_raw),
        inventory_entries=len(inventory),full_source_objects=objects,narrow_source_objects=narrow,
        original_inventory='snapshots/'+name+'/sha256.json',running_at_capture=running,
        process_state_reason=reason,output=str(out) if out else None,completion_markers=completion,
        final_log=final_log,log_skipped_while_running=running is not False)
    jobs.append(record); send('job',record=record)
send('finished',jobs=len(jobs),new_source_objects=len(emitted-known),known_source_objects=len(known),source_gaps=gaps)
'''


def digest(raw): return hashlib.sha256(raw).hexdigest()


def exact(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError('Existing immutable local file differs; preserve and use a new capture/output: '+str(path))
        return False
    if shutil.disk_usage(ROOT).free < len(raw)+256*1024*1024: raise ValueError('Local data-volume free-space floor reached')
    with path.open('xb') as stream: stream.write(raw)
    return True


def encoded(value): return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def local_requirements():
    result=[]
    for path in sorted((ROOT/'results/round8_v1').glob('*/manifest.json')):
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        if not value.get('config',{}).get('arguments',{}).get('legacy_scripts'): continue
        result.append(dict(manifest=path.relative_to(ROOT).as_posix(),manifest_sha256=digest(path.read_bytes()),
            sources={k:v for k,v in value.get('sources',{}).items() if k.endswith('.py')},
            legacy_scripts=value['config']['arguments']['legacy_scripts']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/round8_v1/runtime_sources')
    p.add_argument('--capture',default='current_v1')
    a=p.parse_args(); out=a.output.resolve()
    if not out.is_relative_to(ROOT/'results/round8_v1') or not re.fullmatch('[A-Za-z0-9_]+',a.capture):
        raise ValueError('Use a project-data-disk output and simple new capture name')
    out.mkdir(parents=True,exist_ok=True); (out/'objects').mkdir(exist_ok=True)
    known=set()
    for path in (out/'objects').glob('*.py'):
        if not re.fullmatch('[0-9a-f]{64}',path.stem) or digest(path.read_bytes())!=path.stem:
            raise ValueError('Existing object name/content fingerprint mismatch')
        known.add(path.stem)
    requirements=local_requirements(); jobs=[]; files=[]; legacy=[]; pending=[]; final=None; written=0
    before=shutil.disk_usage(ROOT).free
    client=connect()
    try:
        command=shlex.join(['/usr/bin/python3','-B','-c',REMOTE_SCAN,REMOTE_ROOT+'/stages/round8_20261118_v1',json.dumps(sorted(known))])
        _,stdout,stderr=client.exec_command(command,timeout=60)
        for line in stdout:
            message=json.loads(line); kind=message['kind']
            if kind in ('file','object'):
                raw=base64.b64decode(message.pop('data'),validate=True)
                if len(raw)!=message['bytes'] or digest(raw)!=message['sha256']: raise ValueError('Transport/source hash mismatch')
                relative='objects/'+message['sha256']+'.py' if kind=='object' else message['relative']
                pure=PurePosixPath(relative)
                if pure.is_absolute() or '..' in pure.parts or '\\' in relative: raise ValueError('Unsafe local evidence path')
                if exact(out/relative,raw): written+=len(raw)
                message.update(local_relative=relative); files.append(message)
            elif kind=='job':
                jobs.append(message['record'])
                print(json.dumps(dict(status='CAPTURED_JOB',job=message['record']['job'],
                    source_entries=message['record']['inventory_entries'],running=message['record']['running_at_capture'],
                    final_log_saved=message['record']['final_log'] is not None)),flush=True)
            elif kind=='legacy_source_match': legacy.append(message)
            elif kind=='pending_snapshot': pending.append(message)
            elif kind=='finished': final=message
        status=stdout.channel.recv_exit_status()
        error=stderr.read().decode('utf-8',errors='replace')
        if status or final is None: raise RuntimeError('Read-only capture failed: '+error[-1800:])
    finally: client.close()
    all_objects={path.stem:path for path in (out/'objects').glob('*.py')}
    # Local finished manifests can include legacy imports not present in an SR
    # launcher inventory. Fetch only exact same-name/hash matches, never current
    # working-tree replacements with a different recorded SHA.
    gaps=[]
    requested={expected for item in requirements for expected in item['sources'].values()}-set(all_objects)
    if requested:
        client=connect(); sftp=client.open_sftp()
        try:
            for item in requirements:
                for name,expected in item['sources'].items():
                    if expected in all_objects: continue
                    candidate=item['legacy_scripts'].rstrip('/')+'/'+PurePosixPath(name).name
                    try:
                        with sftp.file(candidate,'rb') as stream: raw=stream.read(2097153)
                    except FileNotFoundError: raw=b''
                    if len(raw)<=2097152 and digest(raw)==expected:
                        path=out/'objects'/(expected+'.py')
                        if exact(path,raw): written+=len(raw)
                        all_objects[expected]=path
                        legacy.append(dict(manifest=item['manifest'],manifest_sha256=item['manifest_sha256'],source_name=name,
                            source_sha256=expected,remote_candidate=candidate,acceptance='exact_manifest_sha256_match'))
                    else: gaps.append(dict(manifest=item['manifest'],source_name=name,expected_sha256=expected,
                        reason='Same-name legacy candidate missing or SHA mismatch; not accepted'))
        finally: sftp.close();client.close()
    local_coverage=[dict(manifest=item['manifest'],manifest_sha256=item['manifest_sha256'],expected_python_sources=len(item['sources']),
        covered_python_sources=sum(v in all_objects for v in item['sources'].values()),
        missing={k:v for k,v in item['sources'].items() if v not in all_objects}) for item in requirements]
    for gap in final['source_gaps']:
        if gap['expected_sha256'] not in all_objects: gaps.append(gap)
    created=datetime.now(timezone.utc).isoformat()
    index=dict(schema='trajkv.round8.runtime-source-index.v1',created_at_utc=created,capture=a.capture,
        jobs=jobs,files=files,legacy_source_matches=legacy,local_quality_manifest_coverage=local_coverage,
        source_gaps=gaps,pending_snapshots=pending,objects=len(all_objects),
        object_bytes=sum(path.stat().st_size for path in all_objects.values()),
        scope='Exact original per-job source inventory and available Python versions, not proof that every listed module executed',
        collection_access='Authorized read-only existing remote.connect route; no remote files or worker states changed',
        excluded_downloads=['pt/native banks','images','weights','checkpoints'],GPU_started=False,remote_write_operations=0)
    collection=dict(schema='trajkv.round8.runtime-source-collection.v1',status='PASS' if not gaps else 'INCOMPLETE_SOURCE_COVERAGE',
        capture=a.capture,created_at_utc=created,jobs=len(jobs),objects=len(all_objects),object_bytes=index['object_bytes'],
        final_logs=sum(job['final_log'] is not None for job in jobs),active_logs_skipped=sum(job['log_skipped_while_running'] for job in jobs),
        source_gaps=len(gaps),local_quality_manifests=len(local_coverage),all_local_manifest_python_hashes_covered=all(not row['missing'] for row in local_coverage),
        runtime_index_sha256=digest(encoded(index)),bytes_written_this_capture=written,free_before=before,
        free_after=shutil.disk_usage(ROOT).free,code_sha256=digest(Path(__file__).read_bytes()),
        GPU_started=False,remote_write_operations=0,credentials_printed=False)
    # Each capture receipt is immutable. The first canonical index remains
    # intact; later snapshots use versioned metadata, never overwrite it.
    exact(out/f'runtime_index_{a.capture}.json',encoded(index)); exact(out/f'COLLECTION_{a.capture}.json',encoded(collection))
    if not (out/'runtime_index.json').exists(): exact(out/'runtime_index.json',encoded(index))
    if not (out/'COLLECTION.json').exists(): exact(out/'COLLECTION.json',encoded(collection))
    print(json.dumps(dict(collection,output=str(out))),flush=True)
    if gaps: raise SystemExit(2)


if __name__=='__main__': main()
