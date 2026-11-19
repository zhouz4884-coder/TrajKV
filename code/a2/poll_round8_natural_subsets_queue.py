"""Poll only the six authorized R8 subset jobs; download ended runs only.

No launch/restart/GPU command is issued here. A transport exception is an
unknown observation, never worker death. Exact existing local files persist.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import shlex
import shutil
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/a2')]
from remote import connect
from round8_data_prepare import read,sha,new
from analyze_round8_candidate_quality import candidate_gate


SCAN=r'''
import json,hashlib,sys
from pathlib import Path
res=[]
for row in json.loads(sys.argv[1]):
 launch=Path(row['launch']);out=Path(row['output']);item=dict(job=row['job'],output=row['output'])
 if not launch.exists():
  item.update(state='NOT_LAUNCHED',snapshot_exists=Path(row['snapshot']).exists());res.append(item);continue
 receipt=json.loads(launch.read_text());pid=receipt['pid'];proc=Path('/proc')/str(pid)
 active=False;state='PID_ABSENT'
 if proc.exists():
  try:
   state=(proc/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
   command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   active=state not in ('Z','X') and row['snapshot'] in command
  except FileNotFoundError:state='EXITED_DURING_POLL'
 raw=out/'records.jsonl';count=0
 if raw.exists():
  with raw.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):count+=block.count(b'\n')
 marker=out/row['required_marker']
 item.update(pid=pid,process_state=state,state='RUNNING' if active else 'ENDED',records=count,
  marker_exists=marker.is_file(),required_marker=row['required_marker'],files=[])
 if not active:
  if marker.is_file():
   item['marker_status']=json.loads(marker.read_text()).get('status')
   for path in sorted(out.rglob('*')):
    if path.is_file() and path.suffix in ('.json','.jsonl'):
     size=path.stat().st_size
     if size>134217728:raise ValueError('Bounded-result download cap')
     digest=hashlib.sha256()
     with path.open('rb') as stream:
      for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
     item['files'].append(dict(remote=str(path),relative=path.relative_to(out).as_posix(),bytes=size,sha256=digest.hexdigest()))
  else:
   log=Path(receipt['log']);item['failure_log_tail']=log.read_text(errors='replace').splitlines()[-14:] if log.exists() else []
 res.append(item)
print(json.dumps(res))
'''


def exact(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=raw:raise ValueError('Existing evidence differs; never overwrite: '+str(path))
  return
 if shutil.disk_usage(ROOT).free<len(raw)+128*1024*1024:raise ValueError('Data-disk free floor')
 with path.open('xb') as stream:stream.write(raw)


def verify(folder,candidate):
 if candidate:
  result=candidate_gate(folder)
  if not result or result['status']!='PASS':raise ValueError('Candidate completion gate did not PASS')
 else:
  marker=read(folder/'complete_round8.json')
  assert marker['status']=='COMPLETE' and marker['encoder_inputs_unchanged'] is True
  assert marker['engine_complete_sha256']==sha(folder/'complete.json') and marker['manifest_sha256']==sha(folder/'manifest.json')
 engine=read(folder/'complete.json');manifest=read(folder/'manifest.json')
 count=sum(1 for line in (folder/'records.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip())
 assert engine['status']=='COMPLETE' and engine['records']==engine['expected']==count
 assert count==len(manifest['selected_ids'])*len(manifest['config']['variants'])
 return count


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download-complete',action='store_true');a=p.parse_args()
 queue=ROOT/'outputs/round8_start_20261118/natural_subsets_queue_v1';pre=read(queue/'preflight.json');rows=pre['jobs']
 for row in rows:row['required_marker']='complete_round8.json' if '_pos1_' in row['job'] else 'complete_candidate.json'
 client=connect()
 try:
  _,stdout,stderr=client.exec_command(shlex.join(['/usr/bin/python3','-B','-c',SCAN,json.dumps(rows)]),timeout=30)
  data=stdout.read().decode();error=stderr.read().decode();status=stdout.channel.recv_exit_status()
  if status:raise RuntimeError('Read-only status failed: '+error[-1400:])
  observed=json.loads(data)
  if a.download_complete:
   sftp=client.open_sftp()
   try:
    for item,row in zip(observed,rows):
     if item['state']!='ENDED' or not item.get('marker_exists') or item.get('marker_status')!='COMPLETE':continue
     local=ROOT/'results/round8_v1'/Path(item['output']).name
     for file in item['files']:
      pure=PurePosixPath(file['relative']);assert not pure.is_absolute() and '..' not in pure.parts
      target=local/file['relative']
      if target.is_file() and sha(target)==file['sha256']:continue
      with sftp.file(file['remote'],'rb') as stream:raw=stream.read()
      assert len(raw)==file['bytes'] and hashlib.sha256(raw).hexdigest()==file['sha256']
      exact(target,raw)
     count=verify(local,'_pos1_' not in item['job'])
     for suffix in ('.json','.launch.json'):
      remote=row['launch'].removesuffix('.launch.json')+suffix
      with sftp.file(remote,'rb') as stream:raw=stream.read()
      exact(queue/'receipts'/(item['job']+suffix),raw)
     launch=read(queue/'receipts'/(item['job']+'.launch.json'))
     with sftp.file(launch['log'],'rb') as stream:raw=stream.read()
     exact(queue/'logs'/(item['job']+'.log'),raw)
     item.update(local_verified_complete=True,downloaded_records=count,local_run=str(local))
   finally:sftp.close()
 finally:client.close()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
 new(queue/'polls'/(stamp+'.json'),dict(status='OBSERVED',jobs=observed,transport_failure_implies_worker_death=False))
 print(json.dumps([dict(job=x['job'],state=x['state'],pid=x.get('pid'),records=x.get('records'),marker=x.get('marker_exists'),
  verified=x.get('local_verified_complete',False),failure_log_tail=x.get('failure_log_tail')) for x in observed]),flush=True)


if __name__=='__main__':main()
