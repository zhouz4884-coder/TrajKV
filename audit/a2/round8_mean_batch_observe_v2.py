"""Bounded status/JSON-only collection for the authorized Mean batch."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/a2')]
from remote import connect


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REMOTE_CODE='''import json,subprocess,hashlib
from pathlib import Path
jobs=json.loads(__import__('sys').argv[1]);base=Path(jobs[0]['argv'][-1]).parents[2];rows=[]
for j in jobs:
 out=Path(j['argv'][-1]);launch=base/'jobs'/(j['name']+'.launch.json');l=json.loads(launch.read_text());p=Path('/proc')/str(l['pid']);log=Path(l['log'])
 files={f.name:dict(size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in out.glob('*') if f.is_file() and f.suffix in ('.json','.jsonl')}
 markers={name:json.loads((out/name).read_text()) for name in ('complete.json','complete_round8.json','complete_mean.json') if (out/name).is_file()}
 record_count=sum(1 for line in (out/'records.jsonl').open() if line.strip()) if (out/'records.jsonl').is_file() else 0
 verified=False
 if 'complete_mean.json' in markers:
  final=markers['complete_mean.json'];post=markers['complete_round8.json'];engine=markers['complete.json']
  assert final['status']=='COMPLETE' and final['backend_family']=='mean' and final['accepted_only_after_this_marker']
  assert post['status']=='COMPLETE' and post['source_inputs_unchanged'] and post['encoder_inputs_unchanged']
  for key,name in [('complete_round8_sha256','complete_round8.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:assert final[key]==files[name]['sha256']
  for key,name in [('engine_complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:assert post[key]==files[name]['sha256']
  assert engine['status']=='COMPLETE' and engine['records']==engine['expected']==record_count
  verified=True
 progress=[line for line in log.read_text().splitlines() if line.startswith('{"done":')] if log.exists() else []
 tail=' | '.join(log.read_text().split('Traceback (most recent call last):')[-1].splitlines()) if log.exists() and not p.exists() and not verified and 'Traceback (most recent call last):' in log.read_text() else None
 rows.append(dict(name=j['name'],pid=l['pid'],pid_exists=p.exists(),pid_state=(p/'stat').read_text().split()[2] if p.exists() else None,
  output=str(out),record_count=record_count,files=files,markers=markers,complete_graph_verified=verified,progress=progress[-2:],failure_tail=tail,
  launch=l))
print(json.dumps(dict(jobs=rows,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
 compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 job_paths=['configs/a2/round8_v1/job_mean_select2_v2_doc.json','configs/a2/round8_v1/job_mean_select2_v2_text.json']
 jobs=[json.loads((ROOT/path).read_text()) for path in job_paths]
 c=connect()
 try:
  _,out,err=c.exec_command(shlex.join([jobs[0]['argv'][0],'-c',REMOTE_CODE,json.dumps(jobs)]),timeout=35)
  data=out.read().decode();error=err.read().decode();rc=out.channel.recv_exit_status()
  if rc:raise RuntimeError('Remote status failed '+str(rc)+': '+error)
  result=json.loads(data);collected=[]
  if a.collect:
   s=c.open_sftp()
   try:
    for row in result['jobs']:
     if not row['complete_graph_verified']:continue
     target=ROOT/'results/round8_v1'/Path(row['output']).name
     target.mkdir(parents=True,exist_ok=True)
     for name,entry in row['files'].items():
      path=target/name
      if path.exists():
       if sha(path)!=entry['sha256']:raise RuntimeError('Existing collected immutable file differs: '+str(path))
       continue
      with s.file(row['output']+'/'+name,'rb') as f:raw=f.read()
      if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise RuntimeError('Remote JSON changed during collection')
      with path.open('xb') as f:f.write(raw)
     from analyze_round8_final_quality import completion_gate
     gate=completion_gate(target)
     if gate['status']!='PASS':raise RuntimeError('Local final gate failed')
     gp=target/'local_collection_gate.json'
     if not gp.exists():
      with gp.open('x',encoding='utf-8') as f:json.dump(gate,f,indent=2)
     collected.append(dict(name=row['name'],target=str(target),gate=gate['status'],files=len(row['files'])))
   finally:s.close()
  result['collected']=collected
  if a.receipt:
   with a.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
  for row in result['jobs']:
   row.pop('launch',None)
   row['markers']={key:dict(status=value.get('status'),records=value.get('records'),expected=value.get('expected')) for key,value in row['markers'].items()}
   row['files']={key:dict(size=value['size']) for key,value in row['files'].items()}
  print(json.dumps(result,ensure_ascii=False))
 finally:c.close()


if __name__=='__main__':main()
