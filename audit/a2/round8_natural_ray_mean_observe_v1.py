"""Authorized four-job status and canonical JSON-only collection."""
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


REMOTE='''import json,subprocess,hashlib
from pathlib import Path
specs=json.loads(__import__('sys').argv[1]);rows=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for spec in specs:
 j=spec['job_data'];sr=Path(j['round8_source_root']);out=Path(j['argv'][-1]);lp=sr/'jobs'/(j['name']+'.launch.json');snap=sr/'jobs'/(j['name']+'.source')
 row=dict(name=j['name'],family=spec['family'],dataset=spec['dataset'],gpu=spec['gpu'],output=str(out),launch_exists=lp.exists(),output_exists=out.exists())
 if not lp.exists():rows.append(row);continue
 l=json.loads(lp.read_text());proc=Path('/proc')/str(l['pid']);argv=(proc/'cmdline').read_bytes().split(bytes([0])) if proc.exists() else []
 argv=[x.decode() for x in argv if x];expected=j['expected_launch_receipt']['argv_executed']
 assert l['argv_executed']==expected and l['source_snapshot']==str(snap)
 if proc.exists():assert argv==expected,('Actual live process argv differs',j['name'],argv)
 assert Path(expected[1]).is_file() and sha(expected[1])==spec['sources'][Path(expected[1]).name]
 log=Path(l['log']);text=log.read_text() if log.exists() else '';progress=[line for line in text.splitlines() if line.startswith('{"done":')]
 files={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in out.glob('*') if p.is_file() and p.suffix in ('.json','.jsonl')}
 markers={name:json.loads((out/name).read_text()) for name in ('complete.json','complete_round8.json',spec['required_final_marker']) if (out/name).is_file()}
 count=sum(1 for line in (out/'records.jsonl').open() if line.strip()) if (out/'records.jsonl').is_file() else 0
 verified=False;coverage=None
 if spec['required_final_marker'] in markers:
  final=markers[spec['required_final_marker']];post=markers['complete_round8.json'];engine=markers['complete.json'];manifest=json.loads((out/'manifest.json').read_text())
  assert final['status']=='COMPLETE' and final['backend_family']==spec['family'] and final['accepted_only_after_this_marker']
  assert post['status']=='COMPLETE' and post['source_inputs_unchanged'] and post['encoder_inputs_unchanged']
  for key,name in [('complete_round8_sha256','complete_round8.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:assert final[key]==files[name]['sha256']
  for key,name in [('engine_complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:assert post[key]==files[name]['sha256']
  assert engine['status']=='COMPLETE' and engine['records']==engine['expected']==count==64
  assert final['source_sha256']==spec['sources'] and final['config_sha256']==manifest['config_sha256']==spec['config_sha256']
  assert manifest['samples_sha256']==spec['samples_sha256'] and manifest['config']['arguments']['max_patches']==8
  assert len(manifest['config']['variants'])==1 and manifest['config']['variants'][0]['name']==spec['variant']
  actual=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines() if line.strip()]
  ids=[r.get('session_id',r.get('id')) for r in actual]
  assert len(set(ids))==64 and set(ids)==set(manifest['selected_ids']) and len(manifest['selected_ids'])==64
  assert all(r['variant']==spec['variant'] and r['variant_config']==manifest['config']['variants'][0] and r['max_patches']==8 for r in actual)
  inventory=json.loads((snap/'sha256.json').read_text())
  for name,h in spec['sources'].items():
   rel='code/'+name if name.startswith('experiments/') else 'code/a2/'+name
   assert inventory[rel]==h and sha(snap/rel)==h
  coverage=dict(records=64,unique_sessions=64,variant=spec['variant'],actual_caps=[8],Nv_min=min(r['visual_tokens'] for r in actual),Nv_max=max(r['visual_tokens'] for r in actual),actual_snapshot_source_inventory_exact=True)
  verified=True
 failure=text.split('Traceback (most recent call last):')[-1] if 'Traceback (most recent call last):' in text and not proc.exists() and not verified else None
 row.update(pid=l['pid'],pid_exists=proc.exists(),pid_state=(proc/'stat').read_text().split()[2] if proc.exists() else None,live_cmdline=argv,
  launch_receipt=l,record_count=count,files=files,markers=markers,complete_graph_verified=verified,coverage=coverage,progress=progress[-2:],failure_traceback=failure)
 rows.append(row)
print(json.dumps(dict(jobs=rows,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
 compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 prepared=json.loads((ROOT/'audit/a2/round8_natural_ray_mean_v1_prepared.json').read_text());specs=[]
 for spec in prepared['outputs']:
  spec=dict(spec,job_data=json.loads((ROOT/spec['job']).read_text()),sources=prepared['source_sha256'][spec['family']]);specs.append(spec)
 c=connect()
 try:
  _,out,err=c.exec_command(shlex.join([prepared['worker_python'],'-c',REMOTE,json.dumps(specs)]),timeout=45)
  raw=out.read().decode();error=err.read().decode();rc=out.channel.recv_exit_status()
  if rc:raise RuntimeError('Remote observation failed '+str(rc)+': '+error)
  result=json.loads(raw);collected=[]
  if a.collect:
   s=c.open_sftp()
   try:
    for row in result['jobs']:
     if not row.get('complete_graph_verified'):continue
     target=ROOT/'results/round8_v1'/Path(row['output']).name;target.mkdir(parents=True,exist_ok=True)
     for name,entry in row['files'].items():
      path=target/name
      if path.exists():
       if sha(path)!=entry['sha256']:raise RuntimeError('Immutable local JSON differs: '+str(path))
       continue
      with s.file(row['output']+'/'+name,'rb') as f:data=f.read()
      if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise RuntimeError('Remote canonical JSON changed during collection')
      with path.open('xb') as f:f.write(data)
     from analyze_round8_final_quality import completion_gate
     gate=completion_gate(target)
     if not gate or gate['status']!='PASS':raise RuntimeError('Independent final gate failed')
     gp=target/'local_collection_gate.json'
     if not gp.exists():
      with gp.open('x',encoding='utf-8') as f:json.dump(gate,f,indent=2)
     collected.append(dict(name=row['name'],path=str(target),final_gate='PASS',coverage=row['coverage']))
   finally:s.close()
  result['collected']=collected
  if a.receipt:
   with a.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
  for row in result['jobs']:
   row.pop('launch_receipt',None)
   row['markers']={key:dict(status=v.get('status'),records=v.get('records'),expected=v.get('expected')) for key,v in row.get('markers',{}).items()}
   row['files']={key:dict(size=v['size']) for key,v in row.get('files',{}).items()}
  print(json.dumps(result))
 finally:c.close()


if __name__=='__main__':main()
