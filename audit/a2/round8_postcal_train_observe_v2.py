"""Read-only actual train status and final-graph JSON collection."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect

REMOTE='''import json,subprocess,hashlib,math,struct
from pathlib import Path
j=json.loads(__import__('sys').argv[1]);sr=Path(j['round8_source_root']);out=Path(j['argv'][-1]);l=json.loads((sr/'jobs'/(j['name']+'.launch.json')).read_text());p=Path('/proc')/str(l['pid'])
argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x] if p.exists() else []
assert l['argv_executed']==j['expected_launch_receipt']['argv_executed'];assert not p.exists() or argv==l['argv_executed']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
log=Path(l['log']).read_text();progress=[]
for line in log.splitlines():
 try:row=json.loads(line)
 except:continue
 if 'phase' in row:progress.append(row)
files={f.name:dict(size=f.stat().st_size,sha256=sha(f)) for f in out.glob('*') if f.is_file() and f.suffix in ('.json','.jsonl')}
probe=json.loads((out/'first_probe.json').read_text()) if (out/'first_probe.json').exists() else None
steps=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines() if line.strip()] if (out/'steps.jsonl').exists() else []
diagnostics=[json.loads(line) for line in (out/'gradients.jsonl').read_text().splitlines() if line.strip()] if (out/'gradients.jsonl').exists() else []
zero_diagnostics=[r for r in diagnostics if r.get('q_squared_norm')==0 or any(v==0 for v in r.get('parameter_gradient_norms',[]))]
verified=False;training=None
if (out/'complete_postcal_train.json').is_file():
 final=json.loads((out/'complete_postcal_train.json').read_text());factors=json.loads((out/'factors.json').read_text());manifest=json.loads((out/'manifest.json').read_text());cfg=json.loads(Path(j['argv'][j['argv'].index('--config')+1]).read_text())
 assert final['status']=='COMPLETE' and final['steps']==200 and final['cases']==70 and final['sources']==32 and final['accepted_only_after_this_marker']
 for name,h in final['canonical_sha256'].items():assert files[name]['sha256']==h
 assert final['config_sha256']==manifest['config_sha256']==sha(j['argv'][j['argv'].index('--config')+1])
 assert final['factors_sha256']==files['factors.json']['sha256'] and final['manifest_sha256']==files['manifest.json']['sha256'] and final['schedule_sha256']==files['schedule.json']['sha256']
 assert final['source_sha256']==manifest['source_sha256']==factors['source_sha256']
 index=json.loads((Path(l['source_snapshot'])/'sha256.json').read_text())
 for name,h in final['source_sha256'].items():
  rel='code/'+name if name.startswith('experiments/') else 'code/a2/'+name
  assert index[rel]==h and sha(Path(l['source_snapshot'])/rel)==h
 assert factors['parameter_shape']==[3,28,4] and factors['parameter_count']==336 and factors['encoder_constant_bytes']==1344 and factors['fixed_stage_count']==42
 table=factors['post_factors'];assert len(table)==3 and all(len(axis)==28 and all(len(head)==4 for head in axis) for axis in table)
 flat=[v for axis in table for layer in axis for v in layer];assert all(math.isfinite(v) for v in flat) and all(v>0 for v in flat[:224])
 assert hashlib.sha256(struct.pack('<336f',*flat)).hexdigest()==factors['table_values_sha256']
 ranking=cfg['bank_settings']['selected_stages'];packed=struct.pack('<224H',*[((int(a=='v')*28+l)*4+h)*2+d for a,l,h,d in ranking]);assert hashlib.sha256(packed).hexdigest()==factors['full_priority_sha256']
 schedule=json.loads((out/'schedule.json').read_text());assert schedule['metadata']['schedule_sha256']==cfg['source_balanced_schedule']['canonical_sha256']
 assert len(steps)==200 and [r['step'] for r in steps]==list(range(200)) and len(schedule['schedule'])==200 and schedule['metadata']['domain_step_counts']=={'doc':100,'text':100}
 assert all(len(r['parameter_gradient_norms'])==3 and all(math.isfinite(v) for v in r['parameter_gradient_norms']) and r['q_gradient_calls']>0 and r['q_gradient_norm']>=0 for r in steps)
 initial=json.loads((out/'initial_kl.json').read_text());last=json.loads((out/'final_kl.json').read_text());assert initial['case_count']==last['case_count']==70 and last['source_balanced_mean']<initial['source_balanced_mean']-1e-8
 assert probe['status']=='PASS' and probe['optimizer_updates']==0 and probe['actual_packed_zero_theta_identity'] and probe['dense_shadow_tolerance_pass']
 ctrl=factors['forward_controls'];assert ctrl['packed_calls']==ctrl['bridge_bitwise_equal'] and ctrl['identity_reader_calls']==ctrl['identity_reader_equal'] and ctrl['identity_reader_calls']>=70*28
 assert final['model_weights_frozen'] and not final['labels_used'] and final['post_encoder_constant_bytes']==1344 and final['train_KL_strictly_decreased']
 assert steps[0]['q_gradient_norm']>0 and all(v>0 for v in steps[0]['parameter_gradient_norms'])
 training=dict(initial_KL=initial['source_balanced_mean'],final_KL=last['source_balanced_mean'],steps=200,cases=70,sources=32,new_post_bytes=1344,forward_controls=ctrl,first_step_seconds=steps[0]['seconds'],last_step_seconds=steps[-1]['seconds'],finite_zero_query_steps=ctrl.get('finite_zero_query_steps'),finite_zero_gradient_steps=ctrl.get('finite_zero_gradient_steps'))
 verified=True
print(json.dumps(dict(pid=l['pid'],pid_exists=p.exists(),pid_state=(p/'stat').read_text().split()[2] if p.exists() else None,actual_argv=argv,launch_receipt=l,files=files,
 first_probe=probe,finite_zero_diagnostics=zero_diagnostics,gradient_attempts=len(diagnostics),completed_steps=len(steps),first_step=steps[0] if steps else None,latest_step=steps[-1] if steps else None,progress=progress[-3:],complete_train_graph_verified=verified,training=training,
 failure=log.split('Traceback (most recent call last):')[-1] if 'Traceback (most recent call last):' in log else None,
 compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 job=json.loads((ROOT/'configs/a2/round8_v1/job_postcal_train32_v3.json').read_text());c=connect()
 try:
  _,out,err=c.exec_command(shlex.join([job['argv'][0],'-c',REMOTE,json.dumps(job)]),timeout=40)
  raw=out.read().decode();error=err.read().decode();rc=out.channel.recv_exit_status()
  if rc:raise RuntimeError('Actual trainer observation failed '+str(rc)+': '+error)
  result=json.loads(raw)
  if a.collect:
   if not result['complete_train_graph_verified']:raise RuntimeError('No valid final training graph; do not collect as completed')
   target=ROOT/'results/round8_v1/postcal_train32_v2';target.mkdir(parents=True,exist_ok=True);s=c.open_sftp()
   try:
    for name,entry in result['files'].items():
     path=target/name
     if path.exists():
      if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Immutable collected file differs')
      continue
     with s.file(job['argv'][-1]+'/'+name,'rb') as f:data=f.read()
     if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise RuntimeError('Remote canonical bytes changed during collection')
     with path.open('xb') as f:f.write(data)
   finally:s.close()
   result['collected']=str(target)
  if a.receipt:
   with a.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
  result.pop('launch_receipt',None);result['files']={k:dict(size=v['size']) for k,v in result['files'].items()}
  print(json.dumps(result))
 finally:c.close()


if __name__=='__main__':main()
