"""Frozen Doc TRAIN-source holdout then InfoVQA validation confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/a2')]
from remote import connect
import round8_postcal_runner as backend

FREEZE_SHA='af1c0c06090522ca92efa57b0d0d85b1ae4445606510cc0bc3e1f1a11f1ff4be'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


REMOTE='''import json,hashlib,subprocess
from pathlib import Path
specs=json.loads(__import__('sys').argv[1]);sources=json.loads(__import__('sys').argv[2]);freeze_sha=__import__('sys').argv[3];rows=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for spec in specs:
 j=spec['job_data'];sr=Path(j['round8_source_root']);gr=Path(j['argv'][-1]).parents[2];out=Path(j['argv'][-1]);lp=sr/'jobs'/(j['name']+'.launch.json');cfg_path=Path(j['argv'][j['argv'].index('--config')+1]);cfg=json.loads(cfg_path.read_text())
 assert sha(cfg_path)==spec['config_sha256']
 freeze_path=gr/'configs/a2/round8_v1/method_freeze_v1.json';assert sha(freeze_path)==freeze_sha;freeze=json.loads(freeze_path.read_text());by={v['name']:v for v in freeze['variants']}
 expected_variants=[by[n] for n in freeze['confirmation_variant_names']];assert cfg['variants']==expected_variants and cfg['method_freeze_sha256']==freeze_sha and cfg['arguments']['max_patches']==8
 sample=Path(cfg['arguments']['samples']);assert sha(sample)==spec['samples_sha256'];samples=json.loads(sample.read_text());ids=[r['id'] for r in samples]
 expected_sources=spec['sources'];expected_records=expected_sources*4
 assert len(ids)==len(set(ids))==expected_sources and all(r['turns']==len(r['questions'])==1 and not any(k in r for k in ('answer','answers','references')) for r in samples)
 assets={};code_frozen={};local_only_frozen_code=[]
 for item in freeze['frozen_files']:
  rel=item['path'].replace('\\\\','/')
  if rel.startswith('results/'):
   assert sha(gr/rel)==item['sha256'];assets[rel]=item['sha256']
  elif rel.startswith('code/'):
   if (sr/rel).is_file():assert sha(sr/rel)==item['sha256'];code_frozen[rel]=item['sha256']
   else:
    assert rel not in {'code/a2/'+n for n in sources if not n.startswith('experiments/')},('Required runtime source missing',rel)
    local_only_frozen_code.append(rel)
 row=dict(pool=spec['pool'],name=j['name'],launch_exists=lp.exists(),output_exists=out.exists(),config_sha256=sha(cfg_path),samples_sha256=sha(sample),freeze_sha256=freeze_sha,frozen_assets_verified=assets,local_only_nonruntime_frozen_code=local_only_frozen_code)
 if not lp.exists():rows.append(row);continue
 l=json.loads(lp.read_text());snap=Path(l['source_snapshot']);expected=list(j['argv']);expected[1]=str(snap/'code/a2'/Path(j['argv'][1]).name);assert l['argv_executed']==expected and str(snap)==str(sr/'jobs'/(j['name']+'.source'))
 proc=Path('/proc')/str(l['pid']);argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v] if proc.exists() else []
 if proc.exists():assert argv==expected
 index=json.loads((snap/'sha256.json').read_text());source_checks={}
 for name,h in sources.items():
  rel='code/'+name if name.startswith('experiments/') else 'code/a2/'+name
  assert index[rel]==sha(snap/rel)==h;source_checks[rel]=h
 snapshot_nonruntime_absent=[]
 for rel,h in code_frozen.items():
  if rel in index:assert index[rel]==sha(snap/rel)==h
  else:
   assert rel not in source_checks,('Required numerical source absent from actual snapshot',rel)
   snapshot_nonruntime_absent.append(rel)
 files={f.name:dict(size=f.stat().st_size,sha256=sha(f)) for f in out.glob('*') if f.is_file() and f.suffix in ('.json','.jsonl')}
 count=sum(1 for line in (out/'records.jsonl').open() if line.strip()) if (out/'records.jsonl').exists() else 0
 log=Path(l['log']).read_text() if Path(l['log']).exists() else '';progress=[line for line in log.splitlines() if line.startswith('{"done":')]
 verified=False;coverage=None
 if (out/'complete_postcal.json').is_file():
  final=json.loads((out/'complete_postcal.json').read_text());post=json.loads((out/'complete_round8.json').read_text());engine=json.loads((out/'complete.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
  assert final['status']=='COMPLETE' and final['accepted_only_after_this_marker'] and final['source_sha256']=={k:v for k,v in sources.items() if k in final['source_sha256']}
  for key,name in [('complete_round8_sha256','complete_round8.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:assert final[key]==files[name]['sha256']
  for key,name in [('engine_complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:assert post[key]==files[name]['sha256']
  assert engine['status']=='COMPLETE' and engine['records']==engine['expected']==count==expected_records
  assert manifest['config_sha256']==spec['config_sha256'] and manifest['samples_sha256']==spec['samples_sha256'] and manifest['selected_ids']==ids and manifest['config']==cfg
  data=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines() if line.strip()];pairs=[(r['session_id'],r['variant']) for r in data];names=[v['name'] for v in expected_variants]
  assert len(pairs)==len(set(pairs))==expected_records and set(pairs)=={(sid,n) for sid in ids for n in names}
  variants={v['name']:v for v in expected_variants};sample_by={r['id']:r for r in samples}
  assert all(r['variant_config']==variants[r['variant']] and r['max_patches']==8 and r['image_id']==sample_by[r['session_id']]['image_id'] and [t['turn'] for t in r['turns']]==[1] for r in data)
  nvs={sid:{r['visual_tokens'] for r in data if r['session_id']==sid} for sid in ids};assert all(len(v)==1 for v in nvs.values())
  assert all(sha(path)==digest for path,digest in manifest['external_codec_inputs'].items())
  assert sha(freeze_path)==freeze_sha and sha(cfg_path)==spec['config_sha256'] and sha(sample)==spec['samples_sha256']
  coverage=dict(records=expected_records,sources=expected_sources,questions=expected_sources,variants=names,exact_frozen_variant_bytes=True,exact_sample_ID_order=True,cap8=True,matched_Nv_per_source=True,Nv_min=min(next(iter(v)) for v in nvs.values()),Nv_max=max(next(iter(v)) for v in nvs.values()))
  verified=True
 row.update(pid=l['pid'],pid_exists=proc.exists(),actual_argv=argv,receipt_argv=l['argv_executed'],launch=l,record_count=count,files=files,progress=progress[-2:],complete_graph_verified=verified,coverage=coverage,
  method_sources_verified=source_checks,frozen_code_snapshot_verified={rel:h for rel,h in code_frozen.items() if rel in index},snapshot_nonruntime_frozen_code_absent=snapshot_nonruntime_absent,
  failure_traceback=log.split('Traceback (most recent call last):')[-1] if 'Traceback (most recent call last):' in log and not proc.exists() and not verified else None)
 rows.append(row)
print(json.dumps(dict(jobs=rows,compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True))))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--include-text',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 freeze=json.loads((ROOT/'configs/a2/round8_v1/method_freeze_v1.json').read_text());assert sha(ROOT/'configs/a2/round8_v1/method_freeze_v1.json')==FREEZE_SHA
 assert all(sha(ROOT/item['path'])==item['sha256'] for item in freeze['frozen_files'])
 prepared=json.loads((ROOT/'audit/a2/round8_confirmation_firstq4_prepared_v1.json').read_text());specs=[]
 for item in prepared['pools']:
  if item['pool'] not in (('doc_train64','infovqa_validation64','text_validation128') if a.include_text else ('doc_train64','infovqa_validation64')):continue
  assert sha(ROOT/item['config'])==item['config_sha256'] and sha(ROOT/item['job'])==item['job_sha256'] and sha(ROOT/item['samples'])==item['samples_sha256']
  specs.append(dict(item,job_data=json.loads((ROOT/item['job']).read_text())))
 sources=backend.source_fingerprints()
 for name in ('session_boundary.py','round7_key_bank.py','round7_key_codec.py','round7_key_attention.py','round6_depth_bank.py','round6_depth_attention.py','metadata_direction_bank.py'):
  sources[name]=sha(ROOT/'code/a2'/name)
 c=connect()
 try:
  _,out,err=c.exec_command(shlex.join([specs[0]['job_data']['argv'][0],'-c',REMOTE,json.dumps(specs),json.dumps(sources),FREEZE_SHA]),timeout=45)
  raw=out.read().decode();error=err.read().decode();rc=out.channel.recv_exit_status()
  if rc:raise RuntimeError('Confirmation observation failed '+str(rc)+': '+error)
  result=json.loads(raw);collected=[]
  if a.collect:
   s=c.open_sftp()
   try:
    for row in result['jobs']:
     if not row.get('complete_graph_verified'):continue
     spec=next(x for x in specs if x['pool']==row['pool']);target=ROOT/'results/round8_v1'/Path(spec['job_data']['argv'][-1]).name;target.mkdir(parents=True,exist_ok=True)
     for name,entry in row['files'].items():
      path=target/name
      if path.exists():assert sha(path)==entry['sha256'];continue
      with s.file(spec['job_data']['argv'][-1]+'/'+name,'rb') as f:data=f.read()
      assert hashlib.sha256(data).hexdigest()==entry['sha256']
      with path.open('xb') as f:f.write(data)
     from analyze_round8_postcal_quality import completion_gate
     gate=completion_gate(target);assert gate and gate['status']=='PASS'
     gp=target/'local_confirmation_collection_gate.json'
     if not gp.exists():
      with gp.open('x',encoding='utf-8') as f:json.dump(gate,f,indent=2)
     collected.append(dict(pool=row['pool'],path=str(target),final_gate='PASS'))
   finally:s.close()
  assert all(sha(ROOT/item['path'])==item['sha256'] for item in freeze['frozen_files']);result['all40_local_frozen_files_unchanged']=True;result['collected']=collected
  if a.receipt:
   with a.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
  for row in result['jobs']:
   row.pop('launch',None);row.pop('method_sources_verified',None);row.pop('frozen_code_snapshot_verified',None);row.pop('frozen_assets_verified',None)
   row['files']={k:dict(size=v['size']) for k,v in row.get('files',{}).items()}
  print(json.dumps(result))
 finally:c.close()


if __name__=='__main__':main()
