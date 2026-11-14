"""First-answer proxies and external128 evaluation fixed before new64 analysis."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def make_job(name,gpu,dep,entrypoint,config):
 queue={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'proxies' if 'continuous' in entrypoint else 'quality','entrypoint':entrypoint,'arguments':['--config',REMOTE_ROOT+'/configs/a2/'+config]}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(queue,indent=2))
 job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2))
def main():
 fp=ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json';frozen=json.loads(fp.read_text());contract=hashlib.sha256(fp.read_bytes()).hexdigest();variants={v['name']:v for v in frozen['variants']}
 selected=['storage_primary','base2_8','k2_lloyd_v1','direction_vq2_v1','lloyd_rms125_joint','calibration_warm']
 confirm=json.loads((ROOT/'configs/a2/confirm64_short.json').read_text(encoding='utf-8'));refs={r['id']:r for r in map(json.loads,(ROOT/'configs/data/docvqa_references.jsonl').read_text(encoding='utf-8').splitlines())}
 labels=[{'id':r['id'],'image_id':r['image_id'],'answer':refs[r['id']]['references'][0]['answer']} for r in confirm]
 label_path=ROOT/'configs/a2/confirm64_teacher_refs.json';label_path.write_text(json.dumps(labels,indent=2))
 ext_names=['storage_primary','full8','base2_8','native_k2v1_8','k2_lloyd_v1','direction_vq2_v1','direction_k1_v2','calibration_warm','k2_global_mad_v1','lloyd_rms125_joint','calib_k2v1_fp32']
 base=json.loads((ROOT/'configs/a2/mechanism_confirm64_s0_v1.json').read_text());jobs=[]
 for shard in (0,1):
  name=f'confirm64_continuous_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32,variants=[variants[n] for n in selected],frozen_contract_sha256=contract)
  cfg['arguments'].update(turns=1,references=REMOTE_ROOT+'/configs/a2/confirm64_teacher_refs.json')
  path=name+'.json';(ROOT/'configs/a2'/path).write_text(json.dumps(cfg,indent=2));make_job(name,shard,f'mechanism_confirm64_s{shard}_v1','run_frozen_continuous.py',path);jobs.append(name)
  name=f'textvqa_eval128_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=64*shard,count=64,variants=[variants[n] for n in ext_names],frozen_contract_sha256=contract)
  cfg['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/textvqa_eval128.json',turns=1)
  path=name+'.json';(ROOT/'configs/a2'/path).write_text(json.dumps(cfg,indent=2));make_job(name,shard,f'confirm64_continuous_s{shard}_v1','run_variants.py',path);jobs.append(name)
 out=ROOT/'audit/a2/frozen_external128_and_proxies.json'
 if out.exists():raise RuntimeError('Followup already frozen')
 result={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'main_contract_sha256':contract,'teacher_images':64,'teacher_variants':selected,'teacher_scope':'canonical first answer only, labelled proxy not accuracy',
  'external_images':128,'external_variants':ext_names,'external_membership_sha256':hashlib.sha256((ROOT/'configs/a2/textvqa_eval128.json').read_bytes()).hexdigest(),
  'external_reference_sha256':hashlib.sha256((ROOT/'configs/a2/textvqa_eval128_refs.json').read_bytes()).hexdigest(),'external_metric':'official MMF ten-annotation soft VQA accuracy','jobs':jobs,
  'declaration':'fixed before viewing new64 outcomes; no parameters selected on external128; no main candidate changes'}
 out.write_text(json.dumps(result,indent=2));print(json.dumps({'teacher_proxy_records':64*7,'external_free_generations':128*len(ext_names),'jobs':jobs}))
if __name__=='__main__':main()
