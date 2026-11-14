"""Additional strong controls fixed before viewing either held-out outcome set."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 path=ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json';f=json.loads(path.read_text());by={v['name']:v for v in f['variants']};names=['kvarn2_k_clipped_v1','kvarn2_8_quality_port'];jobs=[]
 for shard in (0,1):
  cfg=json.loads((ROOT/'configs/a2'/f'textvqa_eval128_s{shard}_v1.json').read_text());cfg['variants']=[by[n] for n in names]
  name=f'textvqa_controls128_s{shard}_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep=f'textvqa_eval128_s{shard}_v1'
  q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
   'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 info={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'variants':names,'main_contract_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'dataset_sha256':hashlib.sha256((ROOT/'configs/a2/textvqa_eval128.json').read_bytes()).hexdigest(),'jobs':jobs,
  'declaration':'additional strong controls, fixed before reading new64 or external128 outcomes; original primary tests and13 total external configurations unchanged in parameters'}
 out=ROOT/'audit/a2/frozen_external_additional_controls.json'
 if out.exists():raise RuntimeError('External additional controls already fixed')
 out.write_text(json.dumps(info,indent=2));print(json.dumps({'additional_generations':256,'jobs':jobs,'timing_wait_for':'both supplemental controls jobs as well'}))
if __name__=='__main__':main()
