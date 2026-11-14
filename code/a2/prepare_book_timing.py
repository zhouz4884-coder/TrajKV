"""Freeze matched-output runtime measurements before new confirmation scores."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def write(path,data):
 if path.exists():raise RuntimeError('Book timing already staged')
 path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
 name='book_timing_gpu0_v1';marker=ROOT/'audit/a2/book_timing_design_v1.json'
 if marker.exists():raise RuntimeError('Book timing already frozen')
 fp=ROOT/'audit/a2/book_confirmation_v1.json';frozen=json.loads(fp.read_text(encoding='utf-8'))
 config=json.loads((ROOT/'configs/a2/next_timing_gpu0_v1.json').read_text());config.update(variants=frozen['variants'],images=6,repeats=3,
  frozen_contract_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),declaration='all18 method configs frozen before new confirmation scores; identical8 outputIDs per question; no task accuracy')
 cp=ROOT/'configs/a2'/f'{name}.json';write(cp,config)
 wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{job}/complete.json','launch':REMOTE_ROOT+f'/jobs/{job}.launch.json'} for job in frozen['jobs']],
  'settle_seconds':3,'entrypoint':'run_timing_multiturn.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
 wp=ROOT/'configs/a2'/f'{name}_wait.json';write(wp,wait)
 job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
 write(ROOT/'configs/jobs'/f'{name}.json',job)
 write(marker,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'job':name,'images':6,'variants':18,'measurement_trials':324,'warmups':108,
  'quality':'not_scored','method_selection':'all frozen confirmation configurations, no selection by latency or future accuracy',
  'config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'scope':'model load excluded, native prefix+encode+question replay+decode included; boundaries check both GPUs quiet'})
 print(json.dumps({'job':name,'measurement_trials':324,'warmups':108}))

if __name__=='__main__':main()
