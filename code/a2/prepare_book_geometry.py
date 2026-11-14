"""Fixed16 development inputs for query-weighted directional diagnostics."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def write(path,data):
 if path.exists():raise RuntimeError('Geometry instrument already staged')
 path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
 marker=ROOT/'audit/a2/book_geometry16_design_v1.json'
 if marker.exists():raise RuntimeError('Geometry design already frozen')
 base=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text());extra=json.loads((ROOT/'configs/a2/book_factorial64_s0_v1.json').read_text())
 lookup={v['name']:v for v in base['variants']+extra['variants']}
 names=['fwht_global','decoder_cold_book_only','decoder_warm_book_only','decoder_cold_joint','decoder_warm_joint','decoder_warm_joint_konly','decoder_warm_joint_vonly'];jobs=[]
 for shard in [0,1]:
  name=f'book_geometry16_s{shard}_v1';config=copy.deepcopy(base);config.update(start=8*shard,count=8,variants=[lookup[n] for n in names])
  cp=ROOT/'configs/a2'/f'{name}.json';write(cp,config);dependency=f'book_factorial64_s{shard}_v1'
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{dependency}/complete.json','launch':REMOTE_ROOT+f'/jobs/{dependency}.launch.json'}],
   'entrypoint':'probe_book_geometry.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';write(wp,wait)
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  write(ROOT/'configs/jobs'/f'{name}.json',job);jobs.append(name)
 write(marker,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'images':16,'selection':'first16 rows of existing development64 membership, no model error filtering',
  'variants':names,'layer_records':16*7*28,'full_model_question_prefill_trials':16*8,'task_accuracy_eligible':False,'extra_fp16_references_retained_for_instrument':True,
  'scope':'centered/full key angle, fixed-native-query score error and endogenous-query additive score identity; no claim of causal energy shares, task accuracy, storage or latency',
  'source_sha256':hashlib.sha256((ROOT/'code/a2/probe_book_geometry.py').read_bytes()).hexdigest(),'jobs':jobs})
 print(json.dumps({'jobs':jobs,'layer_records':16*7*28}))

if __name__=='__main__':main()
