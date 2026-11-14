"""Ablate the same jointly fitted K/V parameters, with exact repeat controls."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
 if path.exists():raise RuntimeError('Do not overwrite factorial experiment: '+str(path))
 path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
 marker=ROOT/'audit/a2/book_factorial64_design_v1.json'
 if marker.exists():raise RuntimeError('Factorial already frozen')
 base=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text(encoding='utf-8'));lookup={v['name']:v for v in base['variants']}
 variants=[lookup[n] for n in ['fwht_global','decoder_warm_joint','native4_8']]
 for condition in ['cold','warm']:
  joint=lookup['decoder_'+condition+'_joint']
  variants += [dict(joint,name='decoder_'+condition+'_joint_konly',apply_value_metadata=False),dict(joint,name='decoder_'+condition+'_joint_vonly',apply_key_decoder=False)]
 jobs=[]
 for shard in [0,1]:
  name=f'book_factorial64_s{shard}_v1';config=copy.deepcopy(base);config.update(start=32*shard,count=32,variants=variants)
  cp=ROOT/'configs/a2'/f'{name}.json';write(cp,config)
  dependency=f'book_continuous64_s{shard}_v1'
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{dependency}/complete.json','launch':REMOTE_ROOT+f'/jobs/{dependency}.launch.json'}],
   'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';write(wp,wait)
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  write(ROOT/'configs/jobs'/f'{name}.json',job);jobs.append(name)
 write(marker,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'variants':variants,'images':64,'actual_sessions':448,'new_unique_sessions':256,'repeated_controls':192,
  'scope':'open development extension motivated by completed joint-vs-book-only results; same fixed joint fit split into K-only/V-only applications; no retraining',
  'vonly_cost':'compact global decoder, no learned headwise table retained; count896B encoder-only V metadata; same numerical base interpretation verified by global/headwise controls',
  'control_requirement':'three repeated methods must match previous first logits, whole token trajectories and actual owner bytes on all64 images',
  'source_sha256':{name:sha(ROOT/'code/a2'/name) for name in ['learned_direction_bank.py','fused_learned_direction.py','fast_wht.py','run_variants.py']},'jobs':jobs})
 print(json.dumps({'jobs':jobs,'sessions':448,'new_variants':4}))

if __name__=='__main__':main()
