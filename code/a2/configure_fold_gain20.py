"""Cost-free head-scale controls for the per-token amplitude interpretation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 c=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());c.update(start=0,count=20);vs=[]
 for enc in ('rtn','mad','lloyd'):
  for mode in ('head_norm','head_residual_norm'):
   vs.append({'name':enc+'_'+mode,'method':'rate','bits_k':1,'bits_v':1,'k_encoder':enc,'k_fold_gain':mode,'v_encoder':'lloyd','v_layout':'channel','cap':8,'store_selector_meta':False})
 c['variants']=vs;name='fold_gain20_gpu0_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(c,indent=2));dep='rotation20_gpu0_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'variants':6,'images':20,'role':'true packed head-scale ablation; two existing metadata slots, no extra payload'}))
if __name__=='__main__':main()
