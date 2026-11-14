"""Prespecified final scale/geometry diagnostics, old development images only."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());base.update(start=0,count=20)
 c=copy.deepcopy(base);c['arguments']['max_tokens']=16
 c['variants']=[{'name':'fp16_geometry','method':'geometrydiag','cap':8,'selector':'none','store_selector_meta':False,'check_calib_official':True}]
 (ROOT/'configs/a2/geometry20_v1.json').write_text(json.dumps(c,indent=2))
 c=copy.deepcopy(base);variants=[]
 for enc in ('rtn','mad'):
  for beta in (.125,.26,.5,1.,2.):
   variants.append({'name':f'{enc}_visual_beta{beta:g}','method':'rate','bits_k':1,'bits_v':1,'k_encoder':enc,'v_encoder':'mad','v_layout':'channel',
    'cap':8,'store_selector_meta':False,'visual_logit_gain':beta})
 c['variants']=variants
 (ROOT/'configs/a2/temperature20_v1.json').write_text(json.dumps(c,indent=2))
 for gpu,task in ((0,'geometry20'),(1,'temperature20')):
  name=task+'_gpu'+str(gpu)+'_v1';dep=f'final100_e2e_s{gpu}_v1'
  q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
   'tasks':[{'name':'diagnostic' if gpu==0 else 'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{task}_v1.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
  j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
 print(json.dumps({'geometry_images':20,'temperature_images':20,'temperature_variants':len(variants),'selection':'first20 mechanism100 before outcomes; genuine full-model GPU forwards'}))
if __name__=='__main__':main()
