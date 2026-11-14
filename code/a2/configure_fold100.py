"""Expand the free midpoint-preserving RMS repair on all100 development images."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=100);variants=[]
 for enc in ('rtn','lloyd'):
  for factor in (.8,1.,1.25):
   variants.append({'name':f'{enc}_fold_rms_{factor:g}','method':'rate','bits_k':1,'bits_v':1,'k_encoder':enc,'k_fold_gain':'head_residual_norm','k_fold_gain_factor':factor,
     'v_encoder':'lloyd','v_layout':'channel','cap':8,'store_selector_meta':False})
 variants.append({'name':'rtn_k1_vlloyd_nofold','method':'rate','bits_k':1,'bits_v':1,'k_encoder':'rtn','v_encoder':'lloyd','v_layout':'channel','cap':8,'store_selector_meta':False})
 cfg['variants']=variants;name='fold100_gpu0_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='calib_packed_gpu0_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':100,'variants':len(variants),'records':100*len(variants),'extra_storage':0}))
if __name__=='__main__':main()
