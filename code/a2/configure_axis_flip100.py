"""Test whether a repaired K1 permits spending the other bit on V instead."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=100);variants=[]
 for enc in ('rtn','lloyd'):
  for factor in (1.,1.25):
   for layout in ('token','channel'):
    variants.append({'name':f'k1_{enc}_rms{factor:g}_v2_{layout}','method':'rate','bits_k':1,'bits_v':2,'k_encoder':enc,'k_fold_gain':'head_residual_norm','k_fold_gain_factor':factor,
      'v_encoder':'rtn','v_layout':layout,'cap':8,'store_selector_meta':False})
 cfg['variants']=variants;name='axis_flip100_gpu1_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='vq_probe20_gpu1_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'kernel','entrypoint':'check_v_coverage.py','arguments':['--legacy-scripts',cfg['arguments']['legacy_scripts'],'--key-bits','1']},
           {'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':100,'variants':len(variants),'records':800,'storage_point':'1.752 actual bits/scalar, same as K2/V1'}))
if __name__=='__main__':main()
