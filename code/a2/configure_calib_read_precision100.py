"""Enlarge the offset-reading precision effect with equally improved controls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=100)
 cfg['variants']=[
  {'name':'calib1_center_fp32_100','method':'calib_packed','bits_k':1,'bits_v':1,'clip_k':.5,'clip_v':.3,'tau':0.,'preserve_center_k':True,'score_fp32':True,'cap':8,'store_selector_meta':False},
  {'name':'calib1_center_fp16_100','method':'calib_packed','bits_k':1,'bits_v':1,'clip_k':.5,'clip_v':.3,'tau':0.,'preserve_center_k':True,'score_fp32':False,'cap':8,'store_selector_meta':False},
  {'name':'calib2_fp32_control','method':'calib_packed','bits_k':2,'bits_v':2,'clip_k':1.,'clip_v':1.,'tau':-3.,'score_fp32':True,'cap':8,'store_selector_meta':False},
  {'name':'calib_k2v1_fp32_control','method':'calib_packed','bits_k':2,'bits_v':1,'clip_k':1.,'clip_v':.3,'tau':0.,'score_fp32':True,'cap':8,'store_selector_meta':False}]
 name='calib_read_precision100_gpu1_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='rotation_v2_probe20_gpu1_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':100,'variants':4,'reference_precision_control_included':True}))
if __name__=='__main__':main()
