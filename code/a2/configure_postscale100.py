"""Same encoded storage, alternate read arithmetic, on the matched100 images."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=100);variants=[]
 specs=[('post_base2',2,2,'rtn','rtn','token',None),('post_k2_vlloyd',2,1,'rtn','lloyd','channel',None),
  ('post_lloyd_joint',1,1,'lloyd','lloyd','channel',None),('post_rtn_rms_vlloyd',1,1,'rtn','lloyd','channel','head_residual_norm'),
  ('post_lloyd_rms_vlloyd',1,1,'lloyd','lloyd','channel','head_residual_norm'),('post_rtn_rms_v2',1,2,'rtn','rtn','token','head_residual_norm')]
 for name,kb,vb,ke,ve,layout,fold in specs:
  v={'name':name,'method':'rate','bits_k':kb,'bits_v':vb,'k_encoder':ke,'v_encoder':ve,'v_layout':layout,'cap':8,'store_selector_meta':False,'postscale_reader':True}
  if fold:v['k_fold_gain']=fold
  variants.append(v)
 cfg['variants']=variants;name='postscale100_gpu0_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='calibrated100_gpu0_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'kernel','entrypoint':'check_postscale.py','arguments':[]},{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':100,'variants':6,'gpu_numeric_checks':36,'actual_stored_bytes':'unchanged from each matched pre-dequant reader configuration'}))
if __name__=='__main__':main()
