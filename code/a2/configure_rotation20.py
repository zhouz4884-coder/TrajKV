"""Bounded response to new direction-code ideas; no held-out images."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=20)
 variants=[]
 for center in (False,True):
  for rotate in (False,True):
   for gain in ('mad','norm'):
    variants.append({'name':f'center{int(center)}_rotate{int(rotate)}_{gain}','method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,
     'center':center,'rotate':rotate,'gain':gain})
 for gain in ('mad','norm'):
  variants.append({'name':'joint_center_rotate_'+gain,'method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,
   'center':True,'rotate':True,'gain':gain,'v_encoder':'lloyd'})
 variants.append({'name':'center_rotate_roundtrip','method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,
  'center':True,'rotate':True,'roundtrip_control':True})
 cfg['variants']=variants;name='rotation20_gpu0_v1'
 (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='state100_native_history_v1'
 queue={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(queue,indent=2))
 job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));print(json.dumps({'images':20,'variants':len(variants),'records':20*len(variants),'scope':'real GPU full-model quality prototype, not compressed runtime'}))
if __name__=='__main__':main()
