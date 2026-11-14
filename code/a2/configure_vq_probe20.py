"""Test direction codebooks on20 matched actual-model development images."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=20);variants=[]
 for dim in (4,8):
  for gain in ('mad','norm'):
   for vcodec in ('fp16','lloyd'):
    v={'name':f'vq_g{dim}_{gain}_v{vcodec}','method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,'center':True,'rotate':True,'gain':gain,
      'direction_book_path':REMOTE_ROOT+f'/results/direction_books_gpu1_v1/books/gaussian_vq_g{dim}_256.json'}
    if vcodec=='lloyd':v['v_encoder']='lloyd'
    variants.append(v)
 cfg['variants']=variants;name='vq_probe20_gpu1_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='textvqa_dev32_gpu1_v2'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':20,'variants':len(variants),'role':'real-model direction-code quality prototypes; actual materialized owners, not packed efficiency claims'}))
if __name__=='__main__':main()
