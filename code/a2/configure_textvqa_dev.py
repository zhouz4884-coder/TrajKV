"""Small external OCR pilot, preceding any method-frozen external evaluation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/final100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=32)
 cfg['arguments']['samples']=REMOTE_ROOT+'/configs/a2/textvqa_dev32.json'
 cfg['variants']=[{'name':'full8','method':'full','cap':8,'selector':'none','store_selector_meta':False},
  {'name':'base2_8','method':'base2','cap':8,'selector':'none','store_selector_meta':False},
  {'name':'native_k2v1_8','method':'rate','bits_k':2,'bits_v':1,'cap':8,'selector':'none','store_selector_meta':False},
  {'name':'k2_vlloyd_channel','method':'rate','bits_k':2,'bits_v':1,'v_layout':'channel','v_encoder':'lloyd','cap':8,'selector':'none','store_selector_meta':False}]
 name='textvqa_dev32_gpu1_v2';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='direction_books_gpu1_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
if __name__=='__main__':main()
