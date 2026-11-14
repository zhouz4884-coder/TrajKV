"""Finite last development round before new64 candidate freezing.

All images belong to mechanism100; confirmation outcomes are unavailable.
No configuration is selected or tuned using the held-out manifest.
"""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text())
 common={'method':'rate','bits_k':1,'bits_v':1,'cap':8,'store_selector_meta':False,'v_layout':'channel'}
 variants=[]
 def add(name,**kwargs):variants.append(dict(common,name=name,**kwargs))
 # Quality references are fixed at the same input resolution.
 variants.extend([{'name':'native4_reference','method':'native4','cap':8,'selector':'none','store_selector_meta':False},
                  {'name':'kvarn2_reference','method':'kvarn2','cap':8,'selector':'none','store_selector_meta':False}])
 for enc in ('mad','lloyd'):
  for layout in ('channel','token'):
   v=dict(common);v.update(name=f'k2_v{enc}_{layout}',bits_k=2,k_encoder='rtn',v_encoder=enc,v_layout=layout);variants.append(v)
  v=dict(common);v.update(name=f'k2_v{enc}_global',bits_k=2,k_encoder='rtn',v_encoder=enc,v_group_size='all');variants.append(v)
 add('k2_vrtn_channel',k_encoder='rtn',v_encoder='rtn')
 variants[-1]['bits_k']=2
 for enc in ('mad','lloyd'):
  add(f'{enc}_origin_norm',k_encoder=enc,v_encoder='mad',k_token_gain='origin_norm')
  add(f'{enc}_origin_shared',k_encoder=enc,v_encoder='mad',k_token_gain='origin_norm',gain_shared=True)
 add('lloyd_joint_none',k_encoder='lloyd',v_encoder='lloyd')
 add('lloyd_joint_centerls',k_encoder='lloyd',v_encoder='lloyd',k_token_gain='ls')
 add('lloyd_joint_origin_norm',k_encoder='lloyd',v_encoder='lloyd',k_token_gain='origin_norm')
 for shard in (0,1):
  name=f'final100_e2e_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=50*shard,count=50,variants=variants)
  (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
  q={'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2))
 print(json.dumps({'variants':len(variants),'images':100,'records':100*len(variants),'names':[v['name'] for v in variants]}))
if __name__=='__main__':main()
