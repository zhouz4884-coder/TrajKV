"""Final development comparison under compression-before-question protocol."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
base=json.loads((ROOT/'configs/a2/dev_selectors.json').read_text());base['arguments']['compress_before_question']=True
v=[]
for name,method,cap in [('full16','full',16),('full8','full',8),('base2','base2',16),('native4_8','native4',8),('kvarn2_8','kvarn2',8),('kvarn2_16','kvarn2',16),('base1','base1',16),('k_only','asym1_k',16),('v_only','asym1_v',16)]:
 v.append({'name':name,'method':method,'selector':'all' if method.startswith('asym') else 'none','fraction':1,'store_selector_meta':False,'cap':cap})
for cap in (8,16):v.append({'name':f'native_k2v1_{cap}','method':'rate','bits_k':2,'bits_v':1,'cap':cap,'store_selector_meta':False})
for s in ('none','random','mass','risk','layer_uniform'):
 v.append({'name':'a2_'+s,'method':'asym','selector':s,'fraction':.25,'global_read_equivalents':32,'page':128,'cap':16,'selector_query_last':True})
for cap in (8,16):v.append({'name':f'hybrid_packed_{cap}','method':'hybrid_packed','clip_v':.3,'cap':cap,'store_selector_meta':False})
v.append({'name':'calib1_post','method':'calibpost','clip_k':.5,'clip_v':.3,'tau':-3.,'cap':16,'store_selector_meta':False})
for name,bk,bv,tau in [('calib_k2v1_tau0',2,1,0),('calib_k2v1_tau3',2,1,-3),('calib_k2v2_tau3',2,2,-3)]:
 v.append({'name':name,'method':'calibpost','bits_k':bk,'bits_v':bv,'clip_k':1.,'clip_v':.3 if bv==1 else 1.,'tau':tau,'cap':8,'store_selector_meta':False})
base['variants']=v
for shard in (0,1):
 c=copy.deepcopy(base);c['start']=shard*4;c['count']=4;name=f'a2_boundary_dev4_s{shard}_v1'
 (ROOT/'configs/a2'/(name+'.json')).write_text(json.dumps(c,indent=2))
 dependency='a2_hybrid_kernel_gpu0_v1' if shard==0 else 'a2_value_probe20_gpu1_v1'
 q={'wait_complete':REMOTE_ROOT+'/results/'+dependency+'/complete.json','wait_launch':REMOTE_ROOT+'/jobs/'+dependency+'.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+name+'.json']}]}
 (ROOT/'configs/a2'/(name+'_queue.json')).write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+'/configs/a2/'+name+'_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/(name+'.json')).write_text(json.dumps(j,indent=2))
(ROOT/'audit/a2/boundary_dev_precommit.json').write_text(json.dumps({'variants':v,'questions':'dev8,37 total','first_token':'compression before first question, no shared FP16 answer token','candidate_selection':'development only; all failures retained; confirmation not started'},indent=2))
print(json.dumps({'variants':len(v),'sessions':len(v)*8}))
