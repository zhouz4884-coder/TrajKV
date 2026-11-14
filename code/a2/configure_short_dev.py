"""One standard short-answer instruction for every method/turn before confirmation."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
PROMPT='Answer the question using a single word or phrase.'
for name in ('dev8','confirm64'):
 rows=json.loads((ROOT/'configs/a2'/(name+'.json')).read_text(encoding='utf-8'))
 for row in rows:
  row['questions']=[q+'\n'+PROMPT for q in row['questions']]
  for item in row['content']:
   if item['type']=='text':item['text']=item['text']+'\n'+PROMPT
  row['answer_format_protocol']='internvl_official_short_phrase_prompt_v1'
 (ROOT/'configs/a2'/(name+'_short.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
base=json.loads((ROOT/'configs/a2/a2_boundary_dev4_s0_v1.json').read_text());base['arguments']['samples']=REMOTE_ROOT+'/configs/a2/dev8_short.json'
v=base['variants']
v=[r for r in v if r['name']!='calib_k2v1_tau3']
v += [{'name':'native2_clipped1_'+str(cap),'method':'hybrid_packed','key_codec':'native2','clip_v':.3,'cap':cap,'store_selector_meta':False} for cap in (8,16)]
v += [{'name':'a2_thumb_random','method':'asym','selector':'thumb_random','fraction':.25,'global_read_equivalents':32,'page':128,'cap':16,'selector_query_last':True},
      {'name':'a2_all','method':'asym','selector':'all','fraction':1,'page':128,'cap':16,'selector_query_last':True},
      {'name':'hybrid_materialized_8','method':'hybrid','value_codec':'calib1','clip_v':.3,'cap':8,'selector':'none','store_selector_meta':False}]
base['variants']=v
for shard in (0,1):
 c=copy.deepcopy(base);c['start']=shard*4;c['count']=4;name=f'a2_short_dev4_s{shard}_v1'
 (ROOT/'configs/a2'/(name+'.json')).write_text(json.dumps(c,indent=2))
 q={'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+name+'.json']}]}
 if shard==0:q.update(wait_complete=REMOTE_ROOT+'/results/a2_factorial_dev_gpu0_v2/complete.json',wait_launch=REMOTE_ROOT+'/jobs/a2_factorial_dev_gpu0_v2.launch.json')
 (ROOT/'configs/a2'/(name+'_queue.json')).write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+'/configs/a2/'+name+'_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/(name+'.json')).write_text(json.dumps(j,indent=2))
audit={'prompt':PROMPT,'source':'https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/eval/vqa/evaluate_vqa.py',
 'applied_to':'all methods, every question including future turns; reference answers/scorer unchanged','date':'2026-11-13',
 'confirm_images_same_members':True,'confirm_inference_not_started':True,'variants':len(v),
 'samples_sha256':{n:hashlib.sha256((ROOT/'configs/a2'/(n+'_short.json')).read_bytes()).hexdigest() for n in ('dev8','confirm64')}}
(ROOT/'audit/a2/short_prompt_precommit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({'variants':len(v),'sessions':len(v)*8}))
