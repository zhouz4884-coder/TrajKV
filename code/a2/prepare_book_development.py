"""Actual packed free evaluation after both table-calibration queues finish."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 marker=ROOT/'audit/a2/book_development_design_v1.json'
 if marker.exists():raise RuntimeError('Book development already defined')
 base=json.loads((ROOT/'configs/a2/trajectory_development64_s0_v1.json').read_text());old={v['name']:v for v in base['variants']}
 variants=[copy.deepcopy(old[n]) for n in ['full8','base2_8','native4_8','k2_lloyd_v1','direction_vq2_v1']]
 common={'method':'learned_direction','bits_v':1,'v_layout':'channel','v_encoder':'lloyd','cap':8,'selector':'none','store_selector_meta':False,
  'direction_book_path':old['direction_vq2_v1']['direction_book_path'],'headwise_if_budget':True,'book_budget_bits':1.8}
 variants.append(dict(common,name='fwht_global'))
 variants.append(dict(common,name='fwht_untrained_headwise',untrained_headwise_control=True))
 for shard,condition in [(0,'cold'),(1,'warm')]:
  for task in ('book_only','joint'):
   variants.append(dict(common,name='decoder_'+condition+'_'+task,decoder_book_path=REMOTE_ROOT+f'/results/next_books_s{shard}_v1/{task}/decoder_books.json'))
 jobs=[]
 for shard in (0,1):
  name=f'book_development64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32,variants=variants)
  cp=ROOT/'configs/a2'/f'{name}.json';cp.write_text(json.dumps(cfg,indent=2))
  pipeline={'tasks':[{'name':'arithmetic','entrypoint':'check_learned_direction.py','arguments':[]},
   {'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}]}
  pp=ROOT/'configs/a2'/f'{name}_pipeline.json';pp.write_text(json.dumps(pipeline,indent=2))
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/next_books_s{s}_v1/complete.json','launch':REMOTE_ROOT+f'/jobs/next_books_s{s}_v1.launch.json'} for s in (0,1)],
   'entrypoint':'run_queue.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+pp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';wp.write_text(json.dumps(wait,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 marker.write_text(json.dumps({'variants':variants,'sessions':64*len(variants),'images':64,'questions':255,'jobs':jobs,
  'scope':'existing open development only; new methods never added post-hoc to running confirm256; future fresh source-documents required',
  'checks':'headwise-untrained and compact-global have identical code interpretation; compare learned tables at same active format and count all constant backing'},indent=2))
 print(json.dumps({'variants':len(variants),'sessions':64*len(variants),'jobs':jobs}))
if __name__=='__main__':main()
