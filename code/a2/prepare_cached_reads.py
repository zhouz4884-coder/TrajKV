"""Predeclare read-selector amortization and dose exploration on new dev64."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 frozen=ROOT/'audit/a2/cached_reads_development_v1.json'
 if frozen.exists():raise RuntimeError('Read design already exists')
 base=json.loads((ROOT/'configs/a2/trajectory_development64_s0_v1.json').read_text());variants=[]
 common={'method':'cached_asym','page':128,'cap':16,'store_selector_meta':True,'selector_query_last':True}
 for name,selector,fraction in [('all','all',.25),('none','none',.25),('periodic25','periodic',.25),('risk25','risk',.25),
  ('risk50','risk',.5),('cached25','cached_risk',.25),('cached50','cached_risk',.5),('prefill_cached25','prefill_cached_risk',.25),('prefill_cached50','prefill_cached_risk',.5)]:
  v=dict(common,name='cachedread_'+name,selector=selector,fraction=fraction)
  if selector!='all':v['global_read_equivalents']=16/fraction
  variants.append(v)
 # Same cached/no-cached none/all payloads must reproduce original endpoints.
 pilot=copy.deepcopy(base);pilot['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/dev8_short.json');pilot.update(start=0,count=2)
 pilot['variants']=[variants[0],variants[1],dict(common,method='asym',selector='all',name='original_all'),dict(common,method='asym',selector='none',name='original_none'),*variants[2:]]
 pp=ROOT/'configs/a2/cached_reads_endpoint_probe_v1.json';pp.write_text(json.dumps(pilot,indent=2))
 jobs=[]
 for shard in (0,1):
  name=f'cached_reads_dev64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32,variants=variants)
  cp=ROOT/'configs/a2'/f'{name}.json';cp.write_text(json.dumps(cfg,indent=2))
  pipeline={'tasks':[{'name':'endpoints','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+pp.name]},
   {'name':'endpoint_audit','entrypoint':'check_cached_endpoints.py','arguments':['--input',REMOTE_ROOT+f'/results/{name}/endpoints']},
   {'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}]}
  qp=ROOT/'configs/a2'/f'{name}_pipeline.json';qp.write_text(json.dumps(pipeline,indent=2))
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/trajectory_development64_s{shard}_v1/complete.json',
   'launch':REMOTE_ROOT+f'/jobs/trajectory_development64_s{shard}_v1.launch.json'}],
   'entrypoint':'run_queue.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+qp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';wp.write_text(json.dumps(wait,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'motivation':'frozen risk policy saved78.56% extra logical reads but fixed-output median session was12.4% slower than all; amortize scoring once per question and sweep dose',
  'data':'new source-disjoint dev64,255questions','expected_sessions':64*len(variants),'variants':variants,'jobs':jobs,
  'comparison_storage':'all read policies own identical base+residual+selector statistics+cached uint8 masks+CPU turn/count tensors',
  'global_budget':'16 full residual forward equivalents for every non-all method, including explicit selection fees; per-forward fractions separate',
  'quality_target':'at least50% aggregate extra logical read saving and ANLS delta>=-0.005 vsall; end-to-end timing evaluated separately',
  'endpoint_probe':'2 old docs, same all/none answer IDs and initial logit hashes with/without cache buffers; no endpoint quality-based selection',
  'scope':'development exploration, method selection before untouched confirm256; no new current64 or Text128 confirmation claim',
  'source_sha256':hashlib.sha256((ROOT/'code/a2/cached_refinement_bank.py').read_bytes()).hexdigest()}
 frozen.write_text(json.dumps(result,indent=2));print(json.dumps({'variants':len(variants),'sessions':64*len(variants),'jobs':jobs}))
if __name__=='__main__':main()
