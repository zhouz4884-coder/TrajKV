"""Complete prefill protection x selector reuse2x2 on already-open dev64."""
import copy,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 marker=ROOT/'audit/a2/read_factorial_development_v1.json'
 if marker.exists():raise RuntimeError('Read factorial already declared')
 base=json.loads((ROOT/'configs/a2/cached_reads_dev64_s0_v1.json').read_text());source=next(v for v in base['variants'] if v['name']=='cachedread_risk50')
 cfg_v=copy.deepcopy(source);cfg_v.update(name='cachedread_prefill_freshrisk50',selector='prefill_risk');jobs=[]
 for shard in (0,1):
  name=f'read_factorial_dev64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32,variants=[cfg_v])
  cp=ROOT/'configs/a2'/f'{name}.json';cp.write_text(json.dumps(cfg,indent=2))
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/density_development64_s{shard}_v1/complete.json','launch':REMOTE_ROOT+f'/jobs/density_development64_s{shard}_v1.launch.json'}],
   'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';wp.write_text(json.dumps(wait,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 marker.write_text(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'new_variant':cfg_v,'sessions':64,'jobs':jobs,
  'motivation':'prefill+cached50 improved development quality; add prefill+fresh50 so phase protection and mask amortization are not conflated',
  'conditions':'same4.0679-bit stored representation,0.5 fraction,16 full-forward-equivalent global budget; prefill full, fresh risk on decode',
  'first_question_control':'same all-refined first-question state as prefill+cached; verify initial-logit identity from complete records',
  'role':'additional adaptive development control, declared before prospective confirm256'},indent=2));print(json.dumps({'variant':cfg_v['name'],'sessions':64,'jobs':jobs}))
if __name__=='__main__':main()
