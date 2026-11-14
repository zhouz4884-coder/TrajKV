"""Matched cap16 references and fixed-codec density transfer on open dev64."""
import copy,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 marker=ROOT/'audit/a2/density_development_design_v1.json'
 if marker.exists():raise RuntimeError('Density controls already declared')
 base=json.loads((ROOT/'configs/a2/trajectory_development64_s0_v1.json').read_text());lookup={v['name']:v for v in base['variants']};variants=[]
 for name in ['full8','base2_8','native4_8','warm_multi_gain','direction_vq2_v1']:
  v=copy.deepcopy(lookup[name]);v['name']=name.replace('_8','')+'_cap16';v['cap']=16;variants.append(v)
 jobs=[]
 for shard in (0,1):
  name=f'density_development64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32,variants=variants)
  cp=ROOT/'configs/a2'/f'{name}.json';cp.write_text(json.dumps(cfg,indent=2))
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/anchors_development64_s{shard}_v3/complete.json','launch':REMOTE_ROOT+f'/jobs/anchors_development64_s{shard}_v3.launch.json'}],
   'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';wp.write_text(json.dumps(wait,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 marker.write_text(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'variants':variants,'sessions':320,'jobs':jobs,
  'motivation':'read-none and read-all nearly tie on new dev64, so add matched Full16/Native4 references and test density transfer without refitting; baseline cap16 Base2 must match cachedread_none',
  'reporting':'different caps are different inputs; report actual total cache bytes, matched quality references and image token counts, not just bit ratios',
  'scope':'open development only; no source-document confirmation or external balanced128 outcome used'},indent=2));print(json.dumps({'variants':5,'sessions':320,'jobs':jobs}))
if __name__=='__main__':main()
