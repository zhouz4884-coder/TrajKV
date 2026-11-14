"""Matched development comparison for the predeclared six calibration fits."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 path=ROOT/'audit/a2/next_development_design_v1.json'
 if path.exists():raise RuntimeError('Development design already exists')
 frozen=json.loads((ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json').read_text());old={v['name']:v for v in frozen['variants']}
 variants=[copy.deepcopy(old[n]) for n in ['full8','base2_8','native4_8','kvarn2_8_quality_port','native_k2v1_8','k2_lloyd_v1','storage_primary','direction_vq2_v1']]
 variants[6]['name']='previous_frozen_primary'
 for shard,condition in [(0,'cold'),(1,'warm')]:
  for task in ['first_gain','multi_gain','multi_affine']:
   variants.append({'name':condition+'_'+task,'method':'affine_rate','bits_k':2,'bits_v':1,'k_encoder':'rtn','v_layout':'channel','v_encoder':'lloyd',
    'cap':8,'selector':'none','store_selector_meta':False,
    'affine_factor_path':REMOTE_ROOT+f'/results/trajectory_next32_s{shard}_v2/{task}/factors.json'})
 base=json.loads((ROOT/'configs/a2/mechanism_confirm64_s0_v1.json').read_text());base.update(variants=variants)
 base['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/docvqa_next_dev64.json',turns=6)
 jobs=[];hashes={}
 for shard in (0,1):
  name=f'trajectory_development64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32)
  cfgpath=ROOT/'configs/a2'/f'{name}.json';cfgpath.write_text(json.dumps(cfg,indent=2));hashes[cfgpath.name]=hashlib.sha256(cfgpath.read_bytes()).hexdigest()
  dependency={'dependencies':[{'complete':REMOTE_ROOT+f'/results/trajectory_next32_s{s}_v2/complete.json','launch':REMOTE_ROOT+f'/jobs/trajectory_next32_s{s}_v2.launch.json'} for s in (0,1)],
   'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cfgpath.name]}
  waitpath=ROOT/'configs/a2'/f'{name}_wait.json';waitpath.write_text(json.dumps(dependency,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+waitpath.name,'--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 membership=json.loads((ROOT/'audit/a2/next_docvqa_source_split_v1.json').read_text());result={'at_utc':datetime.now(timezone.utc).isoformat(),
  'dataset':membership['manifests']['dev64'],'dataset_sha256':membership['input_sha256'][membership['manifests']['dev64']],
  'images':64,'questions':255,'variants':variants,'expected_sessions':64*len(variants),'jobs':jobs,'config_sha256':hashes,
  'status':'prepared, not launched; requires packed affine integration and GPU arithmetic parity check',
  'role':'development only, all observed scores may inform selection; confirm256 still unused'}
 path.write_text(json.dumps(result,indent=2));print(json.dumps({'variants':len(variants),'sessions':result['expected_sessions'],'questions_per_variant':255,'jobs':jobs}))
if __name__=='__main__':main()
