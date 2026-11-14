"""Freeze a complete transfer benchmark without selecting on Text outcomes."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 marker=ROOT/'audit/a2/textvqa_standard5000_methods_v1.json'
 if marker.exists():raise RuntimeError('Full validation methods already frozen')
 transport=json.loads((ROOT/'audit/a2/textvqa_standard5000_transport.json').read_text());assert transport['status']=='PASS'
 frozen=json.loads((ROOT/'audit/a2/book_confirmation_v1.json').read_text(encoding='utf-8'));lookup={v['name']:v for v in frozen['variants']}
 names=['full8','native4_8','base2_8','k2_lloyd_v1','fwht_global','decoder_cold_joint_vonly',frozen['candidate'],'calib2_fp32_tau0'];assert len(set(names))==8
 variants=[lookup[n] for n in names];base=json.loads((ROOT/'configs/a2/book_confirm_text_s0_v1.json').read_text(encoding='utf-8'))
 outputs=[];jobs=[]
 for shard in [0,1]:
  name=f'textvqa_standard5000_s{shard}_v1';config=copy.deepcopy(base);config.update(start=2500*shard,count=2500,variants=variants)
  config['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/textvqa_standard5000.json',turns=1,max_patches=8)
  cp=ROOT/'configs/a2'/f'{name}.json';outputs.append((cp,config));dependency='book_timing_gpu0_v1'
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{dependency}/complete.json','launch':REMOTE_ROOT+f'/jobs/{dependency}.launch.json'}],
   'settle_seconds':3,'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';outputs.append((wp,wait))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  outputs.append((ROOT/'configs/jobs'/f'{name}.json',job));jobs.append(name)
 assert all(not path.exists() for path,data in outputs)
 for path,data in outputs:path.write_text(json.dumps(data,indent=2),encoding='utf-8')
 external={variant[key]:frozen['external_codec_inputs_sha256'][variant[key]] for variant in variants for key in ['direction_book_path','decoder_book_path'] if variant.get(key)}
 result={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'questions_per_method':5000,'methods':8,'actual_free_answers':40000,
  'candidate':frozen['candidate'],'variants':variants,'source_image_ids':3166,'official_metric':'MMF ten-annotator TextVQA soft accuracy',
  'primary_comparison_family':[(frozen['candidate'],name) for name in names if name!=frozen['candidate']]+[('decoder_cold_joint_vonly','fwht_global')],
  'secondary_benchmark_quality_margin':.005,'budget_bits':1.8,
  'method_selection':'DocVQA development only; candidate fixed before future Text256 or full validation outputs; no method/hyperparameter changes based on these Text scores',
  'scope':'complete official5000-QA validation under fixed cap8 vision-only prompt/protocol; includes previously observed legacy and balanced subsets, not an entirely unseen confirmation; independent-source confirmation is reported separately',
  'resource_order':'both quiet timing and all four new confirmation jobs must complete first; one full model per3090; no V100',
  'inference_source_sha256':frozen['inference_source_sha256'],'external_codec_inputs_sha256':external,
  'inputs_sha256':{path:sha(ROOT/path) for path in ['configs/a2/textvqa_standard5000.json','configs/data/textvqa_standard5000_references.json','audit/a2/textvqa_standard5000_membership.json']},
  'job_config_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path,data in outputs},'jobs':jobs,
  'book_confirmation_sha256':sha(ROOT/'audit/a2/book_confirmation_v1.json'),'full_validation_model_outcomes_seen':False}
 marker.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'jobs':jobs,'free_answers':40000,'candidate':result['candidate']}))

if __name__=='__main__':main()
