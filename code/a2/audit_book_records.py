"""Independent frozen identities and closed-form tensor ownership, no inference."""
import hashlib,json
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

@lru_cache(maxsize=16)
def codec_data(path):return json.loads((ROOT/path.split('/VLAtest/TrajKV/',1)[1]).read_text(encoding='utf-8'))

def audit(frozen,runs,variants,samples_sha256):
 configs={v['name']:v for v in variants};records=[];seen=set();source_checks=codec_checks=owner_checks=0;maximum=defaultdict(float)
 amendment_path=ROOT/'audit/a2/metadata_execution_cache_v2.json'
 amendment=json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
 for run in runs:
  folder=ROOT/'results'/run;assert json.loads((folder/'complete.json').read_text())['status']=='COMPLETE'
  manifest=json.loads((folder/'manifest.json').read_text());assert manifest['samples_sha256']==samples_sha256 and not manifest['answers_visible_to_runner']
  config_key=amendment.get('config_mapping',{}).get(run,'configs/a2/'+run+'.json')
  expected_config=frozen['job_config_sha256'][config_key];assert manifest['config_sha256']==expected_config
  for name,digest in frozen['inference_source_sha256'].items():assert manifest['sources']['code/a2/'+name]==digest;source_checks+=1
  expected_external={v[key]:frozen['external_codec_inputs_sha256'][v[key]] for v in variants for key in ['direction_book_path','decoder_book_path','metadata_factor_path'] if v.get(key)}
  assert manifest['external_codec_inputs']==expected_external;codec_checks+=len(expected_external)
  for path,digest in expected_external.items():assert hashlib.sha256((ROOT/path.split('/VLAtest/TrajKV/',1)[1]).read_bytes()).hexdigest()==digest
  for line in (folder/'records.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant']);assert key not in seen;seen.add(key);records.append(r)
   c=configs[r['variant']];assert r['variant_config']==c and r['protocol']=='image_boundary_free_running'
   assert not r['initial_logits_shared_fp16'] and r.get('task_accuracy_eligible') is not False and r['read_ledger_exact'] and r.get('global_extra_budget_ok',True)
   nv=r['visual_tokens'];e=28*4*nv*128;owned=r['owned_initial'];method=r['method'];encoder=0
   assert owned['visual_fp16_denominator']==4*e
   if method=='full':expected=4*e
   elif method in ['base2','native4','rate']:
    kb=c.get('bits_k',2) if method=='rate' else 4 if method=='native4' else 2;vb=c.get('bits_v',1) if method=='rate' else kb
    expected=e*((kb+vb)/8+8/128)
   elif method=='direction_meta':
    group=nv if c.get('v_group_size')=='all' else 128
    expected=e*(3/8+4/group)+28*4*nv*2+28*4*128*2+2048
    if c.get('metadata_factor_path'):encoder=4*codec_data(c['metadata_factor_path'])['parameter_count']
    assert r['decoder_book_bytes']==0 and r['encoder_calibration_parameters_bytes']==encoder and r['value_group_size']==group
   elif method in ['direction','learned_direction']:
    expected=e*(2/8+1/8+4/128)+28*4*nv*2+28*4*128*2+2048
    if method=='direction':expected+=65536
    else:
     budget_ok=16*(expected+28*4*256*4*2+8*nv+896)/(4*e)<=1.8
     active=bool(c.get('decoder_book_path') or c.get('untrained_headwise_control')) and c.get('apply_key_decoder',True) and (budget_ok or not c.get('headwise_if_budget',True))
     assert r['decoder_headwise']==active and r['headwise_budget_ok']==budget_ok
     if active:expected+=28*4*256*4*2
     if c.get('decoder_book_path') and c.get('apply_value_metadata',True):
      data=codec_data(c['decoder_book_path'])
      if data.get('value_head_factors') is not None:encoder=896
     assert r.get('encoder_calibration_parameters_bytes',0)==encoder and not r['rotation_matrix_retained']
   elif method=='calib_packed':
    kb=c['bits_k'];vb=c['bits_v'];expected=e*(kb+vb)/8+8*28*4*128+sum((1<<bits)*2 for bits in {kb,vb})
    assert not r['materialized_quality_port'] and not r['visual_fp16_materialization']
   else:raise ValueError('Unaccounted representation: '+method)
   assert owned['visual_base']==expected and owned['visual_residual']==0 and owned['selector_metadata']==0 and owned['visual_position']==8*nv
   if method!='full':assert not owned.get('full_visual_retained') and not owned.get('diagnostic_fp16_reference')
   bits=16*(expected+8*nv+encoder)/(4*e);maximum[r['variant']]=max(maximum[r['variant']],bits)
   if method in ['learned_direction','direction_meta']:assert bits<=1.8+1e-10
   owner_checks+=1
 return records,{'status':'PASS','records':len(records),'frozen_source_checks':source_checks,'external_codec_checks':codec_checks,
  'independent_owner_formula_cases':owner_checks,'maximum_per_image_budget_bits':dict(maximum)}
