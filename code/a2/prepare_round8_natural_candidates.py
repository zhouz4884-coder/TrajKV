"""Prepare exposed dev64 candidate subsets locally; never launch or refit."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path
from round8_candidate_runner import scoped_backend,validate_subset_config
from round8_budget_check import cost


SPECS={
 'spherical':dict(name='natural_dev64_cd2_v1',parent='joint_cd_select5_v1',variants=['r8_cd1_joint0','r8_cd1_joint1']),
 'field':dict(name='natural_dev64_vfield2_v1',parent='field_joint_select5_v1',variants=['r8_kjoint0_vjoint1','r8_kjoint1_vjoint1']),
 'position':dict(name='natural_dev64_pos1_v1',parent='postfit_select11_v2',variants=['r8_k4pos_vbinary']),
}


def require(value,message):
 if not value: raise ValueError(message)


def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--families',nargs='+',choices=list(SPECS),default=list(SPECS)); a=p.parse_args()
 natural_receipt=read(ROOT/'audit/a2/round8_natural_dev64_firstq3_v1_prepared.json')
 natural_specs={x['dataset']:x for x in natural_receipt['outputs']}
 runtime_path=ROOT/'configs/a2/round8_v1/job_natural_dev64_firstq3_v1_doc.json'; runtime=read(runtime_path)
 source_root=runtime['round8_source_root']; worker=runtime['argv'][0]
 global_root=runtime['argv'][runtime['argv'].index('--output')+1].split('/results/',1)[0]; shared=global_root+'/cache/round8_v1/shared'
 require(source_root.startswith(global_root+'/stages/round8_') and worker.startswith(global_root.rsplit('/',1)[0]+'/_runtime/'),'Use isolated code and8TB P72')
 all_names=[name for spec in SPECS.values() for name in spec['variants']]
 require(len(set(all_names))==len(all_names),'Candidate names must be mutually disjoint')
 for family in a.families:
  spec=SPECS[family]; payloads=[]; outputs=[]; inputs={}; consumed={}; sources={}
  receipt_path=ROOT/f'audit/a2/round8_{spec["name"]}_prepared.json'
  for kind,gpu in [('doc',0),('text',1)]:
   natural_path=ROOT/f'configs/a2/round8_v1/natural_dev64_firstq3_v1_{kind}.json'; natural=read(natural_path)
   parent_path=ROOT/f'configs/a2/round8_v1/{spec["parent"]}_{kind}.json'; parent=read(parent_path)
   by={v['name']:v for v in parent['variants']}; variants=[copy.deepcopy(by[name]) for name in spec['variants']]
   require(not {v['name'] for v in natural['variants']}&set(all_names),'New names collide with prior natural3 controls')
   config=copy.deepcopy(natural); config.update(variants=variants,start=0,count=64)
   config.pop('development_protocol',None)
   config['scope']='Previously exposed development64 first-QA candidate extension; reuse original3 controls, no blind-confirmation claim'
   require(config['arguments']==natural['arguments'] and config['arguments']['max_patches']==16 and config['arguments']['turns']==1,'Preserve all original natural64 arguments')
   samples=local_path(config['arguments']['samples'])
   require(sha(samples)==natural_specs[kind]['samples_sha256'] and config['arguments']['samples'].startswith(global_root+'/configs/'),'Natural64 sample bytes changed')
   for v in variants:
    require(v['selected_stages']==natural['variants'][1]['selected_stages'] and len(v['selected_stages'])==224 and v['res_stages_limit']==43,'Frozen mask/priority changed')
    require(v['metadata_factor_path']==natural['variants'][1]['metadata_factor_path'] and v['budget_bits']==natural['variants'][1]['budget_bits'] and v['cap']==8,'Factor/budget/cap changed')
    for key,value in v.items():
     if key.endswith('_path') and isinstance(value,str): consumed[value]=sha(local_path(value))
   if family in ('spherical','field'):
    config.update(backend_family=family,candidate_subset=True,factorial_complete=False)
    with scoped_backend(family) as backend:
     declarations=validate_subset_config(config,family,backend); sources=backend.source_fingerprints()
    require(len(declarations)==2 and sources['round8_candidate_runner.py']=='2ba4c595072146d276b99cb93b803a64b686b0d21247460ceb50d7ed287ec338','Expected approved subset wrapper')
    entry='round8_candidate_runner.py'; marker='complete_candidate.json'
    if family=='spherical': require(all(v['key_assignment']=='row_cosine_cd' and v['row_cosine_sweeps']==1 and v['row_cosine_hard_accept'] is True for v in variants),'Expected CD1 hard-guard candidates')
    else: require(all(v['key_assignment']=='l2' and v['value_joint_refit'] is True and v['value_joint_rounds']==2 for v in variants),'Expected L2 Vjoint2 candidates')
   else:
    entry='round8_runner.py'; marker='complete_round8.json'
    v=variants[0]; book_path=local_path(v['direction_book_path']); book=read(book_path); centers=np.asarray(book['centers'],dtype=np.float16)
    complete_path=book_path.parent/'complete_books.json'; complete=read(complete_path)
    item=next(x for x in complete['books'] if x['asset']=='k_position4_16' and x['mode']=='reencode')
    require(complete['status']=='COMPLETE' and item['file']==book_path.name and item['sha256']==sha(book_path)
      and centers.shape==(32,16,4) and centers.nbytes==4096 and book['fit_mode']=='reencode','Position control must use exact prior fitted4096B endpoint')
    require(v.get('key_encoding')=='reencode' and v.get('value_encoding')=='binary' and not v.get('joint_gain_refit',False),'Position-only control changed')
    sources={name:sha(ROOT/'code/a2'/name) for name in ('round8_runner.py','round8_bank.py','round8_codec.py')}
   config['natural_candidate_extension']=dict(parent_controls='natural_dev64_firstq3_v1_'+kind,
     additional_variants=spec['variants'],same_original64_first_questions=True,required_final_marker=marker,
     expected_variants_after_merge=3+len(variants),new_source_count_claimed=0,
     decoder_K_book_bytes=4096 if family=='position' else 128,
     extra_book_bytes_vs_old_G4_16=3968 if family=='position' else 0,
     scope='Selected development candidates after observed select results; no Full or baseline is rerun and no new independent source is added.')
   cp=ROOT/f'configs/a2/round8_v1/{spec["name"]}_{kind}.json'; jp=ROOT/f'configs/a2/round8_v1/job_{spec["name"]}_{kind}.json'
   job=copy.deepcopy(runtime); job.update(name='round8_'+spec['name']+'_'+kind,round8_source_root=source_root,cwd=source_root)
   job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
   for key,value in job['env'].items():
    if key in ('TMPDIR','TMP','TEMP','HF_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX') or 'CACHE' in key: require(value.startswith(shared+'/'),'Use shared data-disk caches')
   output=global_root+'/results/round8_v1/'+spec['name']+'_'+kind
   job['argv']=[worker,source_root+'/code/a2/'+entry,'--config',global_root+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
   job['prerequisites']=['Root verifies source and config/asset SHA and an idle GPU before launch',
     'Candidate runs are accepted only after '+marker+' and its exact metadata/records hashes pass; do not use an intermediate backend marker']
   payloads.extend([(cp,config),(jp,job)])
   inputs[natural_path.relative_to(ROOT).as_posix()]=sha(natural_path); inputs[parent_path.relative_to(ROOT).as_posix()]=sha(parent_path)
   outputs.append(dict(dataset=kind,gpu=gpu,config=cp.relative_to(ROOT).as_posix(),job=jp.relative_to(ROOT).as_posix(),
     samples=natural_specs[kind]['samples'],samples_sha256=sha(samples),references_local_only=natural_specs[kind]['references_local_only'],
     references_sha256=natural_specs[kind]['references_sha256'],output=output,sources=64,questions=64,
     variants=len(variants),expected_records=64*len(variants),expected_answers=64*len(variants),required_final_marker=marker,
     expected_variants_when_merged_with_original3=3+len(variants),
     launch_command='python code/launch_round8_data_disk.py '+jp.relative_to(ROOT).as_posix()))
  require(not any(path.exists() for path,_ in payloads) and not receipt_path.exists(),'Preserve existing files; do not rerun same version')
  for path,value in payloads: new(path,value)
  for output in outputs:
   for key in ('config','job'): output[key+'_sha256']=sha(ROOT/output[key])
  budget=cost(1792,28,4,4,16,kpos=32 if family=='position' else 1,stages=43,ranking_entries=224)
  new(receipt_path,dict(schema='trajkv.round8.natural-candidate-extension.v1',status='PREPARED_NOT_EXECUTED',
    family=family,created_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),outputs=outputs,
    source_sha256_at_preparation=sources,consumed_codec_inputs_sha256=consumed,parent_input_sha256=inputs,
    global_config_sample_asset_result_root=global_root,source_root=source_root,worker_python=worker,shared_cache=shared,
    existing_original3_controls_reused=True,no_new_book_fit=True,new_source_count_claimed=0,
    complete_owner_N1792_expected_bytes=budget['complete_visual_owner_bytes'],
    book_bytes=4096 if family=='position' else 128,extra_book_bytes_vs_G4_16=3968 if family=='position' else 0,
    required_final_marker=marker,scope='Exposed development extension only; copied original64 source/first-question identities.',
    future_confirmation_answers_opened=False,GPU_used=False,SSH_used=False,uploaded=False))
  print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',family=family,outputs=outputs,GPU_used=False,SSH_used=False)),flush=True)


if __name__=='__main__': main()
