"""Independent review of actual U3D11 numeric files, not model QA."""
import hashlib,json
from pathlib import Path
D=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
fetch=load(D/'FETCH_RECEIPT.json');remote=load(D/'INDEPENDENT_REMOTE_SHA_REHASH.json');rehash={r['remote_file']:r for r in remote['files']}
assert fetch['remote_hashes_computed_independently'] is True and fetch['remote_files_written'] is False
for item in fetch['files']:
 assert sha(D/item['file'])==item['sha256']==rehash[item['remote_file']]['sha256'] and (D/item['file']).stat().st_size==item['bytes']==rehash[item['remote_file']]['bytes']
m=load(D/'manifest.json');c=load(D/'complete.json');j=load(D/'actual_job.json');l=load(D/'launch_receipt.json');iv=load(D/'source_inventory.json');profile=load(D/'actual_candidate_profile.json')
assert c['status']=='PASS_ALL2_U3D11_FIELDS_READER_BYTES_AND_NONEMPTY_KRE_BRANCH' and c['cases']==2 and c['nonempty_KRE_GPU_stages_observed']==2
assert sha(D/'records.jsonl')==c['records_sha256']=='8ce3d9a1cbb31909b6e363323e11983b9cc381afcf5669616721db47dec0a532'
assert c['quality_QA_added']==m['quality_QA_added']==0 and m['model_images_GT_loaded'] is False and c['diagnostic_fixture_not_task_candidate'] is True
assert l['name']==j['name']==fetch['producer_name']=='bm_R9_BRPE3_GPU2Nv_v1' and l['pid']==2984949 and l['pid']>0
assert sha(D/'actual_job.json')==l['job_sha256'] and sha(D/'source_inventory.json')==l['source_inventory_sha256']
assert l['runtime_python']==j['argv'][0] and str(l['CUDA_VISIBLE_DEVICES'])==j['env']['CUDA_VISIBLE_DEVICES']=='0'
assert '--gpu-check' in l['argv_executed'] and '--cpu-check' not in l['argv_executed']
assert sha(D/'executed_source/code/round9_precision_exchange_v2.py')==iv['code/round9_precision_exchange_v2.py']=='7658d920970a9eba23b4edaed2c110e523ca66326aacafb7031ca188aa37bd99'
assert sha(D/'executed_source/code/round9_precision_exchange_v1.py')==iv['code/round9_precision_exchange_v1.py']=='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'
assert sha(D/'actual_candidate_profile.json')=='349514921518520383c375b9b5ead97ceaab356484f080cb546a91b1e188aa1a'
assert profile['upgraded_layers']==[14,22,26] and profile['U']==3 and profile['D']==11 and profile['profile_role']=='cal_only_DP_selected_U3D11'
assert sha(D/'candidate_high_book.json')==profile['high_book_sha256']=='9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae'
rows=[json.loads(x) for x in (D/'records.jsonl').read_text().splitlines()];assert [r['Nv'] for r in rows]==[256,1792]
comparisons=nonempty=0;table=[];diag_scopes=0
for row in rows:
 n=row['Nv'];assert row['status']=='PASS_FIELDS_READER_ACTUAL_BYTES'
 assert len(row['scopes'])==4
 actual=[x for x in row['scopes'] if x['scope']=='actual_cal_profile'];diag=[x for x in row['scopes'] if x['scope']=='diagnostic_nonempty_KRE_not_task']
 assert {x['mode'] for x in actual}=={'Lookup8','DropD','Exchange'} and len(diag)==1
 for result in row['scopes']:
  mode=result['mode'];meta=result['metadata'];p=meta['round9_precision_exchange'];b0=3816*n+89733+18*n*43
  delta=-1344 if mode=='Lookup8' else -198*n-1283 if mode=='DropD' else 765-6*n
  assert result['status']=='PASS_FIELDS_READER_ACTUAL_BYTES' and result['actual_delta_bytes']==result['expected_delta_bytes']==delta
  assert p['source_sha256']==iv['code/round9_precision_exchange_v2.py'] and p['numerical_fields_v1_sha256']==iv['code/round9_precision_exchange_v1.py']
  assert p['before_unique_bytes']==b0 and p['after_unique_bytes']==meta['returned_owners_unique_tensor_capacity_bytes']==b0+delta
  assert p['actual_delta_bytes']==p['expected_delta_bytes']==delta and p['lookup_bytes']==448 and p['lookup_dtype']=='torch.int8'
  assert p['old_lookup_and_original_pools_released'] is True and p['V20_base_parameters_and_residual_values_unchanged'] is True
  assert p['format_bytes']==(0 if mode=='Lookup8' else 28) and p['donor_descriptor_bytes']==(0 if mode=='Lookup8' else 33)
  assert p['old_low_book_bytes']==128 and p['extra_high_book_bytes']==(2048 if mode=='Exchange' else 0) and p['high_book_storage_shared_across_all_upgraded_layers'] is True
  assert meta['selected_stage_count']==(43 if mode=='Lookup8' else 32) and sum(x[0]=='v' for x in meta['selected_stages'])==20
  if mode!='Lookup8':assert sum(x[0]=='k' for x in meta['selected_stages'])==12 and p['U']==3 and p['D']==11
  assert p['upgraded_layers']==([14,22,26] if mode=='Exchange' else [])
  assert len(result['all28_checks'])==28 and {x['layer'] for x in result['all28_checks']}==set(range(28)) and all(x['field_and_reader_bitwise_equal'] is True for x in result['all28_checks'])
  comparisons+=28;recoded=sum(s['retained_K_stages_reencoded'] for s in p['layer_stats'])
  assert recoded==result['retained_K_stages_reencoded']
  if result['scope'].startswith('diagnostic'):
   assert mode=='Exchange' and p['candidate']['profile_role']=='GPU_nonempty_KRE_fixture_only' and recoded==1
   assert p['candidate']['dropped_k_units']!=profile['dropped_k_units']
   assert result['quality_QA_added']==0;nonempty+=recoded;diag_scopes+=1
  else:assert p['candidate']==profile
  table.append({'Nv':n,'scope':result['scope'],'mode':mode,'actual_complete_bytes':b0+delta,'delta_bytes':delta,'stages':meta['selected_stage_count'],
   'all28_field_reader_bitwise_equal':True,'retained_K_RE_stages':recoded})
assert comparisons==224 and nonempty==2 and diag_scopes==2
result={'schema':'trajkv.round9.BRPE3_GPU2Nv.independent_review.v1','status':'PASS_ACTUAL_U3D11_GPU2NV_FIELDS_READERS_BYTES_AND_NONEMPTY_KRE_INDEPENDENTLY_REVIEWED',
 'actual_GPU':True,'model_quality_QA_added':0,'synthetic_Nv':[256,1792],'actual_cal_profile_reader_comparisons':168,'diagnostic_reader_comparisons':56,
 'all28_reader_field_comparisons':comparisons,'nonempty_KRE_GPU_stages_observed':nonempty,'diagnostic_scopes_do_not_count_as_task_QA_or_cal_selected_candidates':True,
 'records_sha256':sha(D/'records.jsonl'),'source_v2_sha256':iv['code/round9_precision_exchange_v2.py'],'source_v1_sha256':iv['code/round9_precision_exchange_v1.py'],
 'profile_sha256':sha(D/'actual_candidate_profile.json'),'high_book_sha256':sha(D/'candidate_high_book.json'),'cases':table,
 'job_sha256':sha(D/'actual_job.json'),'launch_receipt_sha256':sha(D/'launch_receipt.json'),'source_inventory_sha256':sha(D/'source_inventory.json'),
 'remote_rehash_receipt_sha256':sha(D/'INDEPENDENT_REMOTE_SHA_REHASH.json'),'fetch_receipt_sha256':sha(D/'FETCH_RECEIPT.json'),
 'source_files_and_metadata_fetched':len(fetch['files']),'actual_GPU_affinity':'0',
 'fetch_retry_note':'Initial private fetch reused old gate expectedGPU1 assertion; actual job and launch consistently GPU0. Retry accepted actual affinity and required all existing partial local bytes to match; no remote write or source/worker changes.',
 'private_credentials_printed_or_saved':False,'GPU_worker_or_queue_actions_by_reviewer':False,'remote_files_written':False,'new_independent_GPU_experiment_started':False}
with (D/'INDEPENDENT_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'comparisons':comparisons,'nonempty_KRE':nonempty,'QA':0,'receipt_sha256':sha(D/'INDEPENDENT_REVIEW.json')}))
