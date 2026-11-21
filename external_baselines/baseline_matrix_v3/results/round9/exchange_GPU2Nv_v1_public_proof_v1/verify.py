"""Independent parsing of actual fetched GPU controls; 0 model-quality QA."""
import hashlib,json
from pathlib import Path
D=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
fetch=load(D/'FETCH_RECEIPT.json');remote=load(D/'INDEPENDENT_REMOTE_SHA_REHASH.json')
assert fetch['remote_hashes_computed_independently'] is True and fetch['remote_files_written'] is False
remote_index={x['remote_file']:x for x in remote['files']}
for item in fetch['files']:
 path=D/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']==remote_index[item['remote_file']]['sha256']
m=load(D/'manifest.json');c=load(D/'complete.json');job=load(D/'actual_job.json');launch=load(D/'launch_receipt.json');iv=load(D/'source_inventory.json')
assert c['status']=='PASS_ALL2_PRECISION_EXCHANGE_FIELDS_READER_ACTUAL_BYTES' and c['cases']==2 and c['quality_QA_added']==0
assert sha(D/'records.jsonl')==c['records_sha256']=='bc69d4cb9d9e72de899a5f0e0cf0d6f7a42695af8d344d14d8b23625cb827be5'
assert m['Nv']==[256,1792] and m['model_images_GT_loaded'] is False and m['quality_QA_added']==0
assert sha(D/'executed_source/code/round9_precision_exchange_v1.py')==iv['code/round9_precision_exchange_v1.py']=='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'
assert sha(D/'actual_job.json')==launch['job_sha256'] and sha(D/'source_inventory.json')==launch['source_inventory_sha256']
assert launch['name']==job['name']==fetch['producer_name']=='bm_R9_exchange_GPU2Nv_v1' and launch['pid']>0
assert '--gpu-check' in launch['argv_executed'] and '--cpu-check' not in launch['argv_executed']
profile=load(D/'actual_candidate_profile.json')
assert sha(D/'actual_candidate_profile.json')=='7ec305a76cebf5ca9e358842e11e68fe6a77e6af9f2d83c39af723ff17eb5c22'
assert profile['upgraded_layer']==14 and profile['dropped_k_units']==[[14,0,0],[21,1,0],[16,1,0],[21,2,0]]
assert sha(D/'candidate_high_book.json')==profile['high_book_sha256']=='9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae'
rows=[json.loads(x) for x in (D/'records.jsonl').read_text().splitlines()];assert [r['Nv'] for r in rows]==[256,1792]
table=[];readers=0;upgraded_retained=0
for row in rows:
 n=row['Nv'];assert row['status']=='PASS_FIELDS_READER_BYTES'
 assert {r['mode'] for r in row['modes']}=={'Lookup8','Drop4','Exchange'}
 for result in row['modes']:
  mode=result['mode'];meta=result['metadata'];p=meta['round9_precision_exchange']
  delta=-1344 if mode=='Lookup8' else -72*n-1304 if mode=='Drop4' else 744-8*n
  b0=3816*n+89733+18*n*43
  assert result['status']=='PASS_FIELDS_READER_BYTES' and result['actual_complete_delta']==result['expected_delta']==delta
  assert p['before_unique_bytes']==b0 and p['after_unique_bytes']==meta['returned_owners_unique_tensor_capacity_bytes']==b0+delta
  assert p['actual_delta_bytes']==p['expected_delta_bytes']==delta and p['lookup_dtype']=='torch.int8' and p['lookup_bytes']==448
  assert p['old_lookup_and_original_pools_released'] is True and p['V20_base_parameters_and_residual_values_unchanged'] is True
  assert p['old_low_book_bytes']==128 and p['extra_high_book_bytes']==(2048 if mode=='Exchange' else 0)
  assert p['format_bytes']==(0 if mode=='Lookup8' else 28) and p['donor_descriptor_bytes']==(0 if mode=='Lookup8' else 12)
  assert meta['selected_stage_count']==(43 if mode=='Lookup8' else 39) and sum(x[0]=='v' for x in meta['selected_stages'])==20
  assert len(result['all28_checks'])==28 and {x['layer'] for x in result['all28_checks']}==set(range(28))
  assert all(x['reader_bitwise_equal'] is True and x['field_decode_bitwise_equal'] is True for x in result['all28_checks']);readers+=28
  assert len(p['layer_stats'])==28 and all(x['compact_real_pools'] is True for x in p['layer_stats'])
  upgraded=[s for s in p['layer_stats'] if s['whole_layer_upgraded']]
  assert [s['layer'] for s in upgraded]==([14] if mode=='Exchange' else [])
  if mode=='Exchange':
   old_keep=[x for x in meta['original_selected_stages'] if x[0]=='k' and x[1]==14 and x[1:] not in profile['dropped_k_units']]
   assert upgraded[0]['retained_K_stages_reencoded']==len(old_keep);upgraded_retained+=len(old_keep)
  table.append({'Nv':n,'mode':mode,'stages':meta['selected_stage_count'],'V_stages':20,'B0':b0,'actual_complete_bytes':b0+delta,'delta_bytes':delta,'all28_reader_and_field_bitwise':True})
assert readers==168
result={'schema':'trajkv.round9.exchange_GPU2Nv.independent_review.v1','status':'PASS_ACTUAL_GPU2NV_FIELDS_READERS_AND_COMPLETE_BYTES_INDEPENDENTLY_REVIEWED',
 'actual_GPU':True,'synthetic_geometries':[256,1792],'all28_reader_comparisons':readers,'model_quality_QA_added':0,
 'source_driver_sha256':iv['code/round9_precision_exchange_v1.py'],'records_sha256':sha(D/'records.jsonl'),'profile_sha256':sha(D/'actual_candidate_profile.json'),
 'actual_job_sha256':sha(D/'actual_job.json'),'launch_receipt_sha256':sha(D/'launch_receipt.json'),'source_inventory_sha256':sha(D/'source_inventory.json'),
 'independent_remote_rehash_receipt_sha256':sha(D/'INDEPENDENT_REMOTE_SHA_REHASH.json'),'fetch_receipt_sha256':sha(D/'FETCH_RECEIPT.json'),
 'cases':table,'signed_int8_negative_sentinel_and_mixed16_256_reader_checked':True,
 'upgrade_retained_K_RE_GPU_cases_observed':upgraded_retained,
 'coverage_limitation':'Selected layer14 only had K[14,0,0] and it was dropped. No retained K residual exists on the upgraded layer in this profile; that GPU RE branch is vacuous, not a failure. It is implemented/source- and CPU-field-checked, but future profiles retaining such stages require an actual GPU branch control.',
 'worker_queue_or_remote_write_actions_by_reviewer':False,'independent_GPU_experiment_started_by_reviewer':False,'not_free_generation_or_task_quality':True}
with (D/'INDEPENDENT_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'reader_comparisons':readers,'upgrade_retained_K_RE_GPU_cases':upgraded_retained,'receipt_sha256':sha(D/'INDEPENDENT_REVIEW.json')}))
