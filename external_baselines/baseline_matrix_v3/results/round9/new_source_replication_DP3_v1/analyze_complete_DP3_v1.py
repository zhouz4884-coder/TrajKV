"""Bounded four completed DP3 sources + exact frozen8 reference proof reuse."""
import sys,pathlib,json,hashlib,csv,statistics
sys.dont_write_bytecode=True
ROOT=pathlib.Path('${PROJECT_ROOT}');E=pathlib.Path('${EVIDENCE_ROOT}/results/round9_v1');PREV=pathlib.Path('${EVIDENCE_ROOT}/results/round9/new_source_replication_v1/report');OUT=pathlib.Path('${EVIDENCE_ROOT}/results/round9/new_source_replication_DP3_v1/report');REFS=pathlib.Path('${EVIDENCE_ROOT}/results/round9/new_confirmation_reserve32x2_v1')
sys.path.insert(0,str(ROOT/'ops'))
import round9_compare_dev_v2 as comp
import verify_public_custom_driver_v1 as physical
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,o):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(o,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def stats(x):return {'mean':statistics.mean(x),'median':statistics.median(x),'min':min(x),'max':max(x)} if x else None
DRIVER='ed102ab76346ba70a1b5cbddd5887b3f87e4216e837b9b8ff71bad9e23391abb';HELPER='7658d920970a9eba23b4edaed2c110e523ca66326aacafb7031ca188aa37bd99';PROFILE='349514921518520383c375b9b5ead97ceaab356484f080cb546a91b1e188aa1a'
datasets=['textvqa','infovqa'];modes=['BRPE3_DropD','BRPE3_Exchange'];folders={(d,m):E/'scored'/f'{m}_{d}_new32_v1' for d in datasets for m in modes}
missing=[str(p/'summary.json') for p in folders.values() if not(p/'summary.json').is_file()]
if missing:print(json.dumps({'status':'WAIT_FOUR_ACTUAL_DP3_SCORED_SOURCES','missing':missing,'report_written':False}));raise SystemExit(3)
assert not OUT.exists(),'Immutable DP3 report exists'
old_receipt=json.loads((PREV/'ANALYSIS_RECEIPT.json').read_text());assert sha(PREV/'summary.json')==old_receipt['summary_sha256'] and sha(PREV/'source_index.json')==old_receipt['source_index_sha256']
old=json.loads((PREV/'summary.json').read_text());old_index=json.loads((PREV/'source_index.json').read_text());assert old['status']=='PASS_COMPLETE_EIGHT_NEW_SOURCE_RUNS_AND_MATCHED_PAIRS' and old['actual_new_answer_events']==256
reference_proof={};references={};raw_references={}
for name,item in old_index.items():
 folder=pathlib.Path(item['scored']);assert sha(folder/'summary.json')==item['summary_sha256'] and sha(folder/'scored_records.jsonl')==item['scored_records_sha256']
 raw=pathlib.Path(item['raw']);assert all(sha(raw/k)==v for k,v in item['raw_file_sha256'].items())
 su,rows=comp.load(folder);key=(su['dataset'],su['_raw_manifest']['candidate']['mode']);references[key]=(su,rows,folder)
 raw_references[key]={r['id']:r for r in [json.loads(l) for l in (raw/'records.jsonl').read_text(encoding='utf-8').splitlines()]}
 reference_proof[name]={'summary_sha256':item['summary_sha256'],'physical_proof_reused_from':str(PREV/'summary.json'),'physical_proof_sha256':sha(PREV/'summary.json'),'prior_owner_reconstruction_not_repeated':True}
before={str(pathlib.Path(m.__file__).resolve()):sha(m.__file__) for m in [comp,physical]}
common=['model_spec','torch_version','transformers_version','CUDA_version','python_version','weight_dtype','tf32','seed','attention_implementation','processor_id','processor_version','processor_file_hashes','generation_native','generation_overrides','tokenizer_class','tokenizer_is_fast','tokenizers_version','decode_overrides','runtime_library_source_sha256','frozen_variant_sha256','asset_sha256']
points=list(old['points']);inventory={};pairs={};loaded={}
for ds in datasets:
 base=references[ds,'original'][0];bm=base['_raw_manifest']
 for mode in modes:
  folder=folders[ds,mode];su,rows=comp.load(folder);m=su['_raw_manifest'];raw=pathlib.Path(su['integrity']['raw_dir']);name=f'{mode}_{ds}_new32_v1'
  assert su['status']=='PASS_SCORED_COMPLETE' and len(rows)==su['expected_run_questions']==32 and not su['groups']['overall']['failed_ids']
  assert pathlib.Path(su['integrity']['refs_path']).resolve()==(REFS/(ds+'_confirm32_refs.jsonl')).resolve() and su['integrity']['refs_sha256']==sha(REFS/(ds+'_confirm32_refs.jsonl'))
  assert m['expected_ids']==bm['expected_ids'] and m['inputs_sha256']==bm['inputs_sha256']
  for k in common:assert m[k]==bm[k],('Common protocol/decoder differs',ds,mode,k)
  for k,v in bm['source_hashes'].items():
   if k!='code/round9_runner_v4.py':assert m['source_hashes'][k]==v,('Frozen numeric dependency differs',k)
  assert m['source_hashes']['code/round9_runner_v5.py']==sha(raw/'source_driver.py')==DRIVER and m['source_hashes']['code/round9_precision_exchange_v2.py']==HELPER
  proof=json.loads((raw/'source_retrieval_receipt_round9_v2.json').read_text());iv=json.loads((raw/'source_inventory.json').read_text());launch=json.loads((raw/'launch_receipt.json').read_text())
  assert proof['producer_name']=='bm_R9_'+name and proof['actual_driver']=='code/round9_runner_v5.py'
  assert iv['code/round9_runner_v5.py']==DRIVER and sha(raw/'source_inventory.json')==launch['source_inventory_sha256'] and launch['argv_executed'][2]==m['runtime_source_root']+'/code/round9_runner_v5.py'
  assert m['candidate']['profile_sha256']==sha(raw/'assets/candidate_profile.json')==PROFILE
  profile=json.loads((raw/'assets/candidate_profile.json').read_text());assert profile['profile_role']=='cal_only_DP_selected_U3D11' and profile['upgraded_layers']==[14,22,26] and len(profile['dropped_k_units'])==11 and profile['selection_provenance']['task_scores_or_answers_used'] is False
  assert sha(raw/'assets/candidate_high_book.json')==profile['high_book_sha256']
  rr=[json.loads(l) for l in (raw/'records.jsonl').read_text(encoding='utf-8').splitlines()];amounts=[];bits=[];refbytes=[]
  for r,row in zip(rr,rows):
   assert r['id']==row['id'] and r['status']=='PASS';b=physical.physical_complete_bytes(r,m['model_spec']);n=r['actual_visual_token_count'];b0=3816*n+89733+18*n*43
   delta=-198*n-1283 if mode=='BRPE3_DropD' else 765-6*n
   assert b==row['stored_visual_KV_bytes']==b0+delta<=b0 and r['BRPE3_budget_vs_original_R7_low']['actual_complete_bytes']==b
   q=r['r7_bridge']['round9_precision_exchange'];assert r['r7_bridge']['selected_stage_count']==32 and sum(x[0]=='k' for x in r['r7_bridge']['selected_stages'])==12 and sum(x[0]=='v' for x in r['r7_bridge']['selected_stages'])==20
   assert q['source_sha256']==HELPER and q['lookup_dtype']=='torch.int8' and q['lookup_bytes']==448 and q['format_bytes']==28 and q['donor_descriptor_bytes']==33 and q['old_lookup_and_original_pools_released'] is True and q['V20_base_parameters_and_residual_values_unchanged'] is True
   assert q['extra_high_book_bytes']==(2048 if mode=='BRPE3_Exchange' else 0) and q['high_book_storage_shared_across_all_upgraded_layers'] is True
   assert raw_references[ds,'original'][r['id']]['complete_visual_owner_bytes']==b0
   amounts.append(b);bits.append(r['actual_effective_visual_bits']);refbytes.append(b0)
  loaded[ds,mode]=(su,rows,folder)
  points.append({'dataset':ds,'method':mode,'QA':32,'primary_mean':su['primary_mean'],'primary_ci95':su['primary_ci95'],'metric_id':rows[0]['metric_id'],'mean_complete_bytes':statistics.mean(amounts),'max_complete_bytes':max(amounts),'mean_bits':statistics.mean(bits),'max_bits':max(bits),'strict_all_requests_1p3':all(160*b<=13*r['full_kv_byte_denominator'] for b,r in zip(amounts,rr)),'strict_all_requests_1p8':all(80*b<=9*r['full_kv_byte_denominator'] for b,r in zip(amounts,rr)),'total_bytes_ratio_to_original':sum(amounts)/sum(refbytes),'timing_ms':{k:stats([float(r[k]) for r in rr if r.get(k)is not None]) for k in ['encode_ms','ttft_ms','question_prefill_ms','wall_ms']},'output_tokens':stats([r['output_tokens'] for r in rr]),'output_at64cap':sum(r['output_tokens']==64 for r in rr),'prototype_not_controlled_speed':True,'physical_owner_tables_reconstructed':32})
  inventory[name]={'raw':str(raw),'scored':str(folder),'summary_sha256':sha(folder/'summary.json'),'scored_records_sha256':sha(folder/'scored_records.jsonl'),'raw_file_sha256':su['integrity']['raw_files'],'source_receipt_sha256':sha(raw/'source_retrieval_receipt_round9_v2.json'),'source_inventory_sha256':sha(raw/'source_inventory.json'),'launch_sha256':sha(raw/'launch_receipt.json'),'source_hashes':m['source_hashes'],'profile_sha256':PROFILE,'refs_sha256':su['integrity']['refs_sha256']}
  for anchor in ['original','FP16','BRPE_Drop4','BRPE_Exchange']:
   pair=comp.compare(references[ds,anchor][2],[folder]);assert pair['status']=='PASS_ALL_DECLARED_CANDIDATE_IDS' and pair['candidate_QA']==pair['anchor_reference_QA']==32
   pairs[f'{ds}__{mode}_minus_{anchor}']=pair
 pair=comp.compare(folders[ds,'BRPE3_DropD'],[folders[ds,'BRPE3_Exchange']]);assert pair['status']=='PASS_ALL_DECLARED_CANDIDATE_IDS';pairs[ds+'__BRPE3_Exchange_minus_BRPE3_DropD']=pair
 assert all(sha(p)==v for p,v in before.items()),'Analysis helper source changed'
OUT.mkdir(parents=True,exist_ok=False)
for name,value in pairs.items():write(OUT/(name+'.json'),value)
write(OUT/'source_index.json',{'new_DP3_sources':inventory,'reference_proof_reused':reference_proof})
summary={'schema':'trajkv.round9.new_source_DP3_tournament.v1','status':'PASS_FOUR_DP3_CLOSED_AND_MATCHED_SIX_METHOD_TABLE','new_DP3_answer_events':128,'reference_answer_events_reused':256,'logical_table_answer_events':384,'unique_source_questions':64,'points':points,'paired_comparisons':pairs,'source_index_sha256':sha(OUT/'source_index.json'),'reference_summary_sha256':sha(PREV/'summary.json'),'profile_sha256':PROFILE,'runner_actual_sha256':DRIVER,'helper_actual_sha256':HELPER,'physical_reconstruction':'128 newDP3 owner tables; prior256 owner-table proof reused by exact source/score/raw SHA, not re-reconstructed','profile_fitted_from_old_cal_only_not_fresh_answers':True,'formal500_or_preregistered_confirmation_claimed':False,'unadjusted_paired_CIs_are_descriptive_not_equivalence_or_winner':True,'different_K_base_residual_counts_and_actual_costs_are_not_isobyte':True,'no_cross_dataset_quality_average':True,'analysis_sources':before,'new_GPU_or_remote_mutations':0}
write(OUT/'summary.json',summary)
with (OUT/'table.csv').open('x',encoding='utf-8',newline='')as f:
 w=csv.writer(f);w.writerow(['dataset','method','QA','score_percent','mean_complete_bytes','mean_bits','max_bits','strict_all_1p3','vs_original_pp','CI95pp','higher_lower','encode_median_ms','wall_median_ms'])
 for p in points:
  q=pairs.get(f"{p['dataset']}__{p['method']}_minus_original",{}).get('primary',{})
  w.writerow([p['dataset'],p['method'],32,100*p['primary_mean'],p['mean_complete_bytes'],p['mean_bits'],p['max_bits'],p['strict_all_requests_1p3'],q.get('difference_pp'),q.get('ci95_pp'),[q.get('improved_QA'),q.get('worsened_QA')],p['timing_ms']['encode_ms']['median'],p['timing_ms']['wall_ms']['median']])
lines=['# 新来源32题：DP3与冻结对照','', '新增DP3四格128个真实回答；复用此前同32来源的FP16、原R7、单层Drop4/Exchange共256个回答，不重复推理reference。六方法按各域分别报告，旧32或64开发题不混入。','', '| 域 | 方法 | 主分% | vs原R7 pp | 95%CI pp | 得分升/降题 | mean bits | max bits |','|---|---|---:|---:|---|---|---:|---:|']
for p in points:
 q=pairs.get(f"{p['dataset']}__{p['method']}_minus_original",{}).get('primary')
 lines.append('| '+p['dataset']+' | '+p['method']+' | '+f"{100*p['primary_mean']:.4f}"+' | '+('—'if q is None else f"{q['difference_pp']:+.4f}")+' | '+('—'if q is None else str(q['ci95_pp']))+' | '+('—'if q is None else f"{q['improved_QA']}/{q['worsened_QA']}")+' | '+f"{p['mean_bits']:.6f}"+' | '+f"{p['max_bits']:.6f}"+' |')
lines+=['','DP3的三升级层/十一K donor固定来自旧cal成本表，未用这些新答案重训。表为新来源复测竞赛，非正式500或预注册统计确认；CI未做多重比较校正，不把跨零称等价。','', '基码与残差数不同且实际花费不同：DP3 Exchange相对DP3 DropD逐题多192N+2048字节，并非等字节比较。资源完整计费，旧256份物理证据只按精确SHA复用，新128份owner表重新核；mean bit不代表全部请求达cap。','', '编码、TTFT、wall仅质量运行观察，输出长度/冷shape与并行背景不同，不作受控速度排名。没有跨数据集分数平均。']
with(OUT/'REPORT.zh.md').open('x',encoding='utf-8',newline='\n')as f:f.write('\n'.join(lines)+'\n')
receipt={'status':'PASS_BOUNDED_COMPLETE_DP3_NEW_SOURCE_ANALYSIS','summary_sha256':sha(OUT/'summary.json'),'source_index_sha256':sha(OUT/'source_index.json'),'report_sha256':sha(OUT/'REPORT.zh.md'),'recipe_sha256':sha(__file__),'pair_sha256':{k+'.json':sha(OUT/(k+'.json')) for k in pairs},'new_GPU_calls':0,'remote_mutations':0,'whole_goal_completion_claimed':False}
write(OUT/'ANALYSIS_RECEIPT.json',receipt)
print(json.dumps({'output':str(OUT),'receipt_sha256':sha(OUT/'ANALYSIS_RECEIPT.json'),'new_points':[p for p in points if p['method'].startswith('BRPE3')],'paired_primary':{k:v['primary'] for k,v in pairs.items()}}))
