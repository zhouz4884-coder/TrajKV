"""Completed frozen8x32 new-source tournament, no GPU/network/refitting."""
import sys,pathlib,json,hashlib,statistics,csv
sys.dont_write_bytecode=True
ROOT=pathlib.Path('${PROJECT_ROOT}');E=pathlib.Path('${EVIDENCE_ROOT}/results/round9_v1')
REFS=pathlib.Path('${EVIDENCE_ROOT}/results/round9/new_confirmation_reserve32x2_v1');OUT=pathlib.Path('${EVIDENCE_ROOT}/results/round9/new_source_replication_v1/report')
sys.path.insert(0,str(ROOT/'ops'))
import round9_compare_dev_v2 as comp
import verify_public_custom_driver_v1 as physics
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def stat(x):return {'mean':statistics.mean(x),'median':statistics.median(x),'min':min(x),'max':max(x)} if x else None
DRIVER='cf4885b77b03fd94aadd5bbabc5a1e6d62e3259ba6037b7a708fabab71758961';HELPER='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe';PROFILE='7ec305a76cebf5ca9e358842e11e68fe6a77e6af9f2d83c39af723ff17eb5c22'
modes=['FP16','original','BRPE_Drop4','BRPE_Exchange'];datasets=['textvqa','infovqa']
folders={(ds,m):E/'scored'/f'{m}_{ds}_new32_v1' for ds in datasets for m in modes}
missing=[str(f/'summary.json') for f in folders.values() if not(f/'summary.json').is_file()]
if missing:print(json.dumps({'status':'WAIT_ALL_EIGHT_ACTUAL_SOURCE_SCORE_CLOSURES','missing':missing,'report_written':False}));raise SystemExit(3)
assert not OUT.exists(),'Use a new immutable report directory'
reserve=REFS/'CONFIRMATION_RESERVE_MANIFEST.json';reserve_data=json.loads(reserve.read_text())
assert reserve_data['selection_reads_model_outputs_or_correctness'] is False and reserve_data['refs_decoded_only_after_ID_selection'] is True
assert reserve_data['source_cluster_rawSHA_intersection_with_current160_formal2000_historical_compression_or_confirm']==0
before={str(pathlib.Path(m.__file__).resolve()):sha(m.__file__) for m in [comp,physics]};before[str(reserve)]=sha(reserve)
loaded={};raws={};inventory={};points=[];pairs={};image_hashes=set();cal_images=set()
common=['model_spec','torch_version','transformers_version','CUDA_version','python_version','weight_dtype','tf32','seed','attention_implementation','processor_id','processor_version','processor_file_hashes','generation_native','generation_overrides','tokenizer_class','tokenizer_is_fast','tokenizers_version','decode_overrides','runtime_library_source_sha256','frozen_variant_sha256','asset_sha256']
for ds in datasets:
 references=REFS/(ds+'_confirm32_refs.jsonl');inputs=REFS/(ds+'_confirm32_inputs.jsonl')
 before[str(references)]=sha(references);before[str(inputs)]=sha(inputs)
 for mode in modes:
  name=f'{mode}_{ds}_new32_v1';folder=folders[ds,mode];su,rows=comp.load(folder);m=su['_raw_manifest'];raw=pathlib.Path(su['integrity']['raw_dir'])
  assert su['status']=='PASS_SCORED_COMPLETE' and su['expected_run_questions']==32 and su['groups']['overall']['failed_ids']==[]
  assert pathlib.Path(su['integrity']['refs_path']).resolve()==references.resolve() and su['integrity']['refs_sha256']==sha(references)
  assert su['integrity']['inputs_sha256']==m['inputs_sha256']==sha(inputs)
  assert su['integrity']['scorer_sha256']=='72b5182eeb97e930403a970feebb4f1fadac23d04544c9646c6c9fd6b34bfba6'
  rr=[json.loads(l) for l in (raw/'records.jsonl').read_text(encoding='utf-8').splitlines()];assert len(rr)==len(rows)==32
  proof=json.loads((raw/'source_retrieval_receipt_round9_v2.json').read_text());launch=json.loads((raw/'launch_receipt.json').read_text());iv=json.loads((raw/'source_inventory.json').read_text())
  assert proof['producer_name']=='bm_R9_'+name and proof['actual_driver']=='code/round9_runner_v4.py'
  assert m['source_hashes']['code/round9_runner_v4.py']==sha(raw/'source_driver.py')==iv['code/round9_runner_v4.py']==DRIVER
  assert sha(raw/'source_inventory.json')==launch['source_inventory_sha256'] and launch['argv_executed'][2]==m['runtime_source_root']+'/code/round9_runner_v4.py'
  if mode.startswith('BRPE_'):
   assert m['source_hashes']['code/round9_precision_exchange_v1.py']==HELPER and m['candidate']['profile_sha256']==sha(raw/'assets/candidate_profile.json')==PROFILE
   pr=json.loads((raw/'assets/candidate_profile.json').read_text());assert sha(raw/'assets/candidate_high_book.json')==pr['high_book_sha256']
   cal_images.update(x['image_sha256'] for x in pr['calibration_source_identities'])
  if mode!='FP16':
   reference=loaded[ds,'FP16'][0]['_raw_manifest']
   for k in common:assert m[k]==reference[k],('Runtime/model/processor contract differs',ds,mode,k)
   assert m['source_hashes']==reference['source_hashes'] and m['expected_ids']==reference['expected_ids']
  amounts=[];bits=[]
  for r,row in zip(rr,rows):
   assert r['id']==row['id'] and r['status']=='PASS';b=physics.physical_complete_bytes(r,m['model_spec']);amounts.append(b);bits.append(float(r['actual_effective_visual_bits']))
   assert b==row['stored_visual_KV_bytes'];n=r['actual_visual_token_count'];b0=3816*n+89733+18*n*43
   if mode=='FP16':assert b==r['full_kv_byte_denominator']+8*n
   elif mode=='original':assert b==b0
   else:
    delta=-72*n-1304 if mode=='BRPE_Drop4' else 744-8*n
    assert b==b0+delta<=b0 and r['r7_bridge']['selected_stage_count']==39 and sum(x[0]=='v' for x in r['r7_bridge']['selected_stages'])==20
    assert r['BRPE_budget_vs_original_R7_low']['actual_complete_bytes']==b
   image_hashes.add(r['original_image_sha256'])
  loaded[ds,mode]=(su,rows);raws[ds,mode]=rr
  inventory[name]={'raw':str(raw),'scored':str(folder),'raw_file_sha256':su['integrity']['raw_files'],'summary_sha256':sha(folder/'summary.json'),'scored_records_sha256':sha(folder/'scored_records.jsonl'),'refs_file':str(references),'refs_sha256':sha(references),'inputs_sha256':sha(inputs),'actual_driver_sha256':DRIVER,'source_receipt_sha256':sha(raw/'source_retrieval_receipt_round9_v2.json'),'source_inventory_sha256':sha(raw/'source_inventory.json'),'launch_sha256':sha(raw/'launch_receipt.json'),'source_hashes':m['source_hashes']}
  points.append({'dataset':ds,'method':mode,'QA':32,'primary_mean':su['primary_mean'],'primary_ci95':su['primary_ci95'],'metric_id':rows[0]['metric_id'],'mean_complete_bytes':statistics.mean(amounts),'max_complete_bytes':max(amounts),'mean_bits':statistics.mean(bits),'max_bits':max(bits),'strict_all_requests_1p3':all(160*b<=13*r['full_kv_byte_denominator'] for b,r in zip(amounts,rr)),'strict_all_requests_1p8':all(80*b<=9*r['full_kv_byte_denominator'] for b,r in zip(amounts,rr)),'timing_ms':{k:stat([float(r[k]) for r in rr if r.get(k) is not None]) for k in ['encode_ms','ttft_ms','question_prefill_ms','wall_ms']},'output_tokens':stat([r['output_tokens'] for r in rr]),'output_at64cap':sum(r['output_tokens']==64 for r in rr),'prototype_not_controlled_speed':True,'physical_owner_tables_reconstructed':32})
 for candidate,anchor in [('original','FP16'),('BRPE_Drop4','original'),('BRPE_Exchange','original'),('BRPE_Drop4','FP16'),('BRPE_Exchange','FP16'),('BRPE_Exchange','BRPE_Drop4')]:
  value=comp.compare(folders[ds,anchor],[folders[ds,candidate]]);assert value['status']=='PASS_ALL_DECLARED_CANDIDATE_IDS' and value['anchor_reference_QA']==value['candidate_QA']==32
  pairs[f'{ds}__{candidate}_minus_{anchor}']=value
assert len(image_hashes)==64 and not(image_hashes&cal_images),'Fresh inputs overlap calibration image identities'
assert all(sha(p)==h for p,h in before.items()),'Analysis/refs/source preparation changed'
OUT.mkdir(parents=True,exist_ok=False)
for label,value in pairs.items():write(OUT/(label+'.json'),value)
write(OUT/'source_index.json',inventory)
summary={'schema':'trajkv.round9.new_source_replication_tournament.v1','status':'PASS_COMPLETE_EIGHT_NEW_SOURCE_RUNS_AND_MATCHED_PAIRS','actual_new_answer_events':256,'unique_new_sources':64,'QA_per_cell':32,'datasets':datasets,'methods':modes,'points':points,'paired_comparisons':pairs,'source_index_sha256':sha(OUT/'source_index.json'),'source_preparation_manifest':str(reserve),'source_preparation_manifest_sha256':sha(reserve),'calibration_image_overlap_checked':0,'profiles_frozen_and_not_refitted_on_these_answers':True,'profile_sha256':PROFILE,'actual_new_model_or_GPU_actions_by_analysis':0,'SourceRun_admission_established':False,'formal500_or_preregistered_statistical_confirmation_claimed':False,'statistical_scope':'new-source replication tournament with descriptive source-cluster paired bootstrap; no multiplicity-adjusted winner/ equivalence claim','no_cross_dataset_quality_average':True,'timing_scope':'Instrumented quality-run observations with answer-length/cold-shape/background differences; no controlled speed ranking.','analysis_sources':before,'private_PT_WO_or_images_read':False,'physical_owner_reconstruction':'all256 actual request owner tables, no tensor/GPU replay'}
write(OUT/'summary.json',summary)
with (OUT/'table.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['dataset','method','QA','score_percent','CI95_percent','mean_complete_bytes','mean_bits','max_bits','strict_all_1p3','delta_to_original_pp','paired_CI95_pp','higher_lower_QA','encode_median_ms','wall_median_ms'])
 for p in points:
  d=p['dataset'];m=p['method'];pair=pairs.get(f'{d}__{m}_minus_original',{})
  q=pair.get('primary',{})
  w.writerow([d,m,32,100*p['primary_mean'],[100*x for x in p['primary_ci95']],p['mean_complete_bytes'],p['mean_bits'],p['max_bits'],p['strict_all_requests_1p3'],q.get('difference_pp'),q.get('ci95_pp'),[q.get('improved_QA'),q.get('worsened_QA')],p['timing_ms']['encode_ms']['median'],p['timing_ms']['wall_ms']['median']])
lines=['# Round9 新来源32×2复测','', '每域FP16、原R7、Drop4、Exchange各32个真实新回答，共256。四种方法同源、同图、同问题、同模型/位置/运行库与冻结主评分；refs仅来自这批新冻结名单。没有使用旧dev64 refs，也未用这些答案重训原两个版本。','', '| 域 | 方法 | 主分% | 相对原R7 pp | 配对95%CI pp | 得分上升/下降题 | mean bits | max bits | 全请求≤1.3 |','|---|---|---:|---:|---|---|---:|---:|---|']
for p in points:
 q=pairs.get(f"{p['dataset']}__{p['method']}_minus_original",{}).get('primary')
 lines.append('| '+p['dataset']+' | '+p['method']+' | '+f"{100*p['primary_mean']:.4f}"+' | '+('—'if q is None else f"{q['difference_pp']:+.4f}")+' | '+('—'if q is None else str(q['ci95_pp']))+' | '+('—'if q is None else f"{q['improved_QA']}/{q['worsened_QA']}")+' | '+f"{p['mean_bits']:.6f}"+' | '+f"{p['max_bits']:.6f}"+' | '+str(p['strict_all_requests_1p3'])+' |')
lines+=['','这是新来源复测竞赛，不是正式500题或外部预注册的统计确认。FP16是经验参考点，不是数学质量上界；配对CI跨零不等于等价。各域单独解释，不平均不同主指标。','', '完整bytes含实际命名tensor容量与共享存储去重，256份owner表已重建。Exchange与Drop4成本不同，直接差值不作等字节因果结论。encode/TTFT/wall仅质量运行观察，不作受控提速排名。']
with (OUT/'REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
receipt={'status':'PASS_BOUNDED_EIGHT_NEW_SOURCE_REPLICATION_ANALYSIS','summary_sha256':sha(OUT/'summary.json'),'source_index_sha256':sha(OUT/'source_index.json'),'report_sha256':sha(OUT/'REPORT.zh.md'),'recipe_sha256':sha(__file__),'pair_files_sha256':{n+'.json':sha(OUT/(n+'.json')) for n in pairs},'remote_mutations':0,'GPU_launches':0,'goal_complete_claimed':False}
write(OUT/'ANALYSIS_RECEIPT.json',receipt)
print(json.dumps({'output':str(OUT),'receipt_sha256':sha(OUT/'ANALYSIS_RECEIPT.json'),'points':[{k:p[k] for k in ['dataset','method','primary_mean','mean_bits','max_bits','strict_all_requests_1p3']} for p in points],'pairs':{k:v['primary'] for k,v in pairs.items()}}))
