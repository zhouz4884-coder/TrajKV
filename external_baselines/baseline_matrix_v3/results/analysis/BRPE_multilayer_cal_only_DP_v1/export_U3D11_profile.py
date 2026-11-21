"""Export only the cal-table DP result; no task scores, model or GPU."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).parent
CAL=Path('${EVIDENCE_ROOT}/results/round9/exchange_calibration_v2_local_profile_v1')
OUT=Path('${EVIDENCE_ROOT}/results/round9/BRPE3_U3D11_profile_v1')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=load(HERE/'summary.json');cal=load(CAL/'summary.json');old=load(CAL/'profile.json')
assert s['status']=='PASS_FINITE_CONDITIONAL_TABLE_DP_CAL_ONLY_NOT_MODEL_OR_OWNER'
chosen=next(x for x in s['best_budget_feasible_per_U'] if x['U']==3)
assert chosen['D']==11 and chosen['upgraded_layers']==[14,22,26]
for path,digest in s['source_sha256'].items():assert sha(path)==digest
assert cal['source_provenance']['calibrator']=='8b986799062ef8d3fbc67624344d1740cfea4726b64aa3fc4d995ffaebc54438'
assert cal['source_provenance']['exchange_helper']=='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'
assert cal['source_provenance']['collection_complete']=='c838b3a79cb970233a9579f8972ae4276157fad1001fd9b4e9118945bc0dc521'
profile={'schema':'trajkv.round9.precision_exchange_profile.v2','profile_role':'cal_only_DP_selected_U3D11',
 'base_profile':'r7_joint_k1_gt_cond','upgraded_layers':chosen['upgraded_layers'],'dropped_k_units':chosen['dropped_k_units'],
 'U':3,'D':11,'original_K_stages':23,'retained_K_stages':12,'V_stages_unchanged':20,'total_stages':32,
 'low_book_sha256':old['low_book_sha256'],'base_metadata_factor_sha256':old['base_metadata_factor_sha256'],
 'high_book_path':old['high_book_path'],'high_book_sha256':old['high_book_sha256'],
 'source_provenance':cal['source_provenance'],'collection_provenance':cal['collection_provenance'],
 'selection_provenance':{'method':'CPU exact layer x U x D DP on actual GPU-calibrated conditional costs',
  'DP_script_sha256':sha(HERE/'analyze.py'),'DP_summary_sha256':sha(HERE/'summary.json'),
  'candidate_cost_table_sha256':sha(CAL/'candidate_profiles.json'),'original_single_profile_sha256':sha(CAL/'profile.json'),
  'profile_export_script_sha256':sha(__file__),'task_scores_or_answers_used':False,'new_model_forwards':0,'new_GPU_calls':0},
 'calibration_phase':'suffix_last','calibration_sources':32,'actual_source_table_events':896,
 'base_cal_route_loss':s['base_cal_route_loss'],'predicted_cal_route_loss':chosen['predicted_cal_route_loss'],
 'predicted_cal_change_percent':chosen['predicted_change_percent'],
 'symbolic_complete_byte_delta':'765-6*N','Dmin_all_N_ge256':11,
 'cal_only_fixed_q_head_layer_additive_optimum':True,'whole_model_or_task_optimum_claimed':False,
 'physical_owner_or_GPU_reader_established_for_U3D11':False,'private_shards_copied':False}
OUT.mkdir(parents=True,exist_ok=False)
with (OUT/'profile.json').open('x',encoding='utf-8') as f:json.dump(profile,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
with (OUT/'CPU_SELECTION_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS_CAL_ONLY_DP_U3D11_PROFILE_DERIVED_NO_NEW_GPU',
 'profile_sha256':sha(OUT/'profile.json'),'source_cost_table_sha256':sha(CAL/'candidate_profiles.json'),
 'new_model_forwards':0,'GPU_calls':0,'task_answers_or_scores_read':False,'source_files_modified':False},f,indent=2);f.write('\n')
print(json.dumps({'profile':str(OUT/'profile.json'),'sha256':sha(OUT/'profile.json'),'layers':profile['upgraded_layers'],'donors':profile['dropped_k_units']}))
