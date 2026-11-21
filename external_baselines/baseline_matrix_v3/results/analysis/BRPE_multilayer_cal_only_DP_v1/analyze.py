"""Exact finite-table fixed-q cal DP. No model/tensor loading or owner changes."""
from collections import defaultdict
from itertools import combinations
import csv,hashlib,json,math
from pathlib import Path
SOURCE=Path('${EVIDENCE_ROOT}/results/round9/exchange_calibration_v2_local_profile_v1');OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(p,rows):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(tuple,list,dict)) else v for k,v in row.items()})
rows=load(SOURCE/'candidate_profiles.json');summary=load(SOURCE/'summary.json');complete=load(SOURCE/'complete.json');profile=load(SOURCE/'profile.json')
assert summary['status']=='PASS_CAL_ONLY_CONDITIONAL_EXCHANGE_PROFILE' and summary['actual_GPU'] is True and summary['references_read'] is False
assert sha(SOURCE/'candidate_profiles.json')==summary['candidates_sha256']
assert sha(SOURCE/'summary.json')==complete['summary_sha256'] and sha(SOURCE/'profile.json')==complete['profile_sha256']==summary['profile_sha256']
assert len(rows)==28 and {r['upgraded_layer'] for r in rows}==set(range(28))
assert summary['phases_used']==['suffix_last'] and summary['calibration_sources']==32 and summary['actual_events']==896
base=summary['base_route_loss'];by_layer={r['upgraded_layer']:r for r in rows}
costs={l:{tuple(x['unit']):x['cost'] for x in r['conditional_donor_costs']} for l,r in by_layer.items()}
units=set(costs[0]);assert len(units)==23 and all(set(c)==units for c in costs.values())
assert all(len(u)==3 and u[2]==0 for u in units) and len({u[:2] for u in units})==23
assert all(math.isfinite(v) for c in costs.values() for v in c.values())
baseline={};local_units=defaultdict(list)
for unit in sorted(units):
 values=[costs[l][unit] for l in range(28) if l!=unit[0]]
 assert max(values)-min(values)<1e-12,'non-upgraded donor costs inconsistent across layer hypotheses'
 baseline[unit]=values[0];local_units[unit[0]].append(unit)
for l,row in by_layer.items():
 for x in row['conditional_donor_costs']:assert x['evaluated_under_new_layer_base_and_RE'] is (x['unit'][0]==l)
 chosen=sorted(units,key=lambda u:(costs[l][u],u))[:4]
 pred=base+row['upgrade_keep_all_K_residual_route_delta']+math.fsum(costs[l][u] for u in chosen)
 assert abs(pred-row['route_loss_predicted'])<1e-12

def selected_cost(S,D):
 S=set(S);drop_cost={u:(costs[u[0]][u] if u[0] in S else baseline[u]) for u in units}
 donors=sorted(units,key=lambda u:(drop_cost[u],u))[:D]
 delta=math.fsum(by_layer[l]['upgrade_keep_all_K_residual_route_delta'] for l in S)+math.fsum(drop_cost[u] for u in donors)
 return delta,tuple(donors)
def details(U,D,delta,layers,donors):
 constant=732+3*D;slope=64*U-18*D
 dmin=math.ceil((64*256*U+732)/(18*256-3))
 return {'U':U,'D':D,'upgraded_layers':list(layers),'dropped_k_units':[list(x) for x in donors],
  'predicted_cal_route_loss':base+delta,'predicted_delta':delta,'predicted_change_percent':100*delta/base,
  'retained_K_stages':23-D,'V_stages':20,'total_stages':43-D,
  'byte_constant':constant,'byte_slope_per_N':slope,'delta_bytes_N256':constant+slope*256,'delta_bytes_N1792':constant+slope*1792,
  'all_N_ge256_budget_eligible':constant+slope*256<=0 and slope<=0,'Dmin_robust':dmin}

# DP layer x upgraded-layer count x donor count; each KV-head owns one stage0
# donor, so k cheapest local choices are exact under the selected layer format.
dp={(0,0):(0.,(),())}
for l in range(28):
 nxt={}
 for (U,D),(value,layers,donors) in dp.items():
  for up in (0,1):
   if U+up>6:continue
   sort=sorted(local_units[l],key=lambda u:((costs[l][u] if up else baseline[u]),u))
   for k in range(len(sort)+1):
    key=(U+up,D+k)
    if key[1]>23:continue
    chosen=tuple(sort[:k]);inc=(by_layer[l]['upgrade_keep_all_K_residual_route_delta'] if up else 0.)+math.fsum(costs[l][u] if up else baseline[u] for u in chosen)
    record=(value+inc,layers+((l,) if up else ()),donors+chosen)
    old=nxt.get(key)
    if old is None or record<old:nxt[key]=record
 dp=nxt
states=[details(U,D,*value) for (U,D),value in sorted(dp.items()) if U>0]
best_exact_min=[];best_feasible=[]
for U in range(1,7):
 dmin=math.ceil((64*256*U+732)/(18*256-3))
 best_exact_min.append(next(r for r in states if r['U']==U and r['D']==dmin))
 feasible=[r for r in states if r['U']==U and r['all_N_ge256_budget_eligible']]
 best_feasible.append(min(feasible,key=lambda r:(r['predicted_cal_route_loss'],r['D'],r['upgraded_layers'],r['dropped_k_units'])))

# Independent complete enumeration of the originally requested convenient
# D=4U subfamily for U2/U3. This is not needed by DP but checks its exactness.
enumerated=[];enum_best=[]
for U in (2,3):
 D=4*U;group=[]
 for S in combinations(range(28),U):
  delta,donors=selected_cost(S,D);r=details(U,D,delta,S,donors);group.append(r)
 expected=math.comb(28,U);assert len(group)==expected
 best=min(group,key=lambda r:(r['predicted_cal_route_loss'],r['upgraded_layers'],r['dropped_k_units']))
 assert abs(best['predicted_delta']-dp[U,D][0])<1e-12
 enum_best.append(best);enumerated+=group
one=details(1,4,*dp[1,4]);assert abs(one['predicted_cal_route_loss']-profile['route_loss_predicted_from_conditional_marginals'])<1e-12
assert one['upgraded_layers']==[profile['upgraded_layer']]

# Static cal/byte frontier depends on the N at which byte lines are compared.
frontiers={}
for N in (256,1792):
 field='delta_bytes_N'+str(N);eligible=[r for r in states if r['all_N_ge256_budget_eligible']]
 frontier=[]
 for row in eligible:
  dominated=any(other['predicted_cal_route_loss']<=row['predicted_cal_route_loss'] and other[field]<=row[field] and
    (other['predicted_cal_route_loss']<row['predicted_cal_route_loss'] or other[field]<row[field]) for other in eligible)
  if not dominated:frontier.append(row)
 frontiers[str(N)]=sorted(frontier,key=lambda r:(r[field],r['predicted_cal_route_loss']))
table(OUT/'DP_all_states.csv',states);table(OUT/'enumeration_U2D8_U3D12.csv',enumerated)
table(OUT/'best_exact_Dmin.csv',best_exact_min);table(OUT/'best_budget_feasible_per_U.csv',best_feasible)
result={'schema':'trajkv.round9.BRPE_multilayer_cal_only_DP.v1','status':'PASS_FINITE_CONDITIONAL_TABLE_DP_CAL_ONLY_NOT_MODEL_OR_OWNER',
 'base_cal_route_loss':base,'original_single_layer_reproduced':one,'best_exact_Dmin':best_exact_min,'best_budget_feasible_per_U':best_feasible,
 'u2_d8_u3_d12_full_enumeration_best':enum_best,'full_enumeration_counts':{'U2D8':378,'U3D12':3276},
 'static_cal_byte_frontiers_at_N':frontiers,'DP_final_states':len(states),
 'byte_law':'U>0: 64NU-18ND+2048(high shared once)+28(format)+3D(donors)-1344(lookup)=732+3D+(64U-18D)N',
 'Dmin_law':'ceil((64*256*U+732)/(18*256-3))','Dmin_U1_to_6':[r['D'] for r in best_exact_min],
 'current_convenient_drop4U_law':'732+12U-8UN; U2D8=756-16N, U3D12=768-24N, but U3D11 is already budget-legal',
 'source_sha256':{str(SOURCE/n):sha(SOURCE/n) for n in ('candidate_profiles.json','summary.json','profile.json','complete.json')},
 'calibration_scope':{'sources':32,'events':896,'phase':'suffix_last','dataset_mix':summary['source_dataset_counts'],'fixed_student_q_text':True,'references_read':False},
 'exactness_conditions':['All23 donors are distinct KV-head stage0 units','Per-head/layer source-balanced fixed-q loss is additive','Upgrade all4KVheads of each chosen layer','Donor costs use upgraded-base+RE cost only when that donor layer is upgraded; otherwise original-base cost'],
 'limitations':['Not whole-model/end-to-end task optimum; later-layer q changes are not represented','No new owner or GPU reader implemented or executed','Byte laws are exact symbolic accounting under the stated future layout, not measured multilevel deployment','Current frozen single-layer profile and active jobs unchanged','Selection uses cal-only tables, no task answers or scores'],
 'new_model_forwards':0,'GPU_calls':0,'remote_calls':0,'Git_calls':0}
write(OUT/'summary.json',result)
print(json.dumps({'status':result['status'],'base':base,'Dmin':result['Dmin_U1_to_6'],'best_feasible':[{k:r[k] for k in ('U','D','upgraded_layers','predicted_cal_route_loss','predicted_change_percent','delta_bytes_N256','delta_bytes_N1792')} for r in best_feasible]}))
