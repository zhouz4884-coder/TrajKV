"""Multi-choice layer knapsack fitted on same-query development attention error."""
import copy,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT
path=ROOT/'results/a2_refine_dev_gpu1_v1/axis_calib/records.jsonl'
records=[json.loads(s) for s in path.read_text().splitlines()]
allowed=set(json.loads((ROOT/'configs/a2/isolation.json').read_text())['dev_ids'])
assert len(records)==8 and {r['session_id'] for r in records}==allowed
gains=defaultdict(list)
for r in records:
 for g in r['local_axis_calibration']:gains[g['layer']].append(g)
table=[]
for i in range(28):
 rows=gains[i];assert len(rows)==16
 avg={k:sum(r[k] for r in rows)/len(rows) for k in ('mse_base','mse_kr','mse_vr','mse_both')}
 table.append({'layer':i,'events':len(rows),**avg})
def solve(slots):
 dp={0:(0.,[])}
 for row in table:
  options=[(0,0.,(False,False)),(2,row['mse_base']-row['mse_kr'],(True,False)),
    (1,row['mse_base']-row['mse_vr'],(False,True)),(3,row['mse_base']-row['mse_both'],(True,True))]
  nxt={}
  for used,(value,path0) in dp.items():
   for weight,gain,flags in options:
    new=used+weight
    if new<=slots and (new not in nxt or value+gain>nxt[new][0]):nxt[new]=(value+gain,path0+[flags])
  dp=nxt
 value,flags=dp[slots]
 return {'name':f'calibrated_s{slots}','method':'alloc1','selector':'all','fraction':1,'store_selector_meta':False,
  'slots':slots,'k_layers':[i for i,(k,v) in enumerate(flags) if k],'v_layers':[i for i,(k,v) in enumerate(flags) if v],
  'estimated_local_gain':value,'calibration_scope':'same-query output MSE to retained FP16 visual, first two decode events, dev8; no future answers'}
variants=[solve(s) for s in (12,24,36)]
base=json.loads((ROOT/'configs/a2/dev_selectors.json').read_text());base['variants']=variants+[
 {'name':s+'_128_bounded','method':'asym','selector':s,'fraction':.25,'global_read_equivalents':32,'page':128} for s in ('random','mass','risk')]
(ROOT/'configs/a2/dev_calibrated.json').write_text(json.dumps(base,indent=2))
audit={'calibration_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'calibration_ids':[r['session_id'] for r in records],
 'events':sum(len(r['local_axis_calibration']) for r in records),'layer_table':table,'rules':variants,'objective':'28-layer multi-choice exact-slot knapsack, using measured none/K/V/both same-query MSE'}
(ROOT/'audit/a2/frozen_axis_tables.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'events':audit['events'],'rules':[{k:r[k] for k in ('name','slots','k_layers','v_layers')} for r in variants]}))
