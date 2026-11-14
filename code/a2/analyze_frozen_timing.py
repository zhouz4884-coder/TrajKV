"""Audit fixed-token multi-question timing directly from raw measured trials."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze import write_csv
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='frozen_timing_gpu0_v1');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 folder=ROOT/'results'/a.run;complete=json.loads((folder/'complete.json').read_text());manifest=json.loads((folder/'manifest.json').read_text());cfg=manifest['config']
 provenance={}
 if cfg.get('frozen_contract_sha256'):
  contracts=[p for p in (ROOT/'audit/a2').glob('*.json') if hashlib.sha256(p.read_bytes()).hexdigest()==cfg['frozen_contract_sha256']]
  assert len(contracts)==1;frozen=json.loads(contracts[0].read_text());checks=frozen.get('inference_source_sha256',{})
  for name,digest in checks.items():assert manifest['sources'][name]==digest
  for path,digest in manifest['external_codec_inputs'].items():assert frozen['external_codec_inputs_sha256'][path]==digest
  variants={v['name']:v for v in frozen['variants']}
  assert all(v==variants[v['name']] for v in cfg['variants'])
  provenance={'frozen_inference_source_checks':len(checks),'external_codec_input_checks':len(manifest['external_codec_inputs']),
   'variant_configs_match_frozen_quality':True,'frozen_contract_sha256':cfg['frozen_contract_sha256']}
 canonical=json.loads((folder/'length_controls.json').read_text());schedule=json.loads((folder/'schedule.json').read_text());records=[json.loads(x) for x in (folder/'timing_records.jsonl').read_text().splitlines()]
 assert complete['status']=='COMPLETE' and complete['quality']=='not_scored';assert len(records)==len(schedule)
 expected={(sid,v['name'],r) for sid in canonical for v in cfg['variants'] for r in range(cfg['repeats'])};seen=set();positions={};rows=[]
 for record,planned in zip(records,schedule):
  assert all(record[k]==planned[k] for k in ('phase','repeat','session_id','variant'));assert record['quality']=='not_scored' and record['task_accuracy_eligible'] is False and record['quiet_verified']
  result=record['runner_result'];assert result['protocol']=='image_boundary_fixed_tokens';assert result['read_ledger_exact'] and result.get('read_budget_ok',True) and result.get('global_extra_budget_ok',True)
  assert [t['token_ids'] for t in result['turns']]==[t['token_ids'] for t in canonical[record['session_id']]]
  cap=result.get('max_patches',result.get('variant_config',{}).get('cap'))
  # The producer stores variant configuration in the timing contract rather
  # than mutating the runner result after its measured path.
  cap=next(v['cap'] for v in cfg['variants'] if v['name']==record['variant']);key=(record['session_id'],cap)
  pos=[(t['logical_end'],t['forced_chat_closure']) for t in result['turns']]
  assert key not in positions or positions[key]==pos;positions[key]=pos
  if record['phase']!='measurement':continue
  identity=(record['session_id'],record['variant'],record['repeat']);assert identity not in seen;seen.add(identity)
  intervals=sum(max(0,len(t['token_ids'])-1) for t in result['turns']);it=sum(max(0,len(t['token_ids'])-1)*t['mean_inter_token_ms'] for t in result['turns'])/intervals
  rows.append({'session_id':record['session_id'],'variant':record['variant'],'repeat':record['repeat'],'cap':cap,'session_ms':result['session_ms'],
    'wall_ms_with_setup':record['wall_ms_including_runner_setup'],'first_ttft_ms':result['turns'][0]['ttft_ms'],'mean_it_ms':it,'intervals':intervals,
    'output_tokens':sum(len(t['token_ids']) for t in result['turns']),'encoding_ms':result['encoding_wall_ms'],'peak_allocated':result['peak_allocated'],
    'extra_logical_read_bytes':result['extra_logical_read_bytes']})
 assert seen==expected and len(rows)==complete['measurement_trials'];table=[]
 for v in cfg['variants']:
  subset=[r for r in rows if r['variant']==v['name']];entry={'variant':v['name'],'cap':v['cap'],'images':len(canonical),'measurements':len(subset)}
  for field in ('session_ms','wall_ms_with_setup','first_ttft_ms','mean_it_ms','encoding_ms','peak_allocated','extra_logical_read_bytes'):
   values=np.array([r[field] for r in subset]);entry[field+'_median']=float(np.median(values));entry[field+'_p25']=float(np.quantile(values,.25));entry[field+'_p75']=float(np.quantile(values,.75))
  table.append(entry)
 write_csv(a.output/'measurements.csv',rows);write_csv(a.output/'table.csv',table)
 result={'status':'PASS','measured_trials':len(rows),'warmups':len(records)-len(rows),'output_ids_identical_each_question':True,'logical_positions_identical_at_same_cap':True,
  'quiet_verified_all_trials':True,'quality':'not_scored','table':table,
  'scope':f"{len(canonical)} development images, same{cfg['tokens_per_turn']} forced IDs per question; medians/IQR across{cfg['repeats']} repeats; cap groups{sorted({v['cap'] for v in cfg['variants']})} use separate input settings",
  'producer_config_sha256':manifest['config_sha256'],**provenance}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'PASS','measurements':len(rows),'warmups':len(records)-len(rows)}))
if __name__=='__main__':main()
