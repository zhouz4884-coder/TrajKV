"""Image-clustered summaries of full-model query/key diagnostic probes."""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_cycle import cluster_bootstrap
from analyze import write_csv

def main():
 records={};metrics=['centered_key_cosine','full_key_cosine','centered_key_mse','centered_norm_ratio','query_cosine_to_full',
  'direct_score_rmse_fixed_full_query','state_score_rmse','interaction_score_rmse','total_score_rmse','visual_only_attention_kl_fixed_full_query','visual_top1_match_fixed_full_query']
 for run in ['book_geometry16_s0_v1','book_geometry16_s1_v1']:
  folder=ROOT/'results'/run;receipt=json.loads((folder/'complete.json').read_text());assert receipt['status']=='COMPLETE' and not receipt['task_accuracy_eligible']
  manifest=json.loads((folder/'manifest.json').read_text());assert not manifest['task_accuracy_eligible']
  for line in (folder/'geometry.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant'],r['layer']);assert key not in records and not r['task_accuracy_eligible'];records[key]=r
 assert len(records)==3136;ids=sorted({key[0] for key in records});variants=sorted({key[1] for key in records});assert len(ids)==16 and len(variants)==7
 fixed_checks=0
 for sid in ids:
  for layer in range(28):
   for left,right in [('fwht_global','decoder_warm_joint_vonly'),('decoder_warm_joint','decoder_warm_joint_konly')]:
    a=records[sid,left,layer];b=records[sid,right,layer]
    for field in ['centered_key_cosine','full_key_cosine','centered_key_mse','centered_norm_ratio','direct_score_rmse_fixed_full_query','visual_only_attention_kl_fixed_full_query','visual_top1_match_fixed_full_query']:
     assert a[field]==b[field],(sid,left,right,layer,field)
    fixed_checks+=1
 image_rows=[];layer_rows=[];table=[]
 for variant in variants:
  for sid in ids:image_rows.append({'session_id':sid,'variant':variant,**{key:float(np.mean([records[sid,variant,l][key] for l in range(28)])) for key in metrics}})
  for layer in range(28):layer_rows.append({'layer':layer,'variant':variant,**{key:float(np.mean([records[sid,variant,layer][key] for sid in ids])) for key in metrics}})
  entry={'variant':variant,'images':16,'layers':28}
  for key in metrics:
   boot=cluster_bootstrap([(r['session_id'],r[key]) for r in image_rows if r['variant']==variant],repeats=8000)
   entry.update({key:boot['mean'],key+'_ci_low':boot['ci_lower'],key+'_ci_high':boot['ci_upper']})
  table.append(entry)
 out=ROOT/'results/book_geometry16_analysis';out.mkdir(exist_ok=True,parents=True)
 write_csv(out/'table.csv',table);write_csv(out/'image_metrics.csv',image_rows);write_csv(out/'layer_metrics.csv',layer_rows)
 result={'status':'PASS','layer_records':len(records),'images':16,'task_accuracy_eligible':False,'fixed_key_functions_identical_pairs':fixed_checks,
  'max_additive_identity_residual':max(r['decomposition_max_residual'] for r in records.values()),'table':table,
  'raw_score_limitation':'raw score RMSE is uncentered and sensitive to common-mode shifts; it cannot establish softmax disruption or state dominance; do not interpret component magnitudes as causal shares',
  'scope':'actual packed first-question prefill queries plus extra FP16 keys for diagnostics; source-image bootstrap of layer means; direct/state/interaction score identity is not a loss or causal-energy decomposition; visual-only attention KL is not vocabulary KL'}
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'PASS','fixed_key_pairs':fixed_checks,'table':[{key:r[key] for key in ['variant','centered_key_cosine','centered_norm_ratio','query_cosine_to_full','direct_score_rmse_fixed_full_query','state_score_rmse','total_score_rmse']} for r in table]}))

if __name__=='__main__':main()
