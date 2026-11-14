"""Summarize actual-query local diagnostics at independent image level."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from analyze import write_csv
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='geometry20_gpu0_v1');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 all_rows=[];image_rows=[];checks=[]
 for f in (ROOT/'results'/a.run).rglob('records.jsonl'):
  for line in f.read_text().splitlines():
   r=json.loads(line);assert not r['task_accuracy_eligible'];groups=defaultdict(list)
   for z in r['geometry_diagnostic']:
    groups[z['encoder']].append(z);all_rows.append(dict(image_id=r['image_id'],**z))
   for name,rows in groups.items():
    columns=['normalized_key_mse','mean_cosine','logit_correlation','logit_snr','logit_top1_agreement','logit_kl_visual','top_gap_over_logit_std','visual_output_relative_squared_error']
    image_rows.append({'image_id':r['image_id'],'encoder':name,**{k:float(np.mean([z[k] for z in rows if z[k] is not None])) for k in columns},
      'snr_gt1_and_wrong_top1_fraction':float(np.mean([z['logit_snr']>1 and z['logit_top1_agreement']==0 for z in rows])),
      'snr_lt1_and_correct_top1_fraction':float(np.mean([z['logit_snr']<1 and z['logit_top1_agreement']==1 for z in rows]))})
   checks += [dict(image_id=r['image_id'],**z) for z in r['calib_actual_model_operator_checks']]
 table=[]
 for enc in sorted({r['encoder'] for r in image_rows}):
  rows=[r for r in image_rows if r['encoder']==enc]
  table.append({'encoder':enc,'images':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('image_id','encoder')}})
 write_csv(a.output/'query_head_diagnostics.csv',all_rows);write_csv(a.output/'image_diagnostics.csv',image_rows);write_csv(a.output/'table.csv',table);write_csv(a.output/'official_operator_checks.csv',checks)
 result={'table':table,'official_operator_cases':len(checks),'official_qk_max_abs':max(r['qk_max_abs'] for r in checks),'official_wv_max_abs':max(r['wv_max_abs'] for r in checks),
  'scope':'FP16 trajectory, same real last prefix query; visual-only local attention, not answer quality; image-level summaries'}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
