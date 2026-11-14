"""Free end-to-end GPU answers, separate from teacher-forcing diagnostics."""
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze_cycle import cluster_bootstrap
from costs import visual_owner,role
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-images',type=int);p.add_argument('--start-image',type=int,default=0);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 all_refs=json.loads((ROOT/'configs/a2/mechanism100_refs.json').read_text());selected=all_refs[a.start_image:a.start_image+a.limit_images if a.limit_images else None];refs={r['id']:r for r in selected}
 rows=[];seen=set()
 for run in a.runs:
  for f in (ROOT/'results'/run).rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line)
    if r.get('session_id') not in refs or r.get('protocol')!='image_boundary_free_running' or r.get('task_accuracy_eligible') is False or r.get('primary_quality_eligible') is False:continue
    key=(r['session_id'],r['variant'])
    if key in seen:raise RuntimeError('Duplicate image/variant')
    seen.add(key);score=score_docvqa(r['turns'][0]['answer'],refs[r['session_id']]['answer']);owned=r['owned_initial']
    stored,aux_adjustment=visual_owner(r)
    rows.append({'id':r['session_id'],'image_id':r['image_id'],'variant':r['variant'],'anls':score['score'],'exact':score['exact'],
      'digit_bearing':refs[r['session_id']]['digit_bearing'],'oracle':r.get('oracle',False),'method':r['method'],'role':role(r),'prediction':r['turns'][0]['answer'],'reference':str(refs[r['session_id']]['answer']),
      'stored_visual_bytes':stored,'fp16_visual_bytes':owned['visual_fp16_denominator'],'effective_bits':16*stored/owned['visual_fp16_denominator'],
      'aux_owner_adjustment_bytes':aux_adjustment,'session_ms':r['session_ms'],'encoding_ms':r.get('encoding_wall_ms'),'cuda_peak':r['peak_allocated']})
 table=[]
 for v in sorted({r['variant'] for r in rows}):
  rs=[r for r in rows if r['variant']==v];num=[r for r in rs if r['digit_bearing']]
  boot=cluster_bootstrap([(r['image_id'],r['anls']) for r in rs],repeats=4000)
  table.append({'variant':v,'role':rs[0]['role'],'images':len(rs),'anls':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],'exact':float(np.mean([r['exact'] for r in rs])),
   'numeric_anls':float(np.mean([r['anls'] for r in num])) if num else None,'effective_bits':16*sum(r['stored_visual_bytes'] for r in rs)/sum(r['fp16_visual_bytes'] for r in rs),
   'session_ms_median':float(np.median([r['session_ms'] for r in rs]))})
 for name,data in [('scores',rows),('table',table)]:
  if data:
   with (a.output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (a.output/'summary.json').write_text(json.dumps({'records':len(rows),'table':table,'scope':'old-pool mechanism development, free end-to-end GPU generation; actual n reported per variant; numeric strata diagnostic'},indent=2));print(json.dumps({'records':len(rows),'table':table}))
if __name__=='__main__':main()
