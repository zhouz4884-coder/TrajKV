"""Official10-annotator TextVQA score, independent-image uncertainty."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap
from analyze import write_csv,paired
from costs import visual_owner,role
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 refs={r['id']:r for r in json.loads(a.references.read_text(encoding='utf-8'))};metric=TextVQAAccuracyEvaluator();rows=[];seen=set();by=defaultdict(dict)
 for run in a.runs:
  for f in (ROOT/'results'/run).rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line)
    if r.get('protocol')!='image_boundary_free_running' or r.get('task_accuracy_eligible') is False:continue
    key=(r['session_id'],r['variant']);assert key not in seen;seen.add(key);ref=refs[r['session_id']]
    prediction=r['turns'][0]['answer'];score=metric.eval_pred_list([{'pred_answer':prediction,'gt_answers':ref['answers']}]);stored,aux=visual_owner(r)
    row={'session_id':r['session_id'],'image_id':r['image_id'],'variant':r['variant'],'score':score,'prediction':prediction,
     'actual_visual_bytes':stored,'fp16_visual_bytes':r['owned_initial']['visual_fp16_denominator'],'role':role(r),'aux_adjustment':aux}
    rows.append(row);by[r['variant']][(r['image_id'],1)]=score
 table=[]
 for v in sorted(by):
  rs=[r for r in rows if r['variant']==v];boot=cluster_bootstrap([(r['image_id'],r['score']) for r in rs],repeats=4000)
  table.append({'variant':v,'images':len(rs),'vqa_accuracy':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],
   'effective_bits':16*sum(r['actual_visual_bytes'] for r in rs)/sum(r['fp16_visual_bytes'] for r in rs),'role':rs[0]['role']})
 write_csv(a.output/'scores.csv',rows);write_csv(a.output/'table.csv',table)
 result={'table':table,'metric':'official MMF TextVQA10-annotator soft accuracy, normalization unchanged','references':str(a.references)}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
