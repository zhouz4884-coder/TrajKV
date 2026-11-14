"""New metadata candidates on a previously evaluated transfer cohort."""
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap
from analyze import paired,write_csv
from costs import visual_owner
from audit_book_records import audit

def main():
 frozen=json.loads((ROOT/'audit/a2/metadata_confirmation_v1.json').read_text());lookup={v['name']:v for v in frozen['variants']};variants=[lookup[n] for n in frozen['text_selected_methods']]
 runs=['metadata_confirm_text_s0_v1','metadata_confirm_text_s1_v1'];amendment=ROOT/'audit/a2/metadata_execution_cache_v2.json'
 if amendment.exists():mapping=json.loads(amendment.read_text())['run_mapping'];runs=[mapping.get(name,name) for name in runs]
 records,check=audit(frozen,runs,variants,frozen['input_sha256']['configs/a2/textvqa_future256.json'])
 refs={r['id']:r for r in json.loads((ROOT/'configs/data/textvqa_future256_references.json').read_text(encoding='utf-8'))};expected={(sid,v['name']) for sid in refs for v in variants}
 assert {(r['session_id'],r['variant']) for r in records}==expected and len(records)==frozen['text_free_answers']
 metric=TextVQAAccuracyEvaluator();rows=[];costs=defaultdict(list);by=defaultdict(dict)
 for r in records:
  sid=r['session_id'];ref=refs[sid];assert len(r['turns'])==1 and r['image_id']==ref['image_id']
  score=float(metric.eval_pred_list([{'pred_answer':r['turns'][0]['answer'],'gt_answers':ref['answers']}]))
  row={'session_id':sid,'image_id':r['image_id'],'variant':r['variant'],'score':score,'answer_stratum':ref['answer_stratum'],'prediction':r['turns'][0]['answer']}
  rows.append(row);by[r['variant']][r['image_id'],sid]=score;costs[r['variant']].append(r)
 for row in csv.DictReader((ROOT/'results/book_confirm_text256_analysis/scores.csv').open(encoding='utf-8-sig')):by[row['variant']][row['image_id'],row['session_id']]=float(row['score'])
 table=[]
 for variant,group in costs.items():
  scored=[r for r in rows if r['variant']==variant];boot=cluster_bootstrap([(r['image_id'],r['score']) for r in scored],repeats=8000)
  den=sum(r['owned_initial']['visual_fp16_denominator'] for r in group);store=sum(visual_owner(r)[0]+r['encoder_calibration_parameters_bytes'] for r in group)
  table.append({'variant':variant,'questions':256,'accuracy':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],'budget_bits_including_encoder_constants':16*store/den,
   'numeric_accuracy':float(np.mean([r['score'] for r in scored if r['answer_stratum']=='numeric'])),'text_accuracy':float(np.mean([r['score'] for r in scored if r['answer_stratum']=='text']))})
 comparisons=[]
 for variant in costs:
  for reference in ['fwht_global','base2_8','decoder_warm_joint','full8','native4_8']:comparisons.append({'candidate':variant,'reference':reference,'role':'secondary previously evaluated cohort',**paired(by[variant],by[reference],repeats=8000)})
 out=ROOT/'results/metadata_text256_transfer_analysis';out.mkdir(exist_ok=True,parents=True);write_csv(out/'scores.csv',rows);write_csv(out/'table.csv',table);write_csv(out/'comparisons.csv',comparisons)
 result={'status':'COMPLETE','records':len(records),'table':table,'comparisons':comparisons,'audit':check,'metric':'official MMF10-annotator soft accuracy',
  'scope':'reused already-evaluated256-image transfer cohort; candidate selection solely from Doc development; old controls retain original frozen source records; not a new-image confirmation'}
 (out/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'audit/a2/METADATA_TEXT256_TRANSFER_AUDIT.json').write_text(json.dumps(check,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
