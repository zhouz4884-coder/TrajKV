"""Frozen transfer/full-validation TextVQA scoring with image-clustered tests."""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap
from analyze import paired,write_csv
from analyze_docvqa_confirm64 import holm
from costs import visual_owner
from audit_book_records import audit

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['future256','standard5000'],default='future256');args=parser.parse_args()
 full=args.protocol=='standard5000';stem='textvqa_standard5000' if full else 'textvqa_future256'
 freeze='audit/a2/textvqa_standard5000_methods_v1.json' if full else 'audit/a2/book_confirmation_v1.json'
 frozen=json.loads((ROOT/freeze).read_text(encoding='utf-8'));hashes=frozen['inputs_sha256'] if full else frozen['input_sha256']
 sp='configs/a2/'+stem+'.json';rp='configs/data/'+stem+'_references.json'
 for path in [sp,rp]:assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==hashes[path]
 samples=json.loads((ROOT/sp).read_text(encoding='utf-8'));refs={r['id']:r for r in json.loads((ROOT/rp).read_text(encoding='utf-8'))};ids={r['id'] for r in samples};assert set(refs)==ids
 runs=[f'textvqa_standard5000_s{s}_v1' if full else f'book_confirm_text_s{s}_v1' for s in [0,1]]
 records,audit_result=audit(frozen,runs,frozen['variants'],hashes[sp]);expected={(sid,v['name']) for sid in ids for v in frozen['variants']}
 assert {(r['session_id'],r['variant']) for r in records}==expected and len(records)==len(expected)
 metric=TextVQAAccuracyEvaluator();rows=[];by=defaultdict(dict);costs=defaultdict(list)
 for record in records:
  sid=record['session_id'];variant=record['variant'];ref=refs[sid];assert len(record['turns'])==1 and len(ref['answers'])==10 and record['image_id']==ref['image_id']
  prediction=record['turns'][0]['answer'];score=metric.eval_pred_list([{'pred_answer':prediction,'gt_answers':ref['answers']}])
  row={'session_id':sid,'image_id':record['image_id'],'variant':variant,'score':float(score),'answer_stratum':ref['answer_stratum'],'prediction':prediction}
  rows.append(row);by[variant][record['image_id'],sid]=float(score);costs[variant].append(record)
 table=[]
 for variant in sorted(by):
  scored=[r for r in rows if r['variant']==variant];boot=cluster_bootstrap([(r['image_id'],r['score']) for r in scored],repeats=8000)
  group=costs[variant];den=sum(r['owned_initial']['visual_fp16_denominator'] for r in group);store=sum(visual_owner(r)[0] for r in group);encoder=sum(r.get('encoder_calibration_parameters_bytes',0) for r in group)
  entry={'variant':variant,'questions':len(scored),'unique_image_contents':len({r['image_id'] for r in scored}),'accuracy':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],
   'effective_visual_bits':16*store/den,'budget_bits_including_encoder_constants':16*(store+encoder)/den,'visual_bytes_total':store}
  for stratum in ['numeric','text','binary']:
   subset=[r['score'] for r in scored if r['answer_stratum']==stratum];entry[stratum+'_questions']=len(subset);entry[stratum+'_accuracy']=float(np.mean(subset)) if subset else None
  table.append(entry)
 pairs=frozen['primary_comparison_family'] if full else frozen['primary_superiority_pairs_holm'];comparisons=[]
 for left,right in pairs:comparisons.append({'left':left,'right':right,**paired(by[left],by[right],repeats=8000)})
 for row,pvalue in zip(comparisons,holm([r['p'] for r in comparisons])):row['holm_family']=pvalue
 candidate=frozen['candidate'];lookup={r['variant']:r for r in table};target={'candidate':candidate,'reference':'native4_8','analysis_role':'secondary transfer benchmark',
  'budget_bits':lookup[candidate]['budget_bits_including_encoder_constants'],**paired(by[candidate],by['native4_8'],repeats=8000)}
 target.update(point_target_met=target['budget_bits']<=1.8 and target['delta']>=-.005,noninferiority_95_ci=target['budget_bits']<=1.8 and target['ci_low']>=-.005)
 out=ROOT/('results/textvqa_standard5000_analysis' if full else 'results/book_confirm_text256_analysis');out.mkdir(exist_ok=True,parents=True)
 for name,data in [('scores.csv',rows),('table.csv',table),('comparisons.csv',comparisons),('quality_target.csv',[target])]:write_csv(out/name,data)
 result={'status':'COMPLETE','records':len(records),'questions_per_variant':len(ids),'unique_image_contents':len({r['image_id'] for r in rows}),
  'table':table,'comparisons':comparisons,'secondary_quality_target':target,'audit':audit_result,'metric':'official MMF10-annotator soft accuracy; not ANLS',
  'scope':'complete5000 official validation QA under common cap8 vision-only prompt, includes previously observed subsets' if full else 'new source-image-disjoint256,128numeric+128text, balanced subset not an official overall-distribution estimate'}
 (out/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'audit/a2'/('TEXTVQA_STANDARD5000_AUDIT.json' if full else 'BOOK_CONFIRM_TEXT256_AUDIT.json')).write_text(json.dumps(audit_result,indent=2))
 print(json.dumps({'records':len(records),'table':table,'target':target,'comparisons':comparisons,'audit':audit_result}))

if __name__=='__main__':main()
