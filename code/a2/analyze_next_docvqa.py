"""Complete matched new-source DocVQA analysis, storage and read axes separate."""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze_cycle import cluster_bootstrap
from analyze import paired,write_csv
from costs import visual_owner,role

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--read-all');p.add_argument('--expected-variants',type=int,required=True);p.add_argument('--role',choices=['development','confirmation'],default='development')
 p.add_argument('--references',type=Path,default=ROOT/'configs/data/docvqa_next_references.jsonl');a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 samples=json.loads(a.samples.read_text(encoding='utf-8'));ids={r['id'] for r in samples};assert len(ids)==len(samples)
 refs={r['id']:{int(t['turn']):t['answer'] for t in r['references']} for r in map(json.loads,a.references.read_text(encoding='utf-8').splitlines()) if r['id'] in ids}
 assert set(refs)==ids
 records=[];seen=set();by=defaultdict(dict);costs=defaultdict(list);scored=[]
 for name in a.runs:
  folder=ROOT/'results'/name;assert json.loads((folder/'complete.json').read_text())['status']=='COMPLETE'
  quality=folder/'quality'
  if not quality.is_dir():quality=folder
  manifest=json.loads((quality/'manifest.json').read_text());assert manifest['samples_sha256']==hashlib.sha256(a.samples.read_bytes()).hexdigest()
  variants={v['name']:v for v in manifest['config']['variants']};assert 0<len(variants)<=a.expected_variants
  for path in [quality/'records.jsonl']:
   for line in path.read_text().splitlines():
    r=json.loads(line);sid=r['session_id'];variant=r['variant'];key=(sid,variant)
    assert sid in ids and variant in variants and key not in seen;seen.add(key)
    assert r['protocol']=='image_boundary_free_running' and r.get('task_accuracy_eligible') is not False
    assert r['variant_config']==variants[variant] and r['read_ledger_exact'] and r.get('global_extra_budget_ok',True)
    assert len(r['turns'])==len(refs[sid]);records.append(r);costs[variant].append(r)
    for turn in r['turns']:
     result=score_docvqa(turn['answer'],refs[sid][turn['turn']]);answers=refs[sid][turn['turn']]
     row={'session_id':sid,'image_id':r['image_id'],'source_document_id':next(x['source_document_id'] for x in samples if x['id']==sid),
      'variant':variant,'turn':turn['turn'],'score':result['score'],'exact':result['exact'],'prediction':turn['answer'],'reference':json.dumps(answers),
      'numeric':any(any(ch.isdigit() for ch in x) for x in answers),'visual_tokens':r['visual_tokens']}
     scored.append(row);by[variant][(r['image_id'],turn['turn'])]=result['score']
 assert len(costs)==a.expected_variants
 assert seen=={(sid,v) for sid in ids for v in costs}
 table=[]
 for variant,rows in sorted(costs.items()):
  answers=[r for r in scored if r['variant']==variant];boot=cluster_bootstrap([(r['image_id'],r['score']) for r in answers],repeats=8000)
  denominator=sum(r['owned_initial']['visual_fp16_denominator'] for r in rows);store=sum(visual_owner(r)[0] for r in rows);encoder=sum(r.get('encoder_calibration_parameters_bytes',0) for r in rows)
  table.append({'variant':variant,'images':len(rows),'questions':len(answers),'anls':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],
   'exact':float(np.mean([r['exact'] for r in answers])),'effective_visual_bits':16*store/denominator,'budget_bits_including_encoder_constants':16*(store+encoder)/denominator,
   'encoder_constants_bytes_per_session':encoder/len(rows),'role':role(rows[0]),'visual_bytes_total':store,'extra_reads_total':sum(r['extra_logical_read_bytes'] for r in rows),
   'logical_cache_reads_total':sum(sum(sum(e.get(k,0) for k in ('base_visual_bytes','residual_bytes','text_bytes','reference_bytes','selection_bytes','position_bytes')) for e in r['read_events']) for r in rows),
   'free_session_seconds_median':float(np.median([r['session_ms']/1000 for r in rows])),
   'cache_reused_calls':sum(r.get('selector_cache_reused_calls',0) for r in rows),'cache_fresh_calls':sum(r.get('selector_cache_fresh_calls',0) for r in rows)})
 comparisons=[]
 pairs=[('cold_first_gain','warm_first_gain'),('cold_multi_gain','warm_multi_gain'),('cold_multi_affine','warm_multi_affine'),
  ('cold_multi_gain','cold_first_gain'),('warm_multi_gain','warm_first_gain'),('cold_multi_affine','cold_multi_gain'),('warm_multi_affine','warm_multi_gain')]
 for variant in by:
  if variant.startswith(('cold_','warm_')):
   pairs.extend((variant,ref) for ref in ['full8','k2_lloyd_v1','previous_frozen_primary'])
 for left,right in sorted(set(pairs)):
  if left in by and right in by:comparisons.append({'left':left,'right':right,'analysis_role':a.role,**paired(by[left],by[right],repeats=8000)})
 read_targets=[]
 if a.read_all:
  ref=a.read_all;lookup={r['variant']:r for r in table};owned={(r['session_id'],r['variant']):visual_owner(r)[0] for r in records}
  for variant in sorted(by):
   if variant==ref:continue
   if costs[variant][0]['method']!=costs[ref][0]['method']:continue
   same=all(owned[sid,variant]==owned[sid,ref] for sid in ids);assert same
   saving=1-lookup[variant]['extra_reads_total']/lookup[ref]['extra_reads_total'];comparison=paired(by[variant],by[ref],repeats=8000)
   read_targets.append({'variant':variant,'same_store_each_image':same,'extra_read_saving':saving,
    'total_logical_cache_read_saving':1-lookup[variant]['logical_cache_reads_total']/lookup[ref]['logical_cache_reads_total'],
    'point_target_met':saving>=.5 and comparison['delta']>=-.005,'noninferiority_95_ci':comparison['ci_low']>=-.005,**comparison})
 for filename,rows in [('scores.csv',scored),('table.csv',table),('comparisons.csv',comparisons),('read_targets.csv',read_targets)]:write_csv(a.output/filename,rows)
 result={'status':'COMPLETE','images':len(ids),'source_documents':len({r['source_document_id'] for r in samples}),'sessions':len(records),
  'questions_per_variant':sum(len(refs[sid]) for sid in refs),'scored_answers':len(scored),'variants':len(costs),
  'table':table,'comparisons':comparisons,'read_targets':read_targets,'samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),
  'analysis_role':a.role,'scope':'new source-disjoint '+a.role+', complete coverage; free-time medians descriptive because output lengths differ; no HBM or fixed-output speed claim'}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'COMPLETE','images':len(ids),'sessions':len(records),'table':table,'comparisons':comparisons,'read_targets':read_targets}))
if __name__=='__main__':main()
