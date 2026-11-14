"""Independent image-paired analysis of the prospectively frozen new64 package."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze_cycle import cluster_bootstrap
from analyze_docvqa_confirm64 import holm
from analyze import paired,write_csv
from costs import visual_owner,role
def read_total(r):
 fields=('base_visual_bytes','residual_bytes','text_bytes','reference_bytes','selection_bytes','position_bytes')
 return sum(sum(e.get(k,0) for k in fields) for e in r['read_events'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 frozen=json.loads((ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json').read_text());expected_names={r['name'] for r in frozen['variants']};expected_ids=set(frozen['confirm_ids'])
 refs={r['id']:{z['turn']:z['answer'] for z in r['references']} for r in map(json.loads,(ROOT/'configs/data/docvqa_references.jsonl').read_text(encoding='utf-8').splitlines())}
 samples={r['id']:r for r in json.loads((ROOT/'configs/a2/confirm64_short.json').read_text(encoding='utf-8'))};records=[];seen=set()
 for run in a.runs:
  folder=ROOT/'results'/run
  if json.loads((folder/'complete.json').read_text()).get('status')!='COMPLETE':raise RuntimeError('Partial confirmation')
  for f in folder.rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line);key=(r['session_id'],r['variant'])
    if key in seen:raise RuntimeError('Duplicate confirmation row')
    if r['protocol']!='image_boundary_free_running' or r['session_id'] not in expected_ids or r['variant'] not in expected_names:raise RuntimeError('Unexpected confirmation input')
    if len(r['turns'])!=min(6,len(samples[r['session_id']]['questions'])):raise RuntimeError('Turn coverage differs')
    if not r['read_ledger_exact'] or not r.get('read_budget_ok',True) or not r.get('global_extra_budget_ok',True):raise RuntimeError('Budget or ledger failed')
    seen.add(key);records.append(r)
 if len(records)!=frozen['expected_sessions'] or seen!={(s,v) for s in expected_ids for v in expected_names}:raise RuntimeError('Confirmation incomplete')
 rows=[];by=defaultdict(dict);costs=defaultdict(list)
 for r in records:
  costs[r['variant']].append(r)
  for t in r['turns']:
   score=score_docvqa(t['answer'],refs[r['session_id']][t['turn']]);answers=refs[r['session_id']][t['turn']];answers=answers if isinstance(answers,list) else [answers]
   numeric=any(any(ch.isdigit() for ch in x) for x in answers)
   row={'session_id':r['session_id'],'image_id':r['image_id'],'variant':r['variant'],'turn':t['turn'],'score':score['score'],'exact':score['exact'],
    'numeric':numeric,'prediction':t['answer'],'reference':json.dumps(answers,ensure_ascii=False),'visual_tokens':r['visual_tokens']}
   rows.append(row);by[r['variant']][(r['image_id'],t['turn'])]=score['score']
 table=[]
 for name in sorted(costs):
  rs=costs[name];ss=[r for r in rows if r['variant']==name];boot=cluster_bootstrap([(r['image_id'],r['score']) for r in ss],repeats=8000)
  total=sum(visual_owner(r)[0] for r in rs);den=sum(r['owned_initial']['visual_fp16_denominator'] for r in rs);encoder=sum(r.get('encoder_calibration_parameters_bytes',0) for r in rs)
  def avg(field,default=0):return float(np.mean([r.get(field,default) for r in rs]))
  table.append({'variant':name,'role':role(rs[0]),'images':len(rs),'questions':len(ss),'anls':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],
   'exact':float(np.mean([r['exact'] for r in ss])),'anls_first':float(np.mean([r['score'] for r in ss if r['turn']==1])),
   'anls_later':float(np.mean([r['score'] for r in ss if r['turn']>1])),'anls_numeric':float(np.mean([r['score'] for r in ss if r['numeric']])),
   'effective_visual_bits':16*total/den,'conservative_bits_with_encoder_constants':16*(total+encoder)/den,'visual_store_mean':total/len(rs),
   'encoder_constants_mean':encoder/len(rs),'total_bank_owner_mean':float(np.mean([r['owned_initial']['total_unique'] for r in rs])),
   'extra_reads_total':sum(r['extra_logical_read_bytes'] for r in rs),'selection_reads_total':sum(r.get('selection_logical_read_bytes',0) for r in rs),
   'total_cache_logical_reads':sum(read_total(r) for r in rs),'free_output_tokens':sum(len(t['token_ids']) for r in rs for t in r['turns']),
   'session_ms_median':float(np.median([r['session_ms'] for r in rs])),'encoding_ms_median':float(np.median([r['encoding_wall_ms'] for r in rs])),
   'peak_allocated_max':max(r['peak_allocated'] for r in rs)})
 index={r['variant']:r for r in table};primary=[]
 for left,right in frozen['primary_tests']:
  left=frozen['aliases'].get(left,left);right=frozen['aliases'].get(right,right)
  primary.append({'left':left,'right':right,**paired(by[left],by[right],repeats=8000)})
 for row,pv in zip(primary,holm([r['p'] for r in primary])):row['holm_primary_family']=pv
 secondary=[]
 for v in sorted(by):
  if v=='storage_primary':continue
  secondary.append({'left':'storage_primary','right':v,**paired(by['storage_primary'],by[v],repeats=8000)})
 reads=[]
 full=index['read_all'];full_store={r['session_id']:visual_owner(r)[0] for r in costs['read_all']}
 for v in sorted(x for x in by if x.startswith('read_') and x!='read_all'):
  r=index[v];delta=paired(by[v],by['read_all'],repeats=8000);same=all(visual_owner(x)[0]==full_store[x['session_id']] for x in costs[v])
  saving=1-r['extra_reads_total']/full['extra_reads_total'];reads.append({'variant':v,'same_visual_store_each_image':same,'extra_read_saving':saving,
   'total_logical_cache_read_saving':1-r['total_cache_logical_reads']/full['total_cache_logical_reads'],**delta,
   'point_target_met':same and saving>=.5 and delta['delta']>=-.005,'noninferiority_95_ci_unadjusted':delta['ci_low']>=-.005})
 quality_ref=frozen['quality_reference_confirmation_variant'];quality=paired(by['storage_primary'],by[quality_ref],repeats=8000)
 turns=[]
 for v in sorted(by):
  for t in range(1,7):
   ss=[r for r in rows if r['variant']==v and r['turn']==t]
   if ss:turns.append({'variant':v,'turn':t,'questions':len(ss),'anls':float(np.mean([r['score'] for r in ss])),'exact':float(np.mean([r['exact'] for r in ss]))})
 write_csv(a.output/'scores.csv',rows);write_csv(a.output/'table.csv',table);write_csv(a.output/'primary_comparisons.csv',primary);write_csv(a.output/'secondary_comparisons.csv',secondary);write_csv(a.output/'read_targets.csv',reads);write_csv(a.output/'by_turn.csv',turns)
 out={'sessions':len(records),'answers':len(rows),'unique_images':64,'questions_per_variant':frozen['confirm_questions'],'table':table,'primary_comparisons':primary,
  'quality_target':{'reference':quality_ref,**quality,'point_met':quality['delta']>=-.005,'noninferiority_95_ci_unadjusted':quality['ci_low']>=-.005},
  'read_targets':reads,'scope':'prospectively frozen, current-cycle image-disjoint64; question-weighted scores, whole-image paired uncertainty; no physical DRAM or timing-speed claim from payload counters'}
 (a.output/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps({'sessions':len(records),'answers':len(rows),'quality_target':out['quality_target'],'read_targets':reads}))
if __name__=='__main__':main()
