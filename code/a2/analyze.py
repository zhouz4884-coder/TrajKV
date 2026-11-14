"""Frozen DocVQA scorer, image-cluster paired analysis and full cost tables."""
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze_docvqa_confirm64 import signflip_pvalues,holm
from costs import visual_owner,role
def write_csv(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paired(a,b,repeats=4000):
 keys=sorted(a.keys()&b.keys());groups=defaultdict(list)
 for k in keys:groups[k[0]].append(a[k]-b[k])
 sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(20261113)
 if len(groups)==0:raise ValueError('No paired documents')
 draws=rng.integers(0,len(sums),(repeats,len(sums)));boot=sums[draws].sum(1)/counts[draws].sum(1)
 test=signflip_pvalues(sums,len(keys),repeats=100000)
 return {'delta':float(sums.sum()/len(keys)),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
  'p':test['p_two_sided'][0],'images':len(groups),'questions':len(keys)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate');a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 refs={r['id']:{x['turn']:x['answer'] for x in r['references']} for r in map(json.loads,(ROOT/'configs/data/docvqa_references.jsonl').read_text(encoding='utf-8').splitlines())}
 records=[]
 for name in a.runs:
  folder=ROOT/'results'/name
  for f in folder.rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    if line.strip():
     r=json.loads(line)
     if 'session_id' in r and 'variant' in r and 'turns' in r and 'method' in r and r.get('protocol') in ('free_running','image_boundary_free_running') and r.get('task_accuracy_eligible') is not False and r.get('primary_quality_eligible') is not False:records.append(r)
 scored=[];seen=set();costs=defaultdict(list);by=defaultdict(dict)
 for r in records:
  v=r.get('variant',r['method']);key=(r['session_id'],v)
  if key in seen:raise RuntimeError('Duplicate variant/session '+str(key))
  seen.add(key);costs[v].append(r)
  for t in r['turns']:
   s=score_docvqa(t['answer'],refs[r['session_id']][t['turn']])
   row={'session_id':r['session_id'],'image_id':r['image_id'],'turn':t['turn'],'variant':v,'score':s['score'],'exact':s['exact'],'prediction':t['answer'],'reference':json.dumps(refs[r['session_id']][t['turn']],ensure_ascii=False),'nv':r['visual_tokens']}
   scored.append(row);by[v][(r['image_id'],t['turn'])]=s['score']
 table=[]
 for v in sorted(costs):
  rs=costs[v];ss=[x for x in scored if x['variant']==v]
  def mean(f):return float(np.mean([f(r) for r in rs]))
  table.append({'variant':v,'images':len(rs),'questions':len(ss),'diagnostic':any(r.get('variant_config',{}).get('diagnostic') or r.get('variant_config',{}).get('local_calibration') for r in rs),
   'materialized_quality_port':any(role(r)=='materialized quality prototype' for r in rs),'anls':float(np.mean([x['score'] for x in ss])),
   'exact':float(np.mean([x['exact'] for x in ss])),'anls_t2plus':float(np.mean([x['score'] for x in ss if x['turn']>1])),
   'nv_mean':mean(lambda r:r['visual_tokens']),'owner_initial_mean':mean(lambda r:r['owned_initial']['total_unique']),
   'visual_store_mean':mean(lambda r:visual_owner(r)[0]),
   'session_seconds_median':float(np.median([r['session_ms']/1000 for r in rs])),
   'extra_reads_mean':mean(lambda r:r['extra_logical_read_bytes']),
   'selector_reads_mean':mean(lambda r:r.get('selection_logical_read_bytes',0)),
   'visual_reads_mean':mean(lambda r:sum(e['base_visual_bytes'] for e in r['read_events'])),
   'text_reads_mean':mean(lambda r:sum(e['text_bytes'] for e in r['read_events'])),
   'peak_allocated_max':max(r['peak_allocated'] for r in rs)})
 comparisons=[]
 if a.candidate:
  candidate=a.candidate
  for v in sorted(by):
   if v==candidate:continue
   comparisons.append({'candidate':candidate,'comparator':v,**paired(by[candidate],by[v])})
  adjusted=holm([r['p'] for r in comparisons])
  for row,pv in zip(comparisons,adjusted):row['holm_all_comparators']=pv
 write_csv(a.output/'scores.csv',scored);write_csv(a.output/'table.csv',table);write_csv(a.output/'comparisons.csv',comparisons)
 result={'records':len(records),'scored_questions':len(scored),'table':table,'comparisons':comparisons}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
