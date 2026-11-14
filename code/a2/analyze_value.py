"""Score the prospectively selected 20 digit-bearing questions; image-cluster CIs."""
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze_cycle import cluster_bootstrap
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',default=['a2_value_probe20_gpu1_v1']);a=parser.parse_args()
 refs={r['id']:r for r in json.loads((ROOT/'configs/a2/value_probe20_references.json').read_text())}
 records=[]
 for run in a.runs:
  for p in (ROOT/'results'/run).rglob('records.jsonl'):
   for line in p.read_text().splitlines():
    r=json.loads(line)
    if r.get('session_id') in refs:records.append(r)
 scores=[];groups=defaultdict(list)
 for r in records:
  assert r['protocol']=='image_boundary_free_running' and not r['initial_logits_shared_fp16']
  s=score_docvqa(r['turns'][0]['answer'],refs[r['session_id']]['answer'])
  row={'id':r['session_id'],'image_id':r['image_id'],'variant':r['variant'],'prediction':r['turns'][0]['answer'],'reference':str(refs[r['session_id']]['answer']),'anls':s['score'],'exact':s['exact']}
  scores.append(row);groups[r['variant']].append(row)
 table=[]
 for v,rs in sorted(groups.items()):
  boot=cluster_bootstrap([(r['image_id'],r['anls']) for r in rs],repeats=4000,seed=1212)
  table.append({'variant':v,'questions':len(rs),'images':len({r['image_id'] for r in rs}),'anls':float(np.mean([r['anls'] for r in rs])),'exact':float(np.mean([r['exact'] for r in rs])),'cluster_bootstrap':boot})
 out=ROOT/'results/a2_analysis_value';out.mkdir(exist_ok=True)
 with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
 report={'records':len(records),'questions_selected':20,'images_selected':9,'selection':'first 20 reference answers with ASCII digits in old dev32 order, independent of new compression outcomes','protocol':'image boundary; exactly FP16 opposite axis; materialized diagnostic','table':table}
 (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
