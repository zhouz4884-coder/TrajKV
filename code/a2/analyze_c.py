"""Controlled C outcomes and matched live-owner storage, no benchmark claims."""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze_cycle import cluster_bootstrap
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',default='a2_aux_gpu0_v1');parser.add_argument('--output',type=Path,default=ROOT/'results/a2_analysis_c');a=parser.parse_args()
 p=ROOT/'results'/a.run/'history/records.jsonl'
 rows=[json.loads(s) for s in p.read_text().splitlines()]
 refs={r['id']:r for r in json.loads((ROOT/'configs/a2/c_references.json').read_text())}
 groups=defaultdict(list);scored=[]
 for r in rows:
  metric=score_docvqa(r['prediction'],refs[r['id']][r['target']]);r.update(metric)
  groups[(r['arm'],r['target'])].append(r)
  scored.append({'id':r['id'],'image_id':r['image_id'],'arm':r['arm'],'ordering':r['ordering'],'target':r['target'],
   'prediction':r['prediction'],'reference':refs[r['id']][r['target']],'anls':metric['score'],'exact':metric['exact'],
   'owner':r['owned_before_target']['total_unique'],'text_owner':r['owned_before_target']['text_capacity'],
   'nv':r['visual_tokens'],'context_ms':r['context_ms'],'target_ms':r['target_ms']})
 table=[]
 for (arm,target),rs in sorted(groups.items()):
  table.append({'arm':arm,'target':target,'contexts':len(rs),'unique_images':len({r['image_id'] for r in rs}),
   'anls':float(np.mean([r['score'] for r in rs])),'exact':float(np.mean([r['exact'] for r in rs])),
   'owner_mean':float(np.mean([r['owned_before_target']['total_unique'] for r in rs])),
   'text_owner_mean':float(np.mean([r['owned_before_target']['text_capacity'] for r in rs])),
   'context_ms_mean':float(np.mean([r['context_ms'] for r in rs]))})
 index={(r['id'],r['target'],r['arm']):r for r in rows};savings=[]
 for r in rows:
  if r['arm']=='keep_base2':continue
  base=index.get((r['id'],r['target'],'keep_base2'))
  if base:
   delta=r['owned_before_target']['total_unique']-base['owned_before_target']['total_unique']
   savings.append({'id':r['id'],'image_id':r['image_id'],'target':r['target'],'arm':r['arm'],
    'owner_difference_to_keep_base2':delta,'same_or_lower_stored_bytes':delta<=0,
    'anls_difference':r['score']-base['score'],'exact_difference':r['exact']-base['exact'],
    'logical_positions_equal':r['logical_before_target']==base['logical_before_target']})
 out=a.output;out.mkdir(exist_ok=True)
 for name,data in [('scores',scored),('table',table),('savings',savings)]:
  if data:
   with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (out/'summary.json').write_text(json.dumps({'records':len(rows),'table':table,'scope':'constructed context, paired orderings cluster by original image; raw context cost includes image prefill/replay'},indent=2))
 print(json.dumps({'records':len(rows),'table':table}))
if __name__=='__main__':main()
