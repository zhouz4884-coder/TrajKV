"""Image-level summaries of GPU teacher proxies; no accuracy from forced answers."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 groups=defaultdict(list);seen=set()
 for run in a.runs:
  for f in (ROOT/'results'/run).rglob('proxies.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line);key=(r['session_id'],r['variant'])
    if key in seen:raise RuntimeError('Duplicate teacher proxy')
    assert r['teacher_forced'] and not r['task_accuracy_eligible'];seen.add(key);groups[r['variant']].append(r)
 table=[]
 for v,rows in sorted(groups.items()):
  table.append({'variant':v,'images':len(rows),'mean_reference_token_logprob':float(np.mean([r['mean_reference_token_logprob'] for r in rows])),
   'token_weighted_logprob':sum(r['content_tokens']*r['mean_reference_token_logprob'] for r in rows)/sum(r['content_tokens'] for r in rows),
   'mean_kl_fp16_to_method':float(np.mean([r['mean_kl_fp16_to_method'] for r in rows])),
   'argmax_changes_at_gold_context':sum(r['argmax_changes'] for r in rows),
   'positions_including_eos':sum(r['content_tokens']+1 for r in rows),
   'argmax_change_rate_including_eos':sum(r['argmax_changes'] for r in rows)/sum(r['content_tokens']+1 for r in rows),
   'scope':'canonical reference teacher forcing; KL/log-prob content only; argmax changes include EOS; not free accuracy'})
 with (a.output/'table.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 (a.output/'summary.json').write_text(json.dumps({'proxy_records':len(seen),'table':table},indent=2));print(json.dumps({'proxy_records':len(seen),'table':table}))
if __name__=='__main__':main()
