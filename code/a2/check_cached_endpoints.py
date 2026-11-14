"""Free whole-model endpoint identity and retained-mask ownership checks."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 assert json.loads((a.input/'complete.json').read_text())['status']=='COMPLETE'
 rows=[json.loads(x) for x in (a.input/'records.jsonl').read_text().splitlines()];by={(r['session_id'],r['variant']):r for r in rows};checks=[]
 for sid in sorted({r['session_id'] for r in rows}):
  for endpoint in ('all','none'):
   new=by[sid,'cachedread_'+endpoint];old=by[sid,'original_'+endpoint]
   assert new['initial_logits_sha256']==old['initial_logits_sha256']
   assert [r['token_ids'] for r in new['turns']]==[r['token_ids'] for r in old['turns']]
   owned=new['owned_initial'];delta=owned['total_unique']-old['owned_initial']['total_unique']
   expected=28*4*(new['visual_tokens']//128)+28*2*8
   assert owned['selector_cache_buffers']+owned['selector_cache_host_control']==expected==delta
   checks.append({'session_id':sid,'endpoint':endpoint,'tokens_and_initial_logits_identical':True,'extra_owned_bytes':delta})
 for row in rows:
  assert row['read_ledger_exact'] and row.get('global_extra_budget_ok',True)
  if row['variant'].startswith('cachedread_'):
   assert row['owned_initial']['visual_base']==by[row['session_id'],'cachedread_all']['owned_initial']['visual_base']
   assert row['owned_initial']['selector_metadata']==by[row['session_id'],'cachedread_all']['owned_initial']['selector_metadata']
 report={'status':'PASS','endpoint_checks':checks,'record_checks':len(rows),'scope':'software endpoint/owner/budget checks, no new task-effect claim'}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':'PASS'}));print(json.dumps({'status':'PASS','checks':len(checks),'records':len(rows)}))
if __name__=='__main__':main()
