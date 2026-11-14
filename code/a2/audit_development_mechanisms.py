"""Independent endpoint, charged-prefix and within-source paired checks."""
import csv,json
from pathlib import Path
from collections import defaultdict
from analyze import paired,write_csv
from costs import visual_owner
ROOT=Path(__file__).resolve().parents[2]

def load(names,child=None):
 result={}
 for name in names:
  folder=ROOT/'results'/name
  if child:folder=folder/child
  for line in (folder/'records.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant']);assert key not in result;result[key]=r
 return result

def main():
 reads=load(['cached_reads_dev64_s0_v1','cached_reads_dev64_s1_v1'],'quality')
 reads.update(load(['read_factorial_dev64_s0_v1','read_factorial_dev64_s1_v1']))
 density=load(['density_development64_s0_v1','density_development64_s1_v1'])
 anchors=load(['anchors_development64_s0_v3','anchors_development64_s1_v3'])
 ids={k[0] for k in reads};assert len(ids)==64;endpoints=first_logits=anchor_formulas=0
 coverage=defaultdict(lambda:[0,0])
 for sid in ids:
  a=reads[sid,'cachedread_none'];b=density[sid,'base2_cap16']
  assert a['initial_logits_sha256']==b['initial_logits_sha256'] and [t['token_ids'] for t in a['turns']]==[t['token_ids'] for t in b['turns']];endpoints+=1
  a=reads[sid,'cachedread_prefill_cached50'];b=reads[sid,'cachedread_prefill_freshrisk50']
  assert a['initial_logits_sha256']==b['initial_logits_sha256'];first_logits+=1
  for family in ('affine','direction'):
   a=anchors[sid,'anchor_'+family+'_native'];b=anchors[sid,'anchor_'+family+'_compressed']
   assert a['anchor_tokens']==b['anchor_tokens'] and visual_owner(a)[0]==visual_owner(b)[0]
   assert a.get('actual_fixed_prefix_token_ids',[])==b.get('actual_fixed_prefix_token_ids',[])
  for variant in [v for s,v in anchors if s==sid]:
   r=anchors[sid,variant];m=r['anchor_tokens'];expected=m*(4*28*4*128+8)
   assert r['owned_initial']['anchor_visual_auxiliary']==expected and r['introduced_nonquestion_tokens']<=m and r.get('uncharged_fixed_prompt_tail_tokens',0)==0
   if r['method']!='anchor_full':assert r['owned_initial']['anchor_conservative_visual_bits']<=1.8+1e-9
   anchor_formulas+=1
 for r in reads.values():
  for event in r['read_events']:
   phase='prefill' if event['phase'].endswith('prefill') else 'decode'
   key=(r['variant'],phase);coverage[key][0]+=event.get('selected_head_pages',0);coverage[key][1]+=event.get('available_head_pages',0)
 rows=[{'variant':k[0],'phase':k[1],'selected_head_pages':v[0],'available_head_pages':v[1],'actual_selected_page_fraction':v[0]/v[1]} for k,v in sorted(coverage.items()) if v[1]]
 by=defaultdict(dict)
 for r in csv.DictReader((ROOT/'results/anchors_development64_analysis/scores.csv').open(encoding='utf-8-sig')):by[r['variant']][r['image_id'],int(r['turn'])]=float(r['score'])
 comparisons=[]
 for family in ('affine','direction'):
  for left,right in [('native','compressed'),('native','original'),('compressed','original')]:
   l='anchor_'+family+'_'+left;r='anchor_'+family+'_'+right;comparisons.append({'left':l,'right':r,**paired(by[l],by[r],repeats=8000)})
 out=ROOT/'results/development_mechanism_audit';out.mkdir(exist_ok=True,parents=True);write_csv(out/'coverage.csv',rows);write_csv(out/'anchor_comparisons.csv',comparisons)
 result={'status':'PASS','base2_equals_no_refinement_whole_trajectory':endpoints,'prefill_fresh_equals_cached_first_logits':first_logits,
  'anchor_independent_owner_formula_cases':anchor_formulas,'same_anchor_input_and_store_pairs':128,'coverage':rows,'anchor_comparisons':comparisons,
  'wording':'0.5 is a per-read byte-budget fraction, not necessarily50% selected pages; selection overhead is deducted; use measured coverage',
  'scope':'completed open development only; no confirm256 or balanced128 scores read'}
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
