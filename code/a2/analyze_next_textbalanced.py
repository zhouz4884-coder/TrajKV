"""Complete frozen external128 replication, with prespecified answer strata."""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap
from analyze import write_csv,paired
from costs import visual_owner,role

def main():
 frozen=json.loads((ROOT/'audit/a2/next_confirmation_v1.json').read_text());variants={v['name']:v for v in frozen['text_variants']};aliases=frozen['aliases']
 rp=ROOT/'configs/data/textvqa_balanced128_references.json';sp=ROOT/'configs/a2/textvqa_balanced128.json'
 assert hashlib.sha256(rp.read_bytes()).hexdigest()==frozen['input_sha256']['configs/data/textvqa_balanced128_references.json']
 assert hashlib.sha256(sp.read_bytes()).hexdigest()==frozen['input_sha256']['configs/a2/textvqa_balanced128.json']
 refs={r['id']:r for r in json.loads(rp.read_text(encoding='utf-8'))};expected={(sid,v) for sid in refs for v in variants};seen=set();rows=[];by=defaultdict(dict);source_checks=0
 metric=TextVQAAccuracyEvaluator()
 for name in ('next_textbalanced128_s0_v1','next_textbalanced128_s1_v1'):
  folder=ROOT/'results'/name;assert json.loads((folder/'complete.json').read_text())['status']=='COMPLETE';m=json.loads((folder/'manifest.json').read_text())
  assert m['samples_sha256']==frozen['input_sha256']['configs/a2/textvqa_balanced128.json'] and not m['answers_visible_to_runner']
  for fn,digest in frozen['inference_source_sha256'].items():assert m['sources']['code/a2/'+fn]==digest;source_checks+=1
  required={v[k] for v in variants.values() for k in ('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path','affine_factor_path') if v.get(k)}
  assert set(m['external_codec_inputs'])==required
  for path,digest in m['external_codec_inputs'].items():assert digest==frozen['external_codec_inputs_sha256'][path]
  for line in (folder/'records.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant']);assert key in expected and key not in seen;seen.add(key)
   assert r['variant_config']==variants[r['variant']] and r['protocol']=='image_boundary_free_running' and not r['initial_logits_shared_fp16'] and len(r['turns'])==1
   ref=refs[r['session_id']];score=metric.eval_pred_list([{'pred_answer':r['turns'][0]['answer'],'gt_answers':ref['answers']}]);store,aux=visual_owner(r)
   rows.append({'session_id':r['session_id'],'image_id':r['image_id'],'variant':r['variant'],'stratum':ref['answer_stratum'],'score':score,
    'prediction':r['turns'][0]['answer'],'visual_bytes':store,'fp16_visual_bytes':r['owned_initial']['visual_fp16_denominator'],'role':role(r)})
   by[r['variant']][r['image_id'],1]=score
 assert seen==expected and len(seen)==frozen['expected_text_sessions'];table=[]
 for variant in variants:
  subset=[r for r in rows if r['variant']==variant];boot=cluster_bootstrap([(r['image_id'],r['score']) for r in subset],repeats=8000)
  table.append({'variant':variant,'images':128,'vqa_accuracy':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],
   'numeric_accuracy':sum(r['score'] for r in subset if r['stratum']=='numeric')/64,'text_accuracy':sum(r['score'] for r in subset if r['stratum']=='text')/64,
   'effective_visual_bits':16*sum(r['visual_bytes'] for r in subset)/sum(r['fp16_visual_bytes'] for r in subset),'role':subset[0]['role']})
 pairs=[('new_storage_primary','full8'),('new_storage_primary','base2_8'),('new_storage_primary','k2_lloyd_v1'),
  ('cold_multi_gain','warm_multi_gain'),('cold_multi_gain','cold_first_gain'),('cold_multi_affine','cold_multi_gain'),('cold_multi_affine','warm_multi_affine')]
 comparisons=[]
 for left,right in pairs:
  l=aliases[left];r=aliases[right];comparisons.append({'left_label':left,'right_label':right,'left_variant':l,'right_variant':r,'role':'secondary external replication',**paired(by[l],by[r],repeats=8000)})
 out=ROOT/'results/next_textbalanced128_analysis';out.mkdir(exist_ok=True,parents=True);write_csv(out/'scores.csv',rows);write_csv(out/'table.csv',table);write_csv(out/'comparisons.csv',comparisons)
 result={'status':'PASS','records':len(rows),'images':128,'strata':{'numeric':64,'text':64},'source_checks':source_checks,'table':table,'comparisons':comparisons,
  'metric':'official MMF ten-annotator TextVQA soft accuracy','scope':'prospective content-balanced subset, not overall official-distribution estimate; no tuning on these outcomes'}
 (out/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'audit/a2/NEXT_TEXTBALANCED128_AUDIT.json').write_text(json.dumps({k:result[k] for k in ('status','records','images','strata','source_checks')},indent=2));print(json.dumps(result))
if __name__=='__main__':main()
