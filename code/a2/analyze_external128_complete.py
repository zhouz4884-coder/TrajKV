"""Strict complete external replication; no score from partially finished jobs."""
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap
from analyze import paired,write_csv
from costs import visual_owner,role

def main():
 main_contract=json.loads((ROOT/'audit/a2/frozen_external128_and_proxies.json').read_text(encoding='utf-8'))
 additional=json.loads((ROOT/'audit/a2/frozen_external_additional_controls.json').read_text(encoding='utf-8'))
 primary=json.loads((ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json').read_text(encoding='utf-8'))
 refs_path=ROOT/'configs/a2/textvqa_eval128_refs.json'
 assert hashlib.sha256(refs_path.read_bytes()).hexdigest()==main_contract['external_reference_sha256']
 refs={r['id']:r for r in json.loads(refs_path.read_text(encoding='utf-8'))}
 samples=ROOT/'configs/a2/textvqa_eval128.json'
 assert hashlib.sha256(samples.read_bytes()).hexdigest()==main_contract['external_membership_sha256']
 expected_variants=main_contract['external_variants']+additional['variants'];assert len(expected_variants)==13
 expected={(sid,v) for sid in refs for v in expected_variants};assert len(refs)==128
 frozen_cfg={v['name']:v for v in primary['variants']};metric=TextVQAAccuracyEvaluator()
 records=[];seen=set();rows=[];by=defaultdict(dict);config_checks=0;codec_checks=0;source_checks=0
 core=['run_variants.py','session_boundary.py','masked_bank.py','rate_bank.py','factory.py','fused_rates.py',
  'direction_bank.py','fused_direction.py','calib_packed_bank.py','hybrid_packed_bank.py','fused_hybrid.py','quantizers.py']
 for run in ['textvqa_eval128_s0_v1','textvqa_eval128_s1_v1',*additional['jobs']]:
  folder=ROOT/'results'/run
  assert json.loads((folder/'complete.json').read_text())['status']=='COMPLETE'
  for path in folder.rglob('manifest.json'):
   manifest=json.loads(path.read_text());sources=manifest.get('sources')
   if not sources:continue
   assert manifest['samples_sha256']==main_contract['external_membership_sha256']
   assert manifest['answers_visible_to_runner'] is False
   for name in core:
    assert sources['code/a2/'+name]==primary['source_sha256'][name];source_checks+=1
   variants=manifest['config']['variants']
   required={v[k] for v in variants for k in ('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path') if v.get(k)}
   assert set(manifest['external_codec_inputs'])==required
   for name,h in manifest['external_codec_inputs'].items():
    assert h==primary['external_codec_inputs_sha256'][name];codec_checks+=1
  for path in folder.rglob('records.jsonl'):
   for line in path.read_text().splitlines():
    record=json.loads(line);sid=record['session_id'];variant=record['variant'];key=(sid,variant)
    assert key in expected and key not in seen;seen.add(key)
    assert record['protocol']=='image_boundary_free_running' and record.get('task_accuracy_eligible') is not False
    assert record['variant_config']==frozen_cfg[variant];config_checks+=1
    assert record.get('read_ledger_exact',True) and len(record['turns'])==1
    pred=record['turns'][0]['answer'];score=metric.eval_pred_list([{'pred_answer':pred,'gt_answers':refs[sid]['answers']}])
    stored,aux=visual_owner(record);records.append(record)
    rows.append({'session_id':sid,'image_id':record['image_id'],'variant':variant,'score':score,'prediction':pred,
     'actual_visual_bytes':stored,'fp16_visual_bytes':record['owned_initial']['visual_fp16_denominator'],'role':role(record),'aux_adjustment':aux})
    by[variant][(record['image_id'],1)]=score
 assert seen==expected and all(len(d)==128 for d in by.values())
 table=[]
 for variant in expected_variants:
  subset=[r for r in rows if r['variant']==variant]
  boot=cluster_bootstrap([(r['image_id'],r['score']) for r in subset],repeats=8000)
  table.append({'variant':variant,'images':len(subset),'vqa_accuracy':boot['mean'],'ci_low':boot['ci_lower'],'ci_high':boot['ci_upper'],
   'effective_visual_bits':16*sum(r['actual_visual_bytes'] for r in subset)/sum(r['fp16_visual_bytes'] for r in subset),'role':subset[0]['role']})
 # Replication comparisons are reported as secondary, without relabelling them
 # as the six primary DocVQA tests registered in the main frozen contract.
 comparisons=[]
 for left,right in [('storage_primary','full8'),('storage_primary','base2_8'),('storage_primary','k2_lloyd_v1'),
  ('storage_primary','calibration_warm'),('storage_primary','kvarn2_8_quality_port'),('direction_vq2_v1','full8'),
  ('direction_vq2_v1','k2_lloyd_v1'),('direction_vq2_v1','storage_primary'),('kvarn2_k_clipped_v1','storage_primary')]:
  comparisons.append({'candidate':left,'comparator':right,'analysis_role':'secondary external replication',**paired(by[left],by[right],repeats=8000)})
 out=ROOT/'results/frozen_external128_analysis';out.mkdir(exist_ok=True,parents=True)
 write_csv(out/'scores.csv',rows);write_csv(out/'table.csv',table);write_csv(out/'comparisons.csv',comparisons)
 result={'status':'PASS','records':len(rows),'expected_records':len(expected),'images':128,'config_checks':config_checks,
  'codec_input_checks':codec_checks,'frozen_inference_source_checks':source_checks,'metric':'official MMF TextVQA ten-annotator soft VQA accuracy; not ANLS',
  'table':table,'comparisons':comparisons,'scope':'current-cycle prospective external128; historical whole-project exposure not exhaustively audited',
  'model':'InternVL3-8B-hf FP16, actual CUDA full-model generation; same cap8 per variant',
  'source_inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [refs_path,samples]}}
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'PASS','records':len(rows),'table':table,'comparisons':comparisons}))
if __name__=='__main__':main()
