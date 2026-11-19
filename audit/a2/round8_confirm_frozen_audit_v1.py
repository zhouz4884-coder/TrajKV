"""Post-completion frozen confirmation identity/price audit; no rescoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/a2'))
from analyze_round8_postcal_quality import completion_gate


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise RuntimeError(message)


def main():
 p=argparse.ArgumentParser();p.add_argument('--pool',choices=['doc_train64','infovqa_validation64'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 freeze_path=ROOT/'configs/a2/round8_v1/method_freeze_v1.json';freeze=read(freeze_path);freeze_sha=sha(freeze_path)
 require(freeze_sha=='af1c0c06090522ca92efa57b0d0d85b1ae4445606510cc0bc3e1f1a11f1ff4be','Method freeze differs')
 require(all(sha(ROOT/item['path'])==item['sha256'] for item in freeze['frozen_files']),'One of40 frozen files changed')
 prepared=read(ROOT/'audit/a2/round8_confirmation_firstq4_prepared_v1.json');spec=next(row for row in prepared['pools'] if row['pool']==a.pool)
 folder=ROOT/f'results/round8_v1/confirm_firstq4_v1_{a.pool}';gate=completion_gate(folder);require(gate and gate['status']=='PASS','Strict final postcal gate failed')
 manifest=read(folder/'manifest.json');cfg=read(ROOT/spec['config']);samples=read(ROOT/spec['samples']);by_sample={r['id']:r for r in samples}
 require(sha(ROOT/spec['config'])==spec['config_sha256']==manifest['config_sha256'] and sha(ROOT/spec['samples'])==spec['samples_sha256']==manifest['samples_sha256'],'Prepared/config/sample SHA graph differs')
 frozen_by={v['name']:v for v in freeze['variants']};variants=[frozen_by[name] for name in freeze['confirmation_variant_names']]
 require(cfg==manifest['config'] and cfg['variants']==variants and cfg['method_freeze_sha256']==freeze_sha
  and cfg['arguments']['max_patches']==8 and manifest['selected_ids']==[r['id'] for r in samples],'Frozen variants/caps/source order differ')
 rows=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines() if line.strip()];by={}
 for r in rows:by.setdefault(r['session_id'],{})[r['variant']]=r
 require(len(rows)==256 and set(by)==set(by_sample) and all(set(v)==set(freeze['confirmation_variant_names']) for v in by.values()),'Exact64x4 coverage differs')
 deltas=[];nvs=[];k2stages=[];learnedbytes=[];r7bytes=[]
 for sid,arms in by.items():
  require(all(r['image_id']==by_sample[sid]['image_id'] and r['max_patches']==8 and [t['turn'] for t in r['turns']]==[1] for r in arms.values()),'Same-source firstQA/cap identity differs')
  require(len({r['visual_tokens'] for r in arms.values()})==1,'Methods use different realNv')
  baseline=arms['r7_frozen_joint43_control'];candidate=arms['postcal_learned_R42'];anchor=arms['r7_budget_k2_inc'];nv=candidate['visual_tokens']
  require(baseline['depth_residual_stages']==43 and candidate['depth_residual_stages']==42 and anchor['method']=='round6_depth','Wrong frozen stage/anchor family')
  require(candidate['postcal_factor_file_sha256']==freeze['training']['factors_sha256'] and candidate['postcal_train_complete_sha256']==freeze['training']['complete_sha256'],'Frozen final200 artifacts differ')
  b=baseline['owned_initial']['complete_visual_owner_bytes'];c=candidate['owned_initial']['complete_visual_owner_bytes'];delta=c-b
  require(delta==1344-18*nv and c<=b and candidate['postcal_added_encoder_constant_bytes']==1344 and candidate['postcal_added_retained_tensor_bytes']==1344,'New table charge/R42 saving mismatch')
  require(anchor['depth_residual_actual_budget_bits']<=1.8+1e-12,'K2 rate anchor exceeds declared budget')
  deltas.append(delta);nvs.append(nv);k2stages.append(anchor['depth_residual_stages']);learnedbytes.append(c);r7bytes.append(b)
 role=('Official DocVQA TRAIN project-source holdout; not official validation/test' if a.pool=='doc_train64' else 'Official InfoVQA validation source holdout')
 result=dict(status='PASS',pool=a.pool,data_identity=role,method_freeze_sha256=freeze_sha,all40_frozen_files_unchanged=True,
  sources=64,questions=64,records=256,exact_frozen_variants=freeze['confirmation_variant_names'],exact_config_sample_ID_order=True,
  same_image_realNv_cap8_first_question_per_source=True,Nv_range=[min(nvs),max(nvs)],learned_table_encoder_bytes=1344,
  candidate_minus_frozen_R7_bytes_range=[min(deltas),max(deltas)],candidate_owner_bytes_range=[min(learnedbytes),max(learnedbytes)],oldR7_owner_bytes_range=[min(r7bytes),max(r7bytes)],
  oldK2_anchor=dict(method='round6_depth',book='global256x4',V='binary/all',budget_bits=1.8,stage_prefix='actualNv-affordable dynamic prefix',actual_stage_count_range=[min(k2stages),max(k2stages)]),
  factors_sha256=freeze['training']['factors_sha256'],training_complete_sha256=freeze['training']['complete_sha256'],
  records_sha256=sha(folder/'records.jsonl'),final_marker_sha256=sha(folder/'complete_postcal.json'),config_sha256=spec['config_sha256'],samples_sha256=spec['samples_sha256'],
  new_GT_values_read_by_this_audit=False,method_or_parameter_changes=False,GPU_executed=False,checker_sha256=sha(__file__))
 with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
 print(json.dumps(result))


if __name__=='__main__':main()
