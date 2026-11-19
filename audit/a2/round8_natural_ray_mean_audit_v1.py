"""Local final-gated natural64 source/cap/owner/intervention audit; no GT."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/a2'))
from analyze_round8_final_quality import completion_gate


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(line) for line in Path(p).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise RuntimeError(message)


def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['final_ray','mean'],required=True);p.add_argument('--dataset',choices=['doc','text'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 prepared=read(ROOT/'audit/a2/round8_natural_ray_mean_v1_prepared.json');spec=next(s for s in prepared['outputs'] if s['family']==a.family and s['dataset']==a.dataset)
 folder=ROOT/'results/round8_v1'/Path(spec['output']).name;gate=completion_gate(folder);require(gate and gate['status']=='PASS','Final marker graph gate failed')
 manifest=read(folder/'manifest.json');data=rows(folder/'records.jsonl');samples=read(ROOT/spec['samples'])
 require(manifest['selected_ids']==[r['id'] for r in samples] and manifest['samples_sha256']==sha(ROOT/spec['samples'])==spec['samples_sha256'],'Frozen selected source order/input SHA mismatch')
 source={r['id']:r for r in samples};by={r['session_id']:r for r in data};require(set(by)==set(source) and len(by)==len(data)==64,'Exact64 singleton coverage mismatch')
 baseline=ROOT/f'results/round8_v1/natural_dev64_firstq3_v1_{a.dataset}';old={r['session_id']:r for r in rows(baseline/'records.jsonl') if r['variant']=='r7_frozen_joint43_control'}
 require(set(old)==set(by),'Baseline IDs differ')
 names=[];nv=[];owner=[];head_before=[];head_after=[];expected_variant=manifest['config']['variants'][0];ranking=expected_variant['selected_stages'];selected=ranking[:43]
 kheads={(l,h) for axis,l,h,s in selected if axis=='k'};require(len(ranking)==224 and len(kheads)==23 and sum(t[0]=='v' for t in selected)==20,'Wrong frozen224/43/23K20V')
 for sid,r in by.items():
  b=old[sid];sample=source[sid]
  require(r['image_id']==b['image_id']==sample['image_id'] and r['visual_tokens']==b['visual_tokens']
   and r['max_patches']==b['max_patches']==8 and r['owned_initial']==b['owned_initial']
   and r['first_question_suffix_replay_tokens']==b['first_question_suffix_replay_tokens']>0,'Same source/Nv/cap/owner/question-prefix control failed')
  require([t['turn'] for t in r['turns']]==[1] and r['question_state_scope']=='recomputed_under_codec'
   and r['initial_logits_shared_fp16'] is False and r['read_ledger_exact'] is True and r.get('global_extra_budget_ok',True),'First-question/student-state/read contract failed')
  require(r['variant_config']==expected_variant and r['depth_residual_requested_stages']==ranking and r['depth_residual_selected_stages']==selected
   and r['depth_residual_stages']==43 and r['depth_residual_encoder_ranking_bytes']==448
   and r['depth_residual_K_stages']==23 and r['depth_residual_V_stages']==20
   and r['depth_residual_source_fp16_retained'] is False and r['visual_fp16_materialization'] is False,'Frozen geometry/owner contract failed')
  if a.family=='final_ray':
   stats=r['final_ray_statistics'];actual=[(s['layer'],s['head']) for s in stats]
   require(r['final_ray_mode']=='norm' and r['final_ray_J_sign_unchanged'] is True and r['final_ray_reader_unchanged'] is True
    and r['final_ray_added_retained_tensor_bytes']==0 and len(actual)==len(set(actual))==23 and set(actual)==kheads,'Ray norm head coverage/unchanged-field contract failed')
   require(all(s['mode']=='norm' and s['rows']==r['visual_tokens'] and s['indices_changed']==0 and s['sign_recode'] is False
    and s['initialization_verified_bitwise'] is True and s['physical_optimized_bytes_before']==s['physical_optimized_bytes_after']
    and s['added_retained_tensor_bytes']==0 and s['norm_error_after']<=s['norm_error_before']+1e-8 for s in stats),'Ray actual per-head final-state rule failed')
  else:
   require(r['mean_actual_decode_chunk']==4096 and len(r['mean_correction_statistics'])==28 and r['mean_correction_added_persistent_tensor_bytes']==0,'Mean chunk/layer/byte contract failed')
   for s in r['mean_correction_statistics']:
    require(s['visual_tokens']==r['visual_tokens'] and s['apply_count']==1 and s['actual_decode_chunk']==4096 and len(s['per_head'])==4,'Mean all-real-Nv/once/4-head coverage failed')
    for h in s['per_head']:head_before.append(h['stored_target_residual_l2_before']);head_after.append(h['stored_target_residual_l2_after'])
  nv.append(r['visual_tokens']);owner.append(r['owned_initial']['complete_visual_owner_bytes']);names.append(sid)
 result=dict(status='PASS',family=a.family,dataset=a.dataset,records=64,exact_original_source_order=True,unique_sources=64,
  original_R7_exact_Nv_image_id_cap_full_owned_initial_suffix_replay_matches=64,one_first_question_each=True,
  actual_cap8=True,Nv_min=min(nv),Nv_max=max(nv),complete_owner_byte_range=[min(owner),max(owner)],added_persistent_tensor_bytes=0,
  requested_ranking_entries=224,selected_stages=43,K_stages=23,V_stages=20,ranking_constant_bytes=448,
  final_marker=spec['required_final_marker'],final_marker_sha256=sha(folder/spec['required_final_marker']),records_sha256=sha(folder/'records.jsonl'),
  baseline_records_sha256=sha(baseline/'records.jsonl'),samples_sha256=sha(ROOT/spec['samples']),checker_sha256=sha(__file__),
  GT_references_opened=False,quality_scored_here=False,GPU_used=False,
  source_byte_scope='Same frozen samples/paths/image_id; separate remote_image_sha receipt verifies128 original image bytes without download')
 if head_before:result['mean_header_residual']=dict(heads=len(head_before),before_mean=sum(head_before)/len(head_before),after_mean=sum(head_after)/len(head_after),reduced_heads=sum(a<b for a,b in zip(head_after,head_before)),target='existing stored FP16 source mean; not exact original mean or GPU bias-elimination guarantee')
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))


if __name__=='__main__':main()
