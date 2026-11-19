"""Independent final-marker, same-QA baseline identity and header summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/a2'))
from analyze_round8_final_quality import completion_gate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def check(dataset):
 path=ROOT/('results/round8_v1/mean_select2_v2_'+dataset)
 gate=completion_gate(path)
 if not gate or gate['status']!='PASS':raise RuntimeError('Mean final gate not passed')
 actual=rows(path/'records.jsonl');off={r['session_id']:r for r in actual if r['variant']=='r8_visual_mean_off'}
 on={r['session_id']:r for r in actual if r['variant']=='r8_visual_mean_on'}
 if set(off)!=set(on):raise RuntimeError('off/on source coverage differs')
 checks=[];nonidentity=[]
 baseline_names=[('postres_select2_v1_','r8_postres_direction_off'),('joint_cd_select5_v1_','r8_l2_joint0'),
  ('postfit_cap8_aligned11_v3_','r7_frozen_joint43_control'),('postfit_cap8_aligned11_v3_','r8_k4old_vbinary')]
 for dirname,variant in baseline_names:
  baseline=ROOT/('results/round8_v1/'+dirname+dataset)
  if not (baseline/'records.jsonl').is_file():continue
  bg=completion_gate(baseline)
  if bg and bg['status']!='PASS':raise RuntimeError('Comparator final gate missing')
  old={r['session_id']:r for r in rows(baseline/'records.jsonl') if r['variant']==variant}
  if not old:continue
  if set(off)!=set(old):raise RuntimeError('Comparator source coverage differs')
  counts=dict(answer=0,token_ids=0,initial_logits_sha256=0,owned_initial=0)
  for sid,a in off.items():
   b=old[sid]
   for key in counts:
    value=([t[key] for t in a['turns']]==[t[key] for t in b['turns']]) if key in ('answer','token_ids') else a[key]==b[key]
    counts[key]+=value
    if not value:nonidentity.append(dict(source=sid,baseline=dirname,variant=variant,field=key))
  checks.append(dict(run=baseline.relative_to(ROOT).as_posix(),variant=variant,questions=len(off),exact_matches=counts,
   records_sha256=sha(baseline/'records.jsonl')))
 if not checks or any(c['exact_matches'][key]!=len(off) for c in checks for key in c['exact_matches']):
  raise RuntimeError('Mean off is not an exact preserved baseline: '+json.dumps(nonidentity))
 own=[];before=[];after=[];shift=[];header_bound=[]
 changed=dict(answer=0,token_ids=0,initial_logits_sha256=0)
 for sid,a in off.items():
  b=on[sid]
  if a['owned_initial']!=b['owned_initial']:raise RuntimeError('on/off owned tensor bytes differ')
  own.append(a['owned_initial']['complete_visual_owner_bytes'])
  for key in changed:
   changed[key]+=([t[key] for t in a['turns']]!=[t[key] for t in b['turns']]) if key in ('answer','token_ids') else a[key]!=b[key]
  for layer in b['mean_correction_statistics']:
   for head in layer['per_head']:
    before.append(head['stored_target_residual_l2_before']);after.append(head['stored_target_residual_l2_after'])
    shift.append(head['header_shift_l2']);header_bound.append(head['residual_component_bound_max'])
 return dict(dataset=dataset,status='PASS',questions=len(off),sources=16,records=len(actual),baseline_identity_checks=checks,
  owner_bytes_off_on_equal=True,owner_byte_range=[min(own),max(own)],added_persistent_tensor_bytes=0,
  on_off_changed_questions=changed,header_heads=len(before),header_stored_target_residual_l2_before_mean=statistics.mean(before),
  header_stored_target_residual_l2_after_mean=statistics.mean(after),header_stored_target_residual_l2_before_max=max(before),
  header_stored_target_residual_l2_after_max=max(after),header_shift_l2_mean=statistics.mean(shift),
  header_component_bound_max=max(header_bound),header_residual_reduced_heads=sum(a<b for a,b in zip(after,before)),
  records_sha256=sha(path/'records.jsonl'),final_marker_sha256=sha(path/'complete_mean.json'),
  target='Stored FP16 source mean only; not exact original mean or task guarantee',primary_quality_scored_here=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['doc','text'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=check(a.dataset);result['checker_sha256']=sha(__file__)
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))


if __name__=='__main__':main()
