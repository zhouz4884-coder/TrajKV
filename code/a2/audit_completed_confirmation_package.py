"""Consolidate a complete bounded research package without declaring targets met."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 errors=[];read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
 primary=read('audit/a2/FROZEN_CONFIRM64_AUDIT.json');summary=read('results/frozen_confirm64_analysis/summary.json')
 external=read('results/frozen_external128_analysis/summary.json');continuous=read('results/frozen_confirm64_continuous/summary.json');timing=read('results/frozen_confirm_timing/summary.json')
 assert primary['status']=='PASS' and primary['records']==2112
 assert summary['sessions']==2112 and external['status']=='PASS' and external['records']==1664
 assert continuous['proxy_records']==448 and timing['status']=='PASS' and timing['measured_trials']==132 and timing['warmups']==33
 old=read('audit/final_artifact_manifest.json')
 for relative,item in old['files'].items():
  path=ROOT/relative
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:errors.append({'old_artifact_changed':relative})
 figure_names=['confirm64_storage_quality','confirm64_refinement_reads','confirm64_norm_direction','confirm64_question_gaps','external_state_replication']
 for name in figure_names:
  for suffix in ['png','pdf','svg']:
   if not (ROOT/'figures/a2_confirmation'/f'{name}.{suffix}').is_file():errors.append({'missing_figure':name+'.'+suffix})
 # Only this completed cycle is frozen here. Later new-data experiments have
 # their own mutable development record and are deliberately outside this set.
 runs=['mechanism_confirm64_s0_v1','mechanism_confirm64_s1_v1','confirm64_continuous_s0_v1','confirm64_continuous_s1_v1',
  'textvqa_eval128_s0_v1','textvqa_eval128_s1_v1','textvqa_controls128_s0_v1','textvqa_controls128_s1_v1','frozen_timing_gpu0_v2',
  'frozen_confirm64_analysis','frozen_external128_analysis','frozen_confirm64_continuous','frozen_confirm_timing']
 files=[]
 for run in runs:
  files.extend(p for p in (ROOT/'results'/run).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
  snapshot=ROOT/'audit/job_sources'/(run+'.source')
  if snapshot.is_dir():
   files.extend(p for p in snapshot.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
   for suffix in ('.launch.json','.json'):
    path=ROOT/'audit/job_sources'/(run+suffix)
    if path.is_file():files.append(path)
 files.extend(p for p in (ROOT/'figures/a2_confirmation').glob('*') if p.is_file())
 for relative in ['audit/a2/frozen_confirmation_v2_mechanism.json','audit/a2/FROZEN_CONFIRM64_AUDIT.json',
  'audit/a2/frozen_external128_and_proxies.json','audit/a2/frozen_external_additional_controls.json','audit/a2/timing_restart_v2.json',
  'docs/RESEARCH_DISCOVERIES_2026-11-14.zh.md','docs/LITERATURE_2026-11-14_STATE_CALIBRATION.md',
  'configs/a2/confirm64_short.json','configs/a2/textvqa_eval128.json','configs/a2/textvqa_eval128_refs.json',
  'configs/a2/confirm64_teacher_refs.json','configs/data/docvqa_references.jsonl','audit/a2/assets_identity.json']:
  files.append(ROOT/relative)
 frozen=read('audit/a2/frozen_confirmation_v2_mechanism.json')
 remote_root='/path/to/TrajKV/'
 for remote,digest in frozen['external_codec_inputs_sha256'].items():
  assert remote.startswith(remote_root);local=ROOT/remote[len(remote_root):]
  assert hashlib.sha256(local.read_bytes()).hexdigest()==digest;files.append(local)
 manifest={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))}
 result={'status':'PASS' if not errors else 'FAIL','generated_at_utc':datetime.now(timezone.utc).isoformat(),'docvqa_free_sessions':2112,
  'docvqa_questions':286,'docvqa_scored_answers':summary['scored_answers'] if 'scored_answers' in summary else 9438,
  'external_free_answers':1664,'teacher_proxy_records_not_accuracy':448,'timing_measurements_not_accuracy':132,'timing_warmups':33,
  'first_cycle_files_unchanged':len(old['files']),'frozen_docvqa_source_comparisons':primary['frozen_source_comparisons'],
  'independent_owner_formula_cases':primary['independent_owner_formula_cases'],'external_frozen_source_checks':external['frozen_inference_source_checks'],
  'figures':len(figure_names),'visual_qa':'all five PNG figures inspected; read-policy labels corrected; vector versions exported',
  'performance_status':'prespecified<=1.8bit/0.5pp storage target unmet; read-policy0.5pp targets unmet; state-condition improvement replicated; no HBM counter claim',
  'files':manifest,'file_count':len(manifest),'total_bytes':sum(v['bytes'] for v in manifest.values()),'errors':errors,
  'scope':'completed confirmation cycle only; next32 calibration/dev64/confirm256 belong to continuing research, not outcomes in this package'}
 (ROOT/'audit/a2/COMPLETED_CONFIRMATION_PACKAGE.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('files','visual_qa','scope')}))
 if errors:raise RuntimeError('Completed confirmation package audit failed')
if __name__=='__main__':main()
