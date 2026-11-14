"""Freeze the completed Doc256/Text128/timing cycle independently of new fits."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
 target=ROOT/'audit/a2/ROUND3_CONFIRMATION_PACKAGE.json'
 if target.exists():raise RuntimeError('Completed package already frozen; do not overwrite')
 doc=read('results/next_confirm256_analysis/summary.json');ext=read('results/next_textbalanced128_analysis/summary.json');timing=read('results/next_confirm_timing/summary.json')
 da=read('audit/a2/NEXT_CONFIRM256_AUDIT.json');ea=read('audit/a2/NEXT_TEXTBALANCED128_AUDIT.json')
 assert doc['sessions']==5888 and doc['scored_answers']==22977 and ext['records']==1664
 assert da['status']==ea['status']==timing['status']=='PASS'
 assert timing['measured_trials']==288 and timing['warmups']==96 and timing['frozen_inference_source_checks']==29
 assert not doc['storage_target']['point_target_met']
 primary=next(r for r in doc['read_targets'] if r['variant']==doc['primary_read_variant'])
 assert primary['point_target_met'] and not primary['noninferiority_95_ci']
 preserved={}
 for relative in ['audit/final_artifact_manifest.json','audit/a2/COMPLETED_CONFIRMATION_PACKAGE.json']:
  old=read(relative)
  for name,entry in old['files'].items():assert sha(ROOT/name)==entry['sha256'],name
  preserved[relative]={'files_unchanged':len(old['files']),'manifest_sha256':sha(ROOT/relative)}
 files=[]
 runs=['next_confirm256_s0_v1','next_confirm256_s1_v1','next_textbalanced128_s0_v1','next_textbalanced128_s1_v1','next_timing_gpu0_v1']
 for run in runs:
  assert read('results/'+run+'/complete.json')['status']=='COMPLETE'
  for folder in [ROOT/'results'/run,ROOT/'audit/job_sources'/(run+'.source')]:
   assert folder.is_dir();files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
  for suffix in ['.launch.json','.json']:
   path=ROOT/'audit/job_sources'/(run+suffix)
   if path.exists():files.append(path)
 for folder in ['results/next_confirm256_analysis','results/next_textbalanced128_analysis','results/next_confirm_timing','figures/a3_confirmation']:
  files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
 for name in ['doc256_storage_quality','doc256_read_quality','prefill_effect_development_vs_confirmation','external128_answer_strata','fixed_tokens_timing']:
  for suffix in ['png','pdf','svg']:assert (ROOT/'figures/a3_confirmation'/f'{name}.{suffix}').exists()
 fixed=['audit/a2/next_confirmation_v1.json','audit/a2/post_confirmation_execution_v1.json','audit/a2/NEXT_CONFIRM256_AUDIT.json','audit/a2/NEXT_TEXTBALANCED128_AUDIT.json',
  'audit/a2/next_docvqa_source_split_v1.json','audit/a2/next_docvqa_data_transport.json','audit/a2/textvqa_balanced128_membership.json','audit/a2/textvqa_balanced128_transport.json',
  'configs/a2/docvqa_next_confirm256.json','configs/a2/docvqa_next_cal32.json','configs/a2/docvqa_next_dev64.json',
  'configs/a2/textvqa_balanced128.json','configs/data/docvqa_next_references.jsonl','configs/data/textvqa_balanced128_references.json',
  'configs/a2/next_timing_gpu0_v1.json','docs/ROUND3_CONFIRMATION_2026-11-14.zh.md']
 for relative in fixed:assert (ROOT/relative).is_file();files.append(ROOT/relative)
 frozen=read('audit/a2/next_confirmation_v1.json');remote='/path/to/TrajKV/'
 for name,digest in frozen['external_codec_inputs_sha256'].items():
  assert name.startswith(remote);path=ROOT/name[len(remote):];assert sha(path)==digest;files.append(path)
 # Preserve the analysis implementation, while active development may continue.
 archive=ROOT/'audit/analysis_sources/round3_confirm256'
 scripts=list((ROOT/'code/a2').glob('analyze_next*.py'))+[ROOT/'code/a2'/name for name in ['analyze.py','analyze_frozen_timing.py','plot_next_confirmation.py','costs.py','audit_round3_package.py','textvqa_metric.py','vendor_textvqa_metric.py']]
 scripts += [ROOT/'code/analysis'/name for name in ['analyze_external.py','analyze_cycle.py','analyze_docvqa_confirm64.py']]
 assert all(source.is_file() for source in scripts),[str(p) for p in scripts if not p.is_file()]
 assert not archive.exists() or not any(archive.iterdir());archive.mkdir(parents=True,exist_ok=True)
 for source in scripts:
  dest=archive/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);files.append(dest)
 unique=sorted(set(files));manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in unique}
 result={'status':'PASS','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'doc_free_sessions':5888,'doc_scored_answers':22977,'external_free_answers':1664,
  'timing_measurements_not_accuracy':288,'timing_warmups':96,'preserved':preserved,
  'storage_target_met':False,'read_point_target_met':True,'read_95_ci_noninferiority':False,
  'end_to_end_acceleration_established':False,'physical_hbm_measurement':False,'figures_visually_inspected':5,
  'scope':'completed prospectively frozen confirmation cycle; learned books and their future evaluation are separate continuing research',
  'files':manifest,'file_count':len(manifest),'total_bytes':sum(r['bytes'] for r in manifest.values())}
 target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':main()
