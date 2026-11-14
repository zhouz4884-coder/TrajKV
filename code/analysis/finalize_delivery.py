"""Bind completed research artifacts to a final local delivery receipt."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
EXCLUDE={'audit/final_artifact_manifest.json','audit/final_delivery.json','audit/research_progress.json'}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    validation=json.loads((ROOT/'audit/FINAL_INDEPENDENT_VALIDATION.json').read_text(encoding='utf-8'))
    levels=validation['findings_by_level']
    if levels.get('ERROR',0) or levels.get('PENDING',0):raise RuntimeError('Independent validation has unresolved required work')
    numeric=json.loads((ROOT/'audit/final_numeric_validation.json').read_text(encoding='utf-8'))
    processes=json.loads((ROOT/'audit/final_model_process_validation.json').read_text(encoding='utf-8'))
    if numeric['status']!='PASS_ALL_REGISTERED_PRIMARY_RAW_FLOATS_FINITE':raise RuntimeError('Nonfinite/raw verification incomplete')
    if processes['status']!='PASS_MODEL_STATS_UNCHANGED_AND_NO_PROJECT_GPU_PROCESSES':raise RuntimeError('Workers or model identity unsettled')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    files={}
    for folder in ('code','configs','docs','audit','results','figures','paper'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name.endswith(('.pyc','.download-part')):continue
            rel=p.relative_to(ROOT).as_posix()
            if rel in EXCLUDE:continue
            files[rel]={'bytes':p.stat().st_size,'sha256':digest(p)}
    files['README.md']={'bytes':(ROOT/'README.md').stat().st_size,'sha256':digest(ROOT/'README.md')}
    manifest={'schema':'trajkv.final-artifacts.v1','generated_at':now,
              'scope':'local research code/config/evidence/figures; model/image originals remain at audited external asset paths',
              'files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values())}
    mp=ROOT/'audit/final_artifact_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt={'schema':'trajkv.final-delivery.v1','status':'COMPLETE_RESEARCH_CYCLE',
             'generated_at':now,'report':'docs/RESEARCH_CYCLE_REPORT_2026-11-12.zh.md',
             'goal_scope':'authorized two-week research work package, including implementation, iteration, heldout/confirmation evaluation and final report',
             'publication_claim':'current method advantage not established; no submission/acceptance claim',
             'primary_images':64,'primary_configurations':17,'primary_answers':10880,
             'kvarn_quality_reference_configurations':2,'combined_primary_quality_answers':12160,
             'replication_images':32,'replication_answers':3840,
             'doc_confirm_images':64,'doc_confirm_questions':294,'doc_confirm_answers':3528,
             'quiet_timing_measurements':108,'new_experiments_pending':False,
             'independent_validation':{'errors':0,'pending':0,'findings_by_level':levels,
                  'path':'audit/FINAL_INDEPENDENT_VALIDATION.json','sha256':digest(ROOT/'audit/FINAL_INDEPENDENT_VALIDATION.json')},
             'retained_exclusions':'16 historical development repair records exceeded the requested read allowance; excluded from equal-budget claims, never deleted',
             'hardware_counter_scope':'installed profiler failed CUDA init, zero kernels; DRAM counters null; reads are explicitly logical tensor payloads',
             'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':digest(mp),
             'file_count':len(files),'model_and_image_weights_repacked':False}
    (ROOT/'audit/final_delivery.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'files':len(files),'bytes':manifest['total_bytes'],
                      'errors':0,'pending':0,'receipt':str(ROOT/'audit/final_delivery.json')}))

if __name__=='__main__':main()
