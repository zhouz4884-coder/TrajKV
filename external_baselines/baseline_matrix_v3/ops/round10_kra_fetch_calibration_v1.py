"""Read-only independent KRA calibration source/launch/scalar retrieval.

The single calibration worker has a launcher PID, rather than a sequence
parent with START_CELL/exit0. That distinction is explicit in the receipt.
No private fit PT, native KV, W_O, model weights or images are retrieved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

INSPECT = r'''
import hashlib,json,sys
from collections import Counter
from pathlib import Path,PurePosixPath
r=json.load(sys.stdin);root=Path(r['root']);name=r['job_name']
assert name.replace('_','').replace('-','').isalnum()
def h(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
 return digest.hexdigest()
jobpath=root/'jobs'/(name+'.json');launchpath=root/'jobs'/(name+'.launch.json')
if not launchpath.is_file():
 print(json.dumps({'status':'PENDING_KRA_CALIBRATION_NOT_LAUNCHED'}));raise SystemExit(0)
job=json.loads(jobpath.read_text());launch=json.loads(launchpath.read_text());argv=job['argv']
assert job['name']==launch['name']==name and type(launch['pid']) is int and launch['pid']>0
assert launch['job_sha256']==h(jobpath)
source=root/'jobs'/(name+'.source')
assert launch['source_snapshot']==str(source)
executed=[str(source/Path(a).relative_to(root)) if a.startswith(str(root)+'/code/') and a.endswith('.py') else a for a in argv]
assert executed==launch['argv_executed'] and argv[1]=='-B'
assert argv[2]==str(root/'code/round10_kra_calibrate_v1.py')
out=Path(argv[argv.index('--output')+1]);rel=out.relative_to(root/'results/round10_kra_v1')
assert len(rel.parts)==1 and rel.name.replace('_','').replace('-','').isalnum()
if not (out/'complete.json').is_file():
 print(json.dumps({'status':'PENDING_KRA_CALIBRATION_NOT_CLOSED'}));raise SystemExit(0)
done=json.loads((out/'complete.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
cfgpath=Path(argv[argv.index('--config')+1]);cfg=json.loads(cfgpath.read_text())
assert cfgpath.parent in (root/'configs/round10_v1',root/'configs/round10_kra_v1')
assert manifest['config']==dict(cfg,calibration_config_sha256=h(cfgpath))
assert manifest['runtime_pid']==launch['pid'] and manifest['runtime_source_root']==str(source)
assert manifest['runtime_executable']==executed[0] and manifest['runtime_argv']==[executed[0],*executed[2:]]
assert manifest['schema']=='trajkv.round10.kra_calibration_run.v1'
assert done['schema']=='trajkv.round10.kra_calibration_complete.v1'
assert done['actual_GPU'] is True and done['references_read'] is False
assert done['status'] in ('PASS_KRA_T0_ACTUAL_MODEL_IDENTITY','PASS_KRA_FIT48_REUSED_AND_ACTUAL_MODEL_SELECT16')
technical=done['status']=='PASS_KRA_T0_ACTUAL_MODEL_IDENTITY'
assert technical==manifest['identity_only']==bool('--identity-only' in argv or cfg.get('technical_only',False))
assert done['fit_sources_reused']==48 and done['new_fit_source_captures']==0
assert done['select_sources']==(1 if technical else 16) and done['teacher_cap']==32
assert done['new_free_generation_task_answers']==0
assert done['source_hashes']==manifest['source_hashes']
inventory=source/'sha256.json';assert h(inventory)==launch['source_inventory_sha256'];iv=json.loads(inventory.read_text())
for rel,digest in done['source_hashes'].items():
 assert rel.startswith('code/') and '..' not in Path(rel).parts and iv[rel]==digest==h(source/rel)
assert h(out/'records.jsonl')==done['records_sha256'] and h(out/'identity.json')==done['identity_sha256']
identity=json.loads((out/'identity.json').read_text())
assert identity['status']=='PASS_T0_FIELDS_AND_ACTUAL_MODEL_LOGITS_IDENTICAL' and identity['actual_GPU'] is True
assert identity['original_R7_actual_TF']['actual_raw_logits_sha256']==identity['KRA1_t0_actual_TF']['actual_raw_logits_sha256']
assert identity['original_R7_actual_TF']['position_KL']==identity['KRA1_t0_actual_TF']['position_KL']
assert identity['KRA1_t0_complete_bytes']==identity['original_complete_bytes']+5980
assert identity['actual_metric_increment_bytes']==5980
records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
teachers=[x for x in records if x['phase']=='select_source_teacher'];branches=[x for x in records if x['phase']=='select_branch']
assert len(teachers)==done['select_sources'] and len({(x['dataset'],x['source_id']) for x in teachers})==len(teachers)
assert len(branches)==(1 if technical else 64)
for row in branches:
 detail=row['owner_metadata']['round10_kra_candidate'];n=row['context']['visual_tokens']
 assert row['mode']=='KRA1' and detail['mode']=='KRA1' and detail['calibration_only'] is True
 assert detail['new_persistent_constant_bytes']==detail['actual_extra_persistent_bytes']==5980
 assert detail['before_unique_capacity_bytes']==4590*n+89733
 assert detail['after_unique_capacity_bytes']==row['owner_before_question_suffix']['stored_visual_KV_bytes']==4590*n+89733+5980
 assert all(detail[key] is True for key in ('original_K23_V20_coverage','original_K_base_fields_unchanged','original_V_base_fields_unchanged','all_residual_sign_codes_unchanged','original_V20_alpha_unchanged','original_lookup_ranking_policy_unchanged','reader_unchanged'))
 assert detail['native_FP16_source_or_workspace_retained'] is False
 assert row['loss']['actual_model_TF_source_evaluation'] is True
 assert row['loss']['actual_packed_owner_calls']==28*row['loss']['positions']
 assert row['loss']['question_native_KV_reused'] is False and row['loss']['question_suffix_recomputed'] is True
 assert row['loss']['current_query_and_text_KV'] is True and row['loss']['positions']<=32
for teacher in teachers:
 values=[x for x in branches if (x['dataset'],x['source_id'])==(teacher['dataset'],teacher['source_id'])]
 assert {x['strength'] for x in values}==({0.} if technical else {0.,.25,.5,1.})
 assert len({x['immutable_fields_sha256'] for x in values})==1
if not technical:
 assert Counter(x['dataset'] for x in teachers)==Counter({key:4 for key in ('textvqa','infovqa','docvqa','chartqa')})
 assert h(out/'selected_metric.json')==done['metric_sha256'] and h(out/'selection.json')==done['selection_sha256']
 selection=json.loads((out/'selection.json').read_text());metric=json.loads((out/'selected_metric.json').read_text())
 assert selection['selected_strength']==done['selected_strength'] in (.25,.5,1.)
 assert selection['strength_grid']==[0.,.25,.5,1.] and selection['zero_is_control_not_candidate_gate'] is True
 means={t:sum(x['loss']['source_mean_KL'] for x in branches if x['strength']==t)/16 for t in (0.,.25,.5,1.)}
 assert selection['source_balanced_mean_KL']=={str(k):v for k,v in means.items()}
 assert done['selected_strength']==min((.25,.5,1.),key=lambda t:(means[t],t))
 assert metric['schema']=='trajkv.round10.kra_metric.v1' and metric['strength']==done['selected_strength']
 assert metric['metric_storage_bytes']==5980 and metric['selection_provenance']==selection
logpath=Path(launch['log']);logbody=logpath.read_bytes();matches=[]
for number,line in enumerate(logbody.splitlines(keepends=True),1):
 try:event=json.loads(line)
 except (ValueError,UnicodeDecodeError):continue
 if isinstance(event,dict) and event.get('schema')==done['schema'] and event.get('status')==done['status']:
  assert event==done
  matches.append({'line_number':number,'exact_line_utf8':line.decode(),'exact_line_sha256':hashlib.sha256(line).hexdigest()})
assert len(matches)==1
files=[]
def add(label,path,digest=None,maximum=256*1024**2):
 path=Path(path);assert path.is_file() and path.suffix.lower() not in ('.pt','.pth','.bin','.safetensors') and path.stat().st_size<=maximum
 actual=h(path)
 if digest is not None:assert actual==digest
 files.append({'file':label,'remote_file':str(path),'sha256':actual,'bytes':path.stat().st_size})
for name2 in ('manifest.json','complete.json','records.jsonl','identity.json','metric_fit48.json','metric_strength0_control.json','metric_strength025.json','metric_strength05.json','metric_strength1.json','selected_metric.json','selection.json'):
 if (out/name2).is_file():add(name2,out/name2)
add('source_config.json',cfgpath);add('source_job.json',jobpath,launch['job_sha256'])
add('launch_receipt.json',launchpath);add('source_inventory.json',inventory,launch['source_inventory_sha256'])
add('source_driver.py',source/'code/round10_kra_calibrate_v1.py',done['source_hashes']['code/round10_kra_calibrate_v1.py'])
add('runtime_log.txt',logpath,maximum=16*1024**2)
for rel,digest in done['source_hashes'].items():add('source_files/'+rel,source/rel,digest)
for role in ('fit','select'):
 add('inputs/cal_'+role+'_inputs.jsonl',cfg['cal_'+role+'_inputs'],cfg['cal_'+role+'_sha256'])
add('inputs/CAL_INPUT_MANIFEST_v1.json',cfg['input_manifest_path'])
parent=Path(cfg['closed_collection_dir']);assert h(parent/'complete.json')==cfg['closed_collection_complete_sha256']==done['parent_calibration_complete_sha256']
for name2 in ('complete.json','manifest.json','fit_collection_complete.json'):add('parent_calibration_scalar/'+name2,parent/name2)
assert len(files)==len({x['file'] for x in files})
print(json.dumps({'schema':'trajkv.round10.kra_calibration_retrieval.v1',
 'status':'PASS_REMOTE_KRA_CALIBRATION_LAUNCH_SOURCE_AND_SCALAR_SHA',
 'scope':'KRA calibration not free-generation task quality; no private fit PT/WO/nativeKV',
 'remote_hashes_computed_independently':True,'SourceRun_admission_established':False,
 'job_name':name,'actual_launch_pid':launch['pid'],'actual_argv_executed':executed,
 'source_snapshot':str(source),'closed_status':done['status'],'identity_only':technical,
 'actual_complete_stdout_event':matches[0],
 'single_worker_has_no_sequence_parent_START_CELL_or_exit0_receipt':True,
 'OS_process_exit0_independently_observed':False,'files':files}))
'''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    from remote_control import REMOTE_ROOT, connect
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve())
    client = connect()
    def inspect():
        inp, out, err = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', INSPECT]), timeout=60)
        inp.write(json.dumps({'root': REMOTE_ROOT, 'job_name': args.job_name}))
        inp.flush(); inp.channel.shutdown_write()
        body, error = out.read(), err.read()
        assert out.channel.recv_exit_status() == 0, error.decode(errors='replace')[-2500:]
        return json.loads(body)
    try:
        proof = inspect()
        if proof['status'].startswith('PENDING_'):
            print(json.dumps(proof)); return
        args.output.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        try:
            for item in proof['files']:
                relative = PurePosixPath(item['file'])
                assert not relative.is_absolute() and '..' not in relative.parts
                target = args.output.joinpath(*relative.parts)
                if target.exists():
                    assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
                    continue
                with sftp.open(item['remote_file'], 'rb') as source:
                    body = source.read()
                assert len(body) == item['bytes'] and hashlib.sha256(body).hexdigest() == item['sha256']
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(body)
        finally:
            sftp.close()
        assert inspect() == proof, 'closed remote calibration changed during retrieval'
        receipt = args.output/'remote_kra_calibration_retrieval_v1.json'
        body = (json.dumps(proof, indent=2, ensure_ascii=False) + '\n').encode()
        if receipt.exists():
            assert receipt.read_bytes() == body
        else:
            with receipt.open('xb') as stream:
                stream.write(body)
        print(json.dumps({'status': proof['status'], 'files': len(proof['files']),
                          'closed_status': proof['closed_status'], 'output': str(args.output)}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
