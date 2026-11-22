"""Independently retrieve the already completed middle QA0 technical worker.

Read/hash/SFTP only: no new inference, scoring, queue or remote writes. The
standalone worker has a launcher-recorded PID, not a sequence exit0 receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

INSPECT = r'''
import hashlib,json,math,sys
from pathlib import Path,PurePosixPath
r=json.load(sys.stdin);root=Path(r['root']);name='bm_R10_middle_technical_v1'
out=root/'results/round10_middle_v1/technical_v1'
def h(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
 return digest.hexdigest()
assert (out/'complete.json').is_file()
done=json.loads((out/'complete.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
jobpath=root/'jobs'/(name+'.json');launchpath=root/'jobs'/(name+'.launch.json')
job=json.loads(jobpath.read_text());launch=json.loads(launchpath.read_text());argv=job['argv']
assert job['name']==launch['name']==name and type(launch['pid']) is int and launch['pid']>0
assert launch['job_sha256']==h(jobpath)
source=root/'jobs'/(name+'.source');assert launch['source_snapshot']==str(source)
executed=[str(source/Path(a).relative_to(root)) if a.startswith(str(root)+'/code/') and a.endswith('.py') else a for a in argv]
assert launch['argv_executed']==executed and argv[1]=='-B'
assert argv[2]==str(root/'code/round10_middle_technical_v1.py')
assert argv[argv.index('--output')+1]==str(out)
cfgpath=Path(argv[argv.index('--config')+1]);cfg=json.loads(cfgpath.read_text())
assert cfgpath==root/'configs/round10_middle_v1/technical_v1.json'
assert manifest['config']==dict(cfg,calibration_config_sha256=h(cfgpath))
assert cfg['technical_only'] is True and cfg['model_id']=='M1'
assert done['schema']=='trajkv.round10.middle_technical.v1'
assert done['status']=='PASS_MIDDLE_ONE_SOURCE_ACTUAL_PACKED_FULL_TF' and done['actual_GPU'] is True
assert done['actual_model_parameter_count']==7944373760 and done['weight_dtype']=='float16'
assert done['QA_events']==done['new_free_generation_task_answers']==0 and done['actual_TF_source_evaluations']==2
assert done['references_read'] is False and done['new_parameters_selected'] is False
assert done['original_calibration_four_controls_executed'] is False and done['full_fit48_select16_completed'] is False
assert done['quality_claim'] is False and done['speed_or_peak_benchmark_claim'] is False
assert done['teacher_sources']==1 and 0<done['teacher_tokens']==len(done['teacher_token_ids'])<=32
assert done['teacher_cap']==32 and done['teacher_actual_native_history'] is True
assert done['source_hashes']==manifest['source_hashes']
assert done['middle_runtime_source_hashes_before']==done['middle_runtime_source_hashes_after']
assert done['input_identity_before']==done['input_identity_after']
assert done['native_prefix_identity_before']==done['native_prefix_identity_after'] and done['native_prefix_unchanged'] is True
assert done['native_prefix_identity_before']['all_CUDA_FP16'] is True
assert len(done['native_prefix_identity_before']['layers'])==28
inventory=source/'sha256.json';assert h(inventory)==launch['source_inventory_sha256'];iv=json.loads(inventory.read_text())
sources=dict(done['source_hashes'])
for rel,digest in done['middle_runtime_source_hashes_before'].items():
 assert rel in sources and sources[rel]==digest
for rel,digest in sources.items():
 assert rel.startswith('code/') and '..' not in Path(rel).parts and iv[rel]==digest==h(source/rel)
role=json.loads((out/'MIDDLE_TECHNICAL_ROLE.json').read_text())
assert role['schema']==done['schema'] and role['QA_events']==0 and role['actual_TF_source_evaluations']==2
controls=done['controls'];assert len(controls)==2 and {x['mode'] for x in controls}=={'V128-R43','AddOldOrder'}
n=done['context']['visual_tokens'];b0=4590*n+89733;v128=5038*n+32389;q=(448*n-57344)//(18*n)
assert done['base43_actual_owner']['complete_visual_owner_bytes']==b0
for control in controls:
 mode=control['mode'];amount=v128 if mode=='V128-R43' else b0+18*n*q
 assert control['actual_owner_before_TF']['complete_visual_owner_bytes']==control['actual_owner_after_TF']['complete_visual_owner_bytes']==amount
 assert control['physical_budget_guard']['actual_complete_visual_bytes']==amount
 assert control['packed_fields_unchanged_after_full_TF'] is True and control['native_source_prefix_unchanged'] is True
 assert control['loss']['actual_model_TF_source_evaluation'] is True
 assert control['loss']['actual_packed_owner_calls']==28*done['teacher_tokens']
 assert control['loss']['positions']==done['teacher_tokens']
 assert control['loss']['question_native_KV_reused'] is False and control['loss']['question_suffix_recomputed'] is True
 assert control['loss']['current_query_and_text_KV'] is True
 assert control['actual_query_dense_diagnostic']['dense_decode_is_current_candidate'] is True
 assert control['actual_query_dense_diagnostic']['dense_reference_workspace_retained'] is False
 if mode=='V128-R43':assert control['complete_K_signature_equals_base43'] is True
 else:assert control['original43_code_alpha_preserved'] is True and control['original43_preservation_for_Add_independently_rechecked'] is True
logpath=Path(launch['log']);body=logpath.read_bytes();matches=[]
expected={key:done[key] for key in ('status','teacher_sources','teacher_tokens','actual_TF_source_evaluations','QA_events')}
for number,line in enumerate(body.splitlines(keepends=True),1):
 try:event=json.loads(line)
 except (ValueError,UnicodeDecodeError):continue
 if event==expected:matches.append({'line_number':number,'exact_line_utf8':line.decode(),'exact_line_sha256':hashlib.sha256(line).hexdigest()})
assert len(matches)==1
files=[]
def add(label,path,digest=None,maximum=16*1024**2):
 path=Path(path);assert path.is_file() and path.suffix.lower() in ('.py','.json','.jsonl','.txt','.log') and path.stat().st_size<=maximum
 actual=h(path)
 if digest is not None:assert actual==digest
 files.append({'file':label,'remote_file':str(path),'sha256':actual,'bytes':path.stat().st_size})
for name2 in ('manifest.json','complete.json','MIDDLE_TECHNICAL_ROLE.json'):add(name2,out/name2)
assert not (out/'records.jsonl').exists()
add('source_config.json',cfgpath);add('source_job.json',jobpath,launch['job_sha256'])
add('launch_receipt.json',launchpath);add('source_inventory.json',inventory,launch['source_inventory_sha256'])
add('source_driver.py',source/'code/round10_middle_technical_v1.py',sources['code/round10_middle_technical_v1.py'])
add('runtime_log.txt',logpath)
for rel,digest in sources.items():add('source_files/'+rel,source/rel,digest)
for role,item in done['input_identity_before'].items():
 if role=='source_image':continue
 assert h(item['path'])==item['sha256']
 label='inputs/'+role.replace('.','_')+Path(item['path']).suffix
 add(label,item['path'],item['sha256'])
assert len(files)==len({x['file'] for x in files})
print(json.dumps({'schema':'trajkv.round10.middle_technical_retrieval.v1',
 'status':'PASS_REMOTE_MIDDLE_TECHNICAL_LAUNCH_SOURCE_AND_SCALAR_SHA',
 'scope':'already completed actual8B technical one-source/two-TF/QA0; no task score',
 'remote_hashes_computed_independently':True,'SourceRun_admission_established':False,
 'job_name':name,'launcher_recorded_worker_pid':launch['pid'],'actual_argv_executed':executed,
 'source_snapshot':str(source),'closed_status':done['status'],'actual_complete_stdout_event':matches[0],
 'single_worker_has_no_sequence_parent_START_CELL_or_exit0_receipt':True,
 'OS_process_exit0_independently_observed':False,'worker_pid_is_launcher_record_not_independent_in_model_observation':True,
 'actual_model_parameter_count':done['actual_model_parameter_count'],'actual_GPU_reported':True,
 'QA_events':0,'actual_TF_source_evaluations':2,'teacher_sources':1,'teacher_tokens':done['teacher_tokens'],
 'declared_source_files':len(sources),'middle_runtime_source_files':len(done['middle_runtime_source_hashes_before']),
 'raw_JSONL_created_by_this_worker':False,'private_tensors_weights_images_retrieved':False,'files':files}))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from remote_control import REMOTE_ROOT, connect
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('${EVIDENCE_ROOT}/results/round10/middle_technical_retrieved_v1'))
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve())
    client = connect()
    def inspect():
        inp, out, err = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', INSPECT]), timeout=60)
        inp.write(json.dumps({'root': REMOTE_ROOT})); inp.flush(); inp.channel.shutdown_write()
        body, error = out.read(), err.read()
        assert out.channel.recv_exit_status() == 0, error.decode(errors='replace')[-2500:]
        return json.loads(body)
    try:
        proof = inspect()
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
        assert inspect() == proof, 'closed remote technical evidence changed during retrieval'
        proof.update(retrieval_tool_sha256=sha(__file__), remote_mutations_performed=False,
                     new_GPU_jobs_started=0, task_scoring_performed=False)
        receipt = args.output/'remote_middle_technical_retrieval_v1.json'
        body = (json.dumps(proof, indent=2, ensure_ascii=False) + '\n').encode()
        if receipt.exists():
            assert receipt.read_bytes() == body
        else:
            with receipt.open('xb') as stream:
                stream.write(body)
        print(json.dumps({'status': proof['status'], 'files': len(proof['files']),
            'declared_source_files': proof['declared_source_files'], 'QA_events': 0,
            'actual_TF_source_evaluations': 2, 'output': str(args.output)}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
