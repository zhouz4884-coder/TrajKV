"""Read-only closed Round10 cells, exact source retrieval and frozen CPU scoring.

The parent is an immutable sequence job; a child is not a canonical SourceRun.
No worker, remote file, queue, GPU or Git write is performed. --fetch is explicit.
Every summary snapshot is new; per-cell immutable evidence may be reused.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ops'), str(ROOT/'code')]
DOMAINS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')
COUNTS = {'textvqa':64, 'infovqa':64, 'docvqa':16, 'chartqa':16}
MODES = ('FP16','R7low','R7high','LS-FZ','LS-RE','FB0','FB-W')
CANDIDATES = ('LS-FZ','LS-RE','FB0','FB-W')
PINS = {
    'code/score.py':'72b5182eeb97e930403a970feebb4f1fadac23d04544c9646c6c9fd6b34bfba6',
    'code/compare_runs.py':'ce73705f17cfcbfe66358f3525eb3121e2e141056bc9a043e7423546878d15a1',
    'ops/prepare_remaining_phase_delivery_v1.py':'621b9a3bf29811e3b11292397c1248b19046154a8924dee87cd2fa1ae10c0650',
    'code/round10_runner_v1.py':'22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96',
    'code/round10_value_bank_v2.py':'4c25f1bdd2174e07edac37f522e54005c0139e5e00103d42975bfaf3210de05c',
}

# This program runs on the remote host only to read/hash files and print proof.
# Raw parent state is captured once and returned as a snapshot, since later
# child completions may legally update it. No marker is treated as completion.
INSPECT = r'''
import hashlib,json,sys
from pathlib import Path,PurePosixPath
r=json.load(sys.stdin);root=Path(r['root']);job=r['job'];cell=r['cell'];name=job['name']
assert name.replace('_','').replace('-','').isalnum()
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
outdir=Path(cell['output'])
if not (outdir/'complete.json').is_file():
 print(json.dumps({'status':'PENDING_NO_COMPLETE'}));raise SystemExit(0)
launchpath=root/'jobs'/(name+'.launch.json');launch=json.loads(launchpath.read_text())
source=root/'jobs'/(name+'.source');argv=job['argv']
assert launch['name']==name and type(launch['pid']) is int and launch['pid']>0
assert launch['source_snapshot']==str(source) and launch['job_sha256']==r['job_sha256']
executed=[str(source/Path(a).relative_to(root)) if a.startswith(str(root)+'/code/') and a.endswith('.py') else a for a in argv]
assert launch['argv_executed']==executed and Path(executed[2])==source/'code/round10_sequence_v1.py'
seqpath=Path(argv[argv.index('--config')+1]);assert h(seqpath)==r['sequence_sha256']
seq=json.loads(seqpath.read_text());assert seq['schema']=='trajkv.round10.sequence.v1'
assert [x for x in seq['entries'] if x['name']==cell['name']]==[cell]
parentout=Path(argv[argv.index('--output')+1]);statepath=parentout/'state.json'
statebody=statepath.read_bytes();state=json.loads(statebody)
assert state['source_root']==str(source) and state['plan_sha256']==r['sequence_sha256']
closed=[x for x in state['completed_cells'] if x['name']==cell['name']]
failed=state.get('failed_cell',{})
if not closed and failed.get('name')!=cell['name']:
 print(json.dumps({'status':'PENDING_PARENT_CELL_COMMIT'}));raise SystemExit(0)
assert len(closed)<=1
manifest=json.loads((outdir/'manifest.json').read_text());done=json.loads((outdir/'complete.json').read_text())
cfgpath=Path(cell['config']);cfg=json.loads(cfgpath.read_text())
assert h(cfg['input_manifest_path'])==r['input_manifest_sha256']
assert h(cfgpath)==cell['config_sha256']==manifest['config_sha256'] and cfg==manifest['config']
assert manifest['schema']=='baseline_matrix.raw_inference.v1' and manifest['runner_schema']=='trajkv.round10_research.raw.v1'
assert manifest['runtime_source_root']==str(source) and manifest['source_hashes']==cfg['expected_source_hashes']
assert manifest['source_hashes'][cell['script']]==cell['script_sha256']
assert cell['script']=='code/round10_runner_v1.py' and cfg['method']['family']=='trajkv_round10_research'
assert h(outdir/'manifest.json')==done['manifest_sha256'] and h(outdir/'records.jsonl')==done['records_jsonl_sha256']
records=[json.loads(x) for x in (outdir/'records.jsonl').read_text().splitlines()]
assert len(records)==manifest['expected_QA']==done['expected_QA']==done['recorded_QA']==cell['expected_QA']
assert [x['id'] for x in records]==manifest['expected_ids'] and done['coverage_complete'] is True
assert done['counts']=={s:sum(x['status']==s for x in records) for s in set(x['status'] for x in records)}
if closed:
 assert closed[0]['exit_code']==0 and closed[0]['output']==str(outdir)
 assert closed[0]['recorded_QA']==cell['expected_QA'] and closed[0]['complete_sha256']==h(outdir/'complete.json')
 assert done['status']=='PASS_RAW_COMPLETE' and done['counts']=={'PASS':cell['expected_QA']}
inventory=source/'sha256.json';assert h(inventory)==launch['source_inventory_sha256'];iv=json.loads(inventory.read_text())
assert iv['code/round10_sequence_v1.py']==r['sequence_script_sha256']==h(source/'code/round10_sequence_v1.py')
start=[]
with Path(launch['log']).open('rb') as stream:
 for number,line in enumerate(stream,1):
  try:event=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if event.get('event')=='START_CELL' and event.get('name')==cell['name']:
   start.append({'event':event,'remote_log':launch['log'],'line_number':number,'exact_line_utf8':line.decode(),'exact_line_sha256':hashlib.sha256(line).hexdigest()})
assert len(start)==1 and type(start[0]['event']['pid']) is int and start[0]['event']['pid']>0
expected_child=[executed[0],'-B',str(source/cell['script']),'--config',cell['config'],'--output',cell['output']]
assert start[0]['event']['argv']==expected_child and start[0]['event']['config_sha256']==cell['config_sha256']
files=[]
def add(local,path,digest=None,maximum=8*1024**2):
 path=Path(path);assert path.is_file() and path.stat().st_size<=maximum
 actual=h(path)
 if digest is not None:assert actual==digest
 files.append({'file':local,'remote_file':str(path),'sha256':actual,'bytes':path.stat().st_size})
for n in ('manifest.json','records.jsonl','complete.json'):add(n,outdir/n,maximum=256*1024**2)
add('launch_receipt.json',launchpath);add('source_inventory.json',inventory)
add('source_parent_job.json',root/'jobs'/(name+'.json'),r['job_sha256'])
add('source_sequence_config.json',seqpath,r['sequence_sha256'])
add('source_parent_driver.py',source/'code/round10_sequence_v1.py',r['sequence_script_sha256'])
add('source_config.json',cfgpath,cell['config_sha256'])
for rel,digest in manifest['source_hashes'].items():
 p=PurePosixPath(rel);assert not p.is_absolute() and '..' not in p.parts and rel.startswith('code/') and rel.endswith('.py')
 assert iv[rel]==digest
 add('source_files/'+rel,source/rel,digest)
 if rel==cell['script']:add('source_driver.py',source/rel,digest)
for field in ('direction_book_path','metadata_factor_path','round7_policy_path'):
 add('assets/base_'+field+'.json',cfg['variant'][field],cfg['expected_asset_sha256'][field])
add('assets/original_profile.json',cfg['variant_source']['path'],cfg['variant_source']['sha256'])
for key in ('theta','window_selection'):
 if cfg['candidate'].get(key+'_path'):add('assets/candidate_'+key+'.json',cfg['candidate'][key+'_path'],cfg['candidate'][key+'_sha256'])
add('source_inputs.jsonl',cfg['inputs'],manifest['inputs_sha256'])
add('input_manifest.json',cfg['input_manifest_path'],cfg.get('input_manifest_sha256'))
roots=[('a2',Path(cfg['legacy_a2'])),('experiments',Path(cfg['legacy_experiments'])),('scripts',Path(cfg['legacy_scripts'])),('kivi_quant',Path(cfg['kivi_source']).parent)]
legacy=done.get('executed_legacy_source_sha256',{})
for legacy_file,digest in legacy.items():
 path=Path(legacy_file);role=None
 for label,parent in roots:
  try:role=label+'/'+path.relative_to(parent).as_posix();break
  except ValueError:pass
 if role is None:
  assert path==Path(cfg['legacy_a2']).parent/'bytes.py';role='code/bytes.py'
 add('legacy_sources/'+role,path,digest)
add('legacy_sources/kivi_quant/dynamic_kivi.py',cfg['kivi_source'])
assert len(files)==len({x['file'] for x in files})
print(json.dumps({'status':'PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA','files':files,
 'remote_hashes_computed_independently':True,'parent_producer':name,'cell_name':cell['name'],
 'actual_driver':cell['script'],'actual_child_start':start[0],
 'parent_state_snapshot':{'remote_file':str(statepath),'sha256':hashlib.sha256(statebody).hexdigest(),'body_utf8':statebody.decode()},
 'parent_cell_success':bool(closed),'SourceRun_admission_established':False,'legacy_count':len(legacy)}))
'''


def require(value,message):
    if not value:raise ValueError('ROUND10_HARVEST: '+message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def immutable(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==body,'frozen evidence differs: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(body)


def write(path,value):immutable(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


def check_pins():
    for name,digest in PINS.items():require(sha(ROOT/name)==digest,'frozen helper changed: '+name)


def load_plans(paths):
    """Exact declared full-domain cells only; no inferred or post-hoc methods."""
    result={};origins=[]
    for path in paths:
        plan=read(path)
        require(plan['schema']=='trajkv.round10.job_plan.v1' and plan['phase'] in ('references','candidates') and
                plan['status']=='FROZEN_PRE_LAUNCH' and plan['new_GPU_jobs_started_by_this_preparer'] is False,'only declared free-generation plans')
        origins.append({'path':str(path),'sha256':sha(path)})
        for producer in plan['jobs']:
            jobpath=ROOT/producer['job'];job=read(jobpath)
            require(sha(jobpath)==producer['job_sha256'] and job['name']==producer['name'],'parent job differs')
            seqremote=job['argv'][job['argv'].index('--config')+1]
            require('/configs/round10_v1/' in seqremote and job['argv'][2].endswith('/code/round10_sequence_v1.py'),'wrong parent producer')
            sequencepath=ROOT/'configs/round10_v1'/PurePosixPath(seqremote).name
            sequence=read(sequencepath);require(sequence['schema']=='trajkv.round10.sequence.v1','wrong sequence')
            for cell in sequence['entries']:
                require(cell in plan['cells'] and cell['script']=='code/round10_runner_v1.py' and
                        cell['script_sha256']==PINS[cell['script']],'undeclared or different child source')
                key=(cell['dataset'],cell['mode'])
                require(key not in result and cell['dataset'] in DOMAINS and cell['mode'] in MODES and
                        cell['expected_QA']==COUNTS[cell['dataset']],'duplicate/partial/wrong domain cell')
                result[key]={'cell':cell,'job':job,'job_sha256':producer['job_sha256'],
                    'sequence_sha256':sha(sequencepath),'sequence_script_sha256':sha(ROOT/'code/round10_sequence_v1.py'),
                    'input_manifest_sha256':plan['input_manifest_sha256']}
        require({x['name'] for x in plan['cells']}=={v['cell']['name'] for v in result.values() if v['cell'] in plan['cells']},'plan cell lacks a parent')
    return result,origins


def inspect_remote(client,request):
    from remote_control import REMOTE_ROOT
    stdin,stdout,stderr=client.exec_command(shlex.join(['/usr/bin/python3','-B','-c',INSPECT]),timeout=60)
    stdin.write(json.dumps({'root':REMOTE_ROOT,**request}));stdin.flush();stdin.channel.shutdown_write()
    body,error=stdout.read(),stderr.read();code=stdout.channel.recv_exit_status()
    require(code==0,'remote read-only inspection failed: '+error.decode(errors='replace')[-2500:])
    return json.loads(body)


def fetch_cell(client,item,destination):
    proof=inspect_remote(client,item)
    if proof['status'].startswith('PENDING_'):return proof
    cell=item['cell'];require(proof['cell_name']==cell['name'] and proof['parent_producer']==item['job']['name'],'producer label mismatch')
    sftp=client.open_sftp()
    try:
        for entry in proof['files']:
            p=PurePosixPath(entry['file']);require(not p.is_absolute() and '..' not in p.parts,'unsafe evidence path')
            target=Path(destination).joinpath(*p.parts)
            if target.exists():require(sha(target)==entry['sha256'] and target.stat().st_size==entry['bytes'],'existing evidence changed');continue
            with sftp.open(entry['remote_file'],'rb') as stream:body=stream.read()
            require(len(body)==entry['bytes'] and hashlib.sha256(body).hexdigest()==entry['sha256'],'remote readback bytes differ')
            immutable(target,body)
    finally:sftp.close()
    # Later parent states differ legitimately. Freeze the first exact snapshot;
    # immutable child/source files must remain exactly the same on every poll.
    receipt=Path(destination)/'source_retrieval_receipt_round10_v1.json'
    if receipt.exists():
        previous=read(receipt)
        stable=lambda p:{k:v for k,v in p.items() if k!='parent_state_snapshot'}
        require(stable(previous)==stable(proof),'immutable child/source proof changed')
    else:write(receipt,proof)
    anchored=[{k:e[k] for k in ('file','bytes','sha256')} for e in proof['files'] if e['file'] in ('manifest.json','records.jsonl','complete.json')]
    require(len(anchored)==3,'raw3 proof missing')
    retrieval=Path(destination)/'retrieval_receipt.json'
    if not retrieval.exists():
        complete=read(Path(destination)/'complete.json')
        write(retrieval,{'status':'PASS_REMOTE_RAW_SHA256_EXACT','remote_hashes_computed_independently':True,
            'files':anchored,'raw_status':complete['status'],'expected_QA':complete['expected_QA'],
            'observed_utc':datetime.now(timezone.utc).isoformat(),'relative_output_dir':'results/'+cell['output'].split('/results/',1)[1],
            'SourceRun_admission_established':False})
    else:require(read(retrieval)['files']==anchored,'raw receipt changed')
    return {'status':proof['status'],'cell_name':cell['name']}


def check_local_proof(raw,item):
    proof=read(raw/'source_retrieval_receipt_round10_v1.json');cell=item['cell']
    require(proof['status']=='PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA' and
            proof['remote_hashes_computed_independently'] is True and proof['SourceRun_admission_established'] is False and
            proof['cell_name']==cell['name'] and proof['parent_producer']==item['job']['name'],'wrong custom child proof')
    require(len(proof['files'])==len({e['file'] for e in proof['files']}),'duplicate proof entries')
    for e in proof['files']:
        p=PurePosixPath(e['file']);require(not p.is_absolute() and '..' not in p.parts,'unsafe local proof path')
        target=raw.joinpath(*p.parts);require(sha(target)==e['sha256'] and target.stat().st_size==e['bytes'],'local source/raw proof changed')
    cfg=read(raw/'source_config.json');manifest=read(raw/'manifest.json');closed=read(raw/'complete.json')
    require(sha(raw/'input_manifest.json')==item['input_manifest_sha256'],'declared input manifest differs from independent readback')
    require(sha(raw/'source_config.json')==cell['config_sha256']==manifest['config_sha256'] and cfg==manifest['config'],'config binding differs')
    require(sha(raw/'source_parent_job.json')==item['job_sha256'] and read(raw/'source_parent_job.json')==item['job'],'parent job binding differs')
    require(sha(raw/'source_sequence_config.json')==item['sequence_sha256'] and cell in read(raw/'source_sequence_config.json')['entries'],'sequence binding differs')
    require(sha(raw/'source_driver.py')==cell['script_sha256'] and
            sha(raw/'source_parent_driver.py')==item['sequence_script_sha256'],'actual source driver differs')
    launch=read(raw/'launch_receipt.json');job=item['job'];parent=job['name']
    remote=job['argv'][2].rsplit('/code/',1)[0];snapshot=remote+'/jobs/'+parent+'.source'
    expected=[snapshot+'/code/'+a.rsplit('/code/',1)[1] if a.startswith(remote+'/code/') and a.endswith('.py') else a for a in job['argv']]
    require(launch['name']==parent and type(launch['pid']) is int and launch['pid']>0 and
            launch['job_sha256']==item['job_sha256'] and launch['source_snapshot']==snapshot and
            launch['argv_executed']==expected and manifest['runtime_source_root']==snapshot,'actual parent invocation differs')
    require(sha(raw/'source_inventory.json')==launch['source_inventory_sha256'],'actual snapshot inventory differs')
    inventory=read(raw/'source_inventory.json')
    require(inventory['code/round10_sequence_v1.py']==item['sequence_script_sha256'] and
            all(inventory[k]==v==sha(raw/'source_files'/k) for k,v in manifest['source_hashes'].items()),'snapshot graph differs')
    event=proof['actual_child_start'];start=event['event']
    require(hashlib.sha256(event['exact_line_utf8'].encode()).hexdigest()==event['exact_line_sha256'] and
            json.loads(event['exact_line_utf8'])==start and start['name']==cell['name'] and
            start['config_sha256']==cell['config_sha256'] and type(start['pid']) is int and start['pid']>0 and
            start['argv']==[expected[0],'-B',snapshot+'/'+cell['script'],'--config',cell['config'],'--output',cell['output']],
            'actual child START_CELL evidence differs')
    snap=proof['parent_state_snapshot']
    require(hashlib.sha256(snap['body_utf8'].encode()).hexdigest()==snap['sha256'],'parent state snapshot bytes differ')
    parentstate=json.loads(snap['body_utf8'])
    require(parentstate['plan_sha256']==item['sequence_sha256'] and parentstate['source_root']==snapshot,'parent state scope differs')
    if proof['parent_cell_success']:
        result=[x for x in parentstate['completed_cells'] if x['name']==cell['name']]
        require(len(result)==1 and result[0]['exit_code']==0 and result[0]['complete_sha256']==sha(raw/'complete.json') and
                result[0]['recorded_QA']==cell['expected_QA'] and result[0]['output']==cell['output'],'parent completion does not bind child')
    require(manifest['source_hashes']==cfg['expected_source_hashes'],'declared/actual runtime graph differs')
    require(closed['expected_QA']==closed['recorded_QA']==manifest['expected_QA']==cell['expected_QA'],'wrong complete count')
    expected_mode='original' if cell['mode'] in ('R7low','R7high') else cell['mode']
    require(cfg['candidate']['mode']==expected_mode and cfg['input_role']=='development' and cfg['cohort_kind']=='development' and
            cfg['start']==0 and cfg['input_count']==cfg['count']==COUNTS[cell['dataset']] and
            cfg['candidate'].get('technical_only',False) is False,'partial/technical/mislabeled cell')
    require(cfg['variant_source']['name']==('r7_budget_k2_cond' if cell['mode']=='R7high' else 'r7_joint_k1_gt_cond'),'wrong base profile')
    return manifest,closed,cfg,proof


def stats(values):
    return {'mean':sum(values)/len(values),'median':statistics.median(values),'min':min(values),'max':max(values),'count':len(values)} if values else None


def context(manifest):
    keys=('torch_version','transformers_version','CUDA_version','python_version','hardware','runtime_executable',
          'processor_id','processor_version','processor_file_hashes','generation_native','generation_overrides',
          'weight_dtype','tf32','seed','attention_implementation','tokenizer_class','tokenizer_is_fast','tokenizers_version','decode_overrides')
    return {'model_spec':manifest['model_spec'],'model_files':[{k:x[k] for k in ('file','bytes','sha256')} for x in manifest['model_identity']['files']],
            **{k:manifest[k] for k in keys}}


def identity(row):
    names=('dataset','dataset_revision','split','question_id','source_cluster_id','source_id','original_image_sha256',
           'question_sha256','raw_question_sha256','full_prompt_hash','input_ids_sha256','question_suffix_sha256',
           'native_position_ids_sha256','native_position_ids_shape','visual_positions','actual_visual_token_count',
           'input_tokens','compression_boundary','question_suffix_tokens')
    return {k:row[k] for k in names}


def closed_cell(raw,scored,item,refs_root):
    cell=item['cell'];n=cell['expected_QA'];missing=[x for x in ('manifest.json','records.jsonl','complete.json','source_retrieval_receipt_round10_v1.json','retrieval_receipt.json') if not(raw/x).is_file()]
    empty={'dataset':cell['dataset'],'mode':cell['mode'],'name':cell['name'],'status':'PENDING_REAL_WHOLE_DOMAIN_SOURCE_SCORE',
           'expected_QA':n,'primary_mean':None,'ci95':None,'resources':None,'timing':None,'missing':missing}
    if missing:return empty,None
    m,c,cfg,proof=check_local_proof(raw,item)
    refs=Path(refs_root)/PurePosixPath(cfg['inputs']).name.replace('_inputs.jsonl','_refs.jsonl')
    require(sha(refs.with_name(refs.name.replace('_refs.jsonl','_inputs.jsonl')))==m['inputs_sha256']==sha(raw/'source_inputs.jsonl'),'local references input mirror differs')
    import score
    if not(scored/'summary.json').is_file():score.score_run(raw,refs,scored)
    s=read(scored/'summary.json')
    require(s['integrity']['scorer_sha256']==PINS['code/score.py'] and s['expected_run_questions']==n and
            s['integrity']['raw_files']=={k:sha(raw/k) for k in ('manifest.json','records.jsonl','complete.json')},'score/raw/scorer binding differs')
    require(s['integrity']['refs_sha256']==sha(refs) and s['integrity']['dataset_source_lock_sha256']==sha(Path(refs_root)/'source_lock.json'),'frozen reference source lock changed')
    for name,digest in s['derived_file_hashes'].items():require(sha(scored/name)==digest,'scored rows changed')
    if c['status']!='PASS_RAW_COMPLETE' or s['status']!='PASS_SCORED_COMPLETE':
        require(s['primary_mean'] is None,'failure must not create a primary success-subset score')
        empty.update(status='CLOSED_WITH_FAILURES_NO_PRIMARY',missing=['all_expected_QA_PASS'],raw_status=c['status'],score_status=s['status']);return empty,None
    from prepare_remaining_phase_delivery_v1 import physical_complete_bytes
    from round10_runner_v1 import round10_budget_guard
    rawrows=[];bits=[];capacity=[];scorerows={r['id']:r for r in map(json.loads,(scored/'scored_records.jsonl').read_text(encoding='utf-8').splitlines())}
    for number,line in enumerate((raw/'records.jsonl').open('rb'),1):
        r=json.loads(line);sr=scorerows[r['id']]
        require(r['status']=='PASS' and sr['score_status']=='SCORED_PASS' and sr['raw_jsonl_line_number']==number and
                sr['raw_jsonl_exact_line_sha256']==hashlib.sha256(line).hexdigest(),'exact raw/score line differs')
        actual=physical_complete_bytes(r,m['model_spec']);capacity.append(actual);bits.append(r['actual_effective_visual_bits'])
        require(math.isclose(bits[-1],16*actual/r['full_kv_byte_denominator'],rel_tol=1e-12,abs_tol=1e-12),'actual bits denominator differs')
        round10_budget_guard(cfg['candidate']['mode'],r['actual_visual_token_count'],actual,r['r7_bridge'],cfg['variant_source']['name'])
        if cell['mode'] not in ('FP16','R7low','R7high'):
            bridge=r['r7_bridge'];stages=bridge['selected_stages'];candidate=bridge['round10_value_candidate']
            require(bridge['selected_stage_count']==len(stages)==43 and sum(x[0]=='k' for x in stages)==23 and
                    sum(x[0]=='v' for x in stages)==20 and candidate['mode']==cfg['candidate']['mode'] and
                    candidate['source_sha256']==PINS['code/round10_value_bank_v2.py'] and
                    candidate['native_FP16_source_or_feedback_workspace_retained'] is False,'candidate coverage/source/owner metadata differs')
        require(r['owner_after_generation']['stored_visual_KV_bytes']==actual and r['original_full_prefix_released'] is True and
                r['original_FP16_visual_alias_retained'] is False,'native prefix retained or persistent owner changed')
        rawrows.append(r)
    require(len(rawrows)==n and s['metric_ids']==[cfg['evaluation_contract']['primary_metric_id']],'wrong full-domain score or metric')
    timing={key:stats([r[key] for r in rawrows if r.get(key) is not None]) for key in
        ('prefill_ms','encode_ms','question_prefill_ms','ttft_ms','wall_ms','output_tokens','peak_allocated_bytes','peak_reserved_bytes')}
    timing['pooled_natural_inter_token_ms']=stats([v for r in rawrows for v in r['inter_token_ms']])
    point={**empty,'status':'CLOSED_FULL_DOMAIN_CUSTOM_SOURCE_SCORE_PHYSICAL','missing':[],'primary_mean':s['primary_mean'],
        'ci95':s['primary_ci95'],'resources':{'complete_bytes':stats(capacity),'actual_complete_bits':stats(bits),
        'strict_all_QA_1p3':all(160*b<=13*r['full_kv_byte_denominator'] for b,r in zip(capacity,rawrows)),
        'strict_all_QA_1p8':all(80*b<=9*r['full_kv_byte_denominator'] for b,r in zip(capacity,rawrows))},
        'timing':timing,'raw':str(raw),'scored':str(scored),'summary_sha256':sha(scored/'summary.json'),
        'source_receipt_sha256':sha(raw/'source_retrieval_receipt_round10_v1.json'),'SourceRun_admission_established':False}
    bundle={'context':context(m),'identity':{r['id']:identity(r) for r in rawrows},'scores':scorerows,'scored':scored}
    return point,bundle


def paired(anchor,candidate):
    require(anchor['context']==candidate['context'] and anchor['identity']==candidate['identity'],'pair native runtime/input/position context differs')
    from compare_runs import compare
    result=compare(Path(anchor['scored']),Path(candidate['scored']))
    require(result['status']=='PASS_PAIRED_COMPLETE','whole-domain paired comparison failed')
    changes=[candidate['scores'][key]['score']-left['score'] for key,left in anchor['scores'].items()]
    n=len(changes)
    result['gross_score_change']={'improved_questions':sum(v>0 for v in changes),'degraded_questions':sum(v<0 for v in changes),
        'equal_questions':sum(v==0 for v in changes),'gross_gain_pp':100*sum(max(0,v) for v in changes)/n,
        'gross_loss_pp':100*sum(max(0,-v) for v in changes)/n,'net_pp':100*sum(changes)/n,
        'continuous_scoring_not_manual_semantic_error_judgment':True,'denominator':n}
    return result


def summarize(items,origins,refs_root,output_root,destination):
    require(not Path(destination).exists(),'new summary directory required')
    points=[];bundles={}
    for ds in DOMAINS:
        for mode in MODES:
            key=(ds,mode)
            if key not in items:
                points.append({'dataset':ds,'mode':mode,'status':'PENDING_DECLARED_PLAN_NOT_READY','expected_QA':COUNTS[ds],
                    'primary_mean':None,'ci95':None,'resources':None,'timing':None});continue
            item=items[key];name=item['cell']['name']
            point,bundle=closed_cell(Path(output_root)/'raw'/name,Path(output_root)/'scored'/name,item,refs_root)
            points.append(point)
            if bundle is not None:bundles[key]=bundle
    pairs={};contrasts=[(c,a) for c in CANDIDATES for a in ('FP16','R7low','R7high')]+[('LS-RE','LS-FZ'),('FB-W','FB0')]
    for ds in DOMAINS:
        for candidate,anchor in contrasts:
            key=ds+'__'+candidate+'_minus_'+anchor
            pairs[key]=paired(bundles[ds,anchor],bundles[ds,candidate]) if (ds,anchor) in bundles and (ds,candidate) in bundles else None
    closed=sum(x['primary_mean'] is not None for x in points)
    result={'schema':'trajkv.round10.development_summary.v1','status':'PASS_ALL28_D160_CUSTOM_SOURCE_SCORE_PHYSICAL' if closed==28 else 'PENDING_OR_PARTIAL_D160_NOT_COMPLETE',
        'expected_cells':28,'closed_cells':closed,'expected_answer_events':1120,'closed_answer_events':sum(x['expected_QA'] for x in points if x['primary_mean'] is not None),
        'expected_distinct_development_sources':160,
        'actual_closed_unique_IDs':len({key for bundle in bundles.values() for key in bundle['identity']}),
        'plan_sources':origins,'points':points,'paired_contrasts':list(pairs),
        'no_cross_dataset_quality_average':True,'D160_is_development_not_independent_confirmation':True,
        'paired_CI_and_signflip_unadjusted_not_equivalence_or_algorithm_winner':True,
        'quality_resource_and_timing_are_separate':'All-QA complete bytes; instrumented quality-run host timing with natural output lengths, not controlled speed or HBM traffic.',
        'SourceRun_admission_established':False,'new_GPU_jobs_started':0,'source_sha256':sha(__file__),'frozen_helper_sha256':PINS}
    Path(destination).mkdir(parents=True)
    write(Path(destination)/'summary.json',result);write(Path(destination)/'paired_comparisons.json',pairs)
    table=[]
    for p in points:
        r=p['resources'] or {};capacity=r.get('complete_bytes') or {};b=r.get('actual_complete_bits') or {}
        table.append({'dataset':p['dataset'],'mode':p['mode'],'status':p['status'],'QA':p['expected_QA'],
            'quality_percent':None if p['primary_mean'] is None else 100*p['primary_mean'],
            'CI95_percent':None if p['ci95'] is None else [100*x for x in p['ci95']],
            'mean_complete_bytes':capacity.get('mean'),'max_complete_bytes':capacity.get('max'),
            'mean_actual_complete_bits':b.get('mean'),'max_actual_complete_bits':b.get('max'),
            'strict_all_QA_1p3':r.get('strict_all_QA_1p3'),'strict_all_QA_1p8':r.get('strict_all_QA_1p8')})
    with (Path(destination)/'quality_complete_bytes.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    write(Path(destination)/'instrumented_timing_separate.json',{
        'scope':'Prototype quality-run host timings, natural token lengths, diagnostic synchronization included.',
        'controlled_speed_or_HBM_traffic_benchmark':False,'cross_dataset_speed_ranking_claimed':False,
        'rows':{p['dataset']+'__'+p['mode']:p['timing'] for p in points}})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plans',nargs='+',type=Path,required=True)
    p.add_argument('--refs-root',type=Path,default=Path('${EVIDENCE_ROOT}/results/round10/input_preparation_v1/inputs'))
    p.add_argument('--output-root',type=Path,default=Path('${EVIDENCE_ROOT}/results/round10/harvest_v1'))
    p.add_argument('--summary-output',type=Path,required=True);p.add_argument('--fetch',action='store_true');a=p.parse_args()
    require(a.output_root.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve()) and
            a.summary_output.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve()),'heavy outputs must use E')
    check_pins();items,origins=load_plans(a.plans)
    if a.fetch:
        from remote_control import connect
        client=connect()
        try:
            for item in items.values():
                result=fetch_cell(client,item,a.output_root/'raw'/item['cell']['name'])
                print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:client.close()
    result=summarize(items,origins,a.refs_root,a.output_root,a.summary_output)
    print(json.dumps({'status':result['status'],'closed_cells':result['closed_cells'],'expected_cells':28}),flush=True)


if __name__=='__main__':main()
