"""Round11 v4 proof: bind recorded child parent_PID to launcher parent PID.

All v3 scoring contrasts and all inference/source plans remain unchanged.

V2 contrasts remain available; the new specificity contrast does not mix
spatial coding with JointV. Inference, plans, sources and raw data are unchanged.

Round11 bounded W_O child/source proof, D96 reference reuse and CPU scoring.

Fresh records are accepted only with actual parent snapshot/PID/START_CELL/
exit0 closure. Reused records retain their original Round10 producer proof
and exact source lines; a D96 view never becomes a new inference run. Remote
access is opt-in --fetch and reads only. All writes are immutable local outputs.
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
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'code')]
import round11_wo_runner_v1 as runner

DOMAINS = tuple(runner.COUNTS)
COUNTS = runner.COUNTS
CANDIDATES = runner.MODES
REFERENCES = ('FP16', 'R7low', 'R7high')
MODES = REFERENCES + CANDIDATES
PROOF_NAME = 'source_retrieval_receipt_round11_wo_v1.json'
PINS = {
    'ops/round10_harvest_score_v1.py': '1401b56ff23d28d5a88f9fd81514ee4633f5c28bdb90b16dc9627e1cc239f789',
    'ops/prepare_remaining_phase_delivery_v1.py': '621b9a3bf29811e3b11292397c1248b19046154a8924dee87cd2fa1ae10c0650',
    'code/round10_sequence_v1.py': '2d0dce9da80156fb0224a04d48a9a3e3869ede5e9885885a6a25073ab511fe00',
    'code/score.py': '72b5182eeb97e930403a970feebb4f1fadac23d04544c9646c6c9fd6b34bfba6',
    'code/metrics.py': '98b260c766e095f2b629d603de5cfd6519a91dccaaee471f02486955bb590d47',
}

# Independent Round11 inspector, not string replacement inside an older
# inspector. It binds the declared frozen sequence parent and new child schema.
INSPECT = r'''
import hashlib,json,sys
from pathlib import Path,PurePosixPath
def below(path,parent):
 try:path.relative_to(parent);return True
 except ValueError:return False
r=json.load(sys.stdin);root=Path(r['root']);job=r['job'];cell=r['cell'];name=job['name']
assert name.replace('_','').replace('-','').isalnum()
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=Path(cell['output'])
if not(out/'complete.json').is_file():
 print(json.dumps({'status':'PENDING_NO_COMPLETE'}));raise SystemExit(0)
assert below(out,root/'results/round11_v1')
launchpath=root/'jobs'/(name+'.launch.json');launch=json.loads(launchpath.read_text())
source=root/'jobs'/(name+'.source');argv=job['argv']
assert launch['name']==name and type(launch['pid']) is int and launch['pid']>0
assert launch['source_snapshot']==str(source) and launch['job_sha256']==r['job_sha256']
executed=[str(source/Path(a).relative_to(root)) if a.startswith(str(root)+'/code/') and a.endswith('.py') else a for a in argv]
assert launch['argv_executed']==executed and Path(executed[2])==source/r['parent_script']
seqpath=Path(argv[argv.index('--config')+1]);assert h(seqpath)==r['sequence_sha256']
seq=json.loads(seqpath.read_text());assert seq['schema']==r['sequence_schema']
assert [x for x in seq['entries'] if x['name']==cell['name']]==[cell]
parentout=Path(argv[argv.index('--output')+1]);statepath=parentout/'state.json'
body=statepath.read_bytes();state=json.loads(body)
assert state['source_root']==str(source) and state['plan_sha256']==r['sequence_sha256']
closed=[x for x in state['completed_cells'] if x['name']==cell['name']]
failed=state.get('failed_cell',{})
if not closed and failed.get('name')!=cell['name']:
 print(json.dumps({'status':'PENDING_PARENT_EXIT0_COMMIT'}));raise SystemExit(0)
assert len(closed)<=1
if closed:assert closed[0]['exit_code']==0 and closed[0]['output']==str(out)
else:assert failed['exit_code']!=0 and failed['output']==str(out)
m=json.loads((out/'manifest.json').read_text());done=json.loads((out/'complete.json').read_text())
cfgpath=Path(cell['config']);cfg=json.loads(cfgpath.read_text())
assert below(cfgpath,root/'configs/round11_v1')
assert h(cfgpath)==cell['config_sha256']==m['config_sha256'] and cfg==m['config']
assert m['schema']=='baseline_matrix.raw_inference.v1' and m['runner_schema']=='trajkv.round11_wo_research.raw.v1'
assert m['runtime_source_root']==str(source) and m['source_hashes']==cfg['expected_source_hashes']
assert m['source_hashes'][cell['script']]==cell['script_sha256']
assert cell['script']=='code/round11_wo_runner_v1.py' and cfg['method']['family']=='trajkv_round11_wo_research'
assert cfg['candidate']['mode']==cell['mode'] and cell['mode'] in ('SpatialWO','JointSpatialWO')
assert h(cfg['input_manifest_path'])==r['input_manifest_sha256']==cfg['input_manifest_sha256']
assert h(cfg['source_group_manifest_path'])==r['source_group_manifest_sha256']==cfg['source_group_manifest_sha256']
assert h(out/'manifest.json')==done['manifest_sha256'] and h(out/'records.jsonl')==done['records_jsonl_sha256']
records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
assert len(records)==m['expected_QA']==done['expected_QA']==done['recorded_QA']==cell['expected_QA']
assert [x['id'] for x in records]==m['expected_ids'] and done['coverage_complete'] is True
assert done['counts']=={status:sum(row['status']==status for row in records) for status in set(row['status'] for row in records)}
if closed:
 assert done['status']=='PASS_RAW_COMPLETE' and done['counts']=={'PASS':cell['expected_QA']}
 assert closed[0]['recorded_QA']==cell['expected_QA'] and closed[0]['complete_sha256']==h(out/'complete.json')
else:assert done['status']=='COMPLETE_WITH_FAILURES'
inventory=source/'sha256.json';assert h(inventory)==launch['source_inventory_sha256'];iv=json.loads(inventory.read_text())
assert iv[r['parent_script']]==r['sequence_script_sha256']==h(source/r['parent_script'])
starts=[]
with Path(launch['log']).open('rb') as stream:
 for number,line in enumerate(stream,1):
  try:event=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if event.get('event')=='START_CELL' and event.get('name')==cell['name']:
   starts.append({'event':event,'remote_log':launch['log'],'line_number':number,'exact_line_utf8':line.decode(),'exact_line_sha256':hashlib.sha256(line).hexdigest()})
assert len(starts)==1 and type(starts[0]['event']['pid']) is int and starts[0]['event']['pid']>0
assert starts[0]['event']['argv']==[executed[0],'-B',str(source/cell['script']),'--config',cell['config'],'--output',cell['output']]
assert starts[0]['event']['config_sha256']==cell['config_sha256']
assert m['runtime_process']['PID']==starts[0]['event']['pid']
assert m['runtime_process']['parent_PID']==launch['pid']
assert m['runtime_process']['source_snapshot_root']==str(source)
files=[]
def add(local,path,digest=None,maximum=256*1024**2):
 path=Path(path);assert path.is_file() and path.stat().st_size<=maximum
 actual=h(path)
 if digest is not None:assert actual==digest
 files.append({'file':local,'remote_file':str(path),'sha256':actual,'bytes':path.stat().st_size})
for n in ('manifest.json','records.jsonl','complete.json'):add(n,out/n)
add('launch_receipt.json',launchpath);add('source_inventory.json',inventory)
add('source_parent_job.json',root/'jobs'/(name+'.json'),r['job_sha256'])
add('source_sequence_config.json',seqpath,r['sequence_sha256'])
add('source_parent_driver.py',source/r['parent_script'],r['sequence_script_sha256'])
add('source_config.json',cfgpath,cell['config_sha256'])
for rel,digest in m['source_hashes'].items():
 p=PurePosixPath(rel);assert not p.is_absolute() and '..' not in p.parts and rel.startswith('code/') and rel.endswith('.py')
 assert iv[rel]==digest
 add('source_files/'+rel,source/rel,digest)
 if rel==cell['script']:add('source_driver.py',source/rel,digest)
for field in ('direction_book_path','metadata_factor_path','round7_policy_path'):
 add('assets/base_'+field+'.json',cfg['variant'][field],cfg['expected_asset_sha256'][field])
add('assets/original_profile.json',cfg['variant_source']['path'],cfg['variant_source']['sha256'])
add('source_inputs.jsonl',cfg['inputs'],m['inputs_sha256'])
add('input_manifest.json',cfg['input_manifest_path'],r['input_manifest_sha256'])
add('source_group_manifest.json',cfg['source_group_manifest_path'],r['source_group_manifest_sha256'])
roots=[('a2',Path(cfg['legacy_a2'])),('experiments',Path(cfg['legacy_experiments'])),('scripts',Path(cfg['legacy_scripts'])),('kivi_quant',Path(cfg['kivi_source']).parent)]
legacy=done.get('executed_legacy_source_sha256',{})
for filename,digest in legacy.items():
 path=Path(filename);role=None
 for label,parent in roots:
  try:role=label+'/'+path.relative_to(parent).as_posix();break
  except ValueError:pass
 if role is None:
  assert path==Path(cfg['legacy_a2']).parent/'bytes.py';role='code/bytes.py'
 add('legacy_sources/'+role,path,digest)
add('legacy_sources/kivi_quant/dynamic_kivi.py',cfg['kivi_source'])
assert len(files)==len({entry['file'] for entry in files})
print(json.dumps({'status':'PASS_ROUND11_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA','files':files,
 'remote_hashes_computed_independently':True,'parent_producer':name,'cell_name':cell['name'],
 'actual_driver':cell['script'],'actual_child_start':starts[0],
 'parent_state_snapshot':{'remote_file':str(statepath),'sha256':hashlib.sha256(body).hexdigest(),'body_utf8':body.decode()},
 'parent_cell_success':bool(closed),'SourceRun_admission_established':False,'legacy_count':len(legacy)}))
'''


def require(value, message):
    if not value:
        raise ValueError('ROUND11_HARVEST: ' + message)


def sha(path):
    return runner.sha(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def immutable(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == body, 'immutable artifact changed: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(body)


def write(path, value):
    immutable(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def check_pins():
    require(all(sha(ROOT / role) == digest for role, digest in PINS.items()), 'frozen offline/proof dependency changed')
    gate = read(ROOT / 'audit/metrics_gate_v2.json')
    require(all(sha(ROOT / role) == digest for role, digest in gate['code_hashes'].items()), 'frozen metric gate changed')
    return {**PINS, **runner.runtime_source_manifest(), 'ops/round11_wo_harvest_v1.py': sha(__file__)}


def load_plan(path):
    plan = read(path)
    require(plan.get('schema') == 'trajkv.round11.harvest_plan.v1'
            and plan.get('branch') == 'round11_bounded_WO_v1',
            'explicit shared D96 plan schema and bounded-WO branch required')
    group_path = Path(plan['source_group_manifest_path'])
    require(sha(group_path) == plan['source_group_manifest_sha256'], 'frozen source-group manifest differs')
    group = read(group_path)
    require(group['schema'] == 'trajkv.round11.source_group.v1'
            and group['reference_input_manifest_sha256'] == plan['input_manifest_sha256'], 'source-group lineage differs')
    require(set(plan['domains']) == set(DOMAINS), 'all four source-group domains required')
    for dataset in DOMAINS:
        domain = plan['domains'][dataset]
        for key in ('inputs', 'refs', 'source_lock'):
            require(sha(domain[key]) == domain[key + '_sha256'], 'locked offline ' + key + ' changed')
        require(domain['inputs_sha256'] == group['datasets'][dataset]['input_sha256'], 'offline source-group input differs')
        inputs = [json.loads(line) for line in Path(domain['inputs']).read_text(encoding='utf-8-sig').splitlines()]
        require([row['id'] for row in inputs[:COUNTS[dataset]]] == group['datasets'][dataset]['ids'], 'offline source-group order differs')
        lock = read(domain['source_lock'])
        for key in ('inputs', 'refs'):
            target = Path(domain[key]); entry = lock['files'][target.name]
            require(entry['sha256'] == sha(target) and entry['bytes'] == target.stat().st_size, 'frozen dataset source lock differs')
    entries = plan['entries']
    require(len(entries) == len({(entry['dataset'], entry['mode']) for entry in entries}), 'duplicate dataset/mode views')
    require(all(entry['dataset'] in DOMAINS and entry['mode'] in MODES for entry in entries), 'unknown view')
    for entry in entries:
        require(entry['kind'] == ('fresh' if entry['mode'] in CANDIDATES else 'reused'), 'fresh/reused accounting differs')
        if entry['kind'] == 'fresh':
            item = entry['item']; cell = item['cell']
            require(item.get('parent_script') == 'code/round10_sequence_v1.py'
                    and item.get('sequence_schema') == 'trajkv.round10.sequence.v1'
                    and item['sequence_script_sha256'] == PINS['code/round10_sequence_v1.py'], 'frozen parent identity differs')
            require(cell['mode'] == entry['mode'] and cell['dataset'] == entry['dataset']
                    and cell['script'] == 'code/round11_wo_runner_v1.py'
                    and cell['script_sha256'] == sha(ROOT / cell['script'])
                    and cell['expected_QA'] == COUNTS[entry['dataset']], 'actual new child declaration differs')
            require(item['input_manifest_sha256'] == plan['input_manifest_sha256']
                    and item['source_group_manifest_sha256'] == plan['source_group_manifest_sha256'], 'child source-group binding differs')
        else:
            require(isinstance(entry.get('source_receipt_sha256'), str) and len(entry['source_receipt_sha256']) == 64,
                    'original reference producer receipt must be explicitly pinned')
    return plan, group


def inspect_remote(client, item):
    stdin, stdout, stderr = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', INSPECT]), timeout=60)
    stdin.write(json.dumps(item)); stdin.flush(); stdin.channel.shutdown_write()
    body, error = stdout.read(), stderr.read()
    require(stdout.channel.recv_exit_status() == 0, 'read-only remote source inspection failed: ' + error.decode(errors='replace')[-2000:])
    return json.loads(body)


def fetch_cell(client, item, destination):
    proof = inspect_remote(client, item)
    if proof['status'].startswith('PENDING_'):
        return proof
    sftp = client.open_sftp()
    try:
        for entry in proof['files']:
            relative = PurePosixPath(entry['file'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe local evidence path')
            target = Path(destination).joinpath(*relative.parts)
            if target.exists():
                require(sha(target) == entry['sha256'] and target.stat().st_size == entry['bytes'], 'existing retrieved bytes changed')
                continue
            with sftp.open(entry['remote_file'], 'rb') as stream:
                body = stream.read()
            require(len(body) == entry['bytes'] and hashlib.sha256(body).hexdigest() == entry['sha256'], 'remote readback differs')
            immutable(target, body)
    finally:
        sftp.close()
    receipt = Path(destination) / PROOF_NAME
    if receipt.exists():
        stable = lambda value: {key: item for key, item in value.items() if key != 'parent_state_snapshot'}
        require(stable(read(receipt)) == stable(proof), 'immutable child/source proof changed')
    else:
        write(receipt, proof)
    anchored = [{key: entry[key] for key in ('file', 'bytes', 'sha256')}
                for entry in proof['files'] if entry['file'] in ('manifest.json', 'records.jsonl', 'complete.json')]
    require(len(anchored) == 3, 'exact raw3 retrieval anchors required')
    receipt_path = Path(destination) / 'retrieval_receipt.json'
    if receipt_path.exists():
        require(read(receipt_path)['files'] == anchored, 'raw retrieval anchors changed')
    else:
        write(receipt_path, {'status': 'PASS_REMOTE_RAW_SHA256_EXACT', 'remote_hashes_computed_independently': True,
            'files': anchored, 'observed_utc': datetime.now(timezone.utc).isoformat(),
            'SourceRun_admission_established': False})
    return {'status': proof['status'], 'cell_name': item['cell']['name']}


def check_fresh_proof(raw, item):
    proof = read(raw / PROOF_NAME); cell = item['cell']
    require(proof['status'] == 'PASS_ROUND11_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA'
            and proof['remote_hashes_computed_independently'] is True and type(proof['parent_cell_success']) is bool
            and proof['SourceRun_admission_established'] is False
            and proof['cell_name'] == cell['name'] and proof['parent_producer'] == item['job']['name'], 'wrong independent child proof')
    require(len(proof['files']) == len({entry['file'] for entry in proof['files']}), 'duplicate source-proof files')
    for entry in proof['files']:
        relative = PurePosixPath(entry['file'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe proof path')
        target = raw.joinpath(*relative.parts)
        require(sha(target) == entry['sha256'] and target.stat().st_size == entry['bytes'], 'source/raw evidence bytes changed')
    manifest, done, cfg = read(raw / 'manifest.json'), read(raw / 'complete.json'), read(raw / 'source_config.json')
    require(sha(raw / 'source_config.json') == cell['config_sha256'] == manifest['config_sha256']
            and cfg == manifest['config'], 'actual child config binding differs')
    require(manifest['schema'] == 'baseline_matrix.raw_inference.v1' and manifest['runner_schema'] == runner.SCHEMA
            and cfg['method']['family'] == 'trajkv_round11_wo_research' and cfg['candidate']['mode'] == cell['mode'], 'new raw schema/mode differs')
    require(sha(raw / 'source_parent_job.json') == item['job_sha256']
            and read(raw / 'source_parent_job.json') == item['job'], 'parent actual job binding differs')
    require(sha(raw / 'source_sequence_config.json') == item['sequence_sha256']
            and cell in read(raw / 'source_sequence_config.json')['entries'], 'parent sequence binding differs')
    require(sha(raw / 'source_parent_driver.py') == item['sequence_script_sha256']
            and sha(raw / 'source_driver.py') == cell['script_sha256'], 'executed parent/child driver identity differs')
    launch = read(raw / 'launch_receipt.json'); job = item['job']; remote = item['root'].rstrip('/')
    snapshot = remote + '/jobs/' + job['name'] + '.source'
    expected = [snapshot + '/code/' + arg.rsplit('/code/', 1)[1]
                if arg.startswith(remote + '/code/') and arg.endswith('.py') else arg for arg in job['argv']]
    require(launch['name'] == job['name'] and type(launch['pid']) is int and launch['pid'] > 0
            and launch['job_sha256'] == item['job_sha256'] and launch['source_snapshot'] == snapshot
            and launch['argv_executed'] == expected and manifest['runtime_source_root'] == snapshot, 'actual parent PID/invocation differs')
    require(sha(raw / 'source_inventory.json') == launch['source_inventory_sha256'], 'snapshot inventory bytes differ')
    inventory = read(raw / 'source_inventory.json')
    require(inventory[item['parent_script']] == item['sequence_script_sha256'], 'parent inventory pin differs')
    require(manifest['source_hashes'] == cfg['expected_source_hashes'] == runner.runtime_source_manifest()
            and all(inventory[role] == digest == sha(raw / 'source_files' / role)
                    for role, digest in manifest['source_hashes'].items()), 'complete executed source graph differs')
    event = proof['actual_child_start']; start = event['event']
    require(hashlib.sha256(event['exact_line_utf8'].encode()).hexdigest() == event['exact_line_sha256']
            and json.loads(event['exact_line_utf8']) == start and start['name'] == cell['name']
            and start['config_sha256'] == cell['config_sha256'] and type(start['pid']) is int and start['pid'] > 0
            and start['argv'] == [expected[0], '-B', snapshot + '/' + cell['script'], '--config', cell['config'], '--output', cell['output']],
            'actual child PID/START_CELL bytes differ')
    require(manifest['runtime_process']['PID'] == start['pid']
            and type(manifest['runtime_process']['parent_PID']) is int
            and manifest['runtime_process']['parent_PID'] == launch['pid']
            and manifest['runtime_process']['source_snapshot_root'] == snapshot,
            'runtime child PID/snapshot does not bind actual START_CELL')
    snap = proof['parent_state_snapshot']
    require(hashlib.sha256(snap['body_utf8'].encode()).hexdigest() == snap['sha256'], 'parent state snapshot bytes differ')
    parent = json.loads(snap['body_utf8'])
    require(parent['source_root'] == snapshot and parent['plan_sha256'] == item['sequence_sha256'], 'parent state scope differs')
    matches = [row for row in parent['completed_cells'] if row['name'] == cell['name']]
    if proof['parent_cell_success']:
        require(len(matches) == 1 and matches[0]['exit_code'] == 0
                and matches[0]['complete_sha256'] == sha(raw / 'complete.json')
                and matches[0]['recorded_QA'] == cell['expected_QA'] and matches[0]['output'] == cell['output']
                and done['status'] == 'PASS_RAW_COMPLETE', 'parent exit0 does not bind complete child')
    else:
        failed = parent.get('failed_cell', {})
        require(not matches and failed.get('name') == cell['name'] and failed['exit_code'] != 0
                and failed['output'] == cell['output'] and done['status'] == 'COMPLETE_WITH_FAILURES',
                'parent failed-cell evidence does not bind preserved failed raw')
    require(sha(raw / 'input_manifest.json') == cfg['input_manifest_sha256'] == item['input_manifest_sha256']
            and sha(raw / 'source_group_manifest.json') == cfg['source_group_manifest_sha256'] == item['source_group_manifest_sha256'],
            'original/source-group input manifest differs')
    require(cfg['start'] == 0 and cfg['count'] == cell['expected_QA'] == COUNTS[cell['dataset']]
            and cfg['input_role'] == cfg['cohort_kind'] == 'development'
            and cfg['candidate'].get('technical_only', False) is False
            and cfg['variant_source']['name'] == 'r7_joint_k1_gt_cond', 'partial/technical/wrong-lineage child')
    return manifest, done, cfg, proof


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'min': min(values), 'max': max(values)} if values else None


def selected_view(entry, plan, group, output_root):
    """Return a strictly proven, exact-line D96 subset, without changing raw3."""
    dataset, mode = entry['dataset'], entry['mode']
    if entry['kind'] == 'fresh':
        raw = Path(output_root) / 'raw' / entry['item']['cell']['name']
        if not (raw / PROOF_NAME).is_file():
            return None
        manifest, done, cfg, proof = check_fresh_proof(raw, entry['item'])
        proof_path = raw / PROOF_NAME
    else:
        import round10_harvest_score_v1 as old
        raw = Path(entry['source_raw_dir'])
        proof_path = raw / 'source_retrieval_receipt_round10_v1.json'
        require(sha(proof_path) == entry['source_receipt_sha256'], 'reused original producer proof changed')
        manifest, done, cfg, proof = old.check_local_proof(raw, entry['source_item'])
        require(entry['source_item']['cell']['script'] == 'code/round10_runner_v1.py'
                and manifest['runner_schema'] == 'trajkv.round10_research.raw.v1'
                and cfg['method']['family'] == 'trajkv_round10_research'
                and proof['parent_cell_success'] is True, 'reused reference must retain original Round10 production proof')
        require(cfg['candidate']['mode'] == ('FP16' if mode == 'FP16' else 'original')
                and cfg['variant_source']['name'] == ('r7_budget_k2_cond' if mode == 'R7high' else 'r7_joint_k1_gt_cond'),
                'reference mode/lineage differs')
    require(done['status'] in ('PASS_RAW_COMPLETE', 'COMPLETE_WITH_FAILURES') and done['coverage_complete'] is True
            and done['manifest_sha256'] == sha(raw / 'manifest.json')
            and done['records_jsonl_sha256'] == sha(raw / 'records.jsonl')
            and manifest['references_read'] is False and done['quality_scored'] is False, 'closed unscored raw3 required')
    all_lines = (raw / 'records.jsonl').read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in all_lines]
    require([row['id'] for row in rows] == manifest['expected_ids']
            and len(rows) == done['expected_QA'] == done['recorded_QA'] == manifest['expected_QA']
            and len({row['id'] for row in rows}) == len(rows)
            and done['counts'] == dict(Counter(row['status'] for row in rows)), 'original raw order/coverage differs')
    domain = plan['domains'][dataset]
    require(manifest['inputs_sha256'] == domain['inputs_sha256'] == sha(raw / 'source_inputs.jsonl'), 'reference/fresh input cohort differs')
    ids = group['datasets'][dataset]['ids']; lookup = {row['id']: (number, row, line) for number, (row, line) in enumerate(zip(rows, all_lines), 1)}
    require(set(ids) <= set(lookup), 'source group missing from original producer')
    selected = [lookup[identifier] for identifier in ids]
    if entry['kind'] == 'fresh':
        require([row['id'] for row in rows] == ids, 'fresh QA must be exact D96 slice')
    view = Path(output_root) / 'views' / (dataset + '__' + mode)
    immutable(view / 'selected_records.jsonl', b''.join(line for _, _, line in selected))
    lineage = {'schema': 'trajkv.round11.exact_source_subset.v1', 'inference_run': False,
        'fresh_inference_source': entry['kind'] == 'fresh', 'reused_reference_source': entry['kind'] == 'reused',
        'source_raw_dir': str(raw), 'source_producer': proof['parent_producer'],
        'source_driver': proof['actual_driver'], 'source_receipt_sha256': sha(proof_path),
        'source_raw_sha256': {name: sha(raw / name) for name in ('manifest.json', 'records.jsonl', 'complete.json')},
        'source_runtime_sha256': manifest['source_hashes'], 'source_group_manifest_sha256': plan['source_group_manifest_sha256'],
        'selected_records_sha256': sha(view / 'selected_records.jsonl'), 'ids': ids,
        'source_lines': [{'id': row['id'], 'line_number': number, 'exact_line_sha256': hashlib.sha256(line).hexdigest()}
                         for number, row, line in selected], 'canonical_SourceRun_admission_claimed': False}
    write(view / 'source_lineage.json', lineage)
    return {'manifest': manifest, 'config': cfg, 'rows': selected, 'lineage': lineage,
            'lineage_path': str(view / 'source_lineage.json'), 'view': view, 'raw': raw}


def score_view(entry, selected, plan):
    import round10_harvest_score_v1 as old
    from prepare_remaining_phase_delivery_v1 import physical_complete_bytes
    from metrics import score_question
    from score import summarize_group, FINISHED_STATUSES
    dataset, mode = entry['dataset'], entry['mode']; domain = plan['domains'][dataset]
    references = {row['id']: row for row in map(json.loads, Path(domain['refs']).read_text(encoding='utf-8-sig').splitlines())}
    inputs = {row['id']: row for row in map(json.loads, Path(domain['inputs']).read_text(encoding='utf-8-sig').splitlines())}
    scored, capacities, bits, identities = [], [], [], {}
    for number, row, line in selected['rows']:
        source, ref = inputs[row['id']], references[row['id']]
        for key in ('dataset', 'dataset_revision', 'split', 'source_cluster_id', 'source_id',
                    'question_id', 'original_image_sha256', 'question_sha256'):
            require(row.get(key) == source.get(key), 'raw/source identity differs: ' + key)
        require(ref['source_cluster_id'] == source['source_cluster_id'] and row.get('official_score') is None,
                'reference source group or unscored raw changed')
        require(row['status'] in FINISHED_STATUSES, 'unfinished/unknown raw status')
        if row['status'] != 'PASS':
            scored.append({'id': row['id'], 'dataset': dataset, 'mode': mode,
                'source_cluster_id': ref['source_cluster_id'], 'group': ref.get('group'),
                'score_status': 'NOT_SCORED_RAW_FAILURE', 'score': None,
                'raw_status': row['status'], 'raw_error': row.get('error'),
                'source_raw_jsonl_line_number': number,
                'source_raw_jsonl_exact_line_sha256': hashlib.sha256(line).hexdigest()})
            continue
        require(isinstance(row.get('raw_prediction'), str)
                and isinstance(row.get('output_token_ids'), list) and row['output_token_ids'], 'full-cohort PASS genuine prediction required')
        require(row['raw_question'] == source['question'] and row['question_suffix'] == runner.frozen.SUFFIX
                and row['question_unknown_prefix_verified'] is True and row['original_full_prefix_released'] is True
                and row['original_FP16_visual_alias_retained'] is False
                and row['first_generation_call']['native_first_logits_injected'] is False,
                'question/source-release/actual suffix evidence differs')
        actual = physical_complete_bytes(row, selected['manifest']['model_spec'])
        require(row['owner_after_generation']['stored_visual_KV_bytes'] == actual,
                'persistent owner changed during real generation')
        require(math.isclose(row['actual_effective_visual_bits'], 16 * actual / row['full_kv_byte_denominator'],
                             rel_tol=1e-12, abs_tol=1e-12), 'physical bits denominator differs')
        if entry['kind'] == 'fresh':
            budget = runner.round11_budget_guard(mode, row['actual_visual_token_count'], actual,
                row['r7_bridge'], selected['config']['variant_source']['name'])
            require(row['round10_budget_vs_original_R7_low'] == budget, 'actual Round11 physical guard differs')
        elif mode == 'R7low':
            require(actual == 4590 * row['actual_visual_token_count'] + 89733, 'reused R7low physical B0 differs')
        metric = score_question(dataset, row['raw_prediction'], ref)
        require(metric['metric_id'] == runner.frozen.PRIMARY_METRIC_IDS[dataset], 'frozen primary metric differs')
        scored.append({**metric, 'id': row['id'], 'dataset': dataset, 'mode': mode,
            'source_cluster_id': ref['source_cluster_id'], 'group': ref.get('group'), 'score_status': 'SCORED_PASS',
            'source_raw_jsonl_line_number': number, 'source_raw_jsonl_exact_line_sha256': hashlib.sha256(line).hexdigest(),
            'output_token_ids': row['output_token_ids'], 'raw_prediction': row['raw_prediction'],
            'stored_visual_KV_bytes': actual, 'actual_effective_visual_bits': row['actual_effective_visual_bits']})
        capacities.append(actual); bits.append(row['actual_effective_visual_bits']); identities[row['id']] = old.identity(row)
    primary = summarize_group(scored, 2000, 20261123)
    require(len(scored) == COUNTS[dataset], 'complete D96 scoring denominator required')
    full_pass = primary['failure_questions'] == 0
    summary = {'schema': 'trajkv.round11.offline_source_subset_score.v1', 'dataset': dataset, 'mode': mode,
        'status': 'PASS_FULL_SOURCE_GROUP_SCORE' if full_pass else 'CLOSED_WITH_FAILURES_NO_PRIMARY',
        'expected_QA': len(scored), 'raw_failure_questions': primary['failure_questions'],
        'failure_QA_remain_in_primary_denominator': True,
        'fresh_free_generation_QA': primary['scored_pass_questions'] if entry['kind'] == 'fresh' else 0,
        'reused_reference_views': len(scored) if entry['kind'] == 'reused' else 0,
        'primary_mean': primary['primary']['mean'], 'ci95': primary['primary']['ci95'], 'groups': {'overall': primary},
        'resources': {'complete_bytes': stats(capacities), 'actual_complete_bits': stats(bits),
            'strict_all_QA_1p3': all(160 * b <= 13 * row['full_kv_byte_denominator'] for b, (_, row, _) in zip(capacities, selected['rows'])),
            'strict_all_QA_1p8': all(80 * b <= 9 * row['full_kv_byte_denominator'] for b, (_, row, _) in zip(capacities, selected['rows']))} if full_pass else None,
        'source_lineage_sha256': sha(selected['lineage_path']),
        'refs_sha256': domain['refs_sha256'], 'source_lock_sha256': domain['source_lock_sha256'],
        'scoring_source_pins': check_pins(), 'canonical_SourceRun_admission_claimed': False,
        'controlled_speed_or_HBM_traffic_claimed': False}
    if dataset == 'chartqa':
        for group in ('human', 'aug'):
            subset = [row for row in scored if ('aug' if row.get('group') == 'augmented' else row.get('group')) == group]
            if subset:
                summary['groups'][group] = summarize_group(subset, 2000, 20261123)
    immutable(selected['view'] / 'scored_records.jsonl', b''.join((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode() for row in scored))
    summary['scored_records_sha256'] = sha(selected['view'] / 'scored_records.jsonl')
    write(selected['view'] / 'summary.json', summary)
    bundle = {'context': old.context(selected['manifest']), 'identities': identities,
              'scores': {row['id']: row for row in scored}} if full_pass else None
    return summary, bundle


def paired(anchor, candidate):
    from score import cluster_bootstrap
    require(anchor['context'] == candidate['context'] and anchor['identities'] == candidate['identities'],
            'paired runtime, source, prompt or original position identity differs')
    rows = [{'id': identifier, 'source_cluster_id': row['source_cluster_id'],
             'delta': candidate['scores'][identifier]['score'] - row['score']}
            for identifier, row in anchor['scores'].items()]
    stats_ = cluster_bootstrap(rows, 'delta', 2000, 20261123)
    changes = [row['delta'] for row in rows]
    return {'schema': 'trajkv.round11.paired_source_group.v1', 'status': 'PASS_PAIRED_COMPLETE', **stats_,
            'improved': sum(value > 0 for value in changes), 'degraded': sum(value < 0 for value in changes),
            'equal': sum(value == 0 for value in changes),
            'gross_gain_pp': 100 * sum(max(0, value) for value in changes) / len(changes),
            'gross_loss_pp': 100 * sum(max(0, -value) for value in changes) / len(changes),
            'descriptive_unadjusted_interval': True, 'source_identity_and_runtime_exactly_matched': True}


def summarize(plan_path, output_root, destination):
    pins = check_pins(); plan, group = load_plan(plan_path)
    require(not Path(destination).exists(), 'new immutable summary destination required')
    points, bundles = [], {}
    lookup = {(entry['dataset'], entry['mode']): entry for entry in plan['entries']}
    for dataset in DOMAINS:
        for mode in MODES:
            entry = lookup.get((dataset, mode))
            selected = selected_view(entry, plan, group, output_root) if entry else None
            if selected is None:
                points.append({'dataset': dataset, 'mode': mode, 'status': 'PENDING_PROVEN_SOURCE',
                               'expected_QA': COUNTS[dataset], 'primary_mean': None, 'ci95': None,
                               'fresh_free_generation_QA': 0, 'reused_reference_views': 0, 'resources': None})
                continue
            point, bundle = score_view(entry, selected, plan)
            points.append(point)
            if bundle is not None:
                bundles[(dataset, mode)] = bundle
    comparisons = {}
    for dataset in DOMAINS:
        for candidate in CANDIDATES:
            for anchor in REFERENCES:
                if (dataset, candidate) in bundles and (dataset, anchor) in bundles:
                    comparisons[dataset + '__' + candidate + '_minus_' + anchor] = paired(bundles[(dataset, anchor)], bundles[(dataset, candidate)])
        for candidate, anchor in (('JointSpatialWO', 'SpatialWO'),):
            if (dataset, candidate) in bundles and (dataset, anchor) in bundles:
                comparisons[dataset + '__' + candidate + '_minus_' + anchor] = paired(bundles[(dataset, anchor)], bundles[(dataset, candidate)])
    closed = sum(point['status'] != 'PENDING_PROVEN_SOURCE' for point in points)
    result = {'schema': 'trajkv.round11_wo.harvest_summary.v1',
        'status': 'PASS_ALL_20_PROVEN_VIEWS' if len(bundles) == 20 else 'PARTIAL_OR_FAILED_PROVEN_VIEWS',
        'expected_cells_or_views': 20, 'closed_cells_or_views': closed,
        'successful_complete_cells_or_views': len(bundles),
        'expected_new_free_generation_QA': 192, 'expected_reused_reference_views': 288,
        'actual_closed_new_free_generation_QA': sum(point['fresh_free_generation_QA'] for point in points),
        'actual_closed_reused_reference_views': sum(point['reused_reference_views'] for point in points),
        '480_views_are_not_480_new_QA': True, 'old_288_reference_views_repeat_primary_group': True, 'QA0_teacher_TF_excluded_from_task_quality': True,
        'source_group_manifest_sha256': plan['source_group_manifest_sha256'],
        'plan_sha256': sha(plan_path), 'pins': pins, 'points': points,
        'canonical_SourceRun_admission_claimed': False, 'speed_or_HBM_traffic_claimed': False}
    write(Path(destination) / 'summary.json', result)
    write(Path(destination) / 'paired_comparisons.json', comparisons)
    rows = [{'dataset': point['dataset'], 'mode': point['mode'], 'status': point['status'],
             'selected_QA_or_views': point['expected_QA'], 'fresh_QA': point['fresh_free_generation_QA'],
             'reused_views': point['reused_reference_views'], 'primary_mean': point['primary_mean'], 'ci95': point['ci95'],
             'mean_complete_bytes': (point['resources'] or {}).get('complete_bytes', {}).get('mean')}
            for point in points]
    target = Path(destination) / 'quality_complete_bytes.csv'
    with target.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('${EVIDENCE_ROOT}/results/round11/wo_harvest_v1'))
    parser.add_argument('--summary-output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    require(args.output_root.resolve().is_relative_to(Path('${EVIDENCE_ROOT}/results/round11').resolve())
            and args.summary_output.resolve().is_relative_to(Path('${EVIDENCE_ROOT}/results/round11').resolve()),
            'heavy outputs must use the Round11 E data mirror')
    check_pins(); plan, _ = load_plan(args.plan)
    if args.fetch:
        from remote_control import connect
        client = connect()
        try:
            for entry in plan['entries']:
                if entry['kind'] == 'fresh':
                    print(json.dumps(fetch_cell(client, entry['item'], args.output_root / 'raw' / entry['item']['cell']['name'])), flush=True)
        finally:
            client.close()
    result = summarize(args.plan, args.output_root, args.summary_output)
    print(json.dumps({'status': result['status'], 'closed_cells_or_views': result['closed_cells_or_views']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
