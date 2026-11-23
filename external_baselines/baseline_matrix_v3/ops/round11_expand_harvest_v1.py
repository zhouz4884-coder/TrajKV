"""Fixed additional384 fresh-source guard and frozen offline scoring adapter.

All three references must be newly executed by the data-only expansion wrapper.
The immutable additional384 and D96-selection manifests bind every parent/child proof.
No existing D96 or Round10 prediction can be relabelled as confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'code')]
import round11_expand_runner_v1 as runner
import round11_harvest_v2 as shared
import round10_harvest_score_v1 as identity_helpers

PROOF = 'source_retrieval_receipt_round11_expand384_v1.json'
SEQUENCE_SHA = '2d0dce9da80156fb0224a04d48a9a3e3869ede5e9885885a6a25073ab511fe00'
SCRIPT = 'code/round11_expand_runner_v1.py'
read, sha, write, immutable = shared.read, shared.sha, shared.write, shared.immutable


def require(value, message):
    runner.require(value, 'HARVEST: ' + message)


# Self-contained read-only remote proof generation. All admission values are
# explicit request contracts; no older inspector text is patched or executed.
INSPECT = r'''
import hashlib,json,sys
from pathlib import Path,PurePosixPath
r=json.load(sys.stdin);root=Path(r['root']);job=r['job'];cell=r['cell'];name=job['name']
def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def below(path,parent):
 try:path.relative_to(parent);return True
 except ValueError:return False
assert name.replace('_','').replace('-','').isalnum()
out=Path(cell['output']);assert below(out,root/'results/round11_v1/expand384')
if not(out/'complete.json').is_file():
 print(json.dumps({'status':'PENDING_NO_COMPLETE'}));raise SystemExit(0)
launchpath=root/'jobs'/(name+'.launch.json');launch=json.loads(launchpath.read_text())
source=root/'jobs'/(name+'.source');argv=job['argv'];jobpath=root/'jobs'/(name+'.json')
assert h(jobpath)==launch['job_sha256']==r['job_sha256']
assert launch['name']==name and type(launch['pid']) is int and launch['pid']>0 and launch['source_snapshot']==str(source)
executed=[str(source/Path(arg).relative_to(root)) if arg.startswith(str(root)+'/code/') and arg.endswith('.py') else arg for arg in argv]
assert launch['argv_executed']==executed and Path(executed[2])==source/'code/round10_sequence_v1.py'
seqpath=Path(argv[argv.index('--config')+1]);assert h(seqpath)==r['sequence_sha256']
seq=json.loads(seqpath.read_text());assert seq['schema']=='trajkv.round10.sequence.v1' and cell in seq['entries']
parentout=Path(argv[argv.index('--output')+1]);statepath=parentout/'state.json';body=statepath.read_bytes();state=json.loads(body)
assert state['source_root']==str(source) and state['plan_sha256']==r['sequence_sha256']
success=[row for row in state['completed_cells'] if row['name']==cell['name']];failed=state.get('failed_cell',{})
if not success and failed.get('name')!=cell['name']:
 print(json.dumps({'status':'PENDING_PARENT_CELL_CLOSURE'}));raise SystemExit(0)
assert len(success)<=1
cfgpath=Path(cell['config']);assert below(cfgpath,root/'configs/round11_v1')
cfg=json.loads(cfgpath.read_text());m=json.loads((out/'manifest.json').read_text());done=json.loads((out/'complete.json').read_text())
assert h(cfgpath)==cell['config_sha256']==m['config_sha256'] and cfg==m['config']
assert m['schema']=='baseline_matrix.raw_inference.v1' and m['runner_schema']=='trajkv.round11.expand384.raw.v1'
assert cell['script']=='code/round11_expand_runner_v1.py' and cfg['method']['family']=='trajkv_round11_expand384'
assert m['runtime_source_root']==str(source) and m['source_hashes']==cfg['expected_source_hashes']
assert m['source_hashes'][cell['script']]==cell['script_sha256']
assert cfg['input_role']==cfg['cohort_kind']=='confirmation_expansion_holdout'
assert h(cfg['input_manifest_path'])==cfg['input_manifest_sha256']==r['input_manifest_sha256']
assert h(cfg['selection_manifest_path'])==cfg['selection_manifest_sha256']==r['selection_manifest_sha256']
assert h(cfg['expansion_design_path'])==cfg['expansion_design_sha256']==r['expansion_design_sha256']
assert h(out/'manifest.json')==done['manifest_sha256'] and h(out/'records.jsonl')==done['records_jsonl_sha256']
rows=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
assert len(rows)==done['expected_QA']==done['recorded_QA']==m['expected_QA']==cell['expected_QA']
assert [row['id'] for row in rows]==m['expected_ids'] and done['coverage_complete'] is True
if success:
 assert success[0]['exit_code']==0 and success[0]['output']==str(out)
 assert success[0]['complete_sha256']==h(out/'complete.json') and success[0]['recorded_QA']==len(rows)
 assert done['status']=='PASS_RAW_COMPLETE'
else:assert failed['exit_code']!=0 and failed['output']==str(out) and done['status']=='COMPLETE_WITH_FAILURES'
inventory=source/'sha256.json';assert h(inventory)==launch['source_inventory_sha256'];iv=json.loads(inventory.read_text())
assert h(source/'code/round10_sequence_v1.py')==iv['code/round10_sequence_v1.py']==r['sequence_script_sha256']
starts=[]
with Path(launch['log']).open('rb') as stream:
 for number,line in enumerate(stream,1):
  try:event=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if event.get('event')=='START_CELL' and event.get('name')==cell['name']:
   starts.append({'event':event,'line_number':number,'exact_line_utf8':line.decode(),'exact_line_sha256':hashlib.sha256(line).hexdigest()})
assert len(starts)==1 and starts[0]['event']['config_sha256']==cell['config_sha256']
assert starts[0]['event']['argv']==[executed[0],'-B',str(source/cell['script']),'--config',cell['config'],'--output',cell['output']]
assert m['runtime_process']['PID']==starts[0]['event']['pid'] and m['runtime_process']['source_snapshot_root']==str(source)
assert m['runtime_process']['parent_PID']==launch['pid']
files=[]
def add(label,path,digest=None):
 path=Path(path);assert path.is_file() and path.stat().st_size<=256*1024**2
 actual=h(path)
 if digest is not None:assert actual==digest
 files.append({'file':label,'remote_file':str(path),'sha256':actual,'bytes':path.stat().st_size})
for key in ('manifest.json','records.jsonl','complete.json'):add(key,out/key)
add('source_job.json',jobpath,r['job_sha256']);add('launch_receipt.json',launchpath)
add('source_inventory.json',inventory,launch['source_inventory_sha256']);add('source_config.json',cfgpath,cell['config_sha256'])
add('source_sequence_config.json',seqpath,r['sequence_sha256'])
add('source_parent_driver.py',source/'code/round10_sequence_v1.py',r['sequence_script_sha256'])
for rel,digest in m['source_hashes'].items():
 p=PurePosixPath(rel);assert not p.is_absolute() and '..' not in p.parts and rel.startswith('code/') and rel.endswith('.py')
 assert iv[rel]==digest;add('source_files/'+rel,source/rel,digest)
add('source_inputs.jsonl',cfg['inputs'],cfg['inputs_sha256'])
add('input_manifest.json',cfg['input_manifest_path'],r['input_manifest_sha256'])
add('selection_manifest.json',cfg['selection_manifest_path'],r['selection_manifest_sha256'])
add('expansion_design.json',cfg['expansion_design_path'],r['expansion_design_sha256'])
add('assets/original_profile.json',cfg['variant_source']['path'],cfg['variant_source']['sha256'])
for role,digest in cfg['expected_asset_sha256'].items():add('assets/'+role+'.json',cfg['variant'][role],digest)
for path,digest in done.get('executed_legacy_source_sha256',{}).items():add('legacy_sources/'+digest[:16]+'_'+Path(path).name,path,digest)
assert len(files)==len({entry['file'] for entry in files})
print(json.dumps({'schema':'trajkv.round11.expand384_source_proof.v1','status':'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA',
 'remote_hashes_computed_independently':True,'parent_producer':name,'cell_name':cell['name'],'actual_driver':cell['script'],
 'actual_child_start':starts[0],'parent_cell_success':bool(success),'SourceRun_admission_established':False,
 'parent_state_snapshot':{'sha256':hashlib.sha256(body).hexdigest(),'body_utf8':body.decode()},'files':files}))
'''


def check_pins():
    pins = shared.check_pins()
    require(sha(ROOT / 'code/round10_sequence_v1.py') == SEQUENCE_SHA, 'frozen sequence driver changed')
    return {**pins, **runner.runtime_source_manifest(), 'ops/round11_expand_harvest_v1.py': sha(__file__)}


def load_plan(path):
    plan = read(path)
    digest = plan.get('input_manifest_sha256')
    require(plan.get('schema') == 'trajkv.round11.expand384_harvest_plan.v1'
            and digest == runner.MANIFEST_SHA
            and sha(plan['input_manifest_path']) == digest, 'actual additional384 manifest required')
    runner.design_identity({'expansion_design_path': plan['expansion_design_path'],
                            'expansion_design_sha256': plan['expansion_design_sha256']})
    selection = runner.selection_identity({'selection_manifest_path': plan['selection_manifest_path'],
        'selection_manifest_sha256': plan['selection_manifest_sha256'],
        'expansion_design_sha256': runner.DESIGN_SHA, 'original_R128_manifest_sha256': runner.ORIGINAL_R128_MANIFEST_SHA, 'candidate': {'mode':'FP16'},
        'variant_source': {'name':'r7_joint_k1_gt_cond'}})
    modes = (*runner.REFERENCES, *selection['selected_modes'])
    require(tuple(modes) == runner.MODES, 'additional384 cannot reselect or remove a method')
    manifest = read(plan['input_manifest_path'])
    require(manifest.get('schema') == 'trajkv.round11.expand384_inputs.v1'
            and manifest.get('counts') == runner.COUNTS and manifest.get('total_count') == 384
            and manifest.get('original_selection_sha256') == runner.ORIGINAL_SELECTION_SHA
            and manifest.get('base_R128_manifest_sha256') == runner.ORIGINAL_R128_MANIFEST_SHA
            and manifest.get('expansion_design_sha256') == runner.DESIGN_SHA,
            'new source cohort must preserve original128/selection/design identities')
    require(set(plan['domains']) == set(runner.COUNTS), 'all4 additional384 domains required')
    for dataset, count in runner.COUNTS.items():
        domain = plan['domains'][dataset]
        values = runner.data.load_expand_inputs(domain['inputs'], plan['input_manifest_path'], domain['inputs_sha256'],
                                                expected_role='confirmation_expansion_holdout')
        require(len(values) == count and {row['dataset'] for row in values} == {dataset}, 'actual full144/48 source coverage required')
        for key in ('refs','source_lock'):
            require(sha(domain[key]) == domain[key+'_sha256'], 'offline frozen source/'+key+' differs')
        require(domain['source_lock_sha256'] == manifest['source_lock']['sha256'], 'additional384 source lock differs')
        lock = read(domain['source_lock'])
        for key in ('inputs','refs'):
            entry = lock['files'][Path(domain[key]).name]
            require(entry['sha256'] == domain[key+'_sha256'] and Path(domain[key]).stat().st_size == entry['bytes'], 'source lock byte identity differs')
    entries = plan['entries']
    require(len(entries) == 20 and {(entry['dataset'],entry['mode']) for entry in entries}
            == {(dataset,mode) for dataset in runner.COUNTS for mode in modes}, 'all20 fresh full-domain cells required')
    for entry in entries:
        item = entry['item']; cell = item['cell']
        require(entry['kind'] == 'fresh' and cell['dataset'] == entry['dataset'] and cell['mode'] == entry['mode']
                and cell['expected_QA'] == runner.COUNTS[entry['dataset']]
                and cell['script'] == SCRIPT and cell['script_sha256'] == sha(ROOT/SCRIPT)
                and item['sequence_script_sha256'] == SEQUENCE_SHA
                and item['input_manifest_sha256'] == digest
                and item['selection_manifest_sha256'] == runner.ORIGINAL_SELECTION_SHA
                and item['expansion_design_sha256'] == runner.DESIGN_SHA, 'actual new384 full-cell/source binding differs')
    return plan, modes


def inspect_remote(client, request):
    stdin, stdout, stderr = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', INSPECT]), timeout=60)
    stdin.write(json.dumps(request)); stdin.flush(); stdin.channel.shutdown_write()
    body, error = stdout.read(), stderr.read()
    require(stdout.channel.recv_exit_status() == 0, 'read-only confirmation inspection failed: ' + error.decode(errors='replace')[-2500:])
    return json.loads(body)


def fetch_cell(client, item, output):
    proof = inspect_remote(client, item)
    if proof['status'].startswith('PENDING_'):
        return proof
    sftp = client.open_sftp()
    try:
        for entry in proof['files']:
            relative = PurePosixPath(entry['file']); require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe evidence path')
            target = output.joinpath(*relative.parts)
            if target.exists():
                require(sha(target) == entry['sha256'] and target.stat().st_size == entry['bytes'], 'existing retrieved evidence changed')
                continue
            with sftp.open(entry['remote_file'], 'rb') as stream:
                body = stream.read()
            require(len(body) == entry['bytes'] and hashlib.sha256(body).hexdigest() == entry['sha256'], 'remote readback differs')
            immutable(target, body)
    finally:
        sftp.close()
    receipt = output / PROOF
    if receipt.exists():
        stable = lambda value: {key: item for key, item in value.items() if key != 'parent_state_snapshot'}
        require(stable(read(receipt)) == stable(proof), 'immutable source proof changed')
    else:
        write(receipt, proof)
    return {'status': proof['status'], 'cell_name': item['cell']['name']}


def check_source(raw, item):
    proof = read(raw / PROOF); cell = item['cell']
    require(proof.get('schema') == 'trajkv.round11.expand384_source_proof.v1'
            and proof['actual_driver'] == SCRIPT
            and proof['status'] == 'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA'
            and proof['remote_hashes_computed_independently'] is True and proof['SourceRun_admission_established'] is False
            and proof['parent_producer'] == item['job']['name'] and proof['cell_name'] == cell['name'], 'wrong confirmation producer proof')
    require(len(proof['files']) == len({entry['file'] for entry in proof['files']}), 'duplicate source evidence paths')
    for entry in proof['files']:
        rel = PurePosixPath(entry['file']); require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe evidence path')
        path = raw.joinpath(*rel.parts)
        require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'remote-anchored source/raw changed')
    m, done, cfg = read(raw / 'manifest.json'), read(raw / 'complete.json'), read(raw / 'source_config.json')
    actual_count = runner.COUNTS[cell['dataset']]
    require(cfg['start'] == 0 and cfg['count'] == cfg['input_count'] == m['expected_QA']
            == done['expected_QA'] == done['recorded_QA'] == cell['expected_QA'] == actual_count
            and m['expansion_identity']['count'] == actual_count
            and m['legacy_structural_validator_QA_count_shadow']['actual_count'] == actual_count
            and cfg['max_new_tokens'] == 64,
            'actual config/main/raw/parent coverage must be full144/48; legacy shadow cannot leak into evidence')
    require(m['runner_schema'] == runner.SCHEMA and m['schema'] == 'baseline_matrix.raw_inference.v1'
            and cfg == m['config'] and sha(raw / 'source_config.json') == cell['config_sha256'] == m['config_sha256'], 'confirmation config/raw schema binding differs')
    require(cell['script'] == SCRIPT and runner.logical_mode(cfg) == cell['mode']
            and cfg['input_role'] == cfg['cohort_kind'] == 'confirmation_expansion_holdout', 'reused/development/wrong-mode source cannot be confirmation')
    require(sha(raw / 'source_job.json') == item['job_sha256'] and read(raw / 'source_job.json') == item['job']
            and sha(raw / 'source_sequence_config.json') == item['sequence_sha256']
            and cell in read(raw / 'source_sequence_config.json')['entries']
            and sha(raw / 'source_parent_driver.py') == SEQUENCE_SHA, 'parent job/sequence identity differs')
    launch = read(raw / 'launch_receipt.json'); root = item['root'].rstrip('/'); job = item['job']
    snapshot = root + '/jobs/' + job['name'] + '.source'
    argv = [snapshot + '/code/' + arg.rsplit('/code/', 1)[1]
            if arg.startswith(root + '/code/') and arg.endswith('.py') else arg for arg in job['argv']]
    require(launch['name'] == job['name'] and type(launch['pid']) is int and launch['pid'] > 0
            and launch['source_snapshot'] == snapshot and launch['argv_executed'] == argv
            and launch['job_sha256'] == item['job_sha256'] and m['runtime_source_root'] == snapshot,
            'actual parent PID/snapshot/invocation differs')
    require(sha(raw / 'source_inventory.json') == launch['source_inventory_sha256'], 'source inventory changed')
    inventory = read(raw / 'source_inventory.json')
    require(m['source_hashes'] == cfg['expected_source_hashes'] == runner.runtime_source_manifest()
            and inventory['code/round10_sequence_v1.py'] == SEQUENCE_SHA
            and all(inventory[role] == digest == sha(raw / 'source_files' / role) for role, digest in m['source_hashes'].items()), 'complete source graph differs')
    event = proof['actual_child_start']; start = event['event']
    require(hashlib.sha256(event['exact_line_utf8'].encode()).hexdigest() == event['exact_line_sha256']
            and json.loads(event['exact_line_utf8']) == start and start['name'] == cell['name']
            and start['config_sha256'] == cell['config_sha256'] and type(start['pid']) is int and start['pid'] > 0
            and start['argv'] == [argv[0], '-B', snapshot + '/' + SCRIPT, '--config', cell['config'], '--output', cell['output']]
            and m['runtime_process']['PID'] == start['pid'] and m['runtime_process']['source_snapshot_root'] == snapshot,
            'actual child PID/START_CELL differs')
    require(m['runtime_process']['parent_PID'] == launch['pid'], 'recorded child parent_PID differs from actual sequence parent PID')
    state = proof['parent_state_snapshot']
    require(hashlib.sha256(state['body_utf8'].encode()).hexdigest() == state['sha256'], 'parent state snapshot changed')
    state = json.loads(state['body_utf8'])
    require(state['plan_sha256'] == item['sequence_sha256'] and state['source_root'] == snapshot, 'parent state scope differs')
    if proof['parent_cell_success']:
        matches = [row for row in state['completed_cells'] if row['name'] == cell['name']]
        require(len(matches) == 1 and matches[0]['exit_code'] == 0 and matches[0]['output'] == cell['output']
                and matches[0]['recorded_QA'] == cell['expected_QA'] and matches[0]['complete_sha256'] == sha(raw / 'complete.json')
                and done['status'] == 'PASS_RAW_COMPLETE', 'parent exit0 does not bind raw completion')
    else:
        failed = state.get('failed_cell', {})
        require(failed.get('name') == cell['name'] and failed['exit_code'] != 0 and failed['output'] == cell['output']
                and done['status'] == 'COMPLETE_WITH_FAILURES', 'failed-cell proof differs')
    require(sha(raw / 'input_manifest.json') == cfg['input_manifest_sha256'] == item['input_manifest_sha256']
            and sha(raw / 'selection_manifest.json') == cfg['selection_manifest_sha256'] == item['selection_manifest_sha256'] == runner.ORIGINAL_SELECTION_SHA
            and sha(raw / 'expansion_design.json') == cfg['expansion_design_sha256'] == item['expansion_design_sha256'] == runner.DESIGN_SHA, 'additional384/selection manifest differs')
    require(done['manifest_sha256'] == sha(raw / 'manifest.json') and done['records_jsonl_sha256'] == sha(raw / 'records.jsonl')
            and done['coverage_complete'] is True and done['quality_scored'] is False and m['references_read'] is False,
            'unscored closed exact raw3 required')
    return m, done, cfg


def score_cell(entry, raw, plan):
    from collections import Counter
    from metrics import score_question
    from score import summarize_group, FINISHED_STATUSES
    from prepare_remaining_phase_delivery_v1 import physical_complete_bytes
    m, done, cfg = check_source(raw, entry['item']); dataset = entry['dataset']; count = runner.COUNTS[dataset]
    domain = plan['domains'][dataset]
    inputs = runner.data.load_expand_inputs(domain['inputs'], plan['input_manifest_path'], domain['inputs_sha256'], expected_role='confirmation_expansion_holdout')
    refs = {row['id']: row for row in map(json.loads, Path(domain['refs']).read_text(encoding='utf-8-sig').splitlines())}
    lines = (raw / 'records.jsonl').read_bytes().splitlines(keepends=True); rows = [json.loads(line) for line in lines]
    ids = [row['id'] for row in inputs]
    require([row['id'] for row in rows] == ids == m['expected_ids'] == m['expansion_identity']['ids']
            and len(rows) == count == cfg['input_count'] == cfg['count'] == done['recorded_QA'] == done['expected_QA']
            and cfg['start'] == 0 and done['counts'] == dict(Counter(row['status'] for row in rows)), 'complete exact additional384 domain coverage differs')
    scores, capacities, identities = [], [], {}
    for number, (row, source, line) in enumerate(zip(rows, inputs, lines), 1):
        ref = refs[row['id']]
        require(all(row.get(key) == source.get(key) for key in ('dataset','dataset_revision','split','source_id','source_cluster_id','question_id','original_image_sha256','question_sha256'))
                and ref['source_cluster_id'] == source['source_cluster_id'] and row.get('official_score') is None, 'raw/frozen source identity differs')
        sr = {'id': row['id'], 'source_cluster_id': source['source_cluster_id'], 'group': ref.get('group'),
            'raw_line_number': number, 'raw_line_sha256': hashlib.sha256(line).hexdigest(), 'score': None,
            'score_status': 'NOT_SCORED_RAW_FAILURE', 'raw_status': row['status']}
        require(row['status'] in FINISHED_STATUSES, 'unfinished raw record')
        if row['status'] == 'PASS':
            require(row['raw_question'] == source['question'] and row['question_suffix'] == runner.frozen.SUFFIX
                    and row['question_unknown_prefix_verified'] is True and row['original_full_prefix_released'] is True
                    and row['original_FP16_visual_alias_retained'] is False and row['first_generation_call']['native_first_logits_injected'] is False,
                    'actual native-prefix/suffix/release identity differs')
            actual = physical_complete_bytes(row, m['model_spec'])
            require(row['owner_after_generation']['stored_visual_KV_bytes'] == actual
                    and math.isclose(row['actual_effective_visual_bits'], 16*actual/row['full_kv_byte_denominator'], rel_tol=1e-12, abs_tol=1e-12), 'complete physical owner differs')
            budget = runner.physical_budget_guard(cfg['candidate']['mode'], row['actual_visual_token_count'], actual, row['r7_bridge'], cfg['variant_source']['name'])
            require(row['round10_budget_vs_original_R7_low'] == budget, 'stored executed physical guard differs')
            metric = score_question(dataset, row['raw_prediction'], ref)
            require(metric['metric_id'] == runner.frozen.PRIMARY_METRIC_IDS[dataset], 'frozen primary metric differs')
            sr.update(metric, score_status='SCORED_PASS', output_token_ids=row['output_token_ids'], raw_prediction=row['raw_prediction'])
            capacities.append(actual); identities[row['id']] = identity_helpers.identity(row)
        scores.append(sr)
    aggregate = summarize_group(scores, 2000, 20261123); success = aggregate['failure_questions'] == 0
    point = {'dataset': dataset, 'mode': entry['mode'], 'status': 'PASS_FRESH_EXPAND384_SOURCE_SCORE' if success else 'CLOSED_WITH_FAILURES_NO_PRIMARY',
        'expected_QA': count, 'primary_mean': aggregate['primary']['mean'], 'ci95': aggregate['primary']['ci95'],
        'groups': {'overall': aggregate}, 'resources': {'complete_bytes': shared.stats(capacities)} if success else None,
        'source_receipt_sha256': sha(raw / PROOF), 'fresh_QA': aggregate['scored_pass_questions'],
        'reused_reference_views': 0, 'failed_QA_remain_in_primary_denominator': True,
        'canonical_SourceRun_admission_claimed': False, 'quality_is_frozen_subset_offline_score': True}
    scored = raw.parent.parent / 'scored' / entry['item']['cell']['name']
    immutable(scored / 'scored_records.jsonl', b''.join((json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n').encode() for row in scores))
    point['scored_records_sha256'] = sha(scored / 'scored_records.jsonl'); write(scored / 'summary.json', point)
    return point, {'context': identity_helpers.context(m), 'identities': identities, 'scores': {row['id']: row for row in scores}} if success else None


def summarize(plan_path, output, destination):
    pins = check_pins(); plan, modes = load_plan(plan_path)
    require(not Path(destination).exists(), 'new immutable summary destination required')
    points, bundles = [], {}
    for entry in plan['entries']:
        raw = Path(output) / 'raw' / entry['item']['cell']['name']
        if not (raw / PROOF).exists():
            points.append({'dataset': entry['dataset'], 'mode': entry['mode'], 'status': 'NOT_RUN',
                'expected_QA': runner.COUNTS[entry['dataset']], 'primary_mean': None, 'fresh_QA': 0})
            continue
        point, bundle = score_cell(entry, raw, plan); points.append(point)
        if bundle is not None:
            bundles[(entry['dataset'], entry['mode'])] = bundle
    pairs = {}
    for dataset in runner.COUNTS:
        for mode in modes:
            if mode in runner.REFERENCES:
                continue
            for reference in runner.REFERENCES:
                if (dataset, mode) in bundles and (dataset, reference) in bundles:
                    pairs[dataset+'__'+mode+'_minus_'+reference] = shared.paired(bundles[(dataset, reference)], bundles[(dataset, mode)])
        # Clean specificity contrast: the same frozen base and no JointV
        # difference. This does not choose either mode for confirmation.
        if (dataset, 'Shuffled') in bundles and (dataset, 'Spatial') in bundles:
            pairs[dataset+'__Shuffled_minus_Spatial'] = shared.paired(bundles[(dataset, 'Spatial')], bundles[(dataset, 'Shuffled')])
        if (dataset, 'JointSpatialWO') in bundles and (dataset, 'SpatialWO') in bundles:
            pairs[dataset+'__JointSpatialWO_minus_SpatialWO'] = shared.paired(bundles[(dataset, 'SpatialWO')], bundles[(dataset, 'JointSpatialWO')])
    result = {'schema': 'trajkv.round11.expand384_harvest_summary.v1',
        'status': 'PASS_FRESH_EXPAND384_ALL_CELLS' if len(bundles) == len(plan['entries']) else 'NOT_RUN_OR_PARTIAL',
        'expected_new_QA': 384*len(modes), 'expected_fresh_reference_QA': 1152,
        'actual_closed_fresh_QA': sum(point['fresh_QA'] for point in points), 'reused_reference_views': 0,
        'input_manifest_sha256': plan['input_manifest_sha256'], 'selection_manifest_sha256': plan['selection_manifest_sha256'],
        'expansion_design_sha256': runner.DESIGN_SHA, 'original_R128_manifest_sha256': runner.ORIGINAL_R128_MANIFEST_SHA,
        'plan_sha256': sha(plan_path), 'pins': pins, 'points': points, 'quality_of_pending_cells': 'NOT_RUN',
        'canonical_SourceRun_admission_claimed': False, 'speed_or_HBM_traffic_claimed': False}
    write(Path(destination)/'summary.json', result); write(Path(destination)/'paired_comparisons.json', pairs)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True); p.add_argument('--summary-output', type=Path, required=True)
    p.add_argument('--fetch', action='store_true'); args = p.parse_args()
    require(all(path.resolve().is_relative_to(Path('${EVIDENCE_ROOT}/results/round11').resolve())
                for path in (args.output_root, args.summary_output)), 'all heavy evidence must use E Round11')
    plan, _ = load_plan(args.plan)
    if args.fetch:
        from remote_control import connect
        client = connect()
        try:
            for entry in plan['entries']:
                print(json.dumps(fetch_cell(client, entry['item'], args.output_root/'raw'/entry['item']['cell']['name'])), flush=True)
        finally:
            client.close()
    result = summarize(args.plan, args.output_root, args.summary_output)
    print(json.dumps({'status': result['status'], 'actual_closed_fresh_QA': result['actual_closed_fresh_QA']}), flush=True)


if __name__ == '__main__':
    main()
