"""Source-bound reserve224 replication; the selected plan fixes its scope.

No selection, GPU launch or remote write is performed. All three references
must be new reserve224 predictions. D160 cells cannot enter this namespace.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'code')]
import round10_harvest_score_v1 as frozen
import round10_middle_harvest_v1 as common
import round10_extension_runner_v1 as runner

PINS = {**common.PINS,
    'ops/round10_middle_harvest_v1.py': '4bd78e263c23680f2162efdb1a0277b3029bdb70a53e2bd60038c2b10b191ebe',
    **runner.PARENT_PINS}
COUNTS, DOMAINS = runner.COUNTS, frozen.DOMAINS
REFERENCES = ('FP16', 'R7low', 'R7high')
require, sha, read, write = frozen.require, frozen.sha, frozen.read, frozen.write
SCRIPT = 'code/round10_extension_runner_v1.py'
_old = "assert cell['script']=='code/round10_runner_v1.py' and cfg['method']['family']=='trajkv_round10_research'"
_new = """assert cell['script']=='code/round10_extension_runner_v1.py' and cfg['method']['family']=='trajkv_round10_research'
assert cfg['input_role']=='reserved_unused_extension' and cfg['cohort_kind']=='replication'
assert cell['mode'] in ('FP16','R7low','R7high','LS-FZ','LS-RE','FB0','FB-W','V128-R43','AddOldOrder')"""
require(frozen.INSPECT.count(_old) == 1, 'frozen inspector script hook changed')
INSPECT = frozen.INSPECT.replace(_old, _new)
_asset_hook = "for key in ('theta','window_selection'):"
require(INSPECT.count(_asset_hook) == 1, 'frozen inspector asset hook changed')
INSPECT = INSPECT.replace(_asset_hook,
    "add('assets/extension_parent_config.json',cfg['extension_lineage']['parent_config_path'],cfg['extension_lineage']['parent_config_sha256'])\n" + _asset_hook)


def check_pins():
    for path, expected in PINS.items():
        require(sha(ROOT / path) == expected, 'frozen extension helper changed: ' + path)
    return {**PINS, **runner.runtime_source_manifest(),
            'ops/round10_extension_harvest_v1.py': sha(__file__)}


def load_plans(paths):
    """Return (items, origins, selected_modes) for exactly one fixed scope."""
    items, origins, selected = {}, [], None
    for path in paths:
        plan = read(path)
        require(plan['schema'] == 'trajkv.round10.job_plan.v1' and plan['phase'] == 'extension' and
                plan['status'] == 'FROZEN_PRE_LAUNCH' and plan['new_GPU_jobs_started_by_this_preparer'] is False and
                plan['input_manifest_sha256'] == runner.INPUT_MANIFEST_SHA, 'only declared reserve224 extension plans')
        modes = plan.get('selected_modes')
        require(isinstance(modes, list) and 1 <= len(modes) <= 2 and len(set(modes)) == len(modes) and
                all(mode in runner.CANDIDATE_MODES for mode in modes), 'one or two frozen selected candidate modes required')
        require(selected is None or selected == modes, 'selected candidates changed between plans')
        selected = list(modes)
        origins.append({'path': str(path), 'sha256': sha(path), 'selected_modes': modes})
        found = []
        for producer in plan['jobs']:
            jobpath = ROOT / producer['job']; job = read(jobpath)
            require(sha(jobpath) == producer['job_sha256'] and job['name'] == producer['name'], 'parent job differs')
            seqremote = job['argv'][job['argv'].index('--config') + 1]
            require('/configs/round10_v1/' in seqremote and job['argv'][2].endswith('/code/round10_sequence_v1.py'),
                    'wrong extension sequence producer')
            sequencepath = ROOT / 'configs/round10_v1' / PurePosixPath(seqremote).name
            sequence = read(sequencepath)
            require(sequence['schema'] == 'trajkv.round10.sequence.v1', 'wrong sequence schema')
            for cell in sequence['entries']:
                key = (cell['dataset'], cell['mode'])
                require(cell in plan['cells'] and cell['script'] == SCRIPT and cell['script_sha256'] == sha(ROOT / SCRIPT) and
                        key not in items and cell['dataset'] in DOMAINS and cell['mode'] in (*REFERENCES, *selected) and
                        cell['expected_QA'] == COUNTS[cell['dataset']] and 'extension' in cell['name'].lower() and
                        'extension' in cell['output'].lower(), 'undeclared/duplicate/partial/nonextension cell')
                items[key] = {'cell': cell, 'job': job, 'job_sha256': producer['job_sha256'],
                    'sequence_sha256': sha(sequencepath), 'sequence_script_sha256': sha(ROOT/'code/round10_sequence_v1.py'),
                    'input_manifest_sha256': plan['input_manifest_sha256'], 'selected_modes': selected}
                found.append(cell)
        require(len(found) == len(plan['cells']) and {c['name'] for c in found} == {c['name'] for c in plan['cells']},
                'extension cell lacks one parent')
        require(plan['planned_free_answers'] == sum(cell['expected_QA'] for cell in plan['cells']),
                'extension answer budget differs')
    require(selected is not None and set(items) == {(ds, mode) for ds in DOMAINS for mode in (*REFERENCES, *selected)},
            'the frozen extension scope must contain all four whole domains and new references')
    return items, origins, selected


def inspect_remote(client, request):
    import shlex
    from remote_control import REMOTE_ROOT
    stdin, stdout, stderr = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', INSPECT]), timeout=60)
    stdin.write(json.dumps({'root': REMOTE_ROOT, **request})); stdin.flush(); stdin.channel.shutdown_write()
    body, error = stdout.read(), stderr.read()
    require(stdout.channel.recv_exit_status() == 0, 'read-only extension inspection failed: ' + error.decode(errors='replace')[-2500:])
    return json.loads(body)


def fetch_cell(client, item, destination):
    previous = frozen.inspect_remote; frozen.inspect_remote = inspect_remote
    try:
        return frozen.fetch_cell(client, item, destination)
    finally:
        frozen.inspect_remote = previous


def check_local_proof(raw, item):
    """Frozen proof checks with the explicit reserve-role/count adapter."""
    proof = read(raw/'source_retrieval_receipt_round10_v1.json'); cell = item['cell']
    require(proof['status'] == 'PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA' and
            proof['remote_hashes_computed_independently'] is True and proof['SourceRun_admission_established'] is False and
            proof['cell_name'] == cell['name'] and proof['parent_producer'] == item['job']['name'] and
            proof['actual_driver'] == SCRIPT, 'wrong extension child proof')
    require(len(proof['files']) == len({e['file'] for e in proof['files']}), 'duplicate proof entries')
    for entry in proof['files']:
        relative = PurePosixPath(entry['file'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe local proof path')
        target = raw.joinpath(*relative.parts)
        require(sha(target) == entry['sha256'] and target.stat().st_size == entry['bytes'], 'local source/raw proof changed')
    cfg, manifest, closed = read(raw/'source_config.json'), read(raw/'manifest.json'), read(raw/'complete.json')
    require(sha(raw/'input_manifest.json') == item['input_manifest_sha256'] == runner.INPUT_MANIFEST_SHA,
            'original input manifest differs from independent readback')
    require(sha(raw/'source_config.json') == cell['config_sha256'] == manifest['config_sha256'] and cfg == manifest['config'], 'config binding differs')
    require(sha(raw/'source_parent_job.json') == item['job_sha256'] and read(raw/'source_parent_job.json') == item['job'], 'parent job binding differs')
    require(sha(raw/'source_sequence_config.json') == item['sequence_sha256'] and cell in read(raw/'source_sequence_config.json')['entries'], 'sequence binding differs')
    require(sha(raw/'source_driver.py') == cell['script_sha256'] and sha(raw/'source_parent_driver.py') == item['sequence_script_sha256'], 'actual driver differs')
    launch, job = read(raw/'launch_receipt.json'), item['job']; parent_name = job['name']
    remote = job['argv'][2].rsplit('/code/', 1)[0]; snapshot = remote+'/jobs/'+parent_name+'.source'
    expected = [snapshot+'/code/'+a.rsplit('/code/', 1)[1] if a.startswith(remote+'/code/') and a.endswith('.py') else a for a in job['argv']]
    require(launch['name'] == parent_name and type(launch['pid']) is int and launch['pid'] > 0 and
            launch['job_sha256'] == item['job_sha256'] and launch['source_snapshot'] == snapshot and
            launch['argv_executed'] == expected and manifest['runtime_source_root'] == snapshot, 'actual parent invocation differs')
    require(sha(raw/'source_inventory.json') == launch['source_inventory_sha256'], 'actual snapshot inventory differs')
    inventory = read(raw/'source_inventory.json')
    require(inventory['code/round10_sequence_v1.py'] == item['sequence_script_sha256'] and
            all(inventory[k] == v == sha(raw/'source_files'/k) for k, v in manifest['source_hashes'].items()), 'snapshot graph differs')
    event = proof['actual_child_start']; start = event['event']
    require(hashlib.sha256(event['exact_line_utf8'].encode()).hexdigest() == event['exact_line_sha256'] and
            json.loads(event['exact_line_utf8']) == start and start['name'] == cell['name'] and
            start['config_sha256'] == cell['config_sha256'] and type(start['pid']) is int and start['pid'] > 0 and
            start['argv'] == [expected[0], '-B', snapshot+'/'+cell['script'], '--config', cell['config'], '--output', cell['output']], 'actual child START_CELL differs')
    stateproof = proof['parent_state_snapshot']
    require(hashlib.sha256(stateproof['body_utf8'].encode()).hexdigest() == stateproof['sha256'], 'parent state bytes differ')
    state = json.loads(stateproof['body_utf8'])
    require(state['plan_sha256'] == item['sequence_sha256'] and state['source_root'] == snapshot, 'parent state scope differs')
    if proof['parent_cell_success']:
        result = [r for r in state['completed_cells'] if r['name'] == cell['name']]
        require(len(result) == 1 and result[0]['exit_code'] == 0 and result[0]['complete_sha256'] == sha(raw/'complete.json') and
                result[0]['recorded_QA'] == cell['expected_QA'] and result[0]['output'] == cell['output'], 'parent completion does not bind child')
    require(manifest['source_hashes'] == cfg['expected_source_hashes'] == runner.runtime_source_manifest(), 'extension source graph differs')
    require(closed['expected_QA'] == closed['recorded_QA'] == manifest['expected_QA'] == cell['expected_QA'], 'wrong complete count')
    expected_mode = 'original' if cell['mode'] in ('R7low', 'R7high') else cell['mode']
    require(cfg['candidate']['mode'] == expected_mode and cfg['input_role'] == 'reserved_unused_extension' and
            cfg['cohort_kind'] == 'replication' and cfg['start'] == 0 and
            cfg['input_count'] == cfg['count'] == COUNTS[cell['dataset']] and
            cfg['candidate'].get('technical_only', False) is False, 'partial/development/technical/mislabeled replication')
    require(cfg['variant_source']['name'] == ('r7_budget_k2_cond' if cell['mode'] == 'R7high' else 'r7_joint_k1_gt_cond'), 'wrong base profile')
    parent_path = raw/'assets/extension_parent_config.json'
    require(sha(parent_path) == cfg['extension_lineage']['parent_config_sha256'], 'frozen parent config exact bytes differ')
    runner.validate_parent_pair(cfg, read(parent_path))
    input_manifest = read(raw/'input_manifest.json'); ds = cell['dataset']
    entry = input_manifest['input_files'][f'{ds}_reserve{COUNTS[ds]}']
    require(entry['role'] == cfg['input_role'] and entry['count'] == cfg['input_count'] and
            entry['sha256'] == cfg['inputs_sha256'] == sha(raw/'source_inputs.jsonl') and
            input_manifest['source_lock']['sha256'] == runner.SOURCE_LOCK_SHA, 'frozen reserve role/input/source-lock differs')
    contract = manifest['extension_deployment_contract']
    require(contract['schema'] == 'trajkv.round10.reserve224_replication.v1' and
            contract['input_role_preserved'] == 'reserved_unused_extension' and contract['run_cohort_kind'] == 'replication' and
            contract['parent_config_sha256'] == cfg['extension_lineage']['parent_config_sha256'] and
            contract['scientific_method_id'] == cfg['method']['id'] and
            contract['scientific_method_and_candidate_parameters_unchanged'] is True and
            contract['calibration_parameters_reselected'] is False and contract['development_D160_predictions_reused'] is False,
            'replication manifest changed scientific contract')
    return manifest, closed, cfg, proof


def closed_cell(raw, scored, item, refs_root):
    require(sha(Path(refs_root)/'source_lock.json') == runner.SOURCE_LOCK_SHA, 'original reference source lock changed')
    previous = common.check_local_proof; common.check_local_proof = check_local_proof
    try:
        point, bundle = common.closed_cell(raw, scored, item, refs_root)
    finally:
        common.check_local_proof = previous
    if bundle is not None:
        cfg = read(raw/'source_config.json'); mode = cfg['candidate']['mode']
        for row in bundle['raw'].values():
            if mode in ('LS-FZ', 'LS-RE', 'FB0', 'FB-W'):
                bridge = row['r7_bridge']; detail = bridge['round10_value_candidate']
                require(bridge['selected_stage_count'] == len(bridge['selected_stages']) == 43 and
                        detail['mode'] == mode and detail['source_sha256'] == frozen.PINS['code/round10_value_bank_v2.py'] and
                        detail['native_FP16_source_or_feedback_workspace_retained'] is False, 'primary factory/coverage/source differs')
    return point, bundle


def summarize(items, origins, selected_modes, refs_root, output_root, destination):
    require(not Path(destination).exists(), 'new extension summary directory required')
    require(1 <= len(selected_modes) <= 2 and len(set(selected_modes)) == len(selected_modes) and
            all(mode in runner.CANDIDATE_MODES for mode in selected_modes), 'explicit selected replication scope required')
    pins = check_pins(); modes = (*REFERENCES, *selected_modes)
    points, bundles = [], {}
    for ds in DOMAINS:
        for mode in modes:
            key = (ds, mode)
            if key not in items:
                points.append({'dataset': ds, 'mode': mode, 'status': 'PENDING_DECLARED_PLAN_NOT_READY',
                    'expected_QA': COUNTS[ds], 'primary_mean': None, 'ci95': None, 'resources': None, 'timing': None})
                continue
            item = items[key]; name = item['cell']['name']
            point, bundle = closed_cell(Path(output_root)/'raw'/name, Path(output_root)/'scored'/name, item, refs_root)
            points.append(point)
            if bundle is not None:
                bundles[key] = bundle
    contrasts = [(mode, ref) for mode in selected_modes for ref in REFERENCES]
    if len(selected_modes) == 2:
        contrasts.append((selected_modes[1], selected_modes[0]))
    pairs = {ds+'__'+candidate+'_minus_'+anchor:
        common.paired(bundles[ds, anchor], bundles[ds, candidate]) if (ds, anchor) in bundles and (ds, candidate) in bundles else None
        for ds in DOMAINS for candidate, anchor in contrasts}
    closed = sum(point['primary_mean'] is not None for point in points); expected_cells = 4*len(modes)
    result = {'schema': 'trajkv.round10.extension_replication_summary.v1',
        'status': 'PASS_ALL_RESERVE224_REPLICATION_SOURCE_SCORE_PHYSICAL' if closed == expected_cells else 'PENDING_OR_PARTIAL_RESERVE224_REPLICATION',
        'selected_modes': list(selected_modes), 'expected_cells': expected_cells, 'closed_cells': closed,
        'expected_answer_events': 224*len(modes), 'expected_distinct_replication_sources': 224,
        'expected_new_reference_answer_events': 672, 'expected_new_candidate_answer_events': 224*len(selected_modes),
        'closed_answer_events': sum(p['expected_QA'] for p in points if p['primary_mean'] is not None),
        'actual_closed_unique_IDs': len({key for bundle in bundles.values() for key in bundle['identity']}),
        'frozen_input_role_preserved': 'reserved_unused_extension', 'run_cohort_kind': 'replication',
        'D160_results_mixed_into_replication': False, 'new_reference_QA_required': True,
        'scientific_method_ids_and_calibration_parameters_unchanged': True if closed == expected_cells else None,
        'scientific_method_or_calibration_parameter_changes_permitted': False,
        'closed_cells_lineage_validated': closed,
        'independent_blind_confirmation_claimed': False, 'formal500_or_cross_model_claimed': False,
        'SourceRun_admission_established': False, 'new_GPU_jobs_started': 0,
        'no_cross_dataset_quality_average': True, 'paired_CI_and_signflip_unadjusted_not_equivalence_or_algorithm_winner': True,
        'quality_resource_and_timing_are_separate': 'All-QA physical bytes; instrumented quality-run host timing with natural answer lengths; no controlled speed/HBM traffic claim.',
        'plan_sources': origins, 'points': points, 'paired_contrasts': list(pairs),
        'source_sha256': sha(__file__), 'helper_sha256': pins}
    Path(destination).mkdir(parents=True)
    write(Path(destination)/'summary.json', result); write(Path(destination)/'paired_comparisons.json', pairs)
    table = []
    for point in points:
        resources = point['resources'] or {}; capacity = resources.get('complete_bytes') or {}; bits = resources.get('actual_complete_bits') or {}
        table.append({'dataset': point['dataset'], 'mode': point['mode'], 'status': point['status'], 'QA': point['expected_QA'],
            'quality_percent': None if point['primary_mean'] is None else 100*point['primary_mean'],
            'CI95_percent': None if point['ci95'] is None else [100*x for x in point['ci95']],
            'mean_complete_bytes': capacity.get('mean'), 'max_complete_bytes': capacity.get('max'),
            'mean_actual_complete_bits': bits.get('mean'), 'max_actual_complete_bits': bits.get('max')})
    with (Path(destination)/'quality_complete_bytes.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(table)
    write(Path(destination)/'instrumented_timing_separate.json', {
        'controlled_speed_or_HBM_traffic_benchmark': False,
        'scope': 'Instrumented natural-output quality-run timings, with owner and diagnostic synchronization.',
        'rows': {p['dataset']+'__'+p['mode']: p['timing'] for p in points}})
    require(check_pins() == pins, 'extension helper changed during harvest')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plans', nargs='+', type=Path, required=True)
    parser.add_argument('--refs-root', type=Path, default=Path('${EVIDENCE_ROOT}/results/round10/input_preparation_v1/inputs'))
    parser.add_argument('--output-root', type=Path, default=Path('${EVIDENCE_ROOT}/results/round10/extension_harvest_v1'))
    parser.add_argument('--summary-output', type=Path, required=True); parser.add_argument('--fetch', action='store_true'); a = parser.parse_args()
    require(a.output_root.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve()) and
            a.summary_output.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve()), 'heavy extension output must use E')
    check_pins(); items, origins, selected = load_plans(a.plans)
    if a.fetch:
        from remote_control import connect
        client = connect()
        try:
            for item in items.values():
                print(json.dumps(fetch_cell(client, item, a.output_root/'raw'/item['cell']['name']), ensure_ascii=False), flush=True)
        finally:
            client.close()
    result = summarize(items, origins, selected, a.refs_root, a.output_root, a.summary_output)
    print(json.dumps({'status': result['status'], 'closed_cells': result['closed_cells'], 'expected_cells': result['expected_cells']}), flush=True)


if __name__ == '__main__':
    main()
