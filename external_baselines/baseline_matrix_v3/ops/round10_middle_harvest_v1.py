"""Read-only middle-pair harvest, with exact parent/source/raw/score bindings.

Reuses the frozen primary harvest's retrieval and pairing machinery. The scope
is exactly two D160 middle candidates plus the three original D160 references:
20 cells, 800 answer events of which 320 are new middle-pair GPU answers.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'code')]
import round10_harvest_score_v1 as frozen
import round10_runner_v1 as original_runner
import round10_middle_runner_v1 as middle

FROZEN_HARVEST_SHA = '1401b56ff23d28d5a88f9fd81514ee4633f5c28bdb90b16dc9627e1cc239f789'
DOMAINS, COUNTS = frozen.DOMAINS, frozen.COUNTS
CANDIDATES = middle.MODES
REFERENCES = ('FP16', 'R7low', 'R7high')
MODES = REFERENCES + CANDIDATES
PINS = {**frozen.PINS, 'ops/round10_harvest_score_v1.py': FROZEN_HARVEST_SHA}
require, sha, read, write = frozen.require, frozen.sha, frozen.read, frozen.write
stats, context, identity, paired = frozen.stats, frozen.context, frozen.identity, frozen.paired
ORIGINAL_BUDGET_GUARD = original_runner.round10_budget_guard

# The frozen inspector retains every parent PID/argv/config/snapshot/inventory,
# independent remote hash, exact START_CELL and raw3 check. Only the two
# approved script/mode bindings change. No remote mutation is introduced.
_old_script_assert = "assert cell['script']=='code/round10_runner_v1.py' and cfg['method']['family']=='trajkv_round10_research'"
_new_script_assert = """assert cfg['method']['family']=='trajkv_round10_research'
assert (cell['mode'] in ('FP16','R7low','R7high') and cell['script']=='code/round10_runner_v1.py') or (cell['mode'] in ('V128-R43','AddOldOrder') and cell['script']=='code/round10_middle_runner_v1.py')"""
require(frozen.INSPECT.count(_old_script_assert) == 1, 'frozen inspector hook changed')
INSPECT = frozen.INSPECT.replace(_old_script_assert, _new_script_assert)


def check_pins():
    require(sha(ROOT / 'ops/round10_harvest_score_v1.py') == FROZEN_HARVEST_SHA,
            'frozen harvest changed')
    frozen.check_pins()
    return {**PINS, 'code/round10_middle_runner_v1.py': sha(ROOT / 'code/round10_middle_runner_v1.py'),
            'code/round10_middle_bank_v1.py': sha(ROOT / 'code/round10_middle_bank_v1.py')}


def load_plans(paths):
    """Accept only original reference cells and new middle-pair cells."""
    result, origins = {}, []
    for path in paths:
        plan = read(path)
        require(plan['schema'] == 'trajkv.round10.job_plan.v1' and
                plan['phase'] in ('references', 'middle', 'candidates') and
                plan['status'] == 'FROZEN_PRE_LAUNCH' and
                plan['new_GPU_jobs_started_by_this_preparer'] is False,
                'only frozen declared free-generation plans')
        declared_modes = REFERENCES if plan['phase'] == 'references' else CANDIDATES
        origins.append({'path': str(path), 'sha256': sha(path), 'phase': plan['phase']})
        found = []
        for producer in plan['jobs']:
            jobpath = ROOT / producer['job']
            job = read(jobpath)
            require(sha(jobpath) == producer['job_sha256'] and job['name'] == producer['name'],
                    'parent job differs')
            seqremote = job['argv'][job['argv'].index('--config') + 1]
            require('/configs/round10_v1/' in seqremote and
                    job['argv'][2].endswith('/code/round10_sequence_v1.py'), 'wrong parent producer')
            sequencepath = ROOT / 'configs/round10_v1' / PurePosixPath(seqremote).name
            sequence = read(sequencepath)
            require(sequence['schema'] == 'trajkv.round10.sequence.v1', 'wrong sequence')
            for cell in sequence['entries']:
                script = ('code/round10_runner_v1.py' if cell['mode'] in REFERENCES else
                          'code/round10_middle_runner_v1.py')
                require(cell in plan['cells'] and cell['mode'] in declared_modes and cell['script'] == script and
                        cell['script_sha256'] == sha(ROOT / script), 'undeclared/different child source')
                if cell['mode'] in REFERENCES:
                    require(cell['script_sha256'] == PINS[script], 'reference driver changed')
                key = (cell['dataset'], cell['mode'])
                require(key not in result and cell['dataset'] in DOMAINS and
                        cell['expected_QA'] == COUNTS[cell['dataset']], 'duplicate/partial/wrong domain cell')
                result[key] = {'cell': cell, 'job': job, 'job_sha256': producer['job_sha256'],
                    'sequence_sha256': sha(sequencepath),
                    'sequence_script_sha256': sha(ROOT / 'code/round10_sequence_v1.py'),
                    'input_manifest_sha256': plan['input_manifest_sha256']}
                found.append(cell)
        require(len(found) == len(plan['cells']) and
                {x['name'] for x in found} == {x['name'] for x in plan['cells']}, 'plan cell lacks one parent')
    return result, origins


def inspect_remote(client, request):
    import shlex
    from remote_control import REMOTE_ROOT
    stdin, stdout, stderr = client.exec_command(
        shlex.join(['/usr/bin/python3', '-B', '-c', INSPECT]), timeout=60)
    stdin.write(json.dumps({'root': REMOTE_ROOT, **request}))
    stdin.flush(); stdin.channel.shutdown_write()
    body, error = stdout.read(), stderr.read()
    require(stdout.channel.recv_exit_status() == 0,
            'remote read-only inspection failed: ' + error.decode(errors='replace')[-2500:])
    return json.loads(body)


def fetch_cell(client, item, destination):
    # fetch_cell resolves inspect_remote in its own globals; set this one
    # explicitly while preserving its exact immutable retrieval implementation.
    previous = frozen.inspect_remote
    frozen.inspect_remote = inspect_remote
    try:
        return frozen.fetch_cell(client, item, destination)
    finally:
        frozen.inspect_remote = previous


def check_local_proof(raw, item):
    m, c, cfg, proof = frozen.check_local_proof(raw, item)
    cell = item['cell']
    require(proof['actual_driver'] == cell['script'] and
            m['source_hashes'][cell['script']] == cell['script_sha256'], 'actual driver source differs')
    if cell['mode'] in CANDIDATES:
        require(m['source_hashes'] == middle.runtime_source_manifest(),
                'middle executed source graph differs from frozen local sources')
        contract = m['middle_deployment_contract']
        require(contract['schema'] == 'trajkv.round10.middle_pair.v1' and
                contract['base_driver_sha256'] == middle.BASE_RUNNER_SHA and
                contract['base_variant_is_encoding_lineage_only'] is True and
                m['frozen_base_lineage_numeric_fields_unchanged'] is True and
                m['frozen_low43_profile_is_current_deployment_proof'] is False,
                'middle deployment and original-profile lineage conflated')
    return m, c, cfg, proof


def closed_cell(raw, scored, item, refs_root):
    cell = item['cell']; n = cell['expected_QA']
    missing = [x for x in ('manifest.json', 'records.jsonl', 'complete.json',
        'source_retrieval_receipt_round10_v1.json', 'retrieval_receipt.json') if not (raw / x).is_file()]
    empty = {'dataset': cell['dataset'], 'mode': cell['mode'], 'name': cell['name'],
        'status': 'PENDING_REAL_WHOLE_DOMAIN_SOURCE_SCORE', 'expected_QA': n,
        'primary_mean': None, 'ci95': None, 'resources': None, 'timing': None, 'missing': missing}
    if missing:
        return empty, None
    m, c, cfg, proof = check_local_proof(raw, item)
    refs = Path(refs_root) / PurePosixPath(cfg['inputs']).name.replace('_inputs.jsonl', '_refs.jsonl')
    require(sha(refs.with_name(refs.name.replace('_refs.jsonl', '_inputs.jsonl'))) ==
            m['inputs_sha256'] == sha(raw / 'source_inputs.jsonl'), 'local reference input mirror differs')
    import score
    if not (scored / 'summary.json').is_file():
        score.score_run(raw, refs, scored)
    s = read(scored / 'summary.json')
    require(s['integrity']['scorer_sha256'] == PINS['code/score.py'] and s['expected_run_questions'] == n and
            s['integrity']['raw_files'] == {k: sha(raw / k) for k in ('manifest.json', 'records.jsonl', 'complete.json')},
            'score/raw/scorer binding differs')
    require(s['integrity']['refs_sha256'] == sha(refs) and
            s['integrity']['dataset_source_lock_sha256'] == sha(Path(refs_root) / 'source_lock.json'),
            'frozen reference source lock changed')
    for name, digest in s['derived_file_hashes'].items():
        require(sha(scored / name) == digest, 'scored rows changed')
    if c['status'] != 'PASS_RAW_COMPLETE' or s['status'] != 'PASS_SCORED_COMPLETE':
        require(s['primary_mean'] is None, 'failure must not create success-subset primary')
        empty.update(status='CLOSED_WITH_FAILURES_NO_PRIMARY', missing=['all_expected_QA_PASS'],
                     raw_status=c['status'], score_status=s['status'])
        return empty, None
    require(proof['parent_cell_success'] is True,
            'successful closure requires actual parent exit0 and committed complete binding')
    from prepare_remaining_phase_delivery_v1 import physical_complete_bytes
    rawrows, bits, capacity, budgetrows = [], [], [], []
    scorerows = {r['id']: r for r in map(json.loads, (scored / 'scored_records.jsonl').read_text(encoding='utf-8').splitlines())}
    for number, line in enumerate((raw / 'records.jsonl').open('rb'), 1):
        r = json.loads(line); sr = scorerows[r['id']]
        require(r['status'] == 'PASS' and sr['score_status'] == 'SCORED_PASS' and
                sr['raw_jsonl_line_number'] == number and
                sr['raw_jsonl_exact_line_sha256'] == hashlib.sha256(line).hexdigest(), 'exact raw/score line differs')
        actual = physical_complete_bytes(r, m['model_spec'])
        capacity.append(actual); bits.append(r['actual_effective_visual_bits'])
        require(math.isclose(bits[-1], 16 * actual / r['full_kv_byte_denominator'], rel_tol=1e-12, abs_tol=1e-12) and
                sr['stored_visual_KV_bytes'] == actual and sr['actual_effective_visual_bits'] == bits[-1],
                'physical/raw/scored bytes or denominator differs')
        if cell['mode'] in CANDIDATES:
            budget = middle.round10_budget_guard(cfg['candidate']['mode'], r['actual_visual_token_count'],
                                                actual, r['r7_bridge'], cfg['variant_source']['name'])
            require(r['round10_budget_vs_original_R7_low'] == budget, 'stored middle physical guard differs')
            budgetrows.append({'id': r['id'], **budget})
        else:
            ORIGINAL_BUDGET_GUARD(cfg['candidate']['mode'], r['actual_visual_token_count'],
                                  actual, r['r7_bridge'], cfg['variant_source']['name'])
        require(r['owner_after_generation']['stored_visual_KV_bytes'] == actual and
                r['original_full_prefix_released'] is True and r['original_FP16_visual_alias_retained'] is False,
                'native prefix retained or persistent owner changed')
        require(r['question_unknown_prefix_verified'] is True and
                r['first_generation_call']['native_first_logits_injected'] is False,
                'question-unknown or candidate suffix evidence differs')
        rawrows.append(r)
    require(len(rawrows) == n and s['metric_ids'] == [cfg['evaluation_contract']['primary_metric_id']],
            'wrong full-domain score or metric')
    timing = {key: stats([r[key] for r in rawrows if r.get(key) is not None]) for key in
        ('prefill_ms', 'encode_ms', 'question_prefill_ms', 'ttft_ms', 'wall_ms',
         'output_tokens', 'peak_allocated_bytes', 'peak_reserved_bytes')}
    timing['pooled_natural_inter_token_ms'] = stats([v for r in rawrows for v in r['inter_token_ms']])
    point = {**empty, 'status': 'CLOSED_FULL_DOMAIN_CUSTOM_SOURCE_SCORE_PHYSICAL', 'missing': [],
        'primary_mean': s['primary_mean'], 'ci95': s['primary_ci95'],
        'resources': {'complete_bytes': stats(capacity), 'actual_complete_bits': stats(bits),
            'strict_all_QA_1p3': all(160*b <= 13*r['full_kv_byte_denominator'] for b, r in zip(capacity, rawrows)),
            'strict_all_QA_1p8': all(80*b <= 9*r['full_kv_byte_denominator'] for b, r in zip(capacity, rawrows)),
            'middle_dynamic_formula_records': budgetrows},
        'timing': timing, 'raw': str(raw), 'scored': str(scored),
        'summary_sha256': sha(scored / 'summary.json'),
        'source_receipt_sha256': sha(raw / 'source_retrieval_receipt_round10_v1.json'),
        'SourceRun_admission_established': False}
    bundle = {'context': context(m), 'identity': {r['id']: identity(r) for r in rawrows},
              'scores': scorerows, 'scored': scored, 'raw': {r['id']: r for r in rawrows}}
    return point, bundle


def summarize(items, origins, refs_root, output_root, destination):
    require(not Path(destination).exists(), 'new summary directory required')
    source_pins = check_pins()
    points, bundles = [], {}
    for ds in DOMAINS:
        for mode in MODES:
            key = (ds, mode)
            if key not in items:
                points.append({'dataset': ds, 'mode': mode, 'status': 'PENDING_DECLARED_PLAN_NOT_READY',
                    'expected_QA': COUNTS[ds], 'primary_mean': None, 'ci95': None,
                    'resources': None, 'timing': None})
                continue
            item = items[key]; name = item['cell']['name']
            point, bundle = closed_cell(Path(output_root) / 'raw' / name,
                                       Path(output_root) / 'scored' / name, item, refs_root)
            points.append(point)
            if bundle is not None:
                bundles[key] = bundle
    pairs = {}; checked_middle_pair_domains = []
    contrasts = [(c, a) for c in CANDIDATES for a in REFERENCES] + [('AddOldOrder', 'V128-R43')]
    for ds in DOMAINS:
        for candidate, anchor in contrasts:
            key = ds + '__' + candidate + '_minus_' + anchor
            pairs[key] = (paired(bundles[ds, anchor], bundles[ds, candidate])
                          if (ds, anchor) in bundles and (ds, candidate) in bundles else None)
        if (ds, 'V128-R43') in bundles and (ds, 'AddOldOrder') in bundles:
            left, right = bundles[ds, 'V128-R43'], bundles[ds, 'AddOldOrder']
            require(left['identity'] == right['identity'] and left['context'] == right['context'],
                    'middle pair input/positions/runtime differs')
            for key, lrow in left['raw'].items():
                rrow = right['raw'][key]; n = lrow['actual_visual_token_count']
                budget = middle.middle_budget(n, 'V128-R43')
                require(lrow['stored_visual_KV_bytes'] - rrow['stored_visual_KV_bytes'] ==
                        budget['V128_minus_AddOldOrder_bytes'] < 18*n,
                        'actual middle pair dynamic byte gap differs')
            checked_middle_pair_domains.append(ds)
    closed = sum(p['primary_mean'] is not None for p in points)
    result = {'schema': 'trajkv.round10.middle_development_summary.v1',
        'status': ('PASS_ALL20_D160_MIDDLE_CUSTOM_SOURCE_SCORE_PHYSICAL' if closed == 20 else
                   'PENDING_OR_PARTIAL_MIDDLE_D160_NOT_COMPLETE'),
        'expected_cells': 20, 'closed_cells': closed, 'expected_answer_events': 800,
        'expected_new_middle_answer_events': 320, 'reused_reference_answer_events': 480,
        'closed_answer_events': sum(p['expected_QA'] for p in points if p['primary_mean'] is not None),
        'expected_distinct_development_sources': 160,
        'actual_closed_unique_IDs': len({key for b in bundles.values() for key in b['identity']}),
        'plan_sources': origins, 'points': points, 'paired_contrasts': list(pairs),
        'no_cross_dataset_quality_average': True, 'D160_is_development_not_independent_confirmation': True,
        'paired_CI_and_signflip_unadjusted_not_equivalence_or_algorithm_winner': True,
        'middle_pair_checked_domains': checked_middle_pair_domains,
        'middle_pair_same_input_positions_runtime_checked': len(checked_middle_pair_domains) == 4,
        'middle_pair_exact_byte_equality_claimed': False,
        'middle_pair_dynamic_gap_less_than_one_18N_residual_unit_checked': len(checked_middle_pair_domains) == 4,
        'quality_resource_and_timing_are_separate': 'All-QA complete bytes; instrumented quality-run host timing with natural output lengths, not controlled speed or HBM traffic.',
        'SourceRun_admission_established': False, 'new_GPU_jobs_started': 0,
        'source_sha256': sha(__file__), 'helper_sha256': source_pins}
    Path(destination).mkdir(parents=True)
    write(Path(destination) / 'summary.json', result)
    write(Path(destination) / 'paired_comparisons.json', pairs)
    table = []
    for p in points:
        resources = p['resources'] or {}; capacity = resources.get('complete_bytes') or {}
        bits = resources.get('actual_complete_bits') or {}
        table.append({'dataset': p['dataset'], 'mode': p['mode'], 'status': p['status'], 'QA': p['expected_QA'],
            'quality_percent': None if p['primary_mean'] is None else 100*p['primary_mean'],
            'CI95_percent': None if p['ci95'] is None else [100*x for x in p['ci95']],
            'mean_complete_bytes': capacity.get('mean'), 'max_complete_bytes': capacity.get('max'),
            'mean_actual_complete_bits': bits.get('mean'), 'max_actual_complete_bits': bits.get('max'),
            'strict_all_QA_1p3': resources.get('strict_all_QA_1p3'),
            'strict_all_QA_1p8': resources.get('strict_all_QA_1p8')})
    with (Path(destination) / 'quality_complete_bytes.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(table)
    write(Path(destination) / 'instrumented_timing_separate.json', {
        'scope': 'Prototype quality-run host timings, natural token lengths, diagnostic synchronization included.',
        'controlled_speed_or_HBM_traffic_benchmark': False, 'cross_dataset_speed_ranking_claimed': False,
        'rows': {p['dataset'] + '__' + p['mode']: p['timing'] for p in points}})
    require(check_pins() == source_pins, 'helper source changed during harvest')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plans', nargs='+', type=Path, required=True)
    p.add_argument('--refs-root', type=Path, default=Path('${EVIDENCE_ROOT}/results/round10/input_preparation_v1/inputs'))
    p.add_argument('--output-root', type=Path, default=Path('${EVIDENCE_ROOT}/results/round10/middle_harvest_v1'))
    p.add_argument('--summary-output', type=Path, required=True)
    p.add_argument('--fetch', action='store_true'); a = p.parse_args()
    require(a.output_root.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve()) and
            a.summary_output.resolve().is_relative_to(Path('${EVIDENCE_ROOT}').resolve()), 'heavy outputs must use E')
    check_pins(); items, origins = load_plans(a.plans)
    if a.fetch:
        from remote_control import connect
        client = connect()
        try:
            for item in items.values():
                print(json.dumps(fetch_cell(client, item, a.output_root/'raw'/item['cell']['name']),
                                 ensure_ascii=False), flush=True)
        finally:
            client.close()
    result = summarize(items, origins, a.refs_root, a.output_root, a.summary_output)
    print(json.dumps({'status': result['status'], 'closed_cells': result['closed_cells'],
                      'expected_cells': 20}), flush=True)


if __name__ == '__main__':
    main()
