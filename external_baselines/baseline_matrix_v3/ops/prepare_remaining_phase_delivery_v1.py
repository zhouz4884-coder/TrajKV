"""Local CPU delivery gates for complete P2-114, canonical P3-20 and speed560.

No remote access, credentials, inference, queue writes, archive copies or Git.
Every invocation writes a new small snapshot under E; default is readiness only.
--verify-complete refuses incomplete scopes before any deep scan. Existing
public_phase_contract_v2, exporter_v3 and publisher_v5 remain the delivery path.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('${EVIDENCE_ROOT}').resolve()
sys.path.insert(0, str(ROOT/'code'))
from public_phase_contract_v2 import load, sha, require, resolve, validate_phase, source_closure

P2_SOURCES = [
    ('core40', 'configs/matrix/P2_harvest_targets_core_v1.json', 40,
     'results/harvest/P2_core_v1/state.json', 'results/p2_views/core_v1', 'results/scored/p2_views_core_v1'),
    ('budget16', '${EVIDENCE_ROOT}/results/extension_execution/budget_extension_v1/ledger_v1.json', 16,
     'results/extension_execution/budget_extension_v1/harvest/state.json', 'results/extension_views/budget_extension_v1', 'results/scored/extension_budget_extension_v1'),
    ('source_compat42', '${EVIDENCE_ROOT}/results/extension_execution/source_compat_v1/ledger_v11.json', 42,
     'results/extension_execution/source_compat_v1/harvest/state.json', 'results/extension_views/source_compat_v1', 'results/scored/extension_source_compat_v1'),
    ('M1_completion4', 'configs/matrix/P2_M1_representative_completion_v1.json', 4,
     'results/extension_execution/m1_representative_completion_v1/harvest/state.json', 'results/extension_views/m1_representative_completion_v1', 'results/scored/extension_m1_representative_completion_v1'),
    ('low12', 'configs/matrix/P2_low_budget_completion_12cells_v1.json', 12,
     'results/extension_execution/low_budget_completion_v1/harvest/state.json', 'results/extension_views/low_budget_completion_v1', 'results/scored/extension_low_budget_completion_v1')]
P3 = 'configs/matrix/P3_M1_representative_targets_v3.json'
P3_SHA = '59efc359c35e4af7ad5409bfedfb922a6eb5dbabcfe4361a32043ef77b0ca0fb'
SPEED_RUN = 'm1_canonical_source_representatives_v2'
SPEED_EXPECTED = {('textvqa', 'hardware_smoke'): (2, 40), ('textvqa', 'formal_fixed_cohort'): (16, 320),
                  ('infovqa', 'hardware_smoke'): (2, 40), ('infovqa', 'formal_fixed_cohort'): (8, 160)}
DTYPE_BYTES = {'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8,
               'torch.int64': 8, 'torch.int32': 4, 'torch.int16': 2, 'torch.int8': 1, 'torch.uint8': 1, 'torch.bool': 1}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2); stream.write('\n')


def latest(folder):
    candidates = [(int(m.group(1)), p) for p in folder.glob('ledger_v*.json')
                  if (m := re.fullmatch(r'ledger_v(\d+)\.json', p.name))]
    require(candidates, 'No immutable execution ledger')
    return max(candidates)[1]


def state(path):
    return load(path) if path.is_file() else {'cells': {}}


def target(row, group, cell=None, view=None, scored=None):
    config = row.get('full_target_config') or row.get('config') or row.get('target_config')
    cfg = resolve(config); value = load(cfg)
    require(value['model_id'] == row['model'], 'Wrong model in declared config')
    if row.get('config_sha256'): require(sha(cfg) == row['config_sha256'], 'Frozen target config changed')
    count = int(row.get('count', row.get('target_QA', row.get('expected_QA', 0))))
    require(count > 0, 'Target count absent')
    cell = cell or {}; stem = cfg.stem
    return {'model': row['model'], 'method': row['method'], 'dataset': row['dataset'], 'count': count,
            'config': cfg.relative_to(ROOT).as_posix(), 'config_sha256': sha(cfg), 'partition': group,
            'frozen_declaration_status': row.get('status', 'DECLARED_TARGET'),
            'view': str(resolve(cell.get('view') or view/stem)),
            'scored': str(resolve(cell.get('scored') or scored/stem))}


def quality_matrix(which):
    rows = []; inputs = []
    if which == 'p2':
        for group, source, number, spath, views, scores in P2_SOURCES:
            path = resolve(source); plan = load(path); require(len(plan['rows']) == number, 'Partition count changed')
            cells = state(EVIDENCE/spath)['cells']; inputs.append({'path': str(path), 'sha256': sha(path)})
            for row in plan['rows']:
                config = row.get('full_target_config') or row.get('config') or row.get('target_config')
                rows.append(target(row, group, cells.get(Path(config).stem), EVIDENCE/views, EVIDENCE/scores))
        require(len(rows) == 114 and all(r['count'] == 500 for r in rows), 'P2 must retain all114 x500, including conditional low8')
        label = 'P2_COMPLETE_114_FIXED500'
    else:
        path = ROOT/P3; require(sha(path) == P3_SHA, 'Canonical full declaration changed')
        plan = load(path); require(len(plan['rows']) == 20, 'Full scope must retain all20')
        inputs.append({'path': str(path), 'sha256': sha(path)})
        full = state(EVIDENCE/'results/full_execution/m1_representative_v3/harvest_v3/state.json')['cells']
        native = load(EVIDENCE/'results/harvest/native_F_v1/state.json')['full_cells']
        for row in plan['rows']:
            is_native = row['method'] == 'F'
            cells = native.get(row['dataset'], {}) if is_native else full.get(Path(row['config']).stem, {})
            rows.append(target(row, 'existing_native4' if is_native else 'canonical_compressed16', cells,
                               EVIDENCE/'results/representative_full_views/m1_representative_v3',
                               EVIDENCE/'results/scored/representative_full_m1_representative_v3'))
        require(sum(r['count'] for r in rows) == 78250, 'Full official scope differs')
        require(sum(r['count'] for r in rows if r['method'] != 'F') == 62600, 'Compressed full scope differs')
        label = 'P3_M1_CANONICAL_FIVE_PROFILES_ALL4_FULL'
    require(len({r['config'] for r in rows}) == len(rows), 'Duplicated target config')
    require(len({(r['model'], r['method'], r['dataset']) for r in rows}) == len(rows), 'Duplicated logical target')
    return {'schema': 'baseline_matrix.fixed_matrix.v1', 'phase': label, 'rows': rows,
            'expected_cells': len(rows), 'expected_QA': sum(r['count'] for r in rows),
            'metrics_not_mixed': True, 'source_declarations': inputs,
            'scope': 'Complete declared matrix only; prediction views do not count as new GPU inference; no speed or whole-library completion claim.'}


def readiness(matrix):
    statuses = []; partitions = Counter()
    for row in matrix['rows']:
        raw = Path(row['view']); scored = Path(row['scored']); missing = []
        if not (raw/'complete.json').is_file(): missing.append('closed_raw_or_view')
        else:
            value = load(raw/'complete.json')
            if value.get('status') != 'PASS_RAW_COMPLETE': missing.append('all_raw_QA_PASS')
            if value.get('expected_QA') != row['count'] or value.get('recorded_QA') != row['count']:
                missing.append('complete_raw_expected_and_recorded_count')
        if not (scored/'summary.json').is_file(): missing.append('scored_summary')
        else:
            value = load(scored/'summary.json')
            if value.get('status') != 'PASS_SCORED_COMPLETE' or value.get('expected_run_questions') != row['count']:
                missing.append('all_scored_QA_PASS')
        if not missing: partitions[row['partition']] += 1
        statuses.append({'cell': Path(row['config']).stem, 'partition': row['partition'], 'missing': missing})
    return statuses, dict(partitions)


def physical_complete_bytes(raw, spec):
    owner = raw['owner_before_suffix']; tensors = owner['tensors']
    require(tensors, 'Missing physical owner table')
    for name, item in tensors.items():
        require(math.prod(item['shape'])*DTYPE_BYTES[item['dtype']] == item['logical_bytes'] <= item['storage_bytes'], 'Physical capacity/shape differs')
        require(item['charged_bytes'] in (0, item['storage_bytes']), 'Invalid unique physical charge')
        if item['shared_with'] is None:
            require(item['charged_bytes'] == item['storage_bytes'], 'Unshared physical owner was not fully charged')
        else:
            first = tensors[item['shared_with']]
            require(first['shared_with'] is None and first['charged_bytes'] == first['storage_bytes'] and
                    item['charged_bytes'] == 0 and item['device'] == first['device'] and item['storage_bytes'] == first['storage_bytes'], 'Unbound, cyclic or indirect shared physical owner')
    require(sum(item['charged_bytes'] for item in tensors.values()) == owner['total_bytes'], 'Physical owner charge sum differs')
    denominator = spec['layers']*2*spec['kv_heads']*raw['actual_visual_token_count']*spec['head_dim']*2
    require(raw['full_kv_byte_denominator'] == denominator, 'Native geometry denominator differs')
    if owner['accounting_scope'] == 'native_full_allocation_equals_disjoint_visual_plus_text_regions':
        require(all(item['charged_bytes'] == item['storage_bytes'] == item['logical_bytes'] for name, item in tensors.items() if name.startswith('layers.')), 'Native KV aliases or hides backing capacity')
        require(owner['stored_visual_payload_bytes'] == denominator, 'Native disjoint visual charge differs')
        actual = denominator+sum(item['charged_bytes'] for name, item in tensors.items() if name.startswith('common.'))
    else:
        require(owner['accounting_scope'] == 'global_unique_storages_across_all_layers_and_shared_constants', 'Unknown owner accounting scope')
        actual = sum(item['charged_bytes'] for name, item in tensors.items() if '.visual.' in name or name == 'original_visual_positions' or name.startswith('common.'))
    require(actual == raw['complete_visual_owner_bytes'] == raw['stored_visual_KV_bytes'] == owner['complete_visual_owner_bytes'] > 0, 'Complete visual bytes differ from physical owners')
    require(owner['total_bytes'] == actual+owner['text_kv_bytes']+owner['text_positions_bytes']+owner.get('other_persistent_metadata_bytes', 0), 'Physical owner component balance differs')
    return actual


def quality_aggregate(matrix, artifacts, out):
    from compare_runs import compare, load_scored
    loaded = {}; points = []; by_group = defaultdict(list)
    for row, artifact in zip(matrix['rows'], artifacts):
        summary, scored_rows = load_scored(artifact['scored']); complete_bytes = []; bits = []
        spec = load(artifact['raw']/'manifest.json')['model_spec']
        with (artifact['raw']/'records.jsonl').open('rb') as stream:
            for number, line in enumerate(stream, 1):
                raw = json.loads(line); score = scored_rows[raw['id']]
                require(score['raw_jsonl_line_number'] == number and score['raw_jsonl_exact_line_sha256'] == hashlib.sha256(line).hexdigest(), 'Resource row detached from original')
                actual = physical_complete_bytes(raw, spec)
                require(actual == score['stored_visual_KV_bytes'], 'Scored/physical complete owner bytes differ')
                require(raw['actual_effective_visual_bits'] == score['actual_effective_visual_bits'], 'Scored/raw bits differ')
                require(abs(raw['actual_effective_visual_bits']-16*actual/raw['full_kv_byte_denominator']) < 1e-12, 'Bits denominator differs')
                complete_bytes.append(actual); bits.append(raw['actual_effective_visual_bits'])
        require(len(complete_bytes) == row['count'], 'Resources must retain complete cohort')
        point = {'cell': artifact['cell'], 'model': row['model'], 'dataset': row['dataset'], 'method': row['method'],
                 'QA': row['count'], 'metric_ids': summary['metric_ids'], 'quality': summary['primary_mean'],
                 'quality_CI95': summary['primary_ci95'], 'groups': summary['groups'],
                 'complete_visual_owner_mean_bytes': sum(complete_bytes)/len(complete_bytes),
                 'actual_effective_bits_mean': sum(bits)/len(bits),
                 'actual_effective_bits_min': min(bits), 'actual_effective_bits_max': max(bits),
                 'no_budget_or_quality_subset_selection': True,
                 'all_QA_iso_budget': {str(b): all(v <= b for v in bits) for b in (1.3, 1.8, 2.25)},
                 'raw': str(artifact['raw']), 'scored': str(artifact['scored']), 'summary_sha256': sha(artifact['scored']/'summary.json')}
        loaded[(row['model'], row['dataset'], row['method'])] = artifact; points.append(point)
        by_group[(row['model'], row['dataset'], tuple(summary['metric_ids']))].append(point)
    for key, group in by_group.items():
        anchor = loaded[(key[0], key[1], 'F')]
        for point in group:
            candidate = loaded[(point['model'], point['dataset'], point['method'])]
            paired = compare(anchor['scored'], candidate['scored'])
            require(paired['status'] == 'PASS_PAIRED_COMPLETE', 'Paired quality not complete')
            path = out/'paired_F'/(point['cell']+'.json'); write(path, paired)
            point['paired_F'] = str(path); point['paired_F_sha256'] = sha(path)
            x = point['complete_visual_owner_mean_bytes']; y = point['quality']
            point['descriptive_pareto_frontier_within_model_dataset_metric'] = not any(
                q['complete_visual_owner_mean_bytes'] <= x and q['quality'] >= y and
                (q['complete_visual_owner_mean_bytes'] < x or q['quality'] > y) for q in group)
    write(out/'quality_bytes.json', {'schema': 'baseline_matrix.complete_quality_bytes.v1', 'rows': points,
          'no_cross_dataset_metric_average': True, 'no_wall_speed_inference': True,
          'frontier_is_descriptive_not_significance_or_equivalence': True})
    fields = ['cell', 'model', 'dataset', 'method', 'QA', 'quality', 'quality_CI95', 'metric_ids',
              'complete_visual_owner_mean_bytes', 'actual_effective_bits_min', 'actual_effective_bits_max',
              'actual_effective_bits_mean', 'all_QA_iso_budget', 'descriptive_pareto_frontier_within_model_dataset_metric', 'paired_F_sha256']
    with (out/'quality_bytes.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        writer.writerows({key: point[key] for key in fields} for point in points)


def quality(which, out, verify):
    matrix = quality_matrix(which); mpath = out/'declared_matrix.json'; write(mpath, matrix)
    statuses, partitions = readiness(matrix); ready = all(not r['missing'] for r in statuses)
    contract = {'schema': 'baseline_matrix.public_phase_contract.v2', 'phase_label': matrix['phase'],
                'family': which+'_complete_declared', 'matrix': str(mpath), 'coverage_scope': matrix['scope'],
                'completion_policy': 'all_expected_QA_pass', 'completion_scope': 'declared_matrix_only',
                'whole_library_completion_assessed': False}
    # Per-row explicit view/scored paths are authoritative; no aliases or copies.
    write(out/'phase_contract.json', contract)
    report = {'status': 'READY_FOR_DEEP_CPU_VALIDATION' if ready else 'WAIT_DECLARED_MATRIX_INCOMPLETE',
              'expected_cells': len(statuses), 'closed_marker_and_score_cells': sum(not r['missing'] for r in statuses),
              'expected_view_QA': matrix['expected_QA'], 'partition_closed_counts': partitions, 'cells': statuses,
              'publication_eligible': False, 'whole_external_library_complete': False, 'deep_evidence_verified': False}
    if verify and ready:
        phase, artifacts, _ = validate_phase(contract)
        require(phase['status'] == 'PASS_DECLARED_MATRIX_COMPLETE', 'Deep phase coverage failed')
        closure = source_closure(artifacts)
        write(out/'source_closure.json', [{'path': str(node['path']), 'kind': node['kind'],
              'files': {p.name: sha(p) for p in node['files']}} for node in closure])
        quality_aggregate(matrix, artifacts, out)
        supporting = [out/'source_closure.json', out/'quality_bytes.json', out/'quality_bytes.csv',
                      *sorted((out/'paired_F').glob('*.json')), Path(__file__)]
        write(out/'quality_support.json', {'schema': 'baseline_matrix.public_support_files.v1',
              'label': matrix['phase']+'_paired_quality_and_complete_bytes', 'completed_evidence_only': True,
              'primary_quality_QA_added': 0, 'quality_QA_added': 0,
              'scope': 'CPU recomputed paired complete-cohort comparisons and actual complete visual-owner bytes. Dataset metrics remain separate; descriptive frontier is not statistical significance, speed or extra GPU inference.',
              'files': [{'path': str(path.resolve()), 'sha256': sha(path), 'role': 'complete_quality_bytes_CPU_validation'} for path in supporting]})
        report.update(status='PASS_COMPLETE_DECLARED_QUALITY_PHASE_CPU', phase=phase,
                      deep_evidence_verified=True, publication_eligible=True,
                      unique_original_source_runs=sum(n['kind'] == 'original_inference_run' for n in closure))
    write(out/'readiness.json', report)
    return report


def timing_origin_phases(donors, out, primary_path):
    """Route large original donors through existing scored quality closures.

    exporter_v3 deliberately limits support-list entries to64MiB. Original
    records larger than that must enter via a declared phase source closure.
    A required auxiliary500 view adds no fresh inference or timing-quality QA.
    """
    primary_path = resolve(primary_path); primary = load(primary_path)
    phase, primary_artifacts, _ = validate_phase(primary)
    require(phase['status'] == 'PASS_DECLARED_MATRIX_COMPLETE', 'Timing must attach to a complete primary phase')
    covered = set(); covered_origins = {}
    def cover(nodes):
        for node in nodes:
            covered.update(str(path.resolve()).lower() for path in node['files'])
            if node['kind'] == 'original_inference_run':
                raw = node['path']; complete = load(raw/'complete.json')
                # source_closure has independently rehashed each original raw;
                # retain its three-file identity, not merely a path membership.
                covered_origins[str(raw.resolve()).lower()] = {
                    'manifest.json': sha(raw/'manifest.json'), 'records.jsonl': complete['records_jsonl_sha256'],
                    'complete.json': sha(raw/'complete.json')}
    cover(source_closure(primary_artifacts))
    primary_configs = {str(a['config'].resolve()).lower() for a in primary_artifacts}
    large = {str(Path(path).resolve()).lower() for path in donors
             if (Path(path)/'records.jsonl').stat().st_size > 64*1024**2 and
             str((Path(path)/'records.jsonl').resolve()).lower() not in covered}
    chosen = []; outstanding = set(large)
    for row in quality_matrix('p2')['rows']:
        if not outstanding: break
        if str(resolve(row['config'])).lower() in primary_configs: continue
        raw = Path(row['view']); scored = Path(row['scored'])
        if not (raw/'complete.json').is_file() or not (scored/'summary.json').is_file(): continue
        if load(raw/'complete.json').get('status') != 'PASS_RAW_COMPLETE' or load(scored/'summary.json').get('status') != 'PASS_SCORED_COMPLETE': continue
        manifest = load(raw/'manifest.json')
        origins = {str(resolve(d['raw_dir'])).lower() for d in manifest.get('source_runs', [])}
        origins.add(str(raw.resolve()).lower())
        if not origins & outstanding: continue
        chosen.append(row); outstanding -= origins
    require(not outstanding, 'WAIT: a >64MiB timing donor needs its own closed/scored fixed500 view; no raw truncation or oversized support entry allowed')
    contracts = [primary_path]; auxiliary = []
    if chosen:
        matrix_path = out/'timing_origin_support_matrix.json'
        write(matrix_path, {'schema': 'baseline_matrix.fixed_matrix.v1', 'phase': 'TIMING560_ORIGINAL_PREDICTION_SUPPORT',
              'rows': chosen, 'scope': 'Existing exact fixed500 quality views required only to carry complete original timing-prediction source closures. Not new GPU inference, extra timing QA, P2-114 completion or full official splits.'})
        contract = {'schema': 'baseline_matrix.public_phase_contract.v2', 'phase_label': 'TIMING560_REQUIRED_ORIGINAL_PREDICTION_SUPPORT',
                    'family': 'timing560_original_prediction_support', 'matrix': str(matrix_path),
                    'coverage_scope': load(matrix_path)['scope'], 'completion_policy': 'all_expected_QA_pass',
                    'completion_scope': 'declared_matrix_only', 'whole_library_completion_assessed': False}
        contract_path = out/'timing_origin_support_contract.json'; write(contract_path, contract)
        support_phase, artifacts, _ = validate_phase(contract)
        require(support_phase['status'] == 'PASS_DECLARED_MATRIX_COMPLETE', 'Required timing-origin support phase not complete')
        cover(source_closure(artifacts))
        contracts.append(contract_path); auxiliary.append(str(contract_path))
    require(all(str((Path(path)/'records.jsonl').resolve()).lower() in covered for path in large), 'Large timing original donor missing from selected phase closure')
    for path, donor in donors.items():
        if str(Path(path).resolve()).lower() in covered_origins:
            require(covered_origins[str(Path(path).resolve()).lower()] == donor.hashes, 'Selected timing-origin phase path has different three-file raw identity')
    return covered, {'primary_contract': str(primary_path), 'auxiliary_contracts': auxiliary,
                     'required_current_bundle_contracts': [{'path': str(path), 'sha256': sha(path)} for path in contracts],
                     'auxiliary_existing_fixed500_view_QA': sum(r['count'] for r in chosen),
                     'auxiliary_new_GPU_inference_QA': 0, 'timing_quality_QA_added': 0}


def timing(out, verify, proof_root, primary_contract):
    ledger_path = latest(EVIDENCE/'results/timing_execution'/SPEED_RUN); ledger = load(ledger_path)
    require(ledger['run_id'] == SPEED_RUN and ledger['requested_profile_trials'] == 560 and ledger['historical_E8_timing_not_reused'] is True, 'Wrong canonical timing profile')
    require(sha(ROOT/P3) == P3_SHA, 'Canonical full source profile changed')
    collector = state(EVIDENCE/'results/timing_harvest'/SPEED_RUN/'watcher/state.json')
    current = {}; missing = []; trials = 0; definitions = []
    for row in ledger['timing_rows']:
        cfg = load(ROOT/row['config']); job = load(ROOT/row['job'])
        require(sha(ROOT/row['job']) == row['job_sha256'], 'Timing job changed')
        count = cfg['selection']['count']; key = (row['dataset'], 'hardware_smoke' if count == 2 else 'formal_fixed_cohort')
        require(key in SPEED_EXPECTED and SPEED_EXPECTED[key] == (count, row['expected_trials']), 'Wrong canonical timing cohort')
        require(key not in current and cfg['repeat'] == 3 and cfg['warmup_per_qa'] == 1 and cfg['host_quiet_asserted'] is False, 'Duplicate timing cohort or repetition changed')
        methods = [load(ROOT/'configs'/str(p).replace('\\', '/').split('/configs/', 1)[1]) for p in cfg['eval_configs']]
        canonical = []
        for declared in load(ROOT/P3)['rows']:
            if declared['dataset'] != row['dataset']: continue
            path = ROOT/declared['full_target_config']
            require(sha(path) == declared['config_sha256'], 'Canonical full config fingerprint changed')
            canonical.append(load(path))
        require(sorted([c['method'] for c in methods], key=lambda m: m['id']) == sorted([c['method'] for c in canonical], key=lambda m: m['id']), 'Timing source-default five profiles differ')
        for actual in methods:
            frozen = next(c for c in canonical if c['method'] == actual['method'])
            for field in ('model_id', 'model_revision', 'model_path', 'model_source_lock', 'preprocessing', 'question_suffix', 'max_new_tokens'):
                require(actual[field] == frozen[field], 'Timing/full behavior differs: '+field)
        cell = collector['cells'].get(job['name'], {}); current[key] = (row, cell, job['name'])
        good = cell.get('status') == 'PASS_TOKEN_CONTEXT_OWNER_REPEAT_TIMING_AUDIT' and bool(cell.get('audit'))
        if not good: missing.append({'cohort': row['cohort'], 'missing': 'actual_PASS_raw_and_independent_audit'})
        else:
            trials += cell['raw_trials']
            if not (proof_root/job['name']/'retrieval_receipt.json').is_file():
                missing.append({'cohort': row['cohort'], 'missing': 'independent_executed_dependency_source_proof'})
        definitions.append({'dataset': key[0], 'step': key[1], 'QA_per_method': count, 'trials': row['expected_trials'],
                            'job': row['job'], 'job_sha256': row['job_sha256'], 'actual_PASS': good})
    for key in SPEED_EXPECTED.keys()-current.keys(): missing.append({'dataset': key[0], 'step': key[1], 'missing': 'actual_ledger_entry'})
    real_complete = trials == 560 and len(current) == 4 and all(d['actual_PASS'] for d in definitions)
    readiness_status = ('WAIT_CANONICAL_560_EXECUTED_DEPENDENCY_PROOF' if missing else
                        'READY_CANONICAL_560_FOR_DEEP_CPU_VALIDATION') if real_complete else 'WAIT_CANONICAL_560_REAL_PASS'
    report = {'status': readiness_status, 'expected_trials': 560, 'actual_PASS_trials': trials,
              'raw_trials_are_not_independent_QA': True, 'quality_QA_added': 0, 'legacy656_calib32_oldE8_excluded': True,
              'publication_eligible': False, 'ledger': str(ledger_path), 'ledger_sha256': sha(ledger_path),
              'definitions': definitions, 'missing': missing, 'host_quiet_asserted': False}
    if verify and not missing:
        require(trials == 560 and len(current) == 4, 'Exactly four actual canonical timing nodes required')
        from validate_speed_evidence import audit
        from prediction_cache import SourceRun
        files = {}; scalar = []; donors = {}; line_checks = 0; origin_checks = []
        def add(path, role):
            path = resolve(path); require(path.suffix.lower() in {'.json', '.jsonl', '.csv', '.py', '.md'}, 'Non-public timing evidence file type')
            files[str(path).lower()] = {'path': str(path), 'sha256': sha(path), 'role': role}
        add(ledger_path, 'immutable_canonical_execution_ledger')
        add(Path(__file__), 'canonical560_support_CPU_generator')
        for key, (row, cell, name) in current.items():
            raw = resolve(cell['raw']); result = audit(raw, out/'audit'/name, row)
            require(result['status'] == 'PASS_TOKEN_CONTEXT_OWNER_REPEAT_TIMING_AUDIT' and result['raw_trials'] == row['expected_trials'], 'Timing deep audit failed')
            for entry in load(raw/'speed_retrieval_receipt_v1.json')['files']: add(raw/entry['file'], 'timing_original_raw_or_source')
            add(raw/'speed_retrieval_receipt_v1.json', 'independent_remote_raw_hash_proof')
            proof = proof_root/name; proof_receipt = load(proof/'retrieval_receipt.json')
            require(proof_receipt['status'] == 'PASS_EXECUTED_SOURCE_INSPECTION' and proof_receipt['independent_remote_SHA'] is True and proof_receipt['job'] == name, 'Original executed timing dependency proof required')
            inventory = load(raw/'source_inventory.json'); expected = load(raw/'expected_predictions.json')
            selected = set(load(raw/'manifest.json')['source_hashes'])
            for context in expected['source_contracts_by_method'].values(): selected.update(context['relevant_source_hashes'])
            for filename in selected:
                filename = filename.replace('\\', '/')
                require(sha(proof/'source'/filename) == inventory[filename], 'Executed source dependency closure differs')
            add(proof/'retrieval_receipt.json', 'independent_executed_dependency_proof')
            for entry in proof_receipt['files']:
                require(sha(proof/entry['file']) == entry['sha256'] and (proof/entry['file']).stat().st_size == entry['bytes'], 'Executed dependency bytes changed')
                add(proof/entry['file'], 'executed_source_dependency')
            for record in expected['records']:
                path = Path(record['quality_source_run'])
                if not path.is_absolute() or not path.exists(): path = EVIDENCE/'results'/str(path).replace('\\', '/').split('/results/', 1)[1]
                path = resolve(path)
                if str(path) not in donors: donors[str(path)] = SourceRun(path)
                source = donors[str(path)]; entry = next(e for e in source.entries if e['id'] == record['id'])
                require(source.hashes == record['quality_source_raw_hashes'] and source.contract_sha == record['quality_source_effective_contract_sha256'], 'Timing expected source contract differs')
                require(entry['status'] == 'PASS' and entry['line_sha256'] == record['quality_source_line_sha256'], 'Timing expected source original line differs')
                actual = json.loads(source.line(entry))
                for field in ('output_token_ids', 'input_ids_sha256', 'full_prompt_hash', 'stored_visual_KV_bytes', 'original_image_sha256', 'method', 'model_id'):
                    require(actual[field] == record[field], 'Expected model output differs from original donor')
                require(source.contract == expected['source_contracts_by_method'][record['method']], 'Expected model source context differs')
                origin_checks.append({'expected_file': str(raw/'expected_predictions.json'),
                    'expected_file_sha256': load(raw/'manifest.json')['expected_predictions_sha256'],
                    'id': record['id'], 'method': record['method'], 'model_id': record['model_id'],
                    'original_source': str(path), 'original_raw_hashes': source.hashes,
                    'original_line_sha256': entry['line_sha256'], 'effective_contract_sha256': source.contract_sha,
                    'token_context_bytes_exactly_matched': True})
                line_checks += 1
            for item in result['methods']:
                scalar.append({'dataset': key[0], 'step': key[1], 'method': item['method'],
                               'warmup_trials': item['warmup_trials'], 'measured_trials': item['measured_trials'],
                               'natural_output_tokens_mean': item['output_tokens']['mean'], 'TTFT_host_ms': item['ttft_ms']['median'],
                               'natural_ITL_host_ms': item['pooled_natural_ITL_ms']['mean'],
                               'complete_visual_owner_mean_bytes': item['complete_visual_owner_bytes']['mean'],
                               'host_quiet_asserted': False, 'no_wall_only_or_cross_domain_speed_ranking': True})
            for path in (out/'audit'/name).iterdir():
                if path.is_file(): add(path, 'independent_cpu_timing_validation')
        covered, recipe = timing_origin_phases(donors, out, primary_contract)
        for node in source_closure([{'raw': Path(path)} for path in donors]):
            for path in node['files']:
                if str(path.resolve()).lower() not in covered:
                    require(path.stat().st_size <= 64*1024**2, 'Large original timing donor must enter via a required quality phase closure')
                    add(path, 'expected_model_output_original_quality_origin')
        write(out/'domain_timing.json', {'rows': scalar, 'quality_QA_added': 0, 'raw_trials': 560,
              'warmup_trials': 140, 'measured_trials': 420, 'no_cross_domain_pooling': True})
        add(out/'domain_timing.json', 'browsable_domain_timing_summary')
        with (out/'domain_timing.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scalar[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(scalar)
        add(out/'domain_timing.csv', 'browsable_domain_timing_scalar_table')
        write(out/'timing_expected_origin_binding.json', {'status': 'PASS_EACH_EXPECTED_MODEL_RECORD_ORIGINAL_SOURCE_BINDING',
              'original_source_runs': len(donors), 'record_line_checks': line_checks, 'rows': origin_checks,
              'required_current_bundle_contracts': recipe['required_current_bundle_contracts'],
              'no_new_quality_QA_or_GPU_inference': True})
        add(out/'timing_expected_origin_binding.json', 'exact_expected_record_original_source_binding_audit')
        support = {'schema': 'baseline_matrix.public_support_files.v1', 'label': 'canonical_E64_five_source_profiles_new560',
                   'completed_evidence_only': True, 'primary_quality_QA_added': 0, 'quality_QA_added': 0,
                   'scope': 'Canonical five source profiles: Text2 smoke40 + Text16 formal320 + Info2 smoke40 + Info8 formal160. Natural answers, host_quiet=false; warm1/repeat3. No old E8/legacy656/Calib32; no QA or wall-only speed ranking.',
                   'required_current_bundle_contracts': recipe['required_current_bundle_contracts'],
                   'files': sorted(files.values(), key=lambda f: f['path'].lower())}
        write(out/'timing_support.json', support)
        recipe.update(schema='baseline_matrix.existing_exporter_delivery_recipe.v1',
                      support_file_lists=[str(out/'timing_support.json')], require_complete=True,
                      exporter='ops/export_public_v3.py', verifier='ops/verify_public_export_v3.py',
                      expected_source_donors=[{'raw': path, 'raw_hashes': donors[path].hashes} for path in sorted(donors)],
                      source_original_public_archive_gitLF_identities_separate=True,
                      github_mutation_performed=False)
        write(out/'timing_delivery_recipe.json', recipe)
        report.update(status='PASS_CANONICAL_560_AND_ORIGINAL_DEPENDENCY_CLOSURE_CPU', publication_eligible=True,
                      expected_source_line_checks=line_checks, unique_expected_original_sources=len(donors),
                      required_auxiliary_quality_contracts=recipe['auxiliary_contracts'],
                      attach_to_primary_contract=recipe['primary_contract'])
    write(out/'readiness.json', report); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['p2', 'p3', 'timing560'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-complete', action='store_true')
    parser.add_argument('--timing-proof-root', type=Path, default=EVIDENCE/'results/timing_source_proofs_v1')
    parser.add_argument('--timing-primary-contract', type=Path, default=ROOT/'configs/publication/M1_FP16_all4_full_reference_v1.json')
    args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(EVIDENCE/'results/publication'), 'Use a new E results/publication snapshot')
    out.mkdir(parents=True, exist_ok=False)
    report = timing(out, args.verify_complete, resolve(args.timing_proof_root), args.timing_primary_contract) if args.phase == 'timing560' else quality(args.phase, out, args.verify_complete)
    print(json.dumps({k: report[k] for k in report if k not in {'cells', 'definitions', 'missing'}}))
    if args.verify_complete and not report['publication_eligible']: raise SystemExit(3)


if __name__ == '__main__': main()
