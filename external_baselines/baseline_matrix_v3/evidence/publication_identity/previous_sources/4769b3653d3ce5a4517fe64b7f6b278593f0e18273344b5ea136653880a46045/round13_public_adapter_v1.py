"""Round13 publication admission helpers, without export or network writes.

Frozen R12 source/line identities, physical storage accounting, producer binds
and official metrics are reused. New R13 schemas and physical formats receive
explicit checks. Inventory preflight alone is never publication admission.
Root must supply final view/report/QA0 closure before invoking an exporter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

import round12_public_core_v1 as frozen
from export_public import Sanitizer

SCHEMA = 'trajkv.round13.public_scope.v1'
NAMESPACE = frozen.NAMESPACE
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('${EVIDENCE_ROOT}').resolve()
REQUIREMENTS = ('Root final per-view paired/report closure', 'QA0 and attempts actual source/producer accounting',
                'new R13 public index/seal/export/cold adapters', 'independent remote commit/tree/archive/history verification')
require, sha, near = frozen.require, frozen.sha, frozen.near
LocalSource, PublicSource, metric_modules = frozen.LocalSource, frozen.PublicSource, frozen.metric_modules


def payload(scope):
    return {k: v for k, v in scope.items() if k != 'root_seal'}


def check_final_header(scope):
    """Cheap rejection before source hashing or any large raw JSONL parsing."""
    require(scope.get('schema') == SCHEMA and scope.get('namespace') == NAMESPACE,
            'R13 explicit production scope schema/namespace required')
    require(scope.get('fixture_only') is False and scope.get('research_cutoff_complete') is True,
            'R13 incomplete or fixture stage cannot be prepared for production export')
    require(scope.get('stage_state') in ('READY_FOR_ROOT_SEAL', 'ROOT_SEALED'),
            'R13 unfinished scope state')
    require(scope.get('whole_external_library_completion_claimed') is False
            and scope.get('SourceRun_admission_claimed') is False,
            'R13 stage cannot claim whole external library or SourceRun admission')
    require(scope.get('metric_contract') == 'matrix_v3_metrics_v1'
            and isinstance(scope.get('phase_label'), str) and bool(scope['phase_label'])
            and all(c.isascii() and (c.isalnum() or c in '_-') for c in scope['phase_label']),
            'R13 contract/safe phase label missing')
    require(isinstance(scope.get('cells'), list) and bool(scope['cells'])
            and isinstance(scope.get('files'), list) and bool(scope['files']), 'R13 actual cell/file inventory required')
    identities = [cell['cell_id'] for cell in scope['cells']]
    require(len(set(identities)) == len(identities), 'R13 duplicate cell ID')
    for cell in scope['cells']:
        require(cell.get('kind') in ('new_round13', 'new_round12', 'pilot', 'reused_reference'), 'R13 explicit cell origin')
        require(cell.get('disposition') in ('CLOSED_COMPLETE', 'CLOSED_WITH_FAILURES', 'CLOSED_PILOT_EXCLUDED'),
                'R13 unclosed actual cell')
    for group in scope.get('groups', []):
        require(group.get('disposition') in ('EXECUTED', 'NOT_RUN', 'STOPPED'), 'R13 pending group')
        require(set(group.get('cell_ids', [])) <= set(identities), 'R13 group has an unknown cell')
        if group['disposition'] == 'NOT_RUN':
            require(group.get('reason') and not group.get('cell_ids') and group.get('quality') is None,
                    'R13 unrun group must have reason, zero cells and null quality')
        if group['disposition'] == 'STOPPED':
            require(group.get('reason'), 'R13 stopped group needs an actual reason')
    for attempt in scope.get('attempts', []):
        require(attempt.get('disposition') in ('PRE_MODEL_FAILED', 'NOT_LAUNCHED', 'TECHNICAL_FAILED', 'CLOSED_ABORTED')
                and attempt.get('reason') and attempt.get('files')
                and attempt.get('successful_terminal_answers') == 0, 'R13 unclosed/hiding-answer attempt')
        if attempt['disposition'] in ('PRE_MODEL_FAILED', 'NOT_LAUNCHED'):
            require(attempt.get('actual_model_forwards') == 0 and attempt.get('recorded_task_rows') == 0,
                    'R13 pre-model/unlaunched attempt cannot hide task/model work')
    require({'report', 'quality_table'} <= {row.get('role') for row in scope['files']}
            and scope.get('report_summary'), 'R13 final report and browsable quality table required')
    return True


def inventory_preflight(scope):
    """Full explicit source SHA inventory; no claim of semantic/cold admission."""
    check_final_header(scope)
    source = LocalSource(scope, (ROOT, EVIDENCE))
    for cell in scope['cells']:
        for rel in cell['paths'].values():
            source.path(rel)
        for row in cell.get('source_graph', []) + cell.get('asset_graph', []):
            source.path(row['file'])
        for key, rel in cell.get('producer', {}).items():
            if key in ('launch', 'job', 'snapshot_inventory', 'source_config', 'sequence_config', 'parent_complete'):
                source.path(rel)
    for row in scope['files']:
        if row['role'] in ('inputs', 'refs', 'source_lock'):
            require(row.get('delivery_channel', 'BROWSER_AND_ARCHIVE') == 'BROWSER_AND_ARCHIVE'
                    and row.get('expected_in_git', True) is True, 'R13 reference hidden by archive/ignore rule')
    source.path(scope['report_summary'])
    for rel, digest in frozen.FROZEN_METRIC_SOURCES.items():
        source.original(rel, digest)
    return source, {'status': 'PASS_R13_CLOSED_INVENTORY_PREFLIGHT_ONLY',
        'publication_eligible': False, 'cells': len(scope['cells']), 'files': len(scope['files']),
        'remaining_requirements': list(REQUIREMENTS), 'GitHub_mutation': False, 'new_GPU_calls': 0}


def _r_count(n):
    return min(224, (1670 * n + 65408) // (18 * n))


def _r_bytes(n):
    return (3816 + 18 * _r_count(n)) * n + 89737


def _exception_bytes(n, h, stages):
    if h == 0 and stages == 43:
        return 4590 * n + 89733
    return ((3816 + 18 * stages) * n + 89733
            + h * (57344 - (3808 + 18 * stages)) + 1792 * (-(n - h) % 32)
            + 8 * ((n + 31) // 32) + 4 * h + 4)


def precenter_budget(n, mode):
    require(mode in ('PC', 'PC_DONOR') and n > 0, 'R13 declared conditional PC mode')
    donors = (28936 + 18 * n - 1) // (18 * n)
    added = 28936 if mode == 'PC' else 0
    return {'K_donor_stages': donors, 'new_persistent_rope_and_mu_bytes': added,
            'expected_owner_bytes': _r_bytes(n) - 18 * donors * n + added, 'cap_bytes': _r_bytes(n)}


def physical_r13(row):
    """Recompute actual storage then check new format against original-N cap.

    Inherited Round12 detail can describe a pre-compaction reference bank.
    Only its optional format-specific check is excluded from the generic
    storage audit; the serialized evidence itself is neither changed nor saved.
    New format fields below, actual tensor charges and full model proof remain
    mandatory. No nominal-bits replacement of physical storage is admitted.
    """
    view = dict(row)
    view['r7_bridge'] = {k: v for k, v in row.get('r7_bridge', {}).items() if k != 'round12_static'}
    actual = frozen.physical_row(view, fixture=False, reused=False)
    n = row['actual_visual_token_count']
    require(n > 0 and n % 256 == 0, 'R13 original visual geometry differs')
    meta = row['r7_bridge']
    router = meta['round13_router']
    mode = router['mode']
    require(mode in ('P_best', 'P_random', 'EX_best', 'TOK_best', 'CAL_AFF', 'PC', 'PC_DONOR')
            and router['query_unknown'] is True
            and router['selection_uses_test_question_or_reference'] is False
            and router['original_visual_count'] == n
            and router['technical_only'] is False, 'R13 deployed route/query-free proof differs')
    require(meta['returned_owners_unique_tensor_capacity_bytes'] == actual, 'R13 factory/physical byte disagreement')
    guard = row['round13_physical_budget']
    cap = 4590 * n + 89733 if mode.startswith('P_') else 6382 * n + 89745 if mode == 'CAL_AFF' else _r_bytes(n)
    require(guard['mode'] == mode and guard['N'] == n
            and guard['actual_complete_visual_bytes'] == actual and guard['strict_cap_bytes'] == cap
            and guard['strict_cap_satisfied'] is True and 0 < actual <= cap,
            'R13 actual strict per-request budget failed')
    stages = meta['selected_stages']
    require(len(stages) == meta['selected_stage_count'], 'R13 concrete selected slot count differs')
    if mode.startswith('P_'):
        detail = meta['round13_pruned']
        m = detail['M']
        expected = cap if detail['fallback_original_low'] else 6382 * m + 91657 + 8 * n + 16
        require(detail['N'] == n and detail['actual_owner_bytes'] == expected == actual
                and len(stages) == 43 and router['owner_visual_counts'] == [m] * 28,
                'R13 P physical compact/full-fallback layout differs')
        if not detail['fallback_original_low']:
            require(256 <= m < n and m % 32 == 0
                    and detail['full_native_prefill_before_pruning'] is True
                    and detail['full_A_encoding_before_compaction'] is True
                    and detail['full_N_means_scales_gains_and_residuals_exact_gather'] is True
                    and detail['pruned_visual_rows_must_not_be_native_FP16_text'] is True,
                    'R13 P full-context-then-compact proof differs')
    elif mode == 'TOK_best':
        detail = meta['round13_precision']
        h = detail['high_tokens']
        k = len(detail['same_request_R_K_residual_slots_removed'])
        expected = cap - 18 * k * n + 2048 + 4 * n + 1792 * h
        require(detail['R_cap_bytes'] == cap and detail['actual_owner_bytes'] == expected == actual
                and detail['expected_owner_bytes'] == expected
                and h == min(n, max(0, (18 * k * n - 2048 - 4 * n) // 1792))
                and detail['V_fields_frozen_to_request_R'] is True
                and detail['V_code_scale_min_preserved_without_reencoding'] is True
                and detail['V_residuals_preserved_by_original_layer_head_depth_token_position'] is True
                and all(stage[0] == 'v' for stage in stages), 'R13 TOK pools/refund/V frozen contract differs')
        tensors = row['owner_before_suffix']['tensors']
        high = [v for key, v in tensors.items() if key.endswith('.round13_high_code')]
        low = [v for key, v in tensors.items() if key.endswith('.k_code')]
        require(len(high) == len(low) == 28 and all(v['shape'] == [1, 4, h, 32] for v in high)
                and all(v['shape'] == [1, 4, n-h, 16] for v in low), 'R13 TOK must have actual compact 4/8-bit pools')
    elif mode == 'EX_best':
        detail = meta['round13_exception']
        h, stages_count = detail['exception_rows'], detail['base_stage_count']
        expected = _exception_bytes(n, h, stages_count)
        require(stages_count in (0, 41, 43) and 0 <= h <= n
                and detail['original_R_complete_owner_bytes'] == cap
                and detail['expected_complete_owner_bytes'] == expected == actual
                and detail['V_padding_rows'] == -(n-h) % 32,
                'R13 EX actual replacement/padding budget differs')
    elif mode in ('PC', 'PC_DONOR'):
        detail = meta['round13_precenter']
        plan = precenter_budget(n, mode)
        expected = plan['expected_owner_bytes']
        original = detail['original_R_selected_stages']
        dropped = detail['K_donor_stages_removed']
        keys = [stage for stage in original if stage[0] == 'k']
        donors = plan['K_donor_stages']
        require(detail['mode'] == ('PC' if mode == 'PC' else 'donor')
                and detail['R_cap_bytes'] == cap and detail['actual_owner_bytes'] == expected == actual
                and detail['extra_PC_bytes'] == 28936 and detail['K_donor_stages'] == donors
                and len(original) == _r_count(n) and dropped == keys[-donors:]
                and stages == [stage for stage in original if stage not in dropped]
                and detail['V_selected_stages_preserved'] == [stage for stage in original if stage[0] == 'v']
                and detail['V_all_code_scale_min_and_logical_residuals_bit_preserved'] is True
                and detail['prefix_frequencies_copied_frozen_not_future_query'] is True
                and detail['new_persistent_rope_and_mu_bytes'] == plan['new_persistent_rope_and_mu_bytes']
                and detail['equal_actual_bytes_claimed'] is False
                and detail['phase_tables_persistent'] is False and detail['technical_only'] is False,
                'R13 PC/donor actual funding/V/prefix-frequency contract differs')
        require(detail['phase_tables_ephemeral_Half_bytes_per_attention_call'] == (512 * n if mode == 'PC' else 0),
                'R13 PC transient phase-table bytes must remain explicit')
        tensors = row['owner_before_suffix']['tensors']
        shared = {leaf: [value for key, value in tensors.items() if key.endswith('.' + leaf)]
                  for leaf in ('round13_precenter_mu', 'round13_precenter_freq',
                               'round13_precenter_scaling', 'round13_precenter_format')}
        if mode == 'PC':
            shapes = {'round13_precenter_mu': [28, 4, 1, 128], 'round13_precenter_freq': [64],
                      'round13_precenter_scaling': [1], 'round13_precenter_format': [4]}
            for leaf, shape in shapes.items():
                require(shared[leaf] and all(value['shape'] == shape for value in shared[leaf]),
                        'R13 PC actual shared tensor layout differs: ' + leaf)
        else:
            require(not any(shared.values()), 'R13 donor-only cannot retain unused PC assets')
    else:
        detail = meta['round13_precision']
        expected = 6382 * n + 89745
        require(detail['mode'] == 'CalTransport' and detail['actual_owner_bytes'] == expected == actual
                and detail['K_fields_preserved'] is True and len(stages) == 43
                and detail['extra_deployment_transport_tensor_bytes'] == 0,
                'R13 CalTransport same-price/K/metadata contract differs')
    return {'mode': mode, 'N': n, 'actual_owner_bytes': actual, 'cap_bytes': cap,
            'expected_owner_bytes': expected, 'actual_unique_storage_verified': True}


FIXED500_SCHEMA = 'trajkv.round13.fixed500_comparison.raw.v1'
FIXED500_ROLE = 'round13_observed_fixed500_comparison'
FIXED500_NATIVE_FIELDS = ('id', 'original_image_sha256', 'raw_question_sha256', 'rendered_question_sha256',
    'full_prompt_hash', 'chat_template_hash', 'input_ids_sha256', 'question_suffix_sha256',
    'native_position_ids_sha256', 'compression_boundary', 'actual_visual_token_count', 'model_commit', 'dtype')


def _input_only_500(value):
    """Portable mirror of the frozen input-only loader; opens no model data."""
    forbidden = {'answers', 'labels', 'scores', 'refs', 'references', 'reference_answers', 'reference_path',
        'old_logits_path', 'old_raw_predictions_path', 'prediction_cache_path', 'raw_prediction', 'output_token_ids'}
    if isinstance(value, dict):
        require(not set(value) & forbidden, 'R13 benchmark worker artifact contains output/reference fields')
        for child in value.values():
            _input_only_500(child)
    elif isinstance(value, list):
        for child in value:
            _input_only_500(child)
    elif isinstance(value, str):
        require(not any(word in value.replace('\\', '/') for word in
            ('records.jsonl', 'scored_records', '/r7_matrix500_v2/', '/p2_views/')),
            'R13 old prediction/raw path entered benchmark worker artifact')


def validate_fixed500_contract(cell, source, manifest):
    """Explicit observed-benchmark admission; never reuse the dev role."""
    cfg = manifest['config']
    require(cfg.get('schema') == FIXED500_SCHEMA and cfg.get('input_role') == FIXED500_ROLE
            and cfg.get('cohort_kind') == 'fixed500_comparison'
            and cfg.get('method', {}).get('family') == 'trajkv_round13_fixed500_comparison'
            and cfg['method'].get('information_condition') == 'query_unknown',
            'R13 fixed500 schema/role/family must remain distinct from development')
    require(manifest.get('comparison_role') == FIXED500_ROLE
            and manifest.get('previously_observed_fixed500') is True
            and manifest.get('independent_holdout_claimed') is False
            and manifest.get('phase_is_confirmatory') is False,
            'R13 observed benchmark cannot become a fresh independent holdout')
    _input_only_500(cfg)
    graph = {row['logical']: row['file'] for row in cell.get('source_graph', [])}
    require({'code/round13_fixed500_runner_v1.py', 'code/round13_fixed500_inputs_v1.py'} <= set(graph),
            'R13 actual new fixed500 admission source graph omitted')
    assets = {row['logical']: row['file'] for row in cell.get('asset_graph', [])}
    require(len(assets) == len(cell.get('asset_graph', [])), 'R13 duplicate benchmark asset graph key')
    def asset(stem):
        path, digest = cfg.get(stem + '_path'), cfg.get(stem + '_sha256')
        require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64 and path in assets,
                'R13 benchmark locked asset omitted: ' + stem)
        source.original(assets[path], digest)
        value = source.doc(assets[path])
        _input_only_500(value)
        return value
    inputs_manifest = asset('input_manifest')
    receipt = asset('selection_receipt')
    algorithm = asset('selected_algorithm')
    require(inputs_manifest.get('schema') == 'trajkv.round13.observed_fixed500_inputs.v1'
            and inputs_manifest.get('role') == FIXED500_ROLE
            and inputs_manifest.get('previously_observed') is True
            and inputs_manifest.get('independent_holdout_claimed') is False
            and inputs_manifest.get('answers_in_encoder_inputs') is False
            and inputs_manifest.get('old_predictions_or_logits_in_inference_artifact') is False,
            'R13 full500 input-only identity contract differs')
    name = PurePosixPath(cfg['inputs'].replace('\\', '/')).name
    require(name in inputs_manifest['input_files'] and PurePosixPath(cell['paths']['inputs']).name == name,
            'R13 benchmark input file identity differs')
    entry = inputs_manifest['input_files'][name]
    inputs = [row for row, _, _ in source.lines(cell['paths']['inputs'])]
    require(entry['count'] == cfg['input_count'] == len(inputs) == 500 and entry['role'] == FIXED500_ROLE
            and entry['answers_included'] is False and entry['sha256'] == cfg['inputs_sha256']
            and [row['id'] for row in inputs] == entry['ids']
            and len({row['source_cluster_id'] for row in inputs}) == len({row['id'] for row in inputs}) == 500,
            'R13 full500 frozen source count/order differs')
    require(set(inputs_manifest['native_bindings']) == set(entry['ids']), 'R13 full500 native-binding coverage differs')
    for row in inputs:
        native = inputs_manifest['native_bindings'][row['id']]
        require(row['dataset'] == 'textvqa' and set(FIXED500_NATIVE_FIELDS) <= set(native)
                and native['original_image_sha256'] == row['original_image_sha256']
                and native['raw_question_sha256'] == row['question_sha256']
                and native['model_commit'] == '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
                and native['dtype'] == 'float16' and native['preprocessing'] == {'max_patches': 8}
                and native['question_suffix'] == cfg['question_suffix'], 'R13 complete input-only native500 sidecar differs')
    require(type(cfg.get('start')) is int and type(cfg.get('count')) is int
            and 0 <= cfg['start'] and 0 < cfg['count'] <= 125 and cfg['start'] + cfg['count'] <= 500,
            'R13 explicit legal benchmark shard bounds required')
    require(receipt.get('schema') == 'trajkv.round13.fixed500_selection.v1'
            and receipt.get('status') == 'FROZEN_BEFORE_P_FIXED500'
            and receipt.get('fixed500_is_previously_observed') is True
            and receipt.get('independent_holdout_claimed') is False
            and receipt.get('new_candidate_fixed500_outputs_read_before_selection') is False
            and receipt.get('methods_and_assets_frozen_before_candidate_comparison') is True
            and receipt.get('input_manifest_sha256') == cfg['input_manifest_sha256'],
            'R13 candidate selection must remain frozen before new P500 outputs')
    selected = cfg.get('selected_method_name')
    require(selected == 'TrajKV_R13_Pbest_lowcap_v1' and receipt.get('selected_methods') == [selected]
            and algorithm.get('schema') == 'trajkv.round13.selected_P_algorithm.v1'
            and algorithm.get('name') == selected and cfg['candidate']['mode'] == 'P_best',
            'R13 observed benchmark must use the preselected original P method')
    require(manifest.get('selected_method_name') == selected
            and manifest.get('selection_receipt_sha256') == cfg['selection_receipt_sha256']
            and manifest.get('selected_algorithm_sha256') == cfg['selected_algorithm_sha256'],
            'R13 benchmark manifest selection identity differs')
    require(any(row.get('name') == selected and row.get('sha256') == cfg['selected_algorithm_sha256']
                for row in receipt.get('method_asset_locks', [])), 'R13 algorithm missing from selection asset locks')
    numeric = lambda value: {key: child for key, child in value.items() if not key.endswith('_path')}
    require(numeric(algorithm['candidate']) == numeric(cfg['candidate'])
            and algorithm['variant_profile'] == cfg['variant_source']['name']
            and algorithm['variant_source_sha256'] == cfg['variant_source']['sha256'],
            'R13 selected encoder/selector/f0 numeric lineage changed')
    for group in ('algorithm_source_hashes', 'base_runtime_source_hashes'):
        require(algorithm.get(group), 'R13 selected source group omitted')
        for logical, digest in algorithm[group].items():
            require(manifest['source_hashes'].get(logical) == digest and logical in graph,
                    'R13 selected numeric/runtime source differs from actual benchmark graph')
            source.original(graph[logical], digest)
    for row, _, _ in source.lines(cell['paths']['records']):
        require(row.get('comparison_role') == FIXED500_ROLE and row.get('round13_fixed500_schema') == FIXED500_SCHEMA
                and row.get('previously_observed_fixed500') is True and row.get('independent_holdout_claimed') is False,
                'R13 benchmark row relabeled as development/independent confirmation')
        expected = inputs_manifest['native_bindings'].get(row['id'])
        require(isinstance(expected, dict) and set(FIXED500_NATIVE_FIELDS) <= set(expected), 'R13 row native binding absent')
        if row['status'] == 'PASS':
            require(all(row.get(key) == expected[key] for key in FIXED500_NATIVE_FIELDS)
                    and row['preprocessing'] == expected['preprocessing'], 'R13 own candidate native input identity differs')
            proof = row.get('round13_fixed500_native_input_binding', {})
            require(proof.get('matched') is True and proof.get('mismatches') == {}
                    and proof.get('old_logits_or_output_used_by_model') is False,
                    'R13 own native metadata check or old-output isolation missing')
    return {'evaluation_phase': 'observed_fixed500_comparison', 'selected_method_name': selected,
            'independent_holdout_claimed': False}


def validate_new_cell(cell, source, metric, scorer):
    """One complete R13 actual raw/producer/scored cell, returning compact rows.

    Aggregate view/paired/report/QA0 closure belongs to the explicit Root final
    contract. This helper alone deliberately does not return publication PASS.
    """
    require(cell.get('kind') == 'new_round13', 'new-cell helper does not relabel reused/R12/pilot producers')
    paths = cell['paths']
    manifest, complete = source.doc(paths['manifest']), source.doc(paths['complete'])
    runner_schema = manifest.get('runner_schema')
    require(manifest.get('schema') == 'baseline_matrix.raw_inference.v1'
            and runner_schema in ('trajkv.round13_research.raw.v1', FIXED500_SCHEMA)
            and manifest.get('references_read') is False and manifest.get('prediction_cache_used') is False
            and complete.get('quality_scored') is False and complete.get('coverage_complete') is True,
            'R13 actual reference-free complete producer required')
    cfg = manifest['config']
    if runner_schema == FIXED500_SCHEMA:
        phase = validate_fixed500_contract(cell, source, manifest)
        require(complete.get('comparison_role') == FIXED500_ROLE
                and complete.get('previously_observed_fixed500') is True
                and complete.get('independent_holdout_claimed') is False
                and complete.get('phase_is_confirmatory') is False
                and complete.get('old_prediction_cache_used_by_new_candidate') is False,
                'R13 benchmark completion lost observed-comparison/output-isolation role')
    else:
        require(cfg.get('input_role') == 'round13_development' and cfg.get('cohort_kind') == 'development',
                'R13 development source role differs')
        phase = {'evaluation_phase': 'development', 'independent_holdout_claimed': False}
    require(not set(cfg) & {'refs', 'references', 'answers', 'labels', 'scores'}, 'R13 inference contains labels')
    require(manifest['model_identity']['revision'] == '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
            and manifest['model_residency']['weight_resident_bytes'] == 15888747520, 'R13 full original M1 residency')
    for field, key in (('manifest', 'manifest_sha256'), ('records', 'records_jsonl_sha256')):
        source.original(paths[field], complete[key])
    raw_lines = source.lines(paths['records'])
    raw = [row for row, _, _ in raw_lines]
    expected = manifest['expected_ids']
    require(expected and expected == [row['id'] for row in raw] and len(set(expected)) == len(raw) == manifest['expected_QA']
            == complete['expected_QA'] == complete['recorded_QA']
            and dict(Counter(row['status'] for row in raw)) == complete['counts'], 'R13 raw coverage/count mismatch')
    finished = {'PASS', 'UNSUPPORTED_MODEL', 'HARDWARE_UNSUPPORTED', 'OOM_AT_DECLARED_INPUT',
                'METHOD_PORT_UNVERIFIED', 'INPUT_MISMATCH', 'OWNER_AUDIT_FAILED', 'FAILED_RUN', 'STALE_PROTOCOL'}
    require(all(row['status'] in finished for row in raw), 'R13 unfinished/unknown task status')
    all_pass = all(row['status'] == 'PASS' for row in raw)
    require(cell['disposition'] == ('CLOSED_COMPLETE' if all_pass else 'CLOSED_WITH_FAILURES')
            and complete['status'] == ('PASS_RAW_COMPLETE' if all_pass else 'COMPLETE_WITH_FAILURES'), 'R13 hides task failure')
    receipt = source.doc(paths['remote_receipt'])
    require(receipt.get('remote_hashes_computed_independently') is True and receipt.get('status') in
            ('PASS_REMOTE_RAW_SHA256_EXACT', 'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA'), 'R13 independent readback omitted')
    inventory = {row['file']: row for row in receipt['files']}
    for field in ('manifest', 'records', 'complete'):
        item = inventory[field + ('.jsonl' if field == 'records' else '.json')]
        source.original(paths[field], item['sha256'])
        require(source.path(paths[field]).stat().st_size == item['bytes'], 'R13 readback byte count differs')
    frozen.producer_binding(cell, source, manifest, complete, fixture=False)
    refs = {row['id']: row for row, _, _ in source.lines(paths['refs'])}
    input_lines = source.lines(paths['inputs'])
    inputs = {row['id']: row for row, _, _ in input_lines}
    lock = source.doc(paths['source_lock'])
    for key in ('refs', 'inputs'):
        source.original(paths[key], lock['files'][Path(paths[key]).name]['sha256'])
    source.original(paths['inputs'], manifest['inputs_sha256'])
    require(expected == [row['id'] for row, _, _ in input_lines][cfg['start']:cfg['start'] + cfg['count']], 'R13 source slice differs')
    scored_lines = source.lines(paths['scored_records'])
    scored = {row['id']: row for row, _, _ in scored_lines}
    require(len(scored) == len(scored_lines) and set(scored) == set(expected), 'R13 scored coverage differs')
    summary = source.doc(paths['scored_summary'])
    require(summary['metric_contract_version'] == metric.METRIC_CONTRACT_VERSION, 'R13 metric version differs')
    source.original(paths['scored_records'], summary['derived_file_hashes']['scored_records.jsonl'])
    for field in ('manifest', 'records', 'complete'):
        source.original(paths[field], summary['integrity']['raw_files'][field + ('.jsonl' if field == 'records' else '.json')])
    compact = {}
    for row, line_number, line_sha in raw_lines:
        ident, answer = row['id'], scored[row['id']]
        require(ident in refs and ident in inputs, 'R13 source/reference missing')
        for key in ('id', 'dataset', 'source_cluster_id', 'original_image_sha256', 'question_sha256'):
            require(row[key] == inputs[ident][key], 'R13 raw/source differs')
        require(refs[ident]['source_cluster_id'] == row['source_cluster_id']
                and row['dataset'] == cell['dataset'], 'R13 cluster/domain differs')
        require(answer.get('raw_jsonl_line_number', answer.get('raw_line_number')) == line_number
                and answer.get('raw_jsonl_exact_line_sha256', answer.get('raw_line_sha256')) == line_sha
                and answer['raw_prediction'] == row.get('raw_prediction')
                and answer.get('output_token_ids') == row.get('output_token_ids'), 'R13 score not bound to exact raw line')
        if row['status'] == 'PASS':
            require(row.get('output_token_ids') and isinstance(row.get('raw_prediction'), str)
                    and row.get('official_score') is None, 'R13 PASS is not an unscored free answer')
            result = metric.score_question(row['dataset'], row['raw_prediction'], refs[ident])
            require(answer['score_status'] == 'SCORED_PASS' and answer['metric_id'] == result['metric_id'], 'R13 scored status/metric differs')
            near(answer['score'], result['score'], 'R13 frozen official score')
            physical_r13(row)
        else:
            require(answer['score_status'] != 'SCORED_PASS' and answer.get('score') is None, 'R13 failed row got score')
        merged = {**row, **answer}
        compact[ident] = {k: merged[k] for k in ('id', 'dataset', 'status', 'source_cluster_id', 'score', 'score_status',
            'metric_id', 'stored_visual_KV_bytes', 'actual_effective_visual_bits', 'encode_ms', 'wall_ms', 'ttft_ms',
            'peak_allocated_bytes') if k in merged}
    mean = sum(row['score'] for row in compact.values()) / len(compact) if all_pass else None
    near(mean, summary['primary_mean'], 'R13 official cell mean')
    interval = scorer.cluster_bootstrap(list(compact.values()), 'score',
        summary['bootstrap']['repetitions'], summary['bootstrap']['seed'])['ci95'] if all_pass else None
    near(interval, summary['primary_ci95'], 'R13 official cell interval')
    return {'cell_id': cell['cell_id'], 'kind': 'new_round13', 'dataset': cell['dataset'],
        **phase,
        'selected_rows': compact, 'producer_rows': len(raw), 'successful_terminal_answers': sum(
            row['status'] == 'PASS' and bool(row.get('output_token_ids')) for row in raw),
        'untrusted_generated_answer_rows': sum(row['status'] != 'PASS' and bool(
            row.get('output_token_ids') or row.get('untrusted_generated_output_token_ids') or row.get('untrusted_output_token_ids')) for row in raw),
        'failure_status_rows': sum(row['status'] != 'PASS' for row in raw), 'primary_mean': mean,
        'pilot_excluded': False}


def export_closed_scope(scope_path, *, semantic_closure=None, exporter=None):
    """Explicit hook boundary; missing final Root adapters fail before stage IO."""
    scope = json.loads(Path(scope_path).read_text(encoding='utf-8-sig'))
    check_final_header(scope)
    require(callable(semantic_closure) and callable(exporter),
            'R13_FINAL_CLOSURE_AND_EXPORT_HOOKS_REQUIRED: inventory preflight is not publication admission')
    seal = scope.get('root_seal')
    require(scope['stage_state'] == 'ROOT_SEALED' and isinstance(seal, dict)
            and seal.get('schema') == 'trajkv.round13.root_seal.v1', 'R13 unsealed scope cannot export')
    require(seal.get('source_payload_sha256') == frozen.canonical(payload(scope))
            and seal.get('public_payload_sha256') == frozen.canonical(Sanitizer().tree(payload(scope))),
            'R13 Root source/public cutoff seal differs')
    source, _ = inventory_preflight(scope)
    closure = semantic_closure(scope, source)
    require(closure.get('publication_eligible') is True and closure.get('actual_producer_binding_required') is True,
            'R13 final complete semantic closure required')
    return exporter(scope, source, closure)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    scope = json.loads(a.scope.read_text(encoding='utf-8-sig'))
    _, result = inventory_preflight(scope)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
