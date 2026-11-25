"""Final R13 actual-source semantic closure, portable and CPU-only.

The cell/producer/official-metric checks remain in the unchanged R12 core and
the explicit R13 adapter. View and paired computations below derive from the
general routines in round12_public_core_v1.verify_semantics, with additional
R13 method/reference/event, resource, QA0 and research-cutoff accounting.

verify_semantics accepts only a genuinely closed R13 scope. admit_cells and
recompute_views are separate pure helpers for partial actual-cell checks;
calling them does not create a final scope or confer publication eligibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import PurePosixPath
import statistics

import round13_public_adapter_v1 as adapter

frozen = adapter.frozen
require, near = frozen.require, frozen.near
COUNT_FIELDS = ('producer_rows', 'successful_terminal_answers', 'untrusted_generated_answer_rows', 'failure_status_rows')
RESOURCE_FIELDS = ('stored_visual_KV_bytes', 'actual_effective_visual_bits', 'encode_ms', 'wall_ms',
                   'ttft_ms', 'peak_allocated_bytes', 'prefill_ms', 'question_prefill_ms',
                   'text_kv_bytes', 'strict_cap_bytes')
MANDATORY_COUNTS = tuple('new_' + key for key in COUNT_FIELDS) + tuple(
    prefix + key for prefix in ('r13_method_', 'r13_development_method_', 'r13_benchmark_', 'r12_reference_') for key in COUNT_FIELDS) + (
    'reused_selected_view_rows', 'closed_non_quality_attempts', 'not_run_groups',
    'QA0_native_prefix_model_forwards', 'QA0_model_TF_forwards', 'QA0_cuda_attention_cases',
    'QA0_S0_sources', 'attempt_model_forwards', 'attempt_native_prefix_model_forwards',
    'attempt_model_TF_forwards') + tuple('pilot_' + key for key in COUNT_FIELDS)


def file_sha(source, relative):
    return source.files[relative]['source_sha256']


def _number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            'R13 finite numeric value required: ' + name)
    return float(value)


def _equal(actual, declared, context):
    if isinstance(actual, dict):
        require(isinstance(declared, dict) and set(declared) == set(actual), 'R13 declared fields differ: ' + context)
        for key in actual:
            _equal(actual[key], declared[key], context + '.' + key)
    elif isinstance(actual, (list, tuple)):
        require(isinstance(declared, (list, tuple)) and len(actual) == len(declared), 'R13 declared list differs: ' + context)
        for i, (a, b) in enumerate(zip(actual, declared)):
            _equal(a, b, context + '[' + str(i) + ']')
    elif isinstance(actual, bool) or isinstance(actual, str):
        require(type(actual) is type(declared) and actual == declared, 'R13 declared value differs: ' + context)
    else:
        near(actual, declared, 'R13 ' + context)


def admit_cells(scope, source, metric, scorer):
    """Verify actual cells; safe to call on a partial inventory for testing."""
    totals, admitted, seen = Counter(), {}, set()
    for key in MANDATORY_COUNTS:
        totals[key] = 0
    for cell in scope['cells']:
        ident, kind = cell['cell_id'], cell['kind']
        require(ident not in admitted, 'R13 duplicate actual cell ID')
        if kind == 'new_round13':
            result = adapter.validate_new_cell(cell, source, metric, scorer)
        else:
            require(kind in ('new_round12', 'reused_reference', 'pilot'), 'R13 unknown cell origin')
            result = frozen.validate_cell(cell, source, metric, scorer, fixture=False, require_producer=True)
        if kind != 'reused_reference':
            triple = tuple(file_sha(source, cell['paths'][key]) for key in frozen.CORE_FILES)
            require(triple not in seen, 'R13 same actual producer triple counted twice')
            seen.add(triple)
            prefix = 'pilot_' if kind == 'pilot' else 'new_'
            for key in COUNT_FIELDS:
                totals[prefix + key] += result[key]
            if kind in ('new_round13', 'new_round12'):
                benchmark = result.get('evaluation_phase') == 'observed_fixed500_comparison'
                role = ('r13_benchmark_' if benchmark else 'r13_method_') if kind == 'new_round13' else 'r12_reference_'
                for key in COUNT_FIELDS:
                    totals[role + key] += result[key]
                    if kind == 'new_round13' and not benchmark:
                        totals['r13_development_method_' + key] += result[key]
                manifest = source.doc(cell['paths']['manifest'])
                require(set(result['selected_rows']) == set(manifest['expected_ids']),
                        'R13 new method/reference producer cannot hide answers behind quality_ids')
                if kind == 'new_round12':
                    require(manifest['config']['candidate']['mode'] == 'R',
                            'R13 extra new_round12 rows must be unchanged R reference completion')
        else:
            totals['reused_selected_view_rows'] += len(result['selected_rows'])
        # Keep only scalar resource evidence after the heavy per-row owner
        # proof. All row sources have already been independently validated.
        for raw, _, _ in source.lines(cell['paths']['records']):
            if raw['id'] not in result['selected_rows']:
                continue
            row = result['selected_rows'][raw['id']]
            if kind != 'reused_reference':
                row['evaluation_phase'] = result.get('evaluation_phase', 'development')
            n = raw.get('actual_visual_token_count')
            if isinstance(n, int) and n > 0:
                row['actual_visual_token_count'] = n
                row['original_KV_elements'] = 28 * 4 * 128 * 2 * n
            for key in ('prefill_ms', 'question_prefill_ms'):
                if key in raw:
                    row[key] = raw[key]
            owner = raw.get('owner_before_suffix', {})
            if 'text_kv_bytes' in owner:
                row['text_kv_bytes'] = owner['text_kv_bytes']
            if kind == 'new_round13' and raw['status'] == 'PASS':
                guard = raw['round13_physical_budget']
                row['strict_cap_bytes'] = guard['strict_cap_bytes']
                row['cap_violated'] = raw['stored_visual_KV_bytes'] > row['strict_cap_bytes']
                trace = raw.get('round13_free_generation_trace', {})
                steps, tokens = trace.get('steps', []), raw['output_token_ids']
                require(len(steps) == trace.get('step_count') == len(tokens) > 0
                        and trace.get('all_output_steps_observed') is True
                        and trace.get('used_for_encoder_or_decoding_decisions') is False,
                        'R13 free generation trace incomplete or used by a different policy')
                for i, step in enumerate(steps):
                    require(step['step'] == i, 'R13 generation trace step ordering differs')
                    near(_number(step['top1_logit'], 'top1') - _number(step['top2_logit'], 'top2'),
                         step['top1_top2_margin'], 'R13 raw generation margin')
                row['generation_trace_complete'] = True
                row['generation_trace_steps'] = len(steps)
        admitted[ident] = result
    return admitted, totals


def recompute_views(scope, admitted, scorer, reporter):
    """Recompute all declared quality and comparison views without a seal."""
    views, quality, members_used = {}, [], defaultdict(set)
    for view in scope.get('views', []):
        ident = view['view_id']
        require(ident not in views and view.get('members'), 'R13 duplicate/empty quality view')
        require(type(view.get('bootstrap', 2000)) is int and view.get('bootstrap', 2000) >= 1000,
                'R13 quality CI requires at least1000 source-cluster draws')
        values = {}
        for member in view['members']:
            require(member['cell_id'] in admitted, 'R13 quality member cell absent')
            cell = admitted[member['cell_id']]
            require(not cell['pilot_excluded'], 'R13 excluded pilot cannot re-enter quality')
            chosen = member.get('ids', list(cell['selected_rows']))
            require(len(chosen) == len(set(chosen)), 'R13 member repeats one answer ID')
            for row_id in chosen:
                require(row_id in cell['selected_rows'] and row_id not in values,
                        'R13 duplicate/missing head-tail answer ID in quality view')
                values[row_id] = cell['selected_rows'][row_id]
                members_used[member['cell_id']].add(row_id)
        require(values and all(row['dataset'] == view['dataset'] for row in values.values()), 'R13 mixed/empty quality domain')
        phases = {row['evaluation_phase'] for row in values.values() if row.get('evaluation_phase')}
        require(len(phases) <= 1, 'R13 development and observed500 new answers cannot share one quality view')
        phase = next(iter(phases)) if phases else view.get('evaluation_phase', 'development')
        require(view.get('evaluation_phase', phase) == phase, 'R13 declared evaluation phase differs')
        if phase == 'observed_fixed500_comparison':
            require(view['label'].startswith('FIXED500_'), 'R13 observed500 must have a distinct benchmark view label')
            if scope.get('research_cutoff_complete') is True:
                require(len(values) == 500 and len(view.get('expected_ids', [])) == 500,
                        'R13 final observed benchmark view must close all500 explicitly')
        if view.get('expected_ids') is not None:
            require(len(view['expected_ids']) == len(set(view['expected_ids']))
                    and set(view['expected_ids']) == set(values), 'R13 declared view ID coverage differs')
        complete = all(row['score_status'] == 'SCORED_PASS' for row in values.values())
        mean = sum(float(row['score']) for row in values.values()) / len(values) if complete else None
        interval = scorer.cluster_bootstrap(list(values.values()), 'score', view.get('bootstrap', 2000),
                    view.get('seed', 20261125))['ci95'] if complete else None
        q = {'view_id': ident, 'label': view['label'], 'dataset': view['dataset'], 'questions': len(values),
             'evaluation_phase': phase, 'independent_holdout_claimed': False,
             'source_clusters': len({row['source_cluster_id'] for row in values.values()}),
             'primary_mean': mean, 'primary_ci95': interval,
             'score_status_counts': dict(Counter(row['score_status'] for row in values.values()))}
        for field in RESOURCE_FIELDS:
            nums = [_number(row[field], field) for row in values.values() if row.get(field) is not None]
            full = len(nums) == len(values)
            q[field] = {'n': len(nums), 'complete_coverage': full,
                'mean': statistics.mean(nums) if full else None,
                'median': statistics.median(nums) if full else None,
                'max': max(nums) if full else None} if nums else None
        raw = list(values.values())
        full_elements = all(isinstance(row.get('original_KV_elements'), int)
                            and isinstance(row.get('stored_visual_KV_bytes'), (int, float)) for row in raw)
        q['element_weighted_effective_visual_bits'] = (8 * sum(row['stored_visual_KV_bytes'] for row in raw)
            / sum(row['original_KV_elements'] for row in raw)) if full_elements else None
        audited_caps = [row for row in raw if 'cap_violated' in row]
        traces = [row for row in raw if 'generation_trace_complete' in row]
        q.update(cap_audited_rows=len(audited_caps), cap_violations=sum(row['cap_violated'] for row in audited_caps),
                 generation_trace_audited_rows=len(traces),
                 generation_trace_missing_or_short=sum(not row['generation_trace_complete'] for row in traces))
        for key, value in q.items():
            if key in view:
                _equal(value, view[key], 'declared view ' + ident + '.' + key)
        views[ident] = values
        quality.append(q)
    for ident, cell in admitted.items():
        if cell['kind'] in ('new_round13', 'new_round12'):
            require(members_used[ident] == set(cell['selected_rows']), 'R13 new task answers omitted from every quality view')
    headers = {view['view_id']: view for view in scope.get('views', [])}
    phases_by_view = {row['view_id']: row['evaluation_phase'] for row in quality}
    pairs, seen = [], set()
    for pair in scope.get('comparisons', []):
        a, b = pair['candidate_view'], pair['reference_view']
        require(a in views and b in views and a != b, 'R13 missing/self paired view')
        require((a, b) not in seen and headers[a]['dataset'] == headers[b]['dataset'], 'R13 duplicate/cross-domain pair')
        require(phases_by_view[a] == phases_by_view[b], 'R13 benchmark and development views cannot form a mixed-phase pair')
        if phases_by_view[a] == 'observed_fixed500_comparison' and scope.get('research_cutoff_complete') is True:
            require(set(views[a]) == set(views[b]) and len(views[a]) == 500,
                    'R13 final benchmark pair must use exactly the same500 questions')
        seen.add((a, b))
        require(pair.get('bootstrap', 2000) >= 1000, 'R13 paired CI requires at least1000 cluster draws')
        actual = reporter.paired(views[a], views[b], pair.get('bootstrap', 2000), pair.get('seed', 20261125))
        for key, value in actual.items():
            if key in pair and key not in ('bootstrap', 'seed'):
                _equal(value, pair[key], 'declared pair ' + a + '/' + b + '.' + key)
        pairs.append({'candidate_view': a, 'reference_view': b,
            'candidate': headers[a]['label'], 'reference': headers[b]['label'], 'dataset': headers[a]['dataset'],
            **actual})
    return quality, pairs, views


def _producer_support(item, obj, source):
    p = item.get('producer', {})
    require({'launch', 'job', 'snapshot_inventory', 'entrypoint'} <= set(p), 'R13 QA0 actual producer binding omitted')
    launch, job, inventory = (source.doc(p[key]) for key in ('launch', 'job', 'snapshot_inventory'))
    source.original(p['job'], launch['job_sha256'])
    source.original(p['snapshot_inventory'], launch['source_inventory_sha256'])
    driver = next((value for value in job['argv'] if isinstance(value, str) and value.endswith('.py')), None)
    keys = [key for key in inventory if driver and driver.replace('\\', '/').endswith('/' + key)]
    require(len(keys) == 1, 'R13 QA0 driver missing from actual snapshot inventory')
    source.original(p['entrypoint'], inventory[keys[0]])
    support = p.get('support_files', []) + [p['entrypoint']]
    available = {file_sha(source, relative) for relative in support}
    sources = obj.get('source_sha256', {})
    if isinstance(sources, str):
        require(sources in available, 'R13 QA0 actual source body absent')
    else:
        for logical, digest in sources.items():
            key = logical if '/' in logical else 'code/' + logical
            require(inventory.get(key) == digest and digest in available, 'R13 QA0 source body/snapshot differs')
    for key in ('legacy_source_sha256', 'executed_legacy_source_sha256', 'locks', 'locked_config_and_asset_sha256'):
        for digest in obj.get(key, {}).values():
            require(digest in available, 'R13 QA0 executed source or locked asset omitted')
    if obj.get('config_sha256'):
        require(p.get('source_config'), 'R13 QA0 source config omitted')
        source.original(p['source_config'], obj['config_sha256'])
    return inventory


def _readback_anchor(item, relative, source):
    """Require independent evidence-object identity, not a bare local hash."""
    receipt = source.doc(item['readback_file'])
    require(isinstance(receipt, dict), 'R13 QA0 explicit independent capture receipt required')
    objects = receipt.get('objects', [])
    digest = file_sha(source, relative)
    matches = [row for row in objects if row.get('sha256') == digest
               and row.get('remote_independent_SHA_before_equals_after') is True]
    if not matches and receipt.get('remote_hashes_computed_independently') is True:
        matches = [row for row in receipt.get('files', []) if row.get('sha256') == digest]
    require(matches, 'R13 QA0 evidence lacks an independent remote SHA anchor')
    source.original(relative, digest)


def verify_qa0(scope, source, totals):
    results, ids = [], set()
    for item in scope.get('qa0', []):
        ident, kind = item['qa0_id'], item['kind']
        require(ident not in ids and kind in ('packed_cuda', 'diagnostic_S0', 'calibration', 'model_TF'),
                'R13 duplicate/unknown QA0 item')
        ids.add(ident)
        obj = source.doc(item['file'])
        require(str(obj.get('status', '')).startswith('PASS'), 'R13 failed QA0 belongs to attempts, not successful QA0')
        require(obj.get('task_answer_events', obj.get('new_free_QA', item.get('new_free_QA'))) == 0,
                'R13 QA0 cannot conceal terminal answers')
        _readback_anchor(item, item['file'], source)
        _producer_support(item, obj, source)
        if kind == 'packed_cuda':
            require(obj.get('actual_CUDA') is True and obj.get('model_calls', 0) == 0, 'R13 packed QA0 CUDA/model scope differs')
            cases = obj.get('checks', obj.get('attention', obj.get('attention_checks', obj.get('attention_cases', []))))
            if not cases and 'dense_max_abs' in obj:
                cases = [obj]
            require(cases, 'R13 packed QA0 has no numerical attention case')
            for case in cases:
                if 'dense_max_abs' in case:
                    require(0 <= case['dense_max_abs'] <= case['dense_tolerance'] <= .008,
                            'R13 QA0 actual dense error exceeds declared bounded tolerance')
            totals['QA0_cuda_attention_cases'] += len(cases)
        elif kind == 'diagnostic_S0':
            require(obj.get('schema') == 'trajkv.round13.stage0.v1' and obj.get('actual_CUDA') is True
                    and obj.get('model_parameter_count') == 7944373760
                    and obj.get('references_loaded') is False
                    and obj.get('question_queries_used_only_for_diagnostic_not_production_selection') is True
                    and obj.get('image_only_selector_frozen_before_diagnostic_results') is True,
                    'R13 S0 whole-model/query-information contract differs')
            declared = obj['sources']
            records = item.get('records', [])
            require(len(declared) == obj['completed_sources'] == obj['expected_sources'] == len(records) > 0,
                    'R13 S0 actual source/record coverage differs')
            require([row['id'] for row in records] == [row['id'] for row in declared], 'R13 S0 record ordering differs')
            for record, expected in zip(records, declared):
                row = source.doc(record['file'])
                _readback_anchor(item, record['file'], source)
                require(row['id'] == expected['id'] and row['status'] == expected['status'] == 'PASS_DIAGNOSTIC_ONLY',
                        'R13 S0 local record/source status differs')
            source.original(item['inputs'], obj['input_sha256'])
            inputs = source.lines(item['inputs'])
            require([row['id'] for row, _, _ in inputs] == [row['id'] for row in declared]
                    and all(not set(row) & {'answers', 'refs', 'references', 'labels', 'scores'} for row, _, _ in inputs),
                    'R13 S0 inputs changed or contain reference answers')
            totals['QA0_native_prefix_model_forwards'] += int(obj['native_prefix_forwards'])
            totals['QA0_model_TF_forwards'] += int(obj['native_question_TF_forwards'])
            totals['QA0_S0_sources'] += len(records)
        elif kind == 'calibration':
            totals['QA0_native_prefix_model_forwards'] += int(obj.get('native_prefix_model_forwards', 0))
            totals['QA0_model_TF_forwards'] += int(obj.get('actual_model_forwards', 0))
        else:
            require(type(item.get('actual_model_forwards')) is int and item['actual_model_forwards'] >= 0,
                    'R13 TF actual model-forward accounting missing')
            totals['QA0_model_TF_forwards'] += item['actual_model_forwards']
        results.append({'qa0_id': ident, 'kind': kind, 'file': item['file'], 'new_free_QA': 0})
    return results


def verify_attempts(scope, source, totals):
    results, seen = [], set()
    for item in scope.get('attempts', []):
        ident = item['attempt_id']
        require(ident not in seen, 'R13 duplicate closed attempt')
        seen.add(ident)
        require(item['successful_terminal_answers'] == 0 and item.get('files') and item.get('reason'),
                'R13 terminal answers or unsupported failed attempt hidden from cells')
        models = item.get('actual_model_forwards')
        tasks = item.get('recorded_task_rows')
        require(type(models) is int and models >= 0 and type(tasks) is int and tasks == 0,
                'R13 non-quality attempt requires explicit model and zero task-row accounting')
        for relative in item['files']:
            source.path(relative)
        if item['disposition'] in ('PRE_MODEL_FAILED', 'NOT_LAUNCHED'):
            require(models == 0, 'R13 pre-model/unlaunched attempt performed model forwards')
        else:
            require(item['disposition'] in ('TECHNICAL_FAILED', 'CLOSED_ABORTED'), 'R13 unclosed attempt')
            producer = item.get('producer', {})
            require({'launch', 'job', 'snapshot_inventory', 'entrypoint'} <= set(producer),
                    'R13 failed executed attempt actual producer evidence omitted')
            _producer_support(item, {}, source)
        if item.get('accounting_file'):
            obj = source.doc(item['accounting_file'])
            _readback_anchor(item, item['accounting_file'], source)
            observed = int(obj.get('native_prefix_forwards', obj.get('native_prefix_model_forwards', 0)))
            observed += int(obj.get('native_question_TF_forwards', obj.get('actual_model_forwards', obj.get('model_calls', 0))))
            require(models == observed and obj.get('task_answer_events', obj.get('new_free_QA', 0)) == 0,
                    'R13 closed failed-attempt model counters differ from captured evidence')
            totals['attempt_native_prefix_model_forwards'] += int(obj.get('native_prefix_forwards', obj.get('native_prefix_model_forwards', 0)))
            totals['attempt_model_TF_forwards'] += int(obj.get('native_question_TF_forwards', obj.get('actual_model_forwards', obj.get('model_calls', 0))))
        elif models:
            require(False, 'R13 positive failed-attempt model work requires captured accounting_file')
        totals['closed_non_quality_attempts'] += 1
        totals['attempt_model_forwards'] += models
        results.append({'attempt_id': ident, 'disposition': item['disposition'], 'actual_model_forwards': models,
                        'recorded_task_rows': tasks, 'successful_terminal_answers': 0, 'reason': item['reason']})
    return results


def verify_research_decision(scope, admitted, qa0, attempts, totals, source):
    decision = scope.get('research_decision')
    require(isinstance(decision, dict) and decision.get('schema') == 'trajkv.round13.research_cutoff_decision.v1'
            and decision.get('status') == 'CLOSED' and decision.get('reason')
            and decision.get('independent_confirmation_claimed') is False,
            'R13 explicit closed research cutoff/observed-data decision required')
    groups = scope.get('groups', [])
    by_id = {group['group_id']: group for group in groups}
    require(groups and len(by_id) == len(groups)
            and set(decision['required_group_ids']) == set(by_id), 'R13 required research groups omitted or repeated')
    require(set(decision['required_qa0_ids']) == {row['qa0_id'] for row in qa0}
            and set(decision['required_attempt_ids']) == {row['attempt_id'] for row in attempts},
            'R13 QA0 or failed attempts missing from cutoff ledger')
    require(all(len(decision[key]) == len(set(decision[key])) for key in
                ('required_group_ids', 'required_qa0_ids', 'required_attempt_ids')),
            'R13 cutoff ledger repeats a required group/QA0/attempt ID')
    require({'packed_cuda', 'diagnostic_S0'} <= {row['kind'] for row in qa0}, 'R13 actual packed and S0 closure required')
    grouped = Counter()
    for group in groups:
        for ident in group.get('cell_ids', []):
            require(ident in admitted, 'R13 group references unknown actual cell')
            grouped[ident] += 1
        if group['disposition'] == 'NOT_RUN':
            require(not group['cell_ids'] and group.get('quality') is None and group.get('reason'),
                    'R13 unrun preparation must retain zero cells/null quality and actual reason')
            totals['not_run_groups'] += 1
        elif group['disposition'] == 'STOPPED':
            require(group.get('reason'), 'R13 stopped branch must explain closed cutoff')
        else:
            require(group['disposition'] == 'EXECUTED' and group['cell_ids'], 'R13 executed group has no actual cell')
    for ident, result in admitted.items():
        if result['kind'] != 'reused_reference':
            require(grouped[ident] == 1, 'R13 new producer must belong to exactly one closed research group')
    require(decision.get('plan_bindings'), 'R13 cutoff omitted executed/prepared plan identities')
    actual_names = {}
    for cell in scope['cells']:
        if cell['kind'] == 'reused_reference':
            continue
        name = cell.get('producer', {}).get('cell_name', cell['cell_id'])
        require(name not in actual_names, 'R13 actual planned producer name repeated')
        actual_names[name] = cell
    unrun_names = set()
    for group in groups:
        if group['disposition'] in ('NOT_RUN', 'STOPPED'):
            declared = group.get('planned_cell_names', [])
            require(len(declared) == len(set(declared)), 'R13 repeated unrun planned cell name')
            unrun_names.update(declared)
    require(not (unrun_names & set(actual_names)), 'R13 executed task labeled as unrun preparation')
    for entry in decision['plan_bindings']:
        source_digest = entry.get('sha256')
        require(isinstance(source_digest, str) and len(source_digest) == 64, 'R13 decision plan SHA missing')
        source.original(entry['file'], source_digest)
        obj = source.doc(entry['file'])
        declared = obj.get('all_entries', obj.get('entries', [])) if isinstance(obj, dict) else []
        if entry.get('role') == 'source_context':
            continue
        for task in declared:
            name = task['name']
            require(name in actual_names or name in unrun_names,
                    'R13 bound plan task disappeared from executed or explicit unrun/stopped ledger: ' + name)
            if name in actual_names:
                cell = actual_names[name]
                require(admitted[cell['cell_id']]['producer_rows'] == task['expected_QA'],
                        'R13 actual task count differs from bound executed plan')
                source.original(cell['producer']['source_config'], task['config_sha256'])
    # Actual parent receipts expose every completed child, even if a scope
    # builder accidentally omits one cell from both views and accounting.
    seen_parent = set()
    for cell in actual_names.values():
        producer = cell.get('producer', {})
        parent = producer.get('parent_complete')
        if producer.get('kind') != 'sequence_child' or not parent or parent in seen_parent:
            continue
        seen_parent.add(parent)
        for child in source.doc(parent).get('completed_cells', []):
            require(child['name'] in actual_names, 'R13 actual completed parent child omitted from stage ledger')
    return decision


def verify_report(report, quality, pairs, counts):
    """Validate every declared score, CI, resource and accounting field."""
    require(isinstance(report.get('quality'), list) and isinstance(report.get('paired'), list)
            and isinstance(report.get('counts'), dict), 'R13 report lacks quality/paired/counts')
    actual_quality = {row['view_id']: row for row in quality}
    require(len(report['quality']) == len(actual_quality), 'R13 report omits quality view')
    seen = set()
    for declared in report['quality']:
        ident = declared['view_id']
        require(ident in actual_quality and ident not in seen, 'R13 unknown/repeated report view')
        seen.add(ident)
        _equal(actual_quality[ident], declared, 'report quality ' + ident)
    actual_pairs = {(row['candidate_view'], row['reference_view']): row for row in pairs}
    require(len(report['paired']) == len(actual_pairs), 'R13 report omitted paired comparison')
    seen = set()
    for declared in report['paired']:
        key = declared['candidate_view'], declared['reference_view']
        require(key in actual_pairs and key not in seen, 'R13 unknown/repeated report pair')
        seen.add(key)
        _equal(actual_pairs[key], declared, 'report pair ' + '/'.join(key))
    require(set(report['counts']) == set(counts), 'R13 report count fields differ from complete actual ledger')
    require(all(type(value) is int and value >= 0 for value in report['counts'].values()),
            'R13 report event counts must be nonnegative integers')
    _equal(dict(counts), report['counts'], 'report counts')


def verify_semantics(scope, source, *, allow_fixture=False):
    require(allow_fixture is False, 'R13 final semantic closure never promotes CPU fixtures')
    adapter.check_final_header(scope)
    for relative, digest in frozen.FROZEN_METRIC_SOURCES.items():
        source.original(relative, digest)
    for relative in ('code/metrics.py', 'code/vendor_textvqa_metric.py', 'code/score.py', 'code/round12_report_v1.py'):
        source.path(relative)
    with adapter.metric_modules(source) as (metric, scorer, reporter):
        admitted, totals = admit_cells(scope, source, metric, scorer)
        quality, pairs, views = recompute_views(scope, admitted, scorer, reporter)
    qa0 = verify_qa0(scope, source, totals)
    attempts = verify_attempts(scope, source, totals)
    decision = verify_research_decision(scope, admitted, qa0, attempts, totals, source)
    totals['QA0_plus_attempt_native_prefix_forwards'] = totals['QA0_native_prefix_model_forwards'] + totals['attempt_native_prefix_model_forwards']
    totals['QA0_plus_attempt_model_TF_forwards'] = totals['QA0_model_TF_forwards'] + totals['attempt_model_TF_forwards']
    totals['QA0_plus_attempt_model_forwards'] = totals['QA0_native_prefix_model_forwards'] + totals['QA0_model_TF_forwards'] + totals['attempt_model_forwards']
    for binding in decision['plan_bindings']:
        source.original(binding['file'], binding['sha256'])
    expected = scope.get('expected_accounting')
    require(isinstance(expected, dict) and set(expected) == set(totals), 'R13 expected ledger must cover all actual count fields')
    _equal(dict(totals), expected, 'Root expected accounting')
    verify_report(source.doc(scope['report_summary']), quality, pairs, totals)
    sealed = scope['stage_state'] == 'ROOT_SEALED' and isinstance(scope.get('root_seal'), dict)
    if sealed:
        seal = scope['root_seal']
        require(seal.get('schema') == 'trajkv.round13.root_seal.v1'
                and frozen.canonical(adapter.payload(scope)) in
                    (seal.get('source_payload_sha256'), seal.get('public_payload_sha256')),
                'R13 final Root payload seal differs')
    return {'status': 'PASS_R13_ACTUAL_LOCAL_SEMANTICS', 'fixture_only': False,
        'publication_eligible': sealed, 'actual_producer_binding_required': True,
        'counts': dict(totals), 'quality': quality, 'paired': pairs, 'QA0': qa0, 'attempts': attempts,
        'research_decision': decision, 'physical_rows_verified': sum(
            row['status'] == 'PASS' for cell in admitted.values() for row in cell['selected_rows'].values()),
        'private_tensors_or_GPU_replayed': False, 'whole_external_library_completion_claimed': False}
