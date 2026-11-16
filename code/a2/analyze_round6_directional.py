"""Correct signed-cost finite diagnostics and freeze four static direction ranks.

Read calibration artifacts only. The raw snapshot remains byte unchanged. Full
and Student gradients predict the same Student finite interventions. Comparisons
separate stage additions/removals, which have opposite marginal byte costs.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from configure_round6_controls import (NL, NH, CAPACITY, ranking_bytes, shape_owner)

ROOT = Path(__file__).resolve().parents[2]
NAMES = {(state, objective): 'r6_' + state + '_' + objective + '_direction'
         for state in ('full', 'student') for objective in ('ce', 'margin')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local_path(path):
    normalized = str(path).replace('\\', '/')
    return ROOT / normalized.split('/TrajKV/', 1)[1] if '/TrajKV/' in normalized else Path(path)


def action_key(row):
    action = row.get('action')
    action = action if isinstance(action, dict) else row
    return (action['axis'], int(action['layer']), int(action['head']))


def prediction_metrics(rows):
    predicted = np.array([row['predicted_improvement'] for row in rows], dtype=float)
    actual = np.array([row['actual_improvement'] for row in rows], dtype=float)
    sst = float(np.square(actual - actual.mean()).sum())
    return dict(actions=len(rows), R2=1 - float(np.square(actual - predicted).sum()) / sst if sst > 0 else None,
        strict_three_class_sign_hit_rate=float(np.mean(np.sign(predicted) == np.sign(actual))),
        positive_improvement_hit_rate=float(np.mean((predicted > 0) == (actual > 0))),
        mean_absolute_directional_error=float(np.abs(predicted - actual).mean()),
        actual_positive_fraction=float(np.mean(actual > 0)))


def group_metrics(rows):
    values = prediction_metrics(rows)
    pick = min(range(len(rows)), key=lambda i: (-rows[i]['predicted_improvement'], action_key(rows[i])))
    actual = [row['actual_improvement'] for row in rows]
    pairs = [(i, j) for i in range(len(rows)) for j in range(i) if actual[i] != actual[j]]
    correct, tied = 0., 0
    for i, j in pairs:
        difference = rows[i]['predicted_improvement'] - rows[j]['predicted_improvement']
        tied += difference == 0
        correct += .5 if difference == 0 else float(difference * (actual[i] - actual[j]) > 0)
    span = max(actual) - min(actual)
    values.update(same_signed_marginal_byte_cost=True, pairwise_untied_actual_pairs=len(pairs),
        pairwise_prediction_ties=tied,
        within_question_pairwise_order_accuracy=correct / len(pairs) if pairs else None,
        selected_action=list(action_key(rows[pick])),
        selected_action_actual_improvement=actual[pick], best_observed_action_improvement=max(actual),
        selected_action_regret=max(actual) - actual[pick],
        regret_over_observed_action_range=(max(actual) - actual[pick]) / span if span > 0 else None)
    return values


def analyze(folder):
    raw_path, case_path = folder / 'directional_scores.json', folder / 'cases.jsonl'
    manifest_path, complete_path = folder / 'manifest.json', folder / 'complete.json'
    raw, manifest = read(raw_path), read(manifest_path)
    cases = [json.loads(line) for line in case_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_case = {row['id']: row for row in cases}
    if (raw['status'] != 'COMPLETE' or read(complete_path)['status'] != 'COMPLETE'
            or raw['labels_used'] is not False or manifest['gold_answers_loaded'] is not False
            or len(cases) != 16 or len(by_case) != 16 or set(by_case) != set(manifest['selected_case_ids'])):
        raise ValueError('Expected complete, gold-free, distinct 16-case calibration artifact')
    settings = manifest['config']['mixed_bank_settings']
    source = local_path(manifest['config']['arguments']['samples'])
    if sha(source) != manifest['samples_sha256']:
        raise ValueError('Calibration runner identity changed')
    source_ids = {row['id'] for row in read(source)}
    if not set(by_case) <= source_ids:
        raise ValueError('Unknown calibration case identity')
    universe = {(axis, layer, head) for axis in ('k', 'v') for layer in range(NL) for head in range(NH)}
    lookup, by_spec = {}, defaultdict(list)
    for row in raw['directional_scores']:
        key = (row['id'], row['state'], row['objective'], row['position'], *action_key(row))
        if key in lookup or row['id'] not in by_case or action_key(row) not in universe:
            raise ValueError('Duplicate/unknown signed direction row')
        if not math.isfinite(float(row['predicted_improvement'])):
            raise ValueError('Nonfinite directional prediction')
        lookup[key] = row
        by_spec[(row['id'], row['state'], row['objective'], row['position'])].append(row)
    expected_specs = {(case_id, state, objective, position) for case_id in by_case
        for state in ('full', 'student') for objective, position in [('ce', None), ('margin', 0), ('margin', 1)]}
    if set(by_spec) != expected_specs or any(len(rows) != CAPACITY or set(map(action_key, rows)) != universe for rows in by_spec.values()):
        raise ValueError('Incomplete 224-unit Full/Student x CE/margin0/1 coverage')
    fixed_actions = {tuple(unit) for unit in manifest['config']['finite_interventions']}
    if len(fixed_actions) != 4:
        raise ValueError('Expected four prespecified finite actions')
    finite, reconstructed = [], 0
    raw_seen = set()
    for row in raw['finite_interventions']:
        case = by_case[row['id']]
        action = row['action']
        unit = action_key(row)
        if unit not in fixed_actions or not row['fresh_cache']:
            raise ValueError('Unexpected finite action/cache protocol')
        sign = int(action['after_stage']) - int(action['before_stage'])
        if sign not in (-1, 1):
            raise ValueError('Finite action is not one actual sign stage')
        delta_bytes = sign * 18 * int(case['visual_tokens'])
        if 'delta_bytes' in row and int(row['delta_bytes']) != delta_bytes:
            raise ValueError('Saved marginal bytes disagree with actual stage/N')
        reconstructed += 'delta_bytes' not in row
        key = (row['id'], row['objective'], row['position'], *unit)
        if key in raw_seen:
            raise ValueError('Duplicate finite action')
        raw_seen.add(key)
        for state in ('full', 'student'):
            direction = lookup[(row['id'], state, row['objective'], row['position'], *unit)]
            if any(direction[key] != action[key] for key in ('before_stage', 'after_stage')):
                raise ValueError('Gradient and finite intervention differ')
            if state == 'student' and not math.isclose(direction['predicted_improvement'], row['predicted_improvement'], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Raw Student finite prediction differs from signed row')
            finite.append(dict(id=row['id'], state=state, objective=row['objective'], position=row['position'],
                action=action, visual_tokens=case['visual_tokens'], delta_bytes=delta_bytes,
                delta_bytes_source='(after_stage-before_stage)*18*case_visual_tokens; reconstructed for legacy snapshot when absent',
                predicted_improvement=direction['predicted_improvement'], actual_improvement=row['actual_improvement'],
                actual_finite_baseline='student', gradient_state=state,
                prediction_source='raw finite Student prediction' if state == 'student' else 'Full-state signed direction row for identical Student codec delta'))
    if len(raw_seen) != len(by_case) * 4 * 3:
        raise ValueError('Incomplete four-action x16-case x3-objective finite coverage')
    groups = defaultdict(list)
    for row in finite:
        groups[(row['id'], row['state'], row['objective'], row['position'], row['delta_bytes'])].append(row)
    group_rows = [dict(id=key[0], state=key[1], objective=key[2], position=key[3], delta_bytes=key[4],
                       **group_metrics(rows)) for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))]
    summaries = []
    for state in ('full', 'student'):
        for objective, position in [('ce', None), ('margin', 0), ('margin', 1)]:
            for direction in ('add', 'remove'):
                rows = [row for row in finite if row['state'] == state and row['objective'] == objective
                        and row['position'] == position and (row['delta_bytes'] > 0) == (direction == 'add')]
                selected_groups = [g for g in group_rows if g['state'] == state and g['objective'] == objective
                    and g['position'] == position and (g['delta_bytes'] > 0) == (direction == 'add')]
                pair_count = sum(g['pairwise_untied_actual_pairs'] for g in selected_groups)
                accuracy = sum((g['within_question_pairwise_order_accuracy'] or 0) * g['pairwise_untied_actual_pairs'] for g in selected_groups)
                summaries.append(dict(state=state, objective=objective, position=position, direction=direction,
                    **prediction_metrics(rows), source_cases=len({row['id'] for row in rows}), matched_cost_groups=len(selected_groups),
                    within_question_pairwise_order_accuracy=accuracy / pair_count if pair_count else None,
                    pairwise_untied_actual_pairs=pair_count,
                    mean_matched_cost_selected_action_regret=float(np.mean([g['selected_action_regret'] for g in selected_groups])),
                    max_matched_cost_selected_action_regret=max(g['selected_action_regret'] for g in selected_groups),
                    R2_scope='pooled same objective/position and action direction; within-question group R2 also exported'))
    # Convert removal benefit into retention value before averaging. This
    # produces a common actual-stage1 value at every unit across case budgets.
    ranking_rows, orders = [], {}
    for (state, objective), name in NAMES.items():
        positions = [None] if objective == 'ce' else [0, 1]
        values = {}
        for unit in universe:
            case_values = []
            for case_id in by_case:
                position_values = []
                for position in positions:
                    row = lookup[(case_id, state, objective, position, *unit)]
                    position_values.append((-1 if row['before_stage'] == 1 else 1) * row['predicted_improvement'])
                case_values.append(float(np.mean(position_values)))
            values[unit] = float(np.mean(case_values))
        order = sorted(universe, key=lambda unit: (-values[unit], unit))
        orders[name] = [list(unit) for unit in order]
        ranking_rows.extend(dict(variant=name, rank=i + 1, unit=list(unit), mean_stage1_retention_value=values[unit],
                                 equal_weight_calibration_cases=len(by_case), objective_positions=positions)
                            for i, unit in enumerate(order))
    inputs = {path.relative_to(ROOT).as_posix():sha(path) for path in (raw_path, case_path, manifest_path, complete_path, source)}
    return dict(raw=raw, manifest=manifest, cases=cases, finite=finite, finite_groups=group_rows,
        finite_summary=summaries, rankings=ranking_rows, orders=orders, input_sha256=inputs,
        missing_delta_bytes_reconstructed=reconstructed, raw_sha256=sha(raw_path))


def freeze(data):
    source_path = ROOT / 'configs/a2/round5_v1/mixed_dev64.json'
    source = read(source_path)
    incumbent = next(v for v in source['variants'] if v['name'] == 'mixed_gradient_max')
    manifest = data['manifest']
    if manifest['config']['mixed_bank_settings'] != incumbent:
        raise ValueError('Calibration did not use the unchanged mixed incumbent')
    factors_path, book_path = local_path(incumbent['metadata_factor_path']), local_path(incumbent['direction_book_path'])
    if sha(factors_path) != manifest['metadata_sha256'] or sha(book_path) != manifest['encoder_book_sha256']:
        raise ValueError('Calibration base/factors/book identity mismatch')
    factor_bytes, book_bytes = 4 * read(factors_path)['parameter_count'], 2048
    # Validate shape arithmetic against the independent actual calibration bank
    # receipts, then prove each new ranking is same-cost for every prefix/N.
    for case in data['cases']:
        budget = case['budget']
        cost = shape_owner(incumbent['selected_units'], budget['residual_units'], case['visual_tokens'], factor_bytes, book_bytes)
        if cost['total_including_encoder'] != budget['complete_bytes'] or 18 * case['visual_tokens'] != budget['unit_bytes']:
            raise ValueError('Exact shape cost differs from actual calibration owner receipt')
    prefixes, budgets, variants = {}, {}, []
    for name, order in data['orders'].items():
        value = copy.deepcopy(incumbent)
        value.update(name=name, selected_units=order,
            selection_calibration_sha256=data['raw_sha256'])
        variants.append(value)
        prefixes[name], budgets[name] = [], []
        for prefix in range(CAPACITY + 1):
            owners = [shape_owner(order, prefix, n, factor_bytes, book_bytes) for n in (128, 256)]
            slope = (owners[1]['total_including_encoder'] - owners[0]['total_including_encoder']) // 128
            intercept = owners[0]['total_including_encoder'] - 128 * slope
            prefixes[name].append(dict(prefix=prefix, K_units=sum(unit[0] == 'k' for unit in order[:prefix]),
                V_units=sum(unit[0] == 'v' for unit in order[:prefix]), unique_units=len(set(map(tuple, order[:prefix]))),
                complete_bytes_intercept=intercept, complete_bytes_per_visual_token=slope,
                byte_formula='intercept + per_visual_token*N; includes encoder factors/ranking, dual lookup, positions and decoder book'))
        for nv in range(128, 4097, 128):
            for prefix in range(CAPACITY + 1):
                cost = shape_owner(order, prefix, nv, factor_bytes, book_bytes)
                if cost != shape_owner(incumbent['selected_units'], prefix, nv, factor_bytes, book_bytes):
                    raise ValueError('New ordering changes same-prefix complete owner')
            fixed = shape_owner(order, 0, nv, factor_bytes, book_bytes)
            allowed = math.floor(1.8 * fixed['visual_fp16_denominator'] / 16)
            count = min(CAPACITY, (allowed - fixed['total_including_encoder']) // (18 * nv))
            actual = shape_owner(order, count, nv, factor_bytes, book_bytes)
            budgets[name].append(dict(visual_tokens=nv, affordable_units=count,
                K_units=prefixes[name][count]['K_units'], V_units=prefixes[name][count]['V_units'],
                complete_bytes=actual['total_including_encoder'],
                actual_bits=16 * actual['total_including_encoder'] / actual['visual_fp16_denominator']))
    outputs = {}
    for count in (16, 64):
        outputs[ROOT / f'configs/a2/round6_v1/directional_doc{count}.json'] = dict(source,
            variants=copy.deepcopy(variants), start=0, count=count)
    report = dict(schema='trajkv.round6.signed-direction-static-rankings.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        calibration_case_ids=[case['id'] for case in data['cases']], calibration_cases=16,
        ranking_variants=list(data['orders']),
        ranking_rule='actual stage1 retention value = predicted add improvement for inactive units and minus predicted removal improvement for active units; margin first averages fixed positions0/1 within case; then equal case mean; descending value with lexicographic unit tie-break',
        reference_answers_read=False, dev_or_confirmation_outcomes_read=False,
        online_margin_trigger=False, policies_static=True,
        frozen_book_sha256=sha(book_path), frozen_factors_sha256=sha(factors_path),
        ranking_constant_bytes=448, dual_lookup_bytes=896, encoder_constants_bytes=factor_bytes + 448,
        same_complete_byte_cost_at_every_prefix=True,
        same_number_of_residual_units_at_each_legal_N=True,
        K_V_axis_sequence_may_differ_from_incumbent=True,
        scope='state/objective bridge in the same packed codec and <=1.8bit owner budget; this is not the axis-matched spread/random ablation',
        prefixes=prefixes, budget_costs=budgets,
        ranking_sha256={name:hashlib.sha256(ranking_bytes(order)).hexdigest() for name, order in data['orders'].items()},
        input_sha256=data['input_sha256'] | {source_path.relative_to(ROOT).as_posix():sha(source_path),
            factors_path.relative_to(ROOT).as_posix():sha(factors_path), book_path.relative_to(ROOT).as_posix():sha(book_path)},
        original_run_source_sha256=manifest['sources'],
        analysis_source_sha256=sha(__file__), owner_formula_source_sha256=sha(Path(__file__).with_name('configure_round6_controls.py')))
    return outputs, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'results/round6_v1/student/calibration16')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round6_v1/student/calibration16_analysis_v1')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data = analyze(args.input)
    outputs, report = freeze(data)
    summary = dict(schema='trajkv.round6.directional-finite-analysis.v1',
        finite_actual_baseline='Student decoded incumbent',
        full_prediction='Full-state gradient dotted with the identical Student intervention codec delta',
        student_prediction='raw saved Student prediction; exactly checked against signed direction row',
        missing_delta_bytes_reconstructed=data['missing_delta_bytes_reconstructed'],
        grouping=['case_id', 'state', 'objective', 'position', 'signed_delta_bytes'],
        finite_scope='four prespecified actions x16 calibration cases; each equal-cost action group contains two measured candidates; not an exhaustive or best-of-224 oracle',
        sign_rule='strict three-class sign and positive-improvement binary hit rate both reported; rank ties count half, actual ties omitted',
        R2_caveat='pooled R2 can reflect between-case scale; within-question groups and regret exported separately; no R2 gate',
        calibration_only=True, task_accuracy_eligible=False, finite_summary=data['finite_summary'],
        source_sha256=data['input_sha256'])
    outputs[args.output / 'summary.json'] = summary
    outputs[args.output / 'finite_groups.json'] = data['finite_groups']
    outputs[args.output / 'rankings.json'] = data['rankings']
    encoded = {path:json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8') for path, value in outputs.items()}
    encoded[args.output / 'finite_corrected.jsonl'] = ''.join(json.dumps(row) + '\n' for row in data['finite']).encode()
    report['configuration_sha256'] = {path.relative_to(ROOT).as_posix():hashlib.sha256(payload).hexdigest()
        for path, payload in encoded.items() if path.parent == ROOT / 'configs/a2/round6_v1'}
    audit_path = ROOT / 'audit/a2/round6_directional_v1.json'
    encoded[audit_path] = json.dumps(report, indent=2).encode()
    if not args.dry_run:
        for path in encoded:
            if path.exists():
                raise FileExistsError('Refusing to overwrite an existing frozen file: ' + str(path))
        for path, payload in encoded.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
    print(json.dumps(dict(status='DRY_RUN' if args.dry_run else 'FROZEN',
        variants=list(data['orders']), raw_direction_rows=len(data['raw']['directional_scores']),
        raw_finite_rows=len(data['raw']['finite_interventions']), corrected_state_rows=len(data['finite']),
        matched_cost_groups=len(data['finite_groups']), reconstructed_delta_bytes=data['missing_delta_bytes_reconstructed'],
        configs=list(report['configuration_sha256']), finite_summary=data['finite_summary'])))


if __name__ == '__main__':
    main()
