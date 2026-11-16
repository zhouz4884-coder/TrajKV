"""Read-only source-distortion, compact-owner and depth-policy audit.

Source MSE ratios are codec diagnostics. They are never presented as ANLS,
attention-readout error, free-answer quality, or measured physical DRAM traffic.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


NAMED = {'incumbent', 'v_pair', 'v_mixed'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def audit_record(record):
    stages = [tuple(item) for item in record['depth_residual_selected_stages']]
    nv, owner = int(record['visual_tokens']), record['owned_initial']
    errors, seen = [], set()
    for item in stages:
        if len(item) != 4:
            errors.append('selected stage is not a quadruple')
            continue
        axis, layer, head, stage = item
        if axis not in ('k', 'v') or not isinstance(layer, int) or not isinstance(head, int) or min(layer, head) < 0 or stage not in (0, 1):
            errors.append('invalid axis/layer/head/depth')
        if item in seen:
            errors.append('duplicate stage')
        if stage == 1 and (axis, layer, head, 0) not in seen:
            errors.append('second stage lacks prior first stage')
        seen.add(item)
    stage_bytes = nv * 18
    expected_pool = len(stages) * stage_bytes
    full = int(owner['visual_fp16_denominator'])
    actual = sum(int(owner[key]) for key in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position'))
    actual += int(record['encoder_calibration_parameters_bytes'])
    fixed = actual - int(owner['visual_residual'])
    allowed = math.floor(float(record['depth_residual_budget_bits']) * full / 16)
    actual_bits = 16 * actual / full
    comparisons = {
        'stage_count_matches': len(stages) == int(record['depth_residual_stages']),
        'stage_bytes_matches': stage_bytes == int(record['depth_residual_stage_bytes']),
        'pool_exact_no_padding': int(owner['visual_residual']) == expected_pool,
        'complete_owner_matches': actual == int(record['depth_residual_complete_bytes']),
        'fixed_owner_matches': fixed == int(record['depth_residual_fixed_complete_bytes']),
        'allowed_bytes_matches': allowed == int(record['depth_residual_allowed_bytes']),
        'budget_cap_met': actual <= allowed,
        'actual_bits_matches': close(actual_bits, record['depth_residual_actual_budget_bits']),
        'budget_verified_metadata': record['depth_residual_budget_verified'] is True,
        'all_visual_base_retained': record['depth_residual_full_visual_support'] is True,
        'source_fp16_not_retained': record['depth_residual_source_fp16_retained'] is False,
        'lookup_in_selector_owner': int(owner['selector_metadata']) >= int(record['depth_residual_stage_lookup_bytes']),
        'policy_descriptor_five_bytes': int(record['depth_residual_encoder_policy_bytes']) == 5,
        'K_stage_count_matches': sum(axis == 'k' for axis, _, _, _ in stages) == int(record['depth_residual_K_stages']),
        'V_stage_count_matches': sum(axis == 'v' for axis, _, _, _ in stages) == int(record['depth_residual_V_stages'])}
    requested = record.get('depth_residual_requested_stages')
    if requested is None:
        requested = record['depth_residual_requested_units']
    comparisons['actual_priority_constant_matches'] = int(record['depth_residual_encoder_ranking_bytes']) == 2 * len(requested)
    unit_depth = defaultdict(int)
    for axis, layer, head, stage in stages:
        unit_depth[(axis, layer, head)] = max(unit_depth[(axis, layer, head)], stage + 1)
    lookup = int(record['depth_residual_stage_lookup_bytes'])
    comparisons['lookup_has_two_int32_stages_per_axis_unit'] = lookup % 16 == 0
    available_units = lookup // 8  # lookup has 2 int32 cells per axis/L/H unit.
    comparisons['pool_units_fit_lookup'] = len(unit_depth) <= available_units
    for key, passed in comparisons.items():
        if not passed:
            errors.append(key)
    source_rows = record.get('depth_residual_distortions', record.get('depth_distortions', []))
    curves = []
    distort_units = set()
    for source in source_rows:
        unit = (source['axis'], int(source['layer']), int(source['head']))
        depth, ds = int(source['depth']), [float(d) for d in source['squared_error_by_depth']]
        if unit in distort_units or depth != unit_depth.get(unit) or len(ds) != depth + 1:
            errors.append('distortion unit/depth is inconsistent with selected stages')
        distort_units.add(unit)
        if any(not math.isfinite(d) or d < 0 for d in ds):
            errors.append('distortion contains nonfinite/negative SSE')
        if depth == 2 and not close(source['alpha2'], ds[2] / max(ds[1], 1e-30)):
            errors.append('reported alpha2 does not equal actual D2/D1')
        curves.append(dict(axis=unit[0], layer=unit[1], head=unit[2], depth=depth,
            D0=ds[0] / (nv * 128), D1=ds[1] / (nv * 128),
            D2=ds[2] / (nv * 128) if depth == 2 else None))
    if distort_units != set(unit_depth):
        errors.append('source distortion diagnostics do not cover exactly selected units')
    variant = record.get('variant', record.get('depth_residual_policy', 'unknown'))
    row = dict(session_id=record['session_id'], image_id=record.get('image_id', record['session_id']),
        variant=variant, policy=record['depth_residual_policy'], visual_tokens=nv,
        stages=len(stages), K_stages=sum(axis == 'k' for axis, _, _, _ in stages),
        V_stages=sum(axis == 'v' for axis, _, _, _ in stages),
        complete_bytes=actual, allowed_bytes=allowed, unused_budget_bytes=allowed - actual,
        fixed_bytes=fixed, residual_bytes=expected_pool, stage_bytes=stage_bytes,
        lookup_bytes=lookup, ranking_bytes=int(record['depth_residual_encoder_ranking_bytes']),
        policy_bytes=int(record['depth_residual_encoder_policy_bytes']), actual_bits=actual_bits,
        legal_prefixes=not any('stage' in error or 'depth' in error for error in errors),
        passed=not errors, errors='; '.join(errors))
    count_rows = []
    for axis in ('k', 'v'):
        values = [depth for (unit_axis, _, _), depth in unit_depth.items() if unit_axis == axis]
        count_rows.append(dict(session_id=row['session_id'], image_id=row['image_id'], variant=variant,
            policy=row['policy'], visual_tokens=nv, axis=axis,
            depth0_units=available_units // 2 - len(values), depth1_units=values.count(1),
            depth2_units=values.count(2), selected_stages=sum(values)))
    reads = record.get('read_events', [])
    read_row = dict(session_id=row['session_id'], variant=variant, policy=row['policy'],
        visual_tokens=nv, recorded_layer_events=len(reads),
        calls=sum(int(event.get('calls', 0)) for event in reads),
        query_tokens=sum(int(event.get('query_tokens', 0)) for event in reads),
        **{key: sum(int(event.get(key, 0)) for event in reads) for key in
            ('base_visual_bytes', 'residual_bytes', 'text_bytes', 'selection_bytes', 'position_bytes')})
    return row, count_rows, curves, read_row, frozenset(unit for unit in unit_depth if unit[0] == 'k')


def summarize_curves(observations):
    buckets = defaultdict(list)
    for row in observations:
        buckets[(row['variant'], row['axis'], row['layer'], row['head'])].append(row)
    rows = []
    for (variant, axis, layer, head), group in sorted(buckets.items()):
        second = [row for row in group if row['D2'] is not None]
        sum0, sum1 = sum(r['D0'] for r in group), sum(r['D1'] for r in group)
        rows.append(dict(variant=variant, axis=axis, layer=layer, head=head,
            unique_image_contexts=len(group), depth2_contexts=len(second),
            mean_D0=sum0 / len(group), mean_D1=sum1 / len(group),
            D1_over_D0=sum1 / max(sum0, 1e-30),
            mean_D2=sum(r['D2'] for r in second) / len(second) if second else None,
            D2_over_D1=sum(r['D2'] for r in second) / max(sum(r['D1'] for r in second), 1e-30) if second else None,
            D2_over_D0=sum(r['D2'] for r in second) / max(sum(r['D0'] for r in second), 1e-30) if second else None))
    summary = []
    by_axis = defaultdict(list)
    for row in observations:
        by_axis[(row['variant'], row['axis'])].append(row)
    for (variant, axis), group in sorted(by_axis.items()):
        second = [row for row in group if row['D2'] is not None]
        ratios = [row['D2'] / max(row['D1'], 1e-30) for row in second]
        summary.append(dict(variant=variant, axis=axis, image_unit_contexts=len(group),
            stage2_image_unit_contexts=len(second),
            D1_over_D0=sum(r['D1'] for r in group) / max(sum(r['D0'] for r in group), 1e-30),
            D2_over_D1=sum(r['D2'] for r in second) / max(sum(r['D1'] for r in second), 1e-30) if second else None,
            stage2_ratio_min=min(ratios) if ratios else None,
            stage2_ratio_median=statistics.median(ratios) if ratios else None,
            stage2_ratio_max=max(ratios) if ratios else None))
    return rows, summary


def analyze(records):
    audits, counts, observations, reads, grouped = [], [], [], [], defaultdict(list)
    seen_records, seen_curves = set(), set()
    duplicate_curve_contexts = 0
    for record in records:
        row, unit_rows, curves, read_row, kset = audit_record(record)
        key = (row['variant'], row['session_id'])
        if key in seen_records:
            raise ValueError('Duplicate variant/session raw record: ' + str(key))
        seen_records.add(key)
        audits.append(row)
        counts.extend(unit_rows)
        reads.append(read_row)
        grouped[row['session_id']].append((row, kset))
        for curve in curves:
            curve = dict(curve, variant=row['variant'], image_id=row['image_id'], visual_tokens=row['visual_tokens'])
            # Multiple questions sharing one identical image prefix should not
            # multiply its source-distortion evidence. Distinct actual prefix
            # reconstructions remain separate contexts and are labelled so.
            signature = tuple(curve[key] for key in ('variant', 'image_id', 'visual_tokens', 'axis', 'layer', 'head', 'D0', 'D1', 'D2'))
            if signature in seen_curves:
                duplicate_curve_contexts += 1
                continue
            seen_curves.add(signature)
            observations.append(curve)
    comparisons = []
    for session_id, group in sorted(grouped.items()):
        named = [(row, kset) for row, kset in group if row['policy'] in NAMED]
        if len(named) < 2:
            continue
        same_cost = len({(row['visual_tokens'], row['complete_bytes'], row['stages'], row['ranking_bytes'], row['policy_bytes'], row['lookup_bytes']) for row, _ in named}) == 1
        same_k = len({kset for _, kset in named}) == 1
        comparisons.append(dict(session_id=session_id, policies=sorted(row['policy'] for row, _ in named),
            identical_complete_bytes_and_stage_count=same_cost,
            identical_incumbent_K_unit_set=same_k, passed=same_cost and same_k))
    curves, curve_summary = summarize_curves(observations)
    return dict(status='PASS' if all(row['passed'] for row in audits) and all(row['passed'] for row in comparisons) else 'FAIL',
        byte_rows=audits, counts=counts, curves=curves, curve_summary=curve_summary,
        crosspolicy=comparisons, logical_reads=reads, unique_distortion_contexts=len(observations),
        identical_image_context_references_deduplicated=duplicate_curve_contexts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [path / 'records.jsonl' if path.is_dir() else path for path in args.run]
    records = []
    for path in paths:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                record = json.loads(line)
                if 'depth_residual_selected_stages' in record:
                    records.append(record)
    if not records:
        raise ValueError('No actual depth-bank records in supplied runs')
    result = analyze(records)
    args.output.mkdir(parents=True, exist_ok=False)
    scope = 'Actual source-coordinate codec MSE only: K in centered WHT coordinates, V in native coordinates. Not attention-readout error or ANLS improvement.'
    write_json(args.output / 'byte_audit.json', dict(status=result['status'],
        records=result['byte_rows'], named_policy_comparisons=result['crosspolicy'],
        scope='Complete owner: base + compact selected stage pool + lookup + positions + encoder factors/ranking/policy. Named policies must preserve K units and complete bytes.'))
    write_csv(args.output / 'byte_audit.csv', result['byte_rows'])
    write_json(args.output / 'curves.json', dict(scope=scope, rows=result['curves'],
        per_axis_summary=result['curve_summary'], unique_image_unit_contexts=result['unique_distortion_contexts'],
        identical_image_context_references_deduplicated=result['identical_image_context_references_deduplicated']))
    write_csv(args.output / 'curves.csv', result['curves'])
    write_json(args.output / 'unit_counts.json', result['counts'])
    write_csv(args.output / 'unit_counts.csv', result['counts'])
    write_json(args.output / 'logical_reads.json', dict(rows=result['logical_reads'],
        scope='Recorded logical tensor payload by layer event; all visual base retained and selected residual stages fused into one full visual/text softmax. Not physical DRAM or elapsed runtime.'))
    write_csv(args.output / 'logical_reads.csv', result['logical_reads'])
    write_json(args.output / 'manifest.json', dict(schema='trajkv_round6_depth_analysis_v1', status=result['status'],
        inputs={str(path): sha(path) for path in paths}, source_sha256=sha(__file__),
        records=len(records), source_mse_scope=scope, raw_evidence_modified=False))
    write_json(args.output / 'complete.json', dict(status=result['status'], records=len(records),
        named_policy_comparisons=len(result['crosspolicy'])))
    print(json.dumps(dict(status=result['status'], records=len(records), named_policy_comparisons=len(result['crosspolicy']))), flush=True)
    if result['status'] != 'PASS':
        raise RuntimeError('Actual depth owner or named-policy matching audit failed')


if __name__ == '__main__':
    main()
