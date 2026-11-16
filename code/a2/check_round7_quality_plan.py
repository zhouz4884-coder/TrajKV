"""CPU budget-algebra compatibility of the R7 independent audit and planner.

This does not execute a model, attention reader, GPU kernel, or quality metric.
The two independently implemented planners are compared on the declared grid.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from analyze_round7_quality import independent_plan
from round7_key_layout import plan_key_stages


DIMENSIONS = dict(geometry=[[28, 4], [32, 32]], real_visual_tokens=[128, 1792, 2438, 2448],
    book_words=[16, 256], metric_tensor_bytes=[0, 29120], layout_reference=['actual', 'k2'],
    depth_policy=['incumbent', 'v_pair', 'v_mixed'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    cases, differences = [], []
    grid = itertools.product(DIMENSIONS['geometry'], DIMENSIONS['real_visual_tokens'],
        DIMENSIONS['book_words'], DIMENSIONS['metric_tensor_bytes'],
        DIMENSIONS['layout_reference'], DIMENSIONS['depth_policy'])
    for (layers, heads), nv, words, metric, reference, policy in grid:
        rank = [(axis, layer, head) for layer in range(layers) for head in range(heads) for axis in ('k', 'v')]
        variant = dict(selected_units=rank, depth_policy=policy, budget_bits=1.8, v_group_size='all')
        if reference == 'k2':
            variant['residual_layout_reference'] = 'k2'
        physical_n = (nv + 31) // 32 * 32
        key_bytes = 16 if words == 16 else 32
        base = (layers * heads * (nv * (key_bytes + 2) + physical_n * 16)
                + layers * heads * 128 * 6 + words * 8 + nv * 8)
        factors, full = layers * heads * 12, layers * heads * nv * 512
        kwargs = dict(layers=layers, heads=heads, visual_tokens=nv, visual_full_bytes=full,
            base_bytes=base, factor_bytes=factors + metric, budget_bits=1.8,
            selected_units=rank, depth_policy=policy)
        row = dict(layers=layers, kv_heads=heads, real_visual_tokens=nv,
            physical_value_bit_capacity=physical_n, book_words=words,
            additional_encoder_metric_bytes=metric, layout_reference=reference, depth_policy=policy,
            factor_bytes=factors, full_visual_bytes=full, base_including_position_bytes=base,
            physical_tail_extra_bytes=(physical_n - nv) * layers * heads * 16)
        runtime, audit = None, None
        try:
            runtime = plan_key_stages(**kwargs, layout_reference='k2' if reference == 'k2' else None,
                key_index_bytes=key_bytes, book_bytes=words * 8, reference_factor_bytes=factors)
        except (ValueError, RuntimeError) as error:
            row['runtime_rejection'] = dict(type=type(error).__name__, message=str(error))
        try:
            audit = independent_plan(variant, layers, heads, nv, base, factors, metric, words, full)
        except (ValueError, RuntimeError) as error:
            row['audit_rejection'] = dict(type=type(error).__name__, message=str(error))
        if runtime is None or audit is None:
            row['status'] = 'MATCHED_REJECTION' if runtime is None and audit is None else 'DIFFERENCE'
            if row['status'] == 'DIFFERENCE':
                differences.append(dict(case=len(cases), field='rejection_disagreement'))
        else:
            fields = [('stages', 'selected_stages'), ('fixed', 'fixed_bytes'),
                ('complete', 'predicted_bytes'), ('allowed', 'allowed_bytes'),
                ('lookup', 'lookup_bytes'), ('stage_bytes', 'stage_bytes')]
            mismatch = [left for left, right in fields if audit[left] != runtime[right]]
            row.update(status='DIFFERENCE' if mismatch else 'MATCHED_ACCEPTANCE',
                selected_stage_count=len(audit['stages']), compared_fields=[left for left, _ in fields],
                exact_stage_tuples_equal=audit['stages'] == runtime['selected_stages'],
                audit_fixed_bytes=audit['fixed'], runtime_fixed_bytes=runtime['fixed_bytes'],
                audit_complete_bytes=audit['complete'], runtime_complete_bytes=runtime['predicted_bytes'],
                allowed_bytes=audit['allowed'])
            for field in mismatch:
                differences.append(dict(case=len(cases), field=field))
        cases.append(row)
    if len(cases) != 192:
        raise RuntimeError('Declared grid does not contain exactly 192 cases')
    rejected = sum(row['status'] == 'MATCHED_REJECTION' for row in cases)
    accepted = sum(row['status'] == 'MATCHED_ACCEPTANCE' for row in cases)
    return dict(schema='trajkv_round7_quality_plan_crosscheck_v1',
        status='PASS' if not differences and rejected == 24 and accepted == 168 else 'FAIL',
        scope='CPU budget algebra compatibility only; no model/GPU/reader/task-quality measurement',
        gpu_executed=False, model_executed=False, task_labels_used=False,
        dimensions=DIMENSIONS, expected_cases=192, actual_cases=len(cases),
        matched_acceptances=accepted, matched_rejections=rejected, differences=differences,
        difference_count=len(differences), budget_bits=1.8, value_group_size='all',
        ranking_construction='layer-major, head-major, interleaved K then V; complete 2*L*H units',
        metric_dimension_scope='0 or 29120 additional paid bytes exercises generic owner algebra; 29120 is the actual 28x4 rank-one tensor size, not a 32x32 metric-size claim',
        physical_tail_scope='ceil(real N/32) V words; all positions, key gains and residual stages use real N',
        sources={name: sha(Path(__file__).with_name(name)) for name in
            ('check_round7_quality_plan.py', 'analyze_round7_quality.py', 'round7_key_layout.py', 'round6_depth_codec.py')},
        cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in
        ('status', 'actual_cases', 'matched_acceptances', 'matched_rejections', 'difference_count')}), flush=True)
    if report['status'] != 'PASS':
        raise RuntimeError('Independent quality owner planner compatibility failed')


if __name__ == '__main__':
    main()
