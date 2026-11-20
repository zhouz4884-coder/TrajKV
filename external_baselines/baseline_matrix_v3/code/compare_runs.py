"""Paired, source-cluster comparisons of immutable offline scored runs.

Quality is never selected according to a bytes threshold. Ineligible resource
points retain their entire-cohort score and are explicitly outside iso-budget.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scored(folder):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    raw = folder / 'scored_records.jsonl'
    assert sha(raw) == summary['derived_file_hashes']['scored_records.jsonl']
    rows = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == summary['expected_run_questions']
    assert len({r['id'] for r in rows}) == len(rows)
    return summary, {r['id']: r for r in rows}


def compare(anchor_dir, candidate_dir, draws=2000, seed=20261119):
    base, anchor = load_scored(anchor_dir)
    cand, candidate = load_scored(candidate_dir)
    assert base['metric_contract_version'] == cand['metric_contract_version']
    assert base['model_id'] == cand['model_id'] and base['dataset'] == cand['dataset']
    assert set(anchor) == set(candidate), 'Different cohorts cannot be a full paired comparison'
    same = ('model_commit', 'dataset_revision', 'split', 'question_id', 'original_image_sha256',
            'full_prompt_hash', 'question_suffix_sha256', 'actual_visual_token_count', 'source_cluster_id')
    failures, grouped, resource_rows = [], {}, []
    for identity, left in anchor.items():
        right = candidate[identity]
        for key in same: assert left.get(key) == right.get(key), f'INPUT_MISMATCH {identity} {key}'
        if left['score_status'] != 'SCORED_PASS' or right['score_status'] != 'SCORED_PASS':
            failures.append(identity); continue
        assert left['metric_id'] == right['metric_id']
        value = grouped.setdefault(left['source_cluster_id'], [0., 0])
        value[0] += right['score'] - left['score']; value[1] += 1
        resource_rows.append((left, right))
    result = {'schema': 'baseline_matrix.paired_cluster_comparison.v1',
              'status': 'PASS_PAIRED_COMPLETE' if not failures else 'INCOMPLETE_NO_PRIMARY_DIFFERENCE',
              'model_id': base['model_id'], 'dataset': base['dataset'], 'metric_ids': base['metric_ids'],
              'anchor_method': base['method'], 'candidate_method': cand['method'],
              'expected_QA': len(anchor), 'paired_scored_QA': len(resource_rows), 'failed_ids': failures,
              'anchor_summary_sha256': sha(anchor_dir / 'summary.json'),
              'candidate_summary_sha256': sha(candidate_dir / 'summary.json'),
              'anchor_scored_rows_sha256': sha(anchor_dir / 'scored_records.jsonl'),
              'candidate_scored_rows_sha256': sha(candidate_dir / 'scored_records.jsonl'),
              'statistical_rule': 'question-weighted paired differences; cluster bootstrap and cluster sign flips; CI crossing zero is not equivalence',
              'primary': None, 'created_unix': time.time()}
    if not failures:
        values = np.asarray(list(grouped.values()), dtype=np.float64)
        sums, counts = values[:, 0], values[:, 1]
        delta = float(sums.sum() / counts.sum())
        rng = np.random.default_rng(seed)
        estimates = []
        # Bounded memory for a complete split with thousands of source clusters.
        for start in range(0, draws, 100):
            indices = rng.integers(0, len(values), size=(min(100, draws - start), len(values)))
            estimates.extend((sums[indices].sum(1) / counts[indices].sum(1)).tolist())
        ci = np.percentile(estimates, [2.5, 97.5]).tolist() if len(values) > 1 else None
        observed = abs(float(sums.sum()))
        if len(values) <= 18:
            extreme = 0; total = 1 << len(values)
            for pattern in range(total):
                signed = sum(value * (1 if pattern & (1 << i) else -1) for i, value in enumerate(sums))
                extreme += abs(signed) >= observed - 1e-12
            p = extreme / total; p_type = 'exact_cluster_sign_flip'
        else:
            total = 10000; extreme = 0
            for start in range(0, total, 100):
                signs = rng.integers(0, 2, size=(min(100, total - start), len(values))) * 2 - 1
                extreme += int((np.abs(signs @ sums) >= observed - 1e-12).sum())
            p = (extreme + 1) / (total + 1); p_type = 'Monte_Carlo_cluster_sign_flip_plus_one'
        influence = (sums - delta * counts) / counts.mean()
        sem = float(influence.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
        result['primary'] = {'difference': delta, 'difference_pp': 100 * delta,
            'ci95': ci, 'ci95_pp': [100 * x for x in ci] if ci else None,
            'cluster_count': len(values), 'bootstrap_draws': draws, 'seed': seed,
            'sign_flip_p_unadjusted': float(p), 'sign_flip_type': p_type,
            'requires_family_adjustment_before_significance_claim': True,
            'pilot_MDE80_normal_approx_pp': 100 * 2.8015852181129683 * sem if sem is not None else None}
    if resource_rows:
        bits = [right['actual_effective_visual_bits'] for _, right in resource_rows]
        native = sum(left['stored_visual_KV_bytes'] for left, _ in resource_rows)
        stored = sum(right['stored_visual_KV_bytes'] for _, right in resource_rows)
        result['resources'] = {
            'scope': 'all successfully paired requests; not controlled timing',
            'complete_visual_bytes_ratio_to_native': stored / native,
            'mean_effective_bits': float(np.mean(bits)), 'min_effective_bits': min(bits), 'max_effective_bits': max(bits),
            'iso_budget': {str(target): {'all_QA_eligible': not failures and all(b <= target for b in bits),
                                        'over_budget_QA': sum(b > target for b in bits),
                                        'quality_never_recomputed_on_budget_selected_subset': True}
                           for target in (1.3, 1.8, 2.25)},
            'mean_instrumented_wall_ms': float(np.mean([r['wall_ms'] for _, r in resource_rows])),
            'wall_is_not_canonical_method_speed': True}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261119)
    args = parser.parse_args()
    value = compare(args.anchor, args.candidate, args.draws, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(value))


if __name__ == '__main__': main()
