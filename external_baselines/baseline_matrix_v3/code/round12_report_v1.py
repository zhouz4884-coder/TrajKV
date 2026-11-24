"""Round12 offline dataset lock and paired task/resource report.

Use unchanged code/score.py for official task scores after independently
anchored remote retrieval. This module never fabricates retrieval receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics


def require(value, message):
    if not value:
        raise ValueError('ROUND12_REPORT: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def ensure_score_lock(directory):
    """Produce the exact inputs/refs file lock required by frozen score.py."""
    directory = Path(directory)
    inputs = read(directory / 'ROUND12_INPUT_MANIFEST_v1.json')
    scoring = read(directory / 'ROUND12_SCORING_MANIFEST_v1.json')
    require(inputs['schema'] == 'trajkv.round12.inputs.v1'
            and scoring['schema'] == 'trajkv.round12.offline_scoring_inputs.v1', 'wrong data manifest schemas')
    files = {}
    for name, entry in inputs['input_files'].items():
        ref = scoring['reference_files'][name]
        for filename, expected in ((name, entry), (ref['file'], ref)):
            p = directory / filename
            require(sha(p) == expected['sha256'] and p.stat().st_size == expected['bytes'], 'data bytes changed: ' + filename)
            files[filename] = {'sha256': expected['sha256'], 'bytes': expected['bytes']}
    lock = {'schema': 'trajkv.round12.scoring_source_lock.v1', 'files': files,
            'input_manifest_sha256': sha(directory / 'ROUND12_INPUT_MANIFEST_v1.json'),
            'scoring_manifest_sha256': sha(directory / 'ROUND12_SCORING_MANIFEST_v1.json'),
            'inference_or_GPU_performed': False, 'official_scoring_implementation': 'unchanged code/score.py'}
    path = directory / 'source_lock.json'
    if path.exists():
        require(read(path) == lock, 'existing source lock differs; never overwrite')
    else:
        write_new(path, lock)
    return {'source_lock': str(path), 'sha256': sha(path), 'files': len(files)}


def percentile(values, q):
    values = sorted(values)
    x = q * (len(values) - 1)
    lo, hi = int(x), min(int(x) + 1, len(values) - 1)
    return values[lo] + (x - lo) * (values[hi] - values[lo])


def paired(a, b, repetitions=2000, seed=20261124):
    common = sorted(set(a) & set(b))
    require(bool(common), 'empty paired comparison')
    groups = defaultdict(list)
    improve = worsen = unchanged = 0
    for ident in common:
        x, y = a[ident], b[ident]
        require(x['source_cluster_id'] == y['source_cluster_id'], 'paired source differs: ' + ident)
        if x['score_status'] != 'SCORED_PASS' or y['score_status'] != 'SCORED_PASS':
            return {'n': len(common), 'primary_delta_pp': None, 'ci95_pp': None,
                    'status': 'INCOMPLETE_PAIRED_COVERAGE_NO_SUCCESS_SUBSET_PRIMARY'}
        require(x['metric_id'] == y['metric_id'], 'paired metric differs: ' + ident)
        delta = float(x['score']) - float(y['score'])
        groups[x['source_cluster_id']].append(delta)
        improve += delta > 0
        worsen += delta < 0
        unchanged += delta == 0
    keys = sorted(groups)
    sums = [sum(groups[k]) for k in keys]
    sizes = [len(groups[k]) for k in keys]
    mean = sum(sums) / sum(sizes)
    ci = None
    if len(keys) >= 2:
        rng = random.Random(seed)
        draws = []
        for _ in range(repetitions):
            ix = [rng.randrange(len(keys)) for _ in keys]
            draws.append(sum(sums[i] for i in ix) / sum(sizes[i] for i in ix))
        ci = [100 * percentile(draws, p) for p in (.025, .975)]
    return {'n': len(common), 'source_clusters': len(keys), 'primary_delta_pp': 100 * mean,
            'ci95_pp': ci, 'improved': improve, 'worsened': worsen, 'unchanged': unchanged,
            'candidate_mean_on_common': sum(float(a[i]['score']) for i in common) / len(common),
            'reference_mean_on_common': sum(float(b[i]['score']) for i in common) / len(common),
            'status': 'DESCRIPTIVE_DEVELOPMENT_NOT_CONFIRMATION',
            'bootstrap': {'repetitions': repetitions, 'seed': seed, 'unit': 'source_cluster_id'}}


def build_report(plan_path, output):
    """Plan cells explicitly label current-stage new versus reused score views."""
    plan = read(plan_path)
    require(plan['schema'] == 'trajkv.round12.comparison_plan.v1', 'unknown comparison plan')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    indexed = defaultdict(dict)
    events = []
    for cell in plan['cells']:
        require(cell['origin'] in ('new_round12', 'reused_reference'), 'explicit event origin required')
        directory = Path(cell['scored_dir'])
        summary = read(directory / 'summary.json')
        require(summary['integrity']['status'] == 'PASS_ALL_SHA_AND_EXPECTED_ID_CHECKS'
                and summary['integrity']['fixture_only'] is False,
                'only genuinely retrieved official-scored results admitted')
        path = directory / 'scored_records.jsonl'
        require(sha(path) == summary['derived_file_hashes']['scored_records.jsonl'], 'scored rows changed')
        values = rows(path)
        require(len(values) == summary['expected_run_questions'], 'scored coverage differs')
        selected = set(cell.get('ids') or [r['id'] for r in values])
        require(selected <= {r['id'] for r in values}, 'requested reference view IDs missing')
        key = (cell['label'], summary['dataset'])
        for row in values:
            if row['id'] not in selected:
                continue
            require(row['id'] not in indexed[key], 'duplicate score in method/domain; require explicit run repair selection')
            indexed[key][row['id']] = row
        events.append({'label': cell['label'], 'dataset': summary['dataset'], 'origin': cell['origin'],
                       'scored_view_rows': len(selected), 'source_raw_questions': summary['expected_run_questions'],
                       'summary_sha256': sha(directory / 'summary.json'),
                       'raw_hashes': summary['integrity']['raw_files']})
    quality = []
    for (label, dataset), values in sorted(indexed.items()):
        rs = list(values.values())
        complete = all(r['score_status'] == 'SCORED_PASS' for r in rs)
        result = {'label': label, 'dataset': dataset, 'questions': len(rs),
                  'source_clusters': len({r['source_cluster_id'] for r in rs}),
                  'primary_mean': sum(float(r['score']) for r in rs) / len(rs) if complete else None,
                  'score_status_counts': dict(Counter(r['score_status'] for r in rs)),
                  'primary_is_full_selected_coverage': complete}
        for field in ('stored_visual_KV_bytes', 'actual_effective_visual_bits', 'encode_ms', 'wall_ms',
                      'ttft_ms', 'peak_allocated_bytes'):
            v = [float(r[field]) for r in rs if isinstance(r.get(field), (int, float))]
            result[field] = {'n': len(v), 'complete_coverage': len(v) == len(rs),
                             'mean': sum(v) / len(v) if len(v) == len(rs) else None,
                             'median': statistics.median(v) if len(v) == len(rs) else None,
                             'max': max(v) if len(v) == len(rs) else None,
                             'conditional_available_mean': sum(v) / len(v)} if v else None
        quality.append(result)
    comparisons = []
    labels = sorted({k[0] for k in indexed})
    for label in labels:
        for reference in plan.get('references', ['R7low', 'R7high', 'FP16']):
            if label == reference:
                continue
            for dataset in sorted({k[1] for k in indexed}):
                if (label, dataset) in indexed and (reference, dataset) in indexed:
                    result = paired(indexed[(label, dataset)], indexed[(reference, dataset)],
                                    plan.get('bootstrap', 2000), plan.get('seed', 20261124))
                    comparisons.append({'candidate': label, 'reference': reference, 'dataset': dataset, **result})
    report = {'schema': 'trajkv.round12.development_report.v1', 'status': 'DESCRIPTIVE_CURRENT_COVERAGE',
              'quality': quality, 'paired': comparisons, 'scored_views': events,
              'new_scored_view_rows': sum(r['scored_view_rows'] for r in events if r['origin'] == 'new_round12'),
              'reused_reference_view_rows': sum(r['scored_view_rows'] for r in events if r['origin'] == 'reused_reference'),
              'view_rows_are_not_automatically_new_inference_events': True,
              'comparison_plan_sha256': sha(plan_path),
              'no_cross_domain_macro': True, 'raw_or_scoring_failures_not_removed_from_primary': True}
    write_new(output / 'summary.json', report)
    with (output / 'paired.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['dataset', 'candidate', 'reference', 'n', 'source_clusters', 'primary_delta_pp',
                  'ci95_pp', 'improved', 'worsened', 'unchanged', 'status']
        w = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(comparisons)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ensure-score-lock', type=Path)
    p.add_argument('--comparison-plan', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.ensure_score_lock:
        print(json.dumps(ensure_score_lock(a.ensure_score_lock), ensure_ascii=False, indent=2))
    else:
        require(a.comparison_plan is not None and a.output is not None, 'comparison plan/output required')
        report = build_report(a.comparison_plan, a.output)
        print(json.dumps({'output': str(a.output), 'quality_rows': len(report['quality']),
                          'paired_comparisons': len(report['paired'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
