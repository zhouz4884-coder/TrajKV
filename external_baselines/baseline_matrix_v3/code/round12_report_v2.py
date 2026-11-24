"""Round12 paired report with strict admission of published R11 score views.

V1 remains unchanged. Legacy R11 summaries have a different schema: admission
checks their remote source receipt, raw triplet, exact scored-row hash, and
recomputes selected scores with the CURRENT frozen metrics and separate refs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import round12_report_v1 as base
from metrics import METRIC_CONTRACT_VERSION, score_question

require, sha, read, rows, write_new, paired = base.require, base.sha, base.read, base.rows, base.write_new, base.paired


def load_cell(cell):
    directory = Path(cell['scored_dir'])
    summary = read(directory / 'summary.json')
    scored_path = directory / 'scored_records.jsonl'
    values = rows(scored_path)
    if 'integrity' in summary:
        require(summary['integrity']['status'] == 'PASS_ALL_SHA_AND_EXPECTED_ID_CHECKS'
                and summary['integrity']['fixture_only'] is False, 'unverified official-scored cell')
        require(sha(scored_path) == summary['derived_file_hashes']['scored_records.jsonl'], 'scored rows changed')
        require(len(values) == summary['expected_run_questions'], 'official-scored coverage differs')
        return summary, values
    require(cell['origin'] == 'reused_reference', 'legacy score schema cannot be new Round12 inference')
    require(summary.get('status') in ('PASS_FRESH_R128_SOURCE_SCORE', 'PASS_FRESH_ADDITIONAL384_SOURCE_SCORE',
                                      'PASS_FRESH_EXPAND384_SOURCE_SCORE'), 'unrecognized legacy source score status')
    require(sha(scored_path) == summary['scored_records_sha256'] and len(values) == summary['expected_QA'],
            'legacy scored bytes/coverage differ')
    raw_dir = directory.parent.parent / 'raw' / directory.name
    receipts = list(raw_dir.glob('source_retrieval_receipt_round11_*.json'))
    require(len(receipts) == 1 and sha(receipts[0]) == summary['source_receipt_sha256'],
            'legacy independent receipt identity differs')
    receipt = read(receipts[0])
    require(receipt.get('status') == 'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA'
            and receipt.get('remote_hashes_computed_independently') is True, 'legacy source lacks independent remote SHA')
    inventory = {x['file']: x for x in receipt['files']}
    raw_hashes = {}
    for name in ('manifest.json', 'records.jsonl', 'complete.json'):
        p = raw_dir / name
        require(name in inventory and sha(p) == inventory[name]['sha256']
                and p.stat().st_size == inventory[name]['bytes'], 'legacy raw anchor differs: ' + name)
        raw_hashes[name] = inventory[name]['sha256']
    complete, manifest = read(raw_dir / 'complete.json'), read(raw_dir / 'manifest.json')
    require(complete['manifest_sha256'] == raw_hashes['manifest.json']
            and complete['records_jsonl_sha256'] == raw_hashes['records.jsonl']
            and complete['coverage_complete'] is True, 'legacy raw triplet cross-binding differs')
    selected = set(cell.get('ids') or [r['id'] for r in values])
    references = {r['id']: r for r in rows(cell['refs'])}
    raw_selected = {}
    with (raw_dir / 'records.jsonl').open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            raw = json.loads(line)
            if raw['id'] in selected:
                raw_selected[raw['id']] = (raw, line_number, hashlib.sha256(line).hexdigest())
    require(selected <= set(raw_selected) and selected <= set(references), 'selected legacy source/reference missing')
    admitted = []
    for row in values:
        if row['id'] not in selected:
            continue
        raw, line_number, line_sha = raw_selected[row['id']]
        require(row['raw_line_number'] == line_number and row['raw_line_sha256'] == line_sha,
                'legacy score is not bound to exact raw line')
        require(row['source_cluster_id'] == raw['source_cluster_id'] == references[row['id']]['source_cluster_id']
                and row['output_token_ids'] == raw['output_token_ids']
                and row['raw_prediction'] == raw['raw_prediction'], 'legacy scored prediction/source differs')
        require(row['metric_contract_version'] == METRIC_CONTRACT_VERSION, 'legacy metric version differs')
        actual = score_question(raw['dataset'], raw['raw_prediction'], references[row['id']])
        require(actual['metric_id'] == row['metric_id'] and float(actual['score']) == float(row['score']),
                'legacy score does not match current frozen metric on paired refs')
        enriched = dict(raw, **row)
        enriched['round12_reused_score_admission'] = 'independent_raw_SHA_exact_line_and_current_metric_recompute'
        admitted.append(enriched)
    normalized = dict(summary, expected_run_questions=summary['expected_QA'],
                      integrity={'raw_files': raw_hashes, 'legacy_source_receipt_sha256': sha(receipts[0])})
    return normalized, admitted


def build_report(plan_path, output):
    plan = read(plan_path)
    require(plan['schema'] == 'trajkv.round12.comparison_plan.v1', 'unknown comparison plan')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    indexed, events = defaultdict(dict), []
    for cell in plan['cells']:
        require(cell['origin'] in ('new_round12', 'reused_reference'), 'explicit origin required')
        summary, values = load_cell(cell)
        selected = set(cell.get('ids') or [r['id'] for r in values])
        require(selected <= {r['id'] for r in values}, 'view IDs missing')
        key = (cell['label'], summary['dataset'])
        for row in values:
            if row['id'] in selected:
                require(row['id'] not in indexed[key], 'duplicate view; explicit repair selection required')
                indexed[key][row['id']] = row
        directory = Path(cell['scored_dir'])
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
            vs = [float(r[field]) for r in rs if isinstance(r.get(field), (int, float))]
            result[field] = {'n': len(vs), 'complete_coverage': len(vs) == len(rs),
                             'mean': statistics.mean(vs) if len(vs) == len(rs) else None,
                             'median': statistics.median(vs) if len(vs) == len(rs) else None,
                             'max': max(vs) if len(vs) == len(rs) else None} if vs else None
        quality.append(result)
    comparisons = []
    for label in sorted({k[0] for k in indexed}):
        for reference in plan.get('references', ['R7low', 'R7high', 'FP16']):
            if label == reference:
                continue
            for dataset in sorted({k[1] for k in indexed}):
                if (label, dataset) in indexed and (reference, dataset) in indexed:
                    comparisons.append({'candidate': label, 'reference': reference, 'dataset': dataset,
                        **paired(indexed[(label, dataset)], indexed[(reference, dataset)],
                                 plan.get('bootstrap', 2000), plan.get('seed', 20261124))})
    report = {'schema': 'trajkv.round12.development_report.v2', 'status': 'DESCRIPTIVE_CURRENT_COVERAGE',
              'quality': quality, 'paired': comparisons, 'scored_views': events,
              'new_scored_view_rows': sum(x['scored_view_rows'] for x in events if x['origin'] == 'new_round12'),
              'reused_reference_view_rows': sum(x['scored_view_rows'] for x in events if x['origin'] == 'reused_reference'),
              'view_rows_are_not_automatically_new_inference_events': True,
              'comparison_plan_sha256': sha(plan_path), 'no_cross_domain_macro': True,
              'raw_or_scoring_failures_not_removed_from_primary': True,
              'reporter_source_sha256': sha(__file__), 'paired_helper_source_sha256': sha(base.__file__)}
    write_new(output / 'summary.json', report)
    with (output / 'paired.csv').open('x', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=['dataset', 'candidate', 'reference', 'n', 'source_clusters',
            'primary_delta_pp', 'ci95_pp', 'improved', 'worsened', 'unchanged', 'status'],
            extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(comparisons)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--comparison-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = build_report(a.comparison_plan, a.output)
    print(json.dumps({'output': str(a.output), 'quality_rows': len(report['quality']),
                      'paired_comparisons': len(report['paired'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
