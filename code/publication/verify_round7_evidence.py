"""Local-only R7 numeric recount and calibration/failure scope separation.

Recomputes official per-answer scores and actual owner sums from raw records.
This does not publish, execute models, or turn cached-prefix NLL into E2E data.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/a2'), str(ROOT / 'code/analysis')]
from analyze_external import score_docvqa
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_round7_quality import selected_references

VISUAL = ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position',
          'spectrum_experiment_reference', 'rotation_packed_reference', 'hadamard_constant',
          'axis_packed_reference', 'calib_packed_reference', 'hybrid_packed_reference',
          'kvarn_packed_reference', 'anchor_visual_auxiliary')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value):
    text = str(value).replace('\\', '/')
    if '/TrajKV/' in text:
        return ROOT / text.split('/TrajKV/', 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require(value, message):
    if not value:
        raise ValueError(message)


def close(left, right, context):
    require(math.isfinite(float(left)) and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-10), context)


def actual_owner(record, external):
    owner = record['owned_initial']
    total = sum(int(owner.get(key, 0)) for key in VISUAL) + int(record.get('encoder_calibration_parameters_bytes', 0))
    denominator, nv = int(owner['visual_fp16_denominator']), int(record['visual_tokens'])
    require(denominator > 0 and nv > 0 and int(owner['visual_position']) == nv * 8, 'Invalid actual denominator/positions')
    for scope in (record, owner):
        for key in ('complete_visual_owner_bytes', 'depth_residual_complete_bytes'):
            if key in scope:
                require(int(scope[key]) == total, 'Declared complete owner does not match raw component recount')
    for key in ('actual_visual_bits_including_encoder', 'depth_residual_actual_budget_bits'):
        if key in record:
            close(record[key], 16 * total / denominator, 'Declared bit rate differs from raw owner')
    details = dict(complete=total, denominator=denominator, metric_bytes=0, physical_tail_bytes=0)
    if record['method'] not in ('round7_key', 'round7_tail_key'):
        return details
    variant = record['variant_config']
    factor_path, book_path = local_path(variant['metadata_factor_path']), local_path(variant['direction_book_path'])
    factor, book = read(factor_path), read(book_path)
    require(external[variant['metadata_factor_path']] == sha(factor_path)
            and external[variant['direction_book_path']] == sha(book_path), 'Original factor/book fingerprints changed')
    arrays = [factor.get(key) for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets')]
    arrays = [array for array in arrays if array is not None]
    layers, heads, words = len(arrays[0]), len(arrays[0][0]), len(book['centers'])
    lh = layers * heads
    require(denominator == lh * nv * 512 and words in (16, 256), 'Actual decoder/book geometry mismatch')
    index_bytes = 16 if words == 16 else 32
    require(record['key_index_bytes_per_token'] == index_bytes and record['key_book_words'] == words,
            'Actual nibble/byte index declaration mismatch')
    padded_capacity = ((nv + 31) // 32) * 32 if record['method'] == 'round7_tail_key' else nv
    window = nv if variant.get('v_group_size', 'all') == 'all' else int(variant['v_group_size'])
    require(window in (128, nv) and nv % window == 0, 'Invalid V metadata window')
    base = lh * (nv * (index_bytes + 2) + padded_capacity * 16) + lh * 256 + lh * 512 * (nv // window) + words * 8
    require(int(owner['visual_base']) == base, 'Raw base owner differs from K indices/gains/mean/V words/metadata/book shapes')
    factor_bytes = len(arrays) * lh * 4
    metric_bytes = 0
    if variant.get('key_metric_path'):
        metric_path = local_path(variant['key_metric_path'])
        metric = read(metric_path)
        require(sha(metric_path) == external[variant['key_metric_path']], 'Metric source changed')
        require(len(metric['u']) == layers and all(len(row) == heads and all(len(vector) == 128 for vector in row) for row in metric['u']), 'Metric geometry differs from model')
        metric_bytes = lh * 260
        require(int(record['key_metric_bytes']) == metric_bytes, 'Metric FP16-u/FP32-gamma owner omitted or doubled')
    stages = [tuple(row) for row in record['depth_residual_selected_stages']]
    seen = set()
    for axis, layer, head, stage in stages:
        require(axis in ('k', 'v') and 0 <= layer < layers and 0 <= head < heads and stage in (0, 1), 'Illegal residual stage')
        require((axis, layer, head, stage) not in seen and (stage == 0 or (axis, layer, head, 0) in seen), 'Residual prefix/uniqueness violation')
        seen.add((axis, layer, head, stage))
    require(int(owner['visual_residual']) == len(stages) * nv * 18 and int(owner['selector_metadata']) == lh * 16,
            'Residual/lookup exact tensor charge differs')
    constants = factor_bytes + metric_bytes + int(record['depth_residual_encoder_ranking_bytes']) + int(record['depth_residual_encoder_policy_bytes'])
    require(int(record['encoder_calibration_parameters_bytes']) == constants, 'Encoder constants are not charged exactly once')
    require(record.get('padded_visual_tokens', 0) == 0 and record.get('padded_visual_positions', 0) == 0,
            'Semantic token/position padding in actual packed evidence')
    tail = (padded_capacity - nv) * lh * 16
    if record['method'] == 'round7_tail_key':
        require(record.get('visual_positions_exact_original') is True and record.get('value_centroid_fit_real_tokens_only') is True,
                'Tail words changed real tokens or centroid-fit samples')
        require(int(record['value_tail_padding_actual_bytes']) == tail, 'Physical unused V tail storage was not charged')
    details.update(metric_bytes=metric_bytes, physical_tail_bytes=tail)
    return details


def reference_file(expected):
    matches = [path for path in (ROOT / 'configs/data').rglob('*') if path.is_file() and path.suffix in ('.json', '.jsonl') and sha(path) == expected]
    require(matches, 'Missing original reference fingerprint for independent recount')
    return matches[0]


def quality(path):
    path = path / 'summary.json' if path.is_dir() else path
    summary = read(path)
    require(summary.get('status') == 'COMPLETE' and summary.get('audit', {}).get('status') == 'PASS', 'Incomplete/unaudited quality analysis')
    dataset = summary.get('dataset') or ('text' if any(row.get('accuracy') is not None for row in summary['table']) else 'doc')
    require(dataset in ('doc', 'text'), 'Unknown quality metric domain')
    with (path.parent / 'scores.csv').open(encoding='utf-8-sig') as stream:
        scores = list(csv.DictReader(stream))
    references = selected_references(reference_file(summary['references_sha256']), {row['session_id'] for row in scores})
    metric, scored = TextVQAAccuracyEvaluator(), {}
    values = defaultdict(list)
    for row in scores:
        key = (row['session_id'], row['variant'], int(row['turn']))
        require(key not in scored, 'Duplicate scored answer')
        ref = references[key[0]]
        targets = (ref['answers'] if dataset == 'text' else next(x.get('answer', x.get('answers')) for x in ref['references'] if int(x['turn']) == key[2]))
        require(json.loads(row['reference']) == targets, 'Scored labels differ from original frozen selected reference')
        actual = (float(metric.eval_pred_list([dict(pred_answer=row['prediction'], gt_answers=targets)])) if dataset == 'text'
                  else score_docvqa(row['prediction'], targets)['score'])
        close(row['score'], actual, 'Raw official metric recount differs')
        scored[key] = row
        values[row['variant']].append(actual)
    selected_variants = {row['variant'] for row in summary['table']}
    sums, denominators, raw_keys, owners, run_receipts = defaultdict(int), defaultdict(int), set(), [], []
    for receipt in summary['audit']['runs']:
        run = local_path(receipt['run'])
        manifest = read(run / 'manifest.json')
        require(read(run / 'complete.json')['status'] == 'COMPLETE'
                and sha(run / 'manifest.json') == receipt['manifest_sha256']
                and sha(run / 'records.jsonl') == receipt['records_sha256'], 'Completed inference evidence changed')
        require('cached-prefix' not in str(manifest.get('quality_scope', '')).lower(), 'Cached-prefix calibration was mixed into E2E quality')
        external = manifest.get('external_codec_inputs', {})
        original_count, selected_count = 0, 0
        for line in (run / 'records.jsonl').read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            require('free_running' in record.get('protocol', '') and record.get('primary_quality_eligible') is not False
                    and record.get('task_accuracy_eligible') is not False, 'Non-free/calibration record entered quality')
            owner = actual_owner(record, external)
            original_count += 1
            if record['variant'] not in selected_variants:
                continue
            selected_count += 1
            sums[record['variant']] += owner['complete']; denominators[record['variant']] += owner['denominator']
            owners.append(owner)
            for turn in record['turns']:
                key = (record['session_id'], record['variant'], int(turn['turn']))
                require(key in scored and key not in raw_keys and scored[key]['prediction'] == turn['answer'], 'Raw answer and scored answer identities differ')
                raw_keys.add(key)
        if 'original_record_count' in receipt:
            require(original_count == int(receipt['original_record_count']), 'Original producer record count changed under analysis filtering')
            require(selected_count == int(receipt['analysis_selected_record_count']), 'Selected analysis record count differs from filter receipt')
        run_receipts.append(dict(run=str(run), records_sha256=sha(run / 'records.jsonl'), complete_sha256=sha(run / 'complete.json'),
            original_records_verified=original_count, analysis_selected_records=selected_count))
    require(raw_keys == set(scored), 'Raw/scored complete question coverage differs')
    for row in summary['table']:
        name = row['variant']
        reported = row.get('score', row.get('anls') if dataset == 'doc' else row.get('accuracy'))
        close(statistics.mean(values[name]), reported, 'Quality mean differs from raw answers')
        close(16 * sums[name] / denominators[name], row['budget_bits_including_encoder_constants'], 'Complete owner aggregate differs from summary')
    return dict(summary=str(path), summary_sha256=sha(path), scores_sha256=sha(path.parent / 'scores.csv'), status='PASS',
        dataset=dataset, analysis_role=summary.get('analysis_role'), scored_answers=len(scored), variants=len(values),
        metric_parameters_bytes_observed=sorted({x['metric_bytes'] for x in owners}),
        physical_tail_storage_bytes=sum(x['physical_tail_bytes'] for x in owners), raw_runs=run_receipts)


def calibration(run):
    manifest = read(run / 'manifest.json')
    require(read(run / 'complete.json').get('status') == 'COMPLETE', 'Incomplete calibration diagnostic')
    require('cached-prefix' in str(manifest.get('quality_scope', '')).lower(), 'Calibration scope must explicitly declare cached-prefix diagnostic')
    reports = []
    for path in run.glob('*.json'):
        value = read(path)
        if isinstance(value, dict) and 'primary_quality_eligible' in value:
            require(value['primary_quality_eligible'] is False and value.get('end_to_end_timing_eligible') is False,
                    'Calibration target report is mislabeled as E2E performance')
            reports.append(dict(path=path.name, sha256=sha(path)))
    return dict(path=str(run), manifest_sha256=sha(run / 'manifest.json'), complete_sha256=sha(run / 'complete.json'),
        scope=manifest['quality_scope'], target_reports=reports, primary_quality_eligible=False, end_to_end_timing_eligible=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analyses', nargs='+', type=Path, required=True)
    p.add_argument('--calibration-runs', nargs='*', type=Path, default=[])
    p.add_argument('--failed-runs', nargs='*', type=Path, default=[])
    p.add_argument('--freeze', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    checks = [quality(path) for path in args.analyses]
    calibrations = [calibration(path) for path in args.calibration_runs]
    quality_runs = {Path(run['run']).resolve() for check in checks for run in check['raw_runs']}
    failed = []
    for run in args.failed_runs:
        require(run.is_dir() and run.resolve() not in quality_runs, 'Failed smoke is absent or mixed into primary quality')
        receipt = read(run / 'complete.json') if (run / 'complete.json').is_file() else None
        require(receipt is None or receipt.get('status') != 'COMPLETE', 'Completed run cannot be relabeled as failed smoke')
        files = [dict(path=path.relative_to(run).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
                 for path in sorted(run.rglob('*')) if path.is_file()]
        failed.append(dict(path=str(run), files=files, scope='failed or partial smoke retained; excluded from task quality and primary timing'))
    frozen = None
    if args.freeze:
        value = read(args.freeze)
        require(value.get('schema') == 'trajkv.round7.method-freeze.v1', 'Unsupported freeze schema')
        for item in value['frozen_files']:
            require(sha(local_path(item['path'])) == item['sha256'], 'Frozen method input changed')
        frozen = dict(path=str(args.freeze), sha256=sha(args.freeze), status='UNCHANGED')
    result = dict(schema='trajkv.round7.local-evidence-verification.v1', status='PASS', quality_analyses=checks,
        calibration_diagnostics=calibrations, failed_smokes=failed, freeze=frozen,
        score_owner_numeric_recount=True, total_unique_not_added=True, cached_prefix_not_counted_as_E2E=True,
        public_write_performed=False, GPU_used=False, source_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(status='PASS', analyses=len(checks), scored_answers=sum(x['scored_answers'] for x in checks),
        calibration_runs=len(calibrations), failed_smokes=len(failed), public_write_performed=False)))


if __name__ == '__main__':
    main()
