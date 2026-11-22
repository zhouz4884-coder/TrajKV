"""Portable stdlib reaggregation of existing D160 KRA/low raw scalars.

The earlier math summary was produced by an inline analysis, not this file.
This new implementation independently rebuilds its aggregate numbers. Paths
and byte hashes are recorded only as identities of the files actually opened;
no original E path/hash or private tensor is needed for numerical replay.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

DOMAINS = {'textvqa': 64, 'infovqa': 64, 'docvqa': 16, 'chartqa': 16}
TOLERANCE = 1e-12
SUMMARY_SCHEMA = 'trajkv.round10.existing_raw_math_summary.v1'
IDENTITY_KEYS = ('input_ids_sha256', 'native_position_ids_sha256',
                 'original_image_sha256', 'actual_visual_token_count')


def require(condition, message):
    if not condition:
        raise ValueError('KRA_REAGGREGATE: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def rows(path):
    with Path(path).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            require(line.strip(), 'blank raw JSONL row')
            yield number, json.loads(line), hashlib.sha256(line).hexdigest()


def aggregate(values):
    require(values, 'nonempty source set')
    return {
        'sources': len(values),
        'sources_with_changed_alpha': sum(r['changed_K_alpha_elements'] > 0 for r in values),
        'changed_K_alpha_elements_total': sum(r['changed_K_alpha_elements'] for r in values),
        'changed_K_alpha_elements_mean_per_source': statistics.mean(r['changed_K_alpha_elements'] for r in values),
        'source_equal_mean_changed_fraction': statistics.mean(r['changed_fraction_of_original_K23'] for r in values),
        'source_equal_mean_H_error_relative_decrease': statistics.mean(r['H_error_relative_decrease_per_source'] for r in values),
        'min_source_H_error_relative_decrease': min(r['H_error_relative_decrease_per_source'] for r in values),
        'max_source_H_error_relative_decrease': max(r['H_error_relative_decrease_per_source'] for r in values),
        'actual_H_after_worse_tokens': sum(r['actual_H_after_worse_tokens'] for r in values),
        'sources_with_identical_task_output_tokens': sum(r['task_output_token_ids_equal_original'] for r in values),
        'sources_with_identical_raw_predictions': sum(r['task_raw_prediction_equal_original'] for r in values),
        'metric_bytes_increment_unique': sorted({r['complete_byte_increment_vs_original'] for r in values}),
        'active_metric_units_unique': sorted({r['active_metric_units'] for r in values}),
        'actual_nonzero_gamma_values_unique': sorted({g for r in values for g in r['actual_nonzero_gamma_values']})}


def reaggregate(harvest_root):
    harvest_root = Path(harvest_root).resolve()
    raw_root = harvest_root / 'raw'
    require(raw_root.is_dir(), '--harvest-root must contain raw/')
    values, sources = [], []
    for dataset, count in DOMAINS.items():
        candidate = raw_root / f'KRA1_{dataset}_{count}_v1/records.jsonl'
        control = raw_root / f'R7low_{dataset}_{count}_v1/records.jsonl'
        old = {}
        for line_number, row, _ in rows(control):
            require(row['status'] == 'PASS' and row['dataset'] == dataset and row['id'] not in old,
                    'unique PASS original low source/ID required')
            # Drop the large ownership/metadata tables after extracting needed
            # scalar identity fields; no native tensor or image is opened.
            old[row['id']] = {key: row[key] for key in (*IDENTITY_KEYS, 'id', 'dataset', 'source_id',
                'source_cluster_id', 'output_token_ids', 'raw_prediction', 'stored_visual_KV_bytes')}
        require(len(old) == count, 'complete original low domain required')
        found = set()
        for line_number, row, line_sha in rows(candidate):
            identity = row['id']
            require(row['status'] == 'PASS' and row['dataset'] == dataset and identity in old and identity not in found,
                    'unique same-ID PASS KRA domain required')
            found.add(identity)
            original = old[identity]
            require(all(row[key] == original[key] for key in (*IDENTITY_KEYS, 'source_id', 'source_cluster_id')),
                    'paired input/source/position identity differs')
            detail = row['r7_bridge']['round10_kra_candidate']
            units = detail['per_unit_alpha_changes']
            require(detail['mode'] == 'KRA1' and len(units) == 23 and
                    len({u['unit_index'] for u in units}) == 23, 'K23 unique actual unit records required')
            active = [unit for unit in units if unit['gamma'] > 0]
            require(active and all(math.isfinite(unit['gamma']) for unit in units), 'finite nonzero metric units')
            before = sum(unit['actual_H_original_error_sum'] for unit in active)
            after = sum(unit['actual_H_selected_error_sum'] for unit in active)
            require(math.isfinite(before) and math.isfinite(after) and before >= after >= 0 and before > 0,
                    'valid declared deployed-coordinate H errors')
            n = row['actual_visual_token_count']
            changed = sum(unit['changed_alpha_elements'] for unit in units)
            require(type(n) is int and n > 0 and 0 <= changed <= 23*n and
                    changed == detail['changed_K_alpha_elements'], 'actual alpha change counters differ')
            worse = sum(unit['actual_H_after_worse_tokens'] for unit in active)
            require(worse == detail['actual_H_after_worse_tokens'] == 0, 'deployed coordinate error worsened')
            values.append({'dataset': dataset, 'id': identity, 'source_id': row['source_id'],
                'visual_tokens': n, 'active_metric_units': len(active),
                'actual_nonzero_gamma_values': sorted({unit['gamma'] for unit in active}),
                'changed_K_alpha_elements': changed, 'all_K23_alpha_elements': 23*n,
                'changed_fraction_of_original_K23': changed/(23*n),
                'H_error_sum_over_active_units_original': before,
                'H_error_sum_over_active_units_selected': after,
                'H_error_relative_decrease_per_source': 1-after/before,
                'actual_H_after_worse_tokens': worse,
                'task_output_token_ids_equal_original': row['output_token_ids'] == original['output_token_ids'],
                'task_raw_prediction_equal_original': row['raw_prediction'] == original['raw_prediction'],
                'complete_byte_increment_vs_original': row['stored_visual_KV_bytes']-original['stored_visual_KV_bytes'],
                'opened_candidate_member': candidate.relative_to(harvest_root).as_posix(),
                'opened_candidate_line_number': line_number,
                'actual_opened_line_sha256': line_sha})
        require(found == set(old), 'complete paired candidate/control IDs required')
        for path, role in ((candidate, 'KRA1'), (control, 'R7low')):
            sources.append({'member_relative_to_harvest_root': path.relative_to(harvest_root).as_posix(),
                'role': role, 'dataset': dataset, 'rows': count, 'actual_opened_file_bytes': path.stat().st_size,
                'actual_opened_file_sha256': sha(path)})
    require(len(values) == 160 and len({(r['dataset'], r['source_id']) for r in values}) == 160 and
            Counter(r['dataset'] for r in values) == Counter(DOMAINS), 'source-balanced complete D160 required')
    result = {'schema': SUMMARY_SCHEMA,
        'all_sources_equal_weighted': aggregate(values),
        'per_domain': {dataset: aggregate([r for r in values if r['dataset'] == dataset]) for dataset in DOMAINS}}
    return result, values, sources


def compare(expected, actual):
    checks = []
    def visit(reference, value, path):
        if isinstance(reference, dict):
            require(isinstance(value, dict) and set(reference) == set(value), 'aggregate keys differ at ' + path)
            for key in reference:
                visit(reference[key], value[key], path + '.' + key)
        elif isinstance(reference, list):
            require(isinstance(value, list) and len(reference) == len(value), 'aggregate list differs at ' + path)
            for index, (left, right) in enumerate(zip(reference, value)):
                visit(left, right, path + '[' + str(index) + ']')
        elif type(reference) is int:
            require(type(value) is int and value == reference, 'integer statistic differs at ' + path)
            checks.append({'field': path, 'exact_integer_match': True})
        elif type(reference) is float:
            require(type(value) in (int, float) and math.isfinite(reference) and math.isfinite(value) and
                    math.isclose(reference, value, rel_tol=0., abs_tol=TOLERANCE), 'numeric statistic differs at ' + path)
            checks.append({'field': path, 'expected': reference, 'recomputed': value, 'absolute_difference': abs(value-reference)})
        else:
            require(value == reference, 'non-numeric statistic differs at ' + path)
    require(expected.get('schema') == SUMMARY_SCHEMA and actual['schema'] == SUMMARY_SCHEMA,
            'original math aggregate schema required')
    for key in ('all_sources_equal_weighted', 'per_domain'):
        visit(expected[key], actual[key], key)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--harvest-root', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'NEW output directory required; existing math evidence is never overwritten')
    expected_before = sha(args.expected)
    original = read(args.expected)
    actual, values, sources = reaggregate(args.harvest_root)
    checks = compare(original, actual)
    require(sha(args.expected) == expected_before, 'expected aggregate changed during replay')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output/'RECOMPUTED_AGGREGATES.json', actual)
    write_new(args.output/'RECOMPUTED_PER_SOURCE_SCALARS.json', values)
    receipt = {'schema': 'trajkv.round10.kra_reaggregate_receipt.v1',
        'status': 'PASS_EXISTING_RAW_SOURCE_EQUAL_AGGREGATES_RECOMPUTED',
        'expected_actual_opened_file_sha256': expected_before,
        'reaggregation_source_sha256': sha(__file__),
        'sources': sources, 'candidate_rows': 160, 'control_rows': 160,
        'source_equal_units': 160, 'statistics_checked': len(checks),
        'absolute_tolerance': TOLERANCE, 'relative_tolerance': 0., 'checks': checks,
        'recomputed_aggregate_sha256': sha(args.output/'RECOMPUTED_AGGREGATES.json'),
        'recomputed_per_source_sha256': sha(args.output/'RECOMPUTED_PER_SOURCE_SCALARS.json'),
        'portable_numeric_replay_requires_original_E_paths_or_original_raw_byte_SHA': False,
        'recorded_input_hashes_describe_actual_opened_bodies_not_original_public_fingerprints': True,
        'prior_inline_analysis_was_not_a_saved_script': True,
        'new_GPU_jobs_started': 0, 'new_model_forwards': 0, 'task_scoring_performed': False,
        'native_PT_WO_images_or_weights_opened': False, 'original_expected_file_modified': False}
    write_new(args.output/'RECOMPUTATION_RECEIPT.json', receipt)
    print(json.dumps({'status': receipt['status'], 'statistics_checked': len(checks),
        'sources': 160, 'candidate_rows': 160, 'control_rows': 160,
        'receipt_sha256': sha(args.output/'RECOMPUTATION_RECEIPT.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
