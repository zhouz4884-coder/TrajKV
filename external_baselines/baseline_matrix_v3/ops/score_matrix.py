"""CPU-only batch scoring of fixed matrix cells with immutable raw sources."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from score import score_run, validate_raw, read_json, require, sha
from compare_runs import compare


def existing_score_is_exact(folder, raw, refs):
    summary = read_json(folder / 'summary.json')
    validated = validate_raw(raw, refs)
    require(summary['integrity']['raw_files'] == validated['raw_hashes'], 'Existing score uses different raw')
    require(summary['integrity']['refs_sha256'] == sha(refs), 'Existing score uses different references')
    for name, digest in summary['derived_file_hashes'].items():
        require(sha(folder / name) == digest, 'Existing derived score hash changed')
    for name, digest in summary['integrity']['metric_code_hashes'].items():
        require(sha(ROOT / name) == digest, 'Existing score metric is stale')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--mirror-root', type=Path, default=Path('${EVIDENCE_ROOT}'))
    parser.add_argument('--score-version', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, help='Optional phase directory of original runs or derived prediction views')
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    matrix = read_json(args.matrix); phase = matrix['phase'].lower()
    score_root = args.mirror_root / 'results/scored' / args.score_version
    summary_root = ROOT / 'results/scored' / args.score_version
    rows, pending, summaries = [], [], {}
    for plan in matrix['rows']:
        stem = Path(plan['config']).stem
        raw = (args.raw_root or args.mirror_root / 'results' / phase) / stem
        is_derived = ((raw / 'manifest.json').is_file() and
                      read_json(raw / 'manifest.json').get('schema') == 'baseline_matrix.derived_prediction_view.v1')
        if not (raw / 'retrieval_receipt.json').is_file() and not is_derived:
            pending.append(stem)
            if args.available_only: continue
            raise FileNotFoundError(f'Raw mirror is not independently verified yet: {raw}')
        raw_manifest = read_json(raw / 'manifest.json')
        input_name = PurePosixPath(raw_manifest['config']['inputs'].replace('\\', '/')).name
        refs = ROOT / 'configs/datasets/v3' / input_name.replace('_inputs.jsonl', '_refs.jsonl')
        destination = score_root / stem
        if destination.exists(): summary = existing_score_is_exact(destination, raw, refs)
        else: summary = score_run(raw, refs, destination, summary_dir=summary_root / stem)
        summaries[(plan['model'], plan['method'], plan['dataset'])] = (summary, destination)
        source_rows = [json.loads(line) for line in (raw / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        valid = [row for row in source_rows if row['status'] == 'PASS']
        def mean(key):
            values = [row[key] for row in valid if row.get(key) is not None]
            return sum(values) / len(values) if values else None
        bits = [row['actual_effective_visual_bits'] for row in valid]
        complete_bytes = sum(row['complete_visual_owner_bytes'] for row in valid)
        denominator = sum(row['full_kv_byte_denominator'] for row in valid)
        row = dict(model=plan['model'], dataset=plan['dataset'], method=plan['method'], method_id=summary['method'],
                   phase=matrix['phase'], status=summary['status'], expected_QA=summary['expected_run_questions'],
                   successful_QA=len(valid), source_clusters=summary['groups']['overall']['primary']['source_clusters'],
                   primary_metric=','.join(summary['metric_ids']), quality_percent=summary['primary_mean_percent'],
                   quality_ci95_low_percent=(summary['primary_ci95_percent'] or [None, None])[0],
                   quality_ci95_high_percent=(summary['primary_ci95_percent'] or [None, None])[1],
                   ratio_of_sums_effective_bits=16 * complete_bytes / denominator if denominator else None,
                   mean_effective_bits=mean('actual_effective_visual_bits'),
                   min_effective_bits=min(bits) if bits else None, max_effective_bits=max(bits) if bits else None,
                   mean_complete_visual_bytes=mean('complete_visual_owner_bytes'),
                   mean_payload_bytes=mean('stored_visual_payload_bytes'), mean_coder_metadata_bytes=mean('coder_metadata_bytes'),
                   mean_positions_bytes=mean('positions_bytes'), mean_encoder_constants_bytes=mean('encoder_constants_bytes'),
                   mean_visual_tokens=mean('actual_visual_token_count'), mean_wall_ms=mean('wall_ms'),
                   mean_encode_ms=mean('encode_ms'), mean_prefill_ms=mean('prefill_ms'), mean_ttft_ms=mean('ttft_ms'),
                   mean_peak_allocated_bytes=mean('peak_allocated_bytes'),
                   timing_scope='instrumented_quality_prototype_not_controlled_speed',
                   scored_summary_sha256=sha(destination / 'summary.json'),
                   scored_rows_sha256=sha(destination / 'scored_records.jsonl'),
                   raw_records_sha256=summary['integrity']['raw_files']['records.jsonl'])
        for target in (1.3, 1.8, 2.25):
            row[f'over_{target}_bit_QA'] = sum(bit > target for bit in bits)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('model', 'dataset', 'method', 'status', 'quality_percent', 'mean_effective_bits')}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paired = []
    for row in rows:
        if row['method'] == 'F':
            row.update(delta_to_FP16_pp=0., delta_ci95_low_pp=0., delta_ci95_high_pp=0.)
            continue
        left = summaries.get((row['model'], 'F', row['dataset']))
        right = summaries[(row['model'], row['method'], row['dataset'])]
        if left is None: continue
        result = compare(left[1], right[1])
        paired.append(result)
        safe_name = f"{row['model']}_{row['method']}_{row['dataset']}_versus_F.json"
        (args.output_dir / safe_name).write_text(json.dumps(result, indent=2), encoding='utf-8')
        primary = result['primary']
        row.update(delta_to_FP16_pp=primary['difference_pp'] if primary else None,
                   delta_ci95_low_pp=primary['ci95_pp'][0] if primary and primary['ci95_pp'] else None,
                   delta_ci95_high_pp=primary['ci95_pp'][1] if primary and primary['ci95_pp'] else None)
    result = {'status': 'PASS_SCORED_FIXED_MATRIX' if not pending else 'PARTIAL_AVAILABLE_SCORED_MATRIX',
              'phase': matrix['phase'], 'scope': 'fixed official-split subsets; not full official scores',
              'expected_cells': len(matrix['rows']), 'scored_cells': len(rows), 'pending': pending,
              'expected_QA': sum(plan['count'] for plan in matrix['rows']),
              'scored_QA': sum(row['expected_QA'] for row in rows), 'rows': rows,
              'matrix_sha256': sha(args.matrix), 'metric_contract': 'matrix_v3_metrics_v1',
              'timing_is_controlled': False, 'GPU_inference_performed_by_this_scorer': False,
              'score_sources_are_immutable': True, 'created_unix': time.time()}
    (args.output_dir / 'matrix.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (args.output_dir / 'matrix.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: result[key] for key in ('status', 'expected_cells', 'scored_cells', 'expected_QA', 'scored_QA')}), flush=True)


if __name__ == '__main__': main()
