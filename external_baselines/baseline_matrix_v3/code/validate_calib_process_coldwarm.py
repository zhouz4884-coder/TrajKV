"""CPU diagnostic trace after unchanged independent speed-evidence audit.

First method invocation is not all-disk-cold, and these two-QA processes do not
produce a speed ranking. No model, GPU, or reference answers are used here.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from validate_speed_evidence import audit, load, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('raw', type=Path)
    parser.add_argument('--ledger-row', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); row = load(args.ledger_row)
    assert row['formal_ranking'] is False and row['quality_500_QA'] == 0
    assert row['expected_trials'] == 16 and row['measured_trials'] == 12 and row['warmup_trials'] == 4
    if not (args.output / 'audit.json').exists(): result = audit(args.raw, args.output, row)
    else:
        result = load(args.output / 'audit.json')
        assert all(sha(args.raw / name) == digest for name, digest in result['raw_hashes'].items())
    plan = load(args.raw / 'plan.json'); launch = load(args.raw / 'launch_receipt.json')
    records = [json.loads(line) for line in (args.raw / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(plan['method_ids']) == len(plan['expected_ids']) == 2 and len(records) == 16
    first, trace = {}, []
    for index, record in enumerate(records):
        method = record['method']; first.setdefault(method, index)
        first_use = first[method] == index
        if first_use: assert record['phase'] == 'warmup'
        trace.append({'trial_index': index, 'method': method, 'id': record['id'], 'phase': record['phase'],
            'repeat': record['repeat'], 'method_first_invocation_in_this_process': first_use,
            'status': record['status'], 'output_tokens': record.get('output_tokens'),
            'wall_ms': record.get('wall_ms'), 'TTFT_host_ms': record.get('ttft_ms'),
            'prefill_GPU_stream_ms': record.get('prefill_ms'), 'encode_GPU_stream_ms': record.get('encode_ms'),
            'question_prefill_GPU_stream_ms': record.get('question_prefill_ms'),
            'ITL_host_mean_ms': record.get('mean_inter_token_ms'),
            'start_allocated_bytes': record.get('start_allocated_bytes'),
            'start_reserved_bytes': record.get('start_reserved_bytes'),
            'peak_allocated_bytes': record.get('peak_allocated_bytes')})
    counts = dict(Counter(record['phase'] for record in records)); assert counts == {'warmup': 4, 'measure': 12}
    summary = {'schema': 'baseline_matrix.calib_process_coldwarm_trace.v1',
        'status': 'PASS_PROCESS_FIRST_USE_AND_WARM_TRACE' if result['status'].startswith('PASS_') else 'CLOSED_DIAGNOSTIC_WITH_UNVERIFIED_OR_FAILED_TRIALS',
        'base_audit': result['status'], 'base_audit_sha256': sha(args.output / 'audit.json'),
        'raw_trials': 16, 'warmup_trials': 4, 'measured_trials': 12, 'QA': 2,
        'runtime_PID': launch['pid'], 'job_name': launch['name'],
        'launch_receipt_sha256': sha(args.raw / 'launch_receipt.json'),
        'first_trial_index_by_method': first, 'all_trial_trace': trace,
        'disk_binary_cache': 'Retained shared project cache; no deletion or all-disk-cold claim',
        'process_scope': 'Separate operator launch; first recorded invocation of each method precedes its warm repeats',
        'no_autotune_winner_identity_observed_by_this_trace': True,
        'host_quiet_asserted': result['host_quiet_asserted'], 'formal_ranking': False,
        'quality_500_QA': 0, 'formal_656_trials_added': 0, 'CPU_validator_used_GPU': False,
        'source_benchmark_sha256': result['source_benchmark_sha256'], 'validator_sha256': sha(__file__)}
    file = args.output / 'coldwarm_trace.json'
    if file.exists(): assert load(file) == summary
    else:
        with file.open('x', encoding='utf-8') as stream: json.dump(summary, stream, indent=2)
        with (args.output / 'all_trials.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(trace)
    print(json.dumps({key: summary[key] for key in ('status', 'raw_trials', 'runtime_PID', 'formal_ranking', 'quality_500_QA')}))


if __name__ == '__main__': main()
