"""CPU-only independent recount of final R8 diagnostic/timing evidence.

Checks three completion layers, original hashes, physical work, fixed outputs,
complete visual owner fields, and every reported timing median/quartile.
No model, GPU, SSH, labels or source mutation.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows = lambda path: [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(value, message):
    if not value:
        raise ValueError(message)


def close(left, right):
    require(math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-10), 'Numerical recount mismatch')


def layers(folder, kind):
    final = read(folder / 'complete_final_measurement.json')
    marker = read(folder / final['original_final_marker'])
    sidecar_name = 'backend_timing_sidecar.json' if kind == 'timing' else 'backend_diagnostic_sidecar.json'
    sidecar = read(folder / sidecar_name)
    engine = read(folder / 'complete.json')
    require(final['status'] == marker['status'] == sidecar['status'] == engine['status'] == 'COMPLETE', 'Completion layer missing')
    require(final['kind'] == kind and final['original_final_marker_sha256'] == sha(folder / final['original_final_marker'])
        and marker['sidecar_sha256'] == sha(folder / sidecar_name)
        and marker['original_complete_sha256'] == sha(folder / 'complete.json')
        and marker['original_manifest_sha256'] == sha(folder / 'manifest.json'), 'Completion layers not hash-bound')
    require(sidecar['before'] == sidecar['after'] and marker['source_inputs_unchanged'] is True
        and marker['original_artifacts_unchanged'] is True, 'Source/input mutation')
    for name, digest in sidecar['original_artifact_sha256'].items():
        require(sha(folder / name) == digest, 'Original measurement artifact changed: ' + name)
    for name, digest in engine.get('artifact_sha256', {}).items():
        require(sha(folder / name) == digest, 'Original diagnostic artifact changed: ' + name)
    require(final['adapter_sha256'] == sha(Path(__file__).with_name('round8_final_measurement.py'))
        and final['preserved_support_sha256'] == sha(Path(__file__).with_name('round8_backend_support.py')),
        'Final adapter/support source changed')
    return dict(engine=engine, final=final,
        marker_sha256={name: sha(folder / name) for name in
            ('complete.json', final['original_final_marker'], 'complete_final_measurement.json', sidecar_name)})


def execute(output):
    continuous = ROOT / 'results/round8_v1/final_continuous8_v1'
    quiet = ROOT / 'results/round8_v1/final_quiet6_v1'
    dc, tc = layers(continuous, 'continuous'), layers(quiet, 'timing')
    summary = read(continuous / 'summary.json')
    passes = rows(continuous / 'passes.jsonl')
    records = rows(continuous / 'records.jsonl')
    require(len(passes) == dc['engine']['physical_outer_sessions'] == 88
        and sum(row['physical_model_forwards'] for row in passes) == summary['physical_model_forwards'] == 654
        and all(row['status'] == 'COMPLETE' for row in passes), 'Continuous physical work differs')
    require(len(records) == dc['engine']['records'] == 64 and len({row['source_id'] for row in records}) == 8
        and len({row['case_id'] for row in records}) == 8, 'Continuous source/QA/arm coverage differs')
    groups = defaultdict(list)
    for row in records:
        require(row['task_accuracy_eligible'] is False and row['end_to_end_timing'] is False, 'Diagnostic scope changed')
        groups[(row['target_kind'], row['variant'])].append(row)
    for aggregate in summary['source_macro']:
        selected = groups[(aggregate['target_kind'], aggregate['variant'])]
        require(len(selected) == aggregate['sources'] == 8, 'Continuous source weighting differs')
        for key, expected in aggregate['source_macro'].items():
            close(statistics.mean(row['summary'][key] for row in selected), expected)

    measured = rows(quiet / 'timing_records.jsonl')
    boundaries = rows(quiet / 'quiet_boundaries.jsonl')
    timing = read(quiet / 'summary.json')
    counts = Counter(row['phase'] for row in measured)
    require(len(measured) == tc['engine']['trials'] == 72 and counts == {'warmup': 18, 'measurement': 54}, 'Quiet trial count differs')
    require(all(row['quiet'] is True and row['verified'] is True for row in boundaries), 'A quiet boundary is invalid')
    outputs = defaultdict(set)
    for row in measured:
        require(row['quiet_verified'] is True and row['output_ids_and_logical_positions_verified'] is True
            and row['task_accuracy_eligible'] is False, 'Quiet output/role contract differs')
        result = row['runner_result']; owned = result['owned_initial']; price = row['actual_visual_owner']
        require(result['read_ledger_exact'] is True, 'Actual read ledger missing')
        complete = sum(owned.get(key, 0) for key in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position', 'hadamard_constant'))
        complete += result.get('encoder_calibration_parameters_bytes', 0)
        require(complete == price['complete_visual_bytes'], 'Complete visual owner component mismatch')
        close(16 * complete / owned['visual_fp16_denominator'], price['complete_visual_bits'])
        outputs[row['session_id']].add(json.dumps([(turn['token_ids'], turn['logical_end']) for turn in result['turns']]))
        require(all(len(turn['token_ids']) == 8 for turn in result['turns']), 'Forced token horizon changed')
    require(len(outputs) == 3 and all(len(values) == 1 for values in outputs.values()), 'Actual output IDs or logical positions differ')
    for table in timing['table']:
        subset = [row for row in measured if row['phase'] == 'measurement' and row['variant'] == table['variant']]
        require(len(subset) == 9 and len({row['session_id'] for row in subset}) == 3, 'Quiet source/repetition coverage differs')
        metrics = dict(session_ms=[row['runner_result']['session_ms'] for row in subset],
            encoding_ms=[row['runner_result']['encoding_wall_ms'] for row in subset],
            wall_ms=[row['wall_ms_including_runner_setup'] for row in subset],
            first_ttft_ms=[row['runner_result']['turns'][0]['ttft_ms'] for row in subset],
            inter_token_ms=[row['runner_result']['turns'][0]['mean_inter_token_ms'] for row in subset],
            actual_visual_bits=[row['actual_visual_owner']['complete_visual_bits'] for row in subset])
        for key, values in metrics.items():
            quartiles = statistics.quantiles(values, n=4, method='inclusive')
            for suffix, actual in (('_median', statistics.median(values)), ('_p25', quartiles[0]), ('_p75', quartiles[2])):
                close(actual, table[key + suffix])
    result = dict(status='PASS', schema='trajkv.round8.final-measurement-independent-recount.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        continuous=dict(sources=8, independent_QAs=8, records=64, outer_sessions=88, physical_model_forwards=654,
            source_macro_recomputed=True, completion_sha256=dc['marker_sha256']),
        timing=dict(sources=3, methods=6, warmups=18, measurements=54, scheduled_trials=72,
            additional_Full_canonical=3, total_outer_sessions=75, fixed_tokens=8,
            all_output_ids_and_positions_equal=True, all_quiet_boundaries_verified=True,
            all_medians_and_inclusive_quartiles_recomputed=True, actual_visual_owner_components_recounted=True,
            completion_sha256=tc['marker_sha256']),
        CPU_only=True, GPU_used=False, model_executed=False, references_opened=False,
        task_quality_evidence=False, Git_operations=0, network_operations=0)
    with Path(output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status='PASS', output=str(output), continuous_forwards=654, measurements=54)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output)
