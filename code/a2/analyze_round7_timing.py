"""Strict analysis of only Round7 quiet same-output cost runs, never quality."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'trajkv_round7_quiet_same_output_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_cost_run(manifest, complete):
    # Reject native/free-quality, tail, profiling and old timer schemas first.
    require(manifest.get('schema') == SCHEMA, 'Not a Round7 quiet same-output cost manifest')
    require(manifest.get('quality') == 'not_scored' and manifest.get('task_accuracy_eligible') is False,
            'Manifest does not explicitly exclude task-quality scoring')
    require(complete.get('status') == 'COMPLETE' and complete.get('quality') == 'not_scored'
            and complete.get('task_accuracy_eligible') is False, 'Cost run incomplete or quality-labelled')
    config = manifest['config']
    variants = config['variants']
    require(len(variants) in (5, 6) and len({v['name'] for v in variants}) == len(variants), 'Wrong bounded variant coverage')
    require(config.get('images', 3) == 3 and config.get('repeats', 3) == 3
            and config.get('warmups', 1) == 1 and config.get('tokens_per_turn', 8) == 8
            and config['arguments'].get('turns', 1) == 1, 'Wrong bounded cost protocol')
    require(len(manifest['selected_ids']) == 3 and len(set(manifest['selected_ids'])) == 3, 'Wrong source coverage')
    require(complete.get('trials') == 12 * len(variants) and complete.get('measurement_trials') == 9 * len(variants)
            and complete.get('warmups') == 3 * len(variants) and complete.get('model_loads') == 1, 'Completion counts disagree')
    return variants


def local_path(value):
    value = str(value).replace('\\', '/')
    return ROOT / value.split('/TrajKV/', 1)[1] if '/TrajKV/' in value else Path(value)


def audit_run(folder):
    folder = Path(folder)
    manifest, complete = read(folder / 'manifest.json'), read(folder / 'complete.json')
    variants = require_cost_run(manifest, complete)
    selected, by_name = manifest['selected_ids'], {v['name']: v for v in variants}
    targets, schedule = read(folder / 'length_controls.json'), read(folder / 'schedule.json')
    require(set(targets) == set(selected), 'Canonical forced-ID source coverage differs')
    require(all(len(turns) == 1 and len(turns[0]['token_ids']) == 8 for turns in targets.values()), 'Canonical outputs are not exactly one eight-ID turn')
    records = [json.loads(line) for line in (folder / 'timing_records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(records) == len(schedule) == complete['trials'], 'Raw trial/schedule count mismatch')
    expected = {(phase, repeat, sid, name) for phase, repeats in (('warmup', 1), ('measurement', 3))
                for repeat in range(repeats) for sid in selected for name in by_name}
    seen, positions, measurements = set(), {}, []
    for record, planned in zip(records, schedule):
        for key in ('phase', 'repeat', 'session_id', 'variant'):
            require(record[key] == planned[key], 'Raw record differs from declared randomized schedule')
        identity = tuple(record[key] for key in ('phase', 'repeat', 'session_id', 'variant'))
        require(identity in expected and identity not in seen, 'Unexpected or duplicate trial')
        seen.add(identity)
        sid, name = record['session_id'], record['variant']
        require(record.get('quality') == 'not_scored' and record.get('task_accuracy_eligible') is False
                and record.get('quiet_verified') is True, 'Trial not quiet cost-only work')
        require(record['variant_config'] == by_name[name], 'Trial differs from declared variant')
        for key in ('quiet_before', 'quiet_after'):
            require(record[key].get('verified') is True and record[key].get('quiet') is True
                    and not record[key].get('foreign_processes'), 'Foreign GPU compute process at trial boundary')
        result = record['runner_result']
        require(result.get('protocol') == 'image_boundary_fixed_tokens' and result.get('read_ledger_exact') is True,
                'A free-generation/tail/diagnostic result entered the fixed-output timer')
        require([turn['token_ids'] for turn in result['turns']] == [turn['token_ids'] for turn in targets[sid]], 'Forced IDs differ')
        current = [(turn['logical_end'], turn['forced_chat_closure']) for turn in result['turns']]
        require(sid not in positions or current == positions[sid], 'Logical positions/closure differ at the common cap')
        positions[sid] = current
        owner = result['owned_initial']
        encoder = int(result.get('encoder_calibration_parameters_bytes', 0))
        hadamard = int(owner.get('hadamard_constant', 0))
        total = sum(int(owner.get(key, 0)) for key in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position')) + encoder + hadamard
        require(record['actual_visual_owner']['complete_visual_bytes'] == total
                and record['actual_visual_owner']['hadamard_constant_bytes'] == hadamard
                and record['actual_visual_owner']['encoder_constant_bytes'] == encoder, 'Complete owner constants missed or duplicated')
        if record['phase'] != 'measurement':
            continue
        turn = result['turns'][0]
        measurements.append(dict(session_id=sid, variant=name, method=by_name[name]['method'], repeat=record['repeat'],
            visual_tokens=result['visual_tokens'], output_tokens=len(turn['token_ids']), session_ms=result['session_ms'],
            wall_ms=record['wall_ms_including_runner_setup'], encoding_ms=result['encoding_wall_ms'],
            first_ttft_ms=turn['ttft_ms'], inter_token_ms=turn['mean_inter_token_ms'],
            complete_visual_bytes=total, complete_visual_bits=16 * total / owner['visual_fp16_denominator'],
            hadamard_constant_bytes=hadamard, encoder_constant_bytes=encoder,
            peak_allocated=result['peak_allocated']))
    require(seen == expected and len(measurements) == complete['measurement_trials'], 'Missing source/repeat/method coverage')
    for path, digest in manifest['external_codec_inputs'].items():
        require(local_path(path).is_file() and sha(local_path(path)) == digest, 'Codec or frozen variant contract missing/changed')
    table = []
    for variant in variants:
        group = [row for row in measurements if row['variant'] == variant['name']]
        entry = dict(variant=variant['name'], method=variant['method'], sources=3, measured_trials=9)
        for field in ('session_ms', 'wall_ms', 'encoding_ms', 'first_ttft_ms', 'inter_token_ms', 'complete_visual_bits', 'peak_allocated'):
            values = [row[field] for row in group]
            quartiles = statistics.quantiles(values, n=4, method='inclusive')
            entry.update({field + '_median': statistics.median(values), field + '_p25': quartiles[0], field + '_p75': quartiles[2]})
        table.append(entry)
    return dict(status='PASS', schema=SCHEMA, table=table, measurements=measurements,
        measured_trials=len(measurements), warmup_trials=complete['warmups'], output_ids_identical=True,
        logical_positions_and_closure_identical=True, quiet_boundaries_verified=True,
        quality='not_scored', task_accuracy_eligible=False, free_generation_accuracy=None,
        scope='three old-development source first questions, same forced eight IDs; only full-session/encoding/read cost, not application quality',
        producer_manifest_sha256=sha(folder / 'manifest.json'), raw_records_sha256=sha(folder / 'timing_records.jsonl'),
        analyzer_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    measurements = report.pop('measurements')
    for name, rows in (('measurements.csv', measurements), ('table.csv', report['table'])):
        with (args.output / name).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (args.output / 'summary.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(status=report['status'], measured_trials=report['measured_trials'], warmup_trials=report['warmup_trials'], quality='not_scored')), flush=True)


if __name__ == '__main__':
    main()
