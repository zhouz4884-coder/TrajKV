"""Bounded quiet same-output cost timing; frozen timers/readers stay unchanged.

Three sources, five/six methods, warmup once and measure three times. The
Full-derived repeated/truncated eight-token strings control work, not accuracy.
No function, allocator or CUDA-event instrumentation is installed in trials.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
SUPPORTED = {'full', 'native4', 'base2', 'hybrid_packed', 'round6_depth', 'round7_key'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate(config):
    variants = config['variants']
    if len(variants) not in (5, 6) or len({v['name'] for v in variants}) != len(variants):
        raise ValueError('Declare five or six distinct timing methods')
    if any(v['method'] not in SUPPORTED for v in variants):
        raise ValueError('Unsupported timing bank')
    if config.get('images', 3) != 3 or config.get('warmups', 1) != 1 or config.get('repeats', 3) != 3 or config.get('tokens_per_turn', 8) != 8:
        raise ValueError('Bounded protocol is exactly 3 sources, warm1, measure3, eight forced IDs/turn')
    caps = {int(v.get('cap', 8)) for v in variants}
    if len(caps) != 1 or int(config['arguments'].get('turns', 1)) != 1:
        raise ValueError('This bounded timer uses one complete first question at one common image cap')
    if config['arguments'].get('native_question_history', False):
        raise ValueError('Question states must be replayed from the image boundary')
    return dict(sources=3, variants=len(variants), warmup_trials=3 * len(variants),
        measurement_trials=9 * len(variants), scheduled_trials=12 * len(variants),
        additional_Full_canonical_runs=3, forced_tokens_per_turn=8, turns=1,
        model_loads=1, common_cap=next(iter(caps)), task_accuracy_eligible=False)


def make_schedule(rows, variants):
    schedule = []
    for phase, repeats in (('warmup', 1), ('measurement', 3)):
        for repeat in range(repeats):
            block = [(row, variant) for row in rows for variant in variants]
            random.Random(20261116 + repeat + (10000 if phase == 'measurement' else 0)).shuffle(block)
            schedule.extend((phase, repeat, row, variant) for row, variant in block)
    return schedule


def visual_owner(result):
    owner = result['owned_initial']
    encoder = int(result.get('encoder_calibration_parameters_bytes', 0))
    hadamard = int(owner.get('hadamard_constant', 0))
    complete = sum(int(owner.get(key, 0)) for key in
        ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position')) + encoder + hadamard
    if 'encoder_constant_bytes' in owner and int(owner['encoder_constant_bytes']) != encoder:
        raise RuntimeError('Encoder owner aliases disagree')
    if 'complete_visual_owner_bytes' in owner and int(owner['complete_visual_owner_bytes']) != complete:
        raise RuntimeError('Complete visual owner disagrees with component sum')
    return dict(complete_visual_bytes=complete, complete_visual_bits=16 * complete / owner['visual_fp16_denominator'],
        encoder_constant_bytes=encoder, hadamard_constant_bytes=hadamard,
        denominator=int(owner['visual_fp16_denominator']),
        scope='visual base/residual/lookup/positions plus encoder constants and actual Hadamard; total_unique is not added')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    plan = validate(config)
    if args.plan_only:
        print(json.dumps(plan), flush=True)
        return
    import torch
    import round7_runner  # Register both preserved oldDepth and current Round7.
    from factory import bank_class
    from bank import install_legacy
    from timing_control import quiet_snapshot
    import session_boundary
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if not torch.cuda.is_available():
        raise RuntimeError('Actual quiet timing requires CUDA')
    args.output.mkdir(parents=True, exist_ok=False)
    a = SimpleNamespace(**config['arguments'])
    for key, value in dict(turns=1, max_tokens=8, description_tokens=8, max_patches=plan['common_cap'],
        margin=0., max_replays=0, save_states=False, read_budget_bytes=None, gain_table_data=None).items():
        if not hasattr(a, key):
            setattr(a, key, value)
    a.max_tokens, a.description_tokens = 8, 8
    install_legacy(a.legacy_scripts)
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads(Path(a.samples).read_text(encoding='utf-8-sig'))[:3]
    if len(rows) != 3 or len({row['source_image_id'] for row in rows}) != 3:
        raise ValueError('Need three distinct old-development source images')
    if any(any(key in row for key in ('answer', 'answers', 'references', 'ground_truth')) for row in rows):
        raise ValueError('No GT answer fields may enter timing')
    external = {v[key]: sha(v[key]) for v in config['variants'] for key in
        ('direction_book_path', 'metadata_factor_path', 'key_metric_path') if v.get(key)}
    contract = config.get('frozen_variant_contract')
    if contract:
        if sha(contract['path']) != contract['sha256']:
            raise ValueError('Frozen variant contract bytes changed')
        frozen = {v['name']: v for v in json.loads(Path(contract['path']).read_text(encoding='utf-8-sig'))['variants']}
        if any(frozen.get(v['name']) != v for v in config['variants']):
            raise ValueError('Timing variant differs from the frozen quality configuration')
        external[contract['path']] = contract['sha256']
    source_names = ('round7_timing.py', 'round7_runner.py', 'round6_runner.py', 'session_boundary.py',
        'round6_depth_bank.py', 'round6_depth_attention.py', 'round6_depth_codec.py', 'round7_key_bank.py',
        'round7_key_attention.py', 'round7_key_codec.py', 'round7_key_layout.py', 'round7_key_artifacts.py',
        'hybrid_packed_bank.py', 'fused_hybrid.py', 'factory.py', 'masked_bank.py', 'fast_wht.py', 'quantizers.py')
    sources = {'code/a2/' + name: sha(Path(__file__).with_name(name)) for name in source_names}
    for name in ('code/experiments/bank.py', 'code/experiments/timing_control.py', 'code/bytes.py'):
        sources[name] = sha(ROOT / name)
    image_inputs = []
    for row in rows:
        hashes = [sha(path) for path in row.get('remote_images', row['images'])]
        if row.get('hashes') and row['hashes'] != hashes:
            raise ValueError('Actual image bytes differ from source manifest')
        image_inputs.append(dict(id=row['id'], hashes=hashes))
    model_hash = sha(Path(a.model) / 'config.json')
    manifest = dict(schema='trajkv_round7_quiet_same_output_v1', config=config, plan=plan,
        config_sha256=sha(args.config), samples_sha256=sha(a.samples), model_config_sha256=model_hash,
        selected_ids=[row['id'] for row in rows], image_inputs=image_inputs,
        exact_frozen_variants_verified=bool(contract),
        sources=sources, external_codec_inputs=external, quality='not_scored', task_accuracy_eligible=False,
        scope='model load excluded; input preparation/native causal-prefix extraction/encoding/full question replay/fixed-eight decode included',
        length_control='Full-generated non-EOS IDs repeated/truncated to seven plus canonical EOS; not free application outputs or accuracy',
        quiet_scope='all NVIDIA compute processes checked before and after each trial; no continuous thermal-isolation claim',
        instrumentation='no function, constructor, allocator or CUDA-event instrumentation; actual frozen run_session path',
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    write_json(args.output / 'manifest.json', manifest)
    original_bank = session_boundary.ProgressiveBank
    quiet_rows, records = [], []

    def require_quiet(stage):
        snapshot = quiet_snapshot()
        quiet_rows.append(dict(stage=stage, observed_wall_time=time.time(), **snapshot))
        if not snapshot.get('verified') or not snapshot.get('quiet'):
            raise RuntimeError('Another NVIDIA compute process is active; preserve run and defer timing')
        return snapshot

    def run(row, variant, forced=None):
        cls = bank_class(variant['method'])
        previous_settings = cls.settings
        cls.settings = variant
        session_boundary.ProgressiveBank = cls
        a.max_patches = variant.get('cap', plan['common_cap'])
        try:
            return session_boundary.run_session(model, processor, row, a, variant['method'], 'base', forced=forced)[0]
        finally:
            cls.settings = previous_settings
            session_boundary.ProgressiveBank = original_bank

    try:
        require_quiet('before_model_load')
        processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(a.model, local_files_only=True,
            dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        canonical = {}
        with torch.inference_mode():
            for row in rows:
                require_quiet('canonical_before:' + row['id'])
                full = run(row, dict(method='full', cap=plan['common_cap'], selector='none', store_selector_meta=False))
                ids = [token for token in full['turns'][0]['token_ids'] if token != processor.tokenizer.eos_token_id]
                if not ids:
                    ids = processor.tokenizer.encode('0', add_special_tokens=False)
                canonical[row['id']] = [dict(token_ids=(ids * 8)[:7] + [processor.tokenizer.eos_token_id])]
                require_quiet('canonical_after:' + row['id'])
            write_json(args.output / 'length_controls.json', canonical)
            schedule = make_schedule(rows, config['variants'])
            write_json(args.output / 'schedule.json', [dict(phase=phase, repeat=repeat, session_id=row['id'], variant=v['name'])
                for phase, repeat, row, v in schedule])
            positions = {}
            with (args.output / 'timing_records.jsonl').open('x', encoding='utf-8') as stream:
                for index, (phase, repeat, row, variant) in enumerate(schedule):
                    before = require_quiet('trial_before:' + str(index))
                    started = time.perf_counter()
                    result = run(row, variant, canonical[row['id']])
                    wall = 1000 * (time.perf_counter() - started)
                    after = require_quiet('trial_after:' + str(index))
                    if [turn['token_ids'] for turn in result['turns']] != [turn['token_ids'] for turn in canonical[row['id']]]:
                        raise RuntimeError('Forced output token IDs changed')
                    current = [(turn['logical_end'], turn['forced_chat_closure']) for turn in result['turns']]
                    if row['id'] in positions and current != positions[row['id']]:
                        raise RuntimeError('Same-input logical positions or closure changed')
                    positions[row['id']] = current
                    if result.get('protocol') != 'image_boundary_fixed_tokens' or not result.get('read_ledger_exact'):
                        raise RuntimeError('Actual fixed-output runner protocol/read ledger failed')
                    record = dict(phase=phase, repeat=repeat, session_id=row['id'], variant=variant['name'],
                        variant_config=variant, quality='not_scored', task_accuracy_eligible=False,
                        quiet_verified=True, quiet_before=before, quiet_after=after,
                        output_ids_and_logical_positions_verified=True, wall_ms_including_runner_setup=wall,
                        actual_visual_owner=visual_owner(result), runner_result=result)
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
                    print(json.dumps(dict(trial=index + 1, phase=phase, variant=variant['name'],
                        session_ms=result['session_ms'], output_ids_identical=True, quality='not_scored')), flush=True)
        for path, expected in sources.items():
            if sha(ROOT / path) != expected:
                raise RuntimeError('Timing source changed: ' + path)
        for path, expected in external.items():
            if sha(path) != expected:
                raise RuntimeError('Timing codec input changed: ' + path)
        summary = []
        for variant in config['variants']:
            subset = [r for r in records if r['phase'] == 'measurement' and r['variant'] == variant['name']]
            if len(subset) != 9:
                raise RuntimeError('Missing measured source/repeat coverage')
            entry = dict(variant=variant['name'], method=variant['method'], sources=3, measurements=9)
            metrics = dict(session_ms=[r['runner_result']['session_ms'] for r in subset],
                encoding_ms=[r['runner_result']['encoding_wall_ms'] for r in subset],
                wall_ms=[r['wall_ms_including_runner_setup'] for r in subset],
                first_ttft_ms=[r['runner_result']['turns'][0]['ttft_ms'] for r in subset],
                inter_token_ms=[r['runner_result']['turns'][0]['mean_inter_token_ms'] for r in subset],
                actual_visual_bits=[r['actual_visual_owner']['complete_visual_bits'] for r in subset])
            for key, values in metrics.items():
                quartiles = statistics.quantiles(values, n=4, method='inclusive')
                entry.update({key + '_median': statistics.median(values), key + '_p25': quartiles[0], key + '_p75': quartiles[2]})
            summary.append(entry)
        with (args.output / 'table.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        write_json(args.output / 'summary.json', dict(status='PASS', table=summary, plan=plan,
            output_ids_identical=True, logical_positions_identical=True, quiet_verified_all_trials=True,
            task_accuracy_eligible=False, quality='not_scored', scope=manifest['scope']))
        with (args.output / 'quiet_boundaries.jsonl').open('x', encoding='utf-8') as stream:
            for row in quiet_rows:
                stream.write(json.dumps(row) + '\n')
        write_json(args.output / 'complete.json', dict(status='COMPLETE', trials=len(records),
            measurement_trials=plan['measurement_trials'], warmups=plan['warmup_trials'],
            quality='not_scored', task_accuracy_eligible=False, model_loads=1))
    except BaseException as error:
        write_json(args.output / 'failure.json', dict(status='FAILED_NEW_RUN_REQUIRED', exception_type=type(error).__name__,
            message=str(error), completed_trials=len(records), quality='not_scored', task_accuracy_eligible=False,
            quiet_observations=quiet_rows))
        raise
    finally:
        session_boundary.ProgressiveBank = original_bank


if __name__ == '__main__':
    main()
