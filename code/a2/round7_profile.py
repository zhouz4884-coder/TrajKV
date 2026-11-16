"""Instrumented CUDA-interval/CPU-wall diagnostics for frozen R7 encoders.

One model load, first 2/3 source images, at most four specified methods, and up
to eight naturally generated first-question tokens. This is not a quiet E2E
speed benchmark, a same-output timer, or a task-quality evaluation.
"""
import argparse
import csv
import functools
import hashlib
import importlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
import torch


METHODS = {'round6_depth': ('round6_depth_bank', 'DepthResidualBank', 'round6_depth_attention'),
           'round7_key': ('round7_key_bank', 'KeyBank', 'round7_key_attention'),
           'round7_tail_key': ('round7_tail_key_bank', 'TailKeyBank', 'round7_tail_key_attention'),
           'full': ('masked_bank', 'A2Bank', None)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class FunctionProfiler:
    """Patch actual call-site globals; defer every event read until final sync.

    A callable identity guard suppresses nested aliases of the same function.
    Patches restore exact original objects even after an exception. Optional
    factories let CPU controls verify the wrapper logic without simulating CUDA.
    """
    def __init__(self, event_factory=None, synchronizer=None):
        self.event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self.synchronizer = synchronizer or torch.cuda.synchronize
        self.patches = []
        self.targets = set()
        self.active_functions = set()
        self.calls = []
        self.started_calls = 0
        self.suppressed_nested_calls = 0
        self.closed = False

    def patch(self, module, attribute, component):
        target = (id(module), attribute)
        if target in self.targets or not hasattr(module, attribute):
            return
        original = getattr(module, attribute)
        if not callable(original):
            raise ValueError('Profile target must be callable: ' + attribute)
        if hasattr(original, '_round7_profile_owner'):
            raise ValueError('Refuse to wrap an existing profiling wrapper')
        function_id = id(original)
        module_name = getattr(module, '__name__', type(module).__name__)
        target_name = module_name + '.' + attribute

        @functools.wraps(original)
        def measured(*args, **kwargs):
            if function_id in self.active_functions:
                self.suppressed_nested_calls += 1
                return original(*args, **kwargs)
            self.active_functions.add(function_id)
            wrapper_started = time.perf_counter()
            begin, end = None, None
            failed = False
            cpu_started = None
            row = dict(component=component, target=target_name, call_index=self.started_calls, failed=False)
            self.started_calls += 1
            try:
                begin, end = self.event_factory(), self.event_factory()
                begin.record()
                cpu_started = time.perf_counter()
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                cpu_ended = time.perf_counter()
                row['failed'] = failed
                row['cpu_wall_ms'] = 1000 * (cpu_ended - cpu_started) if cpu_started is not None else None
                if end is not None:
                    try:
                        end.record()
                    except Exception as error:
                        row['event_record_error'] = type(error).__name__ + ': ' + str(error)
                row['wrapper_wall_ms'] = 1000 * (time.perf_counter() - wrapper_started)
                self.calls.append((row, begin, end))
                self.active_functions.remove(function_id)

        measured._round7_profile_owner = self
        measured._round7_profile_original = original
        setattr(module, attribute, measured)
        self.patches.append((module, attribute, original, measured, component))
        self.targets.add(target)

    def patch_encoder(self, bank_module, attention_module=None):
        for attribute, component in (
            ('wht', 'encoder_FWHT'), ('encode_key', 'PQ_and_gain'),
            ('binary_centroids', 'V_Lloyd'), ('binary_centroids_tail', 'V_Lloyd'),
            ('encode_sign_depth', 'residual_sign_depth'),
            ('plan_depth_stages', 'plan'), ('plan_key_stages', 'plan')):
            self.patch(bank_module, attribute, component)
        if attention_module is not None:
            self.patch(attention_module, 'wht', 'reader_FWHT')

    def close(self):
        if self.closed:
            return
        for module, attribute, original, wrapper, component in reversed(self.patches):
            setattr(module, attribute, original)
        self.closed = True

    def finalized(self):
        # No per-call synchronization was inserted by this profiler.
        self.synchronizer()
        rows, components = [], defaultdict(list)
        for raw, begin, end in self.calls:
            row = dict(raw)
            row['gpu_interval_ms'] = (float(begin.elapsed_time(end))
                if begin is not None and end is not None and 'event_record_error' not in row else None)
            rows.append(row)
            components[row['component']].append(row)
        summaries = {}
        for component, group in components.items():
            gpu = [row['gpu_interval_ms'] for row in group if row['gpu_interval_ms'] is not None]
            cpu = [row['cpu_wall_ms'] for row in group if row['cpu_wall_ms'] is not None]
            summaries[component] = dict(calls=len(group), gpu_interval_total_ms=sum(gpu),
                gpu_interval_mean_ms=statistics.mean(gpu) if gpu else None,
                cpu_wall_total_ms=sum(cpu), cpu_wall_mean_ms=statistics.mean(cpu) if cpu else None,
                wrapper_wall_total_ms=sum(row['wrapper_wall_ms'] for row in group),
                failures=sum(row['failed'] for row in group))
        return dict(components=summaries, calls=rows,
            suppressed_nested_alias_calls=self.suppressed_nested_calls,
            original_callables_restored=self.closed and all(getattr(module, attribute) is original
                for module, attribute, original, wrapper, component in self.patches),
            patched_targets=[dict(module=getattr(module, '__name__', type(module).__name__),
                attribute=attribute, component=component,
                original_module=getattr(original, '__module__', None),
                original_qualname=getattr(original, '__qualname__', None))
                for module, attribute, original, wrapper, component in self.patches],
            timing_scope='CUDA current-stream intervals include any host launch gaps inside functions; CPU wall excludes event setup/record; wrapper wall includes instrumentation',
            synchronization_scope='one profiler sync after each completed session; frozen callees retain their own existing synchronization')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def selected_inputs(rows):
    result = []
    for row in rows:
        if any(key in row for key in ('answer', 'answers', 'references', 'ground_truth')):
            raise ValueError('Profiling input must not contain GT answer fields')
        paths = row.get('remote_images', row.get('images', []))
        hashes = [sha(path) for path in paths]
        if not hashes or row.get('hashes') and hashes != row['hashes']:
            raise ValueError('Source image bytes differ from the input manifest')
        result.append(dict(id=row['id'], image_id=row['source_image_id'], image_sha256=hashes))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    variants = config['variants']
    if not 1 <= len(variants) <= 4 or len({variant['name'] for variant in variants}) != len(variants):
        raise ValueError('Specify one to four distinct profiling variants')
    if any(variant['method'] not in METHODS for variant in variants):
        raise ValueError('Only frozen full/round6_depth/round7_key/round7_tail_key banks are supported')
    count = int(config.get('count', 2))
    if count not in (2, 3) or int(config.get('start', 0)) != 0:
        raise ValueError('Use the first two or three old-development source rows')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA encoder profiling requires CUDA')
    for key, value in config['arguments'].items():
        setattr(args, key, Path(value) if key in ('model', 'legacy_scripts', 'kivi_source', 'samples') else value)
    for key, value in dict(max_patches=16, margin=0., max_replays=0, save_states=False,
                           read_budget_bytes=None, gain_table_data=None).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    # Deliberately keep this diagnostic bounded even when reusing a six-turn config.
    args.turns = 1
    args.max_tokens = int(config.get('profile_tokens', 8))
    args.description_tokens = args.max_tokens
    if not 1 <= args.max_tokens <= 8 or getattr(args, 'native_question_history', False):
        raise ValueError('Profile one complete image-boundary question with one to eight free tokens')
    data = json.loads(args.samples.read_text(encoding='utf-8-sig'))
    rows = data[:count]
    if len(rows) != count or len({row['source_image_id'] for row in rows}) != count:
        raise ValueError('Need two or three distinct old-development source images')
    args.output.mkdir(parents=True, exist_ok=False)
    from bank import install_legacy
    install_legacy(args.legacy_scripts)
    import session_boundary
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(4)
    torch.manual_seed(1212)
    torch.backends.cuda.matmul.allow_tf32 = False
    external = {str(variant[key]): sha(variant[key]) for variant in variants
        for key in ('direction_book_path', 'metadata_factor_path', 'key_metric_path') if variant.get(key)}
    source_names = ('round7_profile.py', 'round6_depth_bank.py', 'round6_depth_attention.py',
        'round6_depth_codec.py', 'round7_key_bank.py', 'round7_key_attention.py', 'round7_key_codec.py',
        'round7_key_layout.py', 'round7_key_artifacts.py', 'round7_tail_key_bank.py',
        'round7_tail_key_attention.py', 'round7_tail_key_codec.py', 'fast_wht.py', 'quantizers.py',
        'session_boundary.py', 'masked_bank.py', 'metadata_direction_bank.py', 'learned_direction_bank.py')
    sources = {str((Path(__file__).parent / name).relative_to(ROOT)): sha(Path(__file__).with_name(name))
               for name in source_names if Path(__file__).with_name(name).is_file()}
    sources['code/experiments/bank.py'] = sha(ROOT / 'code/experiments/bank.py')
    model_config = Path(args.model) / 'config.json'
    manifest = dict(schema='trajkv_round7_instrumented_profile_manifest_v1', config=config,
        config_sha256=sha(args.config), samples_sha256=sha(args.samples), selected_ids=[row['id'] for row in rows],
        selected_inputs=selected_inputs(rows), external_codec_inputs=external, sources=sources,
        model_config_sha256=sha(model_config), model_loads_planned=1,
        effective_turns=1, effective_max_free_tokens=args.max_tokens, generation='first-question free-running, early EOS allowed',
        variants_order='fixed declared order within each source; no quiet warmup or randomized timing claim',
        task_quality_eligible=False, task_accuracy_eligible=False, primary_quality_eligible=False,
        quiet_E2E_speed_eligible=False, same_output_timing_eligible=False,
        instrumentation='temporary wrappers at actual bank/reader module call sites; CUDA events and CPU wall; restored after every session',
        old_Depth_PQ_scope='legacy inline PQ/gain has no separate callable; use its overall encoding_wall_ms reference and do not infer PQ time by subtraction',
        cuda_interval_scope='current-stream event interval, including host launch gaps; not summed kernel-active time or measured DRAM traffic',
        scope='instrumented component diagnostics only; final same-output timer remains a separate experiment',
        labels_used=False, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
    write_json(args.output / 'manifest.json', manifest)
    previous_bank = session_boundary.ProgressiveBank
    records, components_rows = [], []
    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
            dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode(), (args.output / 'records.jsonl').open('x', encoding='utf-8') as stream, (args.output / 'calls.jsonl').open('x', encoding='utf-8') as call_stream:
            for row in rows:
                for variant in variants:
                    module_name, class_name, reader_name = METHODS[variant['method']]
                    bank_module = importlib.import_module(module_name)
                    bank_class = getattr(bank_module, class_name)
                    reader_module = importlib.import_module(reader_name) if reader_name else None
                    original_settings = bank_class.settings
                    bank_class.settings = dict(variant)
                    session_boundary.ProgressiveBank = bank_class
                    args.max_patches = int(variant.get('cap', config['arguments'].get('max_patches', 16)))
                    profiler = FunctionProfiler()
                    started = time.perf_counter()
                    try:
                        with profiler:
                            profiler.patch_encoder(bank_module, reader_module)
                            result, _ = session_boundary.run_session(model, processor, row, args,
                                variant['method'], 'base', keep_reference=False)
                        profile = profiler.finalized()
                    finally:
                        profiler.close()
                        bank_class.settings = original_settings
                        session_boundary.ProgressiveBank = previous_bank
                    profiled_wall = 1000 * (time.perf_counter() - started)
                    raw_calls = profile.pop('calls')
                    for call in raw_calls:
                        call_stream.write(json.dumps(dict(session_id=row['id'], variant=variant['name'], **call), allow_nan=False) + '\n')
                    call_stream.flush()
                    record = dict(session_id=row['id'], variant=variant['name'], method=variant['method'],
                        variant_config=variant, visual_tokens=result['visual_tokens'], max_patches=args.max_patches,
                        encoding_wall_ms=result['encoding_wall_ms'], session_ms=result['session_ms'],
                        profiled_run_outer_wall_ms=profiled_wall, generated_tokens=sum(len(turn['token_ids']) for turn in result['turns']),
                        generation_token_ids=[turn['token_ids'] for turn in result['turns']],
                        forced_canonical_chat_closure=[turn['forced_chat_closure'] for turn in result['turns']],
                        reader_calls=sum(event['calls'] for event in result['read_events']),
                        owned_initial=result['owned_initial'], peak_allocated=result['peak_allocated'],
                        component_profile=profile, task_quality_eligible=False,
                        task_accuracy_eligible=False, primary_quality_eligible=False, quiet_E2E_speed_eligible=False,
                        same_output_timing_eligible=False, source='actual frozen bank and reader calls under temporary profiling wrappers')
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
                    for component, summary in profile['components'].items():
                        components_rows.append(dict(session_id=row['id'], variant=variant['name'], method=variant['method'],
                            visual_tokens=result['visual_tokens'], component=component, **summary))
                    print(json.dumps(dict(done=len(records), session_id=row['id'], variant=variant['name'],
                        encoding_wall_ms=record['encoding_wall_ms'], session_ms=record['session_ms'],
                        components=profile['components'], profiling_only=True)), flush=True)
        for name, expected in sources.items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('Frozen source bytes changed during profiling: ' + name)
        for path, expected in external.items():
            if sha(path) != expected:
                raise RuntimeError('Frozen codec input changed during profiling: ' + path)
        if sha(args.config) != manifest['config_sha256'] or sha(args.samples) != manifest['samples_sha256'] or sha(model_config) != manifest['model_config_sha256']:
            raise RuntimeError('Config/samples/model identity changed during profiling')
        aggregate = []
        for variant in variants:
            group = [record for record in records if record['variant'] == variant['name']]
            summaries = {}
            labels = sorted({component for record in group for component in record['component_profile']['components']})
            for component in labels:
                parts = [record['component_profile']['components'][component] for record in group
                         if component in record['component_profile']['components']]
                summaries[component] = dict(calls=sum(part['calls'] for part in parts),
                    gpu_interval_total_ms=sum(part['gpu_interval_total_ms'] for part in parts),
                    cpu_wall_total_ms=sum(part['cpu_wall_total_ms'] for part in parts),
                    instrumented_sessions=len(parts))
            aggregate.append(dict(variant=variant['name'], method=variant['method'], sessions=len(group),
                encoding_wall_ms=[record['encoding_wall_ms'] for record in group],
                encoding_wall_median_ms=statistics.median(record['encoding_wall_ms'] for record in group),
                session_ms=[record['session_ms'] for record in group],
                session_median_ms=statistics.median(record['session_ms'] for record in group), components=summaries,
                legacy_inline_PQ_not_separately_measured=variant['method'] == 'round6_depth'))
        write_csv(args.output / 'components.csv', components_rows)
        write_json(args.output / 'summary.json', dict(status='COMPLETE', variants=aggregate,
            task_quality_eligible=False, task_accuracy_eligible=False, primary_quality_eligible=False,
            quiet_E2E_speed_eligible=False, same_output_timing_eligible=False,
            scope=manifest['scope'], sources_preserved=True, input_hashes_preserved=True,
            originals_restored=all(record['component_profile']['original_callables_restored'] for record in records)))
        write_json(args.output / 'complete.json', dict(status='COMPLETE', records=len(records),
            expected=count * len(variants), model_loads=1, profiling_only=True,
            original_function_bytes_preserved=True, runtime_monkeypatches_restored=True))
    except BaseException as error:
        if not (args.output / 'failure.json').exists():
            write_json(args.output / 'failure.json', dict(status='FAILED_NEW_RUN_REQUIRED',
                exception_type=type(error).__name__, message=str(error), completed_records=len(records),
                task_quality_eligible=False, quiet_E2E_speed_eligible=False,
                retry_scope='Preserve this failed run; use a new output directory for any repair/retry'))
        raise
    finally:
        session_boundary.ProgressiveBank = previous_bank


if __name__ == '__main__':
    main()
