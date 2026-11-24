"""Fresh Round12 confirmation: native-full FP16 and real packed R7/static.

This is a new confirmation admission, never a translation to development.
Compressed generation retains the byte-frozen Round10 driver. FP16 directly
uses the original evaluate.py native-full generate/observer operations, not
an FP16 HybridVisualOwner control. No source/size/winner is selected here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen
import round10_inputs_v1 as old_inputs
import round12_runner_v1 as development
import round12_confirmation_admission_v1 as admission

SCHEMA = admission.STATIC_RUNNER_SCHEMA
ROLE = admission.ROLE
BASELINE_MODES = admission.BASELINE_MODES
CONFIRMATION_MODES = (*BASELINE_MODES, 'static')
BASE_RUNNER_SHA = development.BASE_RUNNER_SHA
NATIVE_EVALUATE_SHA = 'dbb556d87827c146e596b1e01de6c58a7a8b08f788c4c13d18129f4b1f13f75c'
RUNTIME_FILES = tuple(dict.fromkeys((*development.BASE_RUNTIME_FILES,
    'round12_confirmation_runner_v1.py', 'round12_confirmation_admission_v1.py',
    'round12_confirmation_data_v1.py', 'round12_confirmation_native_bindings_v1.py',
    'round12_static_bank_v1.py', 'round12_static_codec_v1.py', 'round12_static_attention_v1.py')))
_BUILD_ORIGINAL = frozen.build_candidate_owners
_WRITE_ORIGINAL = frozen.write_new
_IDENTITY = None
_CONFIG = None

require, sha = admission.require, admission.sha


def runtime_source_manifest():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA
            and sha(ROOT / 'code/evaluate.py') == NATIVE_EVALUATE_SHA, 'frozen native generation dependencies changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def logical_mode(config):
    mode = config.get('confirmation_mode')
    require(mode in CONFIRMATION_MODES, 'explicit confirmation_mode must be FP16/R7low/R7high/static')
    return mode


def validate_selected_algorithm(config, identity):
    if logical_mode(config) != 'static':
        return None
    path, digest = config.get('selected_algorithm_path'), config.get('selected_algorithm_sha256')
    require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
            'static candidate requires frozen selected algorithm path/SHA')
    locks = identity['decision']['method_asset_locks']
    require(any(x['sha256'] == digest for x in locks), 'selected algorithm is not locked by frozen decision')
    spec = admission.read(path)
    require(spec.get('schema') == 'trajkv.round12.selected_static_algorithm.v1'
            and spec.get('name') == config['selected_method_name'], 'selected static identity differs')
    # Paths may be rebound, but every numeric value and content SHA is fixed.
    normalize = lambda c: {k: v for k, v in c.items() if not k.endswith('_path')}
    require(normalize(spec['candidate']) == normalize(config['candidate']), 'candidate parameters or asset SHA changed')
    require(spec['variant_profile'] == config['variant_source']['name']
            and spec['variant_source_sha256'] == config['variant_source']['sha256'], 'selected encoding lineage changed')
    sources = runtime_source_manifest()
    for name, expected in spec['algorithm_source_hashes'].items():
        require(sources.get(name) == expected, 'selected algorithm source changed: ' + name)
    return {'path': path, 'sha256': digest, 'name': spec['name']}


def candidate_assets(config):
    mode = logical_mode(config)
    if mode in ('R7low', 'R7high'):
        return {'confirmation_mode': mode, 'frozen_original_profile': True,
                'parameter_fit_in_inference_runner': False}, {}
    if mode == 'static':
        details, locks = development.candidate_assets(config)
        locks = dict(locks)
        locks[config['selected_algorithm_path']] = config['selected_algorithm_sha256']
        return {'confirmation_mode': mode, **details}, locks
    return {'confirmation_mode': 'FP16', 'native_full_generate': True, 'HybridVisualOwner_used': False}, {}


def validate_config(config):
    global _IDENTITY, _CONFIG
    require(config.get('schema') == SCHEMA, 'confirmation runner schema differs; use exported constant')
    identity = admission.validate_config_admission(config)
    mode = logical_mode(config)
    admission.validate_execution_selection(config, identity)
    require(config.get('method', {}).get('family') == 'trajkv_round12_fresh_confirmation', 'fresh method family required')
    require(config.get('position_owner_layout') == 'single_shared_visual_position_map', 'declared original VP layout required')
    require(config.get('scope'), 'explicit confirmation scope required')
    if mode == 'FP16':
        require(config.get('candidate', {}).get('mode') == 'FP16', 'native FP16 config must name its mode')
    else:
        source = config['variant_source']
        expected_profile = 'r7_budget_k2_cond' if mode == 'R7high' else 'r7_joint_k1_gt_cond'
        require(source.get('name') == expected_profile and source.get('sha256') == frozen.ORIGINAL_CONFIG_SHA,
                'original frozen cond profile lineage required')
        require(config.get('expected_asset_sha256') == frozen.PROFILES[expected_profile]['assets'], 'frozen profile assets differ')
        for field in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source'):
            require(isinstance(config.get(field), str) and bool(config[field]), 'explicit legacy path required: ' + field)
        if mode in ('R7low', 'R7high'):
            require(config.get('candidate', {}).get('mode') == 'original', 'R7 baseline requires original frozen factory')
        else:
            require(config.get('candidate', {}).get('mode') in development.MODES, 'unknown selected static mode')
            validate_selected_algorithm(config, identity)
    candidate_assets(config)
    require(config.get('expected_source_hashes') == runtime_source_manifest(), 'actual confirmation source graph differs')
    _IDENTITY, _CONFIG = identity, config
    return config


def load_runner_inputs(path, manifest_path=None, expected_sha256=None, expected_role=ROLE):
    return admission.load_inputs(path, manifest_path, expected_sha256, expected_role)


def build_candidate_owners(cache, vp, variant, config, assets):
    if logical_mode(config) in ('R7low', 'R7high'):
        return _BUILD_ORIGINAL(cache, vp, variant, config, assets)
    return development.build_candidate_owners(cache, vp, variant, config, assets)


def physical_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    logical = logical_mode(_CONFIG)
    stages = [list(x) for x in metadata.get('selected_stages', [])]
    require(len(stages) == metadata['selected_stage_count'], 'actual stage manifest missing')
    if logical in ('R7low', 'R7high'):
        expected = ((3816 + 18 * len(stages)) * n + 89733 if logical == 'R7low'
                    else (5608 + 18 * len(stages)) * n + 91653)
        require(complete_bytes == expected, 'frozen original physical owner formula differs')
    else:
        original = [list(x) if len(x) == 4 else [*x, 0] for x in _CONFIG['variant']['selected_stages'][:43]]
        require(stages == original if mode != 'R' else stages[:43] == original, 'concrete original low43 coverage differs')
        require(metadata['returned_owners_unique_tensor_capacity_bytes'] == complete_bytes
                and metadata['round12_static']['mode'] == mode, 'selected actual owner/factory differs')
    return {'schema': 'trajkv.round12.fresh_confirmation_budget.v1', 'confirmation_mode': logical,
            'mode': mode, 'N': n, 'complete_visual_owner_bytes': complete_bytes,
            'selected_stages': stages, 'selected_stage_count': len(stages),
            'strict_cap_not_inferred_from_name': True, 'QA_is_fresh': True}


def write_new(path, value):
    path = Path(path)
    if path.name == 'manifest.json':
        value = dict(value)
        value.update(runner_schema=SCHEMA, confirmation_mode=logical_mode(value['config']),
            fresh_confirmation_admission={k: v for k, v in _IDENTITY.items() if k not in ('rows', 'native_bindings', 'decision')},
            source_driver_contract='fresh confirmation admission over frozen whole-model native M1 generation',
            phase_is_confirmatory=True, old_predictions_or_logits_reused=False,
            native_binding_scope='new CPU processor evidence; does not prove bitexact logits',
            base_profile_numeric_fields_unchanged=logical_mode(value['config']) in ('R7low', 'R7high'),
            round12_confirmation_candidate_assets=value.pop('round10_candidate_assets', None))
    elif path.parent.name == 'records':
        value['round12_confirmation_schema'] = SCHEMA
        value['confirmation_mode'] = logical_mode(_CONFIG)
        value['phase_is_confirmatory'] = True
        value['old_prediction_reused'] = False
        if 'round10_budget_vs_original_R7_low' in value:
            value['round12_confirmation_budget'] = value.pop('round10_budget_vs_original_R7_low')
        if value.get('status') == 'PASS':
            proof = admission.validate_native_record(value, _IDENTITY['native_bindings'])
            value['fresh_CPU_native_input_binding'] = proof
            if not proof['matched']:
                value['status'] = 'INPUT_MISMATCH'
                value['error'] = 'Fresh CPU input identity differs: ' + json.dumps(proof['mismatches'])
                value['untrusted_output_token_ids'] = value.get('output_token_ids')
                value['untrusted_raw_prediction'] = value.get('raw_prediction')
                value['output_token_ids'] = value['raw_prediction'] = None
    elif path.name == 'complete.json':
        value = {**value, 'confirmation_schema': SCHEMA, 'phase_is_confirmatory': True,
                 'confirmation_mode': logical_mode(_CONFIG), 'old_predictions_reused': False,
                 'canonical_SourceRun_admission_claimed': False}
    return _WRITE_ORIGINAL(path, value)


@contextmanager
def compressed_driver_overrides():
    profiles = deepcopy(frozen.PROFILES)
    if logical_mode(_CONFIG) == 'static' and _CONFIG['candidate']['mode'] == 'R':
        profiles['r7_joint_k1_gt_cond']['require_43_stages'] = False
    changes = {'SCHEMA': SCHEMA, 'MODES': (*development.MODES, 'original'), 'PROFILES': profiles,
        'candidate_assets': candidate_assets, 'validate_config': validate_config,
        'build_candidate_owners': build_candidate_owners, 'round10_budget_guard': physical_budget_guard,
        'runtime_source_manifest': runtime_source_manifest, 'write_new': write_new}
    saved = {k: getattr(frozen, k) for k in changes}
    loader = old_inputs.load_round10_inputs
    try:
        for key, value in changes.items():
            setattr(frozen, key, value)
        old_inputs.load_round10_inputs = load_runner_inputs
        yield
    finally:
        old_inputs.load_round10_inputs = loader
        for key, value in saved.items():
            setattr(frozen, key, value)


def run_native_fp16(config, config_path, output):
    """Independent full-prompt model.generate, never prefix+FP16-Hybrid."""
    from model_registry import get_model_spec
    spec = get_model_spec('M1')
    selected = _IDENTITY['rows']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    expected = [r['id'] for r in selected]
    source_hashes = runtime_source_manifest()
    locks = {config['inputs']: config['inputs_sha256'], config['input_manifest_path']: config['input_manifest_sha256'],
             config['frozen_decision_path']: config['frozen_decision_sha256'], config['native_bindings_path']: config['native_bindings_sha256']}
    manifest = {'schema': 'baseline_matrix.raw_inference.v1', 'runner_schema': SCHEMA,
        'config': config, 'config_sha256': sha(config_path), 'inputs_sha256': config['inputs_sha256'],
        'expected_ids': expected, 'expected_QA': len(selected), 'model_spec': spec,
        'references_read': False, 'prediction_cache_used': False, 'source_hashes': source_hashes,
        'source_driver_contract': 'evaluate.py native full prompt generation operations; new fresh-source admission',
        'reference_scope': 'native_full_prompt_independent_reference', 'HybridVisualOwner_used': False,
        'phase_is_confirmatory': True, 'confirmation_mode': 'FP16', 'started_unix': time.time(),
        'timing_scope': 'instrumented quality run, not controlled speed benchmark'}
    records, load_error, manifest_written = [], None, False
    record_path = output / 'records.jsonl'
    model = processor = None

    def append(record):
        write_new(output / 'records' / (f"{len(records):06d}_" + hashlib.sha256(record['id'].encode()).hexdigest()[:16] + '.json'), record)
        with record_path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        records.append(record)

    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from evaluate import model_identity, tensor_sha, GenerationObserver, observe_generation
        from m1_protocol import prepare_m1_inputs, m1_generate_native
        from ownership import model_storage, cache_storage
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available(), 'actual CUDA required')
        torch.cuda.set_device(0)
        require(torch.cuda.get_device_capability() == (8, 6), 'declared3090 SM86 required')
        model_id = model_identity(config, spec)
        processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
            device_map={'': 0}, local_files_only=True, attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        require(all(p.dtype == torch.float16 for p in model.parameters()) and not getattr(model, 'is_quantized', False), 'original full FP16 weights required')
        parameters = sum(p.numel() for p in model.parameters())
        require(parameters == 7944373760, 'full M1 parameter identity differs')
        manifest.update(model_identity=model_id, full_model_parameter_count=parameters,
            torch_version=torch.__version__, transformers_version=transformers.__version__, CUDA_version=torch.version.cuda,
            hardware=torch.cuda.get_device_name(), python_version=platform.python_version(), model_residency=model_storage(model),
            runtime_library_source_sha256={str(Path(inspect.getfile(c))): sha(inspect.getfile(c))
                for c in (type(model), type(processor), type(processor.tokenizer))},
            generation_overrides={'do_sample': False, 'max_new_tokens': 64, 'use_cache': True})
        write_new(output / 'manifest.json', manifest)
        manifest_written = True
        for row in selected:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            record = {**row, 'model_id': 'M1', 'model_commit': spec['revision'], 'model_arch': spec['architecture'],
                'dtype': 'float16', 'method': config['method']['id'], 'method_contract': config['method'],
                'preprocessing': config['preprocessing'], 'status': 'RUNNING', 'output_token_ids': None,
                'raw_prediction': None, 'official_score': None, 'score_status': 'NOT_READ_OR_COMPUTED_BY_RUNNER',
                'reference_scope': 'native_full_prompt_independent_reference', 'HybridVisualOwner_used': False,
                'timing_scope': manifest['timing_scope']}
            image = batch = observer = generated = None
            handles, first_logits, first_events = [], [], []
            try:
                require(sha(row['image_path']) == row['original_image_sha256'], 'INPUT_MISMATCH: new image SHA differs')
                with Image.open(row['image_path']) as raw:
                    image = raw.convert('RGB')
                rendered = row['question'] + admission.SUFFIX
                batch = prepare_m1_inputs(processor, image, rendered, image_token_id=spec['image_token_id'], max_patches=8)
                template = processor.chat_template
                record.update(raw_question=row['question'], rendered_question=rendered, question_suffix=admission.SUFFIX,
                    raw_question_sha256=row['question_sha256'], rendered_question_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    full_prompt=batch.prompt, full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest(),
                    chat_template_hash=hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else frozen.canonical_sha(template),
                    input_tokens=batch.full_length, input_token_ids=batch.ids.detach().cpu().reshape(-1).tolist(),
                    input_ids_sha256=tensor_sha(batch.ids), question_suffix_sha256=tensor_sha(batch.ids[:, batch.boundary:]),
                    compression_boundary=batch.boundary, question_suffix_tokens=batch.suffix_length,
                    visual_positions=batch.visual_positions.detach().cpu().tolist(), actual_visual_token_count=int(batch.visual_positions.numel()),
                    native_position_ids_shape=list(batch.positions.shape), native_position_ids_sha256=tensor_sha(batch.positions),
                    native_position_ids=batch.positions.detach().cpu().reshape(-1).tolist(),
                    question_unknown_prefix_verified=_IDENTITY['native_bindings'][row['id']]['question_unknown_prefix_verified'])
                proof = admission.validate_native_record(record, _IDENTITY['native_bindings'])
                require(proof['matched'], 'INPUT_MISMATCH: fresh CPU processor identities differ: ' + json.dumps(proof['mismatches']))
                record['fresh_CPU_native_input_binding'] = proof

                def before(module, positional, kwargs):
                    if first_events:
                        return
                    ids = kwargs.get('input_ids')
                    if ids is None and positional:
                        ids = positional[0]
                    positions = kwargs.get('position_ids')
                    require(ids is not None and tensor_sha(ids) == record['input_ids_sha256']
                            and ids.shape[-1] == batch.full_length, 'INPUT_MISMATCH: native full first call must contain full prompt')
                    first_events.append({'input_tokens': int(ids.shape[-1]), 'input_ids_sha256': tensor_sha(ids),
                        'explicit_position_ids_sha256': tensor_sha(positions) if positions is not None else None,
                        'native_inferred_positions_if_not_explicit': positions is None,
                        'planned_serial_position_ids_sha256': record['native_position_ids_sha256'],
                        'first_call_is_full_native_prompt': True, 'old_logits_injected': False})

                def after(module, positional, out):
                    if first_logits:
                        return
                    first_logits.append(out.logits[0, -1].detach().cpu().contiguous().clone())
                    owner = cache_storage(out.past_key_values, batch.visual_positions)
                    record['owner_after_native_full_prefill'] = owner
                    for field in ('stored_visual_KV_bytes', 'complete_visual_owner_bytes', 'stored_visual_payload_bytes',
                                  'coder_metadata_bytes', 'positions_bytes', 'encoder_constants_bytes', 'text_positions_bytes', 'text_kv_bytes'):
                        record[field] = owner[field]
                    denominator = 28 * 2 * 4 * int(batch.visual_positions.numel()) * 128 * 2
                    record.update(full_kv_byte_denominator=denominator,
                                  actual_effective_visual_bits=16 * owner['stored_visual_KV_bytes'] / denominator)

                observer = GenerationObserver(model, started, batch)
                with observe_generation(model, observer):
                    observer.phase = 'generation'
                    handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after)]
                    try:
                        generated = m1_generate_native(model, batch, max_new_tokens=64)
                    finally:
                        for handle in handles:
                            handle.remove()
                        handles = []
                tokens = generated[0, batch.full_length:].detach().cpu().tolist()
                require(tokens and len(tokens) <= 64 and tokens == observer.stream_ids and first_logits and first_events,
                        'INPUT_MISMATCH: native streamed/returned answers differ')
                logits = first_logits[0]
                require(bool(torch.isfinite(logits).all()), 'NONFINITE_LOGITS: new native first logits')
                target = output / 'first_logits' / (hashlib.sha256(row['id'].encode()).hexdigest()[:24] + '.pt')
                target.parent.mkdir(exist_ok=True)
                require(not target.exists(), 'first logits evidence must not overwrite')
                torch.save({'id': row['id'], 'raw_last_position_logits': logits, 'scope': 'fresh native full prompt before logits processors'}, target)
                top = torch.topk(logits.float(), 5)
                record.update(status='PASS', output_token_ids=tokens, output_tokens=len(tokens),
                    raw_prediction=processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    raw_decode_with_special_tokens=processor.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    first_generation_call=first_events[0], first_logits={'file': str(target), 'sha256': sha(target),
                        'bytes': target.stat().st_size, 'argmax_id': int(logits.argmax()), 'top5_ids': top.indices.tolist(),
                        'top5_logits': top.values.tolist(), 'finite': True}, encode_ms=0.0,
                    outer_forward_calls=observer.calls,
                    ttft_ms=(observer.token_times[0] - started) * 1000,
                    inter_token_ms=[(b - a) * 1000 for a, b in zip(observer.token_times, observer.token_times[1:])])
            except Exception as exc:
                error = str(exc)
                record.update(status='OOM_AT_DECLARED_INPUT' if 'out of memory' in error.lower() else
                    'INPUT_MISMATCH' if 'INPUT_MISMATCH' in error else 'FAILED_RUN', error_type=type(exc).__name__,
                    error=error, traceback=traceback.format_exc(), output_token_ids=None, raw_prediction=None)
            finally:
                for handle in handles:
                    handle.remove()
                record.update(wall_ms=(time.perf_counter() - started) * 1000,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    end_allocated_bytes=torch.cuda.memory_allocated())
                append(record)
                print(json.dumps({'id': row['id'], 'status': record['status'], 'completed': len(records), 'expected': len(selected)}), flush=True)
                image = batch = observer = generated = None
                first_logits.clear()
                gc.collect()
                torch.cuda.empty_cache()
        require(runtime_source_manifest() == source_hashes and all(sha(p) == digest for p, digest in locks.items()),
                'confirmation source/data/decision changed')
    except Exception as exc:
        load_error = {'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if not manifest_written:
            manifest['initialization_error'] = load_error
            write_new(output / 'manifest.json', manifest)
        done = {r['id'] for r in records}
        for row in selected:
            if row['id'] not in done:
                append({**row, 'model_id': 'M1', 'model_commit': spec['revision'], 'method': config['method']['id'],
                    'method_contract': config['method'], 'status': 'FAILED_RUN', 'raw_prediction': None,
                    'output_token_ids': None, 'official_score': None, 'score_status': 'NOT_READ_OR_COMPUTED_BY_RUNNER',
                    'error': load_error})
        if not record_path.exists():
            record_path.open('x').close()
        counts = dict(__import__('collections').Counter(r['status'] for r in records))
        complete = {'status': 'PASS_RAW_COMPLETE' if load_error is None and counts.get('PASS', 0) == len(selected) else 'COMPLETE_WITH_FAILURES',
            'expected_QA': len(selected), 'recorded_QA': len(records), 'counts': counts,
            'coverage_complete': len(records) == len(selected), 'quality_scored': False,
            'failed_QA_must_remain_in_primary_denominator': True, 'load_error': load_error,
            'manifest_sha256': sha(output / 'manifest.json'), 'records_jsonl_sha256': sha(record_path),
            'phase_is_confirmatory': True, 'confirmation_mode': 'FP16',
            'actual_new_terminal_answers': sum(bool(r.get('output_token_ids')) for r in records),
            'native_full_generate': True, 'HybridVisualOwner_used': False, 'completed_unix': time.time()}
        write_new(output / 'complete.json', complete)
    return 0 if complete['status'] == 'PASS_RAW_COMPLETE' else 1


def config_template(mode='static'):
    require(mode in CONFIRMATION_MODES, 'unknown template mode')
    c = admission.data_config_template()
    c.update(schema=SCHEMA, confirmation_mode=mode, selected_method_name=None,
        method={'id': None, 'family': 'trajkv_round12_fresh_confirmation', 'information_condition': 'query_unknown'},
        scope='fresh independent-source confirmation; no old prediction/native hash reuse',
        candidate={'mode': 'FP16' if mode == 'FP16' else 'original' if mode in ('R7low', 'R7high') else None},
        variant_source=None, variant=None, expected_asset_sha256=None,
        legacy_a2=None, legacy_experiments=None, legacy_scripts=None, kivi_source=None,
        selected_algorithm_path=None, selected_algorithm_sha256=None,
        position_owner_layout='single_shared_visual_position_map', expected_source_hashes=None,
        status='UNCONFIGURED_NO_CONFIRMATION_DATA_OR_WINNER_SELECTED')
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-config', action='store_true')
    parser.add_argument('--print-template', choices=CONFIRMATION_MODES)
    args = parser.parse_args()
    if args.print_template:
        print(json.dumps(config_template(args.print_template), ensure_ascii=False, indent=2))
        return 0
    require(args.config is not None, 'explicit config required')
    config = validate_config(admission.read(args.config))
    if args.validate_config:
        print(json.dumps({'status': 'PASS_FRESH_CONFIRMATION_CONFIG_ONLY', 'mode': logical_mode(config),
                          'expected_QA': len(_IDENTITY['rows']), 'model_or_GPU_calls': 0}, ensure_ascii=False))
        return 0
    require(args.output is not None, 'new output directory required for actual inference')
    if logical_mode(config) == 'FP16':
        return run_native_fp16(config, args.config, args.output)
    with compressed_driver_overrides():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
