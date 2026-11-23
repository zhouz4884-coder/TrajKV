"""QA0 actual-8B W_O controls; two no-op and two encoded modes; no task answers.

This is an independent technical entry point. The frozen calibrator is used
only for native source preparation, teacher-forced suffix replay, signatures
and stable KL arithmetic. No fit/select routine or reference scorer is called.
"""
from __future__ import annotations

import argparse
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
import weakref

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_calibrate_v1 as setup
import round11_wo_runner_v1 as runner

SCHEMA = 'trajkv.round11.wo_technical_config.v1'
BASE_CALIBRATOR_SHA = 'abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a'
MODES = runner.MODES
write_new = runner._write_base


def require(value, message):
    runner.require(value, 'ROUND11_TECHNICAL: ' + message)


def runtime_source_manifest():
    require(runner.sha(ROOT / 'code/round10_calibrate_v1.py') == BASE_CALIBRATOR_SHA,
            'frozen preparation/TF helper changed')
    graph = set(runner.runtime_source_manifest()) | set(setup.PINS) | {
        'code/round10_calibrate_v1.py', 'code/round11_wo_technical_v1.py'}
    result = {role: runner.sha(ROOT / role) for role in sorted(graph)}
    require(all(result[role] == digest for role, digest in setup.PINS.items()),
            'frozen technical helper dependency changed')
    return result


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('purpose') == 'actual_model_technical_only'
            and config.get('technical_only') is True, 'explicit technical QA0 scope required')
    require(not (set(config) & (setup.FORBIDDEN | runner.frozen.FORBIDDEN)), 'no references or labels')
    require(config.get('model_id') == 'M1' and type(config.get('seed')) is int and config['seed'] == 0,
            'M1 seed0 only')
    require(config.get('question_suffix') == runner.frozen.SUFFIX
            and config.get('preprocessing', {}).get('max_patches') == 8,
            'original cap8 and suffix required')
    require(config.get('teacher_max_new_tokens') == 4, 'native teacher cap4 required')
    require(config.get('variant_source', {}).get('name') == 'r7_joint_k1_gt_cond'
            and config.get('expected_asset_sha256') == runner.frozen.PROFILES['r7_joint_k1_gt_cond']['assets'],
            'original low43 assets and variant required')
    require(isinstance(config.get('cal_select_inputs'), str)
            and isinstance(config.get('cal_select_sha256'), str) and len(config['cal_select_sha256']) == 64
            and config.get('cal_select_role', 'cal_select') == 'cal_select',
            'locked original no-answer calSelect source required')
    index = config.get('technical_source_index', 0)
    require(type(index) is int and 0 <= index < 16, 'calSelect source index0..15 required')
    require(isinstance(config.get('input_manifest_path'), str)
            and isinstance(config.get('input_manifest_sha256'), str) and len(config['input_manifest_sha256']) == 64,
            'exact original input manifest required')
    require(config.get('expected_source_hashes') == runtime_source_manifest(), 'complete exact technical runtime graph required')
    require(runner.sha(config['input_manifest_path']) == config['input_manifest_sha256'], 'original input manifest changed')
    return config


def native_prefix_identity(native):
    import torch
    return {'logical_length': int(native.get_seq_length()), 'layers': [
        {axis: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'device': str(tensor.device), 'sha256': setup.tensor_sha(tensor),
                'storage_bytes': tensor.untyped_storage().nbytes()}
         for axis, tensor in (('K', layer.keys), ('V', layer.values))}
        for layer in native.layers], 'all28_CUDA_FP16': len(native.layers) == 28 and all(
            tensor.is_cuda and tensor.dtype == torch.float16
            for layer in native.layers for tensor in (layer.keys, layer.values))}


def model_parameter_identity(model):
    """No parameter writes: retain exact per-storage versions and geometry."""
    return [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
             'device': str(value.device), 'version': value._version,
             'requires_grad': value.requires_grad, 'storage_ptr': value.untyped_storage().data_ptr()}
            for name, value in model.named_parameters()]


def teacher4_history(model, batch, native):
    import torch
    from m1_protocol import clone_m1_prefix_cache, m1_generate_cached
    cache = clone_m1_prefix_cache(native)
    raw = []
    def after(module, args, kwargs, result):
        value = getattr(result, 'logits', None)
        require(value is not None and bool(torch.isfinite(value).all()), 'nonfinite native teacher logits')
        raw.append(value[0, -1].detach().float().cpu().clone())
    handle = model.register_forward_hook(after, with_kwargs=True)
    try:
        with setup.first_call_guard(model, batch, cache) as events:
            sequence = m1_generate_cached(model, batch, cache, max_new_tokens=4)
        require(torch.equal(sequence[:, :batch.full_length], batch.ids), 'native teacher altered full prompt')
        tokens = sequence[0, batch.full_length:].detach().cpu().tolist()
        stop = setup.eos_ids(model)
        require(tokens and len(tokens) <= 4 and len(raw) == len(tokens)
                and not any(token in stop for token in tokens[:-1]), 'actual native teacher EOS/cap4 differs')
        return {'token_ids': tokens, 'raw_logits': torch.stack(raw), 'first_call': events[0],
                'EOS_reached': tokens[-1] in stop, 'max_new_tokens': 4,
                'teacher_history_is_actual_native_generate': True}
    finally:
        handle.remove()


def build_cache(native, batch, variant, config, *, mode=None, encode=True):
    from m1_protocol import clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from ownership import assert_no_full_prefix_alias
    clone = clone_m1_prefix_cache(native)
    refs = [weakref.ref(tensor) for layer in clone.layers for tensor in (layer.keys, layer.values)]
    source_storages = {(str(tensor.device), tensor.untyped_storage().data_ptr())
                       for layer in clone.layers for tensor in (layer.keys, layer.values)}
    kwargs = {role: config[role] for role in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source')}
    kwargs['expected_asset_sha256'] = config['expected_asset_sha256']
    if mode is None:
        from r7_legacy_bridge_v2 import build_r7_owners
        owners, metadata = build_r7_owners(clone, batch.visual_positions, variant, **kwargs)
    else:
        from round11_wo_value_v1 import build_round11_wo_owners
        owners, metadata = build_round11_wo_owners(clone, batch.visual_positions, variant, model=runner.current_model(),
            mode=mode, encode=encode, **kwargs)
    require(len(owners) == 28 and metadata.get('selected_stage_count') == 43
            and metadata.get('variant_sha256') == runner.frozen.canonical_sha(variant),
            'actual all28 K23/V20 profile changed')
    signatures = setup.state_signatures(owners)
    cache = HybridVisualCache(clone, batch.visual_positions, owners, consume_native=True)
    runner.frozen.share_original_visual_position_map(cache)
    clone = None
    gc.collect()
    require(not any(ref() is not None for ref in refs), 'serving cache retained cloned native source tensors')
    assert_no_full_prefix_alias(cache, source_storages)
    return cache, owners, metadata, signatures


def replay(model, batch, cache, owners, history):
    import torch
    from attention_bridge import attention_context
    calls = [0]
    outer = []
    def before(module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        positions = kwargs.get('position_ids')
        past = kwargs.get('past_key_values')
        outer.append({'input_tokens': int(ids.shape[-1]) if ids is not None else None,
            'input_ids_sha256': setup.tensor_sha(ids) if ids is not None else None,
            'position_ids_sha256': setup.tensor_sha(positions) if positions is not None else None,
            'cache_tokens_before': int(past.get_seq_length()) if past is not None else None})
    outer_handle = model.register_forward_pre_hook(before, with_kwargs=True)
    originals = []
    for owner in owners:
        old = owner.attend
        originals.append((owner, old))
        def counted(q, *, _old=old, **kwargs):
            calls[0] += 1
            return _old(q, **kwargs)
        owner.attend = counted
    logits = []
    started = time.perf_counter()
    try:
        with attention_context(model, cache=cache):
            for value in setup.feed_teacher(model, batch, cache, history['token_ids']):
                require(bool(torch.isfinite(value).all()), 'actual packed TF logits are nonfinite')
                logits.append(value.detach().float().cpu().clone())
        require(calls[0] == 28 * len(history['token_ids']), 'packed owners not executed all28 at every TF position')
        require(len(outer) == len(history['token_ids']), 'actual outer TF call count differs from teacher positions')
        matrix = torch.stack(logits)
        return matrix, {**setup.full_vocab_kl(history['raw_logits'], matrix),
            'actual_packed_owner_calls': calls[0], 'actual_teacher_forced_positions': len(logits),
            'actual_outer_forward_call_count': len(outer), 'actual_outer_forward_calls': outer,
            'question_suffix_recomputed': True, 'native_question_KV_reused': False,
            'first_suffix_raw_logits_sha256': setup.tensor_sha(matrix[0]),
            'all_TF_raw_logits_sha256': setup.tensor_sha(matrix), 'wall_s': time.perf_counter() - started}
    finally:
        outer_handle.remove()
        for owner, old in originals:
            owner.attend = old


def save_logits(output, name, logits, tokens):
    import torch
    path = output / 'private_logits' / (name + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'raw technical logits must never overwrite')
    torch.save({'schema': 'trajkv.round11.wo_technical_raw_logits.v1',
        'raw_teacher_forced_logits': logits, 'teacher_token_ids': tokens,
        'first_row_is_actual_full_suffix_last_position': True,
        'no_HF_logits_processor_scores': True, 'no_reference_answers': True}, path)
    return {'file': str(path), 'sha256': runner.sha(path), 'bytes': path.stat().st_size,
            'tensor_sha256': setup.tensor_sha(logits), 'shape': list(logits.shape),
            'dtype': str(logits.dtype), 'first_suffix_tensor_sha256': setup.tensor_sha(logits[0])}


def technical_one_source(model, processor, row, variant, config, output):
    import torch
    from ownership import cache_storage
    sources_before = runtime_source_manifest()
    inputs_before = {role: runner.sha(config[role]) for role in ('cal_select_inputs', 'input_manifest_path')}
    require(runner.sha(row['image_path']) == row['original_image_sha256'], 'original image changed')
    batch, native, context = setup.prepare_case(model, processor, row)
    template = processor.chat_template
    context.update(raw_question=row['question'], rendered_question=row['question'] + runner.frozen.SUFFIX,
        full_prompt=batch.prompt, input_token_ids=batch.ids.detach().cpu().reshape(-1).tolist(),
        native_position_ids=batch.positions.detach().cpu().reshape(-1).tolist(),
        native_position_ids_shape=list(batch.positions.shape),
        visual_positions=batch.visual_positions.detach().cpu().tolist(),
        compression_boundary=batch.boundary, question_suffix_tokens=batch.suffix_length,
        chat_template_hash=hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str)
            else runner.frozen.canonical_sha(template),
        native_tile_layout=runner.frozen.validate_native_tile_layout(batch))
    require(row['dataset'] == 'textvqa', 'QA0 uses the frozen first TextVQA calSelect source')
    native_before = native_prefix_identity(native)
    require(native_before['all28_CUDA_FP16'], 'actual all28 CUDA FP16 native prefix required')
    parameters_before = model_parameter_identity(model)
    history = teacher4_history(model, batch, native)
    teacher_logits_file = save_logits(output, 'native_teacher_cap4', history['raw_logits'], history['token_ids'])
    require(native_prefix_identity(native) == native_before, 'native teacher mutated independent original prefix')
    n = int(batch.visual_positions.numel())
    expected_bytes = 4590 * n + 89733
    reference_cache, reference_owners, reference_metadata, reference_signature = build_cache(native, batch, variant, config)
    reference_owner = cache_storage(reference_cache, reference_cache.visual_positions)
    require(reference_owner['complete_visual_owner_bytes'] == expected_bytes, 'original R7 B0 physical bytes differ')
    original_logits, original_loss = replay(model, batch, reference_cache, reference_owners, history)
    original_logits_file = save_logits(output, 'original_R7', original_logits, history['token_ids'])
    require(setup.state_signatures(reference_owners) == reference_signature, 'original R7 TF changed packed fields')
    reference_cache = reference_owners = None
    gc.collect()
    controls = []
    # Both mode0 variants exercise their deployed readers with untouched
    # original fields; encoded modes are separate actual TF evaluations.
    for encode in (False, True):
        for mode in MODES:
            cache, owners, metadata, signature = build_cache(native, batch, variant, config, mode=mode, encode=encode)
            detail = metadata['round11_wo_candidate']
            require(detail['mode'] == mode and detail['encoding_applied'] is encode
                    and detail['control_mode0_only'] is (not encode), 'actual factory no-op/mode flags differ')
            require(signature['K_fields_sha256'] == reference_signature['K_fields_sha256'], 'new factory changed any original K field')
            if not encode:
                require(signature == reference_signature, 'mode0 factory changed an original packed field')
            before = cache_storage(cache, cache.visual_positions)
            require(before['complete_visual_owner_bytes'] == expected_bytes, 'technical owner does not equal actual B0')
            if encode:
                budget = runner.round11_budget_guard(mode, n, expected_bytes, metadata, 'r7_joint_k1_gt_cond')
            else:
                budget = {'mode0': True, 'complete_visual_owner_bytes': expected_bytes, 'same_B0': True}
            logits, loss = replay(model, batch, cache, owners, history)
            logits_file = save_logits(output, ('encoded_' if encode else 'mode0_') + mode, logits, history['token_ids'])
            require(setup.state_signatures(owners) == signature, 'actual packed fields changed during TF')
            after = cache_storage(cache, cache.visual_positions)
            require(after['complete_visual_owner_bytes'] == expected_bytes, 'persistent owner changed during TF')
            delta = (logits - original_logits).abs()
            record = {'mode': mode, 'encoding_applied': encode, 'control_mode0_only': not encode,
                'metadata': metadata, 'state_signature': signature,
                'complete_K_equals_original_R7': True, 'mode0_all_reader_fields_bitwise_identical': not encode,
                'owner_before_TF': before, 'owner_after_TF': after, 'physical_budget': budget,
                'loss_vs_native_teacher': loss, 'first_suffix_logits_equal_original_R7': bool(torch.equal(logits[0], original_logits[0])),
                'raw_TF_and_first_suffix_logits_file': logits_file,
                'all_TF_logits_equal_original_R7': bool(torch.equal(logits, original_logits)),
                'first_suffix_max_abs_vs_original_R7': float(delta[0].max()),
                'all_TF_max_abs_vs_original_R7': float(delta.max()),
                'all_TF_mean_abs_vs_original_R7': float(delta.mean()),
                'mode0_original_R7_all_TF_bitwise_identity_required': not encode,
                'packed_fields_unchanged_during_TF': True, 'cloned_native_source_released': True,
                'original_FP16_visual_alias_retained': False}
            write_new(output / 'controls' / (('encoded_' if encode else 'mode0_') + mode + '.json'), record)
            if not encode:
                require(torch.equal(logits, original_logits),
                        mode + ' mode0 does not exactly reproduce original R7 TF logits; stop before formal QA')
            controls.append(record)
            cache = owners = logits = None
            gc.collect()
            torch.cuda.synchronize()
            require(native_prefix_identity(native) == native_before, 'branch mutated independent native prefix')
    require(model_parameter_identity(model) == parameters_before, 'native weight versions/storage changed')
    require(runtime_source_manifest() == sources_before, 'runtime graph changed during technical replay')
    require({role: runner.sha(config[role]) for role in inputs_before} == inputs_before, 'original locked inputs changed')
    require(runner.sha(row['image_path']) == row['original_image_sha256'], 'source image changed during replay')
    return {'schema': 'trajkv.round11.wo_technical_result.v1', 'status': 'PASS_ACTUAL_MODEL_WO_PACKED_TECHNICAL_QA0',
        'actual_GPU': True, 'actual_GPU_name': torch.cuda.get_device_name(),
        'actual_GPU_capability': list(torch.cuda.get_device_capability()),
        'actual_model_parameter_count': sum(value.numel() for value in model.parameters()),
        'model_capture': runner.capture_identity(),
        'source_context': context, 'native_prefix_before': native_before,
        'native_prefix_after': native_prefix_identity(native), 'native_prefix_unchanged': True,
        'native_weight_versions_storage_unchanged': True, 'original_R7_signature': reference_signature,
        'original_R7_metadata': reference_metadata, 'original_R7_actual_owner': reference_owner,
        'original_R7_loss_vs_native_teacher': original_loss,
        'native_teacher_raw_logits_file': teacher_logits_file,
        'original_R7_raw_logits_file': original_logits_file,
        'teacher_token_ids': history['token_ids'], 'teacher_cap': 4,
        'teacher_actual_native_generate': True, 'teacher_EOS_reached': history['EOS_reached'],
        'teacher_first_call': history['first_call'], 'teacher_sources': 1,
        'teacher_tokens': len(history['token_ids']), 'actual_TF_source_evaluations': 5,
        'mode0_TF_source_evaluations': 2, 'new_encoded_mode_TF_source_evaluations': 2,
        'original_R7_TF_source_evaluations': 1, 'controls': controls,
        'QA_events': 0, 'new_free_generation_task_answers': 0, 'references_read': False,
        'quality_claim': False, 'source_quality_gate_pass_claimed': False,
        'canonical_SourceRun_admission_claimed': False, 'new_parameters_selected': False,
        'speed_or_peak_benchmark_claim': False, 'runtime_source_hashes': sources_before}


def run(config, output):
    validate_config(config)
    from round10_inputs_v1 import load_round10_inputs
    from model_registry import get_model_spec
    from evaluate import model_identity
    rows = load_round10_inputs(config['cal_select_inputs'], config['input_manifest_path'],
        expected_sha256=config['cal_select_sha256'], expected_role='cal_select')
    require(len(rows) == 16, 'original calSelect16 cohort required')
    row = rows[config.get('technical_source_index', 0)]
    variant = runner.frozen.frozen_variant(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    import tokenizers
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from attention_bridge import language_attention_modules
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'actual CUDA required')
    torch.cuda.set_device(0)
    require(torch.cuda.get_device_capability() == (8, 6), 'actual RTX3090 SM86 required')
    identity = model_identity(config, get_model_spec('M1'))
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
        device_map={'': 0}, local_files_only=True, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    require(all(value.dtype == torch.float16 for value in model.parameters())
            and not getattr(model, 'is_quantized', False), 'native unquantized FP16 model required')
    require(len(language_attention_modules(model)[1]) == 28
            and int(model.generation_config.num_beams or 1) == 1, 'native M1 geometry/greedy teacher required')
    manifest = {'schema': 'trajkv.round11.wo_technical_manifest.v1', 'config': config,
        'model_spec': get_model_spec('M1'), 'model_identity': identity,
        'actual_model_parameter_count': sum(value.numel() for value in model.parameters()),
        'model_capture': runner.capture_identity(),
        'actual_GPU_name': torch.cuda.get_device_name(),
        'actual_GPU_capability': list(torch.cuda.get_device_capability()),
        'runtime_source_root': str(ROOT), 'runtime_source_hashes': runtime_source_manifest(),
        'runtime_executable': sys.executable, 'PID': os.getpid(), 'parent_PID': os.getppid(),
        'started_unix': time.time(), 'torch_version': torch.__version__,
        'transformers_version': transformers.__version__, 'tokenizers_version': tokenizers.__version__,
        'python_version': platform.python_version(), 'CUDA_version': torch.version.cuda,
        'native_generation_config': model.generation_config.to_dict(), 'teacher_cap': 4,
        'teacher_generation_overrides': {'do_sample': False, 'max_new_tokens': 4, 'use_cache': True},
        'weight_dtype': 'float16', 'attention_implementation': 'sdpa', 'tf32': False,
        'runtime_library_source_sha256': {str(Path(inspect.getfile(kind))): runner.sha(inspect.getfile(kind))
            for kind in (type(model), type(processor), type(processor.tokenizer))},
        'references_read': False, 'QA_events': 0, 'quality_claim': False}
    write_new(output / 'manifest.json', manifest)
    try:
        result = technical_one_source(model, processor, row, variant, config, output)
    except Exception as exc:
        # Each named TF file is opened only after its whole replay returns.
        # Even a partial save therefore proves that replay occurred; it does
        # not prove a valid tensor file or a completed technical control.
        tf_artifacts = [output / 'private_logits' / (name + '.pt')
                        for name in ('original_R7', *(('mode0_' + m) for m in MODES),
                                     *(('encoded_' + m) for m in MODES))]
        present = [p.name for p in tf_artifacts if p.exists()]
        write_new(output / 'technical_failure.json', {
            'schema': 'trajkv.round11.wo_technical_failure.v1', 'status': 'FAIL_TECHNICAL_NO_QUALITY_ADMISSION',
            'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
            'QA_events': 0, 'quality_claim': False, 'formal_free_QA_may_start': False,
            'actual_TF_source_evaluations_artifact_backed_lower_bound': len(present),
            'TF_lower_bound_evidence_files': present,
            'TF_expected_if_success_not_executed_count': 5,
            'native_teacher_generation_artifact_present': (output / 'private_logits/native_teacher_cap4.pt').exists(),
            'file_presence_is_not_valid_tensor_or_passed_control_claim': True,
            'preserved_partial_controls': sorted(path.name for path in (output / 'controls').glob('*.json'))})
        raise
    require(model_identity(config, get_model_spec('M1')) == identity, 'on-disk original model identity changed')
    write_new(output / 'technical_result.json', result)
    write_new(output / 'complete.json', {'schema': 'trajkv.round11.wo_technical_complete.v1',
        'status': result['status'], 'manifest_sha256': runner.sha(output / 'manifest.json'),
        'technical_result_sha256': runner.sha(output / 'technical_result.json'),
        'QA_events': 0, 'new_free_generation_task_answers': 0, 'actual_TF_source_evaluations': 5,
        'quality_claim': False, 'completed_unix': time.time()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    config['technical_config_sha256'] = runner.sha(args.config)
    with runner.capture_loaded_model():
        result = run(config, args.output)
    print(json.dumps({'status': result['status'], 'QA_events': 0,
                      'actual_TF_source_evaluations': result['actual_TF_source_evaluations']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
