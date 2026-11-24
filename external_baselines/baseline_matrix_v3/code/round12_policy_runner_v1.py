"""Prospective real gated R12 deployment; no thresholds are selected here.

Every source really executes the low probe. a0 reuses that same rolled-back
packed low cache and recomputes the full original suffix. P/Z really rebuild
their selected prefix. No historical prediction or counterfactual splice is
admitted. Frozen v2 numerical helpers and v3 coverage semantics are preserved.
"""
from __future__ import annotations

import argparse
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
import round12_policy_contract_v1 as contract

SCHEMA = 'trajkv.round12_policy.raw.v1'
DEV_ROLE = 'round12_development'
FRESH_ROLE = 'round12_independent_confirmation'
FIXED_ROLE = 'round12_fixed500_comparison'
SUFFIX = '\nAnswer the question using a single word or phrase.'
M1_REVISION = '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
RUNTIME_FILES = (
    'round12_policy_runner_v1.py', 'round12_policy_contract_v1.py',
    'round12_action_runner_v2.py', 'round12_action_runner_v3.py',
    'round12_action_probe_v1.py', 'round12_multiimage_v1.py', 'round12_inputs_v1.py',
    'round12_confirmation_admission_v1.py', 'round12_confirmation_data_v1.py',
    'round12_confirmation_native_bindings_v1.py', 'round10_data_builder_v1.py',
    'round12_fixed500_inputs_v1.py',
    'round10_inputs_v1.py', 'm1_protocol.py', 'cache_bridge.py', 'attention_bridge.py',
    'ownership.py', 'evaluate.py', 'model_registry.py', 'dataset_registry.py',
    'round9_runner_v1.py', 'round9_inputs_v1.py',
    'r7_legacy_bridge_v1.py', 'r7_legacy_bridge_v2.py',
)
FORBIDDEN = {
    'refs', 'references', 'reference_answers', 'reference_path', 'answers', 'labels',
    'scores', 'old_logits_path', 'old_native_bindings_path', 'prediction_cache_path',
}


require, sha = contract.require, contract.sha


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def runtime_source_manifest():
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def validate_config(config):
    require(config.get('schema') == SCHEMA, 'explicit policy-runner schema required')
    require(not (set(config) & FORBIDDEN), 'reference/old-output fields forbidden')
    require('action_mode' not in config and 'candidate' not in config,
            'policy action must come only from its frozen asset, not a forced-action/static leftover')
    require(config.get('model_id') == 'M1' and config.get('model_revision') == M1_REVISION,
            'full frozen M1 revision required')
    require(config.get('seed') == 0 and type(config.get('seed')) is int and
            config.get('max_new_tokens') == 64 and config.get('question_suffix') == SUFFIX,
            'native greedy seed0/cap64/suffix required')
    require(config.get('preprocessing') == {'max_patches': 8}, 'original cap8 required')
    require(config.get('position_owner_layout') == 'single_shared_visual_position_map',
            'single actual visual position owner required')
    require(config.get('method', {}).get('family') == 'trajkv_round12_policy_research' and
            config['method'].get('information_condition') ==
            'query_unknown_prefix_then_query_conditional_action', 'explicit policy information condition required')
    require(config.get('decision_mode') in ('margin_threshold', 'random_manifest'),
            'explicit fixed threshold or random-manifest decision mode required')
    require(config.get('input_role') in (DEV_ROLE, FRESH_ROLE, FIXED_ROLE), 'admitted data role required')
    require(config.get('evaluation_contract', {}).get('version') == 'matrix_v3_metrics_v1',
            'frozen metric contract required')
    for field in ('inputs', 'input_manifest_path', 'policy_asset_path'):
        require(isinstance(config.get(field), str) and config[field], 'asset/input path required: ' + field)
    for field, digest in (('inputs', 'inputs_sha256'), ('input_manifest_path', 'input_manifest_sha256'),
                          ('policy_asset_path', 'policy_asset_sha256')):
        require(contract.valid_sha(config.get(digest)) and sha(config[field]) == config[digest],
                'asset/input hash differs: ' + field)
    require(config.get('expected_source_hashes') == runtime_source_manifest(),
            'complete actual policy runtime graph differs')
    return config


def admit_inputs(config):
    """No model imports and no references/old output files are opened."""
    if config['input_role'] == DEV_ROLE:
        from round12_inputs_v1 import load_round12_inputs
        require(config.get('cohort_kind') == 'development', 'development role/cohort mismatch')
        all_rows = load_round12_inputs(config['inputs'], config['input_manifest_path'],
                                       config['inputs_sha256'])
        manifest = json.loads(Path(config['input_manifest_path']).read_text(encoding='utf-8-sig'))
        native = manifest['native_bindings']
        identity = dict(role=DEV_ROLE, phase_is_confirmatory=False, old_predictions_admitted=False,
                        native_bindings=native, decision=None)
    elif config['input_role'] == FRESH_ROLE:
        import round12_confirmation_admission_v1 as fresh
        identity = fresh.validate_config_admission(config)
        fresh.validate_execution_selection(config, identity)
        all_rows = fresh.load_inputs(config['inputs'], config['input_manifest_path'],
                                     config['inputs_sha256'], FRESH_ROLE)
        bundle = json.loads(Path(config['native_bindings_path']).read_text(encoding='utf-8-sig'))
        native = bundle['bindings']
        identity['phase_is_confirmatory'] = True
    else:
        import round12_fixed500_inputs_v1 as fixed
        identity = fixed.validate_config_admission(config)
        fixed.validate_execution_selection(config, identity)
        all_rows = fixed.load_inputs(config['inputs'], config['input_manifest_path'],
                                     config['inputs_sha256'], FIXED_ROLE)
        native = identity['native_bindings']
        identity.update(phase_is_confirmatory=False, previously_observed_fixed500=True,
                        independent_new_source_claimed=False)
    require(len(all_rows) == config['input_count'], 'full input count differs')
    start, count = config.get('start'), config.get('count')
    require(type(start) is int and type(count) is int and start >= 0 and count > 0 and
            start + count <= len(all_rows), 'valid explicit source slice required')
    chosen = all_rows[start:start + count]
    require(len({r['dataset'] for r in chosen}) == 1 and len({r['id'] for r in chosen}) == len(chosen),
            'one domain and unique source IDs per run required')
    require(all(r['id'] in native for r in all_rows), 'native CPU/input binding missing')
    identity.update(rows=chosen, all_rows=all_rows, native_bindings=native,
                    expected_ids=[r['id'] for r in chosen])
    return identity


def admitted_assets(config, identity):
    asset, evidence = contract.load_policy_asset(config['policy_asset_path'],
                                                  config['policy_asset_sha256'])
    random_asset = random_evidence = None
    if config['decision_mode'] == 'random_manifest':
        path, digest = config.get('random_decisions_path'), config.get('random_decisions_sha256')
        require(isinstance(path, str) and contract.valid_sha(digest), 'explicit random manifest path/SHA required')
        spec, spec_evidence = contract.load_random_allocator_spec(config.get('random_allocator_spec_path'),
            config.get('random_allocator_spec_sha256'), config['policy_asset_sha256'])
        random_asset, random_evidence = contract.load_random_manifest(path, digest,
            policy_asset_sha256=config['policy_asset_sha256'], input_sha256=config['inputs_sha256'],
            ordered_ids=[r['id'] for r in identity['all_rows']], metadata_by_id=identity['native_bindings'],
            allocator_spec=spec, allocator_spec_sha256=config['random_allocator_spec_sha256'],
            native_bindings_sha256=(identity.get('native_bindings_sha256') or
                config.get('native_bindings_sha256') or config['input_manifest_sha256']))
        random_evidence['allocator_spec'] = spec_evidence
        random_evidence['total_serialized_bytes'] = random_evidence['serialized_bytes'] + spec_evidence['serialized_bytes']
    else:
        require(not config.get('random_decisions_path') and not config.get('random_decisions_sha256'),
                'threshold policy cannot silently read random decisions')
        require(not config.get('random_allocator_spec_path') and not config.get('random_allocator_spec_sha256'),
                'threshold policy cannot silently read a random allocator')
    if config['input_role'] != DEV_ROLE:
        selection = identity['decision'] if config['input_role'] == FRESH_ROLE else identity['selection_receipt']
        locks = selection['method_asset_locks']
        require(any(item.get('sha256') == config['policy_asset_sha256'] and
                    Path(item['path']).resolve() == Path(config['policy_asset_path']).resolve()
                    for item in locks), 'policy asset absent from frozen pre-evaluation method locks')
        if random_evidence:
            require(any(item.get('sha256') == config['random_allocator_spec_sha256'] and
                        Path(item['path']).resolve() == Path(config['random_allocator_spec_path']).resolve()
                        for item in locks), 'random allocator spec absent from pre-data method locks')
    return asset, evidence, random_asset, random_evidence


def validate_only(config):
    validate_config(config)
    identity = admit_inputs(config)
    asset, evidence, random_asset, random_evidence = admitted_assets(config, identity)
    return dict(status='PASS_POLICY_CONFIG_AND_ASSET_ADMISSION_ONLY', schema=SCHEMA,
        expected_QA=len(identity['rows']), role=config['input_role'],
        policy_asset=evidence, random_manifest=random_evidence,
        action=asset['action'], feature=asset['feature'],
        model_loaded=False, model_forward_calls=0, GPU_calls=0,
        threshold_action_or_confirmation_sources_selected_by_this_module=False)


def _template_sha(processor):
    value = processor.chat_template
    return hashlib.sha256(value.encode()).hexdigest() if isinstance(value, str) else contract.canonical_sha(value)


def _verify_initial_input(action, processor, image, batch, row, config, identity):
    if config['input_role'] == DEV_ROLE:
        return action.verify_low_input_binding(processor, image, batch, row, config)
    import torch
    if config['input_role'] == FRESH_ROLE:
        import round12_confirmation_admission_v1 as admission
    else:
        import round12_fixed500_inputs_v1 as admission
    rendered = row['question'] + SUFFIX
    record = dict(id=row['id'], original_image_sha256=row['original_image_sha256'],
        raw_question_sha256=row['question_sha256'], rendered_question_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        chat_template_hash=_template_sha(processor), preprocessing={'max_patches': 8},
        model_commit=M1_REVISION, dtype='float16', **action.input_evidence(batch))
    result = admission.validate_native_record(record, identity['native_bindings'])
    require(result['matched'], 'actual low input differs from admitted native metadata: ' + str(result['mismatches']))
    native = identity['native_bindings'][row['id']]
    pixels = batch.inputs['pixel_values']
    if 'pixel_values_sha256' in native:
        require(list(pixels.shape) == native['pixel_values_shape'] and
                action.tensor_sha(pixels) == native['pixel_values_sha256'], 'native image pixels differ')
    alternate_text = 'Describe the objects.' if row['question'] == 'What is visible in the image?' else 'What is visible in the image?'
    alternate = action.prepare_m1_inputs(processor, image, alternate_text + SUFFIX,
                                         image_token_id=151667, max_patches=8)
    require(alternate.boundary == batch.boundary and
            torch.equal(alternate.ids[:, :batch.boundary], batch.ids[:, :batch.boundary]) and
            torch.equal(alternate.visual_positions, batch.visual_positions) and
            torch.equal(alternate.positions[:, :batch.boundary], batch.positions[:, :batch.boundary]),
            'fresh low prefix depends on future question')
    for key in set(batch.inputs) | set(alternate.inputs):
        if key in ('input_ids', 'attention_mask', 'position_ids', 'token_type_ids'):
            continue
        require(key in batch.inputs and key in alternate.inputs, 'alternate native image keys differ')
        left, right = batch.inputs[key], alternate.inputs[key]
        require(torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right,
                'alternate native visual data differs')
    result.update(alternate_question_prefix_ids_positions_pixels_bit_exact=True,
                  original_image_pixels_match_CPU_binding=('pixel_values_sha256' in native),
                  old_prediction_reuse=False)
    return result


def _visual_signature(cache):
    """Pointer/version witness only; numerical decoder acceptance is separate."""
    result = []
    for name, tensor in sorted(cache.named_tensors().items()):
        if '.visual.' not in name and name != 'original_visual_positions':
            continue
        result.append((name, str(tensor.device), tensor.untyped_storage().data_ptr(),
            tensor.untyped_storage().nbytes(), list(tensor.shape), list(tensor.stride()),
            str(tensor.dtype), int(tensor._version)))
    return result


def _copy_owner_fields(record, owner):
    for field in ('stored_visual_payload_bytes', 'coder_metadata_bytes', 'positions_bytes',
                  'encoder_constants_bytes', 'text_positions_bytes', 'text_kv_bytes'):
        record[field] = owner[field]
    complete = int(owner['complete_visual_owner_bytes'])
    n = record['actual_visual_token_count']
    denominator = 28 * 2 * 4 * n * 128 * 2
    record.update(stored_visual_KV_bytes=complete, complete_visual_owner_bytes=complete,
        full_kv_byte_denominator=denominator, actual_effective_visual_bits=16 * complete / denominator,
        per_request_strict_budget={'1.3': 160 * complete <= 13 * denominator,
                                  '1.8': 80 * complete <= 9 * denominator})


def run_policy_row(model, processor, row, config, output_dir, asset, policy_evidence,
                   random_asset, random_evidence, identity):
    """One real sequential deployment, never a splice of other-run answers."""
    import torch
    from PIL import Image
    import round12_action_runner_v2 as action
    low, high = action.variants_from_config(config)
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    record = {**row, 'model_id': 'M1', 'model_commit': M1_REVISION,
        'model_arch': 'InternVLForConditionalGeneration', 'dtype': 'float16',
        'method': config['method']['id'], 'method_contract': config['method'],
        'preprocessing': {'max_patches': 8}, 'status': 'RUNNING',
        'official_score': None, 'score_status': 'NOT_READ_OR_COMPUTED_BY_RUNNER',
        'output_token_ids': None, 'raw_prediction': None,
        'forced_development_counterfactual_action': False,
        'online_selector_deployment_claimed': config['decision_mode'] == 'margin_threshold',
        'frozen_batch_random_control': config['decision_mode'] == 'random_manifest',
        'policy_asset_sha256': policy_evidence['sha256'],
        'random_decisions_sha256': random_evidence['sha256'] if random_evidence else None,
        'current_reference_answers_or_other_branch_outputs_read': False,
        'prediction_splicing_used': False,
        'object_contract': 'single-query onsite decision: retain low for a0 or replace for P/Z; no reversible dual-bank claim',
        'controller_assets_are_global_per_run_not_per_image': True,
        'controller_GPU_tensor_bytes': 0,
        'host_cached_upgrade_bank_bytes_read': 0,
        'physical_host_to_device_bytes_measured': False,
        'model_initialization_cost_scope': 'run manifest; no undisclosed request amortization'}
    cache = batch = image = low_observer = final_observer = None
    final_calls = []
    try:
        require(sha(row['image_path']) == row['original_image_sha256'], 'original source image hash differs')
        with Image.open(row['image_path']) as opened:
            image = opened.convert('RGB')
        require(list(image.size) == [row['width'], row['height']], 'original image dimensions differ')
        rendered = row['question'] + SUFFIX
        batch = action.prepare_m1_inputs(processor, image, rendered, image_token_id=151667, max_patches=8)
        original = action.input_evidence(batch)
        record.update(original_dimensions=list(image.size), original_low_input=original,
            raw_question=row['question'], rendered_question=rendered, question_suffix=SUFFIX,
            raw_question_sha256=row['question_sha256'],
            rendered_question_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            chat_template_hash=_template_sha(processor))
        record['initial_native_input_binding'] = _verify_initial_input(
            action, processor, image, batch, row, config, identity)
        low_observer = action.GenerationObserver(model, begin, batch)
        with action.observe_generation(model, low_observer):
            cache, low_meta = action.pack_native_prefix(model, batch, low, config, low_observer)
            record['low_stage'] = low_meta
            initial_owner = action.cache_storage(cache, cache.visual_positions)
            record['low_owner_before_probe'] = initial_owner
            signature = _visual_signature(cache)
            low_cache_identity = id(cache)
            action.release_pixel_inputs(batch)
            torch.cuda.synchronize()
            record['T_init_wall_ms'] = (time.perf_counter() - begin) * 1000
            record['T_init_scope'] = 'source IO, native input/binding checks, original low prefill and packed encoding'
            low_observer.phase = 'probe'
            t_probe = time.perf_counter()
            probe = action.run_low_probe(model, processor, batch, cache, row['question'])
            torch.cuda.synchronize()
            record['T_probe_wall_ms'] = (time.perf_counter() - t_probe) * 1000
            require(low_observer.stream_ids == probe['generated_probe_tokens'], 'actual low probe stream differs')
        require(cache.get_seq_length() == batch.boundary and _visual_signature(cache) == signature,
                'probe did not restore the same immutable low visual state')
        record.update(probe=probe, initial_question_unknown_prefix_verified=True,
            low_probe_text_KV_rolled_back_to_prefix=True,
            low_visual_pointer_version_signature_before_probe=contract.canonical_sha(signature),
            low_visual_pointer_version_signature_after_probe=contract.canonical_sha(_visual_signature(cache)),
            signature_scope='same actual tensor storages/shapes/versions; not a substitute for numerical acceptance')
        t_decide = time.perf_counter()
        decision = (contract.evaluate_margin_decision(asset, probe) if config['decision_mode'] == 'margin_threshold'
                    else contract.evaluate_random_decision(asset, random_asset, row['id']))
        record['T_decision_wall_ms'] = (time.perf_counter() - t_decide) * 1000
        record['policy_decision'] = decision
        selected = decision['selected_action']
        record['action_mode'] = selected
        if selected == 'a0':
            require(id(cache) == low_cache_identity and cache.get_seq_length() == batch.boundary,
                    'a0 must reuse the same rolled-back low cache')
            record.update(low_meta)
            record.update(original)
            record.update(T_action_wall_ms=0.0, action_prefill_encode_wall_ms=0.0,
                a0_same_low_cache_object=True, a0_complete_original_question_replayed=True,
                a0_extra_native_prefill_calls=0, a0_extra_visual_encoding_calls=0,
                action_implementation='retain_actual_low_cache_after_probe_rollback_then_recompute_full_original_suffix',
                action_prefix_information_condition='original_query_unknown_low_prefix',
                question_unknown_prefix_verified=True)
            record['owner_before_suffix'] = action.cache_storage(cache, cache.visual_positions)
            _copy_owner_fields(record, record['owner_before_suffix'])
            final_observer = action.GenerationObserver(model, begin, batch)
            t_answer = time.perf_counter()
            with action.observe_generation(model, final_observer):
                action._answer(model, processor, batch, cache, row, config,
                               output_dir, record, final_observer)
            torch.cuda.synchronize()
            record['T_answer_wall_ms'] = (time.perf_counter() - t_answer) * 1000
            require(id(cache) == low_cache_identity and _visual_signature(cache) == signature,
                    'a0 replaced or modified the original low visual owner')
        else:
            require(selected in ('P', 'Z') and selected == asset['action'], 'undeclared policy action selected')
            t_action = time.perf_counter()
            # Release low instead of retaining two banks under a one-bank claim.
            cache = batch = None
            gc.collect()
            torch.cuda.empty_cache()
            if selected == 'P':
                batch = action.prepare_m1_inputs(processor, image, rendered, image_token_id=151667, max_patches=8)
                require(action.input_evidence(batch) == original, 'P changes the original input contract')
                variant = high
                record.update(action_implementation='onsite_native_prefill_encode_high_replace_low',
                    action_prefix_information_condition='same_query_unknown_image_prefix_rebuilt',
                    question_unknown_prefix_verified=True)
            else:
                roi = action.select_roi(probe['spatial_weights'], image.size,
                                        original['actual_visual_token_count'] // 256)
                record['roi'] = roi
                crop = image.crop(tuple(roi['bbox_xyxy']))
                batch, multi = action.prepare_multiimage_inputs(processor, image, crop, rendered,
                                                                image_token_id=151667, max_patches=8)
                crop = None
                variant = low
                record.update(native_two_view_protocol=multi,
                    action_implementation='complete_native_two_view_prefix_all_visual_packed_low',
                    action_prefix_information_condition='query_conditioned_ROI_then_native_complete_two_view_prefix',
                    question_unknown_prefix_verified=False)
            record.update(action.input_evidence(batch))
            record['final_action_input_is_original'] = selected == 'P'
            final_observer = action.GenerationObserver(model, begin, batch)
            with action.observe_generation(model, final_observer):
                cache, action_meta = action.pack_native_prefix(model, batch, variant, config, final_observer)
                record.update(action_meta)
                torch.cuda.synchronize()
                record['T_action_wall_ms'] = (time.perf_counter() - t_action) * 1000
                record['action_prefill_encode_wall_ms'] = record['T_action_wall_ms']
                record['owner_before_suffix'] = action.cache_storage(cache, cache.visual_positions)
                _copy_owner_fields(record, record['owner_before_suffix'])
                t_answer = time.perf_counter()
                action._answer(model, processor, batch, cache, row, config,
                               output_dir, record, final_observer)
                torch.cuda.synchronize()
                record['T_answer_wall_ms'] = (time.perf_counter() - t_answer) * 1000
        require(record.get('terminal_generation_completed_before_audit') is True and
                record.get('actual_terminal_generated_token_count', 0) > 0,
                'one actual final terminal answer required')
        final_calls = final_observer.calls
        record['outer_forward_calls'] = [dict(stage='low_probe', **x) for x in low_observer.calls] + [
            dict(stage='selected_path', **x) for x in final_calls]
        record['ttft_ms'] = (final_observer.token_times[0] - begin) * 1000
        record['inter_token_ms'] = [(b - a) * 1000 for a, b in zip(
            final_observer.token_times, final_observer.token_times[1:])]
        record['question_prefill_ms'] = next((x['wall_ms'] for x in final_calls if x['phase'] == 'generation'), None)
        record['reference_scope'] = 'actual_low_probe_then_frozen_decision_then_real_selected_path_full_suffix_generation'
        record['status'] = 'PASS'
    except Exception as exc:
        record.update(status='OOM_AT_DECLARED_INPUT' if 'out of memory' in str(exc).lower() else 'FAILED_RUN',
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            output_token_ids=None, raw_prediction=None)
    finally:
        record.update(wall_ms=(time.perf_counter() - begin) * 1000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), end_allocated_bytes=torch.cuda.memory_allocated(),
            timing_scope='actual initial low setup + native cap3 probe + frozen decision + chosen path + final answer; instrumented prototype')
        record['T_total_wall_ms'] = record['wall_ms']
        cache = batch = image = low_observer = final_observer = None
        gc.collect()
        torch.cuda.empty_cache()
    return record


def _admission_record(identity):
    return {key: value for key, value in identity.items()
            if key not in ('rows', 'all_rows', 'native_bindings')}


def _check_binding_source_identity(identity):
    for role, digest in identity.get('native_binding_source_files', {}).items():
        path = Path(role)
        if not path.is_absolute():
            path = ROOT / path
        require(path.is_file() and sha(path) == digest,
                'fresh CPU binding source differs from current runtime: ' + role)


def config_template():
    """Unconfigured fixed500 template; does not select a policy or data file."""
    return dict(schema=SCHEMA, model_id='M1', model_revision=M1_REVISION,
        model_path=None, model_source_lock=None, seed=0, max_new_tokens=64,
        question_suffix=SUFFIX, preprocessing={'max_patches': 8},
        method={'id': None, 'family': 'trajkv_round12_policy_research',
                'information_condition': 'query_unknown_prefix_then_query_conditional_action'},
        decision_mode=None, policy_asset_path=None, policy_asset_sha256=None,
        variant_source=None, variant=None, high_variant=None, expected_asset_sha256=None,
        legacy_a2=None, legacy_experiments=None, legacy_scripts=None, kivi_source=None,
        input_role=FIXED_ROLE, cohort_kind='fixed500_comparison',
        inputs=None, inputs_sha256=None, input_count=500, start=0, count=None,
        input_manifest_path=None, input_manifest_sha256=None,
        selection_receipt_path=None, selection_receipt_sha256=None, selected_method_name=None,
        position_owner_layout='single_shared_visual_position_map',
        evaluation_contract={'version': 'matrix_v3_metrics_v1', 'primary_metric_id': None},
        expected_source_hashes=None,
        status='UNCONFIGURED_NO_POLICY_THRESHOLD_ACTION_OR_FIXED500_RUN_SELECTED')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', '--check-config', action='store_true')
    parser.add_argument('--print-template', action='store_true')
    args = parser.parse_args()
    if args.print_template:
        require(args.config is None and args.output is None and not args.validate_only,
                'template printing cannot also validate/run a configured policy')
        print(json.dumps(config_template(), ensure_ascii=False, indent=2))
        return 0
    require(args.config is not None, 'configured policy path required')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    validate_config(config)
    identity = admit_inputs(config)
    asset, policy_evidence, random_asset, random_evidence = admitted_assets(config, identity)
    _check_binding_source_identity(identity)
    if args.validate_only:
        report = dict(status='PASS_POLICY_CONFIG_AND_ASSET_ADMISSION_ONLY', schema=SCHEMA,
            expected_QA=len(identity['rows']), role=config['input_role'],
            policy_asset=policy_evidence, random_manifest=random_evidence,
            action=asset['action'], feature=asset['feature'], model_loaded=False,
            model_forward_calls=0, GPU_calls=0,
            threshold_action_or_confirmation_sources_selected_by_this_module=False)
        if args.output is not None:
            write_new(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    require(args.output is not None, 'actual deployment requires a new output directory')
    import torch
    import round12_action_runner_v2 as action
    action.variants_from_config(config)
    rows = identity['rows']
    dataset = rows[0]['dataset']
    require(config['evaluation_contract']['primary_metric_id'] == action.PRIMARY_METRIC_IDS[dataset],
            'native primary metric identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = runtime_source_manifest()
    controller_serialized = policy_evidence['serialized_bytes'] + (
        random_evidence['total_serialized_bytes'] if random_evidence else 0)
    manifest = dict(schema='baseline_matrix.raw_inference.v1', runner_schema=SCHEMA,
        config=config, config_sha256=sha(args.config), inputs_sha256=config['inputs_sha256'],
        expected_ids=[r['id'] for r in rows], expected_QA=len(rows),
        references_read=False, prediction_cache_used=False,
        source_hashes=source_hashes, runtime_source_root=str(ROOT), runtime_executable=sys.executable,
        started_unix=time.time(), phase_is_confirmatory=config['input_role'] == FRESH_ROLE,
        input_admission=_admission_record(identity),
        previously_observed_fixed500=config['input_role'] == FIXED_ROLE,
        independent_new_source_claimed=config['input_role'] == FRESH_ROLE,
        policy_asset=policy_evidence, random_manifest=random_evidence,
        frozen_policy_numerical_asset=asset,
        global_controller_host_artifact_serialized_bytes=controller_serialized,
        controller_GPU_tensor_bytes=0,
        controller_storage_scope='one copy per run/deployment; not silently added to or removed from each visual-owner payload',
        actual_strategy_execution=True, forced_development_counterfactual_actions=False,
        online_selector_deployment_claimed=config['decision_mode'] == 'margin_threshold',
        frozen_batch_random_control=config['decision_mode'] == 'random_manifest',
        source_numerical_action_driver='round12_action_runner_v2.py',
        coverage_contract='expected IDs, recorded count and all status counts closed as in frozen action v3')
    record_path = args.output / 'records.jsonl'
    records = []
    error = None
    written = False

    def append(record):
        write_new(args.output / 'records' / f'{len(records):06d}.json', record)
        with record_path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        records.append(record)

    try:
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6),
                'actual RTX3090 CUDA SM86 required; no CPU substitution')
        torch.cuda.set_device(0)
        spec = action.get_model_spec('M1')
        model_identity = action.model_identity(config, spec)
        t_model = time.perf_counter()
        processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
            device_map={'': 0}, local_files_only=True, attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        torch.cuda.synchronize()
        require(all(p.dtype == torch.float16 for p in model.parameters()) and
                not getattr(model, 'is_quantized', False), 'complete original FP16 weights required')
        require(len(action.language_attention_modules(model)[1]) == 28, 'whole M1 geometry differs')
        manifest.update(model_spec=spec, model_identity=model_identity,
            model_residency=action.model_storage(model), model_load_wall_ms=(time.perf_counter() - t_model) * 1000,
            model_parameter_count=sum(p.numel() for p in model.parameters()),
            torch_version=torch.__version__, transformers_version=transformers.__version__,
            CUDA_version=torch.version.cuda, hardware=torch.cuda.get_device_name(),
            python_version=platform.python_version(), generation_native=model.generation_config.to_dict(),
            runtime_library_source_sha256={str(Path(inspect.getfile(cls))): sha(inspect.getfile(cls))
                for cls in (type(model), type(processor), type(processor.image_processor), type(processor.tokenizer))})
        write_new(args.output / 'manifest.json', manifest)
        written = True
        for row in rows:
            record = run_policy_row(model, processor, row, config, args.output, asset,
                policy_evidence, random_asset, random_evidence, identity)
            append(record)
            print(json.dumps(dict(id=row['id'], status=record['status'],
                selected_action=record.get('action_mode'), completed=len(records), expected=len(rows))), flush=True)
        require(runtime_source_manifest() == source_hashes, 'policy runtime source graph changed')
        require(sha(config['inputs']) == config['inputs_sha256'] and
                sha(config['input_manifest_path']) == config['input_manifest_sha256'], 'no-answer input bytes changed')
        require(sha(config['policy_asset_path']) == config['policy_asset_sha256'], 'frozen policy asset changed')
        if random_evidence:
            require(sha(config['random_decisions_path']) == config['random_decisions_sha256'], 'random manifest changed')
            require(sha(config['random_allocator_spec_path']) == config['random_allocator_spec_sha256'],
                    'pre-data random allocator spec changed')
        if config['input_role'] == FRESH_ROLE:
            require(sha(config['native_bindings_path']) == config['native_bindings_sha256'] and
                    sha(config['frozen_decision_path']) == config['frozen_decision_sha256'],
                    'fresh input/selection assets changed')
            _check_binding_source_identity(identity)
        elif config['input_role'] == FIXED_ROLE:
            require(sha(config['selection_receipt_path']) == config['selection_receipt_sha256'],
                    'frozen fixed500 comparison selection receipt changed')
        require(sha(args.config) == manifest['config_sha256'], 'policy run config changed')
        require(all(sha(path) == digest for path, digest in manifest['runtime_library_source_sha256'].items()),
                'native model/processor/tokenizer source changed')
    except Exception as exc:
        error = dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    if not written:
        manifest['initialization_error'] = error
        write_new(args.output / 'manifest.json', manifest)
    completed_ids = {r['id'] for r in records}
    for row in rows:
        if row['id'] not in completed_ids:
            append({**row, 'model_id': 'M1', 'method': config['method']['id'],
                'status': 'FAILED_RUN', 'output_token_ids': None, 'raw_prediction': None,
                'official_score': None, 'error': error})
    counts = {}
    for record in records:
        counts[record['status']] = counts.get(record['status'], 0) + 1
    complete = dict(status='PASS_RAW_COMPLETE' if error is None and counts.get('PASS') == len(rows)
                    else 'COMPLETE_WITH_FAILURES', expected_QA=len(rows), recorded_QA=len(records), counts=counts,
        coverage_complete=(len(records) == len(rows) and sum(counts.values()) == len(rows) and
                           [r['id'] for r in records] == [r['id'] for r in rows]),
        failed_QA_must_remain_in_primary_denominator=True,
        all_failed_sources_remain_in_primary_denominator=True,
        quality_scored=False, load_error=error, phase_is_confirmatory=config['input_role'] == FRESH_ROLE,
        manifest_sha256=sha(args.output / 'manifest.json'), records_jsonl_sha256=sha(record_path),
        completed_unix=time.time(), actual_strategy_execution=True,
        actual_terminal_generation_attempts=sum(bool(r.get('actual_terminal_generation_invoked')) for r in records),
        actual_new_terminal_answers=sum(bool(r.get('terminal_generation_completed_before_audit')) for r in records),
        actual_successful_terminal_answers=counts.get('PASS', 0),
        actual_probe_tokens=sum(r.get('probe', {}).get('actual_probe_token_count', 0) for r in records),
        upgraded_sources=sum(bool(r.get('policy_decision', {}).get('upgrade')) for r in records),
        a0_sources=sum(r.get('action_mode') == 'a0' for r in records),
        observed_upgrade_rate=sum(bool(r.get('policy_decision', {}).get('upgrade')) for r in records) / len(rows),
        actual_resources_must_be_compared_not_inferred_from_upgrade_rate=True,
        frozen_score_py_compatible_raw_schema=True)
    write_new(args.output / 'complete.json', complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    return 0 if complete['status'] == 'PASS_RAW_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
