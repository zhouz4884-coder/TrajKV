"""Strict first-answer total-KV view/precision exchange over frozen R12 v2.

New research driver only. Active action/static/probe files are unmodified.
The budget is actual original-low cache_storage.total_bytes immediately after
its complete question suffix, before its first answer token. Only CPU view
enumeration precedes commit; old low GPU state is released before a candidate
prefill. An over-budget candidate aborts before sampling, then pays to rebuild
low for a0. Allocator peak/workspaces are reported, not bounded by this KV cap.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import time
import traceback
import weakref

import torch
from PIL import Image

import round12_action_runner_v2 as base
from round12_budget_multiimage_v1 import (plan_budget_views_cpu, move_selected_to_device,
    predicted_a_visual_bytes, predicted_first_suffix_total)
from round12_static_bank_v1 import build_round12_owners

SCHEMA = 'trajkv.round12_budget_action.raw.v2'
MODES = ('Z-budget-K2', 'Zrandom-budget-K2')
RUNTIME_FILES = tuple(dict.fromkeys((*base.RUNTIME_FILES,
    'round12_action_runner_v3.py', 'round12_budget_action_runner_v2.py',
    'round12_budget_multiimage_v1.py', 'round12_static_bank_v1.py',
    'round12_static_codec_v1.py', 'round12_static_attention_v1.py')))
_base_validate = base.validate_config
_base_write = base.write_new


class FirstSuffixBudgetExceeded(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError('ROUND12_BUDGET_ACTION: ' + message)


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('action_mode') in MODES,
            'new explicit budget-action schema/mode required')
    require(config.get('method', {}).get('family') == 'trajkv_round12_budget_action_research',
            'explicit budget-action method family required')
    require(config.get('budget_scope') == 'first_suffix_total_persistent_KV',
            'budget must include complete visual/text/positions/other cache storages')
    candidate = config.get('a_candidate')
    require(isinstance(candidate, dict) and set(candidate) == {'mode', 'key_book_path', 'key_book_sha256'}
            and candidate['mode'] == 'A' and base.sha(candidate['key_book_path']) == candidate['key_book_sha256'],
            'frozen A K2 book asset required')
    shadow = deepcopy(config)
    shadow['schema'] = base.SCHEMA
    shadow['action_mode'] = 'Z-random' if config['action_mode'] == 'Zrandom-budget-K2' else 'Z'
    shadow['method']['family'] = 'trajkv_round12_action_research'
    _base_validate(shadow)
    return config


def variants_from_config(config):
    return base.frozen_variant(config), None


def _cache_weak_witness(cache):
    tensors = cache.named_tensors()
    return dict(cache_ref=weakref.ref(cache),
        owner_refs=[weakref.ref(layer.visual_owner) for layer in cache.layers],
        tensor_refs=[weakref.ref(t) for t in tensors.values()],
        unique_storage_count=len({(str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                                  for t in tensors.values()}),
        tensor_names=list(tensors))


def _assert_released(witness):
    gc.collect()
    torch.cuda.empty_cache()
    live_owners = sum(ref() is not None for ref in witness['owner_refs'])
    live_tensors = sum(ref() is not None for ref in witness['tensor_refs'])
    require(witness['cache_ref']() is None and live_owners == 0 and live_tensors == 0,
            f'old cache retained across candidate commit: owners={live_owners}, tensors={live_tensors}')
    return dict(old_cache_object_released=True, old_owner_objects_released=True,
        old_KV_and_owner_tensor_objects_released=True,
        formerly_named_tensors=len(witness['tensor_names']),
        formerly_unique_storage_count=witness['unique_storage_count'],
        measured_before_any_candidate_prefill_or_CUDA_input_transfer=True,
        no_pointer_nonreuse_assumption=True,
        scope='weak references prove no retained old cache/owners/tensors; allocator reservation and model buffers separately reported')


def _pack_A(model, batch, config, observer):
    observer.phase = 'prefix'
    begin = time.perf_counter()
    out = base.m1_prefix_forward(model, batch)
    torch.cuda.synchronize()
    native = out.past_key_values
    out = None
    prefill_ms = (time.perf_counter() - begin) * 1000
    require(native.get_seq_length() == batch.boundary and len(native.layers) == 28, 'native action prefix differs')
    vp = batch.visual_positions
    native_report = base.cache_storage(native, vp)
    refs = [weakref.ref(t) for layer in native.layers for t in (layer.keys, layer.values)]
    old = {(str(t.device), t.untyped_storage().data_ptr())
           for layer in native.layers for t in (layer.keys, layer.values)}
    begin = time.perf_counter()
    owners, metadata = build_round12_owners(native, vp, base.frozen_variant(config),
        candidate=config['a_candidate'], legacy_a2=config['legacy_a2'],
        legacy_experiments=config['legacy_experiments'], legacy_scripts=config['legacy_scripts'],
        kivi_source=config['kivi_source'], expected_asset_sha256=config['expected_asset_sha256'])
    cache = base.HybridVisualCache(native, vp, owners, consume_native=True)
    owners = native = None
    sharing = base.share_original_visual_position_map(cache)
    gc.collect()
    torch.cuda.synchronize()
    require(not any(ref() is not None for ref in refs), 'candidate retains original full FP16 prefix')
    base.assert_no_full_prefix_alias(cache, old)
    visual = base.cache_storage(cache, cache.visual_positions)['complete_visual_owner_bytes']
    require(visual == predicted_a_visual_bytes(int(vp.numel())), 'actual A owner differs from CPU predictor')
    require(metadata['selected_stage_count'] == 43 and metadata['round12_static']['mode'] == 'A',
            'candidate is not original concrete43 K2/V1 A')
    return cache, dict(prefill_ms=prefill_ms, encode_ms=(time.perf_counter()-begin)*1000,
        native_prefix_owner=native_report, position_owner_layout=sharing, r7_bridge=metadata,
        original_full_prefix_released=True, original_FP16_visual_alias_retained=False)


def _owner_fields(record, cache):
    own = base.cache_storage(cache, cache.visual_positions)
    record['owner_before_suffix'] = own
    for field in ('stored_visual_payload_bytes', 'coder_metadata_bytes', 'positions_bytes',
                  'encoder_constants_bytes', 'text_positions_bytes', 'text_kv_bytes'):
        record[field] = own[field]
    complete = int(own['complete_visual_owner_bytes'])
    n = int(record['actual_visual_token_count'])
    denominator = 28*2*4*n*128*2
    record.update(stored_visual_KV_bytes=complete, complete_visual_owner_bytes=complete,
        full_kv_byte_denominator=denominator, actual_effective_visual_bits=16*complete/denominator,
        per_request_strict_budget={'1.3': 160*complete <= 13*denominator, '1.8': 80*complete <= 9*denominator})


def _guarded_answer(model, processor, batch, cache, row, config, output_dir,
                    record, observer, budget, *, candidate):
    checked = []
    def after_suffix(module, args, output):
        if checked:
            return
        require(cache.get_seq_length() == batch.full_length, 'first answer forward is not exact complete suffix boundary')
        returned_cache = getattr(output, 'past_key_values', None)
        require(returned_cache is None or returned_cache is cache, 'model returned a different candidate cache object')
        report = base.cache_storage(cache, cache.visual_positions)
        measured = int(report['total_bytes'])
        evidence = dict(total_cache_bytes=measured, budget_bytes=budget,
            complete_visual_owner_bytes=int(report['complete_visual_owner_bytes']),
            text_kv_bytes=int(report['text_kv_bytes']), text_positions_bytes=int(report['text_positions_bytes']),
            other_persistent_metadata_bytes=int(report['other_persistent_metadata_bytes']),
            cache_length=int(cache.get_seq_length()), full_length=batch.full_length,
            fits=measured <= budget, before_first_answer_token_sampling=True,
            no_old_low_cache_retained_during_candidate=candidate,
            text_position_sharing_assumed=False)
        if candidate:
            predicted = record['budget_view_CPU_plan']['selected']['predicted_first_suffix_total_bytes']
            evidence['CPU_predicted_total_bytes'] = predicted
            evidence['actual_minus_CPU_predicted_bytes'] = measured - predicted
        evidence['text_position_unique_storages_charged'] = sum(
            item['charged_bytes'] > 0 for name, item in report['tensors'].items()
            if name.endswith('text_positions'))
        checked.append(evidence)
        record['candidate_first_suffix_budget_check' if candidate else 'a0_first_suffix_budget_check'] = evidence
        if measured > budget:
            raise FirstSuffixBudgetExceeded(f'actual first-suffix cache {measured} exceeds original-low {budget}')
        record['first_answer_total_cache_bytes'] = measured
        record['strict_total_cache_budget_satisfied'] = True
    handle = model.register_forward_hook(after_suffix)
    record['actual_terminal_generation_attempt_count'] = record.get('actual_terminal_generation_attempt_count', 0) + 1
    try:
        base._answer(model, processor, batch, cache, row, config, output_dir, record, observer)
    finally:
        handle.remove()
    require(len(checked) == 1 and checked[0]['fits'], 'first-suffix budget guard not executed')


def run_budget_row(model, processor, row, config, output_dir):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    mode = config['action_mode']
    low = base.frozen_variant(config)
    cache = batch = image = low_observer = final_observer = None
    baseline_handle = None
    record = {**row, 'model_id': 'M1', 'model_commit': base.get_model_spec('M1')['revision'],
        'model_arch': 'InternVLForConditionalGeneration', 'dtype': 'float16',
        'method': config['method']['id'], 'method_contract': config['method'],
        'preprocessing': {'max_patches': 8}, 'status': 'RUNNING', 'official_score': None,
        'score_status': 'NOT_READ_OR_COMPUTED_BY_RUNNER', 'action_mode': mode,
        'output_token_ids': None, 'raw_prediction': None,
        'forced_development_counterfactual_action': True, 'online_selector_deployment_claimed': False,
        'persistent_KV_budget_is_not_allocator_peak_budget': True,
        'host_cached_upgrade_bank_bytes_read': 0, 'physical_host_to_device_bytes_measured': False}
    traces = []
    traced_observers = set()
    def add_trace(observer, stage):
        if observer is not None and id(observer) not in traced_observers:
            traces.extend(dict(stage=stage, **x) for x in observer.calls)
            traced_observers.add(id(observer))
    try:
        require(base.sha(row['image_path']) == row['original_image_sha256'], 'original image hash differs')
        with Image.open(row['image_path']) as opened:
            image = opened.convert('RGB')
        rendered = row['question'] + base.SUFFIX
        batch = base.prepare_m1_inputs(processor, image, rendered, image_token_id=151667, max_patches=8)
        original = base.input_evidence(batch)
        record.update(original_dimensions=list(image.size), original_low_input=original,
            raw_question=row['question'], rendered_question=rendered, question_suffix=base.SUFFIX,
            raw_question_sha256=row['question_sha256'], rendered_question_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            chat_template_hash=base.canonical(processor.chat_template))
        record['initial_native_input_binding'] = base.verify_low_input_binding(processor, image, batch, row, config)
        baseline = {}
        low_observer = base.GenerationObserver(model, started, batch)
        with base.observe_generation(model, low_observer):
            cache, low_meta = base.pack_native_prefix(model, batch, low, config, low_observer)
            record['low_stage'] = low_meta
            record['original_low_visual_owner_bytes'] = base.cache_storage(cache, cache.visual_positions)['complete_visual_owner_bytes']
            base.release_pixel_inputs(batch)
            record['T_init_wall_ms'] = (time.perf_counter()-started)*1000
            def capture_original_suffix(module, args, output):
                if baseline:
                    return
                require(cache.get_seq_length() == batch.full_length, 'baseline capture must follow complete original suffix')
                own = base.cache_storage(cache, cache.visual_positions)
                baseline.update(total_bytes=int(own['total_bytes']), owner=own,
                    original_full_length=batch.full_length, original_N=original['actual_visual_token_count'],
                    before_first_answer_token_sampling=True,
                    scope='original low actual complete cache after full suffix; includes visual/text/all positions/other')
            baseline_handle = model.register_forward_hook(capture_original_suffix)
            low_observer.phase = 'probe'
            t_probe = time.perf_counter()
            try:
                probe = base.run_low_probe(model, processor, batch, cache, row['question'])
            finally:
                baseline_handle.remove()
                baseline_handle = None
            torch.cuda.synchronize()
            record['T_probe_wall_ms'] = (time.perf_counter()-t_probe)*1000
            require(low_observer.stream_ids == probe['generated_probe_tokens'], 'native low probe stream differs')
        require(baseline and cache.get_seq_length() == batch.boundary, 'baseline/rollback missing')
        record.update(probe=probe, baseline_actual_first_suffix_cache=baseline,
            original_low_total_KV_budget_bytes=baseline['total_bytes'],
            original_visual_FP16_denominator=28*2*4*original['actual_visual_token_count']*128*2,
            initial_question_unknown_prefix_verified=True)
        add_trace(low_observer, 'initial_low_probe')
        low_observer = None
        random_seed = hashlib.sha256(('TrajKV-R12-random-ROI-v1|'+row['id']).encode()).hexdigest()
        roi = base.select_roi(probe['spatial_weights'], image.size, original['actual_visual_token_count']//256,
            random_seed=random_seed if mode == 'Zrandom-budget-K2' else None)
        record['roi'] = roi
        crop = image.crop(tuple(roi['bbox_xyxy']))
        t_action = time.perf_counter()
        cpu_batch, plan = plan_budget_views_cpu(processor, image, crop, rendered, baseline['total_bytes'])
        crop = None
        record['budget_view_CPU_plan'] = plan
        committed = cpu_batch is not None
        record['candidate_committed'] = committed
        if committed:
            witness = _cache_weak_witness(cache)
            cache = batch = None
            record['commit_old_cache_release'] = _assert_released(witness)
            witness = None
            batch = move_selected_to_device(cpu_batch)
            cpu_batch = None
            record.update(base.input_evidence(batch))
            view_mode = plan['selected']['view_mode']
            has_global = view_mode == 'global_cap1_plus_ROI'
            roi_tiles = plan['selected']['actual_roi_tiles']
            detail_tiles = roi_tiles - int(roi_tiles > 1)
            record.update(action_prefix_information_condition=('query_ROI_and_budget_conditioned_low_global_two_view'
                if has_global else 'query_ROI_and_budget_conditioned_crop_only'),
                question_unknown_prefix_verified=False,
                action_implementation=('CPU_budget_selected_globalcap1_ROI_then_K2_L43'
                if has_global else 'CPU_budget_selected_ROI_only_cap1_then_K2_L43'),
                original_global_context_removed=not has_global,
                original_cap8_pixel_preservation_claimed=False,
                sampling_tradeoff={'original_nonthumbnail_global_tiles': roi['grid']['nonthumbnail_tiles'],
                    'new_global_tiles': int(has_global), 'new_ROI_tiles_including_possible_thumbnail': roi_tiles,
                    'new_ROI_detail_tiles': detail_tiles,
                    'approximate_ROI_sampling_density_vs_original': detail_tiles/(roi['grid']['nonthumbnail_tiles']*roi['area_fraction']),
                    'density_is_not_information_or_quality_guarantee': True})
            final_observer = base.GenerationObserver(model, started, batch)
            exceeded = False
            with base.observe_generation(model, final_observer):
                cache, metadata = _pack_A(model, batch, config, final_observer)
                record.update(metadata)
                _owner_fields(record, cache)
                torch.cuda.synchronize()
                record['T_action_wall_ms'] = (time.perf_counter()-t_action)*1000
                t_answer = time.perf_counter()
                try:
                    _guarded_answer(model, processor, batch, cache, row, config, output_dir,
                        record, final_observer, baseline['total_bytes'], candidate=True)
                except FirstSuffixBudgetExceeded:
                    exceeded = True
                    record['candidate_over_budget_aborted_before_first_token'] = True
                    record['candidate_abort_traceback'] = traceback.format_exc()
                record['candidate_answer_attempt_wall_ms'] = (time.perf_counter()-t_answer)*1000
            add_trace(final_observer, 'budget_candidate')
            if exceeded:
                witness = _cache_weak_witness(cache)
                cache = batch = final_observer = None
                record['aborted_candidate_cache_release'] = _assert_released(witness)
                witness = None
                t_rebuild = time.perf_counter()
                batch = base.prepare_m1_inputs(processor, image, rendered, image_token_id=151667, max_patches=8)
                require(base.input_evidence(batch) == original, 'reconstructed fallback changed original input')
                final_observer = base.GenerationObserver(model, started, batch)
                with base.observe_generation(model, final_observer):
                    cache, metadata = base.pack_native_prefix(model, batch, low, config, final_observer)
                    record.update(metadata)
                add_trace(final_observer, 'onsite_low_fallback_rebuild')
                record['fallback_low_rebuild_wall_ms'] = (time.perf_counter()-t_rebuild)*1000
                record['fallback_kind'] = 'onsite_rebuilt_original_low_after_actual_budget_rejection'
            else:
                record['fallback_kind'] = None
                record['T_answer_wall_ms'] = record['candidate_answer_attempt_wall_ms']
        else:
            record['T_action_wall_ms'] = (time.perf_counter()-t_action)*1000
            record['fallback_kind'] = 'original_low_reused_no_CPU_candidate_fits'
        if not committed or exceeded:
            require(cache.get_seq_length() == batch.boundary, 'a0 fallback must start after probe rollback/original rebuild')
            record.update(base.input_evidence(batch))
            record.update(action_prefix_information_condition='original_query_unknown_low_prefix',
                question_unknown_prefix_verified=True, action_implementation=record['fallback_kind'])
            if not committed:
                record.update(low_meta)
                record['a0_original_low_cache_retained_without_candidate'] = True
            _owner_fields(record, cache)
            final_observer = base.GenerationObserver(model, started, batch)
            t_answer = time.perf_counter()
            with base.observe_generation(model, final_observer):
                _guarded_answer(model, processor, batch, cache, row, config, output_dir,
                    record, final_observer, baseline['total_bytes'], candidate=False)
            record['T_answer_wall_ms'] = (time.perf_counter()-t_answer)*1000
            add_trace(final_observer, 'a0_final_answer')
        record['outer_forward_calls'] = traces
        record['ttft_ms'] = (final_observer.token_times[0]-started)*1000
        record['inter_token_ms'] = [(b-a)*1000 for a,b in zip(final_observer.token_times,final_observer.token_times[1:])]
        record['total_KV_budget_slack_bytes'] = baseline['total_bytes']-record['first_answer_total_cache_bytes']
        record['final_visual_bytes_vs_original_low'] = record['complete_visual_owner_bytes']-record['original_low_visual_owner_bytes']
        record['local_final_bit_label_not_original_image_1p3_claim'] = True
        record['original_source_denominator_effective_visual_bits'] = 16*record['complete_visual_owner_bytes']/record['original_visual_FP16_denominator']
        record['status'] = 'PASS'
    except Exception as exc:
        record.update(status='OOM_AT_DECLARED_INPUT' if 'out of memory' in str(exc).lower() else 'FAILED_RUN',
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            output_token_ids=None, raw_prediction=None)
    finally:
        if baseline_handle is not None:
            baseline_handle.remove()
        if 'outer_forward_calls' not in record:
            add_trace(low_observer, 'unfinished_initial_low_probe')
            add_trace(final_observer, 'unfinished_selected_or_fallback_path')
            record['outer_forward_calls'] = traces
        record.update(wall_ms=(time.perf_counter()-started)*1000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            end_allocated_bytes=torch.cuda.memory_allocated(),
            timing_scope='actual original-low probe + CPU native budget enumeration + committed re-encoding or charged fallback + final answer',
            budget_scope='first-answer persistent cache_storage.total_bytes; not allocator/model/encoding-workspace bound')
        cache = batch = image = low_observer = final_observer = None
        gc.collect()
        torch.cuda.empty_cache()
    return record


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value.update(round12_budget_action_version=2,
            budget_scope='actual original-low first complete suffix total persistent KV',
            CPU_enumeration_not_multiple_GPU_model_attempts=True,
            encoding_workspace_and_peak_reported_outside_persistent_KV_cap=True)
    elif Path(path).name == 'complete.json':
        expected = value['expected_QA']
        value['coverage_complete'] = value['recorded_QA'] == expected and sum(value['counts'].values()) == expected
        value['all_failed_sources_remain_in_primary_denominator'] = True
        value['round12_budget_action_version'] = 2
    return _base_write(path, value)


def main():
    base.SCHEMA = SCHEMA
    base.RUNTIME_FILES = RUNTIME_FILES
    base.validate_config = validate_config
    base.variants_from_config = variants_from_config
    base.run_action_row = run_budget_row
    base.write_new = write_new
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
