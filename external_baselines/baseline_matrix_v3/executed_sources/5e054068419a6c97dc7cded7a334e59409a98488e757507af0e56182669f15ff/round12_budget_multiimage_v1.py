"""CPU-only native view enumeration for strict first-answer KV budgets.

Global context is deliberately cap1. ROI caps1..8 use actual native processor
outputs. Only the selected batch is later transferred to CUDA; no independent
KV prefixes are concatenated and no cap8 pixel-equivalence claim is made.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
import torch

from m1_protocol import M1PreparedBatch, prepare_m1_inputs
from round12_multiimage_v1 import _visual_runs

TEXT_KV_AND_POSITION_BYTES_PER_TOKEN = 57568
A_FORMAT_BYTES = 4


def require(value, message):
    if not value:
        raise ValueError('ROUND12_BUDGET_VIEWS: ' + message)


def predicted_a_visual_bytes(n):
    require(type(n) is int and n > 0 and n % 32 == 0, 'legal complete visual token count required')
    return 6382 * n + 91653 + A_FORMAT_BYTES


def predicted_first_suffix_total(n, full_length):
    require(type(full_length) is int and full_length >= n, 'invalid native full length')
    return predicted_a_visual_bytes(n) + TEXT_KV_AND_POSITION_BYTES_PER_TOKEN * (full_length - n)


def _cpu_batch(inputs, prompt, image_token_id):
    require(all(not isinstance(x, torch.Tensor) or x.device.type == 'cpu' for x in inputs.values()),
            'enumeration must not retain CUDA input tensors')
    vp = (inputs['input_ids'][0] == image_token_id).nonzero(as_tuple=False).flatten()
    require(vp.numel() > 0 and vp.numel() % 256 == 0, 'native tile placeholder count differs')
    boundary = int(vp[-1]) + 1
    positions = torch.arange(inputs['input_ids'].shape[-1], device='cpu')[None]
    prefix = {**inputs, 'input_ids': inputs['input_ids'][:, :boundary],
              'attention_mask': inputs['attention_mask'][:, :boundary]}
    return M1PreparedBatch(inputs, prefix, vp, boundary, positions, prompt)


def prepare_two_view_cpu(processor, image, crop, question, roi_cap, *,
                         image_token_id=151667, global_batch=None):
    require(type(roi_cap) is int and 1 <= roi_cap <= 8, 'ROI cap must be1..8')
    global_batch = global_batch or prepare_m1_inputs(processor, image, question,
        image_token_id=image_token_id, max_patches=1, device='cpu')
    local_batch = prepare_m1_inputs(processor, crop, question,
        image_token_id=image_token_id, max_patches=roi_cap, device='cpu')
    require(int(global_batch.visual_positions.numel()) == 256, 'global cap1 must be one256token tile')
    gp, lp = global_batch.inputs['pixel_values'], local_batch.inputs['pixel_values']
    require(gp.device.type == lp.device.type == 'cpu' and gp.shape[0] == 1 and
            tuple(gp.shape[1:]) == tuple(lp.shape[1:]) == (3, 448, 448), 'native CPU tile geometry differs')
    # Same two-view text as frozen Z. Related/random mode is never in the prompt.
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Original image:\n'}, {'type': 'image', 'image': image},
        {'type': 'text', 'text': '\nMagnified region of the same image:\n'},
        {'type': 'image', 'image': crop}, {'type': 'text', 'text': question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pixels = torch.cat((gp, lp), dim=0)
    counts = [1, int(lp.shape[0])]
    text, pieces, images, videos = processor._insert_media_placeholders(
        [prompt], pixels, None, counts, [], np.cumsum(counts), np.array([0]), np.array([0]))
    require(images == 2 and videos == 0, 'native two-image placeholder closure differs')
    inputs = dict(processor.tokenizer(text, return_tensors='pt', padding_side='left'))
    processor._check_special_mm_tokens(text, inputs, modalities=['image'])
    inputs['pixel_values'] = torch.cat(pieces, dim=0)
    batch = _cpu_batch(inputs, prompt, image_token_id)
    runs = _visual_runs(batch.visual_positions)
    require(list(map(len, runs)) == [256, int(local_batch.visual_positions.numel())],
            'native view groups must be global256 then actual ROI tokens')
    require(torch.equal(inputs['pixel_values'][:1], gp) and
            torch.equal(inputs['pixel_values'][1:], lp), 'native independently processed pixels changed')
    witness = dict(schema='trajkv.round12.budget_native_views.v1',
        view_mode='global_cap1_plus_ROI', global_cap=1, roi_cap=roi_cap,
        actual_global_tiles=1, actual_roi_tiles=int(lp.shape[0]),
        actual_global_visual_tokens=256, actual_roi_visual_tokens=int(local_batch.visual_positions.numel()),
        actual_visual_tokens=int(batch.visual_positions.numel()),
        visual_group_lengths=list(map(len, runs)),
        global_pixels_equal_to_native_cap1=True, ROI_pixels_equal_to_native_selected_cap=True,
        original_cap8_pixel_preservation_claimed=False,
        full_native_two_image_prefix=True, independent_KV_prefix_concatenated=False,
        all_visual_groups_in_owner=True, input_device='cpu',
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    return batch, witness


def plan_budget_views_cpu(processor, image, crop, question, budget_bytes, *, image_token_id=151667):
    require(type(budget_bytes) is int and budget_bytes > 0, 'actual original-low budget required')
    started = time.perf_counter()
    global_batch = prepare_m1_inputs(processor, image, question,
        image_token_id=image_token_id, max_patches=1, device='cpu')
    attempts = []
    best_batch = best_witness = None
    best_rank = None
    for cap in range(1, 9):
        batch, witness = prepare_two_view_cpu(processor, image, crop, question, cap,
            image_token_id=image_token_id, global_batch=global_batch)
        n, length = int(batch.visual_positions.numel()), int(batch.full_length)
        total = predicted_first_suffix_total(n, length)
        item = dict(roi_cap=cap, view_mode=witness['view_mode'], actual_roi_tiles=witness['actual_roi_tiles'],
            N=n, full_length=length, nonvisual_tokens=length-n,
            predicted_A_visual_bytes=predicted_a_visual_bytes(n),
            predicted_text_KV_and_positions_bytes=TEXT_KV_AND_POSITION_BYTES_PER_TOKEN*(length-n),
            predicted_first_suffix_total_bytes=total, fits=total <= budget_bytes,
            all_input_tensors_CPU=True)
        attempts.append(item)
        rank = (witness['actual_roi_tiles'], -total, -cap)
        if total <= budget_bytes and (best_rank is None or rank > best_rank):
            best_rank, best_batch, best_witness = rank, batch, witness
        batch = witness = None
    if best_batch is None:
        batch = prepare_m1_inputs(processor, crop, question, image_token_id=image_token_id,
                                  max_patches=1, device='cpu')
        require(int(batch.visual_positions.numel()) == 256, 'crop-only cap1 must be256tokens')
        n, length = 256, int(batch.full_length)
        total = predicted_first_suffix_total(n, length)
        attempts.append(dict(roi_cap=1, view_mode='ROI_only_cap1', actual_roi_tiles=1,
            N=n, full_length=length, nonvisual_tokens=length-n,
            predicted_A_visual_bytes=predicted_a_visual_bytes(n),
            predicted_text_KV_and_positions_bytes=TEXT_KV_AND_POSITION_BYTES_PER_TOKEN*(length-n),
            predicted_first_suffix_total_bytes=total, fits=total <= budget_bytes,
            all_input_tensors_CPU=True))
        if total <= budget_bytes:
            best_batch = batch
            best_witness = dict(schema='trajkv.round12.budget_native_views.v1',
                view_mode='ROI_only_cap1', global_cap=0, roi_cap=1,
                actual_global_tiles=0, actual_roi_tiles=1, actual_global_visual_tokens=0,
                actual_roi_visual_tokens=256, actual_visual_tokens=256,
                visual_group_lengths=[256], original_global_context_removed=True,
                original_cap8_pixel_preservation_claimed=False,
                independent_KV_prefix_concatenated=False, all_visual_groups_in_owner=True,
                input_device='cpu', prompt_sha256=hashlib.sha256(batch.prompt.encode()).hexdigest())
    chosen = None if best_batch is None else dict(best_witness,
        full_length=int(best_batch.full_length), nonvisual_tokens=int(best_batch.full_length-best_batch.visual_positions.numel()),
        predicted_first_suffix_total_bytes=predicted_first_suffix_total(int(best_batch.visual_positions.numel()), int(best_batch.full_length)),
        predicted_A_visual_bytes=predicted_a_visual_bytes(int(best_batch.visual_positions.numel())))
    report = dict(schema='trajkv.round12.budget_view_plan.v1', baseline_actual_total_bytes=budget_bytes,
        attempts=attempts, selected=chosen,
        selection_rule='maximum actual feasible ROI tile count; lower predicted bytes then lower cap break ties; crop-only cap1 only if no two-view fits',
        CPU_preprocessing_only=True, candidate_CUDA_transfers_during_enumeration=0,
        enumeration_wall_ms=(time.perf_counter()-started)*1000,
        predictor='A_visual(N,format4)+57568*(full_length-N); verified by actual first-suffix guard',
        prediction_is_not_actual_budget_acceptance=True)
    return best_batch, report


def move_selected_to_device(cpu_batch, device='cuda:0'):
    require(all(not isinstance(v, torch.Tensor) or v.device.type == 'cpu' for v in cpu_batch.inputs.values()),
            'selected batch must originate from CPU-only enumeration')
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.inputs.items()}
    vp = cpu_batch.visual_positions.to(device)
    positions = cpu_batch.positions.to(device)
    boundary = cpu_batch.boundary
    prefix = {**inputs, 'input_ids': inputs['input_ids'][:, :boundary],
              'attention_mask': inputs['attention_mask'][:, :boundary]}
    return M1PreparedBatch(inputs, prefix, vp, boundary, positions, cpu_batch.prompt)
