"""Existing packed PQ reader with a calibrated visual-only logit bias."""
import torch
import triton
from fast_wht import wht
from fused_learned_direction import _read


def attention(q, state, book, v_group_size=128, visual_bias=None):
    b, h, n, d = q.shape
    nv = state.visual_pos.numel()
    group = h // state.text_k.shape[1]
    if b != 1 or d != 128 or q.dtype != torch.float16 or q.stride(-1) != 1:
        raise ValueError('Batch1 FP16 D128 required')
    if nv % v_group_size:
        raise ValueError('Complete V groups required')
    qr = wht(q, half=True)
    bias = (q.float() * state.direction_mean.repeat_interleave(group, 1).float()).sum(-1) / d**.5
    if visual_bias is not None:
        if visual_bias.shape != (state.text_k.shape[1],):
            raise ValueError('One visual bias per KV head is required')
        bias = bias + visual_bias.repeat_interleave(group)[None, :, None]
    bias = bias.contiguous()
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _read[(triton.cdiv(n, 16), h)](
        q, qr, bias, state.direction_code, state.direction_gain, book,
        state.v_code, state.v_scale, state.v_min,
        state.text_k, state.text_v, state.visual_pos, state.text_pos,
        output, n, nv, state.text_length, state.length-n,
        q.stride(1), q.stride(2), state.text_k.stride(1), d, group,
        book.shape[-1], 1, True, v_group_size, book.ndim == 3, 16, 32,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output
