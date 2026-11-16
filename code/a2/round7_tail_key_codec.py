"""Real-N whole-image binary V with physical padding in its last bit word.

Centroids see exactly the real tokens. Only false unused code bits are added
before int32 packing; no key/value token, gain, position or residual is padded.
"""
import torch
import torch.nn.functional as F

from quantizers import pack1
from round7_key_codec import (decode_key, decode_sign_stage, encode_key,
                             encode_sign_depth, key_amplitude, pack_nibbles,
                             plan_depth_stages, unpack_nibbles)


def binary_centroids_tail(data, weights=None, iterations=3, group_size=None):
    """Match binary_centroids arithmetic, packing ceil(N/32) words/channel."""
    nv = data.shape[-1]
    group_size = nv if group_size is None else int(group_size)
    if nv <= 0 or group_size not in (nv, 128) or nv % group_size:
        raise ValueError('Positive real N, whole-image or complete g128 groups required')
    shape = data.shape
    g = data.float().reshape(*shape[:-1], -1, group_size)
    w = torch.ones_like(g) if weights is None else weights.float().expand_as(g)
    c = g > (g.amin(-1, keepdim=True) + g.amax(-1, keepdim=True)) / 2
    for step in range(iterations):
        a, b = w * (~c), w * c
        na, nb = a.sum(-1, keepdim=True), b.sum(-1, keepdim=True)
        m0 = (a * g).sum(-1, keepdim=True) / na.clamp_min(1e-12)
        m1 = (b * g).sum(-1, keepdim=True) / nb.clamp_min(1e-12)
        m0 = torch.where(na > 0, m0, m1)
        m1 = torch.where(nb > 0, m1, m0)
        if step + 1 < iterations:
            c = g > (m0 + m1) / 2
    minimum = m0.squeeze(-1).half().contiguous()
    scale = (m1 - m0).squeeze(-1).half().contiguous()
    bits = c.reshape(shape)
    tail = (-nv) % 32
    if tail:
        bits = F.pad(bits, (0, tail), value=False)
    return pack1(bits), scale, minimum


def decode_value_tail(code, scale, minimum, group_size, visual_tokens):
    """FP16 affine decode with physical bit-word padding truncated to real N."""
    nv = int(visual_tokens)
    if nv <= 0 or code.shape[-1] != (nv + 31) // 32 or nv % group_size:
        raise ValueError('Exact ceil(N/32) words and complete metadata groups required')
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    bits = ((code.long()[..., None] >> shifts) & 1).flatten(-2)[..., :nv].half()
    scales = scale.repeat_interleave(group_size, -1)
    minima = minimum.repeat_interleave(group_size, -1)
    if scales.shape != bits.shape or minima.shape != bits.shape:
        raise ValueError('Value metadata must cover exactly real N tokens')
    return ((bits * scales).half() + minima).half().contiguous()


def value_storage_account(layers, heads, visual_tokens, dimension=128):
    if dimension != 128 or min(layers, heads, visual_tokens) <= 0:
        raise ValueError('Positive L/H/N and D128 required')
    padded = (visual_tokens + 31) // 32 * 32
    return dict(words_per_channel=padded // 32, stored_token_bit_capacity=padded,
                real_visual_tokens=visual_tokens, unused_bits_per_channel=padded - visual_tokens,
                value_code_bytes=layers * heads * dimension * (padded // 32) * 4,
                unused_bit_capacity_bytes=layers * heads * dimension * (padded - visual_tokens) // 8,
                padded_visual_tokens=0, padded_visual_positions=0)
