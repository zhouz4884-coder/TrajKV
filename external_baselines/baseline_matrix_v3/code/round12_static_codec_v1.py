"""Round12 packed index/value primitives; no model or task execution.

K payloads are real little-endian bit streams (4/8/10/12 bits per word).
V payloads are channel-major 1/2-bit streams or 64 retained binary channels.
All reconstruction functions expose the exact Half rounding points.
"""
from __future__ import annotations

import torch


def require(value, message):
    if not value:
        raise ValueError('ROUND12_CODEC: ' + message)


def pack_indices(indices, bits):
    require(bits in (4, 8, 10, 12) and indices.ndim >= 1, 'index shape/bits')
    require(indices.shape[-1] * bits % 8 == 0, 'no padded index streams')
    require(not bool(((indices < 0) | (indices >= (1 << bits))).any()), 'index outside declared word book')
    values = indices.to(torch.int64)
    if bits == 8:
        return values.to(torch.uint8).contiguous()
    if bits == 4:
        return (values[..., 0::2] | values[..., 1::2] << 4).byte().contiguous()
    if bits == 10:
        require(values.shape[-1] % 4 == 0, '10-bit groups contain four words')
        v = values.reshape(*values.shape[:-1], -1, 4)
        word = v[..., 0] | (v[..., 1] << 10) | (v[..., 2] << 20) | (v[..., 3] << 30)
        shifts = torch.arange(5, device=values.device, dtype=torch.int64) * 8
        return ((word[..., None] >> shifts) & 255).byte().flatten(-2).contiguous()
    require(values.shape[-1] % 2 == 0, '12-bit groups contain two words')
    v = values.reshape(*values.shape[:-1], -1, 2)
    word = v[..., 0] | (v[..., 1] << 12)
    shifts = torch.arange(3, device=values.device, dtype=torch.int64) * 8
    return ((word[..., None] >> shifts) & 255).byte().flatten(-2).contiguous()


def unpack_indices(payload, bits, words):
    require(payload.dtype == torch.uint8 and payload.shape[-1] * 8 == words * bits, 'actual compact payload size')
    bit = torch.arange(words, device=payload.device, dtype=torch.long) * bits
    byte, shift = bit // 8, bit % 8
    v = payload.long()
    padded = torch.nn.functional.pad(v, (0, 2))
    combined = padded[..., byte] | padded[..., byte + 1] << 8 | padded[..., byte + 2] << 16
    return ((combined >> shift) & ((1 << bits) - 1)).long().contiguous()


def decode_key(payload, gain, book):
    bits = (int(book.shape[0]) - 1).bit_length()
    indices = unpack_indices(payload, bits, 128 // int(book.shape[-1]))
    template = book.float()[indices].flatten(-2)
    return (template * gain.float()).half()


@torch.no_grad()
def encode_key(source, book, gain_factor=None, *, chunk_parts=4096):
    words, dim = map(int, book.shape)
    require((words, dim) in ((16, 4), (256, 4), (1024, 4), (4096, 8)), 'supported global book')
    require(source.shape[-1] == 128 and source.dtype == torch.float32, 'centered WHT float32 D128')
    raw = source.float()
    norm = raw.norm(dim=-1, keepdim=True)
    initial = (norm / 128 ** .5).clamp_min(1e-20)
    parts, centers = (raw / initial).reshape(-1, dim), book.float()
    ids = []
    for start in range(0, len(parts), chunk_parts):
        p = parts[start:start + chunk_parts]
        distance = p.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * (p @ centers.T)
        ids.append(distance.argmin(-1))
    indices = torch.cat(ids).reshape(*raw.shape[:-1], 128 // dim)
    template = centers[indices].reshape(raw.shape)
    gain = norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    # Match the original separate Half metadata and factor-fold roundings.
    gain = (gain.half().float() * (1. if gain_factor is None else gain_factor)).half().contiguous()
    return pack_indices(indices, (words - 1).bit_length()), gain


def pack_channel_codes(codes, bits):
    require(bits in (1, 2) and codes.shape[-1] % (32 // bits) == 0, 'complete V words')
    c = codes.long().reshape(*codes.shape[:-1], -1, 32 // bits)
    shift = torch.arange(32 // bits, device=c.device, dtype=torch.int64) * bits
    return (c << shift).sum(-1).to(torch.int32).contiguous()


def unpack_channel_codes(payload, bits, n):
    require(payload.dtype == torch.int32 and payload.shape[-1] * (32 // bits) == n, 'compact V word count')
    shift = torch.arange(32 // bits, device=payload.device, dtype=torch.int64) * bits
    return ((payload.long()[..., None] >> shift) & ((1 << bits) - 1)).flatten(-2).contiguous()


def half_value(payload, scale, minimum, *, bits=1, levels=None):
    n = payload.shape[-1] * (32 // bits)
    indices = unpack_channel_codes(payload, bits, n)
    values = indices.half() if levels is None else levels[indices.long()].half()
    return ((values * scale).half() + minimum).half()


def normalized_levels(values, device):
    vals = torch.tensor(values, device=device, dtype=torch.float32)
    require(tuple(vals.shape) == (4,) and bool(torch.isfinite(vals).all()) and bool((vals[1:] > vals[:-1]).all()), 'four ordered finite levels')
    vals = (vals - vals[0]) / (vals[-1] - vals[0])
    return vals.half().contiguous()


@torch.no_grad()
def encode_v2_ratio(source, levels, gain, offset, *, iterations=12):
    """Fit affine four fixed-ratio levels, then apply frozen f0 exactly once.

    Metadata scale spans the full outer interval, so original f0 offsets remain
    in the same full-range scale units as the two-centroid representation.
    Codes are assigned before folding, matching the original encoder order.
    """
    require(source.dtype == torch.float16 and source.ndim == 4 and source.shape[-1] == 128, 'native V [1,H,N,128]')
    x = source.float().transpose(2, 3).contiguous()
    n = x.shape[-1]
    require(n % 32 == 0 and tuple(levels.shape) == (4,), 'complete groups and levels')
    mu, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True, unbiased=False)
    lo, scale = mu - 1.5 * std, (3 * std).clamp_min(1e-12)
    code = None
    lv = levels.float()
    for _ in range(iterations):
        normed = (x - lo) / scale.clamp_min(1e-12)
        code = (normed[..., None] - lv).abs().argmin(-1)
        z = lv[code]
        zmean, xmean = z.mean(-1, keepdim=True), x.mean(-1, keepdim=True)
        zvar = (z - zmean).square().mean(-1, keepdim=True)
        fitted = ((z - zmean) * (x - xmean)).mean(-1, keepdim=True) / zvar.clamp_min(1e-20)
        scale = torch.where(zvar > 1e-20, fitted.clamp_min(1e-12), torch.zeros_like(fitted))
        lo = xmean - scale * zmean
    # Final assignments use deployed pre-fold Half endpoints.
    lo, scale = lo.half().contiguous(), scale.half().contiguous()
    errors = []
    for level in levels:
        decoded = ((level * scale).half() + lo).half()
        errors.append((x - decoded.float()).square())
    code = torch.stack(errors, -1).argmin(-1)
    middle = lo.float() + .5 * scale.float()
    folded_scale = (scale.float() * gain).half().contiguous()
    folded_min = (middle + (offset - .5) * folded_scale.float()).half().contiguous()
    return pack_channel_codes(code, 2), folded_scale, folded_min


@torch.no_grad()
def compact_vhalf(payload, scale, minimum):
    n = payload.shape[-1] * 32
    bits = unpack_channel_codes(payload, 1, n)
    a0 = ((torch.zeros_like(scale) * scale).half() + minimum).half()
    a1 = ((torch.ones_like(scale) * scale).half() + minimum).half()
    p = bits.float().mean(-1, keepdim=True)
    delta = p * (1 - p) * (a1.float() - a0.float()).square()
    # Stable tie order: original channel index.
    indices = torch.argsort(delta[..., 0], dim=-1, descending=True, stable=True)[..., :64]
    selected = torch.zeros_like(delta[..., 0], dtype=torch.bool).scatter_(-1, indices, True)
    ordered = selected.nonzero(as_tuple=False)
    compact = payload[selected].reshape(payload.shape[0], payload.shape[1], 64, n // 32).contiguous()
    mu = ((1 - p) * a0.float() + p * a1.float()).half()
    keep = selected[..., None]
    new_scale = torch.where(keep, scale, torch.zeros_like(scale)).contiguous()
    new_min = torch.where(keep, minimum, mu).contiguous()
    b = selected.long().reshape(*selected.shape[:-1], 16, 8)
    shift = torch.arange(8, device=b.device, dtype=torch.int64)
    mask = (b << shift).sum(-1).byte().contiguous()
    return compact, new_scale, new_min, mask


def decode_vhalf(payload, scale, minimum, mask):
    n = payload.shape[-1] * 32
    bit = torch.arange(128, device=mask.device, dtype=torch.long)
    selected = ((mask[..., bit // 8].long() >> (bit % 8)) & 1).bool()
    require(bool((selected.sum(-1) == 64).all()), '64 retained V channels')
    result = minimum.expand(*minimum.shape[:-1], n).clone()
    full_bits = torch.zeros_like(result, dtype=torch.float16)
    unpacked = unpack_channel_codes(payload, 1, n).half()
    full_bits[selected] = unpacked.reshape(-1, n)
    return ((full_bits * scale).half() + minimum).half()
