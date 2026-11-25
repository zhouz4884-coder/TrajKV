"""Round13 compact mixed key indices and deployed-endpoint V transport.

These primitives never retain native visual K/V. All endpoint arithmetic marks
the Half roundings used by the packed readers; analysis is not task evidence.
"""
from __future__ import annotations

import torch
import round12_static_codec_v1 as previous


def require(value, message):
    if not value:
        raise ValueError('ROUND13_PRECISION_CODEC: ' + message)


def tier_rank(high_mask):
    """Signed dense ranks: high >=0, low = -(rank+1), one real int32/N."""
    require(high_mask.dtype == torch.bool and high_mask.ndim == 1, 'one shared token mask')
    high = high_mask.to(torch.int32).cumsum(0).to(torch.int32) - 1
    low = (~high_mask).to(torch.int32).cumsum(0).to(torch.int32) - 1
    return torch.where(high_mask, high, -low - 1).contiguous()


def decode_token_key(low_code, high_code, gain, low_book, high_book, ranks):
    """Dense numerical reference only; deployment reads the two compact pools."""
    require(ranks.dtype == torch.int32 and ranks.ndim == 1, 'charged signed int32 map')
    n = ranks.numel()
    require(low_code.shape[-2] + high_code.shape[-2] == n, 'disjoint complete key pools')
    require(low_code.shape[-1] == 16 and high_code.shape[-1] == 32, 'real 4/8-bit index pools')
    require(gain.shape[-2:] == (n, 1), 'one gain per selected row')
    high = ranks >= 0
    result = gain.new_empty((*gain.shape[:-1], 128))
    if low_code.shape[-2]:
        result[..., ~high, :] = previous.decode_key(low_code, gain[..., ~high, :], low_book)
    if high_code.shape[-2]:
        result[..., high, :] = previous.decode_key(high_code, gain[..., high, :], high_book)
    return result


@torch.no_grad()
def fit_v2_raw(source, levels, *, iterations=12):
    """Exactly the unfurled fitting/assignment half of frozen V2R."""
    require(source.dtype == torch.float16 and source.ndim == 4 and source.shape[-1] == 128,
            'native V [1,H,N,128]')
    x = source.float().transpose(2, 3).contiguous()
    require(x.shape[-1] % 32 == 0 and tuple(levels.shape) == (4,), 'complete groups and four levels')
    mu, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True, unbiased=False)
    minimum, scale = mu - 1.5 * std, (3 * std).clamp_min(1e-12)
    lv = levels.float()
    for _ in range(iterations):
        ids = (((x - minimum) / scale.clamp_min(1e-12))[..., None] - lv).abs().argmin(-1)
        z = lv[ids]
        zmean, xmean = z.mean(-1, keepdim=True), x.mean(-1, keepdim=True)
        zvar = (z - zmean).square().mean(-1, keepdim=True)
        fitted = ((z - zmean) * (x - xmean)).mean(-1, keepdim=True) / zvar.clamp_min(1e-20)
        scale = torch.where(zvar > 1e-20, fitted.clamp_min(1e-12), torch.zeros_like(fitted))
        minimum = xmean - scale * zmean
    minimum, scale = minimum.half().contiguous(), scale.half().contiguous()
    errors = []
    for level in levels:
        decoded = ((level * scale).half() + minimum).half()
        errors.append((x - decoded.float()).square())
    ids = torch.stack(errors, -1).argmin(-1)
    return previous.pack_channel_codes(ids, 2), scale, minimum


def binary_endpoints(scale, minimum):
    zero = ((torch.zeros_like(scale) * scale).half() + minimum).half()
    one = ((torch.ones_like(scale) * scale).half() + minimum).half()
    return zero.float(), one.float()


@torch.no_grad()
def transport_from_binary(raw_scale, raw_min, folded_scale, folded_min, new_scale, new_min,
                          *, nominal_gain):
    """Fit a*x+b to both actual old Half decoder endpoints, then fold it.

    A collapsed raw endpoint cannot identify an affine slope. In that case the
    declared nominal gain is used and the lower deployed endpoint is matched.
    Any old nonconstant folded endpoint in such a case is explicitly counted.
    Only the two existing Half metadata tensors are returned for deployment.
    """
    x0, x1 = binary_endpoints(raw_scale, raw_min)
    y0, y1 = binary_endpoints(folded_scale, folded_min)
    delta = x1 - x0
    identifiable = delta != 0
    fallback = nominal_gain.float().expand_as(delta)
    slope = torch.where(identifiable, (y1 - y0) / torch.where(identifiable, delta, torch.ones_like(delta)), fallback)
    intercept = y0 - slope * x0
    require(bool(torch.isfinite(slope).all()) and bool(torch.isfinite(intercept).all()), 'finite endpoint transport')
    scale = (slope * new_scale.float()).half().contiguous()
    minimum = (slope * new_min.float() + intercept).half().contiguous()
    error = torch.maximum((slope * x0 + intercept - y0).abs(), (slope * x1 + intercept - y1).abs())
    diagnostics = {
        'transport': 'two actual pre/fold Half binary endpoints',
        'identifiable_channels': int(identifiable.sum()),
        'degenerate_channels': int((~identifiable).sum()),
        'degenerate_nonconstant_folded_channels': int(((~identifiable) & (y0 != y1)).sum()),
        'degenerate_fallback': 'nominal frozen gain; intercept matches lower deployed endpoint',
        'old_endpoint_float_max_abs': float(error.max()),
        'slope_min': float(slope.min()), 'slope_max': float(slope.max()),
        'intercept_min': float(intercept.min()), 'intercept_max': float(intercept.max()),
        'folded_into_existing_Half_scale_min_only': True,
    }
    return scale, minimum, diagnostics


@torch.no_grad()
def encode_v2_transport(source, levels, raw_binary_scale, raw_binary_min,
                        folded_binary_scale, folded_binary_min, nominal_gain, *, iterations=12):
    code, scale, minimum = fit_v2_raw(source, levels, iterations=iterations)
    scale, minimum, diagnostics = transport_from_binary(raw_binary_scale, raw_binary_min,
        folded_binary_scale, folded_binary_min, scale, minimum, nominal_gain=nominal_gain)
    return code, scale, minimum, diagnostics
