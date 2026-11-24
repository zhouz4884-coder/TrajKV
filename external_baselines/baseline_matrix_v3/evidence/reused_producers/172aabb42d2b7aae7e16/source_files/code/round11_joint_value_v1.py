"""Round11 numerical contract: real joint search and actual-Half candidates.

The real joint search is global only for frozen real endpoints and a shared
nonnegative amplitude. The deployed Half search is a finite candidate search;
its original-field candidate guarantees token SSE never increases. No task,
attention, runtime or CUDA equivalence claim follows from this guarantee.
"""
from __future__ import annotations

import torch

SHUFFLE_A = 13
SHUFFLE_B = 7
SHUFFLE_A_INV = 5


def require(condition, message):
    if not condition:
        raise ValueError("ROUND11_NUMERICAL: " + message)


def unpack_bits(code, length):
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    return ((code.long()[..., None] >> shifts) & 1).reshape(*code.shape[:-1], -1)[..., :length].bool()


def pack_bits(bits):
    require(bits.shape[-1] % 32 == 0, "packed axis must be divisible32")
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    return (bits.long().reshape(*bits.shape[:-1], -1, 32) << shifts).sum(-1).int().contiguous()


def alpha_mode(alpha):
    """Read the bit, including negative zero; never use alpha < 0."""
    require(alpha.dtype == torch.float16, "alpha must be Half")
    return (alpha.contiguous().view(torch.int16) & -32768) != 0


def alpha_magnitude(alpha):
    require(alpha.dtype == torch.float16, "alpha must be Half")
    return (alpha.contiguous().view(torch.int16) & 32767).view(torch.float16)


def with_alpha_mode(magnitude, mode):
    raw = alpha_magnitude(magnitude).view(torch.int16)
    flag = torch.as_tensor(mode, device=raw.device, dtype=torch.bool)
    return (raw | torch.where(flag, -32768, 0).to(torch.int16)).view(torch.float16)


def word_spatial_index_map(shuffled=False, device=None):
    """N'=p^-1 N p on one 2x16 word; fallback self-edges stay self-edges.

    The returned tensors are transient shape-derived scratch, never assets.
    The reader derives these same integer expressions at compile time.
    """
    index = torch.arange(32, device=device, dtype=torch.int64)
    grid = (SHUFFLE_A * index + SHUFFLE_B) % 32 if shuffled else index
    col = grid % 16
    left = torch.where(col > 0, grid - 1, grid)
    right = torch.where(col < 15, grid + 1, grid)
    vertical = grid ^ 16
    if shuffled:
        left = (SHUFFLE_A_INV * (left - SHUFFLE_B)) % 32
        right = (SHUFFLE_A_INV * (right - SHUFFLE_B)) % 32
        vertical = (SHUFFLE_A_INV * (vertical - SHUFFLE_B)) % 32
    return left, right, vertical


def spatial_neighbor_numerators(base_bits, *, shuffled=False):
    require(base_bits.ndim == 2 and base_bits.shape[0] % 32 == 0, "[N,D], complete words")
    words = base_bits.int().reshape(-1, 32, base_bits.shape[-1])
    left, right, vertical = word_spatial_index_map(shuffled, base_bits.device)
    result = 4 * words + words[:, left] + words[:, right] + 2 * words[:, vertical]
    require(bool(((result >= 0) & (result <= 8)).all()), "numerator out of range")
    return result.reshape_as(base_bits)


def spatial_predictor(base_bits, l0, l1, *, shuffled=False):
    n = spatial_neighbor_numerators(base_bits, shuffled=shuffled).float()
    # Explicit separate operations match enable_fp_fusion=False in the reader.
    term0 = (8.0 - n) * l0.float()
    term1 = n * l1.float()
    return ((term0 + term1) * 0.125).half()


def real_joint_candidates(source, l0, l1):
    """D+1 clipped interval means, O(D log D) work per token.

    For fixed alpha>=0, sign minimization makes the two residual magnitudes
    u=min(|x-l0|,|x-l1|), v=max(...). Their crossing is (u+v)/2. Sorting these
    crossings gives every quadratic interval, including duplicated/empty ones.
    Computation uses float64 and does not claim rational exactness.
    """
    x, a, b = source.double(), l0.double(), l1.double()
    require(x.ndim >= 1 and bool(torch.isfinite(x).all() & torch.isfinite(a).all() & torch.isfinite(b).all()), "finite real inputs")
    r0, r1 = (x - a).abs(), (x - b).abs()
    u, v = torch.minimum(r0, r1), torch.maximum(r0, r1)
    threshold, order = ((u + v) * 0.5).sort(dim=-1, stable=True)
    delta = (v - u).gather(-1, order)
    zero = torch.zeros_like(u[..., :1])
    prefix = torch.cat((zero, delta.cumsum(-1)), -1)
    mean = (u.sum(-1, keepdim=True) + prefix) / x.shape[-1]
    lower = torch.cat((zero, threshold), -1)
    upper = torch.cat((threshold, torch.full_like(zero, float("inf"))), -1)
    return torch.minimum(torch.maximum(mean, lower), upper)


def real_joint_sse(source, l0, l1, alpha):
    x, a, b = source.double(), l0.double(), l1.double()
    error0 = ((x - a).abs() - alpha.double()[..., None]).square()
    error1 = ((x - b).abs() - alpha.double()[..., None]).square()
    return torch.minimum(error0, error1).sum(-1)


def half_candidates(real_candidates, old_alpha):
    rounded = real_candidates.half()
    lower = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
    upper = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
    return torch.cat((old_alpha[:, None], rounded, lower, upper), -1).contiguous()


def decode_fields(base_bits, sign_bits, alpha, l0, l1, *, spatial=False, shuffled=False):
    base = torch.where(base_bits, l1, l0)
    if spatial:
        predictor = spatial_predictor(base_bits, l0, l1, shuffled=shuffled)
        base = torch.where(alpha_mode(alpha)[:, None], predictor, base)
        alpha = alpha_magnitude(alpha)
    return (base.float() + (sign_bits.float() * 2 - 1) * alpha.float()[:, None]).half()


def _validate_fields(source, old_bits, old_sign_bits, old_alpha, l0, l1):
    require(source.ndim == 2 and source.dtype == torch.float16, "source [N,D] Half")
    require(old_bits.shape == old_sign_bits.shape == source.shape and old_alpha.shape == source.shape[:1], "field shapes")
    require(l0.dtype == l1.dtype == old_alpha.dtype == torch.float16, "Half endpoints/alpha")
    require(bool(torch.isfinite(source).all() & torch.isfinite(l0).all() & torch.isfinite(l1).all() & torch.isfinite(old_alpha).all()), "finite fields")
    require(not bool(alpha_mode(old_alpha).any()), "input mode0 alpha signbit must be clear")


def _search_actual(source, l0, l1, old_bits, old_sign_bits, old_alpha, candidates,
                   *, fixed_predictor=None, candidate_chunk=32):
    require(type(candidate_chunk) is int and candidate_chunk > 0, "positive candidate chunk")
    x = source.double()
    old_decode = decode_fields(old_bits, old_sign_bits, old_alpha, l0, l1)
    require(bool(torch.isfinite(old_decode).all()), "original Half decode must be finite")
    best_sse = (x - old_decode.double()).square().sum(-1)
    before_sse = best_sse.clone()
    best_b, best_s, best_a = old_bits.clone(), old_sign_bits.clone(), old_alpha.clone()
    l0c = l0[:, None] if l0.ndim == 2 else l0
    l1c = l1[:, None] if l1.ndim == 2 else l1
    changed = torch.zeros(source.shape[0], device=source.device, dtype=torch.bool)
    invalid_count = 0
    for start in range(0, candidates.shape[-1], candidate_chunk):
        alpha = candidates[:, start:start + candidate_chunk]
        valid_alpha = torch.isfinite(alpha) & ~alpha_mode(alpha)
        invalid_count += int((~valid_alpha).sum())
        amp = alpha.float()[..., None]
        b_old, s_old = old_bits[:, None], old_sign_bits[:, None]
        if fixed_predictor is None:
            initial = torch.where(b_old, l1c, l0c)
        else:
            initial = fixed_predictor[:, None]
        values = (initial.float() + (s_old.float() * 2 - 1) * amp).half()
        error = (x[:, None] - values.double()).square()
        error = torch.where(torch.isfinite(values), error, float("inf"))
        chosen_b = b_old.expand_as(error).clone()
        chosen_s = s_old.expand_as(error).clone()
        for bit in ((False, True) if fixed_predictor is None else (False,)):
            base = (l1c if bit else l0c) if fixed_predictor is None else fixed_predictor[:, None]
            for sign in (False, True):
                value = (base.float() + (1.0 if sign else -1.0) * amp).half()
                e = (x[:, None] - value.double()).square()
                e = torch.where(torch.isfinite(value), e, float("inf"))
                accept = e < error
                error = torch.where(accept, e, error)
                if fixed_predictor is None:
                    chosen_b = torch.where(accept, bit, chosen_b)
                chosen_s = torch.where(accept, sign, chosen_s)
        sse = error.sum(-1)
        sse = torch.where(valid_alpha, sse, float("inf"))
        value, index = sse.min(-1)
        accept = value < best_sse  # strict: equal SSE keeps the original fields
        row = torch.arange(source.shape[0], device=source.device)
        best_sse = torch.where(accept, value, best_sse)
        best_a = torch.where(accept, alpha[row, index], best_a)
        best_b = torch.where(accept[:, None], chosen_b[row, index], best_b)
        best_s = torch.where(accept[:, None], chosen_s[row, index], best_s)
        changed |= accept
    return best_b, best_s, best_a, changed, {
        "before_sse": float(before_sse.sum()), "after_sse": float(best_sse.sum()),
        "max_token_sse_increase": float((best_sse - before_sse).max()),
        "strictly_improved_tokens": int(changed.sum()),
        "invalid_alpha_candidates_rejected": invalid_count,
        "candidate_count_per_token": candidates.shape[-1],
        "comparison_dtype": "float64_actual_Half_output_SSE", "ties": "retain_original_fields"}


@torch.no_grad()
def joint_encode(source, l0, l1, old_bits, old_sign_bits, old_alpha,
                 *, tile_tokens=128, candidate_chunk=32):
    _validate_fields(source, old_bits, old_sign_bits, old_alpha, l0, l1)
    require(type(tile_tokens) is int and tile_tokens > 0, "positive token tile")
    result_b, result_s, result_a, stats = [], [], [], []
    for start in range(0, source.shape[0], tile_tokens):
        sl = slice(start, start + tile_tokens)
        # Endpoint tensors may be channel-only or already expanded per token.
        a = l0 if l0.ndim == 1 else l0[sl]
        b = l1 if l1.ndim == 1 else l1[sl]
        real = real_joint_candidates(source[sl], a, b)
        fields = _search_actual(source[sl], a, b, old_bits[sl], old_sign_bits[sl], old_alpha[sl],
                                half_candidates(real, old_alpha[sl]), candidate_chunk=candidate_chunk)
        result_b.append(fields[0]); result_s.append(fields[1]); result_a.append(fields[2]); stats.append(fields[4])
    return torch.cat(result_b), torch.cat(result_s), torch.cat(result_a), _merge_stats(stats, "joint_real_intervals_then_Half_round_neighbors")


def _merge_stats(items, algorithm):
    result = {"algorithm": algorithm, "tiles": len(items), "candidate_count_per_token": items[0]["candidate_count_per_token"],
              "comparison_dtype": items[0]["comparison_dtype"], "ties": items[0]["ties"]}
    for name in ("before_sse", "after_sse", "strictly_improved_tokens", "invalid_alpha_candidates_rejected"):
        result[name] = sum(x[name] for x in items)
    result["max_token_sse_increase"] = max(x["max_token_sse_increase"] for x in items)
    return result


@torch.no_grad()
def spatial_encode(source, l0, l1, base_bits, old_sign_bits, old_alpha,
                   *, shuffled=False, tile_tokens=128, candidate_chunk=32):
    _validate_fields(source, base_bits, old_sign_bits, old_alpha, l0, l1)
    require(type(tile_tokens) is int and tile_tokens > 0 and tile_tokens % 32 == 0, "spatial tile multiple32")
    predictor = spatial_predictor(base_bits, l0, l1, shuffled=shuffled)
    require(bool(torch.isfinite(predictor).all()), "nonfinite canonical spatial predictor")
    signs, alphas, modes, stats = [], [], [], []
    for start in range(0, source.shape[0], tile_tokens):
        sl = slice(start, start + tile_tokens)
        mad = (source[sl].double() - predictor[sl].double()).abs().mean(-1, keepdim=True)
        proposals = torch.cat((mad, torch.zeros_like(mad)), -1)
        fields = _search_actual(source[sl], l0, l1, base_bits[sl], old_sign_bits[sl], old_alpha[sl],
            half_candidates(proposals, old_alpha[sl]), fixed_predictor=predictor[sl], candidate_chunk=candidate_chunk)
        signs.append(fields[1]); alphas.append(fields[2]); modes.append(fields[3]); stats.append(fields[4])
    alpha = with_alpha_mode(torch.cat(alphas), torch.cat(modes))
    detail = _merge_stats(stats, "frozen_base_spatial_MAD_zero_oldAlpha_Half_neighbors")
    detail.update(mode1_tokens=int(alpha_mode(alpha).sum()), shuffled=bool(shuffled),
                  predictor="Half(((8-n)*float(A0)+n*float(A1))/8)")
    return torch.cat(signs), alpha, detail
