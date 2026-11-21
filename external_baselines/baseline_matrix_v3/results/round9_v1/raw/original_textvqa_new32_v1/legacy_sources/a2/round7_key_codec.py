"""Round7 product-key codes: real nibble packing and explicit amplitude rules.

The norm rule reproduces Round6.  The LS rule is the unconstrained Euclidean
least-squares amplitude for each complete 128-vector, not norm matching.
Residual stages retain the frozen sequential FP16 reconstruction semantics.
"""
import torch

from round6_depth_codec import (decode_sign_stage, encode_sign_depth,
                               plan_depth_stages, validate_stages, validate_units)


def pack_nibbles(indices):
    if indices.dtype != torch.uint8 or indices.shape[-1] % 2:
        raise ValueError('An even final dimension of uint8 indices is required')
    if bool((indices > 15).any()):
        raise ValueError('A 16-word book requires four-bit indices')
    return (indices[..., 0::2] | (indices[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed):
    if packed.dtype != torch.uint8:
        raise ValueError('Nibble payload must be uint8')
    return torch.stack((packed & 15, packed >> 4), -1).flatten(-2).contiguous()


def decode_key(code, gain, book):
    indices = unpack_nibbles(code) if book.shape[0] == 16 else code
    template = book.float()[indices.long()].flatten(-2)
    return (template * gain.float()).half()


def key_amplitude(source, template, mode='norm', u=None, gamma=None):
    source, template = source.float(), template.float()
    if mode == 'norm':
        return source.norm(dim=-1, keepdim=True) / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    if mode == 'projection':
        # Reuse DirectionBank's existing projection-preserving amplitude rule.
        # This is neither a Euclidean optimum nor an unbiasedness guarantee.
        dot = (source * template).sum(-1, keepdim=True)
        norm = source.norm(dim=-1, keepdim=True)
        fallback = norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        return torch.where(dot > 1e-12, norm.square() / dot.clamp_min(1e-12), fallback)
    numerator = (source * template).sum(-1, keepdim=True)
    denominator = template.square().sum(-1, keepdim=True)
    if mode == 'metric_ls':
        if u is None or gamma is None:
            raise ValueError('metric_ls requires an explicit charged rank-one metric')
        numerator = numerator + gamma * (source * u).sum(-1, keepdim=True) * (template * u).sum(-1, keepdim=True)
        denominator = denominator + gamma * (template * u).sum(-1, keepdim=True).square()
    elif mode != 'ls':
        raise ValueError('key_amplitude must be norm/ls/metric_ls/projection')
    return numerator / denominator.clamp_min(1e-20)


def _objective(source, decoded, u=None, gamma=None):
    error = source.float() - decoded.float()
    score = error.square().sum(-1)
    if u is not None:
        score = score + gamma.squeeze(-1) * (error * u).sum(-1).square()
    return score


def encode_key(source, book, *, amplitude='norm', assignment='l2',
               u=None, gamma=None, iterations=1, chunk=2048, gain_factor=None):
    """Encode centered WHT keys, returning only the packed payload and gain.

    l2 uses exactly the legacy initial normalized nearest-word rule.  bcd and
    rank1_bcd visit all 32 blocks at fixed rounded gain, use the exact FP16
    candidate decode, and accept only per-vector objective improvements. The
    gain is re-estimated after each sweep and accepted only if it improves the
    same rounded objective.  Folded metadata factors are included throughout.
    """
    if source.shape[-1] != 128 or tuple(book.shape) not in ((16, 4), (256, 4)):
        raise ValueError('D128 and a global 16x4 or 256x4 book required')
    if assignment not in ('l2', 'bcd', 'rank1_bcd'):
        raise ValueError('key_assignment must be l2/bcd/rank1_bcd')
    if assignment == 'rank1_bcd' and (u is None or gamma is None):
        raise ValueError('Rank-one assignment requires an explicit metric artifact')
    if assignment != 'rank1_bcd' and amplitude != 'metric_ls' and u is not None:
        raise ValueError('A supplied metric must actually be used by assignment or amplitude')
    raw = source.float()
    norm = raw.norm(dim=-1, keepdim=True)
    initial = (norm / 128 ** .5).clamp_min(1e-20)
    parts, centers, codes = (raw / initial).reshape(-1, 4), book.float(), []
    # Preserve the frozen distance expression, chunk size and tie rule.
    for start in range(0, len(parts), 65536):
        part = parts[start:start + 65536]
        distance = part.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * (part @ centers.T)
        codes.append(distance.argmin(-1).byte())
    indices = torch.cat(codes).reshape(*raw.shape[:-1], 32).contiguous()
    template = centers[indices.long()].reshape(raw.shape)
    factor = 1. if gain_factor is None else gain_factor
    # Two distinct FP16 roundings deliberately reproduce the metadata fold.
    gain = (key_amplitude(raw, template, amplitude, u, gamma).half().float() * factor).half().contiguous()
    if assignment != 'l2':
        if iterations < 1 or chunk < 1:
            raise ValueError('Positive BCD iterations and chunk size required')
        metric_u = u if assignment == 'rank1_bcd' else None
        metric_gamma = gamma if metric_u is not None else None
        flat = raw.reshape(-1, 128)
        flat_indices, flat_gain = indices.reshape(-1, 32), gain.reshape(-1, 1)
        flat_u = None if metric_u is None else metric_u.expand_as(raw).reshape(-1, 128)
        flat_gamma = None if metric_u is None else metric_gamma.expand(*raw.shape[:-1], 1).reshape(-1, 1)
        for _ in range(iterations):
            for start in range(0, len(flat), chunk):
                stop = min(start + chunk, len(flat))
                target, ids, amplitude_now = flat[start:stop], flat_indices[start:stop], flat_gain[start:stop]
                local_u = None if flat_u is None else flat_u[start:stop]
                local_gamma = None if flat_gamma is None else flat_gamma[start:stop]
                decoded = (centers[ids.long()].flatten(-2) * amplitude_now.float()).half()
                for block in range(32):
                    sl = slice(4 * block, 4 * block + 4)
                    error = target - decoded.float()
                    candidate = (centers[None] * amplitude_now.float()[:, :, None]).half().float()
                    new_error = target[:, None, sl] - candidate
                    score = new_error.square().sum(-1) - error[:, sl].square().sum(-1, keepdim=True)
                    if local_u is not None:
                        old_projection = (error * local_u).sum(-1, keepdim=True)
                        new_projection = old_projection + ((new_error - error[:, None, sl]) * local_u[:, None, sl]).sum(-1)
                        score += local_gamma * (new_projection.square() - old_projection.square())
                    best_score, best = score.min(-1)
                    accept = best_score < 0
                    ids[:, block] = torch.where(accept, best.byte(), ids[:, block])
                    decoded[:, sl] = torch.where(accept[:, None], candidate[torch.arange(len(target), device=target.device), best].half(), decoded[:, sl])
            template = centers[indices.long()].reshape(raw.shape)
            proposal = (key_amplitude(raw, template, amplitude, u, gamma).half().float() * factor).half()
            current_decoded = (template * gain.float()).half()
            proposal_decoded = (template * proposal.float()).half()
            improve = _objective(raw, proposal_decoded, metric_u, metric_gamma) < _objective(raw, current_decoded, metric_u, metric_gamma)
            gain.copy_(torch.where(improve[..., None], proposal, gain))
    payload = pack_nibbles(indices) if len(book) == 16 else indices
    return payload, gain
