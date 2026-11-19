"""Independent G4_16 full-row cosine coordinate prototype, one/two sweeps.

For each slot propose the word maximizing signed (x dot t)/||t|| from full-row
incremental moments, then recompute that score on all128 coordinates. Optional
hard acceptance compares only this proposal with the incumbent, with exactly
the old norm->FP16->factor->FP16 gain and FP16 base decode. Neither the search
nor its base-only guarantee claims optimality of subsequently recoded residuals.
No book, field width, serving decoder, or retained tensor constant is added.
"""
import torch

from round8_codec import (encode_key as legacy_encode_key, key_amplitude,
                         key_template, pack_nibbles, unpack_nibbles)


def _gain(source, template, factor):
    # Use the original norm reduction, not sqrt of the FP64 moment cache.
    return (key_amplitude(source, template, 'norm').half().float() * factor).half()


def _moments(source_double, template):
    value = template.double()
    return (source_double * value).sum(-1), value.square().sum(-1)


def _score(dot, square):
    return torch.where((square > 0) & torch.isfinite(dot) & torch.isfinite(square),
                       dot / square.clamp_min(torch.finfo(torch.float64).tiny).sqrt(),
                       torch.full_like(dot, -torch.inf))


def _base_sse(source_double, template, gain):
    decoded = (template * gain.float()).half()
    return (source_double - decoded.double()).square().sum(-1)


@torch.no_grad()
def _row_cosine_cd(source, book, *, amplitude='norm', assignment='row_cosine_cd',
               gain_factor=None, chunk=65536, sweeps=1, hard_accept=True,
               row_chunk=8192, return_stats=False):
    """Return the same packed uint8[...,16] and FP16[...,1] gain fields.

    ``assignment=l2`` is an exact direct call to the unchanged old encoder.
    The new branch supports only global16x4/norm. ``hard_accept`` defaults to
    true. Its strict per-row base-SSE guarantee excludes the original bank's
    subsequent sign/MAD residual reconstruction and all full-model objectives.
    """
    if assignment != 'row_cosine_cd' or amplitude != 'norm' or tuple(book.shape) != (16, 4):
        raise ValueError('Spherical prototype requires row_cosine_cd, norm and one global16x4 book')
    if sweeps not in (1, 2) or row_chunk < 1:
        raise ValueError('Exactly one/two sweeps and a positive row chunk required')
    old_code, old_gain = legacy_encode_key(source, book, amplitude='norm',
        assignment='l2', gain_factor=gain_factor, chunk=chunk)
    shape = source.shape
    raw = source.float().reshape(-1, 128)
    indices = unpack_nibbles(old_code).reshape(-1, 32).clone()
    current_gain = old_gain.reshape(-1, 1).clone()
    packed_old = old_code.reshape(-1, 16)
    centers = book.half().float()
    centers64 = centers.double()
    center_norm = centers64.square().sum(-1)
    if gain_factor is None:
        factors = torch.ones((len(raw), 1), device=raw.device, dtype=torch.float32)
    else:
        factor = torch.as_tensor(gain_factor, device=raw.device, dtype=torch.float32)
        factors = torch.broadcast_to(factor, (*shape[:-1], 1)).reshape(-1, 1)
    valid_total = torch.zeros((), device=raw.device, dtype=torch.int64)
    accepted_total = torch.zeros_like(valid_total)
    cos_total = torch.zeros_like(valid_total)
    rejected_total = torch.zeros_like(valid_total)
    nonfinite_total = torch.zeros_like(valid_total)
    before_total = torch.zeros((), device=raw.device, dtype=torch.float64)
    after_total = torch.zeros_like(before_total)
    maximum_increase = torch.zeros_like(before_total)
    for start in range(0, len(raw), row_chunk):
        stop = min(start + row_chunk, len(raw))
        x = raw[start:stop]
        xd = x.double()
        t = key_template(packed_old[start:stop], book).reshape(-1, 128)
        ids = indices[start:stop].clone()
        gain = current_gain[start:stop].clone()
        f = factors[start:stop]
        dot, square = _moments(xd, t)
        z = xd.square().sum(-1)
        valid = (torch.isfinite(x).all(-1) & torch.isfinite(t).all(-1)
                 & torch.isfinite(gain[:, 0]) & torch.isfinite(f[:, 0]) & (f[:, 0] > 0)
                 & torch.isfinite(z) & (z > 0) & torch.isfinite(square) & (square > 0))
        current_score = _score(dot, square)
        old_sse = _base_sse(xd, t, gain)
        current_sse = old_sse.clone()
        valid &= torch.isfinite(old_sse)
        valid_total += valid.sum()
        local_rows = torch.arange(len(x), device=x.device)
        for _ in range(sweeps):
            for slot in range(32):
                sl = slice(slot * 4, slot * 4 + 4)
                previous = t[:, sl].double()
                old_dot = (xd[:, sl] * previous).sum(-1)
                old_square = previous.square().sum(-1)
                candidate_dot = dot[:, None] - old_dot[:, None] + xd[:, sl] @ centers64.T
                candidate_square = square[:, None] - old_square[:, None] + center_norm[None]
                scores = _score(candidate_dot, candidate_square)
                scores = torch.where(valid[:, None], scores, torch.full_like(scores, -torch.inf))
                # Exact incumbent score wins ties; old indices are never
                # changed merely by argmin/argmax's first-index tie rule.
                scores[local_rows, ids[:, slot].long()] = current_score
                best_score, best = scores.max(-1)
                propose = valid & (best != ids[:, slot].long()) & (best_score > current_score)
                proposal = t.clone()
                proposal[:, sl] = torch.where(propose[:, None], centers[best], t[:, sl])
                # A full recomputation prevents incremental cancellation from
                # accepting a false strict improvement or a duplicate word.
                next_dot, next_square = _moments(xd, proposal)
                next_score = _score(next_dot, next_square)
                improves_cos = propose & (next_score > current_score)
                next_gain = _gain(x, proposal, f)
                next_sse = _base_sse(xd, proposal, next_gain)
                finite = torch.isfinite(next_gain[:, 0]) & torch.isfinite(next_sse)
                accept = improves_cos & finite
                if hard_accept:
                    accept &= next_sse < current_sse
                cos_total += improves_cos.sum()
                rejected_total += (improves_cos & finite & (~accept)).sum()
                nonfinite_total += (improves_cos & (~finite)).sum()
                accepted_total += accept.sum()
                ids[:, slot] = torch.where(accept, best.byte(), ids[:, slot])
                t = torch.where(accept[:, None], proposal, t)
                gain = torch.where(accept[:, None], next_gain, gain)
                dot = torch.where(accept, next_dot, dot)
                square = torch.where(accept, next_square, square)
                current_score = torch.where(accept, next_score, current_score)
                current_sse = torch.where(accept, next_sse, current_sse)
        indices[start:stop] = ids
        current_gain[start:stop] = gain
        before_total += torch.where(valid, old_sse, 0.).sum()
        after_total += torch.where(valid, current_sse, 0.).sum()
        increase = torch.where(valid, current_sse - old_sse, 0.)
        maximum_increase = torch.maximum(maximum_increase, increase.max())
    code = pack_nibbles(indices.reshape(*shape[:-1], 32))
    gain = current_gain.reshape(*shape[:-1], 1).contiguous()
    if not return_stats:
        return code, gain
    changed = (code.reshape(len(raw), 16) != old_code.reshape(len(raw), 16)).any(-1)
    stats = dict(schema='trajkv.round8.row-cosine-cd.v1', assignment='row_cosine_cd',
        rows=len(raw), valid_rows=int(valid_total), skipped_rows=len(raw) - int(valid_total),
        sweeps=sweeps, slots_per_row=32, words=16, hard_base_SSE_acceptance=bool(hard_accept),
        strict_full_row_cosine_proposals=int(cos_total), accepted_slot_updates=int(accepted_total),
        cosine_best_proposals_rejected_by_hard_SSE=int(rejected_total), nonfinite_proposals=int(nonfinite_total),
        changed_rows=int(changed.sum()), source_base_SSE_before=float(before_total),
        source_base_SSE_after=float(after_total), max_valid_row_base_SSE_increase=float(maximum_increase),
        gain_rounding='original norm reduction ->FP16 ->frozen FP32 factor ->FP16; returned gains are the same fields evaluated by the guard',
        comparison_precision='signed cosine and hard-SSE sums in FP64; source/templates are actual FP32 and decoded values FP16',
        source_coordinates='caller supplied complete centered WHT vector; no additional rotation/centering',
        guarantee_scope=('accepted complete128-dimensional base-SSE strictly decreases; no guarantee after bank residual regeneration'
            if hard_accept else 'accepted signed full-row cosine strictly increases; no actual-SSE guarantee'),
        limitation='one cosine-best proposal/slot only; no local/global hard-optimality or task-quality claim',
        code_bytes_per_token=16, added_retained_tensor_constant_bytes=0,
        cost_scope='extra encoder compute and transient scratch are paid; serving reader and persistent tensor layout unchanged')
    return code, gain, stats


def encode_key(source, book, *, amplitude='norm', assignment='row_cosine_cd',
               gain_factor=None, chunk=65536, sweeps=1, hard_accept=True,
               row_chunk=8192, return_stats=False):
    """Exact direct legacy dispatch for l2; isolated no-grad new branch otherwise."""
    if assignment == 'l2':
        result = legacy_encode_key(source, book, amplitude=amplitude,
                                   assignment='l2', gain_factor=gain_factor, chunk=chunk)
        return (*result, dict(assignment='l2', direct_legacy_call=True)) if return_stats else result
    return _row_cosine_cd(source, book, amplitude=amplitude, assignment=assignment,
        gain_factor=gain_factor, chunk=chunk, sweeps=sweeps, hard_accept=hard_accept,
        row_chunk=row_chunk, return_stats=return_stats)
