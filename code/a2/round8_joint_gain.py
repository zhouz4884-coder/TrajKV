"""Fixed-index/sign two-field K refit with the actual FP16 decode order.

Only a pre-existing first K sign stage is eligible. The caller retains the
same indices, sign bytes, mean, mask and dictionary. ``old_gain`` is already
the physical gain after metadata folding; no factor is multiplied here.
Import and execution require only Torch, on CPU or CUDA. This module launches
no model, adds no retained parameter, and does not fit a dictionary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _signs(code):
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    return (((code.long()[..., None] >> shifts) & 1).flatten(-2).float() * 2 - 1)


def _hard_decode(template, gain, sign, alpha):
    base = (template * gain.float()).half()
    return (base.float() + sign * alpha.float()[:, None]).half()


def _finite_total(values):
    return float(values.sum()) if bool(torch.isfinite(values).all()) else None


@torch.no_grad()
def joint_refit_key(sourceRotFP32, templateFP32, old_gainF16, sign_codeI32, old_alphaF16):
    """Return (new gain[N,1], new alpha[N], JSON-safe statistics).

    Solve the real two-variable nonnegative least-squares subproblem by
    enumerating its interior, both axes, and zero. Those real solutions are
    candidates, not FP16 optimality certificates: stored fields are rounded
    to FP16, then Q16(Q16(g*t)+beta*s) is scored against the supplied native
    source. The original physical pair is always the first candidate, so
    exact ties and every non-improving/nonfinite row retain its original bits.

    Apply only to a selected first-stage K residual. Calling this function for
    a missing stage or the first stage of a two-stage K chain would change the
    declared intervention and is the caller's error.
    """
    source, template = sourceRotFP32, templateFP32
    gain, alpha, code = old_gainF16, old_alphaF16, sign_codeI32
    if source.dtype != torch.float32 or template.dtype != torch.float32:
        raise ValueError('Native source and complete selected template must be FP32')
    if source.ndim != 2 or source.shape[-1] != 128 or source.shape != template.shape:
        raise ValueError('Equal [N,128] source/template tensors required')
    n = len(source)
    if tuple(gain.shape) != (n, 1) or tuple(alpha.shape) != (n,) or tuple(code.shape) != (n, 4):
        raise ValueError('Existing gain[N,1], alpha[N] and packed sign[N,4] required')
    if gain.dtype != torch.float16 or alpha.dtype != torch.float16 or code.dtype != torch.int32:
        raise ValueError('Existing physical fields must be FP16 gain/alpha and int32 sign')
    if any(x.device != source.device for x in (template, gain, alpha, code)):
        raise ValueError('All existing fields and source must share one device')
    sign = _signs(code)
    valid = (torch.isfinite(source).all(-1) & torch.isfinite(template).all(-1)
             & torch.isfinite(gain[:, 0]) & torch.isfinite(alpha))
    # Double precision normal equations reduce cancellation for almost
    # collinear columns. Det=0 is handled by the two boundary candidates.
    td, sd, zd = template.double(), sign.double(), source.double()
    a = td.square().sum(-1)
    b = (td * sd).sum(-1)
    c = sd.square().sum(-1)
    u = (td * zd).sum(-1)
    v = (sd * zd).sum(-1)
    det = a*c-b.square()
    safe_det = torch.where(det > 0, det, torch.ones_like(det))
    interior_g = (u*c-v*b)/safe_det
    interior_beta = (v*a-u*b)/safe_det
    interior_valid = valid & (det > 0) & (interior_g >= 0) & (interior_beta >= 0)
    axis_g = torch.where(a > 0, u/a.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(a)).clamp_min(0)
    axis_beta = torch.where(c > 0, v/c.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(c)).clamp_min(0)
    zeros = torch.zeros_like(a)

    def score(candidate_gain, candidate_alpha):
        decoded = _hard_decode(template, candidate_gain, sign, candidate_alpha)
        return (zd-decoded.double()).square().sum(-1)

    result_gain, result_alpha = gain.clone(), alpha.clone()
    old_score = score(gain, alpha)
    best_score = old_score.clone()
    accepted_kind = torch.zeros(n, device=source.device, dtype=torch.int64)
    candidates = ((interior_g, interior_beta, interior_valid),
                  (axis_g, zeros, valid & (a > 0)),
                  (zeros, axis_beta, valid & (c > 0)),
                  (zeros, zeros, valid))
    for kind, (real_g, real_beta, feasible) in enumerate(candidates, 1):
        candidate_g, candidate_beta = real_g.half()[:, None], real_beta.half()
        candidate_score = score(candidate_g, candidate_beta)
        finite = (torch.isfinite(candidate_g[:, 0]) & torch.isfinite(candidate_beta)
                  & torch.isfinite(candidate_score))
        # A finite candidate may repair an overflowing old decode. Invalid
        # input rows always preserve their old fields, even if score is NaN.
        improve = feasible & finite & (candidate_score < best_score)
        result_gain = torch.where(improve[:, None], candidate_g, result_gain)
        result_alpha = torch.where(improve, candidate_beta, result_alpha)
        best_score = torch.where(improve, candidate_score, best_score)
        accepted_kind = torch.where(improve, torch.full_like(accepted_kind, kind), accepted_kind)
    improved = accepted_kind > 0
    finite_pair = torch.isfinite(old_score) & torch.isfinite(best_score)
    maximum_increase = float((best_score[finite_pair]-old_score[finite_pair]).max()) if bool(finite_pair.any()) else 0.
    stats = dict(schema='trajkv.round8.joint-gain-fixed-fields.v1', rows=n,
        improved_rows=int(improved.sum()), unchanged_rows=int((~improved).sum()),
        nonfinite_input_rows=int((~valid).sum()), old_nonfinite_decode_rows=int((~torch.isfinite(old_score)).sum()),
        new_nonfinite_decode_rows=int((~torch.isfinite(best_score)).sum()),
        source_sse_before=_finite_total(old_score), source_sse_after=_finite_total(best_score),
        max_finite_row_sse_increase=maximum_increase,
        real_interior_feasible_rows=int(interior_valid.sum()), singular_or_nonpositive_det_rows=int((det <= 0).sum()),
        chosen_candidate_counts={name: int((accepted_kind == index).sum()) for index, name in enumerate(
            ('old_physical_pair', 'nonnegative_interior', 'gain_axis', 'alpha_axis', 'zero'))},
        objective='sum native FP32 source SSE against Q16(Q16(final_effective_gain*template)+alpha*fixed_sign)',
        acceptance='strict per-row improvement after both stored fields and both decode stages are rounded to FP16; old pair wins ties',
        fixed_fields='J, packed sign bytes, dictionary, mean/frame and first-stage selection mask; only existing gain and alpha slots change',
        factor_policy='input gain is the final already-folded physical gain; no factor applied again',
        added_retained_parameters=0, scope='coordinate NNLS candidates with hard-FP16 acceptance; not a full-model quality claim')
    return result_gain.contiguous(), result_alpha.contiguous(), stats


def _pack_sign(sign):
    bits = (sign >= 0).long().reshape(-1, 4, 32)
    shift = torch.arange(32, dtype=torch.int64)
    return (bits << shift).sum(-1).int().contiguous()


def cpu_check():
    """Finite small problems, degeneracies and old-field tie behavior."""
    sign = torch.ones((1, 128), dtype=torch.float32)
    t = torch.tensor([1., -1.]).repeat(64)[None]
    code = _pack_sign(sign)
    examples = []

    def run(name, z, template, gain, alpha):
        ng, na, stats = joint_refit_key(z.float(), template.float(), gain.half(), code, alpha.half())
        assert stats['max_finite_row_sse_increase'] <= 0
        examples.append(dict(name=name, old_gain=gain.tolist(), old_alpha=alpha.tolist(),
                             new_gain=ng.tolist(), new_alpha=na.tolist(), stats=stats))
        return ng, na, stats

    # The two design coefficients are both exactly representable in FP16.
    ng, na, st = run('orthogonal_exact_0.75_1.25', .75*t+1.25*sign, t,
                     torch.ones(1, 1), torch.ones(1))
    assert float(ng) == .75 and float(na) == 1.25 and st['source_sse_after'] == 0
    # Rank-one Gram matrix: a boundary is a complete real NNLS solution.
    ng, na, st = run('det0_collinear_improves', 3*sign, sign,
                     torch.ones(1, 1), torch.full((1,), .5))
    assert st['source_sse_after'] == 0 and st['singular_or_nonpositive_det_rows'] == 1
    ng, na, st = run('zero_template', 2*sign, torch.zeros_like(t),
                     torch.full((1, 1), 3.), torch.full((1,), .5))
    assert float(na) == 2 and st['source_sse_after'] == 0
    ng, na, st = run('boundary_nonnegative_constraint', -t+sign, t,
                     torch.ones(1, 1), torch.ones(1))
    assert float(ng) == 0 and float(na) == 1 and st['source_sse_after'] == 128
    ng, na, st = run('zero_optimum', -t-sign, t,
                     torch.ones(1, 1), torch.ones(1))
    assert float(ng) == 0 and float(na) == 0 and st['source_sse_after'] == 256
    # Boundary g=2,beta=0 decodes identically; strict acceptance must preserve
    # the existing nonunique physical pair g=.75,beta=1.25 bit for bit.
    old_g, old_a = torch.full((1, 1), .75), torch.full((1,), 1.25)
    ng, na, st = run('old_physical_pair_tie_is_preserved', 2*sign, sign, old_g, old_a)
    assert torch.equal(ng, old_g.half()) and torch.equal(na, old_a.half()) and st['improved_rows'] == 0
    bad = t.clone()
    bad[0, 0] = float('nan')
    ng, na, st = run('nonfinite_source_is_preserved', bad, t, old_g, old_a)
    assert torch.equal(ng, old_g.half()) and torch.equal(na, old_a.half()) and st['nonfinite_input_rows'] == 1
    # The actual sign bytes are read-only inputs.
    assert torch.equal(code, _pack_sign(sign))
    return dict(status='PASS', schema='trajkv.round8.joint-gain-cpu-check.v1',
        scope='CPU algebra and exact FP16 toy decode only; no model/GPU/LBG execution', cases=examples,
        no_sign_mutation=True, no_extra_retained_fields=True)


def main():
    p = argparse.ArgumentParser(description='Run only the CPU joint-field numerical audit')
    p.add_argument('--check-output', type=Path, required=True)
    args = p.parse_args()
    if args.check_output.exists():
        raise ValueError('Frozen numerical evidence is never overwritten; choose a new output path')
    args.check_output.parent.mkdir(parents=True, exist_ok=True)
    result = cpu_check()
    args.check_output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], cases=len(result['cases']), output=str(args.check_output))))


if __name__ == '__main__':
    main()
