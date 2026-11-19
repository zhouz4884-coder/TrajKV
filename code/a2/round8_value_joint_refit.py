"""Encoder-only native binary/all V frame and first-alpha coordinate refit.

Fixed: actual channel-bit words, first-residual sign words, visual positions,
all-frame grouping, head selection and field widths. Optimized: only the
existing FP16 minimum[128,1], scale[128,1]>=0 and alpha[N]>=0. No reader/kernel,
codebook or retained tensor field is added. Input metadata is already folded;
the output is physical metadata and must not be folded a second time.

Real proposals are not FP16 optima. For fixed alpha, two-class least-squares
centers are projected onto center1>=center0 using their counts; an empty class
proposes both centers equal to the observed mean. For fixed physical frame,
alpha=max(mean(sign*(source-actual_FP16_base)),0). Accept frame proposals only
on a strict all-N/channel hard-SSE decrease; accept alpha proposals only on a
strict all-128/row decrease. Keep the old physical fields on ties. These
alternating disjoint-block tests make total native reconstruction SSE
nonincreasing, not every individual row/channel throughout both phases, and
make no attention/task guarantee. Signs are never regenerated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch


def _unpack_bits(code, n):
    positions = torch.arange(n, device=code.device, dtype=torch.int64)
    return ((code.long().index_select(-1, positions // 32) >> (positions % 32)) & 1).T.float()


def _unpack_sign(code):
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    return (((code.long()[..., None] >> shifts) & 1).flatten(-2).float() * 2 - 1)


def _base(bits, scale, minimum):
    product = (bits * scale[:, 0].float()[None]).half()
    return (product.float() + minimum[:, 0].float()[None]).half()


def _decoded(bits, scale, minimum, sign, alpha):
    return (_base(bits, scale, minimum).float() + sign * alpha.float()[:, None]).half()


def _errors(source_double, bits, scale, minimum, sign, alpha):
    return (source_double - _decoded(bits, scale, minimum, sign, alpha).double()).square()


def _finite_sum(values):
    return float(values.sum()) if bool(torch.isfinite(values).all()) else None


def _positive_increase(before, after):
    valid = torch.isfinite(before) & torch.isfinite(after)
    if not bool(valid.any()):
        return 0.
    return max(0., float((after[valid] - before[valid]).max()))


def _frame_proposal(source_double, bits, sign, alpha):
    """Count-weighted ordered two-level LS proposal, before physical rounding."""
    y = source_double - alpha.double()[:, None] * sign.double()
    b = bits.double()
    n1 = b.sum(0)
    n0 = len(y) - n1
    mean0 = ((1-b) * y).sum(0) / n0.clamp_min(1)
    mean1 = (b * y).sum(0) / n1.clamp_min(1)
    empty = (n0 == 0) | (n1 == 0)
    crossed = (~empty) & (mean1 < mean0)
    pooled = y.mean(0)  # n0*mean0+n1*mean1 over n0+n1, not unweighted pair mean.
    collapse = empty | crossed
    center0 = torch.where(collapse, pooled, mean0)
    center1 = torch.where(collapse, pooled, mean1)
    return ((center1-center0).clamp_min(0).half()[:, None], center0.half()[:, None],
            empty, crossed)


@torch.no_grad()
def joint_refit_value(sourceNative, bit_codeI32, old_scaleF16, old_minF16,
                      sign_codeI32, old_alphaF16, *, rounds=2):
    """Return (scale[128,1], minimum[128,1], alpha[N], JSON statistics).

    One already-selected V head with exactly one stored sign stage only.
    sourceNative: FP16/FP32[N,128] in native coordinates; bit_codeI32:
    int32[128,N/32] from state.v_code[0,head]; scale/min: physical FP16[128,1]
    after all-frame fold; sign_codeI32: int32[N,4]; alpha: FP16[N]. N must be
    positive and divisible by32, matching current Round8Bank. Inputs are not
    mutated or retained. This function does not check a bank's selection mask;
    the future caller must reject missing/depth2 V stages and non-binary V.
    """
    source, code, scale, minimum, sign_code, alpha = (
        sourceNative, bit_codeI32, old_scaleF16, old_minF16, sign_codeI32, old_alphaF16)
    if source.ndim != 2 or source.shape[-1] != 128 or source.dtype not in (torch.float16, torch.float32):
        raise ValueError('Native FP16/FP32[N,128] source required')
    n = len(source)
    if n <= 0 or n % 32 or rounds not in (2, 3):
        raise ValueError('Positive complete32-token words and exactly2/3 rounds required')
    if tuple(code.shape) != (128, n//32) or code.dtype != torch.int32 or tuple(sign_code.shape) != (n, 4) or sign_code.dtype != torch.int32:
        raise ValueError('Actual int32 channel-bit[128,N/32] and residual-sign[N,4] fields required')
    if tuple(scale.shape) != (128, 1) or tuple(minimum.shape) != (128, 1) or tuple(alpha.shape) != (n,) or any(t.dtype != torch.float16 for t in (scale, minimum, alpha)):
        raise ValueError('Existing FP16 scale/min[128,1] and alpha[N] fields required')
    if any(t.device != source.device for t in (code, scale, minimum, sign_code, alpha)):
        raise ValueError('All actual fields must share one device')
    if any(not bool(torch.isfinite(t).all()) for t in (source, scale, minimum, alpha)) or bool((scale < 0).any()) or bool((alpha < 0).any()):
        raise ValueError('Finite native input/fields and nonnegative initial S/alpha required')
    bits, sign = _unpack_bits(code, n), _unpack_sign(sign_code)
    sd = source.double()
    new_scale, new_minimum, new_alpha = scale.clone(), minimum.clone(), alpha.clone()
    original_error = _errors(sd, bits, new_scale, new_minimum, sign, new_alpha)
    current_error = original_error
    trace = []
    for iteration in range(rounds):
        before_frame = current_error
        candidate_scale, candidate_minimum, empty, crossed = _frame_proposal(sd, bits, sign, new_alpha)
        frame_error = _errors(sd, bits, candidate_scale, candidate_minimum, sign, new_alpha)
        old_columns, candidate_columns = before_frame.sum(0), frame_error.sum(0)
        finite_frame = (torch.isfinite(candidate_scale[:, 0]) & torch.isfinite(candidate_minimum[:, 0])
                        & torch.isfinite(candidate_columns))
        accept_frame = finite_frame & (candidate_columns < old_columns)
        new_scale = torch.where(accept_frame[:, None], candidate_scale, new_scale)
        new_minimum = torch.where(accept_frame[:, None], candidate_minimum, new_minimum)
        current_error = torch.where(accept_frame[None], frame_error, before_frame)
        after_frame = current_error
        actual_base = _base(bits, new_scale, new_minimum)
        real_alpha = ((sd - actual_base.double()) * sign.double()).mean(-1).clamp_min(0)
        candidate_alpha = real_alpha.half()
        alpha_error = _errors(sd, bits, new_scale, new_minimum, sign, candidate_alpha)
        old_rows, candidate_rows = after_frame.sum(-1), alpha_error.sum(-1)
        finite_alpha = torch.isfinite(candidate_alpha) & torch.isfinite(candidate_rows)
        accept_alpha = finite_alpha & (candidate_rows < old_rows)
        new_alpha = torch.where(accept_alpha, candidate_alpha, new_alpha)
        current_error = torch.where(accept_alpha[:, None], alpha_error, after_frame)
        trace.append(dict(round=iteration+1, source_sse_before=_finite_sum(before_frame),
            source_sse_after_frame=_finite_sum(after_frame), source_sse_after_alpha=_finite_sum(current_error),
            accepted_frame_channels=int(accept_frame.sum()), accepted_alpha_rows=int(accept_alpha.sum()),
            empty_class_channels=int(empty.sum()), crossed_centers_count_pooled=int(crossed.sum()),
            rejected_nonfinite_frame_channels=int((~finite_frame).sum()), rejected_nonfinite_alpha_rows=int((~finite_alpha).sum()),
            max_finite_channel_SSE_increase_frame_phase=_positive_increase(old_columns, after_frame.sum(0)),
            max_finite_row_SSE_increase_alpha_phase=_positive_increase(old_rows, current_error.sum(-1))))
    physical_before = sum(t.numel()*t.element_size() for t in (scale, minimum, alpha))
    physical_after = sum(t.numel()*t.element_size() for t in (new_scale, new_minimum, new_alpha))
    stats = dict(schema='trajkv.round8.value-frame-alpha-fixed-fields.v1', rows=n, channels=128, rounds=rounds,
        source_sse_before=_finite_sum(original_error), source_sse_after=_finite_sum(current_error),
        source_base_sse_after=_finite_sum((sd-_base(bits, new_scale, new_minimum).double()).square()),
        old_nonfinite_decode_values=int((~torch.isfinite(original_error)).sum()),
        new_nonfinite_decode_values=int((~torch.isfinite(current_error)).sum()), trace=trace,
        changed_frame_channels=int(((new_scale != scale) | (new_minimum != minimum)).sum()),
        changed_alpha_rows=int((new_alpha != alpha).sum()),
        max_finite_total_SSE_increase=_positive_increase(original_error.sum().reshape(1), current_error.sum().reshape(1)),
        physical_optimized_fields_bytes_before=physical_before, physical_optimized_fields_bytes_after=physical_after,
        added_retained_tensor_bytes=0, code_fields_mutated=False, sign_recode=False, frame='native binary/all',
        decode='Q16(Q16(Q16(S*b)+m)+alpha*fixed_sign); no FMA, all stored fields FP16',
        frame_proposal='ordered two-class unweighted native LS; count-weighted pool if crossed; collapse to observed mean if empty',
        alpha_proposal='max(mean_c(fixed_sign*(native_source-actual_FP16_base)),0)',
        acceptance='strict actual-hard-SSE decrease per frame-channel or alpha-row; old physical fields win ties',
        factor_policy='already-folded physical metadata in/out, no factor/offset reapplied',
        guarantee='total native reconstruction SSE nonincreasing across phases; not individual rows/channels across both phases; no attention/task guarantee',
        limitation='one real proposal per coordinate block then FP16 rounding and acceptance; no FP16/global optimum claim',
        scratch_scope='temporary unpacked bits/sign/native error tensors and extra encoding compute; source not retained')
    return new_scale.contiguous(), new_minimum.contiguous(), new_alpha.contiguous(), stats


def _pack(code):
    grouped = code.long().reshape(*code.shape[:-1], -1, 32)
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    return (grouped << shifts).sum(-1).int().contiguous()


def cpu_check():
    torch.set_num_threads(1); torch.manual_seed(20261118)
    cases = []
    def run(name, source, bits, sign, scale, minimum, alpha, rounds=2):
        code, sign_code = _pack(bits.T), _pack(sign >= 0)
        before_inputs = [t.clone() for t in (source, code, scale, minimum, sign_code, alpha)]
        before = _errors(source.double(), bits.float(), scale, minimum, sign.float(), alpha)
        ns, nm, na, st = joint_refit_value(source, code, scale, minimum, sign_code, alpha, rounds=rounds)
        after = _errors(source.double(), bits.float(), ns, nm, sign.float(), na)
        assert all(torch.equal(a, b) for a, b in zip(before_inputs, (source, code, scale, minimum, sign_code, alpha)))
        assert float(after.sum()) <= float(before.sum()) and bool((ns >= 0).all()) and bool((na >= 0).all())
        assert st['physical_optimized_fields_bytes_before'] == st['physical_optimized_fields_bytes_after']
        assert all(t['max_finite_channel_SSE_increase_frame_phase'] == 0 and t['max_finite_row_SSE_increase_alpha_phase'] == 0 for t in st['trace'])
        cases.append(dict(name=name, stats=st))
        return ns, nm, na, st
    n = 32
    row = torch.arange(n)[:, None]
    bits = (row % 2).expand(n, 128).bool()
    sign = ((row//2 % 2)*2-1).expand(n, 128).float()
    scale, minimum, alpha = torch.ones(128, 1).half(), torch.zeros(128, 1).half(), torch.full((n,), .5).half()
    source = (1+2*bits.float()+1.25*sign).half()
    ns, nm, na, st = run('two_centers_and_alpha_exact', source, bits, sign, scale, minimum, alpha)
    assert st['source_sse_after'] == 0 and bool((ns == 2).all()) and bool((nm == 1).all()) and bool((na == 1.25).all())
    source = (3-4*bits.float()).half()
    ns, nm, na, st = run('crossed_centers_order_constraint', source, bits, sign, scale, minimum, torch.zeros(n).half(), 3)
    assert st['trace'][0]['crossed_centers_count_pooled'] == 128 and bool((ns == 0).all())
    # Unequal class counts require the weighted mean, not (mean0+mean1)/2.
    uneven = torch.zeros(n, 128); uneven[0] = 1
    cs, cm, _, crossed = _frame_proposal((10-12*uneven).double(), uneven, sign, torch.zeros(n).half())
    assert bool(crossed.all()) and bool((cs == 0).all()) and bool((cm == 9.625).all())
    empty_bits = torch.zeros(n, 128, dtype=torch.bool); empty_bits[:, 64:] = True
    source = (2+.75*sign).half()
    ns, nm, na, st = run('empty_class_collapses_to_present_mean', source, empty_bits, sign, scale, minimum, torch.full((n,), .25).half())
    assert st['source_sse_after'] == 0 and st['trace'][0]['empty_class_channels'] == 128 and bool((ns == 0).all())
    # The unused scale/level is not arbitrarily rewritten when decode ties.
    scale = torch.full((128, 1), 3.).half(); minimum = torch.full((128, 1), 2.).half(); minimum[64:] = -1.
    ns, nm, na, st = run('empty_class_tie_preserves_original_bits', torch.full((n, 128), 2.).half(), empty_bits, sign, scale, minimum, torch.zeros(n).half())
    assert torch.equal(ns, scale) and torch.equal(nm, minimum) and st['changed_alpha_rows'] == 0
    ns, nm, na, st = run('fixed_sign_alpha_nonnegative_boundary', (-3*sign).half(), bits, sign, torch.zeros(128, 1).half(), torch.zeros(128, 1).half(), torch.ones(n).half())
    assert bool((na == 0).all())
    # Nested physical additions differ from an unrounded three-term sum.
    def half_scalar(value):
        return struct.unpack('<e', struct.pack('<e', value))[0]
    scale = torch.ones(128, 1).half(); minimum = torch.full((128, 1), 2**-11).half(); alpha = torch.full((n,), 2**-11).half()
    all_bits, all_sign = torch.ones(n, 128), torch.ones(n, 128)
    decoded = _decoded(all_bits, scale, minimum, all_sign, alpha)
    scalar = half_scalar(half_scalar(half_scalar(1.) + 2**-11) + 2**-11)
    assert scalar == 1. and bool((decoded == scalar).all()) and half_scalar(1.+2**-10) != scalar
    # Independent scalar oracle for arbitrary physical decode values.
    for _ in range(64):
        b = int(torch.randint(2, ()).item()); s = 1 if int(torch.randint(2, ()).item()) else -1
        S, m, a = (float(torch.rand(()).half()), float(torch.randn(()).half()), float(torch.rand(()).half()))
        physical = half_scalar(half_scalar(half_scalar(S*b)+m)+a*s)
        vector = _decoded(torch.full((1, 128), b), torch.full((128, 1), S).half(),
            torch.full((128, 1), m).half(), torch.full((1, 128), s), torch.tensor([a]).half())
        assert bool((vector == physical).all())
    for n in (32, 256, 1792):
        for rounds in (2, 3):
            bits = torch.randint(2, (n, 128)).bool(); sign = torch.randint(2, (n, 128)).float()*2-1
            source = torch.randn(n, 128).half() if rounds == 2 else torch.randn(n, 128)
            run(f'random_native_N{n}_rounds{rounds}', source, bits, sign, torch.rand(128, 1).half(), torch.randn(128, 1).half(), torch.rand(n).half(), rounds)
    # Finite original fields may have an overflowing decoded sum; a finite
    # hard candidate can repair it without pretending the old SSE was finite.
    n = 32; bits = torch.ones(n, 128); sign = torch.ones(n, 128)
    ns, nm, na, st = run('overflowing_old_decode_can_be_repaired', torch.full((n, 128), 3.).half(), bits, sign,
        torch.full((128, 1), 65504.).half(), torch.full((128, 1), 65504.).half(), torch.zeros(n).half())
    assert st['source_sse_before'] is None and st['source_sse_after'] == 0
    invalid = False
    try:
        joint_refit_value(torch.zeros(n, 128), _pack(bits.T), -torch.ones(128, 1).half(),
            torch.zeros(128, 1).half(), _pack(sign >= 0), torch.zeros(n).half())
    except ValueError:
        invalid = True
    assert invalid
    return dict(status='PASS', schema='trajkv.round8.value-frame-alpha-cpu-check.v1', cases=cases,
        scalar_physical_decode_oracle_cases=65, weighted_pool_unequal_counts=True,
        negative_initial_scale_rejected=True, original_input_fields_unchanged=True,
        scope='CPU synthetic physical-field arithmetic only; no bank/model/GPU/task/timing measurement',
        all_existing_field_bytes_unchanged=True, added_retained_tensor_bytes=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description='CPU-only native binary/all V frame-alpha arithmetic prototype')
    parser.add_argument('--check-output', type=Path, required=True)
    args = parser.parse_args()
    if args.check_output.exists():
        raise ValueError('Use a new immutable CPU evidence output')
    result = cpu_check()
    args.check_output.parent.mkdir(parents=True, exist_ok=True)
    with args.check_output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='PASS', cases=len(result['cases']), GPU_used=False,
        source_sha256=result['source_sha256'])))


if __name__ == '__main__':
    main()
