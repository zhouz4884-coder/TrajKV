"""CPU codec/byte controls and GPU dense/old-Mixed reader checks.

Run --cpu-only locally. Without that flag, CUDA is mandatory: a CPU result
cannot be mistaken for the production Triton reader check.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
import torch
from round6_depth_codec import encode_sign_depth, plan_depth_stages


def gpu_checks():
    from affine_rate_bank import fold
    from check_fused_progressive import errors
    from fast_wht import torch_wht
    from fused_progressive import decode_last_generic
    from quantizers import binary_centroids
    from mixed_residual_attention import attention as mixed_attention
    from round6_depth_attention import attention
    torch.manual_seed(253)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    patterns = [[], [('v', 3, 0), ('v', 1, 0)], [('k', 3, 0), ('k', 1, 0)],
                [('k', 1, 0), ('v', 1, 0)],
                [('v', 3, 0), ('k', 1, 0), ('v', 0, 0), ('k', 3, 0)],
                [('v', 3, 0), ('v', 3, 1)], [('k', 1, 0), ('k', 1, 1)],
                [('k', 1, 0), ('v', 1, 0), ('v', 1, 1), ('k', 1, 1)],
                [('v', 3, 0), ('k', 1, 0), ('v', 0, 0), ('k', 3, 0),
                 ('v', 3, 1), ('k', 3, 1), ('v', 0, 1)]]
    with torch.inference_mode():
        for nv, nq in ((256, 1), (768, 11), (1792, 1)):
            q = torch.randn(1, nq, 28, 128, device='cuda', dtype=torch.float16).transpose(1, 2)
            nt = 41 + nq
            state = SimpleNamespace(
                direction_code=torch.randint(0, 256, (1, 4, nv, 32), device='cuda', dtype=torch.uint8),
                direction_gain=(.5 + torch.rand(1, 4, nv, 1, device='cuda')).half(),
                direction_mean=.1 * torch.randn(1, 4, 1, 128, device='cuda', dtype=torch.float16),
                visual_pos=torch.arange(0, 2 * nv, 2, device='cuda'),
                text_pos=torch.cat((torch.arange(1, 82, 2, device='cuda'), torch.arange(2 * nv, 2 * nv + nq, device='cuda'))),
                text_length=nt, length=2 * nv + nq,
                text_k=torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.float16),
                text_v=torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.float16))
            source_v = torch.randn(1, 4, nv, 128, device='cuda', dtype=torch.float16)
            source_k = torch.randn_like(source_v)
            rotated_source = torch_wht(source_k.float() - source_k.float().mean(-2, keepdim=True).half().float())
            book = torch.randn(256, 4, device='cuda', dtype=torch.float16)
            base_k = (book[state.direction_code.long()].reshape(1, 4, nv, 128).float() * state.direction_gain.float()).half()
            beta = torch.tensor([-.6, -.2, .2, .6], device='cuda')
            for window in (128, nv):
                rec = binary_centroids(source_v.transpose(2, 3).contiguous(), group_size=window)
                gain = (.85 + .3 * torch.rand(4, 8, device='cuda')).repeat_interleave(16, -1)[None, :, :, None]
                offset = (.1 * torch.randn(4, 8, device='cuda')).repeat_interleave(16, -1)[None, :, :, None]
                rec = fold(rec, 1, gain, offset)
                state.v_code, state.v_scale, state.v_min = rec
                base_v = decode_last_generic(*rec, window, 1).transpose(2, 3).contiguous()
                for selected in patterns:
                    lookup = torch.full((2, 4, 2), -1, device='cuda', dtype=torch.int32)
                    dense_k, dense_v = base_k.clone(), base_v.clone()
                    payload = 0
                    if selected:
                        signs = [None] * len(selected)
                        amplitudes = [None] * len(selected)
                        units = sorted({(axis, head) for axis, head, _ in selected})
                        for axis, head in units:
                            depth = sum(a == axis and h == head for a, h, _ in selected)
                            source = rotated_source[0, head] if axis == 'k' else source_v[0, head]
                            base = base_k[0, head] if axis == 'k' else base_v[0, head]
                            codes, alphas, decoded, _ = encode_sign_depth(source, base, depth)
                            for stage in range(depth):
                                slot = selected.index((axis, head, stage))
                                signs[slot], amplitudes[slot] = codes[stage], alphas[stage]
                                lookup[0 if axis == 'k' else 1, head, stage] = slot
                            (dense_k if axis == 'k' else dense_v)[0, head] = decoded
                        state.depth_residual_code = torch.stack(signs).contiguous()
                        state.depth_residual_alpha = torch.stack(amplitudes).contiguous()
                        payload = state.depth_residual_code.numel() * 4 + state.depth_residual_alpha.numel() * 2
                        if payload != len(selected) * nv * 18:
                            raise RuntimeError('Depth residual pool contains padding/unselected payload')
                    else:
                        for name in ('depth_residual_code', 'depth_residual_alpha'):
                            if hasattr(state, name):
                                delattr(state, name)
                    state.depth_residual_lookup = lookup
                    actual = attention(q, state, book, window, beta)
                    old_equal = None
                    if all(stage == 0 for _, _, stage in selected):
                        state.mixed_residual_lookup = lookup[:, :, 0].contiguous()
                        if selected:
                            state.mixed_residual_code = state.depth_residual_code
                            state.mixed_residual_alpha = state.depth_residual_alpha
                        else:
                            for name in ('mixed_residual_code', 'mixed_residual_alpha'):
                                if hasattr(state, name):
                                    delattr(state, name)
                        old_equal = bool(torch.equal(actual, mixed_attention(q, state, book, window, beta)))
                        if not old_equal:
                            raise RuntimeError('Depth zero/one-stage path changed old MixedResidualAttention output')
                    zv = torch_wht(q).half().float() @ dense_k.repeat_interleave(7, 1).float().transpose(-1, -2) / 128 ** .5
                    zv += (q.float() * state.direction_mean.repeat_interleave(7, 1).float()).sum(-1, keepdim=True) / 128 ** .5
                    zv += beta.repeat_interleave(7)[None, :, None, None]
                    zt = q.float() @ state.text_k[:, :, :nt].repeat_interleave(7, 1).float().transpose(-1, -2) / 128 ** .5
                    z = torch.cat((zv, zt), -1)
                    positions = torch.cat((state.visual_pos, state.text_pos))
                    qpos = torch.arange(state.length - nq, state.length, device='cuda')
                    z.masked_fill_(positions[None, None, None, :] > qpos[None, None, :, None], -torch.inf)
                    values = torch.cat((dense_v, state.text_v[:, :, :nt]), 2).repeat_interleave(7, 1)
                    reference = (z.softmax(-1) @ values.float()).half()
                    err = errors(actual, reference)
                    rows.append({'nv': nv, 'nq': nq, 'value_window': window, 'selected_axis_head_stages': selected,
                        'compact_residual_bytes': payload, 'old_mixed_depth0_or_1_bitwise_equal': old_equal,
                        'stage_lookup_bytes': lookup.numel() * lookup.element_size(), 'error': err,
                        'passed': err['max_abs'] < .003 and err['relative_l2'] < .003})
    return rows

def cpu_controls():
    torch.manual_seed(20261115)
    rows = []
    for magnitude in (1., 128., 1024.):
        source = magnitude * torch.randn(3, 128, 128)
        base = (source + .17 * magnitude * torch.randn_like(source)).half()
        for depth in (0, 1, 2):
            codes, alphas, actual, distortion = encode_sign_depth(source, base, depth)
            expected = base
            for stage in range(depth):
                error = source.float() - expected.float()
                expected_alpha = error.abs().mean(-1).half()
                if not torch.equal(alphas[stage], expected_alpha):
                    raise RuntimeError('Stage amplitude did not use actual previous FP16 reconstruction')
                shifts = torch.arange(32, dtype=torch.int64)
                bits = ((codes[stage].long()[..., None] >> shifts) & 1).reshape_as(source)
                if not torch.equal(bits.bool(), error >= 0):
                    raise RuntimeError('Packed sign words failed round trip, including the signed high bit')
                expected = (expected.float() + (bits.float() * 2 - 1) * expected_alpha.float()[..., None]).half()
            if not torch.equal(actual, expected):
                raise RuntimeError('Sequential FP16 packed decode disagrees with explicit reconstruction')
            payload = sum(code.numel() * code.element_size() + alpha.numel() * alpha.element_size()
                          for code, alpha in zip(codes, alphas))
            if payload != depth * 3 * 128 * 18:
                raise RuntimeError('Residual codec retained nonexistent stages/padding')
            rows.append(dict(magnitude=magnitude, depth=depth, payload_bytes=payload,
                squared_error_by_depth=distortion))
        if not all(rows[-1]['squared_error_by_depth'][i + 1] < rows[-1]['squared_error_by_depth'][i] for i in (0, 1)):
            raise RuntimeError('Random same-input sign refinement did not lower measured reconstruction error')
    # An intentional unrounded second-stage control must differ: otherwise the
    # random fixture would not detect the principal implementation mistake.
    codes, alphas, decoded, _ = encode_sign_depth(source, base, 2)
    error0 = source.float() - base.float()
    hypothetical = base.float() + torch.where(error0 >= 0, 1., -1.) * alphas[0].float()[..., None]
    wrong_alpha2 = (source.float() - hypothetical).abs().mean(-1).half()
    rounding_sensitive = int((wrong_alpha2 != alphas[1]).sum())
    if rounding_sensitive == 0:
        raise RuntimeError('FP16 control does not distinguish the wrong unrounded second-stage path')
    zero = torch.zeros(2, 128, dtype=torch.float16)
    zcodes, zalphas, zdecoded, _ = encode_sign_depth(zero, zero, 2)
    if not torch.equal(zdecoded, zero) or any(bool(alpha.any()) for alpha in zalphas):
        raise RuntimeError('Zero-error residual failed')
    layers, heads = 28, 4
    all_units = [(axis, layer, head) for axis in ('k', 'v') for layer in range(layers) for head in range(heads)]
    first = [item for item in all_units if item[0] == 'k'][:24] + [item for item in all_units if item[0] == 'v'][:20]
    ranked = first + [unit for unit in all_units if unit not in first]
    budget_rows = []
    for nv in (128, 256, 768, 1792, 2048, 4096):
        full_bytes = layers * heads * nv * 128 * 4
        base_bytes = (layers * heads * nv * 50 + layers * heads * 128 * 2
                      + 4 * layers * heads * 128 + 2048 + 8 * nv)
        plans = []
        for policy in ('incumbent', 'v_pair', 'v_mixed'):
            plan = plan_depth_stages(layers=layers, heads=heads, visual_tokens=nv,
                visual_full_bytes=full_bytes, base_bytes=base_bytes, factor_bytes=1344,
                selected_units=ranked, depth_policy=policy)
            stages = plan['selected_stages']
            if len(plan['ranking_constant']) != 448 or len(plan['policy_constant']) != 5:
                raise RuntimeError('Named depth policy constants are not the same actual format size')
            if plan['predicted_bytes'] > plan['allowed_bytes']:
                raise RuntimeError('Complete owner exceeds actual byte cap')
            if plan['allowed_bytes'] - plan['predicted_bytes'] >= plan['stage_bytes']:
                raise RuntimeError('Named budget control left an affordable stage unexplained')
            expected_k = {unit for unit in plan['incumbent_selected_units'] if unit[0] == 'k'}
            actual_k = {stage[:3] for stage in stages if stage[0] == 'k'}
            if actual_k != expected_k or any(stage[3] != 0 for stage in stages if stage[0] == 'k'):
                raise RuntimeError('Named policy changed the frozen incumbent K subset/depth')
            if any(stage[:3] + (0,) not in stages for stage in stages if stage[3] == 1):
                raise RuntimeError('Second stage lacks its prefix')
            plans.append(plan)
            budget_rows.append(dict(nv=nv, policy=policy, stages=len(stages),
                K_stages=sum(item[0] == 'k' for item in stages),
                V_stages=sum(item[0] == 'v' for item in stages),
                V_units=len({item[:3] for item in stages if item[0] == 'v'}),
                complete_bytes=plan['predicted_bytes'], lookup_bytes=plan['lookup_bytes'],
                actual_budget_bits=plan['actual_budget_bits']))
        if len({plan['predicted_bytes'] for plan in plans}) != 1 or len({len(plan['selected_stages']) for plan in plans}) != 1:
            raise RuntimeError('Named depth arms do not have identical complete byte/stage budgets')
    # Deliberately place second-stage entries and head/axis slots out of order.
    explicit = [('v', 3, 1, 0), ('k', 1, 2, 0), ('v', 3, 1, 1), ('k', 1, 2, 1)]
    common = dict(layers=layers, heads=heads, visual_tokens=1792,
                  visual_full_bytes=layers * heads * 1792 * 128 * 4,
                  base_bytes=layers * heads * 1792 * 50 + layers * heads * 128 * 6 + 2048 + 8 * 1792,
                  factor_bytes=1344)
    for limit in range(5):
        plan = plan_depth_stages(**common, selected_stages=explicit, stage_limit=limit)
        if plan['selected_stages'] != explicit[:limit] or len(plan['ranking_constant']) != 8:
            raise RuntimeError('Explicit legal priority prefix was not shortened exactly/no-padding')
    invalid = [[('k', 0, 0, 1)], [('k', 0, 0, 0), ('k', 0, 0, 0)], [('v', 28, 0, 0)], [('v', 0, 0, 2)]]
    rejected = 0
    for bad in invalid:
        try:
            plan_depth_stages(**common, selected_stages=bad)
        except ValueError:
            rejected += 1
    if rejected != len(invalid):
        raise RuntimeError('An illegal depth/prefix/range/duplicate was accepted')
    # Parse exactly the production sources even on a local CPU host without Triton.
    for name in ('round6_depth_bank.py', 'round6_depth_attention.py', 'round6_depth_codec.py', 'round6_depth_check.py'):
        ast.parse(Path(__file__).with_name(name).read_text())
    return dict(passed=True, scope='CPU math/format controls; GPU Triton reader remains a separate required check',
        codec_cases=rows, rounding_sensitive_second_stage_tokens=rounding_sensitive,
        budget_cases=budget_rows, invalid_inputs_rejected=rejected,
        explicit_prefix_limits_checked=5, sources_ast_parsed=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cpu = cpu_controls()
    rows = [] if args.cpu_only else gpu_checks()
    passed = cpu['passed'] and all(row['passed'] for row in rows)
    report = dict(status='PASS' if passed else 'FAIL', cpu_controls=cpu,
        gpu_executed=not args.cpu_only, cases=len(rows), results=rows,
        scope=('CPU integer codec, real sequential FP16 rounding, legal prefix and byte-budget controls only; no GPU reader/task claim'
               if args.cpu_only else 'CPU controls and same-representation actual GPU reader arithmetic; no task accuracy/efficiency claim'),
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ('round6_depth_bank.py', 'round6_depth_attention.py', 'round6_depth_codec.py', 'round6_depth_check.py')})
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status=report['status'], cases=len(rows), gpu_executed=report['gpu_executed'])), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], cases=len(rows), gpu_executed=report['gpu_executed'])), flush=True)
    if not passed:
        raise RuntimeError('Depth codec/reader control failed')


if __name__ == '__main__':
    main()
