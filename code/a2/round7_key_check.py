"""Round7 CPU codec/owner checks and real GPU nibble/dense/legacy reader checks.

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
from round7_key_codec import (encode_sign_depth, plan_depth_stages, encode_key, decode_key,
                              pack_nibbles, unpack_nibbles, key_amplitude)


def gpu_checks():
    from affine_rate_bank import fold
    from check_fused_progressive import errors
    from fast_wht import torch_wht
    from fused_progressive import decode_last_generic
    from quantizers import binary_centroids
    from mixed_residual_attention import attention as mixed_attention
    from round6_depth_attention import attention as depth_attention
    from round7_key_attention import attention
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
        from itertools import product
        for words, (nv, nq) in product((16, 256), ((256, 1), (768, 11), (1792, 1))):
            q = torch.randn(1, nq, 28, 128, device='cuda', dtype=torch.float16).transpose(1, 2)
            nt = 41 + nq
            state = SimpleNamespace(
                direction_code=torch.randint(0, words, (1, 4, nv, 32), device='cuda', dtype=torch.uint8),
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
            book = torch.randn(words, 4, device='cuda', dtype=torch.float16)
            if words == 16:
                state.direction_code = pack_nibbles(state.direction_code)
            expected_index_bytes = 4 * nv * (16 if words == 16 else 32)
            if state.direction_code.numel() != expected_index_bytes:
                raise RuntimeError('Actual key payload differs from declared rate')
            base_k = decode_key(state.direction_code, state.direction_gain, book)
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
                    depth_equal = None if words == 16 else bool(torch.equal(actual, depth_attention(q,state,book,window,beta)))
                    if depth_equal is False:
                        raise RuntimeError('G4_256 reader changed Round6 depth output')
                    old_equal = None
                    if words == 256 and all(stage == 0 for _, _, stage in selected):
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
                    rows.append({'book_words': words, 'key_index_bytes': expected_index_bytes, 'round6_depth_bitwise_equal': depth_equal, 'nv': nv, 'nq': nq, 'value_window': window, 'selected_axis_head_stages': selected,
                        'compact_residual_bytes': payload, 'old_mixed_depth0_or_1_bitwise_equal': old_equal,
                        'stage_lookup_bytes': lookup.numel() * lookup.element_size(), 'error': err,
                        'passed': err['max_abs'] < .003 and err['relative_l2'] < .003})
    return rows

def key_cpu_controls():
    from round7_key_artifacts import factor_values_hash, verify_factor_inheritance
    torch.manual_seed(20261116)
    rows = []
    raw = torch.randn(1, 2, 37, 128)
    factor = torch.tensor([.93, 1.08])[None, :, None, None]
    for words in (16, 256):
        book = torch.randn(words, 4).half()
        code, gain = encode_key(raw, book, gain_factor=factor)
        decoded = decode_key(code, gain, book)
        expected_bytes = 2 * 37 * (16 if words == 16 else 32)
        if code.numel() != expected_bytes:
            raise RuntimeError('Code owner does not implement the declared K1/K2 rate')
        ids = unpack_nibbles(code) if words == 16 else code
        if words == 16 and not torch.equal(pack_nibbles(ids), code):
            raise RuntimeError('Nibble round trip changed indices')
        norm = raw.norm(dim=-1, keepdim=True)
        parts = (raw / (norm / 128 ** .5).clamp_min(1e-20)).reshape(-1, 4)
        centers = book.float()
        distance = parts.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * parts @ centers.T
        old_ids = distance.argmin(-1).byte().reshape(1, 2, 37, 32)
        template = centers[old_ids.long()].reshape(raw.shape)
        old_gain = ((norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half().float() * factor).half()
        if not torch.equal(ids, old_ids) or not torch.equal(gain, old_gain):
            raise RuntimeError('Default G4 assignment/norm rule changed legacy arithmetic')
        ls = key_amplitude(raw, template, 'ls')
        formula = (raw * template).sum(-1, keepdim=True) / template.square().sum(-1, keepdim=True)
        if not torch.equal(ls, formula):
            raise RuntimeError('LS amplitude does not implement per-vector dot(r,t)/dot(t,t)')
        norm_error = (raw - template * key_amplitude(raw, template, 'norm')).square().sum()
        ls_error = (raw - template * ls).square().sum()
        if float(ls_error) > float(norm_error) + 1e-5:
            raise RuntimeError('Unrounded Euclidean LS amplitude failed its mathematical control')
        rows.append(dict(words=words, actual_index_bytes=code.numel(), expected_index_bytes=expected_bytes,
                         legacy_norm_indices_gain_bitwise_equal=True, norm_error=float(norm_error), ls_error=float(ls_error)))
    # Same frozen stage list isolates the actual byte saving, including book.
    nl, nh, nv = 28, 4, 1792
    common = dict(layers=nl, heads=nh, visual_tokens=nv, visual_full_bytes=nl * nh * nv * 512,
                  factor_bytes=1344, selected_stages=[('k', 0, 0, 0), ('v', 1, 1, 0)])
    plans = [plan_depth_stages(**common, base_bytes=nl * nh * nv * (kb + 18) + nl * nh * 128 * 6 + words * 8 + nv * 8)
             for kb, words in ((32, 256), (16, 16))]
    expected_saving = nl * nh * nv * 16 + (256 - 16) * 8
    if plans[0]['predicted_bytes'] - plans[1]['predicted_bytes'] != expected_saving:
        raise RuntimeError('Complete-owner K1 saving is inconsistent with actual packed payload/book')
    # An artifact declaration cannot silently permit a modified inherited factor.
    inherited = dict(key_head_factors=[[1.]], value_head_factors=[[1.]], value_center_offsets=[[0.]],
                     key_gain_mode='centered', value_group_size='all', parameter_count=3, encoder_book_sha256='b' * 64)
    inherited['round7_inheritance'] = dict(schema='trajkv_round7_explicit_factor_inheritance_v1',
        source_factor_sha256='a' * 64, source_encoder_book_sha256='c' * 64,
        target_encoder_book_sha256='b' * 64, inherited_values_sha256=factor_values_hash(inherited))
    verify_factor_inheritance(inherited)
    inherited['value_head_factors'][0][0] = 1.1
    try:
        verify_factor_inheritance(inherited)
    except ValueError:
        pass
    else:
        raise RuntimeError('Modified inherited numerical factor passed its declared hash')
    for assignment in ('bcd', 'rank1_bcd'):
        u = torch.randn(1, 2, 1, 128)
        u = u / u.norm(dim=-1, keepdim=True)
        gamma = torch.ones(1, 2, 1, 1) * 3
        metric = dict(u=u, gamma=gamma) if assignment == 'rank1_bcd' else {}
        code0, gain0 = encode_key(raw[:, :, :3], book, amplitude='ls')
        code1, gain1 = encode_key(raw[:, :, :3], book, amplitude='ls', assignment=assignment, chunk=3, **metric)
        def objective(code, gain):
            error = raw[:, :, :3] - decode_key(code, gain, book).float()
            result = error.square().sum(-1)
            return result + gamma.squeeze(-1) * (error * u).sum(-1).square() if metric else result
        if not bool((objective(code1, gain1) <= objective(code0, gain0) + 1e-4).all()):
            raise RuntimeError('Actual-FP16 BCD increased its declared objective')
    return dict(passed=True, payload_and_ls_cases=rows, same_stage_complete_byte_saving=expected_saving,
                inheritance_tamper_rejected=True, fp16_bcd_monotonic_controls=2)


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
    for name in ('round7_key_bank.py', 'round7_key_attention.py', 'round7_key_codec.py', 'round7_key_check.py', 'round7_key_artifacts.py'):
        ast.parse(Path(__file__).with_name(name).read_text())
    key_controls = key_cpu_controls()
    return dict(passed=True, key_controls=key_controls, scope='CPU math/format controls; GPU Triton reader remains a separate required check',
        codec_cases=rows, rounding_sensitive_second_stage_tokens=rounding_sensitive,
        budget_cases=budget_rows, invalid_inputs_rejected=rejected,
        explicit_prefix_limits_checked=5, sources_ast_parsed=5)



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
            ('round7_key_bank.py', 'round7_key_attention.py', 'round7_key_codec.py', 'round7_key_check.py', 'round7_key_artifacts.py')})
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status=report['status'], cases=len(rows), gpu_executed=report['gpu_executed'])), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], cases=len(rows), gpu_executed=report['gpu_executed'])), flush=True)
    if not passed:
        raise RuntimeError('Depth codec/reader control failed')


if __name__ == '__main__':
    main()
