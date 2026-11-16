"""Tail-word CPU controls and real 32-head LLaVA-size CUDA reader checks."""
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

from quantizers import binary_centroids
from round7_tail_key_codec import (binary_centroids_tail, decode_value_tail,
    value_storage_account, encode_key, decode_key, encode_sign_depth)
from round7_key_layout import plan_key_stages


def cpu_controls():
    torch.manual_seed(20261116)
    cases = []
    for nv in (1, 31, 32, 33, 128, 2438, 2448):
        source = torch.randn(1, 2, 128, nv).half() + 2
        source[..., -1] += 17
        rec = binary_centroids_tail(source, group_size=nv)
        decoded = decode_value_tail(*rec, nv, nv)
        # Independent unpacked binary Lloyd reference over only the real N.
        x = source.float()
        c = x > (x.amin(-1, keepdim=True) + x.amax(-1, keepdim=True)) / 2
        for step in range(3):
            n0, n1 = (~c).sum(-1, keepdim=True), c.sum(-1, keepdim=True)
            a = (x * (~c)).sum(-1, keepdim=True) / n0.clamp_min(1)
            b = (x * c).sum(-1, keepdim=True) / n1.clamp_min(1)
            a = torch.where(n0 > 0, a, b)
            b = torch.where(n1 > 0, b, a)
            if step < 2:
                c = x > (a + b) / 2
        expected_scale, expected_min = (b - a).half(), a.half()
        expected = ((c.half() * expected_scale).half() + expected_min).half()
        if not torch.equal(rec[1], expected_scale) or not torch.equal(rec[2], expected_min) or not torch.equal(decoded, expected):
            raise RuntimeError('Padded bits influenced real-N centroid fitting or FP16 decode')
        unused = (-nv) % 32
        if unused and not bool((rec[0][..., -1].long() >> (32 - unused) == 0).all()):
            raise RuntimeError('Unused final-word bits are not zero')
        account = value_storage_account(1, 2, nv)
        actual_bytes = rec[0].numel() * rec[0].element_size()
        if actual_bytes != account['value_code_bytes'] or decoded.shape[-1] != nv:
            raise RuntimeError('Value owner or real-N decode shape mismatch')
        old_equal = None
        if nv % 32 == 0:
            old_equal = all(torch.equal(a, b) for a, b in zip(rec, binary_centroids(source, group_size=nv)))
            if not old_equal:
                raise RuntimeError('Aligned-N whole V changed the frozen binary codec')
        cases.append(dict(real_N=nv, code_words_per_channel=rec[0].shape[-1], actual_code_bytes=actual_bytes,
            unused_bits_per_channel=unused, true_N_centroids_and_decode_equal=True, old_aligned_codec_equal=old_equal,
            padded_visual_tokens=0))
    rejected = False
    try:
        binary_centroids_tail(torch.randn(1, 1, 128, 2438), group_size=128)
    except ValueError:
        rejected = True
    if not rejected:
        raise RuntimeError('Incomplete g128 metadata groups were accepted')
    budget_cases = []
    layers = heads = 32
    ranking = [(axis, layer, head) for layer in range(layers) for head in range(heads) for axis in ('k', 'v')]
    for nv in (2438, 2448):
        account = value_storage_account(layers, heads, nv)
        common = dict(layers=layers, heads=heads, visual_tokens=nv,
            visual_full_bytes=layers * heads * nv * 512, factor_bytes=3 * layers * heads * 4,
            budget_bits=1.8, selected_units=ranking)
        bases = {kb: layers * heads * (nv * (kb + 2) + account['stored_token_bit_capacity'] * 16)
                      + layers * heads * 128 * 6 + words * 8 + nv * 8
                 for kb, words in ((32, 256), (16, 16))}
        p2 = plan_key_stages(**common, base_bytes=bases[32])
        p1 = plan_key_stages(**common, base_bytes=bases[16], layout_reference='k2', key_index_bytes=16,
                             book_bytes=128, reference_factor_bytes=common['factor_bytes'])
        if p1['selected_stages'] != p2['selected_stages'] or p2['predicted_bytes'] - p1['predicted_bytes'] != layers * heads * nv * 16 + 1920:
            raise RuntimeError('Tail K1/K2 reference layouts or actual owner differences disagree')
        nominal = layers * heads * nv * 16
        if account['value_code_bytes'] - nominal != 16 * ((nv + 31) // 32 * 32 - nv) * layers * heads:
            raise RuntimeError('Physical tail word padding was omitted from actual owner')
        budget_cases.append(dict(real_N=nv, physical_tail_extra_bytes=account['unused_bit_capacity_bytes'],
            K2_complete_bytes=p2['predicted_bytes'], K1_same_layout_complete_bytes=p1['predicted_bytes'],
            K2_bits=p2['actual_budget_bits'], K1_bits=p1['actual_budget_bits'], stages=len(p2['selected_stages'])))
    for name in ('round7_tail_key_bank.py', 'round7_tail_key_attention.py', 'round7_tail_key_codec.py', 'round7_tail_key_check.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    return dict(passed=True, codec_cases=cases, budget_cases=budget_cases,
                invalid_g128_rejected=True, scope='CPU real-N fitting/packing/owner controls only; no GPU/task claim')


def gpu_checks():
    from affine_rate_bank import fold
    from check_fused_progressive import errors
    from fast_wht import torch_wht
    from round7_tail_key_attention import attention
    torch.manual_seed(20261116)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    patterns = [[], [('v', 31, 0), ('v', 31, 1)], [('k', 0, 0), ('k', 0, 1)],
                [('k', 1, 0), ('v', 1, 0), ('k', 1, 1), ('v', 1, 1)]]
    with torch.inference_mode():
        for nv in (2448, 2438):
            for nq in (1, 11):
                heads, nt = 32, 41 + nq
                q = torch.randn(1, nq, heads, 128, device='cuda', dtype=torch.float16).transpose(1, 2)
                source_k = torch.randn(1, heads, nv, 128, device='cuda', dtype=torch.float16)
                source_v = torch.randn_like(source_k)
                mean = source_k.float().mean(-2, keepdim=True).half()
                rotated = torch_wht(source_k.float() - mean.float())
                rec = binary_centroids_tail(source_v.transpose(2, 3).contiguous(), group_size=nv)
                vgain = (.85 + .3 * torch.rand(1, heads, 1, 1, device='cuda'))
                voffset = .1 * torch.randn(1, heads, 1, 1, device='cuda')
                rec = fold(rec, 1, vgain, voffset)
                base_v = decode_value_tail(*rec, nv, nv).transpose(2, 3).contiguous()
                state = SimpleNamespace(direction_mean=mean,
                    visual_pos=torch.arange(0, 2 * nv, 2, device='cuda'),
                    text_pos=torch.cat((torch.arange(1, 82, 2, device='cuda'), torch.arange(2 * nv, 2 * nv + nq, device='cuda'))),
                    text_length=nt, length=2 * nv + nq,
                    text_k=torch.randn(1, heads, 512, 128, device='cuda', dtype=torch.float16),
                    text_v=torch.randn(1, heads, 512, 128, device='cuda', dtype=torch.float16),
                    v_code=rec[0], v_scale=rec[1], v_min=rec[2])
                for words in (256, 16):
                    book = torch.randn(words, 4, device='cuda', dtype=torch.float16)
                    state.direction_code, state.direction_gain = encode_key(rotated, book, amplitude='norm')
                    base_k = decode_key(state.direction_code, state.direction_gain, book)
                    for selected in patterns:
                        lookup = torch.full((2, heads, 2), -1, device='cuda', dtype=torch.int32)
                        dense_k, dense_v = base_k.clone(), base_v.clone()
                        codes, gains = [None] * len(selected), [None] * len(selected)
                        for axis, head in sorted({(axis, head) for axis, head, _ in selected}):
                            depth = sum(a == axis and h == head for a, h, _ in selected)
                            target = rotated[0, head] if axis == 'k' else source_v[0, head]
                            base = base_k[0, head] if axis == 'k' else base_v[0, head]
                            rc, ra, decoded, _ = encode_sign_depth(target, base, depth)
                            for stage in range(depth):
                                slot = selected.index((axis, head, stage))
                                codes[slot], gains[slot] = rc[stage], ra[stage]
                                lookup[int(axis == 'v'), head, stage] = slot
                            (dense_k if axis == 'k' else dense_v)[0, head] = decoded
                        if selected:
                            state.depth_residual_code, state.depth_residual_alpha = torch.stack(codes), torch.stack(gains)
                            if state.depth_residual_code.numel() * 4 + state.depth_residual_alpha.numel() * 2 != len(selected) * nv * 18:
                                raise RuntimeError('Residual token dimension was padded')
                        else:
                            for name in ('depth_residual_code', 'depth_residual_alpha'):
                                if hasattr(state, name):
                                    delattr(state, name)
                        state.depth_residual_lookup = lookup
                        actual = attention(q, state, book, nv)
                        zv = torch_wht(q).half().float() @ dense_k.float().transpose(-1, -2) / 128 ** .5
                        zv += (q.float() * mean.float()).sum(-1, keepdim=True) / 128 ** .5
                        zt = q.float() @ state.text_k[:, :, :nt].float().transpose(-1, -2) / 128 ** .5
                        scores = torch.cat((zv, zt), -1)
                        positions = torch.cat((state.visual_pos, state.text_pos))
                        qpos = torch.arange(state.length - nq, state.length, device='cuda')
                        scores.masked_fill_(positions[None, None, None, :] > qpos[None, None, :, None], -torch.inf)
                        values = torch.cat((dense_v, state.text_v[:, :, :nt]), 2)
                        reference = (scores.softmax(-1) @ values.float()).half()
                        error = errors(actual, reference)
                        account = value_storage_account(1, heads, nv)
                        if state.v_code.numel() * 4 != account['value_code_bytes'] or state.visual_pos.numel() != nv:
                            raise RuntimeError('GPU physical owner or real token positions disagree')
                        cases.append(dict(real_N=nv, query_tokens=nq, query_heads=heads, kv_heads=heads,
                            GQA=1, book_words=words, key_gain='norm', selected_stages=selected,
                            value_code_bytes=state.v_code.numel() * 4,
                            physical_tail_extra_bytes=account['unused_bit_capacity_bytes'],
                            padded_visual_tokens=0, error=error,
                            passed=error['max_abs'] < .003 and error['relative_l2'] < .003))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cpu = cpu_controls()
    gpu = [] if args.cpu_only else gpu_checks()
    passed = cpu['passed'] and all(row['passed'] for row in gpu)
    sources = ('round7_tail_key_bank.py', 'round7_tail_key_attention.py', 'round7_tail_key_codec.py',
               'round7_tail_key_check.py', 'round7_key_layout.py', 'round7_key_codec.py')
    report = dict(status='PASS' if passed else 'FAIL', cpu_controls=cpu, gpu_executed=not args.cpu_only,
        gpu_cases=len(gpu), results=gpu,
        scope='actual representation reader arithmetic and physical owner controls; no task accuracy or speed claim',
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources})
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status=report['status'], gpu_executed=not args.cpu_only, cases=len(gpu))), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], gpu_executed=not args.cpu_only, cases=len(gpu))), flush=True)
    if not passed:
        raise RuntimeError('Tail key codec/reader check failed')


if __name__ == '__main__':
    main()
