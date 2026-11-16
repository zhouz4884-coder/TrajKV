"""Actual CUDA arithmetic check for the LLaVA-shaped packed depth reader.

Nvisual=576, Hquery=Hkv=32, D=128, whole-image V metadata, q=1/11.
This is a same-representation reader check, never a model/task-quality result.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires actual CUDA/Triton execution; there is no CPU substitute')
    import triton
    from affine_rate_bank import fold
    from fast_wht import torch_wht
    from fused_progressive import decode_last_generic
    from quantizers import binary_centroids
    from mixed_residual_attention import attention as old_mixed_attention
    from round6_depth_attention import attention
    from round6_depth_codec import encode_sign_depth, validate_stages
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(2026111532)
    torch.backends.cuda.matmul.allow_tf32 = False
    nv, nh, dim = 576, 32, 128
    patterns = [[],
        [('v', 31, 0)], [('k', 31, 0)],
        [('v', 31, 0), ('v', 31, 1)],
        [('k', 31, 0), ('k', 31, 1)],
        [('k', 31, 0), ('v', 31, 0)],
        [('v', 31, 0), ('k', 31, 0), ('k', 31, 1), ('v', 31, 1)],
        [('v', 31, 0), ('k', 17, 0), ('v', 0, 0), ('k', 31, 0),
         ('v', 31, 1), ('k', 31, 1), ('k', 17, 1)],
        [('k', 0, 0), ('v', 17, 0), ('k', 0, 1), ('v', 17, 1),
         ('v', 31, 0), ('k', 30, 0), ('k', 30, 1)]]
    for selected in patterns:
        validate_stages([(axis, 0, head, stage) for axis, head, stage in selected], 1, nh)
    rows = []
    with torch.inference_mode():
        for nq in (1, 11):
            q = torch.randn(1, nq, nh, dim, device='cuda', dtype=torch.float16).transpose(1, 2)
            source_k = torch.randn(1, nh, nv, dim, device='cuda', dtype=torch.float16)
            source_v = torch.randn_like(source_k)
            mean = source_k.float().mean(-2, keepdim=True).half().contiguous()
            rotated_source = torch_wht(source_k.float() - mean.float())
            prefix_text_tokens, capacity = 37, 128
            qstart = 3 * nv + 19
            nt = prefix_text_tokens + nq
            state = SimpleNamespace(
                direction_code=torch.randint(0, 256, (1, nh, nv, 32), device='cuda', dtype=torch.uint8),
                direction_gain=(.5 + torch.rand(1, nh, nv, 1, device='cuda')).half(),
                direction_mean=mean,
                visual_pos=3 * torch.arange(nv, device='cuda'),
                text_pos=torch.cat((1 + 3 * torch.arange(prefix_text_tokens, device='cuda'),
                                    torch.arange(qstart, qstart + nq, device='cuda'))),
                text_length=nt, length=qstart + nq,
                text_k=torch.randn(1, nh, capacity, dim, device='cuda', dtype=torch.float16),
                text_v=torch.randn(1, nh, capacity, dim, device='cuda', dtype=torch.float16))
            book = torch.randn(256, 4, device='cuda', dtype=torch.float16)
            base_k = (book[state.direction_code.long()].reshape(1, nh, nv, dim).float()
                      * state.direction_gain.float()).half()
            record = binary_centroids(source_v.transpose(2, 3).contiguous(), group_size=nv)
            gain = (.85 + .3 * torch.rand(nh, 8, device='cuda')).repeat_interleave(16, -1)[None, :, :, None]
            offset = (.1 * torch.randn(nh, 8, device='cuda')).repeat_interleave(16, -1)[None, :, :, None]
            record = fold(record, 1, gain, offset)
            state.v_code, state.v_scale, state.v_min = record
            base_v = decode_last_generic(*record, nv, 1).transpose(2, 3).contiguous()
            beta = torch.linspace(-.4, .4, nh, device='cuda')
            for case_index, selected in enumerate(patterns):
                lookup = torch.full((2, nh, 2), -1, device='cuda', dtype=torch.int32)
                dense_k, dense_v = base_k.clone(), base_v.clone()
                if selected:
                    codes, alphas = [None] * len(selected), [None] * len(selected)
                    for axis, head in sorted({(axis, head) for axis, head, _ in selected}):
                        depth = sum(item_axis == axis and item_head == head for item_axis, item_head, _ in selected)
                        source = rotated_source[0, head] if axis == 'k' else source_v[0, head]
                        base = base_k[0, head] if axis == 'k' else base_v[0, head]
                        words, amplitudes, decoded, _ = encode_sign_depth(source, base, depth)
                        for stage in range(depth):
                            slot = selected.index((axis, head, stage))
                            codes[slot], alphas[slot] = words[stage], amplitudes[stage]
                            lookup[0 if axis == 'k' else 1, head, stage] = slot
                        (dense_k if axis == 'k' else dense_v)[0, head] = decoded
                    state.depth_residual_code = torch.stack(codes).contiguous()
                    state.depth_residual_alpha = torch.stack(alphas).contiguous()
                    payload = state.depth_residual_code.numel() * state.depth_residual_code.element_size()
                    payload += state.depth_residual_alpha.numel() * state.depth_residual_alpha.element_size()
                else:
                    payload = 0
                    for name in ('depth_residual_code', 'depth_residual_alpha'):
                        if hasattr(state, name):
                            delattr(state, name)
                if payload != len(selected) * nv * 18:
                    raise RuntimeError('Compact pool has padding or unselected residual stages')
                state.depth_residual_lookup = lookup
                actual = attention(q, state, book, nv, beta)
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
                    old_equal = bool(torch.equal(actual, old_mixed_attention(q, state, book, nv, beta)))
                visual_scores = torch_wht(q).half().float() @ dense_k.float().transpose(-1, -2) / dim ** .5
                visual_scores += (q.float() * mean.float()).sum(-1, keepdim=True) / dim ** .5
                visual_scores += beta[None, :, None, None]
                text_scores = q.float() @ state.text_k[:, :, :nt].float().transpose(-1, -2) / dim ** .5
                scores = torch.cat((visual_scores, text_scores), -1)
                positions = torch.cat((state.visual_pos, state.text_pos))
                query_positions = torch.arange(qstart, qstart + nq, device='cuda')
                scores.masked_fill_(positions[None, None, None, :] > query_positions[None, None, :, None], -torch.inf)
                values = torch.cat((dense_v, state.text_v[:, :, :nt]), 2)
                reference = (scores.softmax(-1) @ values.float()).half()
                diff = actual.float() - reference.float()
                max_abs = float(diff.abs().max())
                relative_l2 = float(diff.norm() / reference.float().norm().clamp_min(1e-30))
                finite = bool(torch.isfinite(actual).all())
                passed = finite and max_abs < .003 and relative_l2 < .003 and old_equal is not False
                row = dict(case=case_index, visual_tokens=nv, query_tokens=nq, Hq=nh, Hkv=nh, GQA=1,
                    value_window=nv, selected_axis_head_stages=selected, residual_stages=len(selected),
                    compact_residual_bytes=payload, expected_compact_bytes=len(selected) * nv * 18,
                    stage_lookup_bytes=lookup.numel() * lookup.element_size(), no_padding=True,
                    zero_or_one_stage_old_mixed_bitwise_equal=old_equal,
                    position_gaps=True, suffix_causal_mask=True,
                    max_abs=max_abs, relative_l2=relative_l2, finite=finite, passed=passed)
                rows.append(row)
                print(json.dumps({key: row[key] for key in ('case', 'query_tokens', 'residual_stages', 'max_abs', 'relative_l2', 'passed')}), flush=True)
    torch.cuda.synchronize()
    passed = all(row['passed'] for row in rows)
    report = dict(status='PASS' if passed else 'FAIL', actual_cuda=True, gpu_executed=True,
        gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__, triton_version=triton.__version__,
        cases=len(rows), max_abs=max(row['max_abs'] for row in rows),
        max_relative_l2=max(row['relative_l2'] for row in rows), results=rows,
        scope='Actual Triton versus dense same packed representation, LLaVA-shaped arithmetic only; no model/task quality or transfer-gain claim',
        dense_reference='FP16 sequential stage reconstruction, FP16 WHT(q), visual mean bias, full visual/text softmax with original gapped positions and suffix causal mask',
        sources={name: sha(Path(__file__).with_name(name)) for name in
            ('round6_second_reader_check.py', 'round6_depth_attention.py', 'round6_depth_codec.py', 'mixed_residual_attention.py')})
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps({key: report[key] for key in
        ('status', 'actual_cuda', 'cases', 'max_abs', 'max_relative_l2')}), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'actual_cuda', 'cases', 'max_abs', 'max_relative_l2')}), flush=True)
    if not passed:
        raise RuntimeError('Second-backbone-shaped packed reader arithmetic failed')


if __name__ == '__main__':
    main()
