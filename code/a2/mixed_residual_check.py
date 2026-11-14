"""GPU same-representation and old-backend checks for compact mixed residuals."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
import torch
from affine_rate_bank import fold
from check_fused_progressive import errors
from fast_wht import torch_wht
from functional_v_attention import attention as base_attention
from fused_progressive import decode_last_generic
from quantizers import binary_centroids, pack1
from sparse_v_residual_attention import attention as sparse_attention
from mixed_residual_attention import attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(253)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    patterns = [[], [('v', 3), ('v', 1)], [('k', 3), ('k', 1)],
                [('k', 1), ('v', 1)], [('v', 3), ('k', 1), ('v', 0), ('k', 3)]]
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
                    lookup = torch.full((2, 4), -1, device='cuda', dtype=torch.int32)
                    dense_k, dense_v = base_k.clone(), base_v.clone()
                    payload = 0
                    if selected:
                        signs, amplitudes = [], []
                        for slot, (axis, head) in enumerate(selected):
                            delta = (rotated_source[0, head].float() - base_k[0, head].float() if axis == 'k'
                                     else source_v[0, head].float() - base_v[0, head].float())
                            words = pack1(delta >= 0)
                            amplitude = delta.abs().mean(-1).half()
                            signs.append(words)
                            amplitudes.append(amplitude)
                            lookup[0 if axis == 'k' else 1, head] = slot
                            shift = torch.arange(32, device='cuda')
                            decoded_sign = (((words.long()[..., None] >> shift) & 1).reshape(nv, 128).float() * 2 - 1)
                            target = dense_k if axis == 'k' else dense_v
                            target[0, head] = (target[0, head].float() + decoded_sign * amplitude.float()[..., None]).half()
                        state.mixed_residual_code = torch.stack(signs).contiguous()
                        state.mixed_residual_alpha = torch.stack(amplitudes).contiguous()
                        payload = state.mixed_residual_code.numel() * 4 + state.mixed_residual_alpha.numel() * 2
                        if payload != len(selected) * nv * 18:
                            raise RuntimeError('Mixed residual pool contains padding/unselected payload')
                    else:
                        for name in ('mixed_residual_code', 'mixed_residual_alpha'):
                            if hasattr(state, name):
                                delattr(state, name)
                    state.mixed_residual_lookup = lookup
                    actual = attention(q, state, book, window, beta)
                    old_equal = None
                    if not selected:
                        old_equal = bool(torch.equal(actual, base_attention(q, state, book, window, beta)))
                    elif len({axis for axis, _ in selected}) == 1:
                        axis = selected[0][0]
                        state.sparse_residual_axis = axis
                        state.v_residual_lookup = lookup[0 if axis == 'k' else 1]
                        state.v_residual_code = state.mixed_residual_code
                        state.v_residual_alpha = state.mixed_residual_alpha
                        old_equal = bool(torch.equal(actual, sparse_attention(q, state, book, window, beta)))
                    if old_equal is False:
                        raise RuntimeError('Mixed no-residual/single-axis path changed the existing backend')
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
                    rows.append({'nv': nv, 'nq': nq, 'value_window': window, 'selected_axis_heads': selected,
                        'compact_residual_bytes': payload, 'old_single_axis_or_base_bitwise_equal': old_equal,
                        'dual_lookup_bytes': lookup.numel() * lookup.element_size(), 'error': err,
                        'passed': err['max_abs'] < .003 and err['relative_l2'] < .003})
    passed = all(row['passed'] for row in rows)
    report = {'status': 'PASS' if passed else 'FAIL', 'cases': len(rows), 'results': rows,
        'scope': 'same-representation GPU arithmetic, no task accuracy/efficiency claim',
        'sources': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                    ('mixed_residual_bank.py', 'mixed_residual_attention.py', 'mixed_residual_check.py')}}
    (a.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (a.output / 'complete.json').write_text(json.dumps({'status': report['status'], 'cases': len(rows)}), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'cases': len(rows)}), flush=True)
    if not passed:
        raise RuntimeError('Mixed residual GPU arithmetic failed')


if __name__ == '__main__':
    main()
