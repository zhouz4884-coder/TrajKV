"""Twenty-four GPU numeric checks for compact selected-head K/V residuals."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/experiments'))
import torch
from affine_rate_bank import fold
from check_fused_progressive import errors
from fast_wht import torch_wht
from functional_v_attention import attention as base_attention
from fused_progressive import decode_last_generic
from quantizers import binary_centroids, pack1
from sparse_v_residual_attention import attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(253)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with torch.inference_mode():
        for nv, nq in [(256, 1), (768, 11), (1792, 1)]:
            q = torch.randn(1, nq, 28, 128, device='cuda', dtype=torch.float16).transpose(1, 2)
            nt = 41+nq
            textpos = torch.cat((torch.arange(1, 82, 2, device='cuda'),
                                 torch.arange(2*nv, 2*nv+nq, device='cuda')))
            state = SimpleNamespace(
                direction_code=torch.randint(0, 256, (1, 4, nv, 32), device='cuda', dtype=torch.uint8),
                direction_gain=(.5+torch.rand(1, 4, nv, 1, device='cuda')).half(),
                direction_mean=.1*torch.randn(1, 4, 1, 128, device='cuda', dtype=torch.float16),
                visual_pos=torch.arange(0, 2*nv, 2, device='cuda'), text_pos=textpos,
                text_length=nt, length=2*nv+nq,
                text_k=torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.float16),
                text_v=torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.float16))
            source = torch.randn(1, 4, nv, 128, device='cuda', dtype=torch.float16)
            book = torch.randn(256, 4, device='cuda', dtype=torch.float16)
            beta = torch.tensor([-.6, -.2, .2, .6], device='cuda')
            source_k = torch.randn_like(source)
            source_k_rotated = torch_wht(source_k.float()-source_k.float().mean(-2, keepdim=True).half().float())
            base_k = (book[state.direction_code.long()].reshape(1, 4, nv, 128).float()*state.direction_gain.float()).half()
            for window in [128, nv]:
                original = binary_centroids(source.transpose(2, 3).contiguous(), group_size=window)
                g = (.85+.3*torch.rand(4, 8, device='cuda')).repeat_interleave(16, -1)[None, :, :, None]
                o = (.1*torch.randn(4, 8, device='cuda')).repeat_interleave(16, -1)[None, :, :, None]
                record = fold(original, 1, g, o)
                state.v_code, state.v_scale, state.v_min = record
                base = decode_last_generic(*record, window, 1).transpose(2, 3).contiguous()
                for axis in ['v', 'k']:
                    for selected in [[], [3, 1]]:
                        lookup = torch.full((4,), -1, device='cuda', dtype=torch.int32)
                        decoded = base.clone()
                        reconstructed_k = base_k.clone()
                        payload_bytes = 0
                        if selected:
                            delta = (source_k_rotated[0, selected].float()-base_k[0, selected].float() if axis == 'k'
                                     else source[0, selected].float()-base[0, selected].float())
                            state.v_residual_code = pack1(delta >= 0)
                            state.v_residual_alpha = delta.abs().mean(-1).half().contiguous()
                            for index, head in enumerate(selected):
                                lookup[head] = index
                            shift = torch.arange(32, device='cuda')
                            sign = (((state.v_residual_code.long()[..., None] >> shift) & 1)
                                    .reshape(len(selected), nv, 128).float()*2-1)
                            if axis == 'k':
                                reconstructed_k[0, selected] = (base_k[0, selected].float()+sign*state.v_residual_alpha.float()[..., None]).half()
                            else:
                                decoded[0, selected] = (base[0, selected].float()+sign*state.v_residual_alpha.float()[..., None]).half()
                            payload_bytes = state.v_residual_code.numel()*4+state.v_residual_alpha.numel()*2
                            if payload_bytes != len(selected)*nv*18:
                                raise RuntimeError('Sparse residual payload is not compact')
                        else:
                            for name in ['v_residual_code', 'v_residual_alpha']:
                                if hasattr(state, name):
                                    delattr(state, name)
                        state.v_residual_lookup = lookup
                        state.sparse_residual_axis = axis
                        output = attention(q, state, book, window, beta)
                        base_equal = None
                        if not selected:
                            base_equal = bool(torch.equal(output, base_attention(q, state, book, window, beta)))
                            if not base_equal:
                                raise RuntimeError('Unselected path changed the base packed output')
                        rotated = reconstructed_k.repeat_interleave(7, 1)
                        zv = torch_wht(q).half().float() @ rotated.float().transpose(-1, -2) / 128**.5
                        zv += (q.float()*state.direction_mean.repeat_interleave(7, 1).float()).sum(-1, keepdim=True) / 128**.5
                        zv += beta.repeat_interleave(7)[None, :, None, None]
                        zt = q.float() @ state.text_k[:, :, :nt].repeat_interleave(7, 1).float().transpose(-1, -2) / 128**.5
                        z = torch.cat((zv, zt), -1)
                        positions = torch.cat((state.visual_pos, state.text_pos))
                        qpos = torch.arange(state.length-nq, state.length, device='cuda')
                        z.masked_fill_(positions[None, None, None, :] > qpos[None, None, :, None], -torch.inf)
                        allvalues = torch.cat((decoded, state.text_v[:, :, :nt]), 2).repeat_interleave(7, 1)
                        reference = (z.softmax(-1) @ allvalues.float()).half()
                        error = errors(output, reference)
                        rows.append(dict(nv=nv, nq=nq, value_window=window, residual_axis=axis, selected_heads=selected,
                                         compact_residual_bytes=payload_bytes, zero_selection_bitwise_equal=base_equal,
                                         error=error, passed=error['max_abs'] < .003 and error['relative_l2'] < .003))
    passed = all(row['passed'] for row in rows)
    report = dict(status='PASS' if passed else 'FAIL', cases=len(rows), results=rows,
                  scope='same-representation GPU arithmetic only; no task accuracy claim',
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ['sparse_v_residual_attention.py', 'sparse_v_residual_bank.py', 'sparse_v_residual_check.py']})
    (args.output/'check.json').write_text(json.dumps(report, indent=2))
    (args.output/'complete.json').write_text(json.dumps(dict(status=report['status'], cases=len(rows))))
    print(json.dumps(dict(status=report['status'], cases=len(rows))), flush=True)
    if not passed:
        raise RuntimeError('Sparse K/V residual packed arithmetic failed')


if __name__ == '__main__':
    main()
