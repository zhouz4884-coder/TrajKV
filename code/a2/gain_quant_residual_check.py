"""GPU checks for real4/8bit gain packing and16bit legacy equivalence."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/experiments'))
import torch
from check_fused_progressive import errors
from fast_wht import torch_wht
from fused_progressive import decode_last_generic
from quantizers import binary_centroids, pack1
from gain_quant_residual_attention import attention, encode_gain, decode_gain, gain_tensors
from sparse_v_residual_attention import attention as legacy_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(289)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with torch.inference_mode():
        for nv, nq in [(256, 1), (256, 7), (1792, 1), (1792, 7)]:
            q = torch.randn(1, nq, 28, 128, device='cuda', dtype=torch.float16).transpose(1, 2)
            nt = 41+nq
            textpos = torch.cat((torch.arange(1, 82, 2, device='cuda'),
                                 torch.arange(2*nv, 2*nv+nq, device='cuda')))
            state = SimpleNamespace(
                direction_code=torch.randint(0, 256, (1, 4, nv, 32), device='cuda', dtype=torch.uint8),
                direction_mean=.1*torch.randn(1, 4, 1, 128, device='cuda', dtype=torch.float16),
                visual_pos=torch.arange(0, 2*nv, 2, device='cuda'), text_pos=textpos,
                text_length=nt, length=2*nv+nq,
                text_k=torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.float16),
                text_v=torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.float16),
                sparse_residual_axis='v')
            full_gain = (.25+2*torch.rand(1, 4, nv, 1, device='cuda')).half()
            full_gain[:, 0, :128] = .625  # zero-range affine group
            source = torch.randn(1, 4, nv, 128, device='cuda', dtype=torch.float16)
            record = binary_centroids(source.transpose(2, 3).contiguous(), group_size=nv)
            state.v_code, state.v_scale, state.v_min = record
            base_v = decode_last_generic(*record, nv, 1).transpose(2, 3).contiguous()
            book = torch.randn(256, 4, device='cuda', dtype=torch.float16)
            beta = torch.tensor([-.6, -.2, .2, .6], device='cuda')
            for bits in (16, 8, 4):
                state.direction_gain_bits = bits
                if bits == 16:
                    state.direction_gain = full_gain.clone()
                    decoded_gain = state.direction_gain.squeeze(-1)
                else:
                    state.direction_gain_code, state.direction_gain_scale, state.direction_gain_min = encode_gain(full_gain.squeeze(-1), bits)
                    if hasattr(state, 'direction_gain'):
                        del state.direction_gain
                    decoded_gain = decode_gain(state.direction_gain_code, state.direction_gain_scale, state.direction_gain_min, bits)
                    if state.direction_gain_code.dtype != torch.int32 or state.direction_gain_scale.dtype != torch.float16:
                        raise RuntimeError('Gain payload dtype differs from promised packed format')
                actual_gain_bytes = sum(t.numel()*t.element_size() for t in gain_tensors(state))
                expected_gain_bytes = 4*(2*nv if bits == 16 else nv*bits//8+4*(nv//128))
                if actual_gain_bytes != expected_gain_bytes:
                    raise RuntimeError('Gain metadata/payload byte count differs')
                if bits != 16 and hasattr(state, 'direction_gain'):
                    raise RuntimeError('Packed gain retained F16 gain owner')
                for selected in ([], [3, 1]):
                    state.v_residual_lookup = torch.full((4,), -1, device='cuda', dtype=torch.int32)
                    decoded_v = base_v.clone()
                    if selected:
                        error_v = source[0, selected].float()-base_v[0, selected].float()
                        state.v_residual_code = pack1(error_v >= 0)
                        state.v_residual_alpha = error_v.abs().mean(-1).half().contiguous()
                        for index, head in enumerate(selected):
                            state.v_residual_lookup[head] = index
                        shifts = torch.arange(32, device='cuda')
                        sign = ((((state.v_residual_code.long() & 0xffffffff)[..., None] >> shifts) & 1)
                                .reshape(len(selected), nv, 128).float()*2-1)
                        decoded_v[0, selected] = (base_v[0, selected].float()+sign*state.v_residual_alpha.float()[..., None]).half()
                    else:
                        for name in ('v_residual_code', 'v_residual_alpha'):
                            if hasattr(state, name):
                                delattr(state, name)
                    output = attention(q, state, book, nv, beta)
                    legacy_equal = None
                    if bits == 16:
                        legacy_equal = bool(torch.equal(output, legacy_attention(q, state, book, nv, beta)))
                        if not legacy_equal:
                            raise RuntimeError('16bit gain path differs from frozen sparse reader')
                    rotated = (book[state.direction_code.long()].reshape(1, 4, nv, 128).float()*decoded_gain[..., None].float()).half().repeat_interleave(7, 1)
                    zv = torch_wht(q).half().float() @ rotated.float().transpose(-1, -2) / 128**.5
                    zv += (q.float()*state.direction_mean.repeat_interleave(7, 1).float()).sum(-1, keepdim=True) / 128**.5
                    zv += beta.repeat_interleave(7)[None, :, None, None]
                    zt = q.float() @ state.text_k[:, :, :nt].repeat_interleave(7, 1).float().transpose(-1, -2) / 128**.5
                    z = torch.cat((zv, zt), -1)
                    positions = torch.cat((state.visual_pos, state.text_pos))
                    qpos = torch.arange(state.length-nq, state.length, device='cuda')
                    z.masked_fill_(positions[None, None, None, :] > qpos[None, None, :, None], -torch.inf)
                    values = torch.cat((decoded_v, state.text_v[:, :, :nt]), 2).repeat_interleave(7, 1)
                    expected = (z.softmax(-1) @ values.float()).half()
                    error = errors(output, expected)
                    rows.append(dict(nv=nv, nq=nq, gain_bits=bits, selected_heads=selected,
                                     gain_bytes=actual_gain_bytes, fp16_legacy_bitwise_equal=legacy_equal,
                                     gain_mse=float((decoded_gain.float()-full_gain.squeeze(-1).float()).square().mean()),
                                     error=error, passed=error['max_abs'] < .003 and error['relative_l2'] < .003))
    passed = all(row['passed'] for row in rows)
    report = dict(status='PASS' if passed else 'FAIL', cases=len(rows), results=rows,
                  scope='same-representation GPU arithmetic; no task quality or speed claim',
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('gain_quant_residual_attention.py', 'gain_quant_residual_bank.py', 'gain_quant_residual_check.py')})
    (args.output/'check.json').write_text(json.dumps(report, indent=2))
    (args.output/'complete.json').write_text(json.dumps(dict(status=report['status'], cases=len(rows))))
    print(json.dumps(dict(status=report['status'], cases=len(rows))), flush=True)
    if not passed:
        raise RuntimeError('Gain-quant residual numeric check failed')


if __name__ == '__main__':
    main()
