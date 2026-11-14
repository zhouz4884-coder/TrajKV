"""GPU numeric checks: grouped metadata folding and visual-only packed bias."""
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
from fused_learned_direction import attention as original_attention
from functional_v_attention import attention
from fused_progressive import decode_last_generic
from quantizers import binary_centroids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(158)
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
            values = torch.randn(1, 4, 128, nv, device='cuda', dtype=torch.float16)
            book = torch.randn(256, 4, device='cuda', dtype=torch.float16)
            for window in [128, nv]:
                original_record = binary_centroids(values, group_size=window)
                for groups in [1, 8]:
                    gain = (.85+.3*torch.rand(4, groups, device='cuda')).repeat_interleave(128//groups, -1)[None, :, :, None]
                    offset = (.1*torch.randn(4, groups, device='cuda')).repeat_interleave(128//groups, -1)[None, :, :, None]
                    record = fold(original_record, 1, gain, offset)
                    state.v_code, state.v_scale, state.v_min = record
                    decoded = decode_last_generic(*record, window, 1).transpose(2, 3).contiguous()
                    zero = attention(q, state, book, window, torch.zeros(4, device='cuda'))
                    legacy = original_attention(q, state, book, 1, True, window)
                    if not torch.equal(zero, legacy):
                        raise RuntimeError('Zero bias changed original packed output')
                    beta = torch.tensor([-.6, -.2, .2, .6], device='cuda')
                    output = attention(q, state, book, window, beta)
                    rotated = (book[state.direction_code.long()].reshape(1, 4, nv, 128).float()*state.direction_gain.float()).half().repeat_interleave(7, 1)
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
                    rows.append(dict(nv=nv, nq=nq, value_window=window, channel_groups=groups,
                                     zero_bias_bitwise_equal=True, error=error,
                                     passed=error['max_abs'] < .003 and error['relative_l2'] < .003))
    passed = all(row['passed'] for row in rows)
    report = dict(status='PASS' if passed else 'FAIL', cases=len(rows), results=rows,
                  scope='same-representation GPU arithmetic only; no task accuracy claim',
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ['functional_v_attention.py', 'functional_v_bank.py', 'functional_v_check.py']})
    (args.output/'check.json').write_text(json.dumps(report, indent=2))
    (args.output/'complete.json').write_text(json.dumps(dict(status=report['status'], cases=len(rows))))
    print(json.dumps(dict(status=report['status'], cases=len(rows))), flush=True)
    if not passed:
        raise RuntimeError('Functional V packed arithmetic failed')


if __name__ == '__main__':
    main()
