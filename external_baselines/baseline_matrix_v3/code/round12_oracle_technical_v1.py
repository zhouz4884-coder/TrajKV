"""CUDA-only QA0 for native-K oracle partitions and packed control identity."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

sys.dont_write_bytecode = True
import torch
import round12_static_codec_v1 as codec


def require(value, message):
    if not value:
        raise ValueError('ROUND12_ORACLE_TECHNICAL: ' + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_sign(payload):
    shifts = torch.arange(32, device=payload.device, dtype=torch.int64)
    return (((payload.long()[..., None] >> shifts) & 1).flatten(-2).float() * 2 - 1)


def dense_scores(q, state, book, wht):
    k = codec.decode_key(state.direction_code, state.direction_gain, book)
    v = codec.half_value(state.v_code, state.v_scale, state.v_min).transpose(2, 3).contiguous()
    for axis, result in enumerate((k, v)):
        for head in range(4):
            for stage in range(2):
                slot = int(state.depth_residual_lookup[axis, head, stage])
                if slot >= 0:
                    sign = unpack_sign(state.depth_residual_code[slot])
                    result[0, head] = (result[0, head].float() + sign * state.depth_residual_alpha[slot].float()[:, None]).half()
    qr = wht(q, half=True)
    bias = (q.float() * state.direction_mean.repeat_interleave(7, 1).float()).sum(-1) / 128 ** .5
    zc = torch.matmul(qr.float(), k.repeat_interleave(7, 1).float().transpose(-2, -1)) / 128 ** .5 + bias[..., None]
    zf = torch.matmul(q.float(), state.oracle_native_k.repeat_interleave(7, 1).float().transpose(-2, -1)) / 128 ** .5
    zt = torch.matmul(q.float(), state.text_k.repeat_interleave(7, 1).float().transpose(-2, -1)) / 128 ** .5
    qpos = torch.arange(state.length - q.shape[-2], state.length, device=q.device)
    future_visual = state.visual_pos[None, None, None] > qpos[None, None, :, None]
    future_text = state.text_pos[None, None, None] > qpos[None, None, :, None]
    zc.masked_fill_(future_visual, -float('inf')); zf.masked_fill_(future_visual, -float('inf'))
    zt.masked_fill_(future_text, -float('inf'))
    values = torch.cat((v.repeat_interleave(7, 1), state.text_v.repeat_interleave(7, 1)), -2).float()
    return zc, zf, zt, values


def make_state(device, n, partial, gen):
    book = (torch.randn((16, 4), device=device, generator=gen) * .5).half()
    source = torch.randn((1, 4, n, 128), device=device, generator=gen).float()
    code, gain = codec.encode_key(source, book)
    state = SimpleNamespace(direction_code=code, direction_gain=gain,
        direction_mean=(torch.randn((1, 4, 1, 128), device=device, generator=gen) * .03).half(),
        v_code=codec.pack_channel_codes(torch.randint(0, 2, (1, 4, 128, n), device=device, generator=gen), 1),
        v_scale=(torch.rand((1, 4, 128, 1), device=device, generator=gen) + .2).half(),
        v_min=(torch.randn((1, 4, 128, 1), device=device, generator=gen) * .2 - .5).half(),
        visual_pos=torch.arange(n, device=device) * (2 if partial else 1),
        text_k=torch.randn((1, 4, 10, 128), device=device, generator=gen).half(),
        text_v=torch.randn((1, 4, 10, 128), device=device, generator=gen).half(),
        text_pos=(torch.arange(10, device=device) * 2 + 1 if partial else torch.arange(n, n + 10, device=device)),
        text_length=10, length=n + 10,
        oracle_native_k=torch.randn((1, 4, n, 128), device=device, generator=gen).half(), oracle_mode=0)
    state.depth_residual_lookup = torch.full((2, 4, 2), -1, device=device, dtype=torch.int32)
    state.depth_residual_lookup[0, 0, 0], state.depth_residual_lookup[0, 0, 1] = 0, 1
    state.depth_residual_lookup[1, 1, 0], state.depth_residual_lookup[1, 1, 1] = 2, 3
    state.depth_residual_code = codec.pack_channel_codes(torch.randint(0, 2, (4, n, 128), device=device, generator=gen), 1)
    state.depth_residual_alpha = (torch.rand((4, n), device=device, generator=gen) * .03).half()
    q = torch.randn((1, 28, 5, 128), device=device, generator=gen).half()
    return q, state, book


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-a2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visual-tokens', type=int, nargs='+', default=[32, 256])
    args = parser.parse_args()
    require(torch.cuda.is_available() and args.device.startswith('cuda'), 'actual CUDA required')
    require(all(n >= 32 and n % 32 == 0 for n in args.visual_tokens), 'complete visual words')
    output = args.output.resolve()
    require(str(output).startswith('/media/') and '/baseline_matrix_v3/results/round12' in str(output), 'project data disk required')
    sys.path.insert(0, str(args.legacy_a2))
    wht = importlib.import_module('fast_wht').wht
    old = importlib.import_module('round7_key_attention').attention
    oracle = importlib.import_module('round12_oracle_attention_v1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    gen = torch.Generator(device=device).manual_seed(1202501)
    started = time.perf_counter()
    rows = []
    for n in args.visual_tokens:
        for partial in (False, True):
            q, state, book = make_state(device, n, partial, gen)
            control = oracle.attention(q, state, book, n)
            frozen = old(q, state, book, n)
            exact = torch.equal(control.view(torch.uint8), frozen.view(torch.uint8))
            require(exact, 'packed control changed arithmetic')
            zc, zf, zt, values = dense_scores(q, state, book, wht)
            _, _, lc, lf = oracle.visual_lse(q, state, book)
            lc_ref, lf_ref = torch.logsumexp(zc, -1)[0], torch.logsumexp(zf, -1)[0]
            err_lc, err_lf = float((lc - lc_ref).abs().max()), float((lf - lf_ref).abs().max())
            require(err_lc <= 2e-5 and err_lf <= 2e-5, 'per-row visual LSE mismatch')
            for mode, name in ((1, 'mass-only'), (2, 'route-only')):
                state.oracle_mode = mode
                actual = oracle.attention(q, state, book, n)
                corrected = (zc + (lf_ref - lc_ref)[None, :, :, None] if mode == 1 else
                             zf + (lc_ref - lf_ref)[None, :, :, None])
                target_lse = lf_ref if mode == 1 else lc_ref
                partition_error = float((torch.logsumexp(corrected, -1)[0] - target_lse).abs().max())
                weights = torch.softmax(torch.cat((corrected, zt), -1), -1)
                dense = torch.matmul(weights, values).half()
                error = (actual.float() - dense.float()).abs()
                require(bool(torch.isfinite(actual).all()) and float(error.max()) <= .008, 'oracle packed/dense output mismatch')
                require(partition_error <= 2e-5, 'target partition not preserved')
                rows.append({'mode': name, 'N': n, 'synthetic_partial_visual_visibility': partial,
                    'same_student_Q_and_text_for_paired_LSE': True,
                    'control_old_reader_bit_exact': exact, 'compressed_LSE_max_abs': err_lc,
                    'native_LSE_max_abs': err_lf, 'corrected_partition_error': partition_error,
                    'packed_dense_output_max_abs': float(error.max()), 'output_tolerance': .008,
                    'native_K_reference_bytes_in_case': state.oracle_native_k.numel() * 2,
                    'packed_V_and_text_values_unchanged': True})
    torch.cuda.synchronize(device)
    report = {'schema': 'trajkv.round12.oracle_technical.v1', 'status': 'PASS_ACTUAL_CUDA_ORACLE_QA0',
        'actual_CUDA': True, 'model_calls': 0, 'task_answer_events': 0,
        'cases': rows, 'GPU': torch.cuda.get_device_name(device), 'elapsed_seconds': time.perf_counter() - started,
        'partial_visibility_cases_are_synthetic_kernel_checks_not_model_protocol': True,
        'trajectory_additivity_claimed': False,
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in
            ('round12_oracle_attention_v1.py', 'round12_oracle_bank_v1.py', 'round12_oracle_runner_v1.py',
             'round12_oracle_technical_v1.py', 'round12_static_codec_v1.py')}}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'ORACLE_CUDA_TECHNICAL_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(path), 'sha256': sha(path), 'QA': 0}), flush=True)


if __name__ == '__main__':
    main()
