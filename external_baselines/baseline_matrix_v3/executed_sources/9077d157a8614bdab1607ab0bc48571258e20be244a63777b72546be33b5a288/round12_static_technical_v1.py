"""Actual CUDA QA0 checks for Round12 bit streams and packed attention.

This driver never loads a model and never creates a task answer. Root controls
remote dispatch, cache locations and evidence collection.
"""
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
        raise ValueError('ROUND12_TECHNICAL: ' + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def pack_checks(device):
    records = []
    gen = torch.Generator(device=device).manual_seed(1201201)
    for bits, words in ((4, 32), (8, 32), (10, 32), (12, 16)):
        ids = torch.randint(0, 1 << bits, (1, 4, 256, words), device=device, generator=gen)
        ids[0, 0, 0, 0], ids[0, 0, 0, 1] = 0, (1 << bits) - 1
        payload = codec.pack_indices(ids, bits)
        decoded = codec.unpack_indices(payload, bits, words)
        require(torch.equal(ids, decoded), 'K bit stream roundtrip ' + str(bits))
        require(payload.shape[-1] == words * bits // 8, 'K padding is forbidden')
        records.append({'bits': bits, 'words_per_token_head': words,
                        'actual_bytes_per_token_head': payload.shape[-1], 'all_indices_exact': True})
    return records


def value_checks(device):
    gen = torch.Generator(device=device).manual_seed(1201202)
    n = 256
    bit = torch.randint(0, 2, (1, 4, 128, n), device=device, generator=gen)
    payload = codec.pack_channel_codes(bit, 1)
    lo = torch.randn((1, 4, 128, 1), device=device, generator=gen).half()
    scale = (torch.rand((1, 4, 128, 1), device=device, generator=gen) + .02).half()
    compact, ss, mm, mask = codec.compact_vhalf(payload, scale, lo)
    old = codec.half_value(payload, scale, lo)
    new = codec.decode_vhalf(compact, ss, mm, mask)
    keep = ss[..., 0] != 0
    require(bits_equal(old[keep], new[keep]), 'kept V channels changed')
    require(bool((new[~keep] == mm.expand_as(new)[~keep]).all()), 'dropped centers not self-contained')
    require(compact.numel() * compact.element_size() == 1 * 4 * n * 8, 'V0.5 payload bytes')
    require(mask.numel() == 4 * 16, 'V mask bytes')
    l = torch.tensor(2., device=device, dtype=torch.float16)
    s = torch.tensor(8., device=device, dtype=torch.float16)
    mu = ((1 - .25) * l.float() + .25 * (s + l).half().float()).half()
    require(float(mu) == 4., 'center illustrative case')
    return {'N': n, 'kept_channel_reconstruction_bit_exact': True,
            'dropped_center_decoder_requires_no_p': True, 'compact_bytes_per_token_head': 8,
            'mask_bytes_per_head': 16, 'center_example_mu': float(mu)}


def _decode_bits(payload, n):
    shifts = torch.arange(32, device=payload.device, dtype=torch.int64)
    return ((payload.long()[..., None] >> shifts) & 1).reshape(*payload.shape[:-1], n)


def _decode_residual(base, state, axis):
    result = base.clone()
    for head in range(4):
        for stage in range(2):
            slot = int(state.depth_residual_lookup[axis, head, stage])
            if slot >= 0:
                sign = _decode_bits(state.depth_residual_code[slot], 128).float() * 2 - 1
                result[0, head] = (result[0, head].float() + sign * state.depth_residual_alpha[slot].float()[:, None]).half()
    return result


def dense_reference(q, state, book, wht):
    n = state.visual_pos.numel()
    k = codec.decode_key(state.direction_code, state.direction_gain, book)
    k = _decode_residual(k, state, 0)
    if state.round12_vmode == 2:
        v = codec.decode_vhalf(state.v_code, state.v_scale, state.v_min, state.round12_v_mask)
    else:
        v = codec.half_value(state.v_code, state.v_scale, state.v_min,
            bits=2 if state.round12_vmode == 1 else 1,
            levels=getattr(state, 'round12_value_levels', None))
    v = _decode_residual(v.transpose(2, 3).contiguous(), state, 1)
    kr, vr = k.repeat_interleave(7, 1), v.repeat_interleave(7, 1)
    qr = wht(q, half=True)
    bias = (q.float() * state.direction_mean.repeat_interleave(7, 1).float()).sum(-1) / 128 ** .5
    sv = torch.matmul(qr.float(), kr.float().transpose(-2, -1)) / 128 ** .5 + bias[..., None]
    st = torch.matmul(q.float(), state.text_k.repeat_interleave(7, 1).float().transpose(-2, -1)) / 128 ** .5
    score = torch.cat((sv, st), -1)
    positions = torch.cat((state.visual_pos, state.text_pos))
    qpos = torch.arange(state.length - q.shape[-2], state.length, device=q.device)
    score.masked_fill_(positions[None, None, None] > qpos[None, None, :, None], -float('inf'))
    probabilities = torch.softmax(score, -1)
    values = torch.cat((vr, state.text_v.repeat_interleave(7, 1)), -2)
    return torch.matmul(probabilities.float(), values.float()).half()


def attention_checks(device, n, a2):
    sys.path.insert(0, str(a2))
    wht = importlib.import_module('fast_wht').wht
    legacy = importlib.import_module('round7_key_attention').attention
    reader = importlib.import_module('round12_static_attention_v1').attention
    gen = torch.Generator(device=device).manual_seed(1201300 + n)
    cases = [('K1V1', 16, 4, 0), ('K2V1', 256, 4, 0),
             ('K2p5V1', 1024, 4, 0), ('G8V1', 4096, 8, 0),
             ('K1V2R', 16, 4, 1), ('K2Vhalf', 256, 4, 2), ('K2p5Vhalf', 1024, 4, 2)]
    records = []
    for name, words, dim, vm in cases:
        book = (torch.randn((words, dim), generator=gen, device=device) * .5).half()
        x = torch.randn((1, 4, n, 128), generator=gen, device=device).float()
        code, gain = codec.encode_key(x, book, chunk_parts=2048)
        bits = torch.randint(0, 2, (1, 4, 128, n), generator=gen, device=device)
        vc = codec.pack_channel_codes(bits, 1)
        vs = (torch.rand((1, 4, 128, 1), generator=gen, device=device) + .1).half()
        vmin = (torch.randn((1, 4, 128, 1), generator=gen, device=device) * .2 - .5).half()
        state = SimpleNamespace(direction_code=code, direction_gain=gain,
            direction_mean=(torch.randn((1, 4, 1, 128), generator=gen, device=device) * .04).half(),
            v_code=vc, v_scale=vs, v_min=vmin, visual_pos=torch.arange(1, n + 1, device=device),
            text_pos=torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.arange(n + 1, n + 10, device=device))),
            text_k=torch.randn((1, 4, 10, 128), generator=gen, device=device).half(),
            text_v=torch.randn((1, 4, 10, 128), generator=gen, device=device).half(),
            text_length=10, length=n + 10, round12_vmode=vm)
        if vm == 1:
            state.round12_value_levels = codec.normalized_levels([-1.5104, -.45278, .45278, 1.5104], device)
            state.v_code = codec.pack_channel_codes(torch.randint(0, 4, (1, 4, 128, n), generator=gen, device=device), 2)
        elif vm == 2:
            state.v_code, state.v_scale, state.v_min, state.round12_v_mask = codec.compact_vhalf(vc, vs, vmin)
        lookup = torch.full((2, 4, 2), -1, dtype=torch.int32, device=device)
        lookup[0, 0, 0], lookup[0, 0, 1], lookup[1, 1, 0], lookup[1, 1, 1] = 0, 1, 2, 3
        state.depth_residual_lookup = lookup
        residual = torch.randint(0, 2, (4, n, 128), device=device, generator=gen)
        state.depth_residual_code = codec.pack_channel_codes(residual, 1)
        state.depth_residual_alpha = (torch.rand((4, n), device=device, generator=gen) * .03).half()
        q = torch.randn((1, 28, 5, 128), generator=gen, device=device).half()
        actual = reader(q, state, book, n)
        reference = dense_reference(q, state, book, wht)
        error = (actual.float() - reference.float()).abs()
        require(bool(torch.isfinite(actual).all()), 'nonfinite packed output ' + name)
        require(float(error.max()) <= .008, 'packed/dense mismatch ' + name + ': ' + str(float(error.max())))
        item = {'name': name, 'N': n, 'key_bytes_per_token_head': code.shape[-1],
                'V_mode': vm, 'dense_max_abs': float(error.max()), 'dense_rmse': float(error.square().mean().sqrt()),
                'dense_reference_tolerance': .008, 'complete_128d_output_checked': True,
                'two_sequential_Half_K_and_V_stages_exercised': True}
        if words in (16, 256) and vm == 0:
            old = legacy(q, state, book, n)
            item['legacy_max_abs'] = float((actual.float() - old.float()).abs().max())
            item['legacy_bit_exact'] = bits_equal(actual, old)
            require(item['legacy_bit_exact'], 'unchanged V1 packed arithmetic changed ' + name)
        records.append(item)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-a2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visual-tokens', type=int, nargs='+', default=[32, 256])
    args = parser.parse_args()
    require(torch.cuda.is_available() and args.device.startswith('cuda'), 'actual CUDA required')
    require(all(n > 0 and n % 32 == 0 for n in args.visual_tokens), 'complete nonpadded Nv')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    device = torch.device(args.device)
    output = args.output.resolve()
    require(str(output).startswith('/media/') and '/baseline_matrix_v3/results/round12' in str(output), 'remote project data-disk output only')
    report = {'schema': 'trajkv.round12.static_technical.v1', 'status': 'RUNNING',
              'actual_CUDA': True, 'model_calls': 0, 'task_answer_events': 0,
              'pack_checks': pack_checks(device), 'value_checks': value_checks(device),
              'attention_checks': [item for n in args.visual_tokens for item in attention_checks(device, n, args.legacy_a2)]}
    torch.cuda.synchronize(device)
    report.update(status='PASS_ACTUAL_CUDA_PACKED_TECHNICAL_QA0', GPU=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, elapsed_seconds=time.perf_counter() - started,
        source_sha256={name: sha(Path(__file__).with_name(name)) for name in
            ('round12_static_technical_v1.py', 'round12_static_codec_v1.py', 'round12_static_attention_v1.py', 'round12_static_bank_v1.py')},
        legacy_source_sha256={name: sha(args.legacy_a2 / name) for name in ('fast_wht.py', 'round7_key_attention.py')})
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'STATIC_CUDA_TECHNICAL_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(path), 'sha256': sha(path), 'QA': 0}), flush=True)


if __name__ == '__main__':
    main()
