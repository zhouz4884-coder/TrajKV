"""Actual CUDA codec/reader checks for Round13, without loading a model.

Root controls launch and final whole-model checks. No task answers are created.
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
import round12_static_codec_v1 as old_codec
import round13_precision_codec_v1 as codec
import round13_precision_bank_v1 as bank


def require(value, message):
    if not value:
        raise ValueError('ROUND13_PRECISION_TECHNICAL: ' + message)


def bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_checks(device):
    reports = []
    for n, k, m in ((256, 55, 106), (768, 51, 97), (1280, 49, 95), (1792, 49, 94), (2304, 49, 94)):
        cap = (3816 + 18 * m) * n + 89737
        item = bank.token_tier_budget(n, k, cap)
        h = item['high_tokens']
        require(item['expected_owner_bytes'] <= cap, 'candidate exceeds reference R actual cap')
        if h < n:
            require(item['expected_owner_bytes'] + 1792 > cap, 'high-token count not maximum legal')
        reports.append(item)
    gen = torch.Generator(device=device).manual_seed(1301301)
    low_book = torch.randn((16, 4), device=device, generator=gen).half()
    high_book = torch.randn((256, 4), device=device, generator=gen).half()
    n = 256
    h = 117
    mask = torch.zeros(n, device=device, dtype=torch.bool)
    mask[torch.randperm(n, device=device, generator=gen)[:h]] = True
    rank = codec.tier_rank(mask)
    require(torch.equal(rank[mask].long(), torch.arange(h, device=device)), 'high ranks not dense')
    require(torch.equal((-rank[~mask] - 1).long(), torch.arange(n - h, device=device)), 'low ranks not dense')
    low_ids = torch.randint(0, 16, (1, 4, n - h, 32), device=device, generator=gen)
    high_ids = torch.randint(0, 256, (1, 4, h, 32), device=device, generator=gen)
    low, high = old_codec.pack_indices(low_ids, 4), old_codec.pack_indices(high_ids, 8)
    gain = torch.rand((1, 4, n, 1), device=device, generator=gen).half()
    decoded = codec.decode_token_key(low, high, gain, low_book, high_book, rank)
    require(bits_equal(decoded[..., ~mask, :], old_codec.decode_key(low, gain[..., ~mask, :], low_book)), 'low rows changed')
    require(bits_equal(decoded[..., mask, :], old_codec.decode_key(high, gain[..., mask, :], high_book)), 'high rows changed')
    require(low.numel() + high.numel() == 4 * ((n - h) * 16 + h * 32), 'compact key pools incorrectly charged')
    require(rank.untyped_storage().nbytes() == 4 * n, 'signed rank actual storage')
    return {'budget_examples': reports, 'nonaligned_high_rows': h, 'rank_and_two_pool_roundtrip': True,
            'two_pool_actual_bytes': low.numel() + high.numel(), 'rank_actual_bytes': rank.untyped_storage().nbytes()}


def transport_checks(device, binary):
    gen = torch.Generator(device=device).manual_seed(1301302)
    n = 256
    source = torch.randn((1, 4, n, 128), generator=gen, device=device).half()
    source[:, 0, :, 0] = 2.  # real degenerate raw binary channel
    raw_code, raw_scale, raw_min = binary(source.transpose(2, 3).contiguous(), group_size=n)
    nominal = torch.tensor([.71, .93, 1.13, 1.41], device=device)[None, :, None, None]
    offset = torch.tensor([.1, -.09, .04, -.02], device=device)[None, :, None, None]
    middle = raw_min.float() + .5 * raw_scale.float()
    old_scale = (raw_scale.float() * nominal).half()
    old_min = (middle + (offset - .5) * old_scale.float()).half()
    levels = old_codec.normalized_levels([-1.5104, -.45278, .45278, 1.5104], device)
    code, scale, minimum = codec.fit_v2_raw(source, levels)
    transported_scale, transported_min, diag = codec.transport_from_binary(raw_scale, raw_min,
        old_scale, old_min, scale, minimum, nominal_gain=nominal)
    require(diag['degenerate_channels'] >= 1, 'collapsed binary endpoint not exercised')
    require(diag['old_endpoint_float_max_abs'] < 2e-6, 'actual binary endpoint map mismatch')
    original_code, original_scale, original_min = old_codec.encode_v2_ratio(source, levels, nominal, offset)
    require(torch.equal(code, original_code), 'V raw assignment changed while testing affine transport')
    require(scale.untyped_storage().nbytes() == transported_scale.untyped_storage().nbytes() and
            minimum.untyped_storage().nbytes() == transported_min.untyped_storage().nbytes(), 'added affine metadata slots')
    changed = int((original_scale != transported_scale).sum()) + int((original_min != transported_min).sum())
    require(changed > 0, 'synthetic nonidentity f0 does not exercise transport correction')
    decoded = old_codec.half_value(code, transported_scale, transported_min, bits=2, levels=levels)
    require(bool(torch.isfinite(decoded).all()), 'transport produces nonfinite Half values')
    # Identity f0 must transport the existing raw four-level metadata unchanged.
    ids, ss, mm, identity = codec.encode_v2_transport(source, levels, raw_scale, raw_min,
        raw_scale, raw_min, torch.ones_like(nominal))
    require(bits_equal(ss, scale) and bits_equal(mm, minimum) and torch.equal(ids, code), 'identity affine transport changed raw V2')
    return {'N': n, 'raw_codes_same_as_original_V2R': True, 'existing_metadata_slots_only': True,
            'identity_affine_bit_exact': True, 'changed_Half_metadata_slots': changed,
            'original_vs_transport_reconstruction_max_abs': float((decoded.float() - old_codec.half_value(
                original_code, original_scale, original_min, bits=2, levels=levels).float()).abs().max()),
            'endpoint_fit': diag, 'identity_endpoint_fit': identity}


def add_v_residual(base, state):
    result = base.clone()
    if not hasattr(state, 'depth_residual_code'):
        return result
    for head in range(4):
        for depth in range(2):
            slot = int(state.depth_residual_lookup[1, head, depth])
            if slot >= 0:
                ids = old_codec.unpack_channel_codes(state.depth_residual_code[slot], 1, 128)
                sign = ids.float() * 2 - 1
                result[0, head] = (result[0, head].float() +
                    sign * state.depth_residual_alpha[slot].float()[:, None]).half()
    return result


def dense_reference(q, state, book, wht):
    k = codec.decode_token_key(state.direction_code, state.round13_high_code, state.direction_gain,
                              book, state.round13_high_book, state.round13_tier_rank)
    v = add_v_residual(old_codec.half_value(state.v_code, state.v_scale, state.v_min).transpose(2, 3).contiguous(), state)
    qr = wht(q, half=True)
    bias = (q.float() * state.direction_mean.repeat_interleave(7, 1).float()).sum(-1) / 128 ** .5
    sv = torch.matmul(qr.float(), k.repeat_interleave(7, 1).float().transpose(-2, -1)) / 128 ** .5 + bias[..., None]
    st = torch.matmul(q.float(), state.text_k.repeat_interleave(7, 1).float().transpose(-2, -1)) / 128 ** .5
    scores = torch.cat((sv, st), -1)
    positions = torch.cat((state.visual_pos, state.text_pos))
    qpos = torch.arange(state.length - q.shape[-2], state.length, device=q.device)
    scores.masked_fill_(positions[None, None, None] > qpos[None, None, :, None], -float('inf'))
    values = torch.cat((v.repeat_interleave(7, 1), state.text_v.repeat_interleave(7, 1)), -2)
    return torch.matmul(torch.softmax(scores, -1).float(), values.float()).half()


def attention_checks(device, n, a2):
    sys.path.insert(0, str(a2))
    wht = importlib.import_module('fast_wht').wht
    reader = importlib.import_module('round13_precision_attention_v1').attention
    old_reader = importlib.import_module('round12_static_attention_v1').attention
    gen = torch.Generator(device=device).manual_seed(1301303 + n)
    low_book = (torch.randn((16, 4), generator=gen, device=device) * .5).half()
    high_book = (torch.randn((256, 4), generator=gen, device=device) * .5).half()
    reports = []
    for high_n in sorted({0, n // 2 + 1, n}):
        high = torch.zeros(n, dtype=torch.bool, device=device)
        high[torch.randperm(n, generator=gen, device=device)[:high_n]] = True
        ranks = codec.tier_rank(high)
        low_n = n - high_n
        lc = old_codec.pack_indices(torch.randint(0, 16, (1, 4, low_n, 32), device=device, generator=gen), 4)
        hc = old_codec.pack_indices(torch.randint(0, 256, (1, 4, high_n, 32), device=device, generator=gen), 8)
        lookup = torch.full((2, 4, 2), -1, device=device, dtype=torch.int32)
        lookup[1, 1, 0], lookup[1, 1, 1] = 0, 1
        state = SimpleNamespace(direction_code=lc,
            direction_gain=(torch.rand((1, 4, n, 1), generator=gen, device=device) + .2).half(),
            direction_mean=(torch.randn((1, 4, 1, 128), generator=gen, device=device) * .04).half(),
            round13_high_code=hc, round13_high_book=high_book, round13_tier_rank=ranks,
            round13_high_count=high_n, round13_low_count=low_n,
            v_code=old_codec.pack_channel_codes(torch.randint(0, 2, (1, 4, 128, n), device=device, generator=gen), 1),
            v_scale=(torch.rand((1, 4, 128, 1), generator=gen, device=device) + .1).half(),
            v_min=(torch.randn((1, 4, 128, 1), generator=gen, device=device) * .2 - .5).half(),
            visual_pos=torch.arange(1, n + 1, device=device),
            text_pos=torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.arange(n + 1, n + 21, device=device))),
            text_k=torch.randn((1, 4, 21, 128), generator=gen, device=device).half(),
            text_v=torch.randn((1, 4, 21, 128), generator=gen, device=device).half(),
            text_length=21, length=n + 21, depth_residual_lookup=lookup, round12_vmode=0,
            depth_residual_code=old_codec.pack_channel_codes(torch.randint(0, 2, (2, n, 128), device=device, generator=gen), 1),
            depth_residual_alpha=(torch.rand((2, n), device=device, generator=gen) * .03).half())
        q = torch.randn((1, 28, 17, 128), generator=gen, device=device).half()
        # Use contiguous post-boundary query positions; full union includes text.
        state.length = n + 21
        actual = reader(q, state, low_book, n)
        reference = dense_reference(q, state, low_book, wht)
        error = (actual.float() - reference.float()).abs()
        require(bool(torch.isfinite(actual).all()) and float(error.max()) <= .008,
                'mixed packed/dense mismatch: ' + str(float(error.max())))
        item = {'N': n, 'high_tokens': high_n, 'low_tokens': low_n,
                'high_count_non32_aligned': bool(high_n % 32), 'dense_max_abs': float(error.max()),
                'dense_rmse': float(error.square().mean().sqrt()), 'dense_tolerance': .008,
                'two_sequential_V_Half_residual_stages': True, 'K_residual_slots': 0,
                'actual_low_index_bytes': lc.numel(), 'actual_high_index_bytes': hc.numel()}
        if high_n in (0, n):
            saved = state.direction_code
            state.direction_code = lc if high_n == 0 else hc
            old = old_reader(q, state, low_book if high_n == 0 else high_book, n)
            state.direction_code = saved
            item['uniform_endpoint_legacy_bit_exact'] = bits_equal(actual, old)
            item['uniform_endpoint_legacy_max_abs'] = float((actual.float() - old.float()).abs().max())
            require(item['uniform_endpoint_legacy_bit_exact'], 'uniform K endpoint reader arithmetic changed')
        reports.append(item)
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--legacy-a2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--visual-tokens', type=int, nargs='+', default=[32, 256])
    args = p.parse_args()
    require(torch.cuda.is_available() and args.device.startswith('cuda'), 'actual CUDA required')
    require(all(n >= 32 and n % 32 == 0 for n in args.visual_tokens), 'complete source V words')
    output = args.output.resolve()
    require(str(output).startswith('/media/') and '/baseline_matrix_v3/results/round13' in str(output),
            'remote project data-disk output only')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(args.legacy_a2))
    binary = importlib.import_module('quantizers').binary_centroids
    started = time.perf_counter()
    device = torch.device(args.device)
    report = {'schema': 'trajkv.round13.precision_technical.v1', 'actual_CUDA': True,
        'model_calls': 0, 'task_answer_events': 0, 'contract': contract_checks(device),
        'transport': transport_checks(device, binary),
        'attention': [item for n in args.visual_tokens for item in attention_checks(device, n, args.legacy_a2)]}
    torch.cuda.synchronize(device)
    report.update(status='PASS_ACTUAL_CUDA_PRECISION_TECHNICAL', GPU=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, elapsed_seconds=time.perf_counter() - started,
        source_sha256={name: sha(Path(__file__).with_name(name)) for name in
            (*bank.RUNTIME_FILES, 'round13_precision_technical_v1.py')})
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'PRECISION_CUDA_TECHNICAL_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(path), 'sha256': sha(path), 'QA': 0}), flush=True)


if __name__ == '__main__':
    main()
