"""Independent Round8 packed-reader arithmetic checks; no model/quality claim.

Default: CUDA is mandatory. --cpu-only validates hard codec arithmetic only.
The GPU grid uses GQA28/4, real Nv256/1792 plus Nv32, q1/11, K4/K8/position
books, binary/cube/joint V and zero/one/two sequential FP16 residual stages.
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
from round8_codec import (cube_book, cube_indices_from_codes, decode_key,
    decode_value_joint8, encode_key, encode_value_joint8, encode_sign_depth,
    key_book_layout, unpack_nibbles)
from quantizers import binary_centroids


def frozen_reference_functions():
    """Execute only pure-Torch frozen decoder/fold definitions on CPU hosts."""
    source = ROOT / 'code/experiments/fused_progressive.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('unpack_last_generic', 'decode_last_generic')]
    env = {'torch': torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), env)
    fold_path = Path(__file__).with_name('affine_rate_bank.py')
    tree = ast.parse(fold_path.read_text(encoding='utf-8'))
    fold = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fold')
    exec(compile(ast.Module(body=[fold], type_ignores=[]), str(fold_path), 'exec'), env)
    return env['decode_last_generic'], env['fold']


def cpu_controls():
    torch.manual_seed(20261118)
    decode_binary, fold = frozen_reference_functions()
    raw = torch.randn(1, 2, 7, 128)
    base = torch.randn(16, 4).half()
    gain_factor = torch.tensor([.93, 1.08])[None, :, None, None]
    code, gain = encode_key(raw, base, gain_factor=gain_factor)
    index = torch.arange(256)
    product = torch.cat((base[index & 15], base[index >> 4]), -1)
    pcode, pgain = encode_key(raw, product, gain_factor=gain_factor)
    if not torch.equal(code, pcode) or not torch.equal(gain, pgain) or not torch.equal(decode_key(code, gain, base), decode_key(code, gain, product)):
        raise RuntimeError('K8 Cartesian encoding/layout changed old K4 values')
    positional = base[None].repeat(32, 1, 1)
    pc, pg = encode_key(raw, positional, gain_factor=gain_factor)
    if not torch.equal(pc, code) or not torch.equal(pg, gain):
        raise RuntimeError('Repeated position books changed K4 assignment')
    source = torch.randn(1, 2, 32, 128).half()
    record = binary_centroids(source.transpose(2, 3).contiguous(), group_size=32)
    record = fold(record, 1, torch.tensor([.89, 1.13])[None, :, None, None],
                  torch.tensor([.12, -.08])[None, :, None, None])
    cube = cube_book()
    codes = cube_indices_from_codes(record[0], 32)
    reference = decode_binary(*record, 32, 1).transpose(2, 3).contiguous()
    if codes.numel() != 2 * 32 * 16 or not torch.equal(decode_value_joint8(codes, record[1], record[2], cube), reference):
        raise RuntimeError('Actual V codes do not transcode bitwise to cube')
    sample = source[:, :, :2].contiguous()
    selected = encode_value_joint8(sample, record[1], record[2], cube, chunk=11)
    for head in range(2):
        for token in range(2):
            for group in range(16):
                sl = slice(group * 8, group * 8 + 8)
                s = record[1][0, head, sl, 0].float()
                m = record[2][0, head, sl, 0].float()
                values = ((cube.float() * s).half().float() + m).half()
                distance = (sample[0, head, token, sl].float()[None] - values.float()).square().sum(-1)
                if int(selected[0, head, token, group]) != int(distance.argmin()):
                    raise RuntimeError('V encoder missed actual-FP16 native nearest word')
    return dict(passed=True, K_Cartesian_bitwise=True, K_position_identity=True,
        V_original_bit_codes_cube_bitwise=True, V_actual_FP16_nearest_blocks=64,
        scope='CPU synthetic codec arithmetic only; no GPU reader or task claim')


def gpu_checks():
    from fast_wht import torch_wht
    from round7_key_attention import attention as old_attention
    from round8_attention import attention
    torch.manual_seed(20261118)
    torch.backends.cuda.matmul.allow_tf32 = False
    decode_binary, fold = frozen_reference_functions()
    patterns = [[], [('v', 3, 0), ('k', 1, 0)],
        [('v', 3, 0), ('k', 1, 0), ('v', 3, 1), ('k', 1, 1)],
        [('k', 1, 0), ('v', 1, 0), ('v', 1, 1), ('k', 1, 1)]]
    results, encoder_checks = [], []
    with torch.inference_mode():
        for nv in (32, 256, 1792):
            hk, hq, d = 4, 28, 128
            source_k = torch.randn(1, hk, nv, d, device='cuda', dtype=torch.float16)
            source_v = torch.randn_like(source_k)
            mean = source_k.float().mean(-2, keepdim=True).half()
            rotated = torch_wht(source_k.float() - mean.float())
            factor = torch.tensor([.91, .98, 1.04, 1.11], device='cuda')[None, :, None, None]
            c4 = torch.randn(16, 4, device='cuda', dtype=torch.float16)
            code4, gain4 = encode_key(rotated, c4, gain_factor=factor)
            ix = torch.arange(256, device='cuda')
            cart = torch.cat((c4[ix & 15], c4[ix >> 4]), -1).contiguous()
            pos = (c4[None].repeat(32, 1, 1).float()
                   + .013 * torch.arange(32, device='cuda')[:, None, None]).half()
            k_specs = [('K4_16', c4, code4, gain4), ('K8_Cartesian', cart, code4, gain4)]
            for name, book in [('K4_256', torch.randn(256, 4, device='cuda', dtype=torch.float16)),
                               ('K8_256', torch.randn(256, 8, device='cuda', dtype=torch.float16)),
                               ('K4_position', pos)]:
                code, gain = encode_key(rotated, book, gain_factor=factor)
                k_specs.append((name, book, code, gain))
            record = binary_centroids(source_v.transpose(2, 3).contiguous(), group_size=nv)
            record = fold(record, 1, torch.tensor([.86, .95, 1.06, 1.14], device='cuda')[None, :, None, None],
                          torch.tensor([-.1, .03, .12, -.06], device='cuda')[None, :, None, None])
            vc, vs, vm = record
            cube = cube_book(device='cuda')
            cube_codes = cube_indices_from_codes(vc, nv)
            binary_values = decode_binary(*record, nv, 1).transpose(2, 3).contiguous()
            cube_values = decode_value_joint8(cube_codes, vs, vm, cube)
            if not torch.equal(binary_values, cube_values):
                raise RuntimeError('GPU cube transcode changed frozen affine values')
            joint_book = torch.randn(256, 8, device='cuda', dtype=torch.float16)
            joint_codes = torch.randint(0, 256, (1, hk, nv, 16), device='cuda', dtype=torch.uint8)
            joint_values = decode_value_joint8(joint_codes, vs, vm, joint_book)
            # Exercise the real GPU encoder, then independently enumerate every
            # actual hard word on a bounded sample rather than repeatedly
            # encoding full Nv for an arithmetic reader check.
            small = source_v[:, :, :2].contiguous()
            sj = encode_value_joint8(small, vs, vm, joint_book, chunk=29)
            for head in range(hk):
                for token in range(2):
                    for group in range(16):
                        sl = slice(group * 8, group * 8 + 8)
                        values = ((joint_book.float() * vs[0, head, sl, 0].float()).half().float() + vm[0, head, sl, 0].float()).half()
                        loss = (small[0, head, token, sl].float()[None] - values.float()).square().sum(-1)
                        if int(sj[0, head, token, group]) != int(loss.argmin()):
                            raise RuntimeError('GPU joint encoder does not choose exact native-FP16 nearest word')
            encoder_checks.append(dict(nv=nv, exact_GPU_nearest_blocks=hk * 2 * 16,
                V_cube_frozen_affine_bitwise=True, actual_joint_code_bytes_per_token=16))
            v_specs = [('V_binary', vc, None, binary_values),
                       ('V_cube', cube_codes, cube, cube_values),
                       ('V_joint', joint_codes, joint_book, joint_values)]
            for nq in (1, 11):
                q = torch.randn(1, nq, hq, d, device='cuda', dtype=torch.float16).transpose(1, 2)
                prefix = min(41, nv)
                nt = prefix + nq
                state = SimpleNamespace(direction_mean=mean,
                    visual_pos=torch.arange(0, 2 * nv, 2, device='cuda'),
                    text_pos=torch.cat((torch.arange(1, 2 * prefix, 2, device='cuda'),
                                       torch.arange(2 * nv, 2 * nv + nq, device='cuda'))),
                    text_length=nt, length=2 * nv + nq,
                    text_k=torch.randn(1, hk, nt + 64, d, device='cuda', dtype=torch.float16),
                    text_v=torch.randn(1, hk, nt + 64, d, device='cuda', dtype=torch.float16),
                    v_scale=vs, v_min=vm)
                beta = torch.tensor([-.15, .03, .08, .17], device='cuda')
                identities = {}
                for kname, book, code, gain in k_specs:
                    state.direction_code, state.direction_gain = code, gain
                    base_k = decode_key(code, gain, book)
                    for vname, vcode, vbook, base_v in v_specs:
                        state.v_code = vcode
                        for pattern_index, selected in enumerate(patterns):
                            lookup = torch.full((2, hk, 2), -1, device='cuda', dtype=torch.int32)
                            dense_k, dense_v = base_k.clone(), base_v.clone()
                            codes, amplitudes = [None] * len(selected), [None] * len(selected)
                            for axis, head in sorted({(axis, head) for axis, head, _ in selected}):
                                depth = sum(a == axis and h == head for a, h, _ in selected)
                                source = rotated[0, head] if axis == 'k' else source_v[0, head]
                                base = base_k[0, head] if axis == 'k' else base_v[0, head]
                                rc, ra, decoded, _ = encode_sign_depth(source, base, depth)
                                for stage in range(depth):
                                    slot = selected.index((axis, head, stage))
                                    codes[slot], amplitudes[slot] = rc[stage], ra[stage]
                                    lookup[int(axis == 'v'), head, stage] = slot
                                (dense_k if axis == 'k' else dense_v)[0, head] = decoded
                            if selected:
                                state.depth_residual_code, state.depth_residual_alpha = torch.stack(codes), torch.stack(amplitudes)
                                pool_bytes = state.depth_residual_code.numel() * 4 + state.depth_residual_alpha.numel() * 2
                                if pool_bytes != len(selected) * 18 * nv:
                                    raise RuntimeError('Residual owner contains unselected/padded token payload')
                            else:
                                pool_bytes = 0
                                for name in ('depth_residual_code', 'depth_residual_alpha'):
                                    if hasattr(state, name):
                                        delattr(state, name)
                            state.depth_residual_lookup = lookup
                            actual = attention(q, state, book, nv, beta, vbook=vbook)
                            legacy_equal = None
                            if kname in ('K4_16', 'K4_256') and vname == 'V_binary':
                                legacy_equal = bool(torch.equal(actual, old_attention(q, state, book, nv, beta)))
                                if not legacy_equal:
                                    raise RuntimeError('R8 original K4/Vbinary path differs from frozen Round7 reader')
                            key = (kname, vname, pattern_index)
                            identities[key] = actual
                            identity_equal = None
                            if vname == 'V_cube':
                                identity_equal = bool(torch.equal(actual, identities[(kname, 'V_binary', pattern_index)]))
                                if not identity_equal:
                                    raise RuntimeError('V cube identity changed actual attention output')
                            if kname == 'K8_Cartesian':
                                identity_equal = bool(torch.equal(actual, identities[('K4_16', vname, pattern_index)]))
                                if not identity_equal:
                                    raise RuntimeError('K8 Cartesian identity changed actual attention output')
                            qr = torch_wht(q).half().float()
                            zv = qr @ dense_k.repeat_interleave(7, 1).float().transpose(-1, -2) / 128 ** .5
                            zv += (q.float() * mean.repeat_interleave(7, 1).float()).sum(-1, keepdim=True) / 128 ** .5
                            zv += beta.repeat_interleave(7)[None, :, None, None]
                            zt = q.float() @ state.text_k[:, :, :nt].repeat_interleave(7, 1).float().transpose(-1, -2) / 128 ** .5
                            score = torch.cat((zv, zt), -1)
                            positions = torch.cat((state.visual_pos, state.text_pos))
                            qpos = torch.arange(state.length - nq, state.length, device='cuda')
                            score.masked_fill_(positions[None, None, None, :] > qpos[None, None, :, None], -torch.inf)
                            values = torch.cat((dense_v, state.text_v[:, :, :nt]), 2).repeat_interleave(7, 1)
                            expected = (score.softmax(-1) @ values.float()).half()
                            delta = actual.float() - expected.float()
                            max_abs = float(delta.abs().max())
                            rel = float(delta.norm() / expected.float().norm().clamp_min(1e-20))
                            layout = key_book_layout(book)
                            results.append(dict(nv=nv, nq=nq, query_heads=hq, kv_heads=hk,
                                K=kname, V=vname, selected_stages=selected,
                                actual_K_code_bytes=code.numel(), K_bytes_per_token=layout['index_bytes'],
                                V_code_bytes=vcode.numel() * vcode.element_size(),
                                V_book_bytes=0 if vbook is None else vbook.numel() * vbook.element_size(),
                                compact_residual_bytes=pool_bytes, legacy_R7_bitwise_equal=legacy_equal,
                                Cartesian_identity_bitwise_equal=identity_equal,
                                max_abs=max_abs, relative_l2=rel,
                                passed=max_abs < .003 and rel < .003))
    return results, encoder_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cpu = cpu_controls()
    if not args.cpu_only and not torch.cuda.is_available():
        raise RuntimeError('Actual GPU reader validation requires CUDA; CPU controls are a separate scope')
    results, encoders = ([], []) if args.cpu_only else gpu_checks()
    passed = cpu['passed'] and all(row['passed'] for row in results)
    names = ('round8_check.py', 'round8_codec.py', 'round8_attention.py', 'round7_key_attention.py',
             'round6_depth_codec.py', 'fast_wht.py', 'quantizers.py')
    report = dict(status='PASS' if passed else 'FAIL', gpu_executed=not args.cpu_only,
        cases=len(results), CPU_controls=cpu, GPU_encoder_controls=encoders, results=results,
        scope='Same-representation synthetic actual reader/codec arithmetic only; no model/task accuracy, quality or speed result',
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status=report['status'],
        gpu_executed=report['gpu_executed'], cases=len(results))), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], gpu_executed=report['gpu_executed'], cases=len(results))), flush=True)
    if not passed:
        raise RuntimeError('Round8 reader control failed')


if __name__ == '__main__':
    main()
