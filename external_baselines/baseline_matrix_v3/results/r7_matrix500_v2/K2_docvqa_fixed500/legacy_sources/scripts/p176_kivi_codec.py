"""Visual-only KIVI adapter using the unmodified official encoder and tail rules."""
from __future__ import annotations
import hashlib
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

EXPECTED_ENCODER_SHA256 = '3f1140b0faff2cb524a3eb684167e403dffd488477b301826497462fd36fe540'


def load_official(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_ENCODER_SHA256:
        raise ValueError(f'KIVI encoder hash mismatch: {digest}')
    spec = importlib.util.spec_from_file_location('p176_official_kivi_pack', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tail_lengths(tokens: int, residual: int) -> tuple[int, int]:
    if tokens < 0 or residual <= 0:
        raise ValueError('invalid token/residual count')
    return tokens % residual, min(tokens, residual)


def storage_bytes(tensors) -> int:
    return sum(t.untyped_storage().nbytes() for t in tensors if t is not None)


def check_groups(data, group_size: int):
    import torch
    if data.shape[-1] % group_size:
        raise ValueError('incomplete official KIVI group')
    groups = data.reshape(*data.shape[:-1], -1, group_size)
    scale = (groups.amax(-1) - groups.amin(-1))
    zeros = int((scale == 0).sum())
    if zeros or not torch.isfinite(groups).all():
        raise ValueError(f'official codec input has {zeros} zero-range groups or nonfinite values')


def decode_last(module, code, scale, minimum, group_size: int, bits: int):
    data = module.unpack_tensor(code, bits, pack_dim=3).to(dtype=scale.dtype)
    shape = data.shape
    grouped = data.reshape(*shape[:-1], -1, group_size)
    return (grouped * scale[..., None] + minimum[..., None]).reshape(shape).contiguous()


def verify_official(module, key, value, group_size: int, bits: int) -> dict:
    import torch
    key_t = key.transpose(2, 3).contiguous()
    check_groups(key_t, group_size)
    check_groups(value, group_size)
    k_code, k_scale, k_min = module.triton_quantize_and_pack_along_last_dim(key_t, group_size, bits)
    k_ref, ks_ref, km_ref = module.quant_and_pack_kcache(key, group_size, bits)
    v_code, v_scale, v_min = module.triton_quantize_and_pack_along_last_dim(value, group_size, bits)
    v_ref, vs_ref, vm_ref = module.quant_and_pack_vcache(value, group_size, bits)
    checks = {
        'key_codes': torch.equal(k_code.transpose(2, 3), k_ref),
        'key_scales': torch.equal(k_scale.transpose(2, 3), ks_ref.squeeze(-2)),
        'key_minima': torch.equal(k_min.transpose(2, 3), km_ref.squeeze(-2)),
        'value_codes': torch.equal(v_code, v_ref),
        'value_scales': torch.equal(v_scale, vs_ref.squeeze(-1)),
        'value_minima': torch.equal(v_min, vm_ref.squeeze(-1)),
        'key_decode': torch.equal(decode_last(module, k_code, k_scale, k_min, group_size, bits).transpose(2, 3),
                                  module.unpack_and_dequant_kcache(k_ref, ks_ref, km_ref, group_size, bits)),
        'value_decode': torch.equal(decode_last(module, v_code, v_scale, v_min, group_size, bits),
                                    module.unpack_and_dequant_vcache(v_ref, vs_ref, vm_ref, group_size, bits)),
    }
    if not all(checks.values()):
        raise RuntimeError(f'official Triton/PyTorch arithmetic mismatch: {checks}')
    return checks


def materialize_visual_codec(source, indices, module, bits: int, group_size: int = 32,
                             residual: int = 128, audit: bool = False):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = int(indices.numel())
    ktail, vtail = tail_lengths(tokens, residual)
    kcount, vcount = tokens - ktail, tokens - vtail
    holder = SimpleNamespace(key_cache=[], value_cache=[])
    packed_layers = []
    encoded_bytes = tail_bytes = visual_bytes = 0
    real_audits = []
    for layer, (key, value) in enumerate(zip(source.key_cache, source.value_cache)):
        k = key.index_select(2, indices).contiguous()
        v = value.index_select(2, indices).contiguous()
        if audit and layer in (0, len(source.key_cache) // 2, len(source.key_cache) - 1):
            real_audits.append(verify_official(module, k[:, :, :min(tokens, 128), :].contiguous(),
                                              v[:, :, :min(tokens, 128), :].contiguous(), group_size, bits))
        key_out, value_out = key.clone(), value.clone()
        buffers = []
        for data, count, axis in ((k, kcount, 'key'), (v, vcount, 'value')):
            if count == 0:
                continue
            quant = data[:, :, :count, :].contiguous()
            if axis == 'key':
                quant = quant.transpose(2, 3).contiguous()
            check_groups(quant, group_size)
            code, scale, minimum = module.triton_quantize_and_pack_along_last_dim(quant, group_size, bits)
            reconstructed = decode_last(module, code, scale, minimum, group_size, bits)
            if axis == 'key':
                reconstructed = reconstructed.transpose(2, 3).contiguous()
            if not torch.isfinite(reconstructed).all():
                raise ValueError('official KIVI reconstruction is nonfinite')
            target = key_out if axis == 'key' else value_out
            target.index_copy_(2, indices[:count], reconstructed)
            buffers.extend((code, scale, minimum))
        encoded_bytes += storage_bytes(buffers)
        exact_k = k[:, :, kcount:, :].contiguous().clone()
        exact_v = v[:, :, vcount:, :].contiguous().clone()
        tail_bytes += storage_bytes((exact_k, exact_v))
        buffers.extend((exact_k, exact_v))
        packed_layers.append(buffers)
        visual_bytes += (k.numel() + v.numel()) * k.element_size()
        holder.key_cache.append(key_out)
        holder.value_cache.append(value_out)
    index_buffer = indices.to(dtype=torch.int32).contiguous()
    total = encoded_bytes + tail_bytes + storage_bytes((index_buffer,))
    torch.cuda.synchronize()
    return holder, {'codec': 'official_kivi_triton_visual_adapter', 'bits': bits,
                    'group_size': group_size, 'residual_length_visual_tokens': residual,
                    'key_exact_tail_tokens': ktail, 'value_exact_tail_tokens': vtail,
                    'packed_code_scale_min_bytes': encoded_bytes, 'exact_tail_bytes': tail_bytes,
                    'index_bytes': storage_bytes((index_buffer,)), 'resident_bytes': total,
                    'resident_ratio': total / visual_bytes, 'visual_full_bytes': visual_bytes,
                    'pack_and_materialize_ms': 1000 * (time.perf_counter() - start),
                    'arithmetic_audits': real_audits,
                    'source_sha256': EXPECTED_ENCODER_SHA256}, (packed_layers, index_buffer)
