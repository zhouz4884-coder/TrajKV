"""CPU-only audit of pinned OnlyTerp primitives. No model/benchmark claim.

Exercises unmodified source functions and a lossless storage-only bitpack
prototype. Original states remain diagnostic references, so packed projections
below are not measured GPU-owner or speed results.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party/turboquant_onlyterp'
COMMIT = '1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj


def tensors(value, prefix='', seen=None):
    import torch
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        yield prefix, value; return
    if id(value) in seen: return
    seen.add(id(value))
    if isinstance(value, dict):
        for key, item in value.items(): yield from tensors(item, prefix + '.' + str(key), seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value): yield from tensors(item, prefix + '.' + str(index), seen)
    elif hasattr(value, '__dict__'):
        for key, item in vars(value).items(): yield from tensors(item, prefix + '.' + key, seen)


def storage(value):
    unique = {}; rows = []
    for name, tensor in tensors(value):
        store = tensor.untyped_storage(); identity = (str(tensor.device), store.data_ptr(), store.nbytes())
        charge = 0 if identity in unique else store.nbytes(); unique[identity] = name
        rows.append({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                     'storage_bytes': store.nbytes(), 'charged_bytes': charge})
    return {'bytes': sum(row['charged_bytes'] for row in rows), 'tensor_fields': rows}


def pack(values, bits):
    import torch
    values = values.long(); count = values.shape[-1]
    output = torch.zeros(*values.shape[:-1], (count * bits + 7) // 8, dtype=torch.uint8)
    for bit in range(bits):
        for coordinate in range(count):
            position = coordinate * bits + bit
            output[..., position // 8] |= (((values[..., coordinate] >> bit) & 1) << (position % 8)).to(torch.uint8)
    return output


def unpack(values, bits, count):
    import torch
    output = torch.zeros(*values.shape[:-1], count, dtype=torch.int64)
    for bit in range(bits):
        for coordinate in range(count):
            position = coordinate * bits + bit
            output[..., coordinate] |= (((values[..., position // 8].long() >> (position % 8)) & 1) << bit)
    return output


def packed_copy(state, regular_bits, outlier_bits, qjl=True):
    if state.pq.is_mixed_precision:
        pairs = [('regular_quantized_indices', regular_bits), ('outlier_quantized_indices', outlier_bits)]
    else: pairs = [('indices', regular_bits)]
    packed, restored = {}, {}
    for name, bits in pairs:
        raw = getattr(state.pq, name); encoded = pack(raw, bits)
        decoded = unpack(encoded, bits, raw.shape[-1]).to(raw.dtype)
        assert (raw == decoded).all()
        packed['pq.' + name] = encoded; restored[name] = decoded
    pq = replace(state.pq, **restored)
    operator = dict(vars(state.pq)); operator.update({name: None for name, _ in pairs})
    if qjl:
        raw = state.qjl.signs; signs = pack(raw, 1)
        recovered = unpack(signs, 1, raw.shape[-1]).to(raw.dtype)
        assert (recovered == raw).all()
        packed['qjl.signs'] = signs
        operator['qjl_r_norm'] = state.qjl.r_norm; operator['qjl_S'] = state.qjl.S
        restored_state = replace(state, pq=pq, qjl=replace(state.qjl, signs=recovered))
    else: restored_state = replace(state, pq=pq)
    return restored_state, {'packed': packed, 'operators': operator}


def compare(a, b):
    import torch
    delta = a.float() - b.float()
    return {'max_abs': delta.abs().max().item(),
            'relative_L2': (delta.norm() / b.float().norm().clamp_min(1e-12)).item(),
            'cosine': torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()}


def minimal_mse_copy(pq, regular_bits, outlier_bits, shared):
    """Lossless categorical/metadata packing; no quantizer or level change."""
    import torch
    token, constants, fields = {}, {}, {}
    def memo(tensor, kind, create):
        key = (kind, tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        if key not in shared: shared[key] = create()
        return shared[key]
    groups = [('regular', regular_bits), ('outlier', outlier_bits)] if pq.is_mixed_precision else [('', regular_bits)]
    for group, bits in groups:
        index_name = group + '_quantized_indices' if group else 'indices'
        norm_name = group + '_norm' if group else 'norm'
        rotation_name = 'rotation_' + group if group else 'rotation'
        codebook_name = 'codebook_' + group if group else 'codebook'
        indices = getattr(pq, index_name); encoded = pack(indices, bits)
        fields[index_name] = unpack(encoded, bits, indices.shape[-1]).to(indices.dtype)
        token[index_name] = encoded; token[norm_name] = getattr(pq, norm_name)
        rot = getattr(pq, rotation_name); new_rot = copy.copy(rot)
        if hasattr(rot, 'signs'):
            signs = memo(rot.signs, 'sign_bits', lambda: pack(((rot.signs+1)/2).long(), 1))
            constants[rotation_name + '.sign_bits'] = signs
            new_rot.signs = unpack(signs,1,rot.d).float()*2-1
        else: constants[rotation_name + '.P'] = rot.P
        fields[rotation_name] = new_rot
        cb = getattr(pq, codebook_name)
        constants[codebook_name + '.centroids'] = cb.centroids
        # Decision boundaries are encoder-only in an immutable visual prefix.
        fields[codebook_name] = replace(cb, boundaries=None)
        if group:
            map_name = group + '_indices'; mapping = getattr(pq, map_name)
            assert mapping.min() >= 0 and mapping.max() < 256
            packed_map = memo(mapping, 'uint8_map', lambda: mapping.to(torch.uint8))
            constants[map_name] = packed_map; fields[map_name] = packed_map.long()
    return replace(pq, **fields), {'tokens': token, 'constants': constants}


def projection(profile, n, d=128):
    whole, tokens, fixed = storage(profile), storage(profile['tokens']), storage(profile['constants'])
    assert whole['bytes'] == tokens['bytes'] + fixed['bytes']
    per_token = tokens['bytes'] / n
    return {'tensor_bytes_at_synthetic_N': whole['bytes'], 'per_new_visual_token_K_plus_V_bytes': per_token,
            'shared_constants_per_layer_head_bytes': fixed['bytes'], 'source_decode_equal_after_metadata_pack': True,
            'M1_28layers_4KVheads_conservative_two_global_int64_position_arrays': {
                str(nv): {'projected_complete_bytes': int(28*4*(per_token*nv + fixed['bytes']) + 16*nv),
                          'projected_complete_bits_per_KV_scalar': 8*(28*4*(per_token*nv + fixed['bytes']) + 16*nv)/(28*4*2*nv*d)}
                for nv in (256,768,1280,1792,2304)},
            'scope': 'Measured CPU tensor sizes projected to a proposed immutable packed owner; not accepted GPU residency or speed',
            'encoder_boundaries_and_original_unpacked_arrays_not_in_proposed_owner': True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    assert output.is_relative_to(Path('${EVIDENCE_ROOT}').resolve()), 'New raw CPU evidence uses E mirror'
    assert not output.exists()
    import torch
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    actual_commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual_commit == COMMIT
    source_hashes = {str(p.relative_to(SOURCE)): sha(p) for p in [SOURCE / 'src/cache.py', SOURCE / 'src/kernels.py',
                    SOURCE / 'src/lut_attention.py', SOURCE / 'src/vllm_integration.py', SOURCE / 'vllm_plugin/attention.py', SOURCE / 'LICENSE']}
    source = module('onlyterp_primary_cache_cpu_audit', SOURCE / 'src/cache.py')
    sys.modules['cache'] = source
    integration = module('onlyterp_integration_cpu_audit', SOURCE / 'src/vllm_integration.py')
    gen = torch.Generator().manual_seed(20261120)
    n, d = 64, 128
    amplitude = torch.ones(d); amplitude[:32] = 6
    k = (torch.randn(n, d, generator=gen) * amplitude).half()
    v = torch.randn(n, d, generator=gen).half()
    q = torch.randn(4, d, generator=gen).half()
    rows = []
    for name, mode, bits, online in [('H_mix12_offline', 'hadamard', 1, False),
                                    ('H_mix23_offline', 'hadamard', 2, False),
                                    ('D_mix12_offline', 'dense', 1, False),
                                    ('D_mix23_offline', 'dense', 2, False),
                                    ('D_mix34_offline', 'dense', 3, False),
                                    ('D_mix23_online', 'dense', 2, True)]:
        start = time.perf_counter()
        cfg = source.TurboQuantConfig(d=d, b_mse=bits, b_outlier=bits+1, mixed_precision=True,
            n_outlier=32, device=torch.device('cpu'), rotation_mode=mode, use_online_codebook=online)
        mixed = cfg.get_mixed_config(0, 0, k)
        rotation = cfg.make_rotation(0, 0); matrix = cfg.make_qjl_matrix(0, 0)
        kc = source.turboquant_encode_internal(k, cfg.codebook, rotation, matrix, mixed=mixed)
        vc = source.turboquant_encode_internal(v, cfg.codebook, rotation, matrix, mixed=mixed)
        ks, kp = packed_copy(kc, bits, bits+1); vs, vp = packed_copy(vc, bits, bits+1)
        k_mse, v_mse = source.polarquant_decode(kc.pq), source.polarquant_decode(vc.pq)
        k_prod = source.turboquant_decode_single(kc)
        kp_decode = source.turboquant_decode_single(ks)
        vp_decode = source.turboquant_decode_single(vs)
        assert torch.equal(k_prod, kp_decode) and torch.equal(source.turboquant_decode_single(vc), vp_decode)
        mse_kc, km = packed_copy(kc, bits, bits+1, qjl=False)
        mse_vc, vm = packed_copy(vc, bits, bits+1, qjl=False)
        assert torch.equal(source.polarquant_decode(mse_kc.pq), k_mse)
        shared = {}
        compact_k, compact_k_profile = minimal_mse_copy(kc.pq, bits, bits+1, shared)
        compact_v, compact_v_profile = minimal_mse_copy(vc.pq, bits, bits+1, shared)
        assert torch.equal(source.polarquant_decode(compact_k), k_mse)
        assert torch.equal(source.polarquant_decode(compact_v), v_mse)
        compact_profile = {'tokens': {'k': compact_k_profile['tokens'], 'v': compact_v_profile['tokens']},
                           'constants': {'k': compact_k_profile['constants'], 'v': compact_v_profile['constants']}}
        payload_bits = (kc.pq.regular_quantized_indices.shape[-1] * bits + kc.pq.outlier_quantized_indices.shape[-1] * (bits+1)) / d
        source_owner, prod_packed, mse_packed = storage({'k': kc, 'v': vc}), storage({'k': kp, 'v': vp}), storage({'k': km, 'v': vm})
        reference_logits = q.float() @ k.float().T / math.sqrt(d)
        source_mse_logits = q.float() @ k_mse.T / math.sqrt(d)
        source_prod_logits = q.float() @ k_prod.T / math.sqrt(d)
        reference_output = reference_logits.softmax(-1) @ v.float()
        row = {'name': name, 'rotation': mode, 'online_codebook': online, 'mse_bits_regular': bits, 'mse_bits_outlier': bits+1,
            'actual_group_dimensions': [len(mixed.regular_indices), len(mixed.outlier_indices)],
            'stored_code_dimensions': [kc.pq.regular_quantized_indices.shape[-1], kc.pq.outlier_quantized_indices.shape[-1]],
            'actual_MSE_index_payload_bits_after_tight_pack': payload_bits,
            'source_index_dtypes': [str(kc.pq.regular_quantized_indices.dtype), str(kc.qjl.signs.dtype)],
            'source_states_unique_tensor_bytes': source_owner['bytes'],
            'lossless_packed_PROD_projection_bytes_all_source_fields': prod_packed['bytes'],
            'lossless_packed_MSE_projection_bytes': mse_packed['bytes'],
            'normalizer': {'KV_scalars': 2*n*d, 'FP16_bytes': 2*n*d*2},
            'packed_MSE_projection_bits_including_constants_N64': mse_packed['bytes'] * 8 / (2*n*d),
            'packed_PROD_projection_bits_including_constants_N64': prod_packed['bytes'] * 8 / (2*n*d),
            'lossless_bitpack_roundtrip_K_and_V': True, 'packed_decode_equal_to_source': True,
            'MSE_K_reconstruction': compare(k_mse, k), 'PROD_K_reconstruction': compare(k_prod, k),
            'MSE_logits': compare(source_mse_logits, reference_logits), 'PROD_logits': compare(source_prod_logits, reference_logits),
            'MSE_attention_output': compare(source_mse_logits.softmax(-1) @ v_mse, reference_output),
            'PROD_K_MSE_V_attention_output': compare(source_prod_logits.softmax(-1) @ v_mse, reference_output),
            'lossless_minimal_MSE_projection': projection(compact_profile,n,d),
            'cpu_wall_ms_not_speed_benchmark': (time.perf_counter()-start)*1000,
            'owner_details': {'source': source_owner, 'packed_prod_projection': prod_packed, 'packed_mse_projection': mse_packed}}
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('name', 'actual_MSE_index_payload_bits_after_tight_pack', 'source_index_dtypes', 'packed_decode_equal_to_source')}), flush=True)
    # The literal high-level Cache uses Hadamard, offline codebooks, immediate
    # compression and list/view states. Count real backing allocations once.
    literal = source.TurboQuantCache(n_layers=1, n_heads=1, d=128, b_mse=2, device=torch.device('cpu'))
    literal.store_batch(0, 0, k, v)
    literal_storage = storage(literal)
    raw_out = literal.compute_attention(0, 0, q[0], qjl_score_weight=1.)
    # Buffer manager uses full PROD K decode, then adds QJL again. Reproduce
    # the same scalar expression and compare against the manager's raw scores.
    cfg = integration.TurboQuantConfig(num_layers=1, num_heads=1, num_kv_heads=1, head_dim=128,
        flush_interval=64, b_mse=2, device='cpu')
    manager = integration.TurboQuantKVManager(cfg)
    manager.store_batch(0, 0, k, v)
    manager_scores = manager._compressed_scores(0, 0, q[0].float())
    entries = manager.tq_cache.cache[0][0]
    full_decode = torch.cat([source.turboquant_decode_single(kc) for kc, _ in entries])
    mse_decode = torch.cat([source.polarquant_decode(kc.pq) for kc, _ in entries])
    mse_scores = mse_decode @ q[0].float() / math.sqrt(d)
    one_qjl_scores = full_decode @ q[0].float() / math.sqrt(d)
    double_qjl_scores = 2*one_qjl_scores - mse_scores
    buffer_check = {'manager_vs_single_QJL': compare(manager_scores, one_qjl_scores),
                    'manager_vs_double_QJL': compare(manager_scores, double_qjl_scores),
                    'double_QJL_max_abs': (manager_scores-double_qjl_scores).abs().max().item(),
                    'single_QJL_max_abs': (manager_scores-one_qjl_scores).abs().max().item(),
                    'buffer_interval': 64, 'default_interval_in_source': 128,
                    'compressed_only_matches_primary_cache': torch.equal(manager.fetch(0,0,q[0].float()), manager.tq_cache.compute_attention(0,0,q[0].float()))}
    # CPU semantics of the experimental Triton butterfly: upper=b-a rather
    # than a-b. This is a source-formula check, not a CUDA kernel execution.
    x = torch.randn(2,128,generator=gen)
    def experimental_butterfly(x):
        y=x.clone(); offsets=torch.arange(128)
        for stage in range(7):
            h=1<<stage; partner=offsets ^ h; old=y.clone()
            y=torch.where((offsets & h)!=0, old-old[:,partner], old+old[:,partner])
        return y / math.sqrt(128)
    butterfly = {'primary_FWHT_roundtrip': compare(source.fwht(source.fwht(x))/128, x),
                 'experimental_formula_roundtrip': compare(experimental_butterfly(experimental_butterfly(x)),x),
                 'experimental_formula_vs_primary': compare(experimental_butterfly(x),source.fwht(x)/math.sqrt(128)),
                 'scope': 'CPU reproduction of literal scalar butterfly expressions, not compiled GPU execution'}
    uniform_cfg=source.TurboQuantConfig(d=128,b_mse=1,mixed_precision=False,device=torch.device('cpu'))
    rot=uniform_cfg.make_rotation(0,0)
    uk=source.polarquant_encode(k,uniform_cfg.codebook,rot)
    uv=source.polarquant_encode(v,uniform_cfg.codebook,rot)
    common={}; uks,ukp=minimal_mse_copy(uk,1,1,common); uvs,uvp=minimal_mse_copy(uv,1,1,common)
    assert torch.equal(source.polarquant_decode(uks),source.polarquant_decode(uk))
    assert torch.equal(source.polarquant_decode(uvs),source.polarquant_decode(uv))
    uniform_projection=projection({'tokens':{'k':ukp['tokens'],'v':uvp['tokens']},
                                   'constants':{'k':ukp['constants'],'v':uvp['constants']}},n,d)
    assert buffer_check['double_QJL_max_abs'] < 1e-4 and buffer_check['single_QJL_max_abs'] > 1e-3
    assert butterfly['primary_FWHT_roundtrip']['relative_L2'] < 1e-6
    assert butterfly['experimental_formula_roundtrip']['relative_L2'] > .1
    assert all(sha(SOURCE / path)==value for path,value in source_hashes.items())
    result = {'status': 'PASS_CPU_SOURCE_AUDIT_WITH_CONFIRMED_LIMITATIONS', 'GPU_used': False,
        'model_or_task_quality_measured': False, 'actual_packed_CUDA_path_validated': False,
        'scope': 'Synthetic source math, physical CPU tensors and lossless packing projections; not E2E benchmark or deployment support',
        'torch_version':torch.__version__,'source_repo':'https://github.com/OnlyTerp/turboquant','source_commit':COMMIT,
        'source_sha256':source_hashes,'harness_sha256':sha(Path(__file__)),'source_unmodified_after_checks':True,'n':n,'d':d,'seed':20261120,
        'paper_arithmetic': {'outlier':32,'outlier_bits':3,'regular':96,'regular_bits':2,'actual_weighted_bits':(32*3+96*2)/128},
        'cases':rows,'uniform_MSE_H1_lossless_projection':uniform_projection,
        'literal_high_level_cache_unique_tensor_storage':literal_storage,
        'literal_high_level_cache_storage_bits_per_KV_scalar':literal_storage['bytes']*8/(2*n*d),
        'buffer_source_check':buffer_check,'experimental_butterfly_source_check':butterfly,
        'diagnostic_FP16_prototypes_retained':True,'do_not_treat_projection_bytes_as_actual_GPU_residency':True}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(rows),'literal_cache_bits':result['literal_high_level_cache_storage_bits_per_KV_scalar'],
                      'buffer_single_vs_double_max_abs':[buffer_check['single_QJL_max_abs'],buffer_check['double_QJL_max_abs']]}),flush=True)


if __name__=='__main__':main()
