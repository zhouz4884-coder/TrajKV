"""CPU-only source-policy, GQA, causality and physical-owner checks."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from methods.turboquant_adapter import _upstream, storage_bytes
from methods.turboquant_buffered_adapter import TurboQuantBufferedAdapter


def main():
    torch.set_grad_enabled(False); torch.set_num_threads(2); torch.manual_seed(20261120)
    _upstream()
    from turboquant.kv_cache import TurboQuantKVCache
    cases = []
    for hkv, hq in ((4, 28), (8, 32)):
        for n, gap, window in ((17, 1, 128), (129, 2, 128), (257, 1, 128), (257, 3, 0)):
            for vb in (2, 4):
                vp = torch.arange(n) * gap + 3
                boundary = int(vp[-1]) + 1
                full_k = torch.randn(1, hkv, boundary, 128).half(); full_v = torch.randn_like(full_k)
                k, v = full_k[:, :, vp], full_v[:, :, vp]
                i = 7
                adapter = TurboQuantBufferedAdapter(recent_window=window, value_bits=vb, seed=42+6*i)
                owner = adapter.encode(k, v, positions=vp, layer_idx=i)
                tail_mask = vp >= boundary - window
                assert owner.k_tail.shape[-2] == int(tail_mask.sum())
                assert torch.equal(owner.tail_positions, vp[tail_mask])
                assert torch.equal(owner.k_tail, k[:, :, tail_mask]) and torch.equal(owner.v_tail, v[:, :, tail_mask])
                original_ptrs = {full_k.untyped_storage().data_ptr(), full_v.untyped_storage().data_ptr(),
                                 k.untyped_storage().data_ptr(), v.untyped_storage().data_ptr()}
                assert all(t.untyped_storage().data_ptr() not in original_ptrs for t in owner.named_tensors().values() if t.numel())
                source = TurboQuantKVCache(128, key_bits=3, value_bits=vb, buffer_size=window,
                                           device=torch.device('cpu'), dtype=torch.float32, layer_idx=i)
                source.prefill(full_k.float(), full_v.float())
                norm_errors = {}
                if owner.quantized_owner is not None:
                    quant_vp = vp[~tail_mask]
                    actual = owner.quantized_owner
                    for field in ('mse_indices','qjl_signs','norms','residual_norms'):
                        expected = getattr(source.key_quantized, field)[:, :, quant_vp]
                        tensor = getattr(actual.key_quantized, field)
                        if tensor.is_floating_point():
                            norm_errors[field] = float((tensor-expected).abs().max())
                            # Full-prefix and selected-row GEMM shapes can
                            # round FP32 residual reconstruction differently.
                            torch.testing.assert_close(tensor, expected, atol=1e-5, rtol=1e-6)
                        else: assert torch.equal(tensor, expected), field
                    for field in ('data','scales','zeros'):
                        assert torch.equal(getattr(actual.value_quantized, field), getattr(source.value_quantized, field)[:, :, quant_vp]), field
                else: assert bool(tail_mask.all())
                text_mask = torch.ones(boundary, dtype=torch.bool); text_mask[vp] = False
                tp = torch.cat((torch.arange(boundary)[text_mask], torch.arange(boundary, boundary+3)))
                tk = torch.cat((full_k[:, :, text_mask], torch.randn(1,hkv,3,128).half()), -2)
                tv = torch.cat((full_v[:, :, text_mask], torch.randn_like(tk[:, :, -3:])), -2)
                q = torch.randn(1,hq,3,128).half(); qp = torch.arange(boundary,boundary+3)
                kh, vh = owner.materialize(torch.float32)
                # Disabling wrapper materialize demonstrates that attend does
                # not route through this full-visual diagnostic decoder.
                owner.materialize = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('full visual decoder called'))
                actual = owner.attend(q, exact_keys=tk, exact_values=tv, exact_positions=tp,
                                      query_positions=qp, backend='torch_dequant')
                K = torch.cat((kh,tk.float()),-2).repeat_interleave(hq//hkv,1)
                V = torch.cat((vh,tv.float()),-2).repeat_interleave(hq//hkv,1)
                mask = torch.cat((vp,tp))[None,:] <= qp[:,None]
                logits = (q.double() @ K.double().transpose(-1,-2)) / (128**.5)
                expected = (torch.softmax(logits.masked_fill(~mask[None,None],-torch.inf),-1) @ V.double()).to(q.dtype)
                torch.testing.assert_close(actual,expected,atol=.001,rtol=.005)
                changed = tv.clone(); changed[:, :, -1] += 100
                output = owner.attend(q, exact_keys=tk, exact_values=changed, exact_positions=tp,
                                     query_positions=qp, backend='torch_dequant')
                assert torch.equal(actual[:, :, 0], output[:, :, 0]), 'Future suffix value leaked to earlier query'
                report = owner.owner_report()
                assert report['all_owner_bytes_including_constants'] == storage_bytes(owner.named_tensors())['total_bytes']
                cases.append({'hkv':hkv,'hq':hq,'nvisual':n,'logical_gap':gap,'window':window,'value_bits':vb,
                              'tail_visual_tokens':int(tail_mask.sum()), 'source_integer_codes_exact':True,
                              'source_FP32_norm_max_abs_errors':norm_errors,
                              'GQA_causal_max_abs_error':float((actual-expected).abs().max()),
                              'complete_owner_bytes':report['all_owner_bytes_including_constants']})
    result = {'status':'PASS_CPU_SOURCE_POLICY_GQA_CAUSAL_BYTES', 'GPU_used':False,
              'target_CUDA_packed_kernel_validated_by_this_gate':False,
              'cases':cases, 'torch_version':torch.__version__,
              'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__),ROOT/'code/methods/turboquant_buffered_adapter.py')}}
    output = ROOT/'audit/turboquant_buffered_cpu_gate_v1.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(cases),'GPU_used':False}))


if __name__ == '__main__':main()
