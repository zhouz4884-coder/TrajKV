"""CPU format/dtype/coded-math checks; no target CUDA or task-score claim."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasAdapter
from methods.calibquant_fp32_affine_adapter import CalibQuantFP32AffineAdapter
from methods.calibquant_adapter import repeat_kv


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.manual_seed(20261120)
    rows=[]
    for bits in (1,2):
        for n in (32,257):
            k=torch.randn(1,4,n,128).half();v=torch.randn_like(k);q=torch.randn(1,28,3,128).half()
            old=CalibQuantFP32BiasAdapter(bits,'torch_unfused').encode(k,v)
            new=CalibQuantFP32AffineAdapter(bits,'torch_unfused').encode(k,v)
            assert all(torch.equal(value,new.owner_tensors()[name]) for name,value in old.owner_tensors().items())
            assert old.byte_report()==new.byte_report()
            raw=new._raw_qk(q)
            s,z=repeat_kv(new.k_scale,7).double(),repeat_kv(new.k_zero,7).double()
            ideal=raw.double()/128**.5-(q.double()*s*z).sum(-1,keepdim=True)/128**.5
            lo,hi=ideal.amin(-1,keepdim=True),ideal.amax(-1,keepdim=True)
            ideal=((ideal-lo)/(hi-lo+1e-8))*(hi-3-lo)+lo
            actual=new.read(q)
            assert actual.dtype==torch.float32 and torch.isfinite(actual).all()
            torch.testing.assert_close(actual.double(),ideal,atol=3e-5,rtol=1e-5)
            w=torch.softmax(torch.randn(1,28,3,n),-1).half()
            raw_v=new._raw_weighted_codes(w)
            vs,vz=repeat_kv(new.v_scale,7).double(),repeat_kv(new.v_zero,7).double()
            ideal_v=raw_v.double()*vs-(vs*vz)*w.double().sum(-1,keepdim=True)
            output=new.weighted_value(w)
            assert output.dtype==torch.float16 and torch.isfinite(output).all()
            torch.testing.assert_close(output,ideal_v.half(),atol=.0005,rtol=.001)
            zero=new.read(torch.zeros_like(q));assert torch.isfinite(zero).all() and bool((zero==0).all())
            rows.append({'bits':bits,'n':n,'encoder_and_physical_owner_unchanged':True,
                         'calibrated_FP32_vs_boundary_conditioned_FP64_max_abs':float((actual.double()-ideal).abs().max()),
                         'weighted_Half_vs_boundary_conditioned_FP64_Half_max_abs':float((output-ideal_v.half()).abs().max()),
                         'zero_query_source_map_finite_without_clamp':True})
    result={'status':'PASS_CPU_FP32_AFFINE_FORMAT_DTYPE_CODED_MATH','GPU_used':False,'quality_scored':False,
            'original_CUDA_kernels_validated_by_this_gate':False,'reference_scope':'FP64 affine/calibration conditioned on the identical hard-FP16 raw kernel boundary',
            'cases':rows,'manifest':CalibQuantFP32AffineAdapter().manifest(),
            'source_sha256':{str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                              (Path(__file__),ROOT/'code/methods/calibquant_fp32_affine_adapter.py')}}
    with (ROOT/'audit/calibquant_fp32_affine_cpu_gate_v1.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(rows),'GPU_used':False}))


if __name__=='__main__':main()
