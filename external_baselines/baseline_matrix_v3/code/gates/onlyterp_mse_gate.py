"""OnlyTerp-MSE exact-source and lossless-owner gate. CPU default, no QA scores."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from methods.turboquant_onlyterp_mse_adapter import OnlyTerpMSEAdapter,pack_indices,unpack_indices,_source,_storage,READER,SOURCE_LOCK


def compare(a,b):
    delta=a.float()-b.float()
    return {'max_abs':delta.abs().max().item(),'relative_L2':(delta.norm()/b.float().norm().clamp_min(1e-12)).item()}


def run(device):
    torch.set_grad_enabled(False);torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(20261120);source=_source();results=[]
    for bits in (1,2,3,4,8):
        codes=torch.randint(0,1<<bits,(3,13),generator=gen,dtype=torch.int64).to(device)
        packed=pack_indices(codes,bits);decoded=unpack_indices(packed,bits,13,dtype=torch.int64)
        assert torch.equal(codes,decoded)
        assert packed.untyped_storage().nbytes()==3*((13*bits+7)//8)
    for n,h,bits,mixed,zero in [(17,1,1,True,False),(65,2,1,True,False),(129,2,2,True,False),
                              (17,1,1,False,False),(17,1,1,True,True),(1,1,1,True,False)]:
        k=torch.randn(1,h,n,128,generator=gen).to(device=device,dtype=torch.float16)
        v=torch.randn(1,h,n,128,generator=gen).to(device=device,dtype=torch.float16)
        if zero:k.zero_();v.zero_()
        positions=torch.arange(n*2,device=device)[::2].clone()
        adapter=OnlyTerpMSEAdapter(bits_regular=bits,mixed_precision=mixed)
        owner=adapter.encode(k,v,positions=positions,layer_idx=2)
        original_store={(str(x.device),x.untyped_storage().data_ptr()) for x in (k,v)}
        assert all((str(x.device),x.untyped_storage().data_ptr()) not in original_store for x in owner.named_tensors().values())
        before={key:(str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()) for key,t in owner.named_tensors().items()}
        actual_k,actual_v=owner.materialize()
        expected_k=[];expected_v=[];cfg=source.TurboQuantConfig(d=128,b_mse=bits,b_outlier=bits+1,
            mixed_precision=mixed,device=torch.device('cpu'),use_online_codebook=False,rotation_mode='hadamard')
        for head in range(h):
            m=cfg.get_mixed_config(2,head,k[0,head]) if mixed else None
            rotation=cfg.make_rotation(2,head)
            expected_k.append(source.turboquant_mse_decode(source.turboquant_mse_encode(k[0,head],cfg.codebook,rotation,mixed=m)))
            expected_v.append(source.turboquant_mse_decode(source.turboquant_mse_encode(v[0,head],cfg.codebook,rotation,mixed=m)))
        expected_k=torch.stack(expected_k)[None];expected_v=torch.stack(expected_v)[None]
        assert torch.equal(actual_k,expected_k) and torch.equal(actual_v,expected_v),'Source decode differs after storage-only pack'
        assert owner.k_norms.dtype==torch.float16 and owner.v_norms.dtype==torch.float16
        assert owner.regular_codebook.centroids.dtype==torch.float32
        assert owner.regular_codebook.boundaries.dtype==torch.float32
        nq=3;hq=h*3;prefix_len=2*n
        q=torch.randn(1,hq,nq,128,generator=gen).to(device=device,dtype=torch.float16)
        exact_k=torch.randn(1,h,n+3,128,generator=gen).to(device=device,dtype=torch.float16)
        exact_v=torch.randn(1,h,n+3,128,generator=gen).to(device=device,dtype=torch.float16)
        exact_pos=torch.cat((torch.arange(1,2*n,2,device=device),torch.arange(prefix_len,prefix_len+3,device=device)))
        qp=torch.arange(prefix_len,prefix_len+3,device=device);scale=.17
        out=owner.attend(q,exact_keys=exact_k,exact_values=exact_v,exact_positions=exact_pos,query_positions=qp,scale=scale)
        all_k=torch.cat((expected_k,exact_k),dim=-2).double().repeat_interleave(3,dim=1)
        all_v=torch.cat((expected_v,exact_v),dim=-2).double().repeat_interleave(3,dim=1)
        scores=q.double() @ all_k.transpose(-1,-2)*scale
        mask=torch.cat((positions,exact_pos))[None,None,None,:]<=qp[None,None,:,None]
        expected=(scores.masked_fill(~mask,-torch.inf).softmax(-1) @ all_v).to(q.dtype)
        torch.testing.assert_close(out,expected,rtol=.002,atol=.002)
        after={key:(str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()) for key,t in owner.named_tensors().items()}
        assert before==after,'Reader retained decoded data or replaced the persistent owner'
        full_report=_storage(owner.named_tensors());declared=owner.owner_report()
        assert full_report['total_bytes']==declared['stored_visual_KV_bytes']
        # Across layers static codebook aliases and original positions must
        # actually deduplicate, rather than being discounted on paper.
        other=adapter.encode(k,v,positions=positions,layer_idx=3)
        both=_storage({**{'first.'+name:t for name,t in owner.named_tensors().items()},
                       **{'second.'+name:t for name,t in other.named_tensors().items()}})
        assert both['total_bytes']<declared['stored_visual_KV_bytes']+other.owner_report()['stored_visual_KV_bytes']
        results.append({'n':n,'kv_heads':h,'bits_regular':bits,'mixed_precision':mixed,'zero':zero,
            'source_K_decode_max_abs':compare(actual_k,expected_k)['max_abs'],
            'source_V_decode_max_abs':compare(actual_v,expected_v)['max_abs'],
            'joint_GQA_causal_FP64_reference':compare(out,expected),'all_source_norm_and_codebook_dtypes_preserved':True,
            'original_full_KV_storage_not_owned':True,'temporary_decode_not_retained':True,
            'owner_report':{key:value for key,value in declared.items() if key!='storage'},
            'actual_unique_tensor_storage_bytes':full_report['total_bytes'],
            'two_layers_actual_unique_shared_bytes':both['total_bytes']})
    base=torch.randn(1,1,17,128,generator=gen).to(device=device,dtype=torch.float16)
    large=torch.arange(34,device=device)
    compact=OnlyTerpMSEAdapter().encode(base,base,positions=large[:17])
    assert compact.positions.untyped_storage().nbytes()==17*8
    try:OnlyTerpMSEAdapter().encode(base,base,positions=torch.zeros(17,device=device,dtype=torch.int64))
    except ValueError:pass
    else:raise AssertionError('Duplicate original positions were accepted')
    try:compact.attend(torch.randn(1,1,1,128,device=device,dtype=torch.float16),backend='experimental_triton')
    except ValueError:pass
    else:raise AssertionError('Silent backend fallback')
    return {'status':'PASS_ONLYTERP_MSE_SOURCE_PACKING_OWNER_REFERENCE_GATE','device':str(device),'GPU_used':device.type=='cuda',
        'CUDA_support_demonstrated':device.type=='cuda','model_or_task_quality_measured':False,
        'source_commit':'1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306','reader_backend':READER,
        'norm_dtype':'FP16 as original primary source; no additional precision cast',
        'cases':results,'categorical_tail_widths_checked':[1,2,3,4,8],
        'codebook_boundaries_and_CPU_GPU_constant_storages_retained_and_counted':True,
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/methods/turboquant_onlyterp_mse_adapter.py',ROOT/SOURCE_LOCK]},
        'torch_version':torch.__version__}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();device=torch.device(args.device)
    if device.type=='cuda':
        assert torch.cuda.get_device_capability(device)==(8,6),'Require actual RTX3090 SM86'
    result=run(device);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'GPU_used':result['GPU_used']}))


if __name__=='__main__':main()
