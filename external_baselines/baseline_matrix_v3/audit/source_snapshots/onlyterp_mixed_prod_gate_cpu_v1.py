"""CPU/CUDA source parity for single Gaussian QJL; no model/task scores."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from methods.turboquant_onlyterp_mixed_prod_adapter import OnlyTerpMixedProdAdapter,READER,SOURCE_LOCK
from methods.turboquant_onlyterp_mse_adapter import _source,_storage,unpack_indices


def difference(a,b):
    delta=a.float()-b.float()
    return {'max_abs':delta.abs().max().item(),'relative_L2':(delta.norm()/b.float().norm().clamp_min(1e-12)).item()}


def run(device):
    torch.set_grad_enabled(False);torch.set_num_threads(1);gen=torch.Generator().manual_seed(20261120)
    source=_source();records=[]
    for n,h,bits,zero in [(17,1,2,False),(65,2,2,False),(17,1,3,False),(65,2,3,False),(17,1,3,True)]:
        k=torch.randn(1,h,n,128,generator=gen).to(device=device,dtype=torch.float16)
        v=torch.randn(1,h,n,128,generator=gen).to(device=device,dtype=torch.float16)
        if zero:k.zero_();v.zero_()
        positions=torch.arange(0,2*n,2,device=device)
        owner=OnlyTerpMixedProdAdapter(bits_regular=bits).encode(k,v,positions=positions,layer_idx=2)
        source_k=[];source_vmse=[];source_vprod=[];q=torch.randn(1,h*3,3,128,generator=gen).to(device=device,dtype=torch.float16)
        cfg=source.TurboQuantConfig(d=128,b_mse=bits,b_outlier=bits+1,mixed_precision=True,device=torch.device('cpu'))
        literal=source.TurboQuantCache(n_layers=3,n_heads=h,d=128,b_mse=bits,b_outlier=bits+1,device=torch.device('cpu')) if device.type=='cpu' else None
        for head in range(h):
            mixed=cfg.get_mixed_config(2,head,k[0,head]);rotation=cfg.make_rotation(2,head)
            matrix=cfg.make_qjl_matrix(2,head).to(device)
            kc=source.turboquant_encode_internal(k[0,head],cfg.codebook,rotation,matrix,mixed=mixed)
            vc=source.turboquant_encode_internal(v[0,head],cfg.codebook,rotation,matrix,mixed=mixed)
            assert torch.equal(owner.gaussian_matrices[head],matrix)
            for kind,expected in [('k',kc),('v',vc)]:
                actual=owner.source_state(kind,head)
                assert torch.equal(actual.qjl.signs,expected.qjl.signs),'Packed QJL signs differ'
                assert torch.equal(actual.qjl.r_norm,expected.qjl.r_norm),'Source residual norm changed'
                assert actual.qjl.signs.dtype==torch.int64 and actual.qjl.r_norm.dtype==torch.float32
                assert torch.equal(source.turboquant_mse_decode(actual.pq),source.turboquant_mse_decode(expected.pq))
            source_k.append(source.turboquant_decode_single(kc))
            source_vmse.append(source.turboquant_mse_decode(vc.pq));source_vprod.append(source.turboquant_decode_single(vc))
            if literal is not None:literal.store_batch(2,head,k[0,head],v[0,head])
        source_k=torch.stack(source_k)[None];source_vmse=torch.stack(source_vmse)[None];source_vprod=torch.stack(source_vprod)[None]
        actual_k,actual_v=owner.materialize()
        assert torch.equal(actual_k,source_k) and torch.equal(actual_v,source_vmse)
        full=replace(owner,value_read_mode='full_prod_decode');fk,fv=full.materialize()
        assert torch.equal(fk,source_k) and torch.equal(fv,source_vprod)
        assert _storage(owner.named_tensors())['total_bytes']==_storage(full.named_tensors())['total_bytes']
        before={name:(str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()) for name,t in owner.named_tensors().items()}
        source_cache_max=0.
        if literal is not None:
            ours=owner.attend(q)
            expected=[]
            for qhead in range(h*3):
                expected.append(torch.stack([literal.compute_attention(2,qhead//3,q[0,qhead,t].float(),qjl_score_weight=1.) for t in range(3)]))
            expected=torch.stack(expected)[None].to(q.dtype)
            torch.testing.assert_close(ours,expected,rtol=.002,atol=.002);source_cache_max=difference(ours,expected)['max_abs']
        text_k=torch.randn(1,h,n+3,128,generator=gen).to(device=device,dtype=torch.float16)
        text_v=torch.randn(1,h,n+3,128,generator=gen).to(device=device,dtype=torch.float16)
        text_positions=torch.cat((torch.arange(1,2*n,2,device=device),torch.arange(2*n,2*n+3,device=device)))
        query_positions=torch.arange(2*n,2*n+3,device=device);mask=torch.cat((positions,text_positions))[None,None,None,:]<=query_positions[None,None,:,None]
        mode_errors={}
        for current,values in [(owner,source_vmse),(full,source_vprod)]:
            output=current.attend(q,exact_keys=text_k,exact_values=text_v,exact_positions=text_positions,query_positions=query_positions)
            all_k=torch.cat((source_k,text_k),dim=-2).double().repeat_interleave(3,dim=1)
            all_v=torch.cat((values,text_v),dim=-2).double().repeat_interleave(3,dim=1)
            scores=q.double()@all_k.transpose(-1,-2)*(128**-.5)
            reference=(scores.masked_fill(~mask,-torch.inf).softmax(-1)@all_v).to(q.dtype)
            torch.testing.assert_close(output,reference,rtol=.002,atol=.002)
            mode_errors[current.value_read_mode]=difference(output,reference)
        after={name:(str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()) for name,t in owner.named_tensors().items()}
        assert before==after,'Reference reader retained or changed data'
        original={(str(t.device),t.untyped_storage().data_ptr()) for t in (k,v)}
        assert all((str(t.device),t.untyped_storage().data_ptr()) not in original for t in owner.named_tensors().values())
        report=owner.owner_report();assert report['projection_matrix_bytes']==h*128*128*4
        assert report['stored_visual_KV_bytes']==_storage(owner.named_tensors())['total_bytes']
        assert owner.v_qjl_signs.numel()==owner.k_qjl_signs.numel()==h*n*16
        assert owner.v_qjl_rnorm.dtype==owner.k_qjl_rnorm.dtype==torch.float32
        records.append({'n':n,'kv_heads':h,'MSE_regular':bits,'MSE_outlier':bits+1,'zero':zero,
            'source_signs_norms_projection_and_PQ_decode_exact':True,'full_source_vector_K_prod_V_modes_exact':True,
            'source_Cache_default_output_max_abs':source_cache_max,'source_Cache_default_checked':literal is not None,
            'single_QJL_GQA_causal_FP64_errors':mode_errors,'V_QJL_stored_and_counted':True,
            'full_original_KV_storage_not_retained':True,'temporary_decode_not_retained':True,
            'owner':{name:value for name,value in report.items() if name!='storage'}})
    return {'status':'PASS_ONLYTERP_MIXED_PROD_SOURCE_PACKING_OWNER_GATE','device':str(device),'GPU_used':device.type=='cuda',
        'model_or_task_quality_measured':False,'source_runtime':'OnlyTerp primary src/cache.py only',
        'source_commit':'1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306','K_QJL_corrections':1,
        'source_V_default':'MSE read with QJL encoded/stored','explicit_V_variant':'full_prod_decode',
        'no_Rademacher_experimental_or_double_QJL_buffer_paths':True,'cases':records,'torch_version':torch.__version__,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            ROOT/'code/methods/turboquant_onlyterp_mixed_prod_adapter.py',ROOT/SOURCE_LOCK]}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();device=torch.device(args.device)
    if device.type=='cuda':assert torch.cuda.get_device_capability(device)==(8,6),'Require actual RTX3090 SM86'
    result=run(device);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'GPU_used':result['GPU_used']}))


if __name__=='__main__':main()
