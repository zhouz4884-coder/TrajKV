"""Mandatory actual CUDA packed-reader mean-header arithmetic; no model.

The correction sees every final FP16 decoded real token, including sequential
K residuals. GPU same-representation tests do not establish task/speed benefit.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from round8_codec import encode_key,decode_key
from round6_depth_codec import encode_sign_depth,decode_sign_stage
from round8_mean_correction import correct_state_mean_inplace,final_packed_key_mean,wht128
from round8_mean_runner import sha,source_fingerprints,tensor_inventory
from round8_check import frozen_reference_functions
from quantizers import binary_centroids


def require(value,message):
    if not value:raise RuntimeError(message)


def owner(state,book):
    values=[book]+[v for v in vars(state).values() if isinstance(v,torch.Tensor)]
    return sum({v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in values}.values())


def dense(q,state,book,decoder):
    from fast_wht import wht
    nv=state.visual_pos.numel();hk=state.text_k.shape[1];g=q.shape[1]//hk;nt=state.text_length
    dk=decode_key(state.direction_code,state.direction_gain,book)
    dv=decoder(state.v_code,state.v_scale,state.v_min,nv,1).transpose(2,3).contiguous()
    for axis,target in ((0,dk),(1,dv)):
        for head in range(hk):
            for stage in range(2):
                slot=int(state.depth_residual_lookup[axis,head,stage])
                if slot>=0:target[0,head]=decode_sign_stage(target[0,head],state.depth_residual_code[slot],state.depth_residual_alpha[slot])
    qr=wht(q,half=True).float()
    visual=qr@dk.repeat_interleave(g,1).float().transpose(-1,-2)/128**.5
    visual+=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1,keepdim=True)/128**.5
    text=q.float()@state.text_k[:,:,:nt].repeat_interleave(g,1).float().transpose(-1,-2)/128**.5
    scores=torch.cat((visual,text),-1)
    pos=torch.cat((state.visual_pos,state.text_pos[:nt]));qp=torch.arange(state.length-q.shape[2],state.length,device=q.device)
    scores.masked_fill_(pos[None,None,None,:]>qp[None,None,:,None],-torch.inf)
    values=torch.cat((dv,state.text_v[:,:,:nt]),2).repeat_interleave(g,1)
    result=(scores.softmax(-1)@values.float()).half()
    return result,scores,dk


def checks(emit):
    from round8_attention import attention
    from round7_key_attention import attention as old_attention
    decoder,fold=frozen_reference_functions();rows=[];corrections=[]
    for nv in (64,256):
        torch.manual_seed(20261119+nv);hk,hq,d=4,28,128
        source_k=torch.randn(1,hk,nv,d,device='cuda',dtype=torch.float16)
        source_v=torch.randn_like(source_k);book=(torch.randn(16,4,device='cuda')*.4+.1).half()
        mu=source_k.float().mean(-2,keepdim=True).half();z=wht128(source_k.float()-mu.float())
        kc,kg=encode_key(z,book,gain_factor=torch.tensor([.91,.98,1.04,1.11],device='cuda')[None,:,None,None])
        vc,vs,vm=fold(binary_centroids(source_v.transpose(2,3).contiguous(),group_size=nv),1,
            torch.ones(1,hk,1,1,device='cuda'),torch.zeros(1,hk,1,1,device='cuda'))
        base_k=decode_key(kc,kg,book);base_v=decoder(vc,vs,vm,nv,1).transpose(2,3).contiguous()
        # K heads respectively have0/1/2/0 stages; V shares one K head and has2.
        kr1,ka1,_,_=encode_sign_depth(z[0,1],base_k[0,1],1)
        kr2,ka2,_,_=encode_sign_depth(z[0,2],base_k[0,2],2)
        vr,va,_,_=encode_sign_depth(source_v[0,2],base_v[0,2],2)
        lookup=torch.full((2,hk,2),-1,device='cuda',dtype=torch.int32)
        lookup[0,1,0]=1;lookup[0,2,0]=3;lookup[0,2,1]=4;lookup[1,2,0]=0;lookup[1,2,1]=2
        state=SimpleNamespace(direction_code=kc,direction_gain=kg,direction_mean=mu.clone(),v_code=vc,v_scale=vs,v_min=vm,
            depth_residual_lookup=lookup,depth_residual_code=torch.stack((vr[0],kr1[0],vr[1],kr2[0],kr2[1])),
            depth_residual_alpha=torch.stack((va[0],ka1[0],va[1],ka2[0],ka2[1])),visual_pos=torch.arange(nv,device='cuda'),
            text_k=torch.randn(1,hk,3,d,device='cuda',dtype=torch.float16),
            text_v=torch.randn(1,hk,3,d,device='cuda',dtype=torch.float16),text_pos=torch.arange(3,device='cuda'))
        # Only final packed state survives into correction and reader tests.
        del source_k,source_v,z,base_k,base_v
        before=tensor_inventory([state],book);owned=owner(state,book);old_mu=state.direction_mean.clone()
        mean_small=final_packed_key_mean(state.direction_code,state.direction_gain,book,lookup[0],
            state.depth_residual_code,state.depth_residual_alpha,chunk=37)
        mean_large=final_packed_key_mean(state.direction_code,state.direction_gain,book,lookup[0],
            state.depth_residual_code,state.depth_residual_alpha,chunk=4096)
        require(torch.equal(mean_small,mean_large),'CUDA small/all-row chunk final FP16 mean differs')
        stats=correct_state_mean_inplace(state,book,chunk=4096)
        require(tensor_inventory([state],book)==before and owner(state,book)==owned,'Non-mean fields/pointers/owner changed')
        new_mu=state.direction_mean.clone();corrections.append(dict(nv=nv,stats=stats,owner_bytes=owned,
            small37_large4096_mean_bitwise=True,actual_correction_chunk=4096,apply_count=1))
        for nq in (1,11):
            torch.manual_seed(20261119+nv+nq)
            q=torch.randn(1,nq,hq,d,device='cuda',dtype=torch.float16).transpose(1,2)
            for nt in (0,41+nq):
                state.text_length=nt;state.length=nv+41+nq
                state.text_pos=torch.arange(nv-41,nv-41+nt,device='cuda')
                state.text_k=torch.randn(1,hk,max(nt,1)+16,d,device='cuda',dtype=torch.float16)
                state.text_v=torch.randn_like(state.text_k)
                outputs={};scores={};errors={};bitwise={}
                for mode,header in (('off',old_mu),('on',new_mu)):
                    state.direction_mean.copy_(header)
                    actual=attention(q,state,book,nv)
                    require(torch.equal(actual,old_attention(q,state,book,nv)),'Unchanged old reader differs')
                    expected,score,dk=dense(q,state,book,decoder)
                    delta=actual.float()-expected.float();maximum=float(delta.abs().max())
                    relative=float(delta.norm()/expected.float().norm().clamp_min(1e-20))
                    require(maximum<.003 and relative<.003,'Actual reader vs same physical-field dense error')
                    outputs[mode]=actual;scores[mode]=score;errors[mode]=dict(max_abs=maximum,relative_l2=relative)
                    bitwise[mode]=True
                # Score differences use the physical FP16 header difference.
                predicted=(q.double()*(new_mu.double()-old_mu.double()).repeat_interleave(7,1)).sum(-1)/128**.5
                visdiff=scores['on'][...,:nv].double()-scores['off'][...,:nv].double()
                common_error=float((visdiff-predicted[...,None]).abs().max())
                require(common_error<2e-5,'Physical-header visual logit common shift formula differs')
                conditional=float((scores['on'][...,:nv].softmax(-1)-scores['off'][...,:nv].softmax(-1)).abs().max())
                require(conditional<2e-6,'Within-visual dense conditional weights changed beyond reduction rounding')
                output_delta=float((outputs['on'].float()-outputs['off'].float()).abs().max())
                if nt:
                    odds=lambda s:s[...,:nv].double().logsumexp(-1)-s[...,nv:].double().logsumexp(-1)
                    odds_error=float((odds(scores['on'])-odds(scores['off'])-predicted).abs().max())
                    require(odds_error<2e-5,'Mixed-segment logodds shift formula differs')
                else:
                    odds_error=None
                    require(output_delta<.003,'No-text gauge output invariance exceeds FP16 reader tolerance')
                # Include query-transform rounding term in the actual header's
                # real-tensor mean-score residual bound; no exact bias claim.
                from fast_wht import wht
                m=dk.double().mean(-2,keepdim=True).repeat_interleave(7,1)
                r=(new_mu.double()+wht128(dk.double().mean(-2,keepdim=True))-old_mu.double()).repeat_interleave(7,1)
                eq=wht(q,half=True).double()-wht128(q.double())
                residual=(q.double()*r).sum(-1)/128**.5+(eq*m).sum(-1)/128**.5
                bound=(q.double().norm(dim=-1)*r.norm(dim=-1)+eq.norm(dim=-1)*m.norm(dim=-1))/128**.5
                require(bool((residual.abs()<=bound+1e-12).all()),'Rounded-Hq mean-score residual bound failed')
                row=dict(nv=nv,nq=nq,text_tokens=nt,modes=errors,old_reader_bitwise_identity=bitwise,
                    K_stages_per_head=[0,1,2,0],V_stages_head2=2,unchanged_non_mean_fields_pointers_owner=True,
                    visual_common_shift_max_abs_error=common_error,visual_conditional_max_abs=conditional,
                    mixed_text_logodds_max_abs_error=odds_error,actual_reader_on_off_output_max_abs=output_delta,
                    rounded_query_real_tensor_mean_score_residual_max=float(residual.abs().max()),
                    rounded_query_residual_bound_max=float(bound.max()),passed=True,
                    scope='actual CUDA synthetic same-representation arithmetic; no task or timing result')
                rows.append(row);emit(row)
    require(len(rows)==8,'Incomplete GPU arithmetic grid')
    return rows,corrections


def sources():
    result=source_fingerprints()
    for name in ('round8_mean_gpu_check.py','round8_check.py','round7_key_attention.py'):result[name]=sha(Path(__file__).with_name(name))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);before=sources()
    try:
        require(torch.cuda.is_available(),'Actual CUDA mandatory; CPU execution is not a substitute')
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode(),(a.output/'cases.jsonl').open('x',encoding='utf-8') as stream:
            def emit(row):
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(json.dumps(dict(nv=row['nv'],nq=row['nq'],nt=row['text_tokens'],passed=row['passed'])),flush=True)
            rows,stats=checks(emit)
        require(before==sources(),'Sources changed during check')
        report=dict(status='PASS',schema='trajkv.round8.mean-CUDA-reader-check.v1',GPU_executed=True,reader_cases=rows,
            correction_cases=stats,source_sha256=before,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
            added_persistent_tensor_bytes=0,scope='actual CUDA synthetic packed helper/reader arithmetic only; no model task quality or latency measurement')
        with (a.output/'check.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
        with (a.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(dict(status='PASS',GPU_executed=True,
            reader_cases=8,report_sha256=sha(a.output/'check.json'),cases_sha256=sha(a.output/'cases.jsonl')),stream)
    except BaseException as error:
        with (a.output/'failure.json').open('x',encoding='utf-8') as stream:json.dump(dict(status='FAILED_NEW_RUN_REQUIRED',
            exception_type=type(error).__name__,message=str(error),source_sha256=before),stream)
        raise


if __name__=='__main__':main()
