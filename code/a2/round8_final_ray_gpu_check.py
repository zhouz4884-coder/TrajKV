"""Actual CUDA final-ray helper and unchanged packed-reader arithmetic.

CUDA is mandatory by default; --gpu is a compatibility spelling. The grid is
Nv64/256 x q1/11 x off/mse/norm/projection =16 same-representation cases.
This is synthetic arithmetic, not a model, quality or timing measurement.
The frozen CPU checker is neither edited nor called as a GPU substitute.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from round8_codec import encode_key,decode_key,key_template
from round6_depth_codec import encode_sign_depth,decode_sign_stage
from round8_final_ray import (refit_key_final_ray,physical_ray_scale,final_decode,
    signs_from_code,metrics,MODES)
from round8_final_ray_runner import source_fingerprints,sha
from round8_check import frozen_reference_functions
from quantizers import binary_centroids


def require(value,message):
    if not value:raise RuntimeError(message)


def cuda_goal_and_fallback_controls():
    # Legal given-J norm/fold/signMAD witness: the non-MSE modes must accept
    # their own goal even when actual SSE rises; MSE must not add a cosine gate.
    z=torch.tensor([1.,0.],device='cuda').repeat(64)[None]
    book=torch.ones(16,4,device='cuda',dtype=torch.float16)
    factor=torch.tensor(.1767955869436264,device='cuda')
    code,gain=encode_key(z,book,gain_factor=factor)
    rc,ra,_,_=encode_sign_depth(z,decode_key(code,gain,book),1)
    sign=signs_from_code(rc[0]);template=key_template(code,book)
    old_y=final_decode(template,gain,ra[0],sign);old=metrics(z.double(),old_y)
    require(float(gain)==.125 and float(ra[0])==.5,'CUDA original norm/MAD witness differs')
    expected={'mse':40/34,'norm':(64/34)**.5,'projection':64/40};goals=[]
    for mode in ('mse','norm','projection'):
        ng,na,stats=refit_key_final_ray(z,book,code,gain,rc[0],ra[0],mode=mode,gain_factor=factor)
        actual_y=final_decode(template,ng,na,sign);actual=metrics(z.double(),actual_y)
        objective='sse' if mode=='mse' else mode
        require(abs(stats['scale_mean']-expected[mode])<1e-12 and stats['accepted_rows']==1,'CUDA goal scale/acceptance mismatch')
        require(bool((actual[objective]<old[objective]).all()),'CUDA own actual objective did not improve')
        if mode in ('norm','projection'):
            require(bool((actual['sse']>old['sse']).all()) and stats['accepted_rows_with_SSE_increase']==1,
                'Hidden CUDA SSE gate on norm/projection')
        else:
            old_cos=old['dot']/(old['z2']*old['y2']).sqrt()
            new_cos=actual['dot']/(actual['z2']*actual['y2']).sqrt()
            require(bool((new_cos<old_cos).all()),'MSE CUDA witness no longer tests a cosine decrease')
        goals.append(dict(mode=mode,stats=stats))
    fallback=[]
    def tensor(value):return torch.full((1,128),value,device='cuda',dtype=torch.float32)
    for name,source,template,g,a,s in (
        ('zero_source',tensor(0),tensor(1),torch.ones(1,1,device='cuda').half(),torch.zeros(1,device='cuda').half(),tensor(1)),
        ('zero_final_state',tensor(1),tensor(1),torch.zeros(1,1,device='cuda').half(),torch.zeros(1,device='cuda').half(),tensor(1)),
        ('negative_dot',tensor(-1),tensor(1),torch.ones(1,1,device='cuda').half(),torch.zeros(1,device='cuda').half(),tensor(1)),
        ('zero_dot',torch.tensor([1.,-1.],device='cuda').repeat(64)[None],tensor(1),torch.ones(1,1,device='cuda').half(),torch.zeros(1,device='cuda').half(),tensor(1)),
        ('candidate_underflow_zero',tensor(1e-12),tensor(1),torch.ones(1,1,device='cuda').half(),torch.zeros(1,device='cuda').half(),tensor(1)),
        ('candidate_field_overflow',tensor(100),tensor(1),torch.ones(1,1,device='cuda').half(),torch.full((1,),.99951171875,device='cuda').half(),tensor(-1))):
        for mode in ('mse','norm','projection'):
            ng,na,stats=physical_ray_scale(source,template,g,s,a,mode)
            require(torch.equal(ng,g) and torch.equal(na,a) and stats['accepted_rows']==0,'CUDA fallback changed original fields')
            fallback.append(dict(name=name,mode=mode,original_fields_bitwise=True))
    return dict(actual_CUDA_goal_witnesses=goals,actual_CUDA_fallback_cases=fallback,
        norm_projection_have_no_SSE_gate=True,mse_has_no_cosine_gate=True,
        scope='CUDA physical-goal witnesses and conservative fallback policies, not model quality')


def cuda_reader_checks(emit):
    from fast_wht import torch_wht
    from round8_attention import attention
    from round7_key_attention import attention as old_attention
    decoder,fold=frozen_reference_functions();rows=[];helper_rows=[]
    torch.manual_seed(20261118);torch.backends.cuda.matmul.allow_tf32=False
    for nv in (64,256):
        hk,hq,d=4,28,128
        source_k=torch.randn(1,hk,nv,d,device='cuda',dtype=torch.float16)
        source_v=torch.randn_like(source_k);source_v[0,1,:,0]=.125
        mean=source_k.float().mean(-2,keepdim=True).half()
        rotated=torch_wht(source_k.float()-mean.float())
        book=torch.randn(16,4,device='cuda',dtype=torch.float16)
        factor=torch.tensor([.91,.98,1.04,1.11],device='cuda')
        # Whole-layer initialization, then slice the selected head; the helper
        # verifies GIVEN J without rerunning argmin on a different batch shape.
        kc,kg=encode_key(rotated,book,gain_factor=factor[None,:,None,None])
        vc,vs,vm=fold(binary_centroids(source_v.transpose(2,3).contiguous(),group_size=nv),1,
            torch.tensor([.86,.95,1.06,1.14],device='cuda')[None,:,None,None],
            torch.tensor([-.1,.03,.12,-.06],device='cuda')[None,:,None,None])
        base_k=decode_key(kc,kg,book);base_v=decoder(vc,vs,vm,nv,1).transpose(2,3).contiguous()
        kr,ka,_,_=encode_sign_depth(rotated[0,1],base_k[0,1],1)
        vr,va,_,_=encode_sign_depth(source_v[0,1],base_v[0,1],1)
        old_final=decode_sign_stage(base_k[0,1],kr[0],ka[0])
        old_metrics=metrics(rotated[0,1].double(),old_final)
        original_J,original_sign=kc.clone(),torch.stack((vr[0],kr[0])).clone()
        physical_bytes=kc.numel()+kg.numel()*2+vc.numel()*4+vs.numel()*2+vm.numel()*2+2*nv*18
        for mode in MODES:
            gain,alpha,stats=refit_key_final_ray(rotated[0,1],book,kc[0,1],kg[0,1],kr[0],ka[0],mode=mode,gain_factor=factor[1])
            final=decode_sign_stage(decode_key(kc[0,1],gain,book),kr[0],alpha)
            actual_metrics=metrics(rotated[0,1].double(),final)
            if mode=='off':
                require(torch.equal(gain,kg[0,1]) and torch.equal(alpha,ka[0]),'CUDA off fields changed')
            else:
                objective='sse' if mode=='mse' else mode
                require(bool((actual_metrics[objective]<=old_metrics[objective]).all()),'CUDA per-row actual own objective worsened')
            helper_rows.append(dict(nv=nv,mode=mode,whole_layer_to_head_given_J=True,stats=stats))
            lookup=torch.full((2,hk,2),-1,device='cuda',dtype=torch.int32)
            # Same K/V head, intentionally distinct/interleaved pool slots.
            lookup[1,1,0]=0;lookup[0,1,0]=1
            state=SimpleNamespace(direction_code=kc.clone(),direction_gain=kg.clone(),direction_mean=mean,
                v_code=vc.clone(),v_scale=vs.clone(),v_min=vm.clone(),depth_residual_lookup=lookup,
                depth_residual_code=original_sign.clone(),depth_residual_alpha=torch.stack((va[0],ka[0])))
            state.direction_gain[0,1].copy_(gain);state.depth_residual_alpha[1].copy_(alpha)
            require(torch.equal(state.direction_code,original_J) and torch.equal(state.depth_residual_code,original_sign),
                'CUDA final-ray changed J/sign')
            require(torch.equal(state.v_code,vc) and torch.equal(state.v_scale,vs) and torch.equal(state.v_min,vm)
                and torch.equal(state.depth_residual_alpha[0],va[0]),'CUDA final-ray changed V fields')
            owned=state.direction_code.numel()+state.direction_gain.numel()*2+state.v_code.numel()*4+state.v_scale.numel()*2+state.v_min.numel()*2
            owned+=state.depth_residual_code.numel()*4+state.depth_residual_alpha.numel()*2
            require(owned==physical_bytes,'Actual CUDA field byte inventory changed')
            for nq in (1,11):
                prefix=41;nt=prefix+nq
                # Identical query/text inputs for every mode at this Nv/nq.
                torch.manual_seed(20261118+nv+nq)
                q=torch.randn(1,nq,hq,d,device='cuda',dtype=torch.float16).transpose(1,2)
                state.visual_pos=torch.arange(0,2*nv,2,device='cuda')
                state.text_pos=torch.cat((torch.arange(1,2*prefix,2,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
                state.text_length,state.length=nt,2*nv+nq
                state.text_k=torch.randn(1,hk,nt+16,d,device='cuda',dtype=torch.float16)
                state.text_v=torch.randn_like(state.text_k)
                require(not hasattr(state,'visual_k') and not hasattr(state,'visual_v'),'Reader retained source FP16 K/V')
                actual=attention(q,state,book,nv)
                off_identity=None
                if mode=='off':
                    off_identity=bool(torch.equal(actual,old_attention(q,state,book,nv)))
                    require(off_identity,'Frozen R7 off reader identity differs')
                # Dense reconstruction uses CURRENT physical gain/alpha, not
                # pre-refit base or an ideal real rescaling of old final y.
                dk=decode_key(state.direction_code,state.direction_gain,book)
                dv=decoder(state.v_code,state.v_scale,state.v_min,nv,1).transpose(2,3).contiguous()
                dk[0,1]=decode_sign_stage(dk[0,1],state.depth_residual_code[1],state.depth_residual_alpha[1])
                dv[0,1]=decode_sign_stage(dv[0,1],state.depth_residual_code[0],state.depth_residual_alpha[0])
                qr=torch_wht(q).half().float()
                zv=qr@dk.repeat_interleave(7,1).float().transpose(-1,-2)/d**.5
                zv+=(q.float()*mean.repeat_interleave(7,1).float()).sum(-1,keepdim=True)/d**.5
                zt=q.float()@state.text_k[:,:,:nt].repeat_interleave(7,1).float().transpose(-1,-2)/d**.5
                score=torch.cat((zv,zt),-1);positions=torch.cat((state.visual_pos,state.text_pos))
                qpos=torch.arange(state.length-nq,state.length,device='cuda')
                score.masked_fill_(positions[None,None,None,:]>qpos[None,None,:,None],-torch.inf)
                values=torch.cat((dv,state.text_v[:,:,:nt]),2).repeat_interleave(7,1)
                expected=(score.softmax(-1)@values.float()).half()
                delta=actual.float()-expected.float();maximum=float(delta.abs().max())
                relative=float(delta.norm()/expected.float().norm().clamp_min(1e-20))
                row=dict(nv=nv,nq=nq,mode=mode,query_heads=hq,kv_heads=hk,actual_unchanged_reader=True,
                    max_abs=maximum,relative_l2=relative,passed=maximum<.003 and relative<.003,
                    old_R7_off_bitwise_identity=off_identity,J_sign_V_unchanged=True,
                    actual_stored_fields_bytes=owned,added_retained_tensor_bytes=0,
                    scope='synthetic same-representation actual CUDA reader vs current physical-field dense decode')
                rows.append(row);emit(row)
    require(len(rows)==16 and all(row['passed'] for row in rows),'Actual CUDA final-ray reader grid failed')
    return rows,helper_rows


def sources():
    result=source_fingerprints()
    for name in ('round8_final_ray_gpu_check.py','round8_final_ray_check.py','round8_check.py','round7_key_attention.py'):
        result[name]=sha(Path(__file__).with_name(name))
    return result


def main():
    parser=argparse.ArgumentParser(description='Mandatory actual CUDA helper/reader check; no model')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true',help='compatibility flag; CUDA is always mandatory')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    before=sources()
    try:
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA is mandatory; no CPU substitute')
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode(),(args.output/'cases.jsonl').open('x',encoding='utf-8') as stream:
            def emit(row):
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(json.dumps(dict(nv=row['nv'],nq=row['nq'],mode=row['mode'],passed=row['passed'],max_abs=row['max_abs'])),flush=True)
            controls=cuda_goal_and_fallback_controls();rows,helpers=cuda_reader_checks(emit)
        require(sources()==before,'Frozen source changed during CUDA check')
        report=dict(status='PASS',schema='trajkv.round8.final-ray-CUDA-reader-check.v1',GPU_executed=True,
            reader_cases=rows,helper_cases=helpers,CUDA_goal_and_fallback_controls=controls,
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,source_sha256=before,
            scope='actual CUDA synthetic numerical/helper/reader evidence only; no HF model/task accuracy/timing measurement')
        (args.output/'check.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_executed=True,reader_cases=16,
            CUDA_own_goal_witnesses=3,CUDA_fallback_cases=18,report_sha256=sha(args.output/'check.json'),
            cases_sha256=sha(args.output/'cases.jsonl'))),encoding='utf-8')
        print(json.dumps(dict(status='PASS',GPU_executed=True,reader_cases=16)),flush=True)
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps(dict(status='FAILED_NEW_RUN_REQUIRED',exception_type=type(error).__name__,
            message=str(error),source_sha256=before,scope='synthetic CUDA check, no model/task output')),encoding='utf-8')
        raise


if __name__=='__main__':main()
