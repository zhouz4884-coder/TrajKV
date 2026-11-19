"""CPU controls and optional actual-CUDA reader checks for physical postcal.

Synthetic arithmetic only. No HF model, training, task labels or timing claim.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from round8_postcal_fields import FIELD_NAMES,identity_factors,postcal_fields,decode_postcal_visual_fields
from round8_postcal_runner import sha,source_fingerprints

def require(value,message):
    if not value:raise RuntimeError(message)
def bits_equal(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def fixture(nv,device,residual=True):
    torch.manual_seed(20261119+nv);h=4
    state=SimpleNamespace(direction_code=torch.randint(0,256,(1,h,nv,16),device=device,dtype=torch.uint8),
        direction_gain=(.1+torch.rand(1,h,nv,1,device=device)).half(),direction_mean=torch.randn(1,h,1,128,device=device).half(),
        v_code=torch.randint(-(2**31),2**31-1,(1,h,128,nv//32),device=device,dtype=torch.int32),
        v_scale=(.1+torch.rand(1,h,128,1,device=device)).half(),v_min=torch.randn(1,h,128,1,device=device).half(),
        depth_residual_lookup=torch.full((2,h,2),-1,device=device,dtype=torch.int32))
    state.v_min[0,0,0,0]=-0.
    if residual:
        state.depth_residual_lookup[0,1,0]=2;state.depth_residual_lookup[0,1,1]=0
        state.depth_residual_lookup[1,1,0]=3;state.depth_residual_lookup[1,3,0]=1
        state.depth_residual_code=torch.randint(-(2**31),2**31-1,(4,nv,4),device=device,dtype=torch.int32)
        state.depth_residual_alpha=(.05+torch.rand(4,nv,device=device)*.1).half()
    book=torch.randn(16,4,device=device).half()
    return state,book
def independent_decode(state,book):
    code=state.direction_code;indices=torch.stack((code&15,code>>4),-1).flatten(-2)
    key=(book[indices.long()].flatten(-2).float()*state.direction_gain.float()).half()
    nv=key.shape[-2];position=torch.arange(nv,device=book.device,dtype=torch.int64)
    level=((state.v_code.long().index_select(-1,position//32)>>(position%32))&1).transpose(-1,-2).float()
    value=((level*state.v_scale[...,0][:,:,None,:].float()).half().float()+state.v_min[...,0][:,:,None,:].float()).half()
    outk,outv=[],[]
    for head in range(4):
        k,v=key[0,head],value[0,head]
        for axis in range(2):
            current=k if axis==0 else v
            for stage in range(2):
                slot=int(state.depth_residual_lookup[axis,head,stage])
                if slot<0:continue
                words=state.depth_residual_code[slot].long()[...,None]
                signs=(((words>>torch.arange(32,device=book.device))&1).reshape(nv,128).float()*2-1)
                current=(current.float()+signs*state.depth_residual_alpha[slot].float()[:,None]).half()
            if axis==0:k=current
            else:v=current
        outk.append(k);outv.append(v)
    return torch.stack(outk)[None],torch.stack(outv)[None]

def arithmetic(device):
    reports=[]
    for residual in (False,True):
        state,book=fixture(64,device,residual);before={k:v.clone() for k,v in vars(state).items() if isinstance(v,torch.Tensor)}
        identity=identity_factors(4,device=device);same=postcal_fields(state,identity)
        require(all(bits_equal(getattr(same,k),v) for k,v in before.items()),'Theta0 physical fields not bitwise identical')
        factors=torch.tensor([[.85,1.1,.95,1.2],[1.1,.8,1.2,.9],[.05,-.04,.03,.02]],device=device,dtype=torch.float32)
        new=postcal_fields(state,factors)
        require(all(bits_equal(getattr(state,k),v) for k,v in before.items()),'Functional transform mutated input fields')
        require(all(getattr(new,k) is getattr(state,k) for k in ('direction_code','direction_mean','v_code','depth_residual_lookup')),'Unchanged ownership differs')
        require(bits_equal(new.direction_gain,(state.direction_gain.float()*factors[0][None,:,None,None]).half()),'K physical factor formula differs')
        expected_min=(state.v_min.float()+(.5*(1-factors[1])+factors[2]*factors[1])[None,:,None,None]*state.v_scale.float()).half()
        require(bits_equal(new.v_scale,(state.v_scale.float()*factors[1][None,:,None,None]).half()) and bits_equal(new.v_min,expected_min),'V originalS formula differs')
        if residual:
            expected=torch.stack((state.depth_residual_alpha[0].float()*factors[0,1],state.depth_residual_alpha[1].float()*factors[1,3],
                state.depth_residual_alpha[2].float()*factors[0,1],state.depth_residual_alpha[3].float()*factors[1,1])).half()
            require(bits_equal(new.depth_residual_alpha,expected),'Interleaved axis/head/stage alpha scaling differs')
        key,value=decode_postcal_visual_fields(state,factors,book,64);ek,ev=independent_decode(new,book)
        require(bits_equal(key,ek) and bits_equal(value,ev),'Differentiable decode differs from current physical fields')
        parameter=identity.clone().requires_grad_();k,v=decode_postcal_visual_fields(state,parameter,book,64)
        torch.manual_seed(917);loss=(k.float()*torch.randn_like(k.float())).sum()+(v.float()*torch.randn_like(v.float())).sum();loss.backward()
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool((parameter.grad.abs().sum(-1)>0).all()),'Theta0 autograd surrogate missing/nonfinite/zero block')
        reports.append(dict(residual=residual,identity_all_fields_bitwise=True,signed_zero_identity=True,input_unchanged=True,
            unchanged_J_sign_mean_Vcode=True,interleaved_axis_and_second_stage=True,physical_decode_bitwise=True,
            identity_autograd_block_l1=parameter.grad.abs().sum(-1).tolist(),
            gradient_scope='Torch cast/autograd surrogate; not a derivative certificate for discrete FP16 loss'))
    state,_=fixture(64,device,True);bad=[]
    for case in ('negative_gain','nonfinite_shift','overflow','duplicate_slot','stage1_without_stage0'):
        factors=identity_factors(4,device=device);current=SimpleNamespace(**vars(state))
        if case=='negative_gain':factors[0,0]=-1
        elif case=='nonfinite_shift':factors[2,0]=float('nan')
        elif case=='overflow':factors[0,0]=1e10
        else:
            current.depth_residual_lookup=state.depth_residual_lookup.clone()
            if case=='duplicate_slot':current.depth_residual_lookup[1,3,0]=2
            else:current.depth_residual_lookup[0,1,0]=-1
        try:postcal_fields(current,factors)
        except ValueError:bad.append(case)
        else:raise RuntimeError('Invalid physical contract accepted: '+case)
    return dict(cases=reports,rejected_invalid_cases=bad)

def gpu_readers():
    from round8_attention import attention
    from fast_wht import torch_wht
    rows=[]
    for nv in (64,256):
        state,book=fixture(nv,'cuda',True)
        for nq in (1,11):
            torch.manual_seed(20261119+nv+nq);prefix=41;nt=prefix+nq
            q=torch.randn(1,28,nq,128,device='cuda',dtype=torch.float16)
            state.visual_pos=torch.arange(0,2*nv,2,device='cuda')
            state.text_pos=torch.cat((torch.arange(1,2*prefix,2,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
            state.text_length,state.length=nt,2*nv+nq
            state.text_k=torch.randn(1,4,nt+16,128,device='cuda',dtype=torch.float16);state.text_v=torch.randn_like(state.text_k)
            original=attention(q,state,book,nv)
            for mode in ('identity','nonidentity'):
                factors=identity_factors(4,device='cuda') if mode=='identity' else torch.tensor([[.85,1.1,.95,1.2],[1.1,.8,1.2,.9],[.05,-.04,.03,.02]],device='cuda')
                current=postcal_fields(state,factors);actual=attention(q,current,book,nv)
                bitwise=bool(torch.equal(actual,original)) if mode=='identity' else None
                if mode=='identity':require(bitwise,'Actual unchanged reader theta0 output differs bitwise')
                key,value=independent_decode(current,book)
                qr=torch_wht(q).half().float();mean=current.direction_mean.repeat_interleave(7,1).float()
                zv=qr@key.repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
                zv+=(q.float()*mean).sum(-1,keepdim=True)/(128**.5)
                zt=q.float()@current.text_k[:,:,:nt].repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
                scores=torch.cat((zv,zt),-1);positions=torch.cat((current.visual_pos,current.text_pos));qpos=torch.arange(current.length-nq,current.length,device='cuda')
                scores.masked_fill_(positions[None,None,None,:]>qpos[None,None,:,None],-torch.inf)
                values=torch.cat((value,current.text_v[:,:,:nt]),2).repeat_interleave(7,1)
                expected=(scores.softmax(-1)@values.float()).half();diff=actual.float()-expected.float()
                maximum=float(diff.abs().max());relative=float(diff.norm()/expected.float().norm().clamp_min(1e-20))
                row=dict(nv=nv,nq=nq,mode=mode,max_abs=maximum,relative_l2=relative,passed=maximum<.003 and relative<.003,
                    actual_reader_unchanged=True,old_R42_identity_bitwise=bitwise,J_sign_mean_Vcode_unchanged=True)
                require(row['passed'],'Actual packed reader vs transformed physical dense decode failed');rows.append(row);print(json.dumps(row),flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sources=source_fingerprints();sources['round8_postcal_check.py']=sha(__file__)
    try:
        if args.gpu and not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
        torch.backends.cuda.matmul.allow_tf32=False
        controls=arithmetic('cuda' if args.gpu else 'cpu');readers=gpu_readers() if args.gpu else []
        before=source_fingerprints();before['round8_postcal_check.py']=sha(__file__);require(before==sources,'Sources changed during numeric check')
        owners=[dict(nv=nv,old43=4590*nv+89733,postcal42=4572*nv+89733+1344,delta=-18*nv+1344) for nv in (256,768,1280,1792,2304)]
        require(all(x['postcal42']<=x['old43'] for x in owners),'Postcal R42 price bound failed')
        report=dict(status='PASS',GPU_executed=args.gpu,controls=controls,reader_cases=readers,owner_controls=owners,source_sha256=sources,
            scope='Synthetic physical fields/autograd surrogate and optional actualCUDA reader only; no HF training/task/timing measurement')
        (args.output/'check.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_executed=args.gpu,reader_cases=len(readers),report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
        print(json.dumps(dict(status='PASS',GPU_executed=args.gpu,reader_cases=len(readers))),flush=True)
    except BaseException as e:
        (args.output/'failure.json').write_text(json.dumps(dict(status='FAILED_NEW_RUN_REQUIRED',exception_type=type(e).__name__,message=str(e),source_sha256=sources)),encoding='utf-8');raise
if __name__=='__main__':main()
