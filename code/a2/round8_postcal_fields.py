"""Differentiable physical post-residual fields; integer codes never change.

The table is absolute FP32[3,H]: positive fK, positive fV, finite V shift o.
This module imports Torch only and never imports a model, Triton or a bank.
"""
from types import SimpleNamespace
import torch

FIELD_NAMES=('direction_code','direction_gain','direction_mean','v_code','v_scale','v_min',
    'depth_residual_code','depth_residual_alpha','depth_residual_lookup')

class _IdentitySignedZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, proposal, identity):
        # m+0 can canonicalize -0 to +0. Preserve the identity storage bits,
        # retaining the affine proposal's derivative at that same value.
        return torch.where(identity,original,proposal)
    @staticmethod
    def backward(ctx, gradient):return None,gradient,None

def identity_factors(heads, *, device=None):
    return torch.stack((torch.ones(heads,device=device,dtype=torch.float32),
        torch.ones(heads,device=device,dtype=torch.float32),
        torch.zeros(heads,device=device,dtype=torch.float32)))

def postcal_fields(state, factors_3H):
    """Return a new state, sharing unchanged fields and preserving autograd.

    Existing physical tensors are not mutated. Lookup axes are K then V;
    arbitrary interleaved compact slots and sequential stage0/1 are supported.
    The actual all-frame binary-V FP16 fields are mandatory.
    """
    if not isinstance(factors_3H,torch.Tensor) or factors_3H.dtype!=torch.float32 or factors_3H.ndim!=2 or factors_3H.shape[0]!=3:
        raise ValueError('Absolute FP32[3,H] post factors required')
    heads=factors_3H.shape[1];device=factors_3H.device
    if heads<1 or not bool(torch.isfinite(factors_3H).all()) or not bool((factors_3H[:2]>0).all()):
        raise ValueError('Positive finite K/V factors and finite shift required')
    residual_present=hasattr(state,'depth_residual_alpha')
    if residual_present!=hasattr(state,'depth_residual_code'):raise ValueError('Both compact residual fields required together')
    for name in FIELD_NAMES:
        if name in ('depth_residual_code','depth_residual_alpha') and not residual_present:continue
        if not hasattr(state,name):raise ValueError('Missing packed state field: '+name)
        if getattr(state,name).device!=device:raise ValueError('One device required')
    gain,scale,minimum=state.direction_gain,state.v_scale,state.v_min
    lookup=state.depth_residual_lookup
    if gain.dtype!=torch.float16 or gain.ndim!=4 or gain.shape[:2]!=(1,heads) or gain.shape[-1]!=1:
        raise ValueError('Actual FP16[1,H,N,1] K row gain required')
    nv=gain.shape[-2]
    alpha=state.depth_residual_alpha if residual_present else torch.empty((0,nv),device=device,dtype=torch.float16)
    residual_code=state.depth_residual_code if residual_present else torch.empty((0,nv,4),device=device,dtype=torch.int32)
    if tuple(scale.shape)!=(1,heads,128,1) or minimum.shape!=scale.shape or scale.dtype!=torch.float16 or minimum.dtype!=torch.float16:
        raise ValueError('Actual all-frame FP16[1,H,128,1] V scale/min required')
    if lookup.dtype!=torch.int32 or tuple(lookup.shape)!=(2,heads,2) or bool((lookup < -1).any()):
        raise ValueError('Actual int32[2,H,2] K/V compact stage lookup required')
    if alpha.dtype!=torch.float16 or alpha.ndim!=2 or alpha.shape[1]!=nv or tuple(residual_code.shape)!=(alpha.shape[0],nv,4):
        raise ValueError('Actual compact residual int32[M,N,4]/FP16[M,N] required')
    if residual_code.dtype!=torch.int32 or state.v_code.dtype!=torch.int32:
        raise ValueError('Unchanged integer binary codes required')
    if bool(((lookup[:,:,1]>=0)&(lookup[:,:,0]<0)).any()):raise ValueError('Stage1 requires same-axis/head stage0')
    for tensor in (gain,scale,minimum,alpha):
        if not bool(torch.isfinite(tensor).all()):raise ValueError('Finite physical fields required')
    fk,fv,offset=factors_3H.unbind(0)
    flat=lookup.reshape(-1);valid=flat>=0;slots=flat[valid].long();count=alpha.shape[0]
    if len(slots)!=count or not torch.equal(slots.sort().values,torch.arange(count,device=device)):
        raise ValueError('Every compact residual slot must map to exactly one axis/head/stage')
    axis_head=torch.stack((fk,fv))[:,:,None].expand(2,heads,2).reshape(-1)
    slot_factor=torch.zeros(count,device=device,dtype=torch.float32).scatter(0,slots,axis_head[valid])
    new_gain=(gain.float()*fk[None,:,None,None]).half().contiguous()
    new_scale=(scale.float()*fv[None,:,None,None]).half().contiguous()
    shift=(.5*(1-fv)+offset*fv)[None,:,None,None]
    proposal=(minimum.float()+shift*scale.float()).half()
    unchanged=((fv==1)&(offset==0))[None,:,None,None]
    new_minimum=_IdentitySignedZero.apply(minimum,proposal,unchanged).contiguous()
    new_alpha=(alpha.float()*slot_factor[:,None]).half().contiguous()
    for tensor in (new_gain,new_scale,new_minimum,new_alpha):
        if not bool(torch.isfinite(tensor).all()):raise ValueError('Postcal physical field overflow')
    result=SimpleNamespace(**vars(state))
    result.direction_gain,result.v_scale,result.v_min=new_gain,new_scale,new_minimum
    if residual_present:result.depth_residual_alpha=new_alpha
    return result

def decode_postcal_visual_fields(state, factors_3H, book, visual_tokens):
    """Return actual nested-half (Krotated,Vnative), preserving table gradients.

    The unchanged K mean is intentionally separate. Caller restores it in
    native coordinates or applies its exact mean bias in rotated attention.
    """
    from round8_codec import decode_key
    from round6_depth_codec import decode_sign_stage
    current=postcal_fields(state,factors_3H);nv=int(visual_tokens)
    if nv!=current.direction_gain.shape[-2] or nv%32:
        raise ValueError('Actual complete visual length divisible by32 required')
    if tuple(current.v_code.shape)!=(1,factors_3H.shape[1],128,nv//32):raise ValueError('Actual binary V words required')
    key=decode_key(current.direction_code,current.direction_gain,book)
    pos=torch.arange(nv,device=book.device,dtype=torch.int64)
    words=current.v_code.long().index_select(-1,pos//32)
    levels=((words>>(pos%32))&1).transpose(-1,-2).float()
    scale=current.v_scale[...,0][:,:,None,:].float();minimum=current.v_min[...,0][:,:,None,:].float()
    value=((levels*scale).half().float()+minimum).half()
    keys,values=[],[]
    lookup=current.depth_residual_lookup.detach().cpu().tolist()
    for head in range(factors_3H.shape[1]):
        k,v=key[0,head],value[0,head]
        for stage in range(2):
            slot=lookup[0][head][stage]
            if slot>=0:k=decode_sign_stage(k,current.depth_residual_code[slot],current.depth_residual_alpha[slot])
            slot=lookup[1][head][stage]
            if slot>=0:v=decode_sign_stage(v,current.depth_residual_code[slot],current.depth_residual_alpha[slot])
        keys.append(k);values.append(v)
    return torch.stack(keys)[None],torch.stack(values)[None]
