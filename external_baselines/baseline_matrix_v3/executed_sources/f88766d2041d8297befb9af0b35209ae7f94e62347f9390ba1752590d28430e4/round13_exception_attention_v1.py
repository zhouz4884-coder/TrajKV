"""One online softmax over compact low rows, native visual exceptions, text.

Only the low segment sees centered/WHT query and f0-folded statistics. Native
exception keys already include original RoPE; native values remain unchanged.
The low segment iterates original row blocks and maps them to physically
compact payloads by a charged exception bitmap and block-prefix ranks.
"""
import torch
import triton
import triton.language as tl
from fast_wht import wht


@triton.jit
def _popc(x):
    return tl.inline_asm_elementwise('popc.b32 $0, $1;', constraints='=r,r',
                                   args=[x], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _read(Q, QR, BIAS, KC, GAIN, BOOK, VC, VS, VM, RC, RA, LK, LV,
          NK, NV, MASK, PREFIX, HIGH, TK, TV, VP, TP, O,
          nq, no, nm, nh, nt, qstart,
          qs1:tl.constexpr, qs2:tl.constexpr, ts1:tl.constexpr,
          D:tl.constexpr, G:tl.constexpr, VWORDS:tl.constexpr,
          BM:tl.constexpr, BN:tl.constexpr):
    head=tl.program_id(1); owner=head//G
    ik=tl.load(LK+owner*2); ik2=tl.load(LK+owner*2+1)
    iv=tl.load(LV+owner*2); iv2=tl.load(LV+owner*2+1)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    dims=tl.arange(0,D); cols=tl.arange(0,BN)
    q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
    qr=tl.load(QR+(head*nq+rows[:,None])*D+dims[None,:],rows[:,None]<nq,0)
    bias=tl.load(BIAS+head*nq+rows,rows<nq,0)
    maxv=tl.full((BM,),-float('inf'),tl.float32)
    den=tl.full((BM,),0.,tl.float32); acc=tl.full((BM,D),0.,tl.float32)
    low_blocks=tl.cdiv(no,BN); high_blocks=tl.cdiv(nh,BN)
    for block in range(low_blocks+high_blocks+tl.cdiv(nt,BN)):
        if block<low_blocks:
            original=block*BN+cols
            word=tl.load(MASK+block)
            before=tl.load(PREFIX+block)
            bit=cols.to(tl.uint32)
            selected=((word.to(tl.uint32)>>bit)&1)!=0
            lower=(tl.full((BN,),1,tl.uint32)<<bit)-1
            rank=before+_popc(word.to(tl.uint32)&lower)
            offset=original-rank
            valid=(original<no)&(~selected)
            gain=tl.load(GAIN+owner*nm+offset,valid,0).to(tl.float32)
            packed=tl.load(KC+(owner*nm+offset[None,:])*16+dims[:,None]//8,valid[None,:],0).to(tl.int32)
            index=(packed>>(4*((dims[:,None]//4)%2)))&15
            template=tl.load(BOOK+index*4+dims[:,None]%4,valid[None,:],0).to(tl.float32)
            k=(template*gain[None,:]).to(tl.float16)
            if ik>=0:
                wr=tl.load(RC+(ik*nm+offset[None,:])*4+dims[:,None]//32,valid[None,:],0)
                sign=(wr>>(dims[:,None]%32))&1
                a=tl.load(RA+ik*nm+offset,valid,0).to(tl.float32)
                k=(k.to(tl.float32)+(2*sign.to(tl.float32)-1)*a[None,:]).to(tl.float16)
            if ik2>=0:
                wr=tl.load(RC+(ik2*nm+offset[None,:])*4+dims[:,None]//32,valid[None,:],0)
                sign=(wr>>(dims[:,None]%32))&1
                a=tl.load(RA+ik2*nm+offset,valid,0).to(tl.float32)
                k=(k.to(tl.float32)+(2*sign.to(tl.float32)-1)*a[None,:]).to(tl.float16)
            vw=tl.load(VC+(owner*D+dims[None,:])*VWORDS+offset[:,None]//32,valid[:,None],0)
            code=(vw>>(offset[:,None]%32))&1
            scale=tl.load(VS+owner*D+dims[None,:])
            minimum=tl.load(VM+owner*D+dims[None,:])
            v=((code.to(tl.float16)*scale).to(tl.float16)+minimum).to(tl.float16)
            if iv>=0:
                wr=tl.load(RC+(iv*nm+offset[:,None])*4+dims[None,:]//32,valid[:,None],0)
                sign=(wr>>(dims[None,:]%32))&1
                a=tl.load(RA+iv*nm+offset,valid,0).to(tl.float32)
                v=(v.to(tl.float32)+(2*sign.to(tl.float32)-1)*a[:,None]).to(tl.float16)
            if iv2>=0:
                wr=tl.load(RC+(iv2*nm+offset[:,None])*4+dims[None,:]//32,valid[:,None],0)
                sign=(wr>>(dims[None,:]%32))&1
                a=tl.load(RA+iv2*nm+offset,valid,0).to(tl.float32)
                v=(v.to(tl.float32)+(2*sign.to(tl.float32)-1)*a[:,None]).to(tl.float16)
            score=tl.dot(qr,k)*(D**-.5)+bias[:,None]
            pos=tl.load(VP+original,valid,0)
        elif block<low_blocks+high_blocks:
            offset=(block-low_blocks)*BN+cols; valid=offset<nh
            k=tl.load(NK+(owner*nh+offset[None,:])*D+dims[:,None],valid[None,:],0)
            v=tl.load(NV+(owner*nh+offset[:,None])*D+dims[None,:],valid[:,None],0)
            original=tl.load(HIGH+offset,valid,0)
            pos=tl.load(VP+original,valid,0)
            score=tl.dot(q,k)*(D**-.5)
        else:
            offset=(block-low_blocks-high_blocks)*BN+cols; valid=offset<nt
            k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
            v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
            pos=tl.load(TP+offset,valid,0)
            score=tl.dot(q,k)*(D**-.5)
        score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
        nmv=tl.maximum(maxv,tl.max(score,1)); safe=tl.where(nmv==-float('inf'),0.,nmv)
        alpha=tl.exp(maxv-safe); weight=tl.exp(score-safe[:,None])
        acc=acc*alpha[:,None]+tl.dot(weight.to(tl.float16),v)
        den=den*alpha+tl.sum(weight,1); maxv=nmv
    tl.store(O+(head*nq+rows[:,None])*D+dims[None,:],acc/tl.maximum(den[:,None],1e-30),rows[:,None]<nq)


def attention(q,state,book,v_group_size=None):
    b,h,n,d=q.shape
    no=state.visual_pos.numel(); nm=int(state.round13_ordinary_rows)
    nh=state.round13_native_k.shape[-2]; g=h//state.text_k.shape[1]
    if b!=1 or d!=128 or q.dtype!=torch.float16 or q.stride(-1)!=1:
        raise ValueError('Round13 exception requires batch1 FP16 D128')
    if tuple(book.shape)!=(16,4) or state.direction_code.shape[-1]!=16 or no%32:
        raise ValueError('Unchanged G4_16 complete original image required')
    if tuple(state.depth_residual_lookup.shape)!=(2,4,2) or nh+nm!=no:
        raise ValueError('Complete exception partition and stage lookup required')
    qr=wht(q,half=True)
    bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5).contiguous()
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    rc=getattr(state,'depth_residual_code',state.v_code)
    ra=getattr(state,'depth_residual_alpha',state.v_scale)
    _read[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_code,state.direction_gain,book,
        state.v_code,state.v_scale,state.v_min,rc,ra,state.depth_residual_lookup[0],state.depth_residual_lookup[1],
        state.round13_native_k,state.round13_native_v,state.round13_exception_mask,
        state.round13_exception_prefix,state.round13_exception_rows,state.text_k,state.text_v,
        state.visual_pos,state.text_pos,out,n,no,nm,nh,state.text_length,state.length-n,
        q.stride(1),q.stride(2),state.text_k.stride(1),d,g,(nm+31)//32,16,32,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out


def dense_reference(q,state,book):
    """Transient Torch diagnostic; never used as the experiment reader."""
    no=int(state.visual_pos.numel()); nm=int(state.round13_ordinary_rows)
    high=state.round13_exception_rows.long()
    keep=torch.ones(no,device=q.device,dtype=torch.bool); keep[high]=False
    normal=keep.nonzero(as_tuple=False).flatten()
    code=state.direction_code
    indices=torch.stack((code&15,code>>4),-1).flatten(-2)
    kr=(book.float()[indices.long()].flatten(-2)*state.direction_gain.float()).half()
    shift=torch.arange(32,device=q.device,dtype=torch.int64)
    vb=((state.v_code.long()[...,None]>>shift)&1).flatten(-2)[...,:nm]
    vv=((vb.half()*state.v_scale).half()+state.v_min).half().transpose(2,3).contiguous()
    for axis,values in ((0,kr),(1,vv)):
        for head in range(4):
            for depth in range(2):
                slot=int(state.depth_residual_lookup[axis,head,depth])
                if slot>=0:
                    words=state.depth_residual_code[slot]
                    bits=((words.long()[...,None]>>shift)&1).flatten(-2)
                    delta=(2*bits.float()-1)*state.depth_residual_alpha[slot,:,None].float()
                    values[0,head]=(values[0,head].float()+delta).half()
    g=q.shape[1]//4
    qr=wht(q,half=True)
    logits_low=torch.matmul(qr.float(),kr.repeat_interleave(g,1).float().transpose(-2,-1))/(128**.5)
    bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1)/(128**.5)
    logits_low=logits_low+bias[...,None]
    logits_high=torch.matmul(q.float(),state.round13_native_k.repeat_interleave(g,1).float().transpose(-2,-1))/(128**.5)
    logits_text=torch.matmul(q.float(),state.text_k.repeat_interleave(g,1).float().transpose(-2,-1))/(128**.5)
    logits=torch.cat((logits_low,logits_high,logits_text),-1)
    positions=torch.cat((state.visual_pos[normal],state.visual_pos[high],state.text_pos))
    qpos=torch.arange(state.length-q.shape[-2],state.length,device=q.device)
    logits=logits.masked_fill(positions[None,:]>qpos[:,None],-float('inf'))
    values=torch.cat((vv,state.round13_native_v,state.text_v),2).repeat_interleave(g,1).float()
    return torch.matmul(logits.softmax(-1),values).half()
