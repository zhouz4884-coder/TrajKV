"""Independent packed reader for source-example KIVI g32/R32 visual port.

K metadata groups the token axis; V metadata groups each token's channels.
This new kernel does not modify the frozen g128 technical-anchor reader.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _read_example(Q,KC,KS,KM,VC,VS,VM,KT,VT,TK,TV,VP,TP,O,
          nq,nv,nkq,nvq,nt,qstart,
          qs0:tl.constexpr,qs1:tl.constexpr,qs2:tl.constexpr,
          ts0:tl.constexpr,ts1:tl.constexpr,
          HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BITS:tl.constexpr,GROUP:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr):
    bh=tl.program_id(1);batch=bh//HQ;h=bh%HQ;kh=h//(HQ//HK);owner=batch*HK+kh
    rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
    q=tl.load(Q+batch*qs0+h*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
    running_m=tl.full((BM,),-float('inf'),tl.float32);running_l=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,D),tl.float32);vb=tl.cdiv(nv,BN)
    for block in range(vb+tl.cdiv(nt,BN)):
        slot=(block%vb)*BN+cols
        if block<vb:
            valid=slot<nv;pos=tl.load(VP+slot,valid,0)
            kvalid=valid & (slot<nkq)
            kc=tl.load(KC+owner*D*(nkq//(32//BITS))+dims[:,None]*(nkq//(32//BITS))+slot[None,:]//(32//BITS),kvalid[None,:],0)
            ki=(kc>>(BITS*(slot[None,:]%(32//BITS))))&((1<<BITS)-1)
            ks=tl.load(KS+owner*D*(nkq//GROUP)+dims[:,None]*(nkq//GROUP)+slot[None,:]//GROUP,kvalid[None,:],0)
            km=tl.load(KM+owner*D*(nkq//GROUP)+dims[:,None]*(nkq//GROUP)+slot[None,:]//GROUP,kvalid[None,:],0)
            quant_k=((ki.to(tl.float16)*ks).to(tl.float16)+km).to(tl.float16)
            tail_k=tl.load(KT+owner*(nv-nkq)*D+(slot[None,:]-nkq)*D+dims[:,None],(valid & (slot>=nkq))[None,:],0)
            k=tl.where(kvalid[None,:],quant_k,tail_k)
            vvalid=valid & (slot<nvq)
            vc=tl.load(VC+owner*nvq*(D//(32//BITS))+slot[:,None]*(D//(32//BITS))+dims[None,:]//(32//BITS),vvalid[:,None],0)
            vi=(vc>>(BITS*(dims[None,:]%(32//BITS))))&((1<<BITS)-1)
            # g32 means FOUR independent V scales/minima per D128 token.
            vs=tl.load(VS+owner*nvq*(D//GROUP)+slot[:,None]*(D//GROUP)+dims[None,:]//GROUP,vvalid[:,None],0)
            vm=tl.load(VM+owner*nvq*(D//GROUP)+slot[:,None]*(D//GROUP)+dims[None,:]//GROUP,vvalid[:,None],0)
            quant_v=((vi.to(tl.float16)*vs).to(tl.float16)+vm).to(tl.float16)
            tail_v=tl.load(VT+owner*(nv-nvq)*D+(slot[:,None]-nvq)*D+dims[None,:],(valid & (slot>=nvq))[:,None],0)
            v=tl.where(vvalid[:,None],quant_v,tail_v)
        else:
            slot=(block-vb)*BN+cols;valid=slot<nt;pos=tl.load(TP+slot,valid,0)
            k=tl.load(TK+batch*ts0+kh*ts1+slot[None,:]*D+dims[:,None],valid[None,:],0)
            v=tl.load(TV+batch*ts0+kh*ts1+slot[:,None]*D+dims[None,:],valid[:,None],0)
        score=tl.dot(q,k)*(D**-0.5)
        score=tl.where(valid[None,:] & (pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
        new_m=tl.maximum(running_m,tl.max(score,1));safe_m=tl.where(new_m==-float('inf'),0.,new_m)
        alpha=tl.exp(running_m-safe_m);weight=tl.exp(score-safe_m[:,None])
        acc=acc*alpha[:,None]+tl.dot(weight.to(tl.float16),v)
        running_l=running_l*alpha+tl.sum(weight,1);running_m=new_m
    out=acc/tl.maximum(running_l[:,None],1.e-30)
    tl.store(O+((batch*HQ+h)*nq+rows[:,None])*D+dims[None,:],out,rows[:,None]<nq)


def attention(q,state,text_k,text_v,text_positions,query_positions,scale=None):
    if not q.is_cuda or q.dtype!=torch.float16 or q.shape[0]!=1 or q.shape[-1]!=128 or q.stride(-1)!=1:
        raise ValueError('Source-example packed reader requires CUDA batch1 FP16 D128')
    if state.group_size!=32 or state.residual_length!=32 or state.bits!=2:
        raise ValueError('This source-example port is restricted to b2/g32/R32')
    if state.nv<=0 or state.positions.ndim!=1 or state.positions.numel()!=state.nv:
        raise ValueError('Nonempty visual owner and original positions are required')
    if text_k.ndim!=4 or text_k.shape[0]!=1 or text_k.shape[1]!=state.k_tail.shape[1] or text_k.shape[-1]!=128 or text_k.shape!=text_v.shape:
        raise ValueError('Exact text must match visual-owner KV geometry')
    if text_positions.ndim!=1 or text_positions.numel()!=text_k.shape[-2] or text_positions.dtype not in (torch.int32,torch.int64):
        raise ValueError('One integer original position per exact text token is required')
    factor=128**-.5 if scale is None else float(scale)
    if not math.isclose(factor,128**-.5,rel_tol=1e-12):raise ValueError('Unexpected native attention scale')
    query_positions=query_positions.flatten()
    if len(query_positions)!=q.shape[-2] or not len(query_positions):raise ValueError('One original position per query is required')
    start=int(query_positions[0])
    if not torch.equal(query_positions,torch.arange(start,start+q.shape[-2],device=q.device)):raise ValueError('Queries must append contiguous logical positions')
    if q.shape[1]%text_k.shape[1]:raise ValueError('Nonintegral GQA ratio')
    tk,tv=text_k.contiguous(),text_v.contiguous();b,h,n,d=q.shape
    if tk.dtype!=q.dtype or tv.dtype!=q.dtype or tk.device!=q.device or tv.device!=q.device:raise ValueError('Exact text dtype/device differs')
    vp=state.positions.contiguous();tp=text_positions.to(device=q.device,dtype=torch.int64).contiguous()
    out=torch.empty((b,h,n,d),device=q.device,dtype=q.dtype)
    _read_example[(triton.cdiv(n,16),b*h)](q,state.k_code,state.k_scale,state.k_min,
        state.v_code,state.v_scale,state.v_min,state.k_tail,state.v_tail,tk,tv,vp,tp,out,
        n,state.nv,state.nkq,state.nvq,tk.shape[-2],start,*q.stride()[:3],*tk.stride()[:2],
        h,tk.shape[1],d,state.bits,state.group_size,16,32,num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out
