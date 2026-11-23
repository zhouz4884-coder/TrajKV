"""Packed sign/VQ residual keys in rotated query space, FP32 mean restoration."""
import torch,triton
import triton.language as tl
from fast_wht import wht
@triton.jit
def _read(Q,QR,BIAS,KC,GAIN,BOOK,VC,VS,VM,TK,TV,VP,TP,O,nq,nv,nt,qstart,
          qs1:tl.constexpr,qs2:tl.constexpr,ts1:tl.constexpr,D:tl.constexpr,G:tl.constexpr,KDIM:tl.constexpr,
          VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,HEADWISE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 head=tl.program_id(1);owner=head//G;rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
 q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
 qr=tl.load(QR+(head*nq+rows[:,None])*D+dims[None,:],rows[:,None]<nq,0)
 bias=tl.load(BIAS+head*nq+rows,rows<nq,0)
 m=tl.full((BM,),-float('inf'),tl.float32);l=tl.full((BM,),0.,tl.float32);acc=tl.full((BM,D),0.,tl.float32);vb=tl.cdiv(nv,BN)
 for block in range(vb+tl.cdiv(nt,BN)):
  if block<vb:
   offset=block*BN+cols;valid=offset<nv;gain=tl.load(GAIN+owner*nv+offset,valid,0).to(tl.float32)
   if KDIM==0:
    word=tl.load(KC+(owner*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
    code=(word>>(dims[:,None]%32))&1;k=((2*code.to(tl.float32)-1)*gain[None,:]).to(tl.float16)
   else:
    index=tl.load(KC+(owner*nv+offset[None,:])*(D//KDIM)+dims[:,None]//KDIM,valid[None,:],0).to(tl.int32)
    book_offset=owner*256*KDIM if HEADWISE else 0
    value=tl.load(BOOK+book_offset+index*KDIM+dims[:,None]%KDIM,valid[None,:],0).to(tl.float32);k=(value*gain[None,:]).to(tl.float16)
   if VB==16:v=tl.load(VC+(owner*nv+offset[:,None])*D+dims[None,:],valid[:,None],0)
   else:
    if VCHANNEL:
     wordv=tl.load(VC+(owner*D+dims[None,:])*(nv//(32//VB))+offset[:,None]//(32//VB),valid[:,None],0)
     codev=(wordv>>(VB*(offset[:,None]%(32//VB))))&((1<<VB)-1)
     scale=tl.load(VS+(owner*D+dims[None,:])*(nv//VG)+offset[:,None]//VG,valid[:,None],0)
     minimum=tl.load(VM+(owner*D+dims[None,:])*(nv//VG)+offset[:,None]//VG,valid[:,None],0)
    else:
     wordv=tl.load(VC+(owner*nv+offset[:,None])*(D//(32//VB))+dims[None,:]//(32//VB),valid[:,None],0)
     codev=(wordv>>(VB*(dims[None,:]%(32//VB))))&((1<<VB)-1)
     scale=tl.load(VS+owner*nv+offset,valid,0)[:,None];minimum=tl.load(VM+owner*nv+offset,valid,0)[:,None]
    v=((codev.to(tl.float16)*scale).to(tl.float16)+minimum).to(tl.float16)
   score=tl.dot(qr,k)*(D**-.5)+bias[:,None];pos=tl.load(VP+offset,valid,0)
  else:
   offset=(block-vb)*BN+cols;valid=offset<nt
   k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
   v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
   score=tl.dot(q,k)*(D**-.5);pos=tl.load(TP+offset,valid,0)
  score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
  nm=tl.maximum(m,tl.max(score,1));safe=tl.where(nm==-float('inf'),0.,nm);alpha=tl.exp(m-safe);w=tl.exp(score-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(w.to(tl.float16),v);l=l*alpha+tl.sum(w,1);m=nm
 tl.store(O+((head*nq+rows[:,None])*D+dims[None,:]),acc/tl.maximum(l[:,None],1e-30),rows[:,None]<nq)
def attention(q,state,book,vbits=1,v_channel=True,v_group_size=128):
 b,h,n,d=q.shape;nv=state.visual_pos.numel();g=h//state.text_k.shape[1]
 if b!=1 or d!=128 or q.dtype!=torch.float16 or q.stride(-1)!=1:raise ValueError('Batch1 FP16 D128 required')
 if vbits!=16 and ((nv if v_channel else d)%v_group_size or (not v_channel and v_group_size!=d)):raise ValueError('Complete V groups required')
 qr=wht(q,half=True);bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5).contiguous()
 v=(state.visual_v,q,q) if vbits==16 else (state.v_code,state.v_scale,state.v_min);out=torch.empty_like(q,memory_format=torch.contiguous_format)
 _read[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_code,state.direction_gain,book if book is not None else q,*v,
  state.text_k,state.text_v,state.visual_pos,state.text_pos,out,n,nv,state.text_length,state.length-n,
  q.stride(1),q.stride(2),state.text_k.stride(1),d,g,0 if book is None else book.shape[-1],vbits,v_channel,v_group_size,book.ndim==3,16,32,
  num_warps=4,num_stages=2,enable_fp_fusion=False)
 return out
