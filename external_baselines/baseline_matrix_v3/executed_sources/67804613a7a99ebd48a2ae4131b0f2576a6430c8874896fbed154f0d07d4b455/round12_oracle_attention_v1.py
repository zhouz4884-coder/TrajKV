"""Round12 oracle-only packed mass/routing interventions.

Native post-RoPE visual K is an explicitly charged above-budget reference.
The control path retains the original packed arithmetic exactly.
"""
import torch, triton
import triton.language as tl
from fast_wht import wht


@triton.jit
def _read(Q,QR,BIAS,KC,GAIN,BOOK,VC,VS,VM,VLEVELS,VMASK,NK,LSEC,LSEF,RC,RA,LOOKUP_K,LOOKUP_V,TK,TV,VP,TP,O,nq,nv,nt,qstart,
          qs1:tl.constexpr,qs2:tl.constexpr,ts1:tl.constexpr,D:tl.constexpr,G:tl.constexpr,KDIM:tl.constexpr,
          VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,HEADWISE:tl.constexpr,KPACK:tl.constexpr,KBITS:tl.constexpr,KBYTES:tl.constexpr,VMODE:tl.constexpr,OMODE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 head=tl.program_id(1);owner=head//G;index_k=tl.load(LOOKUP_K+owner*2);index_k2=tl.load(LOOKUP_K+owner*2+1);index_v=tl.load(LOOKUP_V+owner*2);index_v2=tl.load(LOOKUP_V+owner*2+1)
 rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
 q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
 qr=tl.load(QR+(head*nq+rows[:,None])*D+dims[None,:],rows[:,None]<nq,0)
 bias=tl.load(BIAS+head*nq+rows,rows<nq,0)
 if VMODE==2:
  mask_byte=tl.load(VMASK+owner*16+dims//8).to(tl.int32)
  keep=((mask_byte>>(dims%8))&1)!=0
  rank=tl.cumsum(keep.to(tl.int32),0)-1
 m=tl.full((BM,),-float('inf'),tl.float32);l=tl.full((BM,),0.,tl.float32);acc=tl.full((BM,D),0.,tl.float32);vb=tl.cdiv(nv,BN)
 for block in range(vb+tl.cdiv(nt,BN)):
  if block<vb:
   offset=block*BN+cols;valid=offset<nv;gain=tl.load(GAIN+owner*nv+offset,valid,0).to(tl.float32)
   if KDIM==0:
    word=tl.load(KC+(owner*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
    code=(word>>(dims[:,None]%32))&1;k=((2*code.to(tl.float32)-1)*gain[None,:]).to(tl.float16)
   else:
    if KPACK==2:
     packed=tl.load(KC+(owner*nv+offset[None,:])*(D//(KDIM*2))+dims[:,None]//(KDIM*2),valid[None,:],0).to(tl.int32)
     index=(packed>>(4*((dims[:,None]//KDIM)%2)))&15
    elif KBITS==8:
     index=tl.load(KC+(owner*nv+offset[None,:])*KBYTES+dims[:,None]//KDIM,valid[None,:],0).to(tl.int32)
    else:
     bit_offset=(dims[:,None]//KDIM)*KBITS
     byte_offset=bit_offset//8
     start=(owner*nv+offset[None,:])*KBYTES+byte_offset
     b0=tl.load(KC+start,valid[None,:],0).to(tl.int32)
     b1=tl.load(KC+start+1,valid[None,:]&(byte_offset+1<KBYTES),0).to(tl.int32)
     b2=tl.load(KC+start+2,valid[None,:]&(byte_offset+2<KBYTES),0).to(tl.int32)
     index=((b0|(b1<<8)|(b2<<16))>>(bit_offset%8))&((1<<KBITS)-1)
    book_offset=owner*256*KDIM if HEADWISE else 0
    value=tl.load(BOOK+book_offset+index*KDIM+dims[:,None]%KDIM,valid[None,:],0).to(tl.float32);k=(value*gain[None,:]).to(tl.float16)
   if index_k>=0:
    word_k=tl.load(RC+(index_k*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
    sign_k=(word_k>>(dims[:,None]%32))&1
    alpha_k=tl.load(RA+index_k*nv+offset,valid,0).to(tl.float32)
    k=(k.to(tl.float32)+(2*sign_k.to(tl.float32)-1)*alpha_k[None,:]).to(tl.float16)
   if index_k2>=0:
    word_k=tl.load(RC+(index_k2*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
    sign_k=(word_k>>(dims[:,None]%32))&1
    alpha_k=tl.load(RA+index_k2*nv+offset,valid,0).to(tl.float32)
    k=(k.to(tl.float32)+(2*sign_k.to(tl.float32)-1)*alpha_k[None,:]).to(tl.float16)
   if VB==16:v=tl.load(VC+(owner*nv+offset[:,None])*D+dims[None,:],valid[:,None],0)
   else:
    if VCHANNEL:
     if VMODE==2:
      word_v=tl.load(VC+(owner*64+rank[None,:])*(nv//32)+offset[:,None]//32,valid[:,None]&keep[None,:],0)
     else:
      word_v=tl.load(VC+(owner*D+dims[None,:])*(nv//(32//VB))+offset[:,None]//(32//VB),valid[:,None],0)
     code_v=(word_v>>(VB*(offset[:,None]%(32//VB))))&((1<<VB)-1)
     scale=tl.load(VS+(owner*D+dims[None,:])*(nv//VG)+offset[:,None]//VG,valid[:,None],0)
     minimum=tl.load(VM+(owner*D+dims[None,:])*(nv//VG)+offset[:,None]//VG,valid[:,None],0)
    else:
     word_v=tl.load(VC+(owner*nv+offset[:,None])*(D//(32//VB))+dims[None,:]//(32//VB),valid[:,None],0)
     code_v=(word_v>>(VB*(dims[None,:]%(32//VB))))&((1<<VB)-1)
     scale=tl.load(VS+owner*nv+offset,valid,0)[:,None];minimum=tl.load(VM+owner*nv+offset,valid,0)[:,None]
    if VMODE==1:
     level=tl.load(VLEVELS+code_v).to(tl.float16)
     v=((level*scale).to(tl.float16)+minimum).to(tl.float16)
    else:
     v=((code_v.to(tl.float16)*scale).to(tl.float16)+minimum).to(tl.float16)
   if index_v>=0:
    word_r=tl.load(RC+(index_v*nv+offset[:,None])*(D//32)+dims[None,:]//32,valid[:,None],0)
    sign_v=(word_r>>(dims[None,:]%32))&1
    alpha_v=tl.load(RA+index_v*nv+offset,valid,0).to(tl.float32)
    v=(v.to(tl.float32)+(2*sign_v.to(tl.float32)-1)*alpha_v[:,None]).to(tl.float16)
   if index_v2>=0:
    word_r=tl.load(RC+(index_v2*nv+offset[:,None])*(D//32)+dims[None,:]//32,valid[:,None],0)
    sign_v=(word_r>>(dims[None,:]%32))&1
    alpha_v=tl.load(RA+index_v2*nv+offset,valid,0).to(tl.float32)
    v=(v.to(tl.float32)+(2*sign_v.to(tl.float32)-1)*alpha_v[:,None]).to(tl.float16)
   if OMODE==2:
    native_k=tl.load(NK+(owner*nv+offset[None,:])*D+dims[:,None],valid[None,:],0)
    fix=tl.load(LSEC+head*nq+rows,rows<nq,0)-tl.load(LSEF+head*nq+rows,rows<nq,0)
    score=tl.dot(q,native_k)*(D**-.5)+fix[:,None]
   else:
    score=tl.dot(qr,k)*(D**-.5)+bias[:,None]
    if OMODE==1:
     fix=tl.load(LSEF+head*nq+rows,rows<nq,0)-tl.load(LSEC+head*nq+rows,rows<nq,0)
     score=score+fix[:,None]
   pos=tl.load(VP+offset,valid,0)
  else:
   offset=(block-vb)*BN+cols;valid=offset<nt
   k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
   v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
   score=tl.dot(q,k)*(D**-.5);pos=tl.load(TP+offset,valid,0)
  score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
  nm=tl.maximum(m,tl.max(score,1));safe=tl.where(nm==-float('inf'),0.,nm);alpha=tl.exp(m-safe);weight=tl.exp(score-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(weight.to(tl.float16),v);l=l*alpha+tl.sum(weight,1);m=nm
 tl.store(O+((head*nq+rows[:,None])*D+dims[None,:]),acc/tl.maximum(l[:,None],1e-30),rows[:,None]<nq)



@triton.jit
def _visual_lse(Q,QR,BIAS,KC,GAIN,BOOK,RC,RA,LOOKUP_K,NK,VP,LC,LF,nq,nv,qstart,
                qs1:tl.constexpr,qs2:tl.constexpr,D:tl.constexpr,G:tl.constexpr,
                KPACK:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 head=tl.program_id(1);owner=head//G
 rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
 q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
 qr=tl.load(QR+(head*nq+rows[:,None])*D+dims[None,:],rows[:,None]<nq,0)
 bias=tl.load(BIAS+head*nq+rows,rows<nq,0)
 index_k=tl.load(LOOKUP_K+owner*2);index_k2=tl.load(LOOKUP_K+owner*2+1)
 mc=tl.full((BM,),-float('inf'),tl.float32);mf=tl.full((BM,),-float('inf'),tl.float32)
 lc=tl.full((BM,),0.,tl.float32);lf=tl.full((BM,),0.,tl.float32)
 for block in range(tl.cdiv(nv,BN)):
  offset=block*BN+cols;valid=offset<nv
  gain=tl.load(GAIN+owner*nv+offset,valid,0).to(tl.float32)
  if KPACK==2:
   packed=tl.load(KC+(owner*nv+offset[None,:])*16+dims[:,None]//8,valid[None,:],0).to(tl.int32)
   index=(packed>>(4*((dims[:,None]//4)%2)))&15
  else:
   index=tl.load(KC+(owner*nv+offset[None,:])*32+dims[:,None]//4,valid[None,:],0).to(tl.int32)
  value=tl.load(BOOK+index*4+dims[:,None]%4,valid[None,:],0).to(tl.float32)
  kc=(value*gain[None,:]).to(tl.float16)
  if index_k>=0:
   word=tl.load(RC+(index_k*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
   sign=(word>>(dims[:,None]%32))&1
   alpha=tl.load(RA+index_k*nv+offset,valid,0).to(tl.float32)
   kc=(kc.to(tl.float32)+(2*sign.to(tl.float32)-1)*alpha[None,:]).to(tl.float16)
  if index_k2>=0:
   word=tl.load(RC+(index_k2*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
   sign=(word>>(dims[:,None]%32))&1
   alpha=tl.load(RA+index_k2*nv+offset,valid,0).to(tl.float32)
   kc=(kc.to(tl.float32)+(2*sign.to(tl.float32)-1)*alpha[None,:]).to(tl.float16)
  kf=tl.load(NK+(owner*nv+offset[None,:])*D+dims[:,None],valid[None,:],0)
  zc=tl.dot(qr,kc)*(D**-.5)+bias[:,None]
  zf=tl.dot(q,kf)*(D**-.5)
  pos=tl.load(VP+offset,valid,0)
  seen=valid[None,:]&(pos[None,:]<=qstart+rows[:,None])
  zc=tl.where(seen,zc,-float('inf'));zf=tl.where(seen,zf,-float('inf'))
  nc=tl.maximum(mc,tl.max(zc,1));nf=tl.maximum(mf,tl.max(zf,1))
  sc=tl.where(nc==-float('inf'),0.,nc);sf=tl.where(nf==-float('inf'),0.,nf)
  lc=lc*tl.exp(mc-sc)+tl.sum(tl.exp(zc-sc[:,None]),1)
  lf=lf*tl.exp(mf-sf)+tl.sum(tl.exp(zf-sf[:,None]),1)
  mc=nc;mf=nf
 tl.store(LC+head*nq+rows,mc+tl.log(tl.maximum(lc,1e-30)),rows<nq)
 tl.store(LF+head*nq+rows,mf+tl.log(tl.maximum(lf,1e-30)),rows<nq)


def visual_lse(q,state,book):
 b,h,n,d=q.shape;nv=state.visual_pos.numel();g=h//state.text_k.shape[1]
 if tuple(book.shape) not in ((16,4),(256,4)):
  raise ValueError('Oracle LSE supports frozen G4_16/G4_256 only')
 qr=wht(q,half=True)
 bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5).contiguous()
 lc=torch.empty((h,n),device=q.device,dtype=torch.float32);lf=torch.empty_like(lc)
 rc=getattr(state,'depth_residual_code',state.v_code);ra=getattr(state,'depth_residual_alpha',state.v_scale)
 _visual_lse[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_code,state.direction_gain,book,rc,ra,
  state.depth_residual_lookup[0],state.oracle_native_k,state.visual_pos,lc,lf,n,nv,state.length-n,
  q.stride(1),q.stride(2),d,g,2 if book.shape[0]==16 else 1,16,32,
  num_warps=4,num_stages=2,enable_fp_fusion=False)
 return qr,bias,lc,lf


def attention(q,state,book,v_group_size=128,visual_bias=None):
 b,h,n,d=q.shape;nv=state.visual_pos.numel();g=h//state.text_k.shape[1]
 if b!=1 or d!=128 or q.dtype!=torch.float16 or q.stride(-1)!=1:
  raise ValueError('Batch1 FP16 D128 required')
 if tuple(book.shape) not in ((16,4),(256,4)) or state.direction_code.dtype!=torch.uint8:
  raise ValueError('Frozen compact G4 payload required')
 if visual_bias is not None:
  raise ValueError('Oracle has no unrelated external visual bias')
 mode=int(state.oracle_mode)
 if mode not in (0,1,2) or state.oracle_native_k.shape!=(1,4,nv,128):
  raise ValueError('Oracle native K/mode mismatch')
 if any(hasattr(state,key) for key in ('visual_k','visual_v')):
  raise ValueError('Original full native source must be released')
 if mode:
  qr,bias,lc,lf=visual_lse(q,state,book)
 else:
  qr=wht(q,half=True)
  bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5).contiguous()
  # Pointer placeholders are never read by the constexpr control path.
  lc=lf=bias
 out=torch.empty_like(q,memory_format=torch.contiguous_format)
 rc=getattr(state,'depth_residual_code',state.v_code);ra=getattr(state,'depth_residual_alpha',state.v_scale)
 _read[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_code,state.direction_gain,book,state.v_code,state.v_scale,state.v_min,
  state.v_min,state.v_code,state.oracle_native_k,lc,lf,rc,ra,state.depth_residual_lookup[0],state.depth_residual_lookup[1],
  state.text_k,state.text_v,state.visual_pos,state.text_pos,out,n,nv,state.text_length,state.length-n,
  q.stride(1),q.stride(2),state.text_k.stride(1),d,g,4,1,True,v_group_size,False,
  2 if book.shape[0]==16 else 1,4 if book.shape[0]==16 else 8,16 if book.shape[0]==16 else 32,
  0,mode,16,32,num_warps=4,num_stages=2,enable_fp_fusion=False)
 return out
