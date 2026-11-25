"""Round13 correctness-first packed precenter reader, one joint softmax.

Phase tables are transient Torch-native FP32 trig->Half scratch on every call.
This reader deliberately trades speed for a single encode/decode rounding
contract. It does not claim zero-cost RoPE reconstruction.
"""
import torch, triton
import triton.language as tl
from fast_wht import wht
from round13_precenter_bank_v1 import phase_tables


@triton.jit
def _position_fields(MU,MEAN,KF,COS,SIN,owner,offset,dims,valid,
                     D:tl.constexpr,BN:tl.constexpr,ROTATION_CONSTANT:tl.constexpr):
 mu=tl.load(MU+owner*D+dims).to(tl.float32)
 aa=tl.load(MEAN+owner*D+dims).to(tl.float32)
 gg=tl.load(KF+owner).to(tl.float32)
 if ROTATION_CONSTANT:
  ss=tl.broadcast_to(mu[:,None],(D,BN)).to(tl.float16)
 else:
  mup=tl.load(MU+owner*D+(dims+64)%D).to(tl.float32)
  mup=tl.where(dims<64,-mup,mup)
  co=tl.load(COS+offset[None,:]*D+dims[:,None],valid[None,:],0).to(tl.float32)
  si=tl.load(SIN+offset[None,:]*D+dims[:,None],valid[None,:],0).to(tl.float32)
  p0=(mu[:,None]*co).to(tl.float16)
  p1=(mup[:,None]*si).to(tl.float16)
  ss=(p0.to(tl.float32)+p1.to(tl.float32)).to(tl.float16)
 bb=(aa[:,None]+gg*(ss.to(tl.float32)-aa[:,None])).to(tl.float16)
 return ss,bb


@triton.jit
def _materialize_fields(MU,MEAN,KF,COS,SIN,S,B,N,D:tl.constexpr,
                        BN:tl.constexpr,ROTATION_CONSTANT:tl.constexpr):
 owner=tl.program_id(1);offset=tl.program_id(0)*BN+tl.arange(0,BN)
 dims=tl.arange(0,D);valid=offset<N
 ss,bb=_position_fields(MU,MEAN,KF,COS,SIN,owner,offset,dims,valid,D,BN,ROTATION_CONSTANT)
 tl.store(S+(owner*N+offset[None,:])*D+dims[:,None],ss,valid[None,:])
 tl.store(B+(owner*N+offset[None,:])*D+dims[:,None],bb,valid[None,:])


@triton.jit
def _read(Q,QR,BIAS,MEAN,MU,KF,COS,SIN,KC,GAIN,BOOK,VC,VS,VM,VLEVELS,VMASK,RC,RA,LOOKUP_K,LOOKUP_V,TK,TV,VP,TP,O,nq,nv,nt,qstart,
          qs1:tl.constexpr,qs2:tl.constexpr,ts1:tl.constexpr,D:tl.constexpr,G:tl.constexpr,KDIM:tl.constexpr,
          VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,HEADWISE:tl.constexpr,KPACK:tl.constexpr,KBITS:tl.constexpr,KBYTES:tl.constexpr,VMODE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,ROTATION_CONSTANT:tl.constexpr):
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
   pos=tl.load(VP+offset,valid,0)
   if ROTATION_CONSTANT:
    score=tl.dot(qr,k)*(D**-.5)+bias[:,None]
   else:
    _,bb=_position_fields(MU,MEAN,KF,COS,SIN,owner,offset,dims,valid,D,BN,False)
    score=tl.dot(qr,k)*(D**-.5)+tl.dot(q,bb)*(D**-.5)
  else:
   offset=(block-vb)*BN+cols;valid=offset<nt
   k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
   v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
   score=tl.dot(q,k)*(D**-.5);pos=tl.load(TP+offset,valid,0)
  score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
  nm=tl.maximum(m,tl.max(score,1));safe=tl.where(nm==-float('inf'),0.,nm);alpha=tl.exp(m-safe);weight=tl.exp(score-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(weight.to(tl.float16),v);l=l*alpha+tl.sum(weight,1);m=nm
 tl.store(O+((head*nq+rows[:,None])*D+dims[None,:]),acc/tl.maximum(l[:,None],1e-30),rows[:,None]<nq)



def attention(q,state,book,v_group_size=None):
 b,h,n,d=q.shape;nv=state.visual_pos.numel();g=h//state.text_k.shape[1]
 if b!=1 or d!=128 or q.dtype!=torch.float16 or tuple(book.shape)!=(16,4):
  raise ValueError('M1 FP16 G4_16 precenter prototype required')
 if nv%32 or int(state.round12_vmode)!=0:
  raise ValueError('Original whole-image V1 and complete original visual rows required')
 constant=bool(state.round13_precenter_rotation_constant)
 cos,sin=phase_tables(state.visual_pos,state.round13_precenter_freq,
  state.round13_precenter_scaling,rotation_constant=constant)
 qr=wht(q,half=True)
 bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5).contiguous()
 out=torch.empty_like(q,memory_format=torch.contiguous_format)
 rc=getattr(state,'depth_residual_code',state.v_code)
 ra=getattr(state,'depth_residual_alpha',state.v_scale)
 _read[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_mean,state.round13_precenter_mu,
  state.round13_precenter_gain,cos,sin,state.direction_code,state.direction_gain,book,
  state.v_code,state.v_scale,state.v_min,state.v_min,state.v_code,rc,ra,
  state.depth_residual_lookup[0],state.depth_residual_lookup[1],state.text_k,state.text_v,
  state.visual_pos,state.text_pos,out,n,nv,state.text_length,state.length-n,q.stride(1),q.stride(2),
  state.text_k.stride(1),d,g,4,1,True,nv,False,2,4,16,0,16,32,constant,
  num_warps=4,num_stages=2,enable_fp_fusion=False)
 return out


def materialize_position_fields(state,*,return_kernel=False):
 """Transient technical instrument for exact deployed Half field comparison."""
 n=state.visual_pos.numel()
 constant=bool(state.round13_precenter_rotation_constant)
 cos,sin=phase_tables(state.visual_pos,state.round13_precenter_freq,
  state.round13_precenter_scaling,rotation_constant=constant)
 ss=state.direction_mean.new_empty((1,4,n,128));bb=torch.empty_like(ss)
 compiled=_materialize_fields[(triton.cdiv(n,32),4)](state.round13_precenter_mu,state.direction_mean,
  state.round13_precenter_gain,cos,sin,ss,bb,n,128,32,constant,num_warps=4,enable_fp_fusion=False)
 return (ss,bb,compiled) if return_kernel else (ss,bb)
