"""Round11 packed M1 attention: immutable R7 K/text math plus V stage0 mode.

Kernel copied from R7 b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0.
The new source is self-contained; it does not execute/import old kernel source.
Only pinned fast_wht is obtained through the admitted, already-loaded R7 module.
Spatial values are register-local; no visual FP16 cache or image workspace.
"""
import hashlib
from pathlib import Path
import sys
import torch, triton
import triton.language as tl
from round11_joint_value_v1 import SHUFFLE_A, SHUFFLE_B, SHUFFLE_A_INV

R7_READER_SHA256 = "b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0"
WHT_SHA256 = "9dc73998f66d5b7bafc96a5aa3db57974d1ca8e604e432e7539625f104d90ee3"
_verified_wht = None


def _legacy_wht():
 global _verified_wht
 if _verified_wht is None:
  old = sys.modules.get("round7_key_attention")
  if old is None or not callable(getattr(old, "wht", None)):
   raise RuntimeError("Round11 requires the admitted R7 legacy reader to be loaded first")
  source = Path(old.__file__).resolve()
  wht_source = Path(sys.modules[old.wht.__module__].__file__).resolve()
  for path, digest in ((source, R7_READER_SHA256), (wht_source, WHT_SHA256)):
   if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
    raise RuntimeError("Round11 frozen numerical source changed: " + path.name)
  _verified_wht = old.wht
 return _verified_wht


@triton.jit
def _read(Q,QR,BIAS,KC,GAIN,BOOK,VC,VS,VM,RC,RA,LOOKUP_K,LOOKUP_V,TK,TV,VP,TP,O,nq,nv,nt,qstart,
          qs1:tl.constexpr,qs2:tl.constexpr,ts1:tl.constexpr,D:tl.constexpr,G:tl.constexpr,KDIM:tl.constexpr,
          VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,HEADWISE:tl.constexpr,KPACK:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,SHUFFLED:tl.constexpr,
          SHUFFLE_A:tl.constexpr,SHUFFLE_B:tl.constexpr,SHUFFLE_A_INV:tl.constexpr):
 head=tl.program_id(1);owner=head//G;index_k=tl.load(LOOKUP_K+owner*2);index_k2=tl.load(LOOKUP_K+owner*2+1);index_v=tl.load(LOOKUP_V+owner*2);index_v2=tl.load(LOOKUP_V+owner*2+1)
 rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
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
    if KPACK==2:
     packed=tl.load(KC+(owner*nv+offset[None,:])*(D//(KDIM*2))+dims[:,None]//(KDIM*2),valid[None,:],0).to(tl.int32)
     index=(packed>>(4*((dims[:,None]//KDIM)%2)))&15
    else:
     index=tl.load(KC+(owner*nv+offset[None,:])*(D//KDIM)+dims[:,None]//KDIM,valid[None,:],0).to(tl.int32)
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
     word_v=tl.load(VC+(owner*D+dims[None,:])*(nv//(32//VB))+offset[:,None]//(32//VB),valid[:,None],0)
     code_v=(word_v>>(VB*(offset[:,None]%(32//VB))))&((1<<VB)-1)
     scale=tl.load(VS+(owner*D+dims[None,:])*(nv//VG)+offset[:,None]//VG,valid[:,None],0)
     minimum=tl.load(VM+(owner*D+dims[None,:])*(nv//VG)+offset[:,None]//VG,valid[:,None],0)
    else:
     word_v=tl.load(VC+(owner*nv+offset[:,None])*(D//(32//VB))+dims[None,:]//(32//VB),valid[:,None],0)
     code_v=(word_v>>(VB*(dims[None,:]%(32//VB))))&((1<<VB)-1)
     scale=tl.load(VS+owner*nv+offset,valid,0)[:,None];minimum=tl.load(VM+owner*nv+offset,valid,0)[:,None]
    v=((code_v.to(tl.float16)*scale).to(tl.float16)+minimum).to(tl.float16)
   if index_v>=0:
    word_r=tl.load(RC+(index_v*nv+offset[:,None])*(D//32)+dims[None,:]//32,valid[:,None],0)
    sign_v=(word_r>>(dims[None,:]%32))&1
    # Interpret the stored Half bits before any promotion; -0 selects P1.
    alpha_half=tl.load(RA+index_v*nv+offset,valid,0)
    alpha_bits=alpha_half.to(tl.uint16,bitcast=True)
    mode=(alpha_bits & 32768)!=0
    alpha_v=(alpha_bits & 32767).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)
    # Both modes retain the original rounded base endpoints. In particular A1
    # is Half(Half(1*scale)+minimum), never a raw/unfolded centroid.
    a0=((tl.full((BN,D),0,tl.float16)*scale).to(tl.float16)+minimum).to(tl.float16)
    a1=((tl.full((BN,D),1,tl.float16)*scale).to(tl.float16)+minimum).to(tl.float16)
    local=offset%32
    if SHUFFLED:
     local=(SHUFFLE_A*local+SHUFFLE_B)%32
    left=tl.where(local%16==0,local,local-1)
    right=tl.where(local%16==15,local,local+1)
    vertical=local^16
    if SHUFFLED:
     left=(SHUFFLE_A_INV*(left-SHUFFLE_B))&31
     right=(SHUFFLE_A_INV*(right-SHUFFLE_B))&31
     vertical=(SHUFFLE_A_INV*(vertical-SHUFFLE_B))&31
    bl=(word_v>>left[:,None])&1
    br=(word_v>>right[:,None])&1
    bv=(word_v>>vertical[:,None])&1
    numerator=4*code_v+bl+br+2*bv
    predicted=(((8-numerator).to(tl.float32)*a0.to(tl.float32)+numerator.to(tl.float32)*a1.to(tl.float32))/8).to(tl.float16)
    predictor=tl.where(mode[:,None],predicted,v)
    v=(predictor.to(tl.float32)+(2*sign_v.to(tl.float32)-1)*alpha_v[:,None]).to(tl.float16)
   if index_v2>=0:
    word_r=tl.load(RC+(index_v2*nv+offset[:,None])*(D//32)+dims[None,:]//32,valid[:,None],0)
    sign_v=(word_r>>(dims[None,:]%32))&1
    alpha_v=tl.load(RA+index_v2*nv+offset,valid,0).to(tl.float32)
    v=(v.to(tl.float32)+(2*sign_v.to(tl.float32)-1)*alpha_v[:,None]).to(tl.float16)
   score=tl.dot(qr,k)*(D**-.5)+bias[:,None];pos=tl.load(VP+offset,valid,0)
  else:
   offset=(block-vb)*BN+cols;valid=offset<nt
   k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
   v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
   score=tl.dot(q,k)*(D**-.5);pos=tl.load(TP+offset,valid,0)
  score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
  nm=tl.maximum(m,tl.max(score,1));safe=tl.where(nm==-float('inf'),0.,nm);alpha=tl.exp(m-safe);weight=tl.exp(score-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(weight.to(tl.float16),v);l=l*alpha+tl.sum(weight,1);m=nm
 tl.store(O+((head*nq+rows[:,None])*D+dims[None,:]),acc/tl.maximum(l[:,None],1e-30),rows[:,None]<nq)


def attention(q,state,book,v_group_size=128,visual_bias=None):
 b,h,n,d=q.shape;nv=state.visual_pos.numel();g=h//state.text_k.shape[1]
 if (h, state.text_k.shape[1], g)!=(28,4,7) or nv%256 or v_group_size!=nv:
  raise ValueError('Round11 requires M1 Hq28/Hkv4/GQA7, 16x16 tiles and all-frame V groups')
 if tuple(book.shape)!=(16,4) or state.depth_residual_lookup.dtype!=torch.int32:
  raise ValueError('Round11 preserves G4_16 and the physically charged I32 lookup')
 if state.v_code.dtype!=torch.int32 or state.v_scale.dtype!=torch.float16 or state.v_min.dtype!=torch.float16:
  raise ValueError('Round11 requires original packed V1 and Half all-frame metadata')
 if tuple(state.v_scale.shape)!=(1,4,128,1) or tuple(state.v_min.shape)!=(1,4,128,1):
  raise ValueError('Round11 V endpoints must use the all-frame metadata frame')
 if b!=1 or d!=128 or q.dtype!=torch.float16 or q.stride(-1)!=1:
  raise ValueError('Batch1 FP16 D128 required')
 if tuple(book.shape) not in ((16,4),(256,4)) or state.direction_code.shape[-1]!=(16 if book.shape[0]==16 else 32):
  raise ValueError('Expected actual uint8 G4_256 or nibble-packed G4_16 payload')
 if state.direction_code.dtype!=torch.uint8 or any(hasattr(state,name) for name in ('visual_k','visual_v')):
  raise ValueError('Packed payload and no FP16 visual source required at read')
 if nv%v_group_size or tuple(state.depth_residual_lookup.shape)!=(2,h//g,2):
  raise ValueError('Complete V groups and a [2,Hkv,2] compact stage lookup required')
 qr=_legacy_wht()(q,half=True);bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5)
 if visual_bias is not None:bias=bias+visual_bias.repeat_interleave(g)[None,:,None]
 bias=bias.contiguous();out=torch.empty_like(q,memory_format=torch.contiguous_format)
 rc=getattr(state,'depth_residual_code',state.v_code);ra=getattr(state,'depth_residual_alpha',state.v_scale)
 _read[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_code,state.direction_gain,book,state.v_code,state.v_scale,state.v_min,
  rc,ra,state.depth_residual_lookup[0],state.depth_residual_lookup[1],state.text_k,state.text_v,state.visual_pos,state.text_pos,
  out,n,nv,state.text_length,state.length-n,q.stride(1),q.stride(2),state.text_k.stride(1),d,g,book.shape[-1],1,True,v_group_size,
  book.ndim==3,2 if book.shape[0]==16 else 1,16,32,bool(getattr(state,'round11_shuffled',False)),SHUFFLE_A,SHUFFLE_B,SHUFFLE_A_INV,num_warps=4,num_stages=2,enable_fp_fusion=False)
 return out
