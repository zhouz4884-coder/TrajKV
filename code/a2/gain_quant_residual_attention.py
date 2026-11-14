"""Packed K2 keys with quantized per-token gains and selected V residuals."""
import torch,triton
import triton.language as tl
from fast_wht import wht
def encode_gain(values,bits):
 """Affine min/max grouping over128 tokens; packed4/8-bit, F16 metadata."""
 if bits not in (4,8) or values.dtype!=torch.float16 or values.shape[-1]%128:raise ValueError('F16 gain rows with complete128 groups and4/8bits required')
 if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite final gain')
 shape=values.shape;group=values.float().reshape(*shape[:-1],-1,128)
 minimum=group.amin(-1).half().contiguous()
 scale=((group.amax(-1)-minimum.float())/((1<<bits)-1)).half().contiguous()
 code=((group-minimum.float()[...,None])/scale.float().clamp_min(1e-20)[...,None]).round().clamp(0,(1<<bits)-1).long().reshape(shape)
 per_word=32//bits;shift=torch.arange(per_word,device=values.device,dtype=torch.int64)*bits
 packed=(code.reshape(*shape[:-1],-1,per_word)<<shift).sum(-1).to(torch.int32).contiguous()
 return packed,scale,minimum


def decode_gain(code,scale,minimum,bits):
 """Reference with separate F16 multiply/add, matching the packed reader."""
 if bits not in (4,8):raise ValueError('Packed gain uses4/8bits')
 shift=torch.arange(32//bits,device=code.device,dtype=torch.int64)*bits
 unpacked=(((code.long()&0xffffffff)[...,None]>>shift)&((1<<bits)-1)).reshape(*code.shape[:-1],-1)
 group=unpacked.reshape(*unpacked.shape[:-1],-1,128).half()
 result=(group*scale[...,None]).half()+minimum[...,None]
 return result.half().reshape(unpacked.shape).contiguous()


def gain_tensors(state):
 if getattr(state,'direction_gain_bits',16)==16:return [state.direction_gain]
 return [state.direction_gain_code,state.direction_gain_scale,state.direction_gain_min]


@triton.jit
def _read(Q,QR,BIAS,KC,GAIN,GS,GM,BOOK,VC,VS,VM,RC,RA,LOOKUP,TK,TV,VP,TP,O,nq,nv,nt,qstart,
          qs1:tl.constexpr,qs2:tl.constexpr,ts1:tl.constexpr,D:tl.constexpr,G:tl.constexpr,KDIM:tl.constexpr,
          VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,HEADWISE:tl.constexpr,GAIN_BITS:tl.constexpr,RESIDUAL_K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 head=tl.program_id(1);owner=head//G;residual_index=tl.load(LOOKUP+owner);rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
 q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
 qr=tl.load(QR+(head*nq+rows[:,None])*D+dims[None,:],rows[:,None]<nq,0)
 bias=tl.load(BIAS+head*nq+rows,rows<nq,0)
 m=tl.full((BM,),-float('inf'),tl.float32);l=tl.full((BM,),0.,tl.float32);acc=tl.full((BM,D),0.,tl.float32);vb=tl.cdiv(nv,BN)
 for block in range(vb+tl.cdiv(nt,BN)):
  if block<vb:
   offset=block*BN+cols;valid=offset<nv
   if GAIN_BITS==16:
    gain=tl.load(GAIN+owner*nv+offset,valid,0).to(tl.float32)
   else:
    gword=tl.load(GAIN+owner*(nv//(32//GAIN_BITS))+offset//(32//GAIN_BITS),valid,0)
    gcode=(gword>>(GAIN_BITS*(offset%(32//GAIN_BITS))))&((1<<GAIN_BITS)-1)
    gscale=tl.load(GS+owner*(nv//128)+offset//128,valid,0)
    gminimum=tl.load(GM+owner*(nv//128)+offset//128,valid,0)
    gain=((gcode.to(tl.float16)*gscale).to(tl.float16)+gminimum).to(tl.float16).to(tl.float32)
   if KDIM==0:
    word=tl.load(KC+(owner*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
    code=(word>>(dims[:,None]%32))&1;k=((2*code.to(tl.float32)-1)*gain[None,:]).to(tl.float16)
   else:
    index=tl.load(KC+(owner*nv+offset[None,:])*(D//KDIM)+dims[:,None]//KDIM,valid[None,:],0).to(tl.int32)
    book_offset=owner*256*KDIM if HEADWISE else 0
    value=tl.load(BOOK+book_offset+index*KDIM+dims[:,None]%KDIM,valid[None,:],0).to(tl.float32);k=(value*gain[None,:]).to(tl.float16)
   if RESIDUAL_K:
    if residual_index>=0:
     rkword=tl.load(RC+(residual_index*nv+offset[None,:])*(D//32)+dims[:,None]//32,valid[None,:],0)
     rkbit=(rkword>>(dims[:,None]%32))&1
     rkalpha=tl.load(RA+residual_index*nv+offset,valid,0).to(tl.float32)
     k=(k.to(tl.float32)+(2*rkbit.to(tl.float32)-1)*rkalpha[None,:]).to(tl.float16)
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
   if not RESIDUAL_K:
    if residual_index>=0:
     rword=tl.load(RC+(residual_index*nv+offset[:,None])*(D//32)+dims[None,:]//32,valid[:,None],0)
     rbit=(rword>>(dims[None,:]%32))&1
     residual_alpha=tl.load(RA+residual_index*nv+offset,valid,0).to(tl.float32)
     v=(v.to(tl.float32)+(2*rbit.to(tl.float32)-1)*residual_alpha[:,None]).to(tl.float16)
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
def attention(q,state,book,v_group_size=128,visual_bias=None):
 b,h,n,d=q.shape;nv=state.visual_pos.numel();g=h//state.text_k.shape[1]
 if b!=1 or d!=128 or q.dtype!=torch.float16 or q.stride(-1)!=1:raise ValueError('Batch1 FP16 D128 required')
 if nv%v_group_size:raise ValueError('Complete V groups required')
 qr=wht(q,half=True);bias=(q.float()*state.direction_mean.repeat_interleave(g,1).float()).sum(-1).div(d**.5)
 if visual_bias is not None:bias=bias+visual_bias.repeat_interleave(g)[None,:,None]
 bias=bias.contiguous();out=torch.empty_like(q,memory_format=torch.contiguous_format)
 rc=getattr(state,'v_residual_code',state.v_code);ra=getattr(state,'v_residual_alpha',state.v_scale)
 gain_bits=getattr(state,'direction_gain_bits',16)
 gains=gain_tensors(state);gp=gains[0];gs=gains[1] if gain_bits!=16 else q;gm=gains[2] if gain_bits!=16 else q
 _read[(triton.cdiv(n,16),h)](q,qr,bias,state.direction_code,gp,gs,gm,book,state.v_code,state.v_scale,state.v_min,rc,ra,state.v_residual_lookup,
  state.text_k,state.text_v,state.visual_pos,state.text_pos,out,n,nv,state.text_length,state.length-n,
  q.stride(1),q.stride(2),state.text_k.stride(1),d,g,book.shape[-1],1,True,v_group_size,book.ndim==3,gain_bits,False,16,32,
  num_warps=4,num_stages=2,enable_fp_fusion=False)
 return out
