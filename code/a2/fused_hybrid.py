"""Packed KVarN2 visual keys + clipped channel-wise one-bit visual values.

All visual support and FP16 text remain. Visual scores use Hadamard-rotated
queries; text scores use original queries. Stable tiled online softmax follows
the FlashAttention algorithm, as in the project's frozen P202 implementation.
No materialized visual K/V or full attention matrix is created by this reader.
"""
import torch,triton
import triton.language as tl

@triton.jit
def _read(Q,QR,KC,KS,KZ,KR,VC,VS,VZ,TK,TV,VP,TP,O,
          nq,nv,nt,qstart,
          qs0:tl.constexpr,qs1:tl.constexpr,qs2:tl.constexpr,
          ts0:tl.constexpr,ts1:tl.constexpr,
          H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,NATIVE_K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 head=tl.program_id(1);owner=head//G
 rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
 q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
 if NATIVE_K:qr=q
 else:qr=tl.load(QR+(head*nq+rows[:,None])*D+dims[None,:],rows[:,None]<nq,0)
 m=tl.full((BM,),-float('inf'),tl.float32);l=tl.full((BM,),0.,tl.float32);acc=tl.full((BM,D),0.,tl.float32)
 vb=tl.cdiv(nv,BN)
 for block in range(vb+tl.cdiv(nt,BN)):
  if block<vb:
   offset=block*BN+cols;valid=offset<nv;tile=owner*(nv//128)+offset//128
   if NATIVE_K:
    packed=tl.load(KC+(owner*D+dims[:,None])*(nv//16)+offset[None,:]//16,valid[None,:],0)
    code=(packed>>(2*(offset[None,:]%16)))&3
    scale=tl.load(KS+(owner*D+dims[:,None])*(nv//128)+offset[None,:]//128,valid[None,:],0)
    zero=tl.load(KZ+(owner*D+dims[:,None])*(nv//128)+offset[None,:]//128,valid[None,:],0)
    k=((code.to(tl.float16)*scale).to(tl.float16)+zero).to(tl.float16)
   else:
    packed=tl.load(KC+(tile[None,:]*D+dims[:,None])*32+(offset[None,:]%128)//4,valid[None,:],0)
    code=(packed>>(2*(offset[None,:]%4)))&3
    scale=tl.load(KS+tile[None,:]*D+dims[:,None],valid[None,:],0).to(tl.float32)
    zero=tl.load(KZ+tile[None,:]*D+dims[:,None],valid[None,:],0).to(tl.float32)
    row=tl.load(KR+tile*128+offset%128,valid,0).to(tl.float32)
    k=((code.to(tl.float32)*scale+zero)*row[None,:]).to(tl.float16)
   vcode=tl.load(VC+(owner*nv+offset[:,None])*(D//8)+dims[None,:]//8,valid[:,None],0)
   vi=(vcode>>(7-dims[None,:]%8))&1
   vs=tl.load(VS+owner*D+dims[None,:]).to(tl.float16)
   vz=tl.load(VZ+owner*D+dims[None,:]).to(tl.float16)
   v=((vi.to(tl.float16)-vz).to(tl.float16)*vs).to(tl.float16)
   pos=tl.load(VP+offset,valid,0);qx=qr
  else:
   offset=(block-vb)*BN+cols;valid=offset<nt
   k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
   v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
   pos=tl.load(TP+offset,valid,0);qx=q
  score=tl.dot(qx,k)*(D**-.5)
  score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
  nm=tl.maximum(m,tl.max(score,1));safe=tl.where(nm==-float('inf'),0.,nm)
  alpha=tl.exp(m-safe);w=tl.exp(score-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(w.to(tl.float16),v);l=l*alpha+tl.sum(w,1);m=nm
 result=acc/tl.maximum(l[:,None],1e-30)
 tl.store(O+((head*nq+rows[:,None])*D+dims[None,:]),result,rows[:,None]<nq)

def attention(q,state,key_record,value_record,hadamard=None,native_k=False):
 b,h,n,d=q.shape
 if b!=1 or d!=128 or q.dtype!=torch.float16 or state.visual_pos.numel()%128 or q.stride(-1)!=1:raise ValueError('Batch1 FP16 d128, complete g128 required')
 if native_k:
  qr=q;kargs=(*key_record,q)
 else:
  if key_record.full_tokens!=state.visual_pos.numel():raise ValueError('Hybrid reader requires no FP16 visual tail')
  if key_record.bits!=2 or key_record.axis!='k':raise ValueError('KVarN2 keys required')
  qr=(q.float()@hadamard).half().contiguous();p=key_record.packed
  kargs=(p['q_packed_uint8'],p['s_col_K'],p['zp_K'],p['s_row_K'])
 o=torch.empty_like(q, memory_format=torch.contiguous_format)
 _read[(triton.cdiv(n,16),h)](q,qr,*kargs,*value_record,
  state.text_k,state.text_v,state.visual_pos,state.text_pos,o,n,state.visual_pos.numel(),state.text_length,state.length-n,
  *q.stride()[:3],*state.text_k.stride()[:2],h,h//state.text_k.shape[1],d,native_k,16,32,
  num_warps=4,num_stages=2,enable_fp_fusion=False)
 return o
