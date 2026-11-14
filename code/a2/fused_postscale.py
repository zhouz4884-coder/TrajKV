"""All-support packed attention with FP32 affine offsets after code matmuls.

Same codes and metadata as the pre-dequantization reader. K scales multiply Q,
K offsets enter scores in FP32; V scale/offset contributions accumulate in FP32.
This avoids materializing affine reconstructed KV in FP16 before its matmul.
"""
import torch,triton
import triton.language as tl
@triton.jit
def _read(Q,KC,KS,KM,VC,VS,VM,TK,TV,VP,TP,O,nq,nv,nt,qstart,
          qs1:tl.constexpr,qs2:tl.constexpr,ts1:tl.constexpr,D:tl.constexpr,G:tl.constexpr,
          KB:tl.constexpr,VB:tl.constexpr,KG:tl.constexpr,VG:tl.constexpr,VCHANNEL:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr):
 head=tl.program_id(1);owner=head//G;rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D);cols=tl.arange(0,BN)
 q=tl.load(Q+head*qs1+rows[:,None]*qs2+dims[None,:],rows[:,None]<nq,0)
 m=tl.full((BM,),-float('inf'),tl.float32);l=tl.full((BM,),0.,tl.float32);acc=tl.full((BM,D),0.,tl.float32);vb=tl.cdiv(nv,BN)
 for block in range(vb+tl.cdiv(nt,BN)):
  if VCHANNEL:
   vs=tl.full((D,),0.,tl.float32);vm=tl.full((D,),0.,tl.float32)
  else:
   vs=tl.full((BN,),0.,tl.float32);vm=tl.full((BN,),0.,tl.float32)
  if block<vb:
   offset=block*BN+cols;valid=offset<nv
   packed=tl.load(KC+(owner*D+dims[:,None])*(nv//(32//KB))+offset[None,:]//(32//KB),valid[None,:],0)
   code=((packed>>(KB*(offset[None,:]%(32//KB))))&((1<<KB)-1)).to(tl.float16)
   scale=tl.load(KS+(owner*D+dims)*(nv//KG)+(block*BN)//KG).to(tl.float32)
   minimum=tl.load(KM+(owner*D+dims)*(nv//KG)+(block*BN)//KG).to(tl.float32)
   scaled_q=(q.to(tl.float32)*scale[None,:]).to(tl.float16)
   bias=tl.sum(q.to(tl.float32)*minimum[None,:],1)
   score=(tl.dot(scaled_q,code)+bias[:,None])*(D**-.5)
   if VCHANNEL:
    packedv=tl.load(VC+(owner*D+dims[None,:])*(nv//(32//VB))+offset[:,None]//(32//VB),valid[:,None],0)
    v=((packedv>>(VB*(offset[:,None]%(32//VB))))&((1<<VB)-1)).to(tl.float16)
    vs=tl.load(VS+(owner*D+dims)*(nv//VG)+(block*BN)//VG).to(tl.float32)
    vm=tl.load(VM+(owner*D+dims)*(nv//VG)+(block*BN)//VG).to(tl.float32)
   else:
    packedv=tl.load(VC+(owner*nv+offset[:,None])*(D//(32//VB))+dims[None,:]//(32//VB),valid[:,None],0)
    v=((packedv>>(VB*(dims[None,:]%(32//VB))))&((1<<VB)-1)).to(tl.float16)
    vs=tl.load(VS+owner*nv+offset,valid,0).to(tl.float32);vm=tl.load(VM+owner*nv+offset,valid,0).to(tl.float32)
   pos=tl.load(VP+offset,valid,0)
  else:
   offset=(block-vb)*BN+cols;valid=offset<nt
   k=tl.load(TK+owner*ts1+offset[None,:]*D+dims[:,None],valid[None,:],0)
   v=tl.load(TV+owner*ts1+offset[:,None]*D+dims[None,:],valid[:,None],0)
   score=tl.dot(q,k)*(D**-.5);pos=tl.load(TP+offset,valid,0)
  score=tl.where(valid[None,:]&(pos[None,:]<=qstart+rows[:,None]),score,-float('inf'))
  nm=tl.maximum(m,tl.max(score,1));safe=tl.where(nm==-float('inf'),0.,nm);alpha=tl.exp(m-safe);w=tl.exp(score-safe[:,None])
  if block<vb:
   if VCHANNEL:contribution=tl.dot(w.to(tl.float16),v)*vs[None,:]+tl.sum(w,1)[:,None]*vm[None,:]
   else:contribution=tl.dot((w*vs[None,:]).to(tl.float16),v)+tl.sum(w*vm[None,:],1)[:,None]
  else:contribution=tl.dot(w.to(tl.float16),v)
  acc=acc*alpha[:,None]+contribution;l=l*alpha+tl.sum(w,1);m=nm
 result=acc/tl.maximum(l[:,None],1e-30)
 tl.store(O+((head*nq+rows[:,None])*D+dims[None,:]),result,rows[:,None]<nq)

def attention(q,state,bits=1,bits_v=1,v_channel=True,k_group_size=128,v_group_size=128):
 b,h,n,d=q.shape
 if b!=1 or d!=128 or q.dtype!=torch.float16 or q.stride(-1)!=1:raise ValueError('Batch1 FP16 D128 required')
 nv=state.visual_pos.numel()
 if nv%32 or nv%k_group_size or (nv if v_channel else d)%v_group_size or k_group_size%32 or (v_channel and v_group_size%32):raise ValueError('Aligned complete affine groups required')
 if not v_channel and v_group_size!=d:raise ValueError('Postscale token V uses one full-channel group')
 o=torch.empty_like(q,memory_format=torch.contiguous_format)
 _read[(triton.cdiv(n,16),h)](q,state.k_code,state.k_scale,state.k_min,state.v_code,state.v_scale,state.v_min,
  state.text_k,state.text_v,state.visual_pos,state.text_pos,o,n,nv,state.text_length,state.length-n,
  q.stride(1),q.stride(2),state.text_k.stride(1),d,h//state.text_k.shape[1],bits,bits_v,k_group_size,v_group_size,v_channel,16,32,
  num_warps=4,num_stages=2,enable_fp_fusion=False)
 return o
