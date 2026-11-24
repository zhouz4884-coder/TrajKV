"""Normalized128-dimensional Walsh-Hadamard transforms, without a matrix owner."""
import math
import torch,triton
import triton.language as tl

def torch_wht(x):
 if x.shape[-1]!=128:raise ValueError('D128 required')
 shape=x.shape;y=x.float()
 for width in (1,2,4,8,16,32,64):
  z=y.reshape(*shape[:-1],128//(2*width),2,width);left=z[...,0,:];right=z[...,1,:]
  y=torch.cat((left+right,left-right),-1).reshape(shape)
 return y*(128**-.5)

@triton.jit
def _butterfly(value,BR:tl.constexpr,WIDTH:tl.constexpr):
 paired=tl.permute(tl.reshape(value,(BR,128//(2*WIDTH),2,WIDTH)),(0,1,3,2))
 left,right=tl.split(paired)
 return tl.reshape(tl.permute(tl.join(left+right,left-right),(0,1,3,2)),(BR,128))

@triton.jit
def _transform(X,Y,ROWS,N,SH,SN,BR:tl.constexpr):
 row=tl.program_id(0)*BR+tl.arange(0,BR);dim=tl.arange(0,128)
 value=tl.load(X+(row[:,None]//N)*SH+(row[:,None]%N)*SN+dim[None,:],row[:,None]<ROWS,0).to(tl.float32)
 value=_butterfly(value,BR,1)
 value=_butterfly(value,BR,2)
 value=_butterfly(value,BR,4)
 value=_butterfly(value,BR,8)
 value=_butterfly(value,BR,16)
 value=_butterfly(value,BR,32)
 value=_butterfly(value,BR,64)
 tl.store(Y+row[:,None]*128+dim[None,:],value*(128**-.5),row[:,None]<ROWS)

def wht(x,half=False):
 if x.ndim!=4 or x.shape[0]!=1 or x.shape[-1]!=128 or x.stride(-1)!=1:raise ValueError('Batch1[...,128] required')
 h,n=x.shape[1:3];out=torch.empty(x.shape,device=x.device,dtype=torch.float16 if half else torch.float32)
 rows=h*n;br=1 if rows<256 else 4
 _transform[(triton.cdiv(rows,br),)](x,out,rows,n,x.stride(1),x.stride(2),br,num_warps=4)
 return out
