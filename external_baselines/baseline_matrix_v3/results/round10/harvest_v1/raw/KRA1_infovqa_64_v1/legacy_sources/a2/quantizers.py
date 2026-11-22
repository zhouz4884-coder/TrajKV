"""Scalar binary centroid fitting, using the existing packed affine layout.

Lloyd iterations and weighted squared-error centroids are standard quantization
building blocks. This module changes encoding only: the decoder still owns one
bit per scalar and FP16 (centroid0, centroid1-centroid0) per 128 values.
"""
import torch
def pack1(codes):
 if codes.shape[-1]%32:raise ValueError('Complete 32-bit words required')
 c=codes.to(torch.int64).reshape(*codes.shape[:-1],-1,32)
 shift=torch.arange(32,device=codes.device,dtype=torch.int64)
 return (c<<shift).sum(-1).to(torch.int32).contiguous()
def binary_centroids(data,weights=None,iterations=3,group_size=128):
 if data.shape[-1]%group_size:raise ValueError('Complete quantization groups required')
 shape=data.shape;g=data.float().reshape(*shape[:-1],-1,group_size)
 w=torch.ones_like(g) if weights is None else weights.float().expand_as(g)
 c=(g>(g.amin(-1,keepdim=True)+g.amax(-1,keepdim=True))/2)
 for step in range(iterations):
  a=w*(~c);b=w*c;na=a.sum(-1,keepdim=True);nb=b.sum(-1,keepdim=True)
  m0=(a*g).sum(-1,keepdim=True)/na.clamp_min(1e-12)
  m1=(b*g).sum(-1,keepdim=True)/nb.clamp_min(1e-12)
  m0=torch.where(na>0,m0,m1);m1=torch.where(nb>0,m1,m0)
  if step+1<iterations:c=g>(m0+m1)/2
 minimum=m0.squeeze(-1).half().contiguous();scale=(m1-m0).squeeze(-1).half().contiguous()
 return pack1(c.reshape(shape)),scale,minimum

def mean_abs_levels(data,centered=True,group_size=128):
 """Optimal symmetric levels around a fixed mean/zero threshold, not general Lloyd-Max."""
 if data.shape[-1]%group_size:raise ValueError('Complete quantization groups required')
 g=data.float().reshape(*data.shape[:-1],-1,group_size)
 center=g.mean(-1,keepdim=True) if centered else torch.zeros_like(g[...,:1])
 delta=g-center;amplitude=delta.abs().mean(-1,keepdim=True)
 codes=delta>0
 return pack1(codes.reshape(data.shape)),(2*amplitude).squeeze(-1).half().contiguous(),(center-amplitude).squeeze(-1).half().contiguous()

def affine_rtn(data,bits,group_size):
 """Explicit extension for non-g128 metadata ablations; original g128 uses upstream."""
 if bits not in (1,2,4) or data.shape[-1]%group_size:raise ValueError('Invalid RTN layout')
 g=data.float().reshape(*data.shape[:-1],-1,group_size);minimum=g.amin(-1).half()
 scale=((g.amax(-1)-minimum.float())/((1<<bits)-1)).half()
 c=((g-minimum.float()[...,None])/scale.float().clamp_min(1e-20)[...,None]).round().clamp(0,(1<<bits)-1).long().reshape(data.shape)
 pack=32//bits;grouped=c.reshape(*c.shape[:-1],-1,pack);shift=torch.arange(pack,device=data.device,dtype=torch.int64)*bits
 code=(grouped<<shift).sum(-1).int().contiguous()
 return code,scale.contiguous(),minimum.contiguous()
