"""CalibQuant post-scale operation order, discrete materialized codes quality port.

Preserves quantization's scale/zero outside the K and V dot products, matching
the published implementation's FP16 operation order more closely than a
materialized dequantized-KV approximation. Still no official serving claim.
"""
import torch
from calib_bank import CalibBank,unique_bytes
class CalibPostBank(CalibBank):
 def _initialize(self,*args,**kwargs):
  super()._initialize(*args,**kwargs)
  for s,rec in zip(self.states,self.calib_records):
   for axis,packed in (('k',rec[0]),('v',rec[3])):
    bits=type(self).settings.get('bits_'+axis,1)
    shift=torch.arange(8-bits,-1,-bits,device=packed.device)
    codes=((packed.long()[...,None]>>shift)&((1<<bits)-1)).reshape(*packed.shape[:-1],packed.shape[-1]*(8//bits)).half()
    setattr(s,'visual_'+axis,codes)
  self.score_range_statistics={'heads':0,'range_below3':0,'minimum_range':float('inf')}
 def read(self,q,i):
  s=self.states[i];rec=self.calib_records[i];g=q.shape[1]//s.visual_k.shape[1]
  ks,kz=rec[1].transpose(1,2).repeat_interleave(g,1),rec[2].transpose(1,2).repeat_interleave(g,1)
  vs,vz=rec[4].transpose(1,2).repeat_interleave(g,1),rec[5].transpose(1,2).repeat_interleave(g,1)
  k=s.visual_k.repeat_interleave(g,1);v=s.visual_v.repeat_interleave(g,1)
  z=((q*ks)@k.transpose(-1,-2))/q.shape[-1]**.5-((ks*kz)*q).sum(-1,keepdim=True)/q.shape[-1]**.5
  if q.shape[-2]==1:
   low=z.amin((-2,-1),keepdim=True);high=z.amax((-2,-1),keepdim=True);tau=type(self).settings.get('tau',-3.)
   # Record diagnostic on the first eight decode events only; no policy uses it.
   if self.score_range_statistics['heads']<224:
    ranges=(high-low).float();self.score_range_statistics['heads']+=ranges.numel()
    self.score_range_statistics['range_below3']+=int((ranges<3).sum())
    self.score_range_statistics['minimum_range']=min(self.score_range_statistics['minimum_range'],float(ranges.min()))
   if tau:z=(z-low)/(high-low+1e-8)*(high-low+tau)+low
  tk=s.text_k[:,:,:s.text_length].repeat_interleave(g,1);tv=s.text_v[:,:,:s.text_length].repeat_interleave(g,1)
  zt=(q@tk.transpose(-1,-2))/q.shape[-1]**.5
  qp=torch.arange(s.length-q.shape[-2],s.length,device=q.device)
  zt=zt.masked_fill(s.text_pos[:s.text_length][None,None,None,:]>qp[None,None,:,None],-torch.inf)
  scores=torch.cat((z,zt),-1);weight=scores.float().softmax(-1).to(q.dtype);nv=z.shape[-1]
  wv,wt=weight[...,:nv],weight[...,nv:]
  outv=(wv@v)*vs-(vs*vz)*wv.sum(-1,keepdim=True)
  result=outv+wt@tv
  e=self.events[(self.turn,self.phase,self.pass_kind,i)];e['calls']+=1;e['query_tokens']+=q.shape[-2]
  e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length)
  e['base_visual_bytes']+=unique_bytes((s.visual_k,s.visual_v))+unique_bytes((rec[1],rec[2],rec[4],rec[5]))
  e['text_bytes']+=2*s.text_length*s.text_k.shape[1]*s.text_k.shape[-1]*s.text_k.element_size()
  return result
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='CalibQuant post-scale numerical quality port',calib_score_range=self.score_range_statistics,
    calib_tau=type(self).settings.get('tau',-3.),calib_clip_k=type(self).settings.get('clip_k',.5),calib_clip_v=type(self).settings.get('clip_v',.3),
    calib_scope='official quant/encode and post-scale algebra; PyTorch discrete FP16 code matmuls; modern multi-turn HF; tau for one-query decode only')
  return r
