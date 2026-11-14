"""Unmodified CalibQuant packed GPU operators in the common HF protocol.

Owns codes and metadata only, with no persistent visual FP16 reconstruction.
Multi-query prefix rows invoke the official single-query operators separately.
Mean-preserving clipping and FP32 score algebra are explicit own ablations.
"""
import torch
from masked_bank import A2Bank,unique_bytes
from calib_bank import load_official
class CalibPackedBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  cfg=type(self).settings;self.official=load_official(legacy_scripts);encode=getattr(self.official.encode,'_torchdynamo_orig_callable',self.official.encode)
  self.bits_k=cfg.get('bits_k',2);self.bits_v=cfg.get('bits_v',2);self.records=[];self.constant_groups=0
  self.values={b:torch.arange(1<<b,device=self.states[0].visual_k.device,dtype=torch.float16) for b in {self.bits_k,self.bits_v}}
  for s in self.states:
   rec=[]
   for axis,bits in (('k',self.bits_k),('v',self.bits_v)):
    x=getattr(s,'visual_'+axis).transpose(1,2).contiguous();center=x.mean(1,keepdim=True) if cfg.get('preserve_center_'+axis,False) else None
    default=(.5 if axis=='k' else .3) if bits==1 else 1.
    input=x if center is None else (x-center).contiguous()
    code,scale,zero=self.official.quant(input,bits,per_channel=True,scale=cfg.get('clip_'+axis,default),post_scale=True)
    constant=scale==0;self.constant_groups+=int(constant.sum())
    if bool(constant.any()):
     code=torch.where(constant,torch.zeros_like(code),code);scale=torch.where(constant,torch.ones_like(scale),scale)
     zero=torch.where(constant,-input[:,:1],zero)
    if center is not None:zero=(zero-center/scale).half().contiguous()
    packed=encode(code.transpose(1,2).long(),bits=bits,datum_per_elem=8//bits)
    rec.append((packed,scale.transpose(1,2).contiguous(),zero.transpose(1,2).contiguous()))
    if not bool(torch.isfinite(scale).all() and torch.isfinite(zero).all()):raise RuntimeError('Invalid packed quantization metadata')
    delattr(s,'visual_'+axis)
   self.records.append(rec)
  self.method=method
 def _visual_tensors(self,s):
  if hasattr(s,'visual_k'):return [s.visual_k,s.visual_v]
  i=next(j for j,t in enumerate(self.states) if t is s)
  return [t for rec in self.records[i] for t in rec]+list(self.values.values())
 def read(self,q,i):
  s=self.states[i];g=q.shape[1]//s.text_k.shape[1];kr,vr=self.records[i]
  kc,ks,kz=[t.repeat_interleave(g,1) for t in kr];vc,vs,vz=[t.repeat_interleave(g,1) for t in vr]
  qs=q*ks
  dots=torch.cat([self.official.fused_kernel(self.values[self.bits_k],qs[:,:,j:j+1].contiguous(),kc,BITS=self.bits_k,DATUM_PER_ELM=8//self.bits_k) for j in range(q.shape[-2])],2)
  high=type(self).settings.get('score_fp32',False)
  z=(dots.float()/q.shape[-1]**.5-((ks.float()*kz.float())*q.float()).sum(-1,keepdim=True)/q.shape[-1]**.5 if high else
     dots/q.shape[-1]**.5-((ks*kz)*q).sum(-1,keepdim=True)/q.shape[-1]**.5)
  tau=type(self).settings.get('tau',-3.)
  if q.shape[-2]==1 and tau:
   lo=z.amin(-1,keepdim=True);hi=z.amax(-1,keepdim=True);z=(z-lo)/(hi-lo+1e-8)*(hi-lo+tau)+lo
  tk=s.text_k[:,:,:s.text_length].repeat_interleave(g,1);tv=s.text_v[:,:,:s.text_length].repeat_interleave(g,1)
  zt=(q@tk.transpose(-1,-2))/q.shape[-1]**.5
  qp=torch.arange(s.length-q.shape[-2],s.length,device=q.device);zt=zt.masked_fill(s.text_pos[:s.text_length][None,None,None,:]>qp[None,None,:,None],-torch.inf)
  weight=torch.cat((z,zt),-1).float().softmax(-1).to(q.dtype);nv=s.visual_pos.numel();wv,wt=weight[...,:nv],weight[...,nv:]
  dotsv=torch.cat([self.official.fused_kernel_wv(self.values[self.bits_v],wv[:,:,j:j+1].contiguous(),vc,BITS=self.bits_v,DATUM_PER_ELM=8//self.bits_v) for j in range(q.shape[-2])],2)
  result=(dotsv*vs-(vs*vz)*wv.sum(-1,keepdim=True))+wt@tv
  e=self.events[(self.turn,self.phase,self.pass_kind,i)];e['calls']+=1;e['query_tokens']+=q.shape[-2]
  e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length);e['base_visual_bytes']+=unique_bytes(self._visual_tensors(s))
  e['text_bytes']+=2*s.text_length*s.text_k.shape[1]*s.text_k.shape[-1]*s.text_k.element_size()
  expansion=2*unique_bytes([kc,ks,kz,vc,vs,vz])+q.numel()*q.element_size()*2
  e['selection_bytes']+=expansion;self.extra_read_bytes+=expansion
  return result
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='official CalibQuant packed QK/WV operators, row-chunked multi-query prefix',
   materialized_quality_port=False,visual_fp16_materialization=False,lowbit_deployment_candidate=True,
   calib_bits_k=self.bits_k,calib_bits_v=self.bits_v,preserve_center_k=type(self).settings.get('preserve_center_k',False),score_fp32=type(self).settings.get('score_fp32',False),constant_groups=self.constant_groups,
   expansion_scope='logical packed GQA replication read/write and query-scale temporaries charged; row-chunked operator work; not physical DRAM or official serving performance',
   calib_scope='unmodified quant/encode/QK/WV primitives, persistent visual-only HF adaptation; tau at single-query decode only; not original model/serving reproduction')
  return r
