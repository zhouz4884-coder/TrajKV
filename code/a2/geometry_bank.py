"""Same-query, actual-model geometry measurements under FP16 trajectories.

This is a diagnostic forward, with all candidate keys reconstructed temporarily.
It does not enter compressed runtime, storage, or free-answer accuracy tables.
"""
import math,torch
from masked_bank import A2Bank
from quantizers import mean_abs_levels,binary_centroids
from fused_progressive import decode_last_generic

class GeometryBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  self.geometry=[];self.measured=set();self.calib_checks=[];self.legacy_scripts=legacy_scripts

 def read(self,q,i):
  if i not in self.measured:
   self.measured.add(i);s=self.states[i];x=s.visual_k;data=x.transpose(2,3).contiguous()
   xx=x.float();query=q[:,:,-1:].float();g=q.shape[1]//x.shape[1]
   ztrue=query@xx.repeat_interleave(g,1).transpose(-1,-2)/math.sqrt(x.shape[-1])
   true_lp=ztrue.log_softmax(-1);true_w=true_lp.exp();rtrue=xx.norm(dim=-1)
   states=[]
   for enc in ('rtn1','sign','mad','lloyd','rtn2'):
    bits=2 if enc=='rtn2' else 1
    rec=(mean_abs_levels(data,centered=enc=='mad') if enc in ('mad','sign') else
         binary_centroids(data) if enc=='lloyd' else self.codec.triton_quantize_and_pack_along_last_dim(data,128,bits))
    quant=decode_last_generic(*rec,128,bits).transpose(2,3).contiguous().float();states.append((enc,quant))
    if enc in ('mad','lloyd'):
     norm=quant.norm(dim=-1).clamp_min(1e-20)
     states.append((enc+'_norm',quant*(rtrue/norm)[...,None]))
     ls=(xx*quant).sum(-1)/quant.square().sum(-1).clamp_min(1e-20)
     states.append((enc+'_origin_ls',quant*ls.clamp_min(0)[...,None]))
   for name,y in states:
    zy=query@y.repeat_interleave(g,1).transpose(-1,-2)/math.sqrt(x.shape[-1]);delta=zy-ztrue
    lp=zy.log_softmax(-1);weight=lp.exp();rquant=y.norm(dim=-1);cos=(xx*y).sum(-1)/(rtrue*rquant).clamp_min(1e-20)
    # Visual-only output is a local proxy: identical true V for all candidates.
    v=s.visual_v.float().repeat_interleave(g,1)
    output_true=true_w@v;output_error=(weight@v-output_true).square().sum(-1)
    normerr=(rtrue-rquant).square();angerr=2*rtrue*rquant*(1-cos.clamp(-1,1))
    for head in range(query.shape[1]):
     owner=head//g;a=ztrue[0,head,0];b=zy[0,head,0];ac=a-a.mean();bc=b-b.mean()
     den=ac.norm()*bc.norm();cor=float((ac*bc).sum()/den) if float(den)>1e-20 else None
     ratio=float(a.std(unbiased=False)/delta[0,head,0].std(unbiased=False).clamp_min(1e-20))
     gap=a.topk(2).values;var=xx[0,owner].square().sum(-1).mean()
     self.geometry.append({'layer':i,'query_head':head,'kv_head':owner,'encoder':name,'n_visual':x.shape[-2],
      'normalized_key_mse':float((xx[0,owner]-y[0,owner]).square().sum(-1).mean()/var.clamp_min(1e-20)),
      'mean_cosine':float(cos[0,owner].mean()),'radial_energy':float(normerr[0,owner].mean()),'angular_energy':float(angerr[0,owner].mean()),
      'logit_correlation':cor,'logit_snr':ratio,'logit_top1_agreement':int(a.argmax()==b.argmax()),
      'logit_kl_visual':float((true_w[0,head,0]*(true_lp[0,head,0]-lp[0,head,0])).sum()),
      'top_gap_over_logit_std':float((gap[0]-gap[1])/a.std(unbiased=False).clamp_min(1e-20)),
      'visual_output_relative_squared_error':float(output_error[0,head,0]/output_true[0,head,0].square().sum().clamp_min(1e-20)),
      'scope':'same last question-prefix query, FP16 path, visual-only attention; local diagnostic'})
   if i in (0,14,27) and type(self).settings.get('check_calib_official',False):
    self._calib_check(q[:,:,-1:].contiguous(),s,i)
  return super().read(q,i)

 def _calib_check(self,q,s,i):
  from calib_bank import load_official
  m=load_official(self.legacy_scripts);encode=getattr(m.encode,'_torchdynamo_orig_callable',m.encode)
  g=q.shape[1]//s.visual_k.shape[1];values=torch.tensor([0.,1.],device=q.device,dtype=q.dtype)
  rows={}
  for axis,clip in (('k',.5),('v',.3)):
   x=getattr(s,'visual_'+axis).transpose(1,2).contiguous()
   c,scale,zero=m.quant(x,1,per_channel=True,scale=clip,post_scale=True)
   code=c.transpose(1,2).contiguous();packed=encode(code.long(),bits=1,datum_per_elem=8).repeat_interleave(g,1)
   rows[axis]=(code.repeat_interleave(g,1),packed,scale.transpose(1,2).repeat_interleave(g,1))
  code,packed,sc=rows['k'];qs=q*sc
  official=m.fused_kernel(values,qs,packed,BITS=1,DATUM_PER_ELM=8);port=qs@code.transpose(-1,-2)
  w=port.float().softmax(-1).half();cv,pv,_=rows['v'];ov=m.fused_kernel_wv(values,w,pv,BITS=1,DATUM_PER_ELM=8);vv=w@cv
  self.calib_checks.append({'layer':i,'qk_max_abs':float((official-port).abs().max()),'wv_max_abs':float((ov-vv).abs().max()),
    'scope':'official packed operators vs same real-model discrete code matmuls, not original CalibQuant full reproduction'})

 def result_metadata(self):
  r=super().result_metadata();r.update(geometry_diagnostic=self.geometry,calib_actual_model_operator_checks=self.calib_checks,
    diagnostic_only=True,task_accuracy_eligible=False,backend='actual-model FP16 query geometry diagnostics')
  return r
