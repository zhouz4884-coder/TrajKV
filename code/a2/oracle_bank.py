"""Diagnostic radial/angular interventions on true FP16 visual keys.

All oracle arms own materialized FP16 visual KV. They establish mechanism
contrasts, never low-bit storage or deployment-quality claims.
"""
import torch
from masked_bank import A2Bank
from quantizers import mean_abs_levels,binary_centroids
from fused_progressive import decode_last_generic
class OracleKBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  cfg=type(self).settings;self.norm_stats=[];encoder=cfg.get('encoder','mad');mode=cfg['oracle_mode']
  for layer,s in enumerate(self.states):
   x=s.visual_k;data=x.transpose(2,3).contiguous()
   rec=binary_centroids(data) if encoder=='lloyd' else mean_abs_levels(data,centered=encoder!='sign')
   q=decode_last_generic(*rec,128,1).transpose(2,3).contiguous();xf=x.float();qf=q.float()
   rt=xf.norm(dim=-1);rq=qf.norm(dim=-1);ut=xf/rt[...,None].clamp_min(1e-20);uq=qf/rq[...,None].clamp_min(1e-20)
   if mode=='quant_norm_quant_direction':new=q
   elif mode=='true_norm_quant_direction':new=(rt[...,None]*uq).half()
   elif mode=='quant_norm_true_direction':new=(rq[...,None]*ut).half()
   elif mode=='true_norm_true_direction':new=x
   else:raise ValueError('Unknown radial/angular intervention')
   s.visual_k=new.contiguous()
   def corr(a,b):
    aa=a-a.mean();bb=b-b.mean();den=aa.square().sum().sqrt()*bb.square().sum().sqrt()
    return float((aa*bb).sum()/den) if float(den)>1e-12 else None
   cosine=(ut*uq).sum(-1).clamp(-1,1);radial=(rt-rq).square();angular=2*rt*rq*(1-cosine)
   for h in range(x.shape[1]):
    r0=rt[0,h];r1=rq[0,h];g0=r0.reshape(-1,128);g1=r1.reshape(-1,128)
    w0=(g0-g0.mean(-1,keepdim=True)).flatten();w1=(g1-g1.mean(-1,keepdim=True)).flatten()
    self.norm_stats.append({'layer':layer,'kv_head':h,'true_norm_cv':float(r0.std(unbiased=False)/r0.mean().clamp_min(1e-20)),
     'quant_norm_cv':float(r1.std(unbiased=False)/r1.mean().clamp_min(1e-20)),'norm_correlation_global':corr(r0,r1),'norm_correlation_within_group':corr(w0,w1),
     'mean_cosine':float(cosine[0,h].mean()),'radial_squared_error':float(radial[0,h].mean()),'angular_squared_error':float(angular[0,h].mean()),
     'total_squared_error':float((xf[0,h]-qf[0,h]).square().sum(-1).mean())})
  self.method=method
 def result_metadata(self):
  r=super().result_metadata();r.update(oracle=True,lowbit_deployment_candidate=False,oracle_mode=type(self).settings['oracle_mode'],
    oracle_encoder=type(self).settings.get('encoder','mad'),radial_angular_statistics=self.norm_stats,
    backend='FP16 materialized radial/angular causal diagnostic; V exactly original FP16')
  return r
