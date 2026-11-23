"""Actual K2/V1 or K1/V2 bases, and K1+1-bit-residual/V1 representations."""
from masked_bank import A2Bank,unique_bytes
from fused_progressive import decode_last_generic
from fused_rates import attention
from quantizers import binary_centroids,mean_abs_levels,affine_rtn
import hashlib,json
from pathlib import Path
import torch
class RateBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  cfg=type(self).settings;self.bits=cfg.get('bits_k',1);self.vbits=cfg.get('bits_v',1);self.kr_bits=cfg.get('kr_bits',0)
  self.layer_kbits=cfg.get('layer_bits_k',[self.bits]*len(self.states));self.layer_vbits=cfg.get('layer_bits_v',[self.vbits]*len(self.states));self.encoder_stats=[]
  self.v_channel=cfg.get('v_layout','token')=='channel'
  learned=None;learned_value=None;self.encoder_parameter_bytes=0;self.calibration_hash=None
  if cfg.get('k_fold_factor_path'):
   path=Path(cfg['k_fold_factor_path']);data=json.loads(path.read_text());self.calibration_hash=hashlib.sha256(path.read_bytes()).hexdigest()
   learned=torch.tensor(data['head_factors'],device=self.states[0].visual_k.device,dtype=torch.float32)
   if learned.shape!=(len(self.states),self.states[0].visual_k.shape[1]) or not bool(torch.isfinite(learned).all() and (learned>0).all()):raise ValueError('Invalid calibrated head factors')
   self.encoder_parameter_bytes=learned.numel()*learned.element_size()
  if cfg.get('v_fold_factor_path'):
   path=Path(cfg['v_fold_factor_path']);data=json.loads(path.read_text());learned_value=torch.tensor(data['value_head_factors'],device=self.states[0].visual_k.device,dtype=torch.float32)
   if learned_value.shape!=(len(self.states),self.states[0].visual_k.shape[1]) or not bool(torch.isfinite(learned_value).all() and (learned_value>0).all()):raise ValueError('Invalid calibrated value factors')
   self.encoder_parameter_bytes+=learned_value.numel()*learned_value.element_size()
  nv=self.states[0].visual_pos.numel()
  self.k_group_size=nv if cfg.get('k_group_size')=='all' else cfg.get('k_group_size',128)
  self.v_group_size=(nv if self.v_channel else 128) if cfg.get('v_group_size')=='all' else cfg.get('v_group_size',128)
  self.refine_k=True;self.refine_v=False;active=set(cfg.get('k_layers',range(len(self.states))));vactive=set(cfg.get('v_layers',[]));self.refinement_flags=[]
  from p201_refinement import quantize
  from p202_refinement import pack
  for i,s in enumerate(self.states):
   rk=rv=None
   for axis,bits in (('k',self.layer_kbits[i]),('v',self.layer_vbits[i])):
    x=getattr(s,'visual_'+axis);channel_layout=axis=='k' or (axis=='v' and self.v_channel)
    data=x.transpose(2,3).contiguous() if channel_layout else x
    group_size=self.k_group_size if axis=='k' else self.v_group_size
    encoded_input=data
    if axis=='k' and cfg.get('k_clip',1.)!=1.:
     group=data.float().reshape(*data.shape[:-1],-1,128);lo=group.amin(-1,keepdim=True);hi=group.amax(-1,keepdim=True);mid=(lo+hi)/2;beta=cfg['k_clip']
     encoded_input=group.clamp(min=mid+beta*(lo-mid),max=mid+beta*(hi-mid)).half().reshape(data.shape).contiguous()
    layer_encoders=cfg.get('layer_encoders_'+axis)
    encoder=layer_encoders[i] if layer_encoders is not None else cfg.get(axis+'_encoder','rtn')
    selected_layers=cfg.get('centroid_layers')
    if selected_layers is not None and i not in selected_layers:encoder='rtn'
    if encoder in ('mad','sign'):
     if bits!=1:raise ValueError('MAD/sign levels require one-bit indices')
     rec=mean_abs_levels(encoded_input,centered=encoder=='mad',group_size=group_size)
    elif encoder in ('lloyd','weighted_lloyd'):
     if bits!=1:raise ValueError('Binary centroids require one-bit indices')
     weights=None
     if encoder=='weighted_lloyd':
      if axis!='v':raise ValueError('Output-column weighting currently defined for V')
      if self.v_channel:raise ValueError('A fixed channel weight cancels inside a channel group; use unweighted Lloyd')
      module=self._parent.modules[i];wo=module.o_proj.weight
      d=data.shape[-1];h=data.shape[1];groups=wo.shape[1]//(h*d)
      weights=wo.float().square().sum(0).reshape(h,groups,d).sum(1)
      weights=(weights/weights.mean(-1,keepdim=True).clamp_min(1e-12))[None,:,None,None,:]
     rec=binary_centroids(encoded_input,weights=weights,iterations=3,group_size=group_size)
    else:
     rec=(self.codec.triton_quantize_and_pack_along_last_dim(encoded_input,128,bits) if group_size==128 else affine_rtn(encoded_input,bits,group_size))
     if encoder=='folded_sign':
      if bits!=1:raise ValueError('Folded sign prototype requires one-bit indices')
      base=decode_last_generic(*rec,128,bits);error=(data.float()-base.float()).reshape(*data.shape[:-1],-1,128)
      alpha=error.abs().mean(-1)
      rec=(rec[0],(rec[1].float()-2*alpha).clamp_min(0).half().contiguous(),(rec[2].float()+alpha).half().contiguous())
    if axis=='k' and cfg.get('k_fold_gain'):
     base=decode_last_generic(*rec,group_size,bits).float();center=(rec[2].float()+.5*((1<<bits)-1)*rec[1].float()).repeat_interleave(group_size,-1)
     centered=cfg['k_fold_gain']=='head_residual_norm'
     target=data.float()-center if centered else data.float();template=base-center if centered else base
     beta=(target.square().mean((-2,-1),keepdim=True)/template.square().mean((-2,-1),keepdim=True).clamp_min(1e-20)).sqrt()*cfg.get('k_fold_gain_factor',1.)
     if learned is not None:beta=beta*learned[i][None,:,None,None]
     scale=(rec[1].float()*beta).half().contiguous()
     minimum=((rec[2].float()+.5*((1<<bits)-1)*rec[1].float())-.5*((1<<bits)-1)*scale.float() if centered else rec[2].float()*beta).half().contiguous()
     rec=(rec[0],scale,minimum)
    if axis=='v' and learned_value is not None:
     middle=rec[2].float()+.5*((1<<bits)-1)*rec[1].float();scale=(rec[1].float()*learned_value[i][None,:,None,None]).half().contiguous()
     rec=(rec[0],scale,(middle-.5*((1<<bits)-1)*scale.float()).half().contiguous())
    for name,t in zip(('code','scale','min'),rec):setattr(s,axis+'_'+name,t)
    if axis=='k' and cfg.get('k_token_gain'):
     if bits!=1 or self.kr_bits:raise ValueError('Token gain requires K1 without K residual')
     base=decode_last_generic(*rec,group_size,1).float();center=(rec[2].float()+.5*rec[1].float()).repeat_interleave(group_size,-1)
     origin=cfg['k_token_gain'].startswith('origin')
     template=base if origin else base-center;target=data.float() if origin else data.float()-center;reduce=(1,2) if cfg.get('gain_shared',False) else 2
     denom=template.square().sum(reduce)
     if cfg['k_token_gain'] in ('norm','origin_norm'):gain=(target.square().sum(reduce)/denom.clamp_min(1e-20)).sqrt()
     else:gain=(target*template).sum(reduce)/denom.clamp_min(1e-20)
     gain=torch.where(denom>1e-20,gain,torch.ones_like(gain));s.k_gain=gain.half().contiguous()
     if cfg['k_token_gain']=='origin_ls':s.k_gain=s.k_gain.clamp_min(0)
     if cfg.get('gain_shared',False):s.k_gain=s.k_gain[:,None,:]
     if not bool(torch.isfinite(s.k_gain).all()):raise RuntimeError('Nonfinite token gain')
    if axis=='k' and self.kr_bits and i in active:
     base=decode_last_generic(*rec,group_size,bits);_,encoded=quantize(data.float()-base.float(),self.kr_bits)
     rk=list(encoded)
    if axis=='v' and i in vactive:
     base=decode_last_generic(*rec,group_size,bits);base=base.transpose(2,3) if channel_layout else base
     error=(x.float()-base.float()).reshape(*x.shape[:-1],-1,128)
     alpha=error.abs().mean(-1).half();rv=[pack((error>=0).byte(),1),alpha,alpha]
    delattr(s,'visual_'+axis)
   self.refinement_flags.append((rk is not None,rv is not None))
   self.refinements[i]=(rk or rv)+(rv or rk) if rk is not None or rv is not None else None
  self.method=method
 def read(self,q,i):
  s=self.states[i];r=self.refinements[i];e=self.events[(self.turn,self.phase,self.pass_kind,i)]
  e['calls']+=1;e['query_tokens']+=q.shape[-2];e['base_visual_bytes']+=unique_bytes(self._visual_tensors(s))
  e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length)
  e['text_bytes']+=sum(t[:,:,:s.text_length].numel()*t.element_size() for t in (s.text_k,s.text_v))
  fk,fv=self.refinement_flags[i];active=(r[:3] if fk else [])+(r[3:] if fv else []) if r else []
  n=unique_bytes(active);e['residual_bytes']+=n;self.extra_read_bytes+=n
  if type(self).settings.get('postscale_reader',False):
   if fk or fv or hasattr(s,'k_gain'):raise ValueError('Postscale comparison currently uses base affine representations only')
   from fused_postscale import attention as post_attention
   return post_attention(q,s,bits=self.layer_kbits[i],bits_v=self.layer_vbits[i],v_channel=self.v_channel,k_group_size=self.k_group_size,v_group_size=self.v_group_size)
  return attention(q,s,refinement=r,bits=self.layer_kbits[i],bits_v=self.layer_vbits[i],kr_bits=self.kr_bits or 2,refine_k=fk,refine_v=fv,
    v_channel=self.v_channel,k_gain=getattr(s,'k_gain',None),visual_logit_gain=type(self).settings.get('visual_logit_gain',1.),text_logit_gain=type(self).settings.get('text_logit_gain',1.),
    k_group_size=self.k_group_size,v_group_size=self.v_group_size,gain_centered=not (type(self).settings.get('k_token_gain') or '').startswith('origin'))
 def _visual_tensors(self,s):
  return super()._visual_tensors(s)+([s.k_gain] if hasattr(s,'k_gain') else [])
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='mixed native axis widths and sparse static layer residuals',base_bits_k=self.bits,base_bits_v=self.vbits,k_residual_bits=self.kr_bits,
    k_clip=type(self).settings.get('k_clip',1.),v_refined_layers=[i for i,(k,v) in enumerate(self.refinement_flags) if v],
    layer_bits_k=self.layer_kbits,layer_bits_v=self.layer_vbits,k_encoder=type(self).settings.get('k_encoder','rtn'),v_encoder=type(self).settings.get('v_encoder','rtn'),
    layer_encoders_k=type(self).settings.get('layer_encoders_k'),layer_encoders_v=type(self).settings.get('layer_encoders_v'),
    v_layout='channel' if self.v_channel else 'token',k_token_gain=type(self).settings.get('k_token_gain'),gain_shared=type(self).settings.get('gain_shared',False),
    k_group_size=self.k_group_size,v_group_size=self.v_group_size,k_fold_gain=type(self).settings.get('k_fold_gain'),k_fold_gain_factor=type(self).settings.get('k_fold_gain_factor',1.),
    encoder_calibration_parameters_bytes=self.encoder_parameter_bytes,encoder_calibration_sha256=self.calibration_hash,
    calibration_storage_scope='shared encoder-only factors folded into existing FP16 metadata; no retained decoder factor tensor',
    postscale_reader=type(self).settings.get('postscale_reader',False))
  return r
