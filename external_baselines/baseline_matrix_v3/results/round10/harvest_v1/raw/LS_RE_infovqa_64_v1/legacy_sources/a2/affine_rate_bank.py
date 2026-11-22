"""Next-phase metadata-only affine adaptation of the frozen packed format.

The code assignments are unchanged. Shared encoder parameters are folded into
existing FP16 scale/minimum tensors and are not retained by the decoder.
"""
import hashlib,json
from pathlib import Path
import torch
from rate_bank import RateBank

def fold(record,bits,gain,offset=None):
 code,scale,minimum=record
 middle=minimum.float()+.5*((1<<bits)-1)*scale.float()
 new_scale=(scale.float()*gain).half().contiguous()
 shift=0. if offset is None else offset
 new_minimum=(middle+(shift-.5*((1<<bits)-1))*new_scale.float()).half().contiguous()
 return code,new_scale,new_minimum

class AffineRateBank(RateBank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  cfg=type(self).settings;path=Path(cfg['affine_factor_path']);fit=json.loads(path.read_text())
  if fit.get('base_rms',True):raise ValueError('Identity-initialized affine runtime requires base_rms=False')
  if (cfg.get('bits_k'),cfg.get('bits_v'),cfg.get('v_layout'),cfg.get('v_encoder'))!=(2,1,'channel','lloyd'):raise ValueError('Current affine calibration is K2/channel-Lloyd V1 only')
  plain=dict(cfg)
  for key in ('k_fold_gain','k_fold_factor_path','v_fold_factor_path','k_token_gain'):
   if plain.get(key):raise ValueError('Do not combine old fold/gain paths with affine calibration')
  # Parent creates precisely the original RTN/Lloyd codes and metadata.
  super()._initialize(model,cache,visual,method,legacy_scripts,kivi_source,False)
  device=self.states[0].k_scale.device;shape=(len(self.states),self.states[0].k_scale.shape[1]);params={}
  for key in ('head_factors','value_head_factors','key_center_offsets','value_center_offsets'):
   values=fit.get(key)
   if values is None:continue
   tensor=torch.tensor(values,device=device,dtype=torch.float32)
   if tensor.shape!=shape or not bool(torch.isfinite(tensor).all()):raise ValueError('Invalid affine calibration parameter: '+key)
   if 'factors' in key and not bool((tensor>0).all()):raise ValueError('Gains must be positive')
   params[key]=tensor
  if not {'head_factors','value_head_factors'}<=params.keys():raise ValueError('Both K and V gain arrays are required')
  for i,state in enumerate(self.states):
   for axis,bits,gain_key,offset_key in [('k',2,'head_factors','key_center_offsets'),('v',1,'value_head_factors','value_center_offsets')]:
    gain=params[gain_key][i][None,:,None,None];offset=params.get(offset_key)
    if offset is not None:offset=offset[i][None,:,None,None]
    record=fold(tuple(getattr(state,axis+'_'+field) for field in ('code','scale','min')),bits,gain,offset)
    for field,value in zip(('code','scale','min'),record):setattr(state,axis+'_'+field,value)
  self.encoder_parameter_bytes=sum(t.numel()*t.element_size() for t in params.values())
  self.calibration_hash=hashlib.sha256(path.read_bytes()).hexdigest();self.affine_fit_summary={k:fit.get(k) for k in ('calibration_mode','target_tokens_max','train_center','parameter_count','cases','images','labels_used','base_rms')}
  if fit['parameter_count']*4!=self.encoder_parameter_bytes:raise ValueError('Encoder-parameter accounting mismatch')
 def result_metadata(self):
  result=super().result_metadata();result.update(backend='existing packed RTN/Lloyd format with metadata-only affine calibration',
   affine_calibration=self.affine_fit_summary,affine_decoder_parameter_tensors=0,
   affine_center_scope='per layer/KV head offset in final metadata scale units; assignments unchanged')
  return result
