"""Global product-code keys with encoder-only functional metadata fitting."""
import hashlib,json
from pathlib import Path
import torch
from masked_bank import A2Bank
from learned_direction_bank import LearnedDirectionBank
from quantizers import binary_centroids
from fast_wht import wht
from affine_rate_bank import fold

class MetadataDirectionBank(LearnedDirectionBank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  A2Bank._initialize(self,model,cache,visual,'full',legacy_scripts,kivi_source,False)
  cfg=type(self).settings;device=self.states[0].visual_k.device;nv=visual.numel()
  self.book=torch.tensor(json.loads(Path(cfg['direction_book_path']).read_text())['centers'],device=device,dtype=torch.float16).contiguous()
  assert self.book.shape==(256,4)
  self.decoder_books=None;self.decoder_hash=None;self.headwise_budget_ok=False;self.encoder_parameter_bytes=0
  self.vbits=1;self.v_channel=True;self.vgroup=nv if cfg.get('v_group_size')=='all' else 128
  if cfg.get('bits_v',1)!=1 or cfg.get('v_layout','channel')!='channel':raise ValueError('Functional metadata format is channel V1')
  self.metadata_hash=None;params={};self.key_gain_mode='none'
  if cfg.get('metadata_factor_path'):
   path=Path(cfg['metadata_factor_path']);data=json.loads(path.read_text());self.metadata_hash=hashlib.sha256(path.read_bytes()).hexdigest()
   if data['encoder_book_sha256']!=hashlib.sha256(Path(cfg['direction_book_path']).read_bytes()).hexdigest():raise ValueError('Wrong encoder book for factors')
   if data['value_group_size']!=cfg.get('v_group_size',128):raise ValueError('Calibration value-group mismatch')
   self.key_gain_mode=data['key_gain_mode'];assert self.key_gain_mode in ['none','centered','whole']
   for key in ['key_head_factors','value_head_factors','value_center_offsets']:
    if data.get(key) is None:continue
    parameter=torch.tensor(data[key],device=device,dtype=torch.float32)
    assert parameter.shape==(len(self.states),self.states[0].visual_k.shape[1]) and bool(torch.isfinite(parameter).all())
    if 'factors' in key:assert bool((parameter>0).all())
    params[key]=parameter
   self.encoder_parameter_bytes=sum(t.numel()*t.element_size() for t in params.values());assert self.encoder_parameter_bytes==4*data['parameter_count']
  for i,state in enumerate(self.states):
   source=state.visual_k.float();state.direction_mean=source.mean(-2,keepdim=True).half().contiguous()
   rotated=wht(source-state.direction_mean.float());norm=rotated.norm(dim=-1,keepdim=True);initial=(norm/(128**.5)).clamp_min(1e-20)
   parts=(rotated/initial).reshape(-1,4);book=self.book.float();indices=[]
   for start in range(0,len(parts),65536):
    part=parts[start:start+65536];distance=part.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*(part@book.T)
    indices.append(distance.argmin(-1).byte())
   state.direction_code=torch.cat(indices).reshape(*rotated.shape[:-1],32).contiguous();template=book[state.direction_code.long()].reshape(rotated.shape)
   state.direction_gain=(norm/template.norm(dim=-1,keepdim=True).clamp_min(1e-20)).half().contiguous()
   if 'key_head_factors' in params:
    gain=params['key_head_factors'][i][None,:,None,None];state.direction_gain=(state.direction_gain.float()*gain).half().contiguous()
    if self.key_gain_mode=='whole':state.direction_mean=(state.direction_mean.float()*gain).half().contiguous()
   del state.visual_k
   record=binary_centroids(state.visual_v.transpose(2,3).contiguous(),group_size=self.vgroup)
   if 'value_head_factors' in params:record=fold(record,1,params['value_head_factors'][i][None,:,None,None],params['value_center_offsets'][i][None,:,None,None])
   for key,tensor in zip(['code','scale','min'],record):setattr(state,'v_'+key,tensor)
   del state.visual_v
  self.method=method
 def result_metadata(self):
  result=super().result_metadata();result.update(backend='matrix-free global product code with metadata-only functional calibration',
   metadata_calibration_sha256=self.metadata_hash,key_gain_mode=self.key_gain_mode,value_group_size=self.vgroup,
   headwise_fallback_scope='no learned decoder table; global fixed book for every image',
   calibration_parameter_scope='FP32 encoder-only factors folded into existing FP16 key row gains/means and value scale/minimum; counted separately in eligibility')
  return result
