"""Packed product-code keys, matrix-free rotation and optional learned decoder."""
import hashlib,json
from pathlib import Path
import torch
from masked_bank import A2Bank,unique_bytes
from quantizers import binary_centroids,affine_rtn
from affine_rate_bank import fold
from fast_wht import wht
from fused_learned_direction import attention

class LearnedDirectionBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  cfg=type(self).settings;device=self.states[0].visual_k.device;nv=visual.numel()
  self.book=torch.tensor(json.loads(Path(cfg['direction_book_path']).read_text())['centers'],device=device,dtype=torch.float16).contiguous()
  if self.book.shape!=(256,4):raise ValueError('Current product code uses256 vectors of4 dimensions')
  self.decoder_books=None;self.encoder_parameter_bytes=0;self.decoder_hash=None;value_gain=value_offset=None
  layers=len(self.states);heads=self.states[0].visual_k.shape[1];elements=layers*heads*nv*128
  proposed=(elements*(2/8+1/8+4/128)+layers*heads*nv*2+layers*heads*128*2+2048+layers*heads*256*4*2+8*nv)
  self.headwise_budget_ok=16*(proposed+layers*heads*2*4)/self.visual_full_bytes<=cfg.get('book_budget_bits',1.8)
  if cfg.get('decoder_book_path'):
   path=Path(cfg['decoder_book_path']);data=json.loads(path.read_text());self.decoder_hash=hashlib.sha256(path.read_bytes()).hexdigest()
   if cfg.get('apply_key_decoder',True) and (self.headwise_budget_ok or not cfg.get('headwise_if_budget',True)):self.decoder_books=torch.tensor(data['head_decoder_centers'],device=device,dtype=torch.float16).contiguous()
   expected=(len(self.states),self.states[0].visual_k.shape[1],256,4)
   if self.decoder_books is not None and self.decoder_books.shape!=expected:raise ValueError('Decoder-book architecture mismatch')
   if cfg.get('apply_value_metadata',True) and data.get('value_head_factors') is not None:
    value_gain=torch.tensor(data['value_head_factors'],device=device,dtype=torch.float32)
    value_offset=torch.tensor(data['value_center_offsets'],device=device,dtype=torch.float32)
    self.encoder_parameter_bytes=(value_gain.numel()+value_offset.numel())*4
  elif cfg.get('untrained_headwise_control',False) and (self.headwise_budget_ok or not cfg.get('headwise_if_budget',True)):
   self.decoder_books=self.book[None,None].repeat(len(self.states),self.states[0].visual_k.shape[1],1,1).contiguous()
  if self.decoder_books is not None and not bool(torch.isfinite(self.decoder_books).all()):raise RuntimeError('Invalid decoder table')
  self.vbits=cfg.get('bits_v',1);self.v_channel=cfg.get('v_layout','channel')=='channel';self.vgroup=nv if cfg.get('v_group_size')=='all' else 128
  if self.vbits!=1 or not self.v_channel or self.vgroup!=128:raise ValueError('Current calibrated format is channel V1,g128')
  for i,state in enumerate(self.states):
   original=state.visual_k.float();state.direction_mean=original.mean(-2,keepdim=True).half().contiguous()
   rotated=wht(original-state.direction_mean.float());norm=rotated.norm(dim=-1,keepdim=True)
   initial=(norm/(128**.5)).clamp_min(1e-20);parts=(rotated/initial).reshape(-1,4);book=self.book.float();indices=[]
   for start in range(0,len(parts),65536):
    part=parts[start:start+65536];dist=part.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*(part@book.T)
    indices.append(dist.argmin(-1).to(torch.uint8))
   state.direction_code=torch.cat(indices).reshape(*rotated.shape[:-1],32).contiguous()
   template=book[state.direction_code.long()].reshape(rotated.shape)
   state.direction_gain=(norm/template.norm(dim=-1,keepdim=True).clamp_min(1e-20)).half().contiguous()
   del state.visual_k
   value=state.visual_v.transpose(2,3).contiguous() if self.v_channel else state.visual_v
   rec=binary_centroids(value,group_size=self.vgroup) if self.vbits==1 else affine_rtn(value,self.vbits,self.vgroup)
   if value_gain is not None:
    if self.vbits!=1 or not self.v_channel:raise ValueError('Joint value fit is channel V1')
    rec=fold(rec,1,value_gain[i][None,:,None,None],value_offset[i][None,:,None,None])
   for name,tensor in zip(('code','scale','min'),rec):setattr(state,'v_'+name,tensor)
   del state.visual_v
  self.method=method
 def _visual_tensors(self,state):
  if hasattr(state,'visual_k'):return [state.visual_k,state.visual_v]
  arrays=[state.direction_code,state.direction_gain,state.direction_mean,state.v_code,state.v_scale,state.v_min,self.book]
  if self.decoder_books is not None:arrays.append(self.decoder_books)
  return arrays
 def read(self,q,i):
  state=self.states[i];event=self.events[(self.turn,self.phase,self.pass_kind,i)]
  event['calls']+=1;event['query_tokens']+=q.shape[-2]
  payload=[state.direction_code,state.direction_gain,state.direction_mean,state.v_code,state.v_scale,state.v_min]
  decoder=self.book if self.decoder_books is None else self.decoder_books[i]
  event['base_visual_bytes']+=unique_bytes(payload)+decoder.numel()*decoder.element_size()
  event['position_bytes']+=8*(state.visual_pos.numel()+state.text_length)
  event['text_bytes']+=2*state.text_length*state.text_k.shape[1]*state.text_k.shape[-1]*state.text_k.element_size()
  cost=q.numel()*12+q.shape[1]*q.shape[-1]*6+q.shape[1]*q.shape[-2]*4
  event['selection_bytes']+=cost;self.extra_read_bytes+=cost
  return attention(q,state,decoder,self.vbits,self.v_channel,self.vgroup)
 def result_metadata(self):
  cfg=type(self).settings
  result=super().result_metadata();result.update(backend='matrix-freeFWHT packed product code with optional per-layer/KV-head decoder books',
   visual_fp16_materialization=False,rotation_matrix_retained=False,direction_book_dimension=4,key_payload_bits=2,value_payload_bits=self.vbits,
   decoder_headwise=self.decoder_books is not None,decoder_calibration_sha256=self.decoder_hash,
   headwise_budget_ok=self.headwise_budget_ok,headwise_fallback_scope='select compact global book on small images when the full headwise table would exceed1.8bits; inferred from visual length, no extra flag',
   encoder_calibration_parameters_bytes=self.encoder_parameter_bytes,
   decoder_book_bytes=0 if self.decoder_books is None else self.decoder_books.numel()*self.decoder_books.element_size(),
   encoder_book_bytes=self.book.numel()*self.book.element_size(),
   apply_key_decoder_setting=cfg.get('apply_key_decoder',True),apply_value_metadata_setting=cfg.get('apply_value_metadata',True),
   encoding_assignment='fixed data-independent Gaussian encoder book; original row-norm gains; learned book changes reconstruction only',
   constant_scope='both encoder and decoder books charged once per cache; no stored Hadamard matrix; query FWHT scratch is charged separately')
  return result
