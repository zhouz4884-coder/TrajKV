"""Budgeted question-independent FP16 prefix states beside packed visual KV."""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from masked_bank import A2Bank,unique_bytes
from affine_rate_bank import AffineRateBank
from direction_bank import DirectionBank

def anchor_count(model,nv,cfg):
 text=model.config.get_text_config();layers=text.num_hidden_layers;heads=text.num_key_value_heads;d=text.hidden_size//text.num_attention_heads
 if cfg.get('anchor_mode','none')=='none':return 0
 if cfg['method']=='anchor_full':
  if not cfg.get('anchor_reference_method'):return 0
  cfg=dict(cfg,method=cfg['anchor_reference_method'])
 e=layers*heads*nv*d;encoder=0
 if cfg['method']=='anchor_affine':
  fit=json.loads(Path(cfg['affine_factor_path']).read_text());encoder=4*fit['parameter_count']
  base=e*((2+1)/8+8/128)+8*nv
 elif cfg['method']=='anchor_direction':
  book=json.loads(Path(cfg['direction_book_path']).read_text());dim=book.get('dimension',4)
  base=e*(8/dim)/8+layers*heads*nv*2+layers*heads*d*2+65536+256*dim*2+e*(1/8+4/128)+8*nv
 else:raise ValueError('Unsupported anchor base')
 allowance=cfg.get('anchor_budget_bits',1.8)*(2*e)/8-base-encoder
 per_token=4*layers*heads*d+8
 return max(0,min(cfg.get('anchor_max_tokens',16),int(allowance//per_token)))

class AnchorMixin:
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  parent_method={'anchor_affine':'affine_rate','anchor_direction':'direction','anchor_full':'full'}[method]
  super()._initialize(model,cache,visual,parent_method,legacy_scripts,kivi_source,False);self.method=method
  cfg=type(self).settings;self.anchor_requested=cfg['_anchor_count'];self.anchor_mode=cfg.get('anchor_mode','none')
  self.anchor_start=int(visual[-1])+1;self.anchor_pos=torch.arange(self.anchor_start,self.anchor_start+self.anchor_requested,device=visual.device)
  self.anchors=[];self.anchor_capture_ms=0.;self.anchor_copy_bytes=0
  if self.anchor_mode=='native' and self.anchor_requested:self.capture_anchors()
 def capture_anchors(self):
  if self.anchors:raise RuntimeError('Anchor states already captured')
  torch.cuda.synchronize();started=time.perf_counter()
  for state in self.states:
   pos=state.text_pos[:state.text_length];chosen=(pos>=self.anchor_start)&(pos<self.anchor_start+self.anchor_requested)
   idx=chosen.nonzero().flatten();assert idx.numel()==self.anchor_requested
   ak=state.text_k.index_select(2,idx).contiguous();av=state.text_v.index_select(2,idx).contiguous();self.anchors.append((ak,av))
   keep=(~chosen).nonzero().flatten();n=keep.numel()
   key=state.text_k.index_select(2,keep);value=state.text_v.index_select(2,keep);positions=pos.index_select(0,keep)
   state.text_k[:,:,:n]=key;state.text_v[:,:,:n]=value;state.text_pos[:n]=positions;state.text_length=n
  torch.cuda.synchronize();self.anchor_capture_ms+=1000*(time.perf_counter()-started)
 def read(self,q,i):
  if q.shape[0]!=1:raise ValueError('Anchor prototype uses batch1')
  # Batch0 is the only valid batch, so its stride can be canonicalized without
  # changing any address. Avoid specializing prefill kernels on that dead stride.
  q=q.as_strided(q.shape,(0,*q.stride()[1:]))
  if not self.anchors:return super().read(q,i)
  state=self.states[i];ak,av=self.anchors[i];n=state.text_length
  saved=(state.text_k,state.text_v,state.text_pos,state.text_length)
  # This prototype uses the unchanged packed reader. Extra concatenation work
  # is timed and explicitly charged; no new fused-kernel speed claim.
  active=n+self.anchor_requested;capacity=state.text_k.shape[2]+type(self).settings.get('anchor_max_tokens',16)
  # Keep the stride fixed across decode lengths. The original variable-size
  # cat changed a Triton constexpr stride and compiled a new kernel per token.
  tk=torch.empty((1,state.text_k.shape[1],capacity,state.text_k.shape[-1]),device=q.device,dtype=state.text_k.dtype)
  tv=torch.empty_like(tk);tp=torch.empty(capacity,device=q.device,dtype=state.text_pos.dtype)
  tk[:,:,:n].copy_(state.text_k[:,:,:n]);tk[:,:,n:active].copy_(ak)
  tv[:,:,:n].copy_(state.text_v[:,:,:n]);tv[:,:,n:active].copy_(av)
  tp[:n].copy_(state.text_pos[:n]);tp[n:active].copy_(self.anchor_pos)
  copy_bytes=2*(active*state.text_k.shape[1]*state.text_k.shape[-1]*state.text_k.element_size()*2+active*tp.element_size())
  state.text_k,state.text_v,state.text_pos,state.text_length=tk,tv,tp,n+self.anchor_requested
  try:output=super().read(q,i)
  finally:state.text_k,state.text_v,state.text_pos,state.text_length=saved
  event=self.events[(self.turn,self.phase,self.pass_kind,i)];event['selection_bytes']+=copy_bytes
  self.extra_read_bytes+=copy_bytes;self.anchor_copy_bytes+=copy_bytes
  return output
 def owned(self):
  result=super().owned();arrays=[t for pair in getattr(self,'anchors',[]) for t in pair]
  if arrays:arrays.append(self.anchor_pos)
  added=unique_bytes(arrays);result['anchor_visual_auxiliary']=added;result['total_unique']+=added
  if self.method in ('anchor_affine','anchor_direction') and hasattr(self,'anchor_requested'):
   fields=('visual_base','visual_residual','selector_metadata','hadamard_constant','visual_position','anchor_visual_auxiliary')
   conservative=sum(result.get(k,0) for k in fields)+getattr(self,'encoder_parameter_bytes',0)
   bits=16*conservative/self.visual_full_bytes;result['anchor_conservative_visual_bits']=bits
   if bits>type(self).settings.get('anchor_budget_bits',1.8)+1e-9:raise RuntimeError('Actual compact anchor backing exceeds declared visual budget')
  return result
 def result_metadata(self):
  result=super().result_metadata();result.update(anchor_tokens=self.anchor_requested,anchor_mode=self.anchor_mode,
   anchor_capture_wall_ms=self.anchor_capture_ms,anchor_concat_logical_copy_bytes=self.anchor_copy_bytes,
   anchor_scope='fixed text prefix independent of actual question; native or compressed-image state formation; dedicated compact FP16 K/V and original positions',
   anchor_storage_scope='all dedicated anchor K/V/position backing charged as visual auxiliary, in addition to common text capacity',
   anchor_runtime='unchanged packed visual reader with charged temporary text concatenation; entire model on GPU')
  return result

class AnchoredAffineBank(AnchorMixin,AffineRateBank):pass
class AnchoredDirectionBank(AnchorMixin,DirectionBank):pass
class AnchoredFullBank(AnchorMixin,A2Bank):pass
