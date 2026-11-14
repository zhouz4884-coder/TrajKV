"""Actual packed center/rotation/direction cache, no persistent FP16 keys."""
import json,sys,torch
from pathlib import Path
from masked_bank import A2Bank,unique_bytes
from quantizers import pack1,binary_centroids,mean_abs_levels,affine_rtn
from fused_direction import attention
ROOT=Path(__file__).resolve().parents[2]
class DirectionBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  sys.path.insert(0,str(ROOT/'third_party/reference_ports'));from kvarn_reference import _hadamard
  cfg=type(self).settings;self.hadamard=_hadamard(128,str(self.states[0].visual_k.device));self.book=None;self.query_scales=None
  self.vbits=cfg.get('bits_v',2);self.v_channel=cfg.get('v_layout','token')=='channel';nv=self.states[0].visual_pos.numel()
  self.vgroup=nv if cfg.get('v_group_size')=='all' else 128
  if cfg.get('direction_book_path'):
   data=json.loads(Path(cfg['direction_book_path']).read_text());self.book=torch.tensor(data['centers'],device=self.hadamard.device,dtype=torch.float16).contiguous()
  if cfg.get('query_scale_path'):
   data=json.loads(Path(cfg['query_scale_path']).read_text());self.query_scales=torch.tensor(data['rms_scales'],device=self.hadamard.device,dtype=torch.float32).pow(cfg.get('query_scale_power',1.)).half().contiguous()
  for i,s in enumerate(self.states):
   x=s.visual_k.float();s.direction_mean=x.mean(-2,keepdim=True).half().contiguous();residue=x-s.direction_mean.float()
   scale=1. if self.query_scales is None else self.query_scales[i][None,:,None,:].float();z=(residue*scale)@self.hadamard;norm=z.norm(dim=-1,keepdim=True);mode=cfg.get('gain','norm')
   if self.book is None:
    s.direction_code=pack1(z>=0)
    gain=norm/128**.5 if mode=='norm' else norm.square()/z.abs().sum(-1,keepdim=True).clamp_min(1e-20) if mode=='projection' else z.abs().mean(-1,keepdim=True)
   else:
    initial=(norm/128**.5).clamp_min(1e-20);parts=(z/initial).reshape(-1,self.book.shape[-1]);book=self.book.float();indices=[]
    for start in range(0,len(parts),65536):
     part=parts[start:start+65536];dist=part.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*(part@book.T);indices.append(dist.argmin(-1).to(torch.uint8))
    s.direction_code=torch.cat(indices).reshape(*z.shape[:-1],128//book.shape[-1]).contiguous();template=book[s.direction_code.long()].reshape(z.shape);dot=(z*template).sum(-1,keepdim=True)
    if mode=='norm':gain=norm/template.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    elif mode=='projection':gain=torch.where(dot>1e-12,norm.square()/dot.clamp_min(1e-12),norm/template.norm(dim=-1,keepdim=True).clamp_min(1e-20))
    else:gain=dot/template.square().sum(-1,keepdim=True).clamp_min(1e-20)
   s.direction_gain=gain.half().contiguous()
   if not bool(torch.isfinite(s.direction_gain).all()):raise RuntimeError('Invalid direction gain')
   del s.visual_k
   if self.vbits!=16:
    data=s.visual_v.transpose(2,3).contiguous() if self.v_channel else s.visual_v;encoder=cfg.get('v_encoder','rtn')
    rec=(binary_centroids(data,group_size=self.vgroup) if encoder=='lloyd' else mean_abs_levels(data,group_size=self.vgroup) if encoder=='mad' else
         self.codec.triton_quantize_and_pack_along_last_dim(data,128,self.vbits) if self.vgroup==128 else affine_rtn(data,self.vbits,self.vgroup))
    for name,t in zip(('code','scale','min'),rec):setattr(s,'v_'+name,t)
    del s.visual_v
  self.method=method
 def _visual_tensors(self,s):
  if hasattr(s,'visual_k'):return [s.visual_k,s.visual_v]
  t=[s.direction_code,s.direction_gain,s.direction_mean,self.hadamard]
  if self.book is not None:t.append(self.book)
  if self.query_scales is not None:t.append(self.query_scales)
  return t+([s.visual_v] if self.vbits==16 else [s.v_code,s.v_scale,s.v_min])
 def read(self,q,i):
  s=self.states[i];e=self.events[(self.turn,self.phase,self.pass_kind,i)];e['calls']+=1;e['query_tokens']+=q.shape[-2]
  payload=[s.direction_code,s.direction_gain,s.direction_mean]+([s.visual_v] if self.vbits==16 else [s.v_code,s.v_scale,s.v_min])
  if self.book is not None:payload.append(self.book)
  e['base_visual_bytes']+=unique_bytes(payload);e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length)
  e['text_bytes']+=2*s.text_length*s.text_k.shape[1]*s.text_k.shape[-1]*s.text_k.element_size()
  overhead=unique_bytes([self.hadamard])+q.numel()*20+q.shape[1]*q.shape[-1]*6+q.shape[1]*q.shape[-2]*4
  if self.query_scales is not None:overhead+=self.query_scales[i].numel()*self.query_scales.element_size()+q.shape[1]*q.shape[-1]*6+q.numel()*4
  e['selection_bytes']+=overhead;self.extra_read_bytes+=overhead
  return attention(q,s,self.hadamard,self.book,None if self.query_scales is None else self.query_scales[i][None,:,None,:],self.vbits,self.v_channel,self.vgroup)
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='packed residual direction + FP32 visual mean restoration, original FP16 text',
   visual_fp16_materialization=False,key_payload_bits=1 if self.book is None else 8/self.book.shape[-1],value_payload_bits=self.vbits,
   direction_book_dimension=None if self.book is None else self.book.shape[-1],gain_rule=type(self).settings.get('gain','norm'),
   rotation_payload_scope='shared FP32 Hadamard, optional FP16 query metric, mean and row gain in actual owner; predeclared H/query/mean/metric temporary payload charge, not physical DRAM',
   value_axis_original_fp16=self.vbits==16)
  return r
