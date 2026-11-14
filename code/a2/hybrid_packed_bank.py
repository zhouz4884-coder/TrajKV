"""Executable packed combination; only codes/metadata own visual storage."""
import sys
from pathlib import Path
from masked_bank import A2Bank,unique_bytes
from calib_bank import load_official
from fused_hybrid import attention
ROOT=Path(__file__).resolve().parents[2]
class HybridPackedBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  sys.path.insert(0,str(ROOT/'third_party/reference_ports'))
  from kvarn_reference import encode_kvarn,_hadamard
  self.native_k=type(self).settings.get('key_codec','kvarn2')=='native2'
  self.key_records=[];self.value_records=[];self.model_checks=[];self.hadamard=None if self.native_k else _hadamard(128,str(self.states[0].text_k.device))
  m=load_official(legacy_scripts);encoder=getattr(m.encode,'_torchdynamo_orig_callable',m.encode)
  for s in self.states:
   if self.native_k:self.key_records.append(self.codec.triton_quantize_and_pack_along_last_dim(s.visual_k.transpose(2,3).contiguous(),128,2))
   else:self.key_records.append(encode_kvarn(s.visual_k,axis='k',bits=2))
   x=s.visual_v.transpose(1,2).contiguous();codes,scale,zero=m.quant(x,1,per_channel=True,scale=type(self).settings.get('clip_v',.3),post_scale=True)
   packed=encoder(codes.transpose(1,2).long(),bits=1,datum_per_elem=8)
   self.value_records.append((packed,scale.transpose(1,2).contiguous(),zero.transpose(1,2).contiguous()))
   del s.visual_k,s.visual_v
  self.method=method
 def _visual_tensors(self,s):
  if hasattr(s,'visual_k'):return [s.visual_k,s.visual_v]
  i=next(j for j,t in enumerate(self.states) if t is s)
  keys=list(self.key_records[i]) if self.native_k else list(self.key_records[i].storage_tensors().values())
  return keys+list(self.value_records[i])
 def read(self,q,i):
  s=self.states[i];e=self.events[(self.turn,self.phase,self.pass_kind,i)];e['calls']+=1;e['query_tokens']+=q.shape[-2]
  e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length)
  e['base_visual_bytes']+=unique_bytes(self._visual_tensors(s))
  e['text_bytes']+=sum(t[:,:,:s.text_length].numel()*t.element_size() for t in (s.text_k,s.text_v))
  # Separate optional-selection field carries the explicit rotation input/output
  # payload charge. It is included in the common extra-read ledger.
  cost=0 if self.native_k else unique_bytes([self.hadamard])+q.numel()*(2+4+4+2)
  e['selection_bytes']+=cost;self.extra_read_bytes+=cost
  out=attention(q,s,self.key_records[i],self.value_records[i],self.hadamard,native_k=self.native_k)
  if type(self).settings.get('diagnostic_compare',False) and not any(r['layer']==i for r in self.model_checks):
   import torch
   from types import SimpleNamespace
   from fused_progressive import decode_last_generic
   from fused_masked import attention as reference_attention
   from kvarn_reference import decode_kvarn
   kr=self.key_records[i]
   keys=decode_last_generic(*kr,128,2).transpose(2,3).contiguous() if self.native_k else decode_kvarn(kr)
   packed,scale,zero=self.value_records[i];shift=torch.arange(7,-1,-1,device=q.device)
   codes=((packed.long()[...,None]>>shift)&1).reshape(*keys.shape)
   values=((codes.half()-zero).half()*scale).half().contiguous()
   view=SimpleNamespace(visual_k=keys,visual_v=values,visual_pos=s.visual_pos,text_k=s.text_k,text_v=s.text_v,text_pos=s.text_pos,text_length=s.text_length,length=s.length)
   expected=reference_attention(q,view);delta=out.float()-expected.float()
   arrays=list(kr) if self.native_k else list(kr.storage_tensors().values())
   row={'layer':i,'nq':q.shape[-2],'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/expected.float().norm().clamp_min(1e-12)),
    'key_layouts':[{'shape':list(t.shape),'stride':list(t.stride()),'contiguous':t.is_contiguous()} for t in arrays],
    'value_layouts':[{'shape':list(t.shape),'stride':list(t.stride()),'contiguous':t.is_contiguous()} for t in self.value_records[i]]}
   self.model_checks.append(row);print(__import__('json').dumps({'model_kernel_check':row}),flush=True)
  return out
 def owned(self):
  r=super().owned();n=unique_bytes([self.hadamard]) if getattr(self,'hadamard',None) is not None else 0
  r['hadamard_constant']=n;r['total_unique']+=n
  r['packed_visual_ratio_including_aux']=(r['visual_base']+n)/self.visual_full_bytes;r['runtime']='fused_packed_hybrid_rotated_queries'
  return r
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='packed KVarN2 keys and clipped per-channel V1, all visual support',
   clip_v=type(self).settings.get('clip_v',.3),visual_fp16_materialization=False,
   key_codec='native2' if self.native_k else 'kvarn2',rotation_scope='none' if self.native_k else 'FP32 q@Hadamard then FP16; FP16 rotated key dequant; original q for text',
   rotation_logical_payload_bytes=r['selection_logical_read_bytes'],selection_logical_read_bytes=0,
   model_kernel_checks=self.model_checks,diagnostic_materialized_comparison=bool(self.model_checks))
  return r
