"""KVarN2 keys plus one-bit values, numerical combination prototype.

Materialized visual and packed references are both counted. This tests the
combination before building a packed hybrid reader; no serving-memory claim.
"""
import sys
from pathlib import Path
from masked_bank import A2Bank,unique_bytes
from fused_progressive import decode_last_generic
from calib_bank import load_official
ROOT=Path(__file__).resolve().parents[2]
class HybridBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  sys.path.insert(0,str(ROOT/'third_party/reference_ports'))
  from kvarn_reference import encode_kvarn,decode_kvarn
  self.hybrid_packed=[];self.key_records=[];kind=type(self).settings.get('value_codec','native1')
  m=load_official(legacy_scripts) if kind=='calib1' else None
  for s in self.states:
   record=encode_kvarn(s.visual_k,axis='k',bits=2);s.visual_k=decode_kvarn(record)
   self.key_records.append(record);arrays=list(record.storage_tensors().values())
   if kind=='native1':
    enc=self.codec.triton_quantize_and_pack_along_last_dim(s.visual_v,128,1)
    s.visual_v=decode_last_generic(*enc,128,1);arrays+=list(enc)
   else:
    x=s.visual_v.transpose(1,2).contiguous();clip=type(self).settings.get('clip_v',.3)
    code,scale,zero=m.quant(x,1,per_channel=True,scale=clip,post_scale=True)
    encoder=getattr(m.encode,'_torchdynamo_orig_callable',m.encode)
    packed=encoder(code.transpose(1,2).long(),bits=1,datum_per_elem=8)
    s.visual_v=(scale*(code-zero)).transpose(1,2).contiguous();arrays += [packed,scale,zero]
   self.hybrid_packed.append(arrays)
  self.method=method
 def owned(self):
  r=super().owned();n=unique_bytes([t for group in getattr(self,'hybrid_packed',[]) for t in group])
  r['hybrid_packed_reference']=n;r['total_unique']+=n;r['encoded_reference_ratio']=n/self.visual_full_bytes
  r['runtime']='materialized_hybrid_quality_reference';return r
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='KVarN2 K + one-bit V numerical combination, materialized',
   value_codec=type(self).settings.get('value_codec','native1'),encoded_memory_is_not_runtime_memory=True)
  return r
