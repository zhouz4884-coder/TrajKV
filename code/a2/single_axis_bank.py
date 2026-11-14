"""Diagnostic only: exactly FP16 K with V quantized, or exactly FP16 V with K quantized."""
from masked_bank import A2Bank,unique_bytes
from fused_progressive import decode_last_generic
from quantizers import binary_centroids,mean_abs_levels
class SingleAxisBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  z=type(self).settings;axis=z['axis'];bits=z['axis_bits'];layout=z['layout'];self.axis_records=[]
  for s in self.states:
   x=getattr(s,'visual_'+axis);data=x.transpose(2,3).contiguous() if layout=='channel' else x
   if z.get('clip') is not None:
    from calib_bank import load_official
    m=load_official(legacy_scripts);raw=x.transpose(1,2).contiguous()
    codes,scale,zero=m.quant(raw,bits,per_channel=True,scale=z['clip'],post_scale=True)
    encoder=getattr(m.encode,'_torchdynamo_orig_callable',m.encode)
    packed=encoder(codes.transpose(1,2).long(),bits=bits,datum_per_elem=8//bits)
    setattr(s,'visual_'+axis,(scale*(codes-zero)).transpose(1,2).contiguous());self.axis_records.append([packed,scale,zero]);continue
   encoder=z.get('encoder','rtn')
   if encoder in ('mad','sign'):rec=mean_abs_levels(data,centered=encoder=='mad')
   elif encoder=='lloyd':rec=binary_centroids(data)
   else:rec=self.codec.triton_quantize_and_pack_along_last_dim(data,128,bits)
   decoded=decode_last_generic(*rec,128,bits);decoded=decoded.transpose(2,3).contiguous() if layout=='channel' else decoded
   setattr(s,'visual_'+axis,decoded);self.axis_records.append(list(rec))
  self.method=method
 def owned(self):
  r=super().owned();n=unique_bytes([t for group in getattr(self,'axis_records',[]) for t in group]);r['axis_packed_reference']=n;r['total_unique']+=n
  r['runtime']='single_axis_materialized_diagnostic';return r
 def result_metadata(self):
  r=super().result_metadata();r.update(backend='single-axis visual numerical diagnostic',compressed_axis=type(self).settings['axis'],
   axis_layout=type(self).settings['layout'],axis_bits=type(self).settings['axis_bits'],other_visual_axis_exact_original_fp16=True)
  return r
