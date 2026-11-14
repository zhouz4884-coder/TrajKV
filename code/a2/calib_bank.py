"""CalibQuant visual quantization + decode-score calibration numerical port.

Official quant/encode functions are loaded from the pinned prior checkout.
This is the calibquant forward, not the separate QJL baseline in that file.
Modern HF prefix positions replace the original hard-coded span. Materialized
KV and packed records are both counted; this is a quality reference, not serving.
"""
import hashlib,importlib.util,functools
from pathlib import Path
import torch
from masked_bank import A2Bank,unique_bytes

CALIB_SHA='b4cca4467a88ba40d784bcc50309fb510a935859fc2cf297d944d1e743f2c194'
@functools.lru_cache(maxsize=4)
def load_official(legacy):
    p=Path(legacy).parent/'third_party/CalibQuant/calibquant.py'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=CALIB_SHA:raise RuntimeError('CalibQuant source changed')
    spec=importlib.util.spec_from_file_location('_a2_official_calib',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

class CalibBank(A2Bank):
    def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
        super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,keep_reference=False)
        self.method=method;self.calib_records=[];self.calib_checks=[]
        m=load_official(legacy_scripts);encoder=getattr(m.encode,'_torchdynamo_orig_callable',m.encode)
        for i,s in enumerate(self.states):
            rec=[]
            for axis,factor in (('k',type(self).settings.get('clip_k',.5)),('v',type(self).settings.get('clip_v',.3))):
                x=getattr(s,'visual_'+axis).transpose(1,2).contiguous()
                axis_bits=type(self).settings.get('bits_'+axis,1)
                q,scale,zero=m.quant(x,axis_bits,per_channel=True,scale=factor,post_scale=True)
                deq,_,_=m.quant(x,axis_bits,per_channel=True,scale=factor,post_scale=False)
                packed=encoder(q.transpose(1,2).long(),bits=axis_bits,datum_per_elem=8//axis_bits)
                # Independent MSB-first unpack validates the upstream encoder.
                shift=torch.arange(8-axis_bits,-1,-axis_bits,device=x.device)
                restored=((packed.long()[...,None]>>shift)&((1<<axis_bits)-1)).reshape(q.transpose(1,2).shape)
                same=torch.equal(restored,q.transpose(1,2).long())
                reconstructed=scale*(restored.transpose(1,2).to(x.dtype)-zero)
                err=float((deq.float()-reconstructed.float()).abs().max())
                finite=bool(torch.isfinite(deq).all())
                if not same or err!=0 or not finite:raise RuntimeError('Calib numerical/packing mismatch')
                setattr(s,'visual_'+axis,deq.transpose(1,2).contiguous())
                rec.extend((packed,scale,zero));self.calib_checks.append({'layer':i,'axis':axis,'packed_roundtrip':same,'dequant_max_abs':err,'finite':finite})
            self.calib_records.append(rec)
    def read(self,q,i):
        s=self.states[i];groups=q.shape[1]//s.visual_k.shape[1]
        k=torch.cat((s.visual_k,s.text_k[:,:,:s.text_length]),2).repeat_interleave(groups,1)
        v=torch.cat((s.visual_v,s.text_v[:,:,:s.text_length]),2).repeat_interleave(groups,1)
        positions=torch.cat((s.visual_pos,s.text_pos[:s.text_length]));nv=s.visual_pos.numel()
        scores=(q@k.transpose(-1,-2))/q.shape[-1]**.5
        if q.shape[-2]==1:
            z=scores[...,:nv];low=z.amin((-2,-1),keepdim=True);high=z.amax((-2,-1),keepdim=True)
            scores[...,:nv]=(z-low)/(high-low+1e-8)*(high-low-3)+low
        qp=torch.arange(s.length-q.shape[-2],s.length,device=q.device)
        scores=scores.masked_fill(positions[None,None,None,:]>qp[None,None,:,None],-torch.inf)
        result=scores.float().softmax(-1).to(q.dtype)@v
        e=self.events[(self.turn,self.phase,self.pass_kind,i)];e['calls']+=1;e['query_tokens']+=q.shape[-2]
        e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length)
        e['base_visual_bytes']+=unique_bytes((s.visual_k,s.visual_v));e['text_bytes']+=2*s.text_length*s.text_k.shape[1]*s.text_k.shape[-1]*s.text_k.element_size()
        return result
    def owned(self):
        r=super().owned();encoded=unique_bytes([t for x in getattr(self,'calib_records',[]) for t in x]);r['calib_packed_reference']=encoded;r['total_unique']+=encoded
        r['encoded_reference_ratio']=encoded/self.visual_full_bytes;r['runtime']='calib_materialized_reference';return r
    def result_metadata(self):
        r=super().result_metadata();r.update(backend='CalibQuant numerical port, not QJL or official serving',calib_quant_sha256=CALIB_SHA,
          calib_checks=self.calib_checks,calib_scope='official visual quant/encode; normalize tau=-3 for one-query decode; multi-query text prefill uses quantized visual without score normalization',
          calib_adaptations=['native visual positions, not fixed LLaVA span','modern GQA prefix retained across turns','materialized quality execution'])
        return r
