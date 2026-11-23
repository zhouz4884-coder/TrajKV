"""Single-owner packed visual cache with transactional text appends.

Reuses the historical, frozen P202 fused attention and official KIVI codec.
All visual tokens are exposed. No selector or answer-aware policy is imported.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bytes import unique_storage_bytes as unique_bytes

def install_legacy(path):
    path=str(Path(path).resolve())
    if path not in sys.path:sys.path.insert(0,path)

class ProgressiveBank:
    """Attach a packed, all-support backend to a native Transformers cache.

    Reference snapshots are opt-in diagnostic artifacts and explicitly counted.
    The online packed paths discard the original visual FP16 tensors.
    """
    def __init__(self,*args,**kwargs):
        self._parent=None
        try:self._initialize(*args,**kwargs)
        except Exception:
            if self._parent is not None:self._parent.close(self.cache)
            raise

    def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,
                 keep_reference=False):
        install_legacy(legacy_scripts)
        from p185_model_state import PhaseState
        from p176_kivi_codec import load_official,decode_last
        from p201_refinement import quantize
        from p202_refinement import pack
        self.cache=cache
        self._parent=PhaseState(model,cache,visual,{},'single_full')
        self.states=self._parent.states;self.cache=cache;self.model=model
        self.method=method
        self.bits=1 if method.startswith(('base1','asym1')) else (4 if method=='native4' else 2)
        self.extended_kernel=self.bits==1 or method.endswith(('_k','_v'))
        self.refine_k=not method.endswith('_v');self.refine_v=not method.endswith('_k')
        self.refinements=[];self.references=[];self.kvarn_records=[]
        self.refine_layers=set();self.reference_layers=set()
        self.hide_visual=False
        self.probe_controller=None;self.read_budget_bytes=None;self.extra_read_bytes=0
        self.probe_skipped=[]
        self.phase='initial';self.turn=1;self.pass_kind='committed'
        self.events=defaultdict(lambda:dict(calls=0,query_tokens=0,base_visual_bytes=0,
                                           residual_bytes=0,text_bytes=0,reference_bytes=0))
        self.codec=load_official(Path(kivi_source))
        self.visual_full_bytes=0
        for s in self.states:
            original=(s.visual_k,s.visual_v)
            self.visual_full_bytes+=sum(t.numel()*t.element_size() for t in original)
            self.references.append(tuple(t.clone() for t in original) if keep_reference else None)
            arrays=[]
            if method=='full':
                self.refinements.append(None);continue
            if method.startswith('kvarn'):
                port=Path(__file__).resolve().parents[2]/'third_party/reference_ports'
                sys.path.insert(0,str(port))
                from kvarn_reference import encode_kvarn,decode_kvarn
                records=[encode_kvarn(t,axis=a,bits=int(method[-1])) for a,t in zip(('k','v'),original)]
                self.kvarn_records.append(records)
                # Numerical mechanism reference: the serving attention consumes
                # materialized values. Both owners are counted, never packed-only.
                s.visual_k,s.visual_v=[decode_kvarn(r) for r in records]
                self.refinements.append(None);continue
            for axis,x in zip(('k','v'),original):
                data=x.transpose(2,3).contiguous() if axis=='k' else x
                if data.shape[-1]%128:raise ValueError('g128 fused path needs complete groups')
                code,scale,minimum=self.codec.triton_quantize_and_pack_along_last_dim(data,128,self.bits)
                for name,t in zip(('code','scale','min'),(code,scale,minimum)):setattr(s,axis+'_'+name,t)
                if method.startswith('asym'):
                    if self.bits==1:
                        from fused_progressive import decode_last_generic
                        decoded=decode_last_generic(code,scale,minimum,128,self.bits)
                    else:decoded=decode_last(self.codec,code,scale,minimum,128,self.bits)
                    error=data.float()-decoded.float()
                    if axis=='k':
                        _,encoded=quantize(error,2);arrays.extend(encoded)
                    else:
                        group=error.reshape(*error.shape[:-1],-1,128)
                        alpha=group.abs().mean(-1).half();sign=(group>=0).byte()
                        arrays.extend((pack(sign,1),alpha,alpha))
            if arrays and method.endswith('_k'):arrays[3:]=arrays[:3]
            if arrays and method.endswith('_v'):arrays[:3]=arrays[3:]
            self.refinements.append(arrays if arrays else None)
            del s.visual_k,s.visual_v
        for i,module in enumerate(self._parent.modules):
            module._p185_read=lambda q,i=i:self.read(q,i)

    def _visual_tensors(self,s):
        if hasattr(s,'k_code'):
            return [getattr(s,a+'_'+b) for a in ('k','v') for b in ('code','scale','min')]
        return [s.visual_k,s.visual_v]

    def read(self,q,i):
        if self.extended_kernel:
            from fused_progressive import attention
        else:
            from p202_fused import attention
        s=self.states[i];refinement=self.refinements[i] if i in self.refine_layers else None
        controller=self.probe_controller
        if controller is not None:
            if i==0:controller.reset_event(turn=self.turn,phase=self.phase,input_start=s.length-q.shape[-2],input_count=q.shape[-2])
            if controller.should_refine(i):refinement=self.refinements[i]
        base_bytes=0 if self.hide_visual else unique_bytes(self._visual_tensors(s))
        text_bytes=sum(t[:,:,:s.text_length].numel()*t.element_size() for t in (s.text_k,s.text_v))
        selected=(torch.empty((q.shape[0],s.text_k.shape[1],0),device=q.device,dtype=torch.long)
                  if self.hide_visual else None)
        key=(self.turn,self.phase,self.pass_kind,i)
        e=self.events[key];e['calls']+=1;e['query_tokens']+=q.shape[-2]
        e['text_bytes']+=text_bytes
        if i in self.reference_layers:
            if self.references[i] is None:raise ValueError('No diagnostic visual reference retained')
            from types import SimpleNamespace
            view=SimpleNamespace(visual_k=self.references[i][0],visual_v=self.references[i][1],
                 visual_pos=s.visual_pos,text_k=s.text_k,text_v=s.text_v,text_pos=s.text_pos,
                 text_length=s.text_length,length=s.length)
            e['reference_bytes']+=0 if self.hide_visual else unique_bytes(self.references[i])
            return attention(q,view,selected=selected)
        e['base_visual_bytes']+=0 if self.hide_visual else unique_bytes(self._visual_tensors(s))
        active=[]
        if refinement:
            if self.refine_k:active.extend(refinement[:3])
            if self.refine_v:active.extend(refinement[3:])
        residual_bytes=0 if self.hide_visual else unique_bytes(active)
        if self.read_budget_bytes is not None and self.extra_read_bytes+residual_bytes>self.read_budget_bytes:
            refinement=None;residual_bytes=0
        e['residual_bytes']+=residual_bytes
        self.extra_read_bytes+=residual_bytes+(base_bytes+text_bytes if self.pass_kind in ('replay','repair','probe') else 0)
        flags=dict(refine_k=self.refine_k,refine_v=self.refine_v) if self.extended_kernel else {}
        out=attention(q,s,selected=selected,refinement=refinement,bits=self.bits,**flags)
        if controller is not None and i==controller.layer and not self.hide_visual and self.refinements[i]:
            full_refinement=self.refinements[i]
            probe_active=[]
            if self.refine_k:probe_active.extend(full_refinement[:3])
            if self.refine_v:probe_active.extend(full_refinement[3:])
            probe_residual=unique_bytes(probe_active);extra=base_bytes+text_bytes+probe_residual
            if self.read_budget_bytes is not None and self.extra_read_bytes+extra>self.read_budget_bytes:
                self.probe_skipped.append(dict(turn=self.turn,phase=self.phase,layer=i,reason='read_budget_exhausted'))
                return out
            enhanced=attention(q,s,selected=selected,refinement=full_refinement,bits=self.bits,**flags)
            probe=self.events[(self.turn,self.phase,'probe',i)]
            probe['calls']+=1;probe['query_tokens']+=q.shape[-2]
            probe['base_visual_bytes']+=base_bytes;probe['residual_bytes']+=probe_residual;probe['text_bytes']+=text_bytes
            self.extra_read_bytes+=extra
            later=sum(unique_bytes(r or []) for r in self.refinements[i+1:])
            decision=controller.at_probe(out,enhanced,dict(layer=i,same_query=True,
                 probe_read_bytes=2*(base_bytes+text_bytes)+probe_residual,upgrade_bytes=later,
                 cost_scope='logical tensor payload, not hardware HBM traffic',feature_availability='online'))
            if decision['use_enhanced_probe']:out=enhanced
        return out

    def lengths(self):return [(s.length,s.text_length) for s in self.states]

    def rollback(self,lengths):
        for s,(logical,text) in zip(self.states,lengths):
            if logical>s.length or text>s.text_length:raise ValueError('Rollback cannot grow history')
            s.length=logical;s.text_length=text

    def snapshot(self,cpu=False):
        out=[]
        for s in self.states:
            values=[s.text_k[:,:,:s.text_length].clone(),s.text_v[:,:,:s.text_length].clone(),
                    s.text_pos[:s.text_length].clone()]
            if cpu:values=[t.cpu() for t in values]
            out.append(dict(length=s.length,text_length=s.text_length,values=values))
        return out

    def restore(self,snapshot):
        for s,r in zip(self.states,snapshot):
            n=r['text_length']
            if n>s.text_k.shape[2]:
                capacity=n+s.reserve
                s.text_k=s.text_k.new_empty((*s.text_k.shape[:2],capacity,s.text_k.shape[-1]))
                s.text_v=torch.empty_like(s.text_k);s.text_pos=s.text_pos.new_empty(capacity)
            for name,value in zip(('text_k','text_v','text_pos'),r['values']):
                target=getattr(s,name)
                if name=='text_pos':target[:n].copy_(value)
                else:target[:,:,:n].copy_(value)
            s.length=r['length'];s.text_length=n

    def owned(self):
        visual=[];residual=[];text=[];position=[]
        for s,r in zip(self.states,self.refinements):
            visual.extend(self._visual_tensors(s));residual.extend(r or [])
            text.extend((s.text_k,s.text_v));position.extend((s.visual_pos,s.text_pos))
        kvarn=[]
        for records in self.kvarn_records:
            for r in records:kvarn.extend(r.storage_tensors().values())
        reference=[t for pair in self.references if pair for t in pair]
        categories={'visual_base':visual,'visual_residual':residual,'text_capacity':text,
                    'position':position,'kvarn_packed_reference':kvarn,'diagnostic_fp16_reference':reference}
        result={k:unique_bytes(v) for k,v in categories.items()}
        result['total_unique']=unique_bytes([t for values in categories.values() for t in values])
        result['visual_fp16_denominator']=self.visual_full_bytes
        result['packed_visual_ratio']=(result['visual_base']+result['visual_residual'])/self.visual_full_bytes
        if self.kvarn_records:
            result['materialized_visual_ratio']=result['visual_base']/self.visual_full_bytes
            result['encoded_reference_ratio']=result['kvarn_packed_reference']/self.visual_full_bytes
            result['packed_visual_ratio']=None
        result['full_visual_retained']=any(pair is not None for pair in self.references)
        result['runtime']='materialized_kvarn_reference' if self.kvarn_records else 'fused_packed_visual_append_only_fp16_text'
        return result

    def event_rows(self):
        return [dict(turn=k[0],phase=k[1],pass_kind=k[2],layer=k[3],**v)
                for k,v in sorted(self.events.items())]

    def close(self):self._parent.close(self.cache)
