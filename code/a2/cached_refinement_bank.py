"""Amortize query-based page choice over one question, keeping all base support."""
import torch
from masked_bank import A2Bank,unique_bytes

class CachedRefinementBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'asym',legacy_scripts,kivi_source,False)
  self.method=method
  self.cached_masks=[torch.zeros((1,s.text_k.shape[1],s.visual_pos.numel()//self.page),device=s.text_k.device,dtype=torch.uint8) for s in self.states]
  # The exact host-side control state also owns a fixed tensor representation.
  # Every comparison policy allocates these buffers, including none/all.
  self.cached_control=torch.full((len(self.states),2),-1,dtype=torch.int64,device='cpu')
  self.cache_fresh_calls=0;self.cache_reuse_calls=0

 def select(self,q,i,remaining=None):
  if self.selector=='prefill_risk':
   if self.phase.endswith('prefill'):
    ref=self.refinements[i];active=(ref[:3] if self.refine_k else [])+(ref[3:] if self.refine_v else [])
    total=unique_bytes(active);cells=self.cached_masks[i].numel()
    return (None,total,0,cells,0) if remaining is None or total<=remaining else (None,0,0,0,1)
   original=self.selector
   try:self.selector='risk';return super().select(q,i,remaining)
   finally:self.selector=original
  if self.selector not in ('cached_risk','prefill_cached_risk'):return super().select(q,i,remaining)
  s=self.states[i];cells=self.cached_masks[i].numel();ref=self.refinements[i]
  active=(ref[:3] if self.refine_k else [])+(ref[3:] if self.refine_v else [])
  total=unique_bytes(active);unit=total//cells;control_bytes=16
  if remaining is not None and remaining<control_bytes:return None,0,0,0,1
  fresh=int(self.cached_control[i,0])!=self.turn or self.phase.endswith('prefill')
  if not fresh:
   count=int(self.cached_control[i,1]);self.cache_reuse_calls+=1
   fee=control_bytes+(cells if count else 0);cost=count*unit
   if remaining is not None and fee+cost>remaining:
    return None,0,control_bytes if remaining>=control_bytes else 0,0,1
   return (self.cached_masks[i] if count else None),cost,fee,count,int(count==0)
  self.cache_fresh_calls+=1
  # Parent selector pays for scoring, topk and its temporary output mask.
  # Additional copy traffic and the retained control update are charged here.
  store_fee=2*cells+control_bytes
  if remaining is not None and remaining<store_fee:return None,0,control_bytes,0,1
  available=max(0,int(total*self.fraction)-store_fee)
  if remaining is not None:available=min(available,max(0,remaining-store_fee))
  original=self.selector
  try:
   self.selector='risk';mask,residual,selection,count,skip=super().select(q,i,available)
  finally:self.selector=original
  if mask is None:self.cached_masks[i].zero_()
  else:self.cached_masks[i].copy_(mask)
  self.cached_control[i,0]=self.turn;self.cached_control[i,1]=count
  fee=selection+store_fee
  if remaining is not None and fee>remaining:raise RuntimeError('Cached selection fee exceeds available budget')
  if original=='prefill_cached_risk' and self.phase.endswith('prefill'):
   if remaining is None or total+fee<=remaining:return None,total,fee,cells,0
  return (self.cached_masks[i] if count else None),residual,fee,count,skip

 def owned(self):
  result=super().owned();buffers=getattr(self,'cached_masks',[]);control=getattr(self,'cached_control',None)
  added=unique_bytes(buffers+([control] if control is not None else []))
  result['selector_cache_buffers']=unique_bytes(buffers)
  result['selector_cache_host_control']=unique_bytes([control]) if control is not None else 0
  result['selector_metadata']+=added;result['total_unique']+=added
  result['visual_with_metadata_ratio']=(result['visual_base']+result['visual_residual']+result['selector_metadata'])/self.visual_full_bytes
  return result

 def result_metadata(self):
  result=super().result_metadata();result.update(backend='same progressive packed payload with question-amortized selector option',
   selector_cache_fresh_calls=self.cache_fresh_calls,selector_cache_reused_calls=self.cache_reuse_calls,
   selector_cache_scope='mask chosen from current-question prefill q; reused on later decode reads until the next question; all policies own identical buffers',
   selector_cache_byte_scope='GPU uint8 head-page masks plus fixed CPU int64 turn/count controls; both included in selector_metadata',
   budget_scope='global residual+selection byte cap; cached policies pay full scoring on prefill, copy/store, and later mask/control reads')
  return result
