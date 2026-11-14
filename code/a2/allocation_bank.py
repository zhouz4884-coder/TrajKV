"""Actual static layer allocation: unused axis refinements are physically removed.

One K-r2 layer costs exactly two V-sign layers for this g128/d128 codec. Base1
is 1.25 effective bits/scalar including scale/min. Layer subsets produce equal
encoded byte points without pretending that K-r2 means a one-bit increment.
"""
from masked_bank import A2Bank
class AllocationBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'asym1',legacy_scripts,kivi_source,False)
  cfg=type(self).settings;k=set(cfg['k_layers']);v=set(cfg['v_layers']);self.axis_flags=[]
  if max(k|v,default=-1)>=len(self.states):raise ValueError('Layer allocation out of range')
  for i,r in enumerate(self.refinements):
   flags=(i in k,i in v);self.axis_flags.append(flags)
   if flags==(False,False):self.refinements[i]=None
   elif flags==(True,False):self.refinements[i]=list(r[:3])+list(r[:3])
   elif flags==(False,True):self.refinements[i]=list(r[3:])+list(r[3:])
  self.method=method
 def read(self,q,i):
  saved=(self.refine_k,self.refine_v);self.refine_k,self.refine_v=self.axis_flags[i]
  try:return super().read(q,i)
  finally:self.refine_k,self.refine_v=saved
 def result_metadata(self):
  r=super().result_metadata();r['axis_allocation']=[{'layer':i,'k_residual2':k,'v_sign':v} for i,(k,v) in enumerate(self.axis_flags)]
  r['allocation_slots']=sum(2*k+v for k,v in self.axis_flags);r['allocation_scope']='static layer table; unused residuals removed, no query selector metadata'
  return r
