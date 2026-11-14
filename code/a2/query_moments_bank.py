"""Unlabelled calibration queries from actual old-development FP16 forwards."""
from masked_bank import A2Bank
class QueryMomentsBank(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False);self.moments=[];self.measured=set()
 def read(self,q,i):
  if i not in self.measured:
   self.measured.add(i);h=self.states[i].text_k.shape[1]
   # Equal weight to each grouped query head at the final prefix position.
   z=q[:,:,-1:].float().reshape(1,h,-1,q.shape[-1])
   self.moments.append({'layer':i,'count_per_kv_head':z.shape[2],'sum_q2':z.square().sum(2)[0].cpu().tolist()})
  return super().read(q,i)
 def result_metadata(self):
  r=super().result_metadata();r.update(query_second_moments=self.moments,task_accuracy_eligible=False,
    calibration_scope='unlabelled final prefix queries on20 old development images; not future test queries')
  return r
