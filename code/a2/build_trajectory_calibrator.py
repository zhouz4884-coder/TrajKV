"""Derive a next-phase multi-token calibrator without changing frozen methods."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 source=Path(__file__).with_name('calibrate_endogenous.py');text=source.read_text(encoding='utf-8');fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
 def replace(old,new):
  nonlocal text
  if text.count(old)!=1:raise RuntimeError('Calibration source changed: '+old[:60])
  text=text.replace(old,new)
 replace("target=out.logits[:,-1].float().log_softmax(-1).cpu();layers=[]",'''teacher_cache=out.past_key_values;teacher_ids=[];teacher_lp=[]
   import run_p175_modern_replication as oldbase
   stop=oldbase.stop_ids(model,processor);prompt_length=f['input_ids'].shape[-1]
   for target_step in range(c.get('target_tokens',8)):
    teacher_lp.append(out.logits[:,-1].float().log_softmax(-1).cpu());token=int(out.logits[:,-1].argmax(-1));teacher_ids.append(token)
    if token in stop or target_step+1>=c.get('target_tokens',8):break
    with attention_bridge(model,native=True):
     out=model(input_ids=torch.tensor([[token]],device=model.device),position_ids=torch.tensor([[prompt_length+target_step]],device=model.device),past_key_values=teacher_cache,use_cache=True,return_dict=True,logits_to_keep=1)
   target=torch.stack(teacher_lp,1)
   if len(teacher_ids)>1:suffix=torch.cat((suffix,torch.tensor([teacher_ids[:-1]],device=suffix.device)),1)
   layers=[]''')
 replace("for layer in out.past_key_values.layers:","for layer in teacher_cache.layers:")
 replace("codes=unpack(rec,kbits,x.shape)","if not c.get('base_rms',True):beta=torch.ones_like(beta)\n    codes=unpack(rec,kbits,x.shape)")
 replace("{'id':row['id'],'visual':visual.cpu()", "{'id':row['id'],'image_id':row['image_id'],'visual':visual.cpu()")
 replace("out=model(input_ids=suffix,position_ids=position,past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1)","out=model(input_ids=suffix,position_ids=position,past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=case['target'].shape[1])")
 replace("lp=out.logits[:,-1].float().log_softmax(-1)","lp=out.logits.float().log_softmax(-1)")
 replace("(loss*128).backward();factors.grad.div_(128)","penalty=factors.square().mean()+(value_factors.square().mean() if train_value else 0.)\n  objective=loss+c.get('regularization',0.)*penalty\n  (objective*128).backward();factors.grad.div_(128)")
 replace("'steps':len(records),'images':len(cases),'model_weights_frozen':True", "'steps':len(records),'cases':len(cases),'images':len({r['image_id'] for r in cases}),'model_weights_frozen':True")
 replace("'scope':'calibration feasibility/fit, first answer FP16 distribution; dense differentiable student; no task accuracy or packed efficiency claim'", "'scope':'next-phase unlabelled teacher answer trajectory calibration; dense differentiable student; no task accuracy or packed efficiency claim','target_tokens_max':c.get('target_tokens',8),'base_rms':c.get('base_rms',True)")
 dest=source.with_name('calibrate_trajectory.py')
 if dest.exists():raise RuntimeError('Derived next-phase source already exists')
 dest.write_text('# Derived from frozen first-token calibration; independent next-phase program.\n'+text,encoding='utf-8')
 info={'source_sha256':fingerprint,'derived_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
  'changes':['Full-generated answer positions, up to8 tokens','identity initializer for K2 if requested','small positive-factor regularization if requested','12-image disjoint calibration pool possible'],
  'stage':'prepared for after current confirmation/external/timing, not launched'}
 (ROOT/'audit/a2/next_trajectory_calibrator.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
if __name__=='__main__':main()
