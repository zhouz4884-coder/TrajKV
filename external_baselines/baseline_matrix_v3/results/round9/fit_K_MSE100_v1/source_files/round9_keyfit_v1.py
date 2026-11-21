"""Same-capacity post-WHT G4_16 key fitting: MSE vs functional objective.

No new bit slots/headwise book, frozen K factors/V/43 coverage. Each candidate
is encoded with the original norm/L2 encoder and real sign-depth recomputation.
Assignment and hardRE sign/alpha are stopped; gradients use fixed-index STE.
"""
from copy import deepcopy
from pathlib import Path
import inspect,json
import torch
import round9_calibration_v1 as local

def require(v,m):local.require(v,'KEYFIT: '+m)

def key_fields(sample,centers,cfg,grad=False):
 s=sample;source=s.get('visual_k_rotated')
 require(source is not None and source.dtype==torch.float32 and tuple(source.shape)==tuple(s['visual_k'].shape),
         'exact original fast_wht native-centered FP32 visual_k_rotated required')
 book=centers.half();original=s['r7_fields'];l=s['layer_idx']
 f=torch.tensor(s['original_factors']['key_head_factors'][l],device=source.device,dtype=torch.float32)[None,:,None,None]
 with torch.no_grad():
  code,gain=cfg['encode_key'](source,book,amplitude='norm',assignment='l2',gain_factor=f)
  base=cfg['decode_key'](code,gain,book)
  result=dict(original);result['direction_code']=code;result['direction_gain']=gain
  result['depth_residual_code']=original['depth_residual_code'].detach().clone()
  result['depth_residual_alpha']=original['depth_residual_alpha'].detach().clone()
  for head in range(4):
   slots=[int(original['depth_residual_lookup'][0,head,stage]) for stage in range(2)]
   used=[x for x in slots if x>=0];require(slots[0]>=0 or slots[1]<0,'K parent-before-child')
   if used:
    codes,alphas,_,_=cfg['encode_sign_depth'](source[0,head],base[0,head],len(used))
    for depth,slot in enumerate(used):result['depth_residual_code'][slot].copy_(codes[depth]);result['depth_residual_alpha'][slot].copy_(alphas[depth])
 if grad:
  # The full-vector norm rule, two Half roundings and original head factor
  # remain in the proxy; forward gain is taken from the actual encoder.
  ids=torch.stack((code&15,code>>4),-1).flatten(-2).long()
  template=book.float()[ids].flatten(-2)
  proposal=((source.norm(dim=-1,keepdim=True)/template.norm(dim=-1,keepdim=True).clamp_min(1e-20)).half().float()*f).half()
  result['direction_gain']=local.hard_forward_ste(proposal,gain)
 return result,book

def _loss(s,centers,cfg,objective,grad=False):
 fields,book=key_fields(s,centers,cfg,grad)
 if objective=='mse':
  decoded=local.decode_key_rotated(fields,book)
  return (decoded.float()-s['visual_k_rotated']).square().mean(),0.
 proxy=local.dense_heads(s,fields,book)
 with torch.no_grad():actual=cfg['packed_reader'](s,fields,book)
 require(bool(torch.isfinite(actual).all()),'nonfinite actual packed output')
 output=local.hard_forward_ste(proxy,actual) if grad else actual
 pred=local.project(output,local._wo(cfg,s['layer_idx'],s['q'].device))
 target=local.teacher_projected(s,cfg)
 return (pred.float()-target.float()).square().mean(),float((proxy.detach().float()-actual.float()).abs().max())

def fit(samples,cfg,*,objective):
 require(objective in ('mse','functional'),'exact MSE or functional branch')
 module_source_before=local.sha(__file__)
 samples,steps,outer=local._prepare(samples,cfg)
 require(outer==1,'first K fit uses one declared frozen student snapshot; refresh is a separate refit')
 for s in samples:require('visual_k_rotated' in s,'collector lacks exact K rotated source')
 for name in ('encode_key','decode_key','encode_sign_depth'):require(callable(cfg.get(name)),'unchanged callback missing '+name)
 device=cfg.get('device','cuda:0');torch.manual_seed(int(cfg.get('seed',9)))
 torch.backends.cuda.matmul.allow_tf32=False
 initial_book_sha=cfg.get('initial_book_sha') or samples[0]['student_snapshot'].get('direction_book_path')
 require(initial_book_sha=='af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589','original frozen Gaussian G4_16 source required for matched branches')
 require(all(s['student_snapshot'].get('direction_book_path')==initial_book_sha for s in samples),'mixed initial codebook source identity')
 original=samples[0]['book'].float();require(original.shape==(16,4),'global64-parameter book only')
 require(all(torch.equal(s['book'],samples[0]['book']) for s in samples),'mixed initial codebooks')
 centers=torch.nn.Parameter(original.to(device).clone());norm=centers.detach().norm()
 optimizer=torch.optim.Adam([centers],lr=float(cfg.get('learning_rate',.003)))
 source_before=local._provenance(samples,cfg)
 def evaluate():
  weights=local.source_balanced_weights(samples);value=0.
  with torch.no_grad():
   for raw in samples:
    s=local.to_device(raw,device);loss,_=_loss(s,centers,cfg,objective)
    require(bool(torch.isfinite(loss)),'nonfinite hard calibration loss');value+=weights[id(raw)]*float(loss)
  return value
 initial=evaluate();best_loss=initial;best=centers.detach().clone();history=[]
 groups={}
 for s in samples:groups.setdefault(s['source_id'],{}).setdefault((s['id'],s['step'],s['phase']),[]).append(s)
 order=sorted(groups);seen_sources=set();seen_layers=set()
 for step in range(steps):
  sid=order[step%len(order)];keys=sorted(groups[sid]);batch=groups[sid][keys[(step//len(order))%len(keys)]]
  require({s['layer_idx'] for s in batch}==set(range(28)),'sampled source/phase lacks a layer')
  optimizer.zero_grad(set_to_none=True);loss_value=0.;bridge=0.
  for raw in batch:
   s=local.to_device(raw,device);loss,delta=_loss(s,centers,cfg,objective,True);loss=loss/len(batch)
   require(bool(torch.isfinite(loss)),'nonfinite hard forward loss');loss.backward()
   loss_value+=float(loss.detach());bridge=max(bridge,delta);seen_layers.add(s['layer_idx'])
  require(centers.grad is not None and bool(torch.isfinite(centers.grad).all()),'missing/nonfinite book gradient')
  optimizer.step()
  with torch.no_grad():
   require(bool(torch.isfinite(centers).all()) and float(centers.norm())>1e-20,'nonfinite/zero book proposal')
   # One identical declared normalization for both branches; not per-word
   # sphere projection and not a change in dictionary capacity.
   centers.mul_(norm/centers.norm())
  seen_sources.add(sid);history.append({'step':step+1,'source_id':sid,'loss_before_update':loss_value,
        'proxy_vs_packed_max_abs':bridge,'objective':objective})
  if (step+1)%int(cfg.get('checkpoint_every',32))==0 or step+1==steps:
   candidate=evaluate()
   if candidate<best_loss:best_loss=candidate;best=centers.detach().clone()
   history.append({'phase':'full_source_balanced_hard_checkpoint','step':step+1,'loss':candidate,'best_loss':best_loss})
 if steps:require(seen_layers==set(range(28)),'not all28 layers updated')
 physical=best.half();source_after=local._provenance(samples,cfg)
 require(source_before['callback_sources']==source_after['callback_sources'] and source_before['module_sha256']==source_after['module_sha256'],'training dependencies changed')
 require(local.sha(__file__)==module_source_before,'K fit source changed during training')
 require(all(local.sha(path)==digest for path,digest in cfg.get('legacy_source_hashes',{}).items()),'legacy callback source changed')
 result={'schema':'trajkv.direction-book.v1','original_book_sha256':initial_book_sha,'dimension':4,'centroids':16,'nominal_bits_per_scalar':1,
   'centers':physical.cpu().tolist(),'round9_fit':{'objective':objective,'parameter_count':64,'stored_dtype':'float16',
   'initial_hard_loss':initial,'best_hard_loss':best_loss,'updates':steps,'source_ids_seen':sorted(seen_sources),
   'layers_updated':sorted(seen_layers),'normalization':'single global L2 norm fixed to original; identical in both branches',
   'assignment':'unchanged encode_key norm/L2 recomputed every candidate','K_residuals':'actual original selected slots reencoded from each new actual Half base',
   'V_and_K_head_factors_changed':False,'student_snapshot_frozen':True,'task_quality_claim':False,
   'module_sha256':local.sha(__file__),'train_provenance':source_after},'history':history}
 if cfg.get('output'):
  out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
  local.write_new(out/'key_book.json',{k:v for k,v in result.items() if k!='history'});local.write_new(out/'fit_history.json',history)
 return result
