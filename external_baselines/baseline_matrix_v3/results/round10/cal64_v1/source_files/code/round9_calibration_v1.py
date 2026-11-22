"""Round9 same-student-state local V fitting. No model/remote/task scorer.

fit(samples,cfg) consumes private calibration shards, replaces the existing224
V factors, and leaves112 K factors fixed. The LS warm start is conditional and
real-valued in(g,h=g*offset); all candidate acceptance/checkpoints use hardRE
and the unchanged packed-reader callback. STE gradients are explicitly a proxy.
"""
from __future__ import annotations
from copy import copy,deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,inspect,json,math
import torch
import torch.nn.functional as F

SCHEMA='trajkv.round9.local_sample.v1'
FIELDS=('direction_code','direction_gain','direction_mean','v_code','v_scale','v_min','depth_residual_code','depth_residual_alpha','depth_residual_lookup')
BASE_METADATA_SHA='8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea'
def require(x,m):
 if not x:raise ValueError('ROUND9_CALIBRATION: '+m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_new(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def load_sample_shards(paths,expected_sha256=None):
 """Expand Root collector calibration_case.v1; preserve shared CPU tensors."""
 samples=[]
 for path in paths:
  path=Path(path);digest=sha(path)
  if expected_sha256 is not None:require(expected_sha256[str(path)]==digest,'private shard SHA differs')
  case=torch.load(path,map_location='cpu',weights_only=True)
  require(case['schema']=='trajkv.round9.calibration_case.v1' and case['all_28_layers_captured'],'wrong collection case')
  source=case['source']
  for event in case['events']:
   i=event['layer_idx'];layer=case['layers'][i]
   sample={**layer,**event,'book':case['book'],'visual_positions':case['visual_positions'],
           'selected_stages':case['selected_stages'],'original_factors':case['original_factors'],
           '_shard_path':str(path),'_shard_sha256':digest}
   require(sample['id']==source['id'] and sample['source_id']==source['source_id'] and sample['image_sha256']==source['original_image_sha256'],'event/source identity differs')
   validate_sample(sample);samples.append(sample)
 return samples


def load_collection(directory):
 directory=Path(directory);complete=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
 require(complete['status']=='PASS_CALIBRATION_32_REAL_MODEL_SOURCES' and complete['actual_GPU'] is True and complete['references_read'] is False,'real calibration collection not closed')
 paths=[x['file'] for x in complete['sources']];require(len(paths)==32,'exact32 closed private shards')
 samples=load_sample_shards(paths,{x['file']:x['sha256'] for x in complete['sources']})
 require(sha(complete['wo_file'])==complete['wo_sha256'],'private W_O SHA differs')
 wo=torch.load(complete['wo_file'],map_location='cpu',weights_only=True)
 return samples,wo,{'collection_complete':str(directory/'complete.json'),'collection_complete_sha256':sha(directory/'complete.json'),'private_wo_sha256':complete['wo_sha256'],'private_weights_must_not_be_published':True}


def validate_sample(s):
 require(s.get('schema')==SCHEMA and s.get('purpose')=='calibration','only declared calibration samples')
 require(not(set(s)&{'answers','labels','reference_answers','development_scores'}),'labels/development scores forbidden')
 require(isinstance(s.get('source_id'),str) and isinstance(s.get('id'),str) and len(s.get('image_sha256',''))==64,'source identity required')
 require(type(s.get('layer_idx')) is int and 0<=s['layer_idx']<28,'M1 layer required')
 q=s['q'];require(q.ndim==4 and tuple(q.shape[:2])==(1,28) and q.shape[-1]==128 and q.dtype==torch.float16,'actual M1 FP16 post-RoPE q required')
 require(s['q_rotated'].shape==q.shape and s['q_rotated'].dtype==torch.float16,'actual frozen Half WHT query required')
 k,v=s['visual_k'],s['visual_v'];require(k.shape==v.shape and tuple(k.shape[:2])==(1,4) and k.shape[-1]==128 and k.dtype==v.dtype==torch.float16,'native visual KV geometry')
 n=k.shape[-2];require(n>0 and n%32==0 and s['visual_positions'].numel()==n,'real complete32 token words required')
 tk,tv=s['exact_keys'],s['exact_values'];require(tk.shape==tv.shape and tuple(tk.shape[:2])==(1,4) and tk.shape[-1]==128 and tk.dtype==tv.dtype==torch.float16,'same-student exact text KV')
 for name,length in [('visual_positions',n),('exact_positions',tk.shape[-2]),('query_positions',q.shape[-2])]:
  require(s[name].dtype==torch.long and s[name].ndim==1 and s[name].numel()==length,'original serial positions required')
 require(math.isclose(float(s['scaling']),128**-.5,rel_tol=0,abs_tol=1e-12),'M1 scale changed')
 fields=s['r7_fields'];require(set(FIELDS).issubset(fields),'missing physical R7 fields')
 require(fields['direction_code'].dtype==torch.uint8 and tuple(fields['direction_code'].shape)==(1,4,n,16),'actual nibble K payload')
 require(fields['depth_residual_lookup'].dtype==torch.int32 and tuple(fields['depth_residual_lookup'].shape)==(2,4,2),'actual compact lookup')
 require(fields['depth_residual_code'].dtype==torch.int32 and fields['depth_residual_code'].shape[1:]==(n,4) and fields['depth_residual_alpha'].shape==fields['depth_residual_code'].shape[:2],'actual residual or explicit empty pool')
 require(torch.equal(s['raw_v_code'],fields['v_code']),'unfolded/raw V codes must equal frozen base bits')
 require(tuple(s['raw_v_scale'].shape)==tuple(s['raw_v_min'].shape)==(1,4,128,1),'raw all-frame V scale/min')
 require(tuple(s['book'].shape)==(16,4) and s['book'].dtype==torch.float16,'frozen low G4_16 only')
 for name in ('q','q_rotated','visual_k','visual_v','exact_keys','exact_values','raw_v_scale','raw_v_min'):
  require(bool(torch.isfinite(s[name]).all()),'nonfinite sample '+name)
 return s


def to_device(s,device):
 result=dict(s)
 for k,v in s.items():
  if isinstance(v,torch.Tensor):result[k]=v.to(device)
 result['r7_fields']={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in s['r7_fields'].items()}
 return result

def unpack_sign(code):
 shifts=torch.arange(32,device=code.device,dtype=torch.long)
 return ((code.long()[...,None]>>shifts)&1).flatten(-2).float()*2-1

def unpack_value(code,n):
 pos=torch.arange(n,device=code.device)
 return ((code.long().index_select(-1,pos//32)>>(pos%32))&1).transpose(-1,-2).float()

def decode_key_rotated(fields,book):
 packed=fields['direction_code'];indices=torch.stack((packed&15,packed>>4),-1).flatten(-2).long()
 template=book.float()[indices].flatten(-2)
 decoded=(template*fields['direction_gain'].float()).half()
 heads=[]
 for h in range(4):
  item=decoded[0,h]
  for depth in range(2):
   slot=int(fields['depth_residual_lookup'][0,h,depth])
   if slot>=0:item=(item.float()+unpack_sign(fields['depth_residual_code'][slot])*fields['depth_residual_alpha'][slot,:,None].float()).half()
  heads.append(item)
 return torch.stack(heads)[None]

def decoded_value(fields):
 n=fields['direction_gain'].shape[-2];b=unpack_value(fields['v_code'],n)
 value=((b*fields['v_scale'][...,0][:,:,None,:].float()).half().float()+fields['v_min'][...,0][:,:,None,:].float()).half()
 rows=[]
 for h in range(4):
  v=value[0,h]
  for depth in range(2):
   slot=int(fields['depth_residual_lookup'][1,h,depth])
   if slot>=0:v=(v.float()+unpack_sign(fields['depth_residual_code'][slot])*fields['depth_residual_alpha'][slot,:,None].float()).half()
  rows.append(v)
 return torch.stack(rows)[None]

def _repeat(x):return x.repeat_interleave(7,1)
def probabilities(s,fields=None,book=None,native=False):
 q=s['q'].float();scale=float(s['scaling'])
 if native:zv=q@_repeat(s['visual_k']).float().transpose(-1,-2)*scale
 else:
  kr=decode_key_rotated(fields,book)
  zv=s['q_rotated'].float()@_repeat(kr).float().transpose(-1,-2)*scale
  zv=zv+(q*_repeat(fields['direction_mean']).float()).sum(-1,keepdim=True)*scale
 zt=q@_repeat(s['exact_keys']).float().transpose(-1,-2)*scale
 positions=torch.cat((s['visual_positions'],s['exact_positions']))
 visible=positions[None,:]<=s['query_positions'][:,None]
 logits=torch.cat((zv,zt),-1).masked_fill(~visible[None,None],-torch.inf)
 require(bool(visible.any(-1).all()),'empty causal attention support')
 return logits.softmax(-1)

def dense_heads(s,fields=None,book=None,native=False):
 p=probabilities(s,fields,book,native);n=s['visual_positions'].numel()
 v=s['visual_v'] if native else decoded_value(fields)
 return (p[...,:n]@_repeat(v).float()+p[...,n:]@_repeat(s['exact_values']).float()).half()

def project(heads,weight,ideal=False):
 x=heads.transpose(1,2).reshape(heads.shape[0],heads.shape[-2],-1)
 weight=weight.detach()
 return F.linear(x.float(),weight.float()) if ideal else F.linear(x.to(weight.dtype),weight)

class _ExactHardForward(torch.autograd.Function):
 @staticmethod
 def forward(ctx,proxy,actual):return actual
 @staticmethod
 def backward(ctx,gradient):return gradient,None

def hard_forward_ste(proxy,actual):return _ExactHardForward.apply(proxy,actual)

def _ops(cfg):
 if 'value_ops' in cfg:return cfg['value_ops']
 import round9_value_bank_v1 as ops
 return ops

def value_fields(s,theta,cfg,reencode=True):
 ops=_ops(cfg);state=SimpleNamespace(**s['r7_fields'])
 raw=(s['raw_v_code'],s['raw_v_scale'],s['raw_v_min'])
 # Frozen helper indexes a table. Training uses a layer-local4-vector;
 # package it as one table row and select0, preserving its autograd graph.
 g,o=theta['value_head_factors'],theta['value_center_offsets']
 if g.ndim==1:
  require(g.shape==o.shape==(4,),'layer-local V4 shape')
  packet={'value_head_factors':g[None,:],'value_center_offsets':o[None,:]};index=0
 else:
  require(g.shape==o.shape==(28,4),'full V28x4 shape');packet=theta;index=s['layer_idx']
 code,scale,minimum=ops.apply_theta_to_fields(raw,packet,layer_idx=index,fold_fn=cfg.get('fold_fn'))
 state.v_code,state.v_scale,state.v_min=code,scale,minimum
 if reencode:state,_=ops.reencode_value_residuals(state,s['visual_v'],s['selected_stages'],s['layer_idx'])
 return vars(state)

def make_legacy_callbacks(legacy_a2,legacy_experiments,legacy_scripts):
 """Resolve unchanged functions only; no model or bank hook is installed."""
 import sys
 before=list(sys.path)
 try:
  sys.path[:0]=[str(legacy_a2),str(legacy_experiments),str(legacy_scripts),str(Path(legacy_a2).parent)]
  from affine_rate_bank import fold
  from round7_key_codec import encode_key,decode_key,encode_sign_depth
  from round7_key_attention import attention
 finally:sys.path[:]=before
 def packed_reader(s,fields,book):
  state=SimpleNamespace(**fields);state.visual_pos=s['visual_positions']
  state.text_k,state.text_v=s['exact_keys'],s['exact_values'];state.text_pos=s['exact_positions']
  state.text_length=s['exact_keys'].shape[-2];state.length=int(s['query_positions'][-1])+1
  return attention(s['q'],state,book,int(s['visual_positions'].numel()))
 dependencies={}
 for fn in (fold,encode_key,decode_key,encode_sign_depth,attention):
  path=inspect.getsourcefile(fn);dependencies[path]=sha(path)
 return {'fold_fn':fold,'encode_key':encode_key,'decode_key':decode_key,'encode_sign_depth':encode_sign_depth,'packed_reader':packed_reader,'legacy_source_hashes':dependencies}

def source_balanced_weights(samples):
 groups={}
 for s in samples:groups.setdefault(s['source_id'],{}).setdefault(s['id'],[]).append(s)
 result={}
 for sources in groups.values():
  for events in sources.values():
   for s in events:result[id(s)]=1/(len(groups)*len(sources)*len(events))
 return result

def _wo(cfg,layer,device):
 values=cfg['wo_by_layer'];w=values(layer) if callable(values) else values[layer]
 require(tuple(w.shape)==(3584,3584),'exact M1 W_O weight required; do not duplicate it per event')
 cache=cfg.setdefault('_wo_runtime_cache',{});key=(layer,str(device))
 if key not in cache:cache[key]=w.to(device).detach()
 return cache[key]

def teacher_projected(s,cfg):
 if cfg.get('teacher_reader'):
  with torch.no_grad():heads=cfg['teacher_reader'](s)
 else:
  with torch.no_grad():heads=dense_heads(s,native=True)
 return project(heads,_wo(cfg,s['layer_idx'],s['q'].device)).detach()

def _hard_value_output(s,theta,cfg,grad=False):
 fields=value_fields(s,theta,cfg,True)
 with torch.no_grad():actual=cfg['packed_reader'](s,fields,s['book'])
 require(tuple(actual.shape)==tuple(s['q'].shape) and bool(torch.isfinite(actual).all()),'packed callback output shape/nonfinite')
 proxy=dense_heads(s,fields,s['book'])
 result=hard_forward_ste(proxy,actual) if grad else actual
 return project(result,_wo(cfg,s['layer_idx'],s['q'].device)),float((proxy.detach().float()-actual.float()).abs().max())


def solve_positive_gain_ls(H,r,initial,*,gain_floor=1e-4,ridge=1e-5):
 """Eight-variable per-layer convex proxy:4 g>=floor,4 h unconstrained."""
 H,r,initial=H.double(),r.double(),initial.double();require(H.shape==(8,8) and r.shape==initial.shape==(8,),'LS8 geometry')
 regularizer=ridge*max(float(H.diag().mean()),1e-12);A=H+regularizer*torch.eye(8,device=H.device,dtype=torch.double);b=r+regularizer*initial
 best=None;bestval=float('inf')
 for mask in range(16):
  active=[i for i in range(4) if mask&(1<<i)];free=[i for i in range(8) if i not in active]
  x=initial.clone()
  if active:x[active]=gain_floor
  rhs=b[free]-(A[free][:,active]@x[active] if active else 0)
  x[free]=torch.linalg.lstsq(A[free][:,free],rhs).solution
  if not bool(torch.isfinite(x).all()) or bool((x[:4]<gain_floor-1e-8).any()):continue
  value=float(.5*x@A@x-b@x)
  if value<bestval:best,bestval=x,value
 require(best is not None,'no finite positive-gain LS proposal')
 return best.float()


def value_ls_warmstart(samples,initial_g,initial_h,cfg):
 """Real affine proxy with CURRENT hardRE residuals frozen; not FP16 optimum."""
 device=cfg.get('device','cuda:0');weights=source_balanced_weights(samples)
 H=torch.zeros(28,8,8,dtype=torch.float64);rhs=torch.zeros(28,8,dtype=torch.float64)
 with torch.no_grad():
  for original in samples:
   s=to_device(original,device);l=s['layer_idx'];g,h=initial_g[l].to(device),initial_h[l].to(device)
   theta={'value_head_factors':g,'value_center_offsets':h/g}
   fields=value_fields(s,theta,cfg,True);p=probabilities(s,fields,s['book']);n=s['visual_positions'].numel()
   bits=unpack_value(s['raw_v_code'],n);scale=s['raw_v_scale'][...,0][:,:,None,:].float();mid=s['raw_v_min'][...,0][:,:,None,:].float()+.5*scale
   # Exact hard current decoded values; hold its rounding/residual remainder
   # constant while making an affine g/h proposal about this current point.
   Xg=scale*(bits-.5);Xh=scale.expand_as(bits)
   base=decoded_value(fields).float()-Xg*g[None,:,None,None]-Xh*h[None,:,None,None]
   const=(p[...,:n]@_repeat(base)+p[...,n:]@_repeat(s['exact_values']).float())
   columns=[]
   for coeff in (Xg,Xh):
    for head in range(4):
     column=torch.zeros_like(const);sl=slice(head*7,(head+1)*7)
     column[:,sl]=p[:,sl,:,:n]@coeff[:,head:head+1].float()
     columns.append(project(column,_wo(cfg,l,device),ideal=True).reshape(-1))
   X=torch.stack(columns,-1).double();target=teacher_projected(s,cfg).float()
   y=(target-project(const,_wo(cfg,l,device),ideal=True)).reshape(-1).double()
   weight=weights[id(original)]/y.numel();H[l]+=(weight*X.T@X).cpu();rhs[l]+=(weight*X.T@y).cpu()
 proposals=[]
 for l in range(28):
  proposals.append(solve_positive_gain_ls(H[l],rhs[l],torch.cat((initial_g[l].cpu(),initial_h[l].cpu())),
      gain_floor=cfg.get('gain_floor',1e-4),ridge=cfg.get('ls_ridge',1e-5)))
 params=torch.stack(proposals);return params[:,:4],params[:,4:]


def _prepare(samples,cfg):
 samples=list(samples);require(samples,'empty calibration data')
 for s in samples:validate_sample(s)
 require({s['layer_idx'] for s in samples}==set(range(28)),'training must cover all28 layers')
 require(callable(cfg.get('packed_reader')),'actual unchanged packed-reader callback is mandatory')
 require(callable(cfg.get('fold_fn')),'unchanged actual fold function is mandatory')
 steps=int(cfg.get('steps',100));require(0<=steps<=200,'finite prototype/fit update budget0..200')
 outer=int(cfg.get('outer_loops',1));require(1<=outer<=3 and (outer==1 or callable(cfg.get('refresh_samples'))),'refresh callback required for multiple student-state outer loops')
 if not cfg.get('fixture_only',False):
  require(str(cfg.get('device','cuda:0')).startswith('cuda') and torch.cuda.is_available(),'real calibration requires CUDA')
  require(torch.cuda.get_device_capability()==(8,6),'actual SM86 required')
  require(len({s['image_sha256'] for s in samples})==32,'real calibration requires32 declared image sources')
  require(all(len(s.get('_shard_sha256',''))==64 for s in samples),'real calibration requires frozen private-shard hashes')
 return samples,steps,outer

def _initial(cfg,samples):
 base=cfg.get('initial_factors') or samples[0]['original_factors']
 if isinstance(base,(str,Path)):base=json.loads(Path(base).read_text(encoding='utf-8-sig'))
 init=cfg.get('initial_theta') or base
 if isinstance(init,(str,Path)):init=json.loads(Path(init).read_text(encoding='utf-8-sig'))
 g=torch.tensor(init['value_head_factors'],dtype=torch.float32);o=torch.tensor(init['value_center_offsets'],dtype=torch.float32)
 require(g.shape==o.shape==(28,4) and bool(torch.isfinite(g).all()) and bool((g>0).all()) and bool(torch.isfinite(o).all()),'absolute original V table required')
 return base,g,g*o

def _provenance(samples,cfg):
 callback_sources={}
 for name in ('packed_reader','fold_fn','encode_key','decode_key','encode_sign_depth','teacher_reader'):
  f=cfg.get(name)
  if f:
   p=inspect.getsourcefile(f)
   if p and Path(p).is_file():callback_sources[name]={'file':str(p),'sha256':sha(p)}
 ops=_ops(cfg)
 for name in ('apply_theta_to_fields','decode_value_fields','reencode_value_residuals'):
  p=inspect.getsourcefile(getattr(ops,name))
  if p and Path(p).is_file():callback_sources['value_ops.'+name]={'file':str(p),'sha256':sha(p)}
 return {'module_sha256':sha(__file__),'callback_sources':callback_sources,
  'private_train_shards':sorted({(s.get('_shard_path','live'),s.get('_shard_sha256','caller_supplied_live_state')) for s in samples}),
  'calibration_ids':sorted({s['id'] for s in samples}),'source_ids':sorted({s['source_id'] for s in samples}),
  'student_snapshots':list({canonical(s['student_snapshot']):s['student_snapshot'] for s in samples}.values()),
  'train_labels_or_development_answers_read':False,'teacher_backend':'caller exact teacher_reader' if cfg.get('teacher_reader') else 'dense FP32 attention from same native FP16 visual/student text tensors, rounded then frozen W_O',
  'hard_RE_sign_alpha_gradient':'stop-gradient; integer assignment/FP16 cast differentiated only through declared STE proxy',
  'collection_provenance':cfg.get('collection_provenance'),
  'legacy_source_hashes':cfg.get('legacy_source_hashes',{}),
  'fixture_only':bool(cfg.get('fixture_only',False))}


def fit(samples,cfg):
 """Return value_theta.v1 + history. Caller owns collection, models and GPU."""
 samples,steps,outer=_prepare(samples,cfg);base,g0,h0=_initial(cfg,samples);device=cfg.get('device','cuda:0')
 torch.manual_seed(int(cfg.get('seed',9)));torch.backends.cuda.matmul.allow_tf32=False
 provenance_before=_provenance(samples,cfg)
 logg=torch.nn.ParameterList([torch.nn.Parameter(g0[l].to(device).log()) for l in range(28)])
 hh=torch.nn.ParameterList([torch.nn.Parameter(h0[l].to(device).clone()) for l in range(28)])
 optimizer=torch.optim.Adam([*logg,*hh],lr=float(cfg.get('learning_rate',.003)))
 def theta(l):
  g=logg[l].exp();return {'value_head_factors':g,'value_center_offsets':hh[l]/g}
 def tables():return torch.stack([x.detach().exp().cpu() for x in logg]),torch.stack([x.detach().cpu() for x in hh])
 def evaluate(current):
  weights=source_balanced_weights(current);total=0.
  with torch.no_grad():
   for raw in current:
    s=to_device(raw,device);pred,_=_hard_value_output(s,theta(s['layer_idx']),cfg)
    target=teacher_projected(s,cfg);loss=(pred.float()-target.float()).square().mean()
    require(bool(torch.isfinite(loss)),'nonfinite actual calibration loss');total+=weights[id(raw)]*float(loss)
  return total
 history=[];initial_loss=evaluate(samples);best_loss=initial_loss;best=tables()
 if cfg.get('ls_warmstart',True):
  pg,ph=value_ls_warmstart(samples,g0,h0,cfg)
  with torch.no_grad():
   for l in range(28):logg[l].copy_(pg[l].to(device).log());hh[l].copy_(ph[l].to(device))
  candidate=evaluate(samples);accepted=candidate<best_loss
  history.append({'phase':'conditional_real_LS_then_actual_hardRE','loss':candidate,'accepted':accepted})
  if accepted:best_loss=candidate;best=tables()
  else:
   with torch.no_grad():
    for l in range(28):logg[l].copy_(g0[l].to(device).log());hh[l].copy_(h0[l].to(device))
 seen=set();seen_sources=set();done=0
 for epoch in range(outer):
  if epoch:
   # Checkpoints from different student snapshots are not compared by their
   # raw loss. Start this new frozen-state inner problem at the prior best.
   with torch.no_grad():
    for l in range(28):logg[l].copy_(best[0][l].to(device).log());hh[l].copy_(best[1][l].to(device))
   samples=list(cfg['refresh_samples']({'value_head_factors':best[0].tolist(),'value_center_offsets':(best[1]/best[0]).tolist()},epoch))
   for sample in samples:validate_sample(sample)
   require({sample['layer_idx'] for sample in samples}==set(range(28)),'refresh lost layer coverage')
   best_loss=evaluate(samples);best=tables()
  groups={}
  for sample in samples:groups.setdefault(sample['source_id'],{}).setdefault((sample['id'],sample['step'],sample['phase']),[]).append(sample)
  source_order=sorted(groups)
  for group in groups.values():
   for events in group.values():require({x['layer_idx'] for x in events}==set(range(28)),'each sampled source/phase must contain all28 layers')
  count=steps//outer+(epoch<steps%outer)
  for local in range(count):
   source_id=source_order[done%len(source_order)];options=sorted(groups[source_id]);key=options[(done//len(source_order))%len(options)]
   batch=groups[source_id][key];optimizer.zero_grad(set_to_none=True);batch_loss=0.;max_bridge=0.;zero=True
   for raw in batch:
    s=to_device(raw,device);l=s['layer_idx'];pred,delta=_hard_value_output(s,theta(l),cfg,True)
    target=teacher_projected(s,cfg);loss=(pred.float()-target.float()).square().mean()/len(batch)
    require(bool(torch.isfinite(loss)),'nonfinite loss');loss.backward();batch_loss+=float(loss.detach());max_bridge=max(max_bridge,delta);seen.add(l)
   for l in range(28):
    active=[p.grad for p in (logg[l],hh[l]) if p.grad is not None]
    require(active and all(bool(torch.isfinite(x).all()) for x in active),'missing/nonfinite factor gradient')
    zero=zero and all(not bool(x.any()) for x in active)
   optimizer.step();require(all(bool(torch.isfinite(p).all()) for p in [*logg,*hh]),'nonfinite parameter update')
   seen_sources.add(source_id);done+=1;history.append({'phase':'actual_packed_forward_STE_full28_update','step':done,'source_id':source_id,
       'loss_before_update':batch_loss,'proxy_vs_packed_max_abs':max_bridge,'finite_zero_gradient':zero})
  value=evaluate(samples)
  if value<best_loss:best_loss=value;best=tables()
  history.append({'phase':'full_source_balanced_actual_checkpoint','outer':epoch,'loss':value,'best_loss':best_loss})
 if steps:require(seen==set(range(28)),'update schedule did not cover all layers')
 g,h=best;require(bool(torch.isfinite(g).all()) and bool((g>0).all()),'invalid final gains')
 provenance=_provenance(samples,cfg)
 require(provenance['module_sha256']==provenance_before['module_sha256'] and provenance['callback_sources']==provenance_before['callback_sources'],'training source changed during fit')
 require(all(sha(path)==digest for path,digest in cfg.get('legacy_source_hashes',{}).items()),'legacy callback source changed during fit')
 result={'schema':'trajkv.round9.value_theta.v1','base_profile':'r7_joint_k1_gt_cond','base_metadata_factor_sha256':cfg.get('base_metadata_factor_sha256',BASE_METADATA_SHA),
  'parameter_count':224,'effective_existing_table_parameter_count':336,'value_head_factors':g.tolist(),'value_center_offsets':(h/g).tolist(),
  'fixed_key_head_factors_sha256':canonical(base['key_head_factors']),'fit_provenance':provenance,
  'fit_summary':{'initial_actual_loss':initial_loss,'best_actual_loss':best_loss,'updates':done,'layers_updated':sorted(seen),
    'sources_seen_by_optimizer':sorted(seen_sources),'LS_uses_all_calibration_sources':bool(cfg.get('ls_warmstart',True)),'student_state_refreshes':outer-1,'frozen_student_snapshot_only':outer==1,'objective':'same student q/text, restored visual KV target, projected attention output error',
    'free_generation_or_task_quality_claim':False,'real_LS_not_FP16_global_optimum':True},'history':history}
 if cfg.get('output'):
  output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
  write_new(output/'value_theta.json',{k:v for k,v in result.items() if k!='history'});write_new(output/'fit_history.json',history)
 return result
