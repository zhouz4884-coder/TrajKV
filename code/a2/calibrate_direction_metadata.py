"""Label-free full-model calibration of folded key/value metadata.

This optimizer uses differentiable dense attention and teacher contexts. Only
held-out actual packed free generation can establish task/efficiency results.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
from transformers.cache_utils import DynamicCache
from bank import install_legacy
from quantizers import binary_centroids
from fast_wht import torch_wht
from fused_learned_direction import attention as packed_attention
from types import SimpleNamespace
import session_boundary

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 c=json.loads(args.config.read_text());args.output.mkdir(exist_ok=False,parents=True);a=c['arguments'];install_legacy(a['legacy_scripts'])
 from p180_multiturn_core import attention_bridge
 from p184_working_set import attention as custom_attention
 import run_p175_modern_replication as oldbase
 torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 processor=AutoProcessor.from_pretrained(a['model'],local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a['model'],local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 for parameter in model.parameters():parameter.requires_grad_(False)
 config=model.config.get_text_config();nl=config.num_hidden_layers;nh=config.num_key_value_heads;dim=config.hidden_size//config.num_attention_heads
 assert dim==128;modules=[layer.self_attn for layer in model.model.language_model.layers]
 book=torch.tensor(json.loads(Path(c['encoder_book_path']).read_text())['centers'],device='cuda',dtype=torch.float16)
 assert book.shape==(256,4);rows=json.loads(Path(a['samples']).read_text(encoding='utf-8'))[:c.get('images',83)];cases=[]
 def unpack1(rec,shape):
  shifts=torch.arange(32,device='cuda');return ((rec[0].long()[...,None]>>shifts)&1).reshape(shape).half()
 with torch.no_grad():
  for row in rows:
   f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in session_boundary.features(processor,row,8).items()}
   visual=(f['input_ids'][0]==model.config.image_token_id).nonzero().flatten();prompt_length=f['input_ids'].shape[-1]
   boundary=prompt_length-1 if c.get('mode')=='native_question_history' else int(visual[-1])+1
   suffix=f['input_ids'][:,boundary:].clone()
   with attention_bridge(model,native=True):out=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
   teacher_cache=out.past_key_values;ids=[];target=[];stop=oldbase.stop_ids(model,processor)
   for step in range(c.get('target_tokens',8)):
    target.append(out.logits[:,-1].float().log_softmax(-1).cpu());token=int(out.logits[:,-1].argmax(-1));ids.append(token)
    if token in stop or step+1>=c.get('target_tokens',8):break
    with attention_bridge(model,native=True):out=model(input_ids=torch.tensor([[token]],device='cuda'),position_ids=torch.tensor([[prompt_length+step]],device='cuda'),past_key_values=teacher_cache,use_cache=True,return_dict=True,logits_to_keep=1)
   if len(ids)>1:suffix=torch.cat((suffix,torch.tensor([ids[:-1]],device='cuda')),1)
   layers=[]
   for layer in teacher_cache.layers:
    key=layer.keys[:,:,:boundary].clone();value=layer.values[:,:,:boundary].clone();source=key.index_select(2,visual).float()
    mean=source.mean(-2,keepdim=True).half();z=torch_wht(source-mean.float());norm=z.norm(dim=-1,keepdim=True);initial=(norm/(128**.5)).clamp_min(1e-20)
    parts=(z/initial).reshape(-1,4);indices=[];bf=book.float()
    for start in range(0,len(parts),65536):
     part=parts[start:start+65536];distance=part.square().sum(-1,keepdim=True)+bf.square().sum(-1)[None]-2*(part@bf.T)
     indices.append(distance.argmin(-1).byte())
    codes=torch.cat(indices).reshape(*z.shape[:-1],32);template=bf[codes.long()].reshape(z.shape)
    gain=(norm/template.norm(dim=-1,keepdim=True).clamp_min(1e-20)).half()
    vx=value.index_select(2,visual).transpose(2,3).contiguous();value_group=visual.numel() if c.get('value_group_size',128)=='all' else 128;vr=binary_centroids(vx,group_size=value_group)
    layers.append({'key':key.cpu(),'value':value.cpu(),'codes':codes.cpu(),'gain':gain.cpu(),'mean':mean.cpu(),
     'v_packed':vr[0].cpu(),'v_codes':unpack1(vr,vx.shape).cpu(),'value_group':value_group,'v_scale':vr[1].cpu(),'v_center':(vr[2].float()+.5*vr[1].float()).cpu()})
   nv=visual.numel();e=nl*nh*nv*128
   proposed=e*(2/8+1/8+4/128)+nl*nh*nv*2+nl*nh*128*2+2048+nl*nh*256*4*2+8*nv+nl*nh*2*4
   active=not c.get('headwise_if_budget',True) or 16*proposed/(4*e)<=c.get('book_budget_bits',1.8)
   cases.append({'id':row['id'],'image_id':row['image_id'],'boundary':boundary,'visual':visual.cpu(),'suffix':suffix.cpu(),'target':torch.stack(target,1),'layers':layers,'headwise_active':active})
   del f,out,teacher_cache,key,value,source,z,vx,vr,layers
 key_mode=c.get('key_gain_mode','none');assert key_mode in ['none','centered','whole'];kg=torch.nn.Parameter(torch.zeros(nl,nh,device='cuda')) if key_mode!='none' else None
 joint=c.get('train_value',True);vg=torch.nn.Parameter(torch.zeros(nl,nh,device='cuda')) if joint else None;vo=torch.nn.Parameter(torch.zeros(nl,nh,device='cuda')) if joint else None
 parameters=([kg] if kg is not None else [])+([vg,vo] if joint else []);assert parameters;optimizer=torch.optim.Adam(parameters,lr=c.get('learning_rate',.003));logs=[]
 def loss_for(case):
  cache=DynamicCache(config=config);visual=case['visual'].to('cuda');side=[]
  for i,entry in enumerate(case['layers']):
   cache.update(entry['key'].to('cuda'),entry['value'].to('cuda'),i)
   side.append({k:(v.to('cuda') if isinstance(v,torch.Tensor) else v) for k,v in entry.items() if k not in ('key','value')})
  suffix=case['suffix'].to('cuda');total=case['boundary']+suffix.shape[-1];allpos=torch.arange(total,device='cuda');mask=torch.ones(total,device='cuda',dtype=torch.bool);mask[visual]=False;textpos=allpos[mask]
  qpos=torch.arange(case['boundary'],total,device='cuda');positions=torch.cat((visual,textpos));heads=torch.arange(nh,device='cuda')[None,:,None,None]
  def read(q,index):
   entry=side[index];group=q.shape[1]//nh
   decoder=book;template=book.float()[entry['codes'].long()].reshape(1,nh,len(visual),128)
   key_factor=kg[index][None,:,None,None].exp() if kg is not None else 1.
   gain=(entry['gain'].float()*key_factor).half();mean=(entry['mean'].float()*(key_factor if key_mode=='whole' else 1.)).half()
   rotated=(template*gain.float()).half().repeat_interleave(group,1)
   qr=torch_wht(q).half().float();scores_v=qr@rotated.float().transpose(-1,-2)/(128**.5)
   scores_v+=(q.float()*mean.repeat_interleave(group,1).float()).sum(-1,keepdim=True)/(128**.5)
   key=cache.layers[index].keys.index_select(2,textpos).repeat_interleave(group,1);value=cache.layers[index].values.index_select(2,textpos).repeat_interleave(group,1)
   scale=entry['v_scale'].float()*(vg[index][None,:,None,None].exp() if joint else 1.);scale=scale.half()
   minimum=(entry['v_center']+((vo[index][None,:,None,None] if joint else 0.)-.5)*scale.float()).half()
   decoded=((entry['v_codes']*scale.repeat_interleave(case['layers'][index]['value_group'],-1)).half()+minimum.repeat_interleave(case['layers'][index]['value_group'],-1)).half().transpose(2,3).contiguous().repeat_interleave(group,1)
   scores=torch.cat((scores_v,q.float()@key.float().transpose(-1,-2)/(128**.5)),-1)
   scores=scores.masked_fill(positions[None,None,None,:]>qpos[None,None,:,None],-torch.inf)
   weights=(scores-scores.amax(-1,keepdim=True)).exp();values=torch.cat((decoded,value),2)
   dense=((weights.half().float()@values.float())/weights.sum(-1,keepdim=True)).half()
   if not c.get('packed_forward',True):return dense
   # Exact deployed forward; the dense derivative is the quantizer's STE.
   # Keep text strides fixed so decode length does not trigger repeated JIT.
   nt=textpos.numel();assert nt<=512
   with torch.no_grad():
    tk=torch.empty((1,nh,512,128),device='cuda',dtype=torch.float16);tv=torch.empty_like(tk)
    tk[:,:,:nt].copy_(cache.layers[index].keys.index_select(2,textpos));tv[:,:,:nt].copy_(cache.layers[index].values.index_select(2,textpos))
   state=SimpleNamespace(direction_code=entry['codes'],direction_gain=gain,direction_mean=mean,
    v_code=entry['v_packed'],v_scale=scale,v_min=minimum,text_k=tk,text_v=tv,text_pos=textpos,visual_pos=visual,text_length=nt,length=total)
   exact=packed_attention(q,state,decoder,1,True,case['layers'][index]['value_group'])
   result=(dense.float()+(exact.float()-dense.float()).detach()).half()
   if c.get('assert_forward_identity',False) and not torch.equal(result.detach(),exact):raise RuntimeError('Surrogate changed the packed forward')
   return result
  for i,module in enumerate(modules):module._p185_read=lambda q,i=i:read(q,i)
  try:
   with custom_attention(model):out=model(input_ids=suffix,position_ids=qpos[None],past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=case['target'].shape[1])
  finally:
   for module in modules:del module._p185_read
  lp=out.logits.float().log_softmax(-1);target=case['target'].to('cuda')
  return (target.exp()*(target-lp)).sum(-1).mean()
 manifest={'config':c,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'samples_sha256':hashlib.sha256(Path(a['samples']).read_bytes()).hexdigest(),
  'labels_used':False,'model_weights_frozen':True,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
  'scope':'actual GPU full-model offline quantizer calibration; optional exact packed forward with dense STE; not serving efficiency or task accuracy'}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for step in range(c.get('steps',2)):
  optimizer.zero_grad(set_to_none=True);started=time.perf_counter();loss=loss_for(cases[step%len(cases)])
  penalty=(kg.square().mean() if kg is not None else 0.)+(vg.square().mean()+vo.square().mean() if joint else 0.)
  objective=loss+c.get('regularization',.01)*penalty
  if not objective.requires_grad:raise RuntimeError('Functional metadata gradient detached')
  (objective*128).backward()
  for parameter in parameters:
   parameter.grad.div_(128)
   if not bool(torch.isfinite(parameter.grad).all()):raise RuntimeError('Nonfinite functional-metadata gradient')
  grad=float(sum(parameter.grad.square().sum() for parameter in parameters).sqrt())
  if grad==0:raise RuntimeError('Zero functional-metadata gradient')
  optimizer.step()
  with torch.no_grad():
   if kg is not None:kg.clamp_(-c.get('key_log_bound',.1 if key_mode=='whole' else .7),c.get('key_log_bound',.1 if key_mode=='whole' else .7))
   if joint:vg.clamp_(-.7,.7);vo.clamp_(-.35,.35)
  torch.cuda.synchronize();entry={'step':step,'id':cases[step%len(cases)]['id'],'kl_before_update':float(loss.detach()),'gradient_norm':grad,'seconds':time.perf_counter()-started};logs.append(entry);print(json.dumps(entry),flush=True)
  del loss,objective
 with torch.no_grad():final=[float(loss_for(case)) for case in cases]
 assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
 result={'status':'COMPLETE','labels_used':False,'model_weights_frozen':True,'images':len({r['image_id'] for r in cases}),'cases':len(cases),'steps':len(logs),
  'key_head_factors':kg.detach().exp().cpu().tolist() if kg is not None else None,'value_head_factors':vg.detach().exp().cpu().tolist() if joint else None,
  'value_center_offsets':vo.detach().cpu().tolist() if joint else None,'key_gain_mode':key_mode,'value_group_size':c.get('value_group_size',128),'parameter_count':sum(p.numel() for p in parameters),
  'calibration_mode':c.get('mode','compressed_question'),'target_tokens_max':c.get('target_tokens',8),'final_kl':final,'steps_log':logs,
  'format':'fixed global Gaussian product code, folded F16 key row gain/mean and value scale/minimum only',
  'encoder_book_sha256':hashlib.sha256(Path(c['encoder_book_path']).read_bytes()).hexdigest(),'calibration_ids':[r['id'] for r in cases],
  'packed_forward':c.get('packed_forward',True),'headwise_if_budget':c.get('headwise_if_budget',True),'book_budget_bits':c.get('book_budget_bits',1.8),
  'scope':'teacher-context offline calibration only; actual packed free evaluation on disjoint images required'}
 (args.output/'factors.json').write_text(json.dumps(result,indent=2));(args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','steps':len(logs),'cases':len(cases)}))
if __name__=='__main__':main()
