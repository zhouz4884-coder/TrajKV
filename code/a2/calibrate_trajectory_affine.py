# Derived from frozen first-token calibration; independent next-phase program.
"""Feasibility/label-free calibration of folded head gains through real queries.

Model weights are frozen. Student question states form through the quantized
image prefix; positive head factors and optional center offsets are optimized against FP16 logits.
Calibration uses dense differentiable caches; final quality must be evaluated
with the actual packed runtime, never with this calibration representation.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
from transformers.cache_utils import DynamicCache
import session_boundary
from fused_progressive import decode_last_generic
from bank import install_legacy
from quantizers import binary_centroids

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(exist_ok=False,parents=True)
 arguments=c['arguments'];install_legacy(arguments['legacy_scripts']);from p180_multiturn_core import attention_bridge
 from p176_kivi_codec import load_official
 codec=load_official(Path(arguments['kivi_source']));torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 (a.output/'manifest.json').write_text(json.dumps({'config':c,'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
  'samples_sha256':hashlib.sha256(Path(arguments['samples']).read_bytes()).hexdigest(),'labels_used':False,'actual_cap':8,
  'scope':'frozen-model quantizer-parameter calibration; actual GPU model; dense differentiable caches, not serving efficiency',
  'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
 processor=AutoProcessor.from_pretrained(arguments['model'],local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(arguments['model'],local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 for param in model.parameters():param.requires_grad_(False)
 rows=json.loads(Path(arguments['samples']).read_text())[:c.get('images',1)];cases=[];kbits=c.get('bits_k',1);vbits=c.get('bits_v',2);train_value=c.get('train_value',False)
 def unpack(rec,bits,shape):
  shift=torch.arange(32//bits,device=rec[0].device)*bits
  return ((rec[0].long()[...,None]>>shift)&((1<<bits)-1)).reshape(shape).half()
 with torch.no_grad():
  for row in rows:
   f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in session_boundary.features(processor,row,8).items()}
   visual=torch.nonzero(f['input_ids'][0]==model.config.image_token_id).flatten()
   boundary=f['input_ids'].shape[-1]-1 if c.get('mode')=='native_question_history' else int(visual[-1])+1
   suffix=f['input_ids'][:,boundary:].clone()
   with attention_bridge(model,native=True):out=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
   teacher_cache=out.past_key_values;teacher_ids=[];teacher_lp=[]
   import run_p175_modern_replication as oldbase
   stop=oldbase.stop_ids(model,processor);prompt_length=f['input_ids'].shape[-1]
   for target_step in range(c.get('target_tokens',8)):
    teacher_lp.append(out.logits[:,-1].float().log_softmax(-1).cpu());token=int(out.logits[:,-1].argmax(-1));teacher_ids.append(token)
    if token in stop or target_step+1>=c.get('target_tokens',8):break
    with attention_bridge(model,native=True):
     out=model(input_ids=torch.tensor([[token]],device=model.device),position_ids=torch.tensor([[prompt_length+target_step]],device=model.device),past_key_values=teacher_cache,use_cache=True,return_dict=True,logits_to_keep=1)
   target=torch.stack(teacher_lp,1)
   if len(teacher_ids)>1:suffix=torch.cat((suffix,torch.tensor([teacher_ids[:-1]],device=suffix.device)),1)
   layers=[]
   for layer in teacher_cache.layers:
    k=layer.keys[:,:,:boundary].clone();v=layer.values[:,:,:boundary].clone();x=k.index_select(2,visual).transpose(2,3).contiguous()
    rec=codec.triton_quantize_and_pack_along_last_dim(x,128,kbits);base=decode_last_generic(*rec,128,kbits).float();center=rec[2].float()+.5*((1<<kbits)-1)*rec[1].float();expanded=center.repeat_interleave(128,-1)
    beta=((x.float()-expanded).square().mean((-2,-1),keepdim=True)/(base-expanded).square().mean((-2,-1),keepdim=True).clamp_min(1e-20)).sqrt()
    if not c.get('base_rms',True):beta=torch.ones_like(beta)
    codes=unpack(rec,kbits,x.shape)
    vx=v.index_select(2,visual);vdata=vx.transpose(2,3).contiguous() if vbits==1 else vx
    vr=binary_centroids(vdata) if vbits==1 else codec.triton_quantize_and_pack_along_last_dim(vdata,128,vbits)
    decoded=decode_last_generic(*vr,128,vbits);v[:,:,visual]=decoded.transpose(2,3).contiguous() if vbits==1 else decoded
    layer_data={'source_key':k.cpu(),'source_value':v.cpu(),'codes':codes.cpu(),'scale':rec[1].cpu(),'center':center.cpu(),'beta':beta.cpu()}
    if train_value:
     if vbits!=1:raise ValueError('Joint value calibration currently uses channel V1')
     layer_data.update(value_codes=unpack(vr,1,vdata.shape).cpu(),value_scale=vr[1].cpu(),value_center=(vr[2].float()+.5*vr[1].float()).cpu())
    layers.append(layer_data)
   cases.append({'id':row['id'],'image_id':row['image_id'],'visual':visual.cpu(),'boundary':boundary,'suffix':suffix.cpu(),'target':target,'layers':layers})
   del out,f,k,v,x,base,rec,vr,vx,layers
 factors=torch.nn.Parameter(torch.zeros(28,4,device='cuda',dtype=torch.float32));value_factors=torch.nn.Parameter(torch.zeros_like(factors)) if train_value else None
 train_center=c.get('train_center',False)
 key_centers=torch.nn.Parameter(torch.zeros_like(factors)) if train_center else None
 value_centers=torch.nn.Parameter(torch.zeros_like(factors)) if train_center and train_value else None
 params=[x for x in (factors,value_factors,key_centers,value_centers) if x is not None]
 opt=torch.optim.Adam(params,lr=c.get('learning_rate',.02));records=[]
 def loss_for(case):
  cache=DynamicCache(config=model.config.get_text_config());visual=case['visual'].to('cuda')
  for i,r in enumerate(case['layers']):
   code=r['codes'].to('cuda');scale=r['scale'].to('cuda').float();center=r['center'].to('cuda').float();beta=r['beta'].to('cuda').float()*factors[i][None,:,None,None].exp()
   new_scale=(scale*beta).half();offset=key_centers[i][None,:,None,None] if train_center else 0.
   minimum=(center+(offset-.5*((1<<kbits)-1))*new_scale.float()).half()
   kh=((code*new_scale.repeat_interleave(128,-1)).half()+minimum.repeat_interleave(128,-1)).half().transpose(2,3).contiguous()
   k=r['source_key'].to('cuda').clone();k[:,:,visual]=kh;v=r['source_value'].to('cuda')
   if train_value:
    vs=(r['value_scale'].to('cuda').float()*value_factors[i][None,:,None,None].exp()).half();offset=value_centers[i][None,:,None,None] if train_center else 0.
    vm=(r['value_center'].to('cuda')+(offset-.5)*vs.float()).half()
    vh=((r['value_codes'].to('cuda')*vs.repeat_interleave(128,-1)).half()+vm.repeat_interleave(128,-1)).half().transpose(2,3).contiguous()
    v=v.clone();v[:,:,visual]=vh
   cache.update(k,v,i)
  suffix=case['suffix'].to('cuda');position=torch.arange(case['boundary'],case['boundary']+suffix.shape[-1],device='cuda')[None]
  out=model(input_ids=suffix,position_ids=position,past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=case['target'].shape[1])
  lp=out.logits.float().log_softmax(-1);target=case['target'].to('cuda');return (target.exp()*(target-lp)).sum(-1).mean()
 for step in range(c.get('steps',3)):
  case=cases[step%len(cases)];opt.zero_grad(set_to_none=True);start=time.perf_counter();loss=loss_for(case)
  if not loss.requires_grad:raise RuntimeError('Cache path detached calibration parameters')
  # Fixed loss scaling preserves small FP16 activation gradients. Parameters and
  # optimizer state remain FP32; no pretrained parameter receives a gradient.
  penalty=sum(parameter.square().mean() for parameter in params)
  objective=loss+c.get('regularization',0.)*penalty
  (objective*128).backward()
  for parameter in params:parameter.grad.div_(128)
  grad=float(factors.grad.norm());vgrad=float(value_factors.grad.norm()) if train_value else 0.;finite=bool(torch.isfinite(factors.grad).all()) and (not train_value or bool(torch.isfinite(value_factors.grad).all()))
  finite=finite and all(bool(torch.isfinite(parameter.grad).all()) for parameter in params)
  if not finite or grad==0:raise RuntimeError('Invalid affine-parameter gradient')
  opt.step()
  with torch.no_grad():
   factors.clamp_(-.7,.7)
   if train_value:value_factors.clamp_(-.7,.7)
   if train_center:key_centers.clamp_(-.35,.35)
   if value_centers is not None:value_centers.clamp_(-.35,.35)
  torch.cuda.synchronize();record={'step':step,'id':case['id'],'kl_before_update':float(loss.detach()),'gradient_norm':grad,'value_gradient_norm':vgrad,'seconds':time.perf_counter()-start}
  records.append(record);print(json.dumps(record),flush=True);del loss
 with torch.no_grad():final_losses=[float(loss_for(case)) for case in cases]
 result={'status':'COMPLETE','steps':len(records),'cases':len(cases),'images':len({r['image_id'] for r in cases}),'model_weights_frozen':True,'labels_used':False,
  'parameter_count':sum(parameter.numel() for parameter in params),
  'key_center_offsets':key_centers.detach().cpu().tolist() if train_center else None,
  'value_center_offsets':value_centers.detach().cpu().tolist() if value_centers is not None else None,
  'train_center':train_center,'head_factors':factors.detach().exp().cpu().tolist(),
  'value_head_factors':value_factors.detach().exp().cpu().tolist() if train_value else None,'bits_k':kbits,'bits_v':vbits,'calibration_ids':[r['id'] for r in cases],
  'calibration_mode':c.get('mode','compressed_question'),'steps_log':records,'final_kl':final_losses,
  'scope':'next-phase unlabelled teacher answer trajectory calibration; dense differentiable student; no task accuracy or packed efficiency claim','target_tokens_max':c.get('target_tokens',8),'base_rms':c.get('base_rms',True),'center_offset_scope':'per layer/KV head, fraction of the final metadata scale, codes unchanged'}
 (a.output/'factors.json').write_text(json.dumps(result,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','steps':len(records),'images':len(cases)}))
if __name__=='__main__':main()
