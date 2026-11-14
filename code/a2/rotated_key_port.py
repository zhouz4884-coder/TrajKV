"""Full-model quality prototype: centered residual, rotation, one-bit direction.

The prototype materializes reconstructed visual KV. Its actual owner is counted
and it is explicitly excluded from low-bit deployment claims. Packed runtime
is warranted only if this finite mechanism experiment is promising.
"""
import json,math,sys,torch
from pathlib import Path
from masked_bank import A2Bank,unique_bytes
from quantizers import pack1,binary_centroids
from fused_progressive import decode_last_generic
ROOT=Path(__file__).resolve().parents[2]

class RotatedKeyPort(A2Bank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  super()._initialize(model,cache,visual,'full',legacy_scripts,kivi_source,False)
  sys.path.insert(0,str(ROOT/'third_party/reference_ports'))
  from kvarn_reference import _hadamard
  cfg=type(self).settings;self.rotation=_hadamard(128,str(self.states[0].visual_k.device)) if cfg.get('rotate',False) else None
  self.query_scales=None
  if cfg.get('query_scale_path'):
   calibrated=json.loads(Path(cfg['query_scale_path']).read_text())
   self.query_scales=torch.tensor(calibrated['rms_scales'],device=self.states[0].visual_k.device,dtype=torch.float32).pow(cfg.get('query_scale_power',1.)).half().contiguous()
   if self.query_scales.shape!=(28,4,128) or not bool((self.query_scales>0).all()):raise ValueError('Invalid query calibration')
  self.direction_records=[];self.value_records=[];self.center_statistics=[]
  self.direction_book=None
  if cfg.get('direction_book_path'):
   book=json.loads(Path(cfg['direction_book_path']).read_text());self.direction_book=torch.tensor(book['centers'],device=self.states[0].visual_k.device,dtype=torch.float16).contiguous()
   if self.direction_book.shape[0]!=256 or 128%self.direction_book.shape[1]:raise ValueError('Invalid direction book')
  for layer,s in enumerate(self.states):
   x=s.visual_k.float();mu=x.mean(-2,keepdim=True).half() if cfg.get('center',False) else None
   residue=x if mu is None else x-mu.float();scale=1. if self.query_scales is None else self.query_scales[layer][None,:,None,:].float()
   transformed=residue*scale;z=transformed if self.rotation is None else transformed@self.rotation
   nr=z.norm(dim=-1,keepdim=True);sign=z>=0
   if cfg.get('roundtrip_control',False):reconstructed=z;record=[]
   elif self.direction_book is not None:
    initial=(nr/math.sqrt(z.shape[-1])).clamp_min(1e-20);subvectors=(z/initial).reshape(-1,self.direction_book.shape[1])
    book=self.direction_book.float();selected=[]
    for start in range(0,len(subvectors),65536):
     part=subvectors[start:start+65536];distance=part.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*(part@book.T)
     selected.append(distance.argmin(-1).to(torch.uint8))
    index=torch.cat(selected).reshape(*z.shape[:-1],128//book.shape[-1]).contiguous();template=book[index.long()].reshape(z.shape)
    mode=cfg.get('gain','norm');dot=(z*template).sum(-1,keepdim=True)
    if mode=='norm':gain=nr/template.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    elif mode=='projection':gain=torch.where(dot>1e-12,nr.square()/dot.clamp_min(1e-12),nr/template.norm(dim=-1,keepdim=True).clamp_min(1e-20))
    else:gain=dot/template.square().sum(-1,keepdim=True).clamp_min(1e-20)
    gain=gain.half().contiguous()
    reconstructed=template*gain.float();record=[index,gain]+([mu] if mu is not None else [])
   else:
    mode=cfg.get('gain','norm')
    gain=(nr/math.sqrt(z.shape[-1]) if mode=='norm' else nr.square()/z.abs().sum(-1,keepdim=True).clamp_min(1e-20) if mode=='projection' else z.abs().mean(-1,keepdim=True)).half().contiguous()
    code=pack1(sign);record=[code,gain]+([mu] if mu is not None else [])
    # Decode the actual packed words, not the pre-pack Boolean array.
    shift=torch.arange(32,device=x.device);decoded=((code.long()[...,None]>>shift)&1).reshape(z.shape)
    reconstructed=(2*decoded.float()-1)*gain.float()
   reconstructed=reconstructed if self.rotation is None else reconstructed@self.rotation.T
   reconstructed=reconstructed/scale
   if mu is not None:reconstructed=reconstructed+mu.float()
   self.direction_records.append(record);s.visual_k=reconstructed.half().contiguous()
   if cfg.get('bits_v')==2:
    channel=cfg.get('v_layout','token')=='channel';data=s.visual_v.transpose(2,3).contiguous() if channel else s.visual_v
    rec=self.codec.triton_quantize_and_pack_along_last_dim(data,128,2);self.value_records.append(list(rec));decoded=decode_last_generic(*rec,128,2)
    s.visual_v=decoded.transpose(2,3).contiguous() if channel else decoded
   elif cfg.get('v_encoder')=='lloyd':
    rec=binary_centroids(s.visual_v.transpose(2,3).contiguous());self.value_records.append(list(rec))
    s.visual_v=decode_last_generic(*rec,128,1).transpose(2,3).contiguous()
   else:self.value_records.append([])
   normx=x.norm(dim=-1);normres=residue.norm(dim=-1)
   self.center_statistics.append({'layer':layer,'mean_residual_energy_fraction':float(residue.square().sum()/x.square().sum()),
    'key_norm_mean':float(normx.mean()),'residual_norm_mean':float(normres.mean()),
    'key_mse':float((x-s.visual_k.float()).square().mean()),'mean_key_cosine':float(torch.nn.functional.cosine_similarity(x,s.visual_k.float(),dim=-1).mean())})
  self.method=method

 def owned(self):
  r=super().owned();t=[v for group in getattr(self,'direction_records',[]) for v in group]+[v for group in getattr(self,'value_records',[]) for v in group]
  if getattr(self,'rotation',None) is not None:t.append(self.rotation)
  if getattr(self,'query_scales',None) is not None:t.append(self.query_scales)
  if getattr(self,'direction_book',None) is not None:t.append(self.direction_book)
  n=unique_bytes(t);r['rotation_packed_reference']=n;r['total_unique']+=n;r['runtime']='materialized_center_rotation_quality_prototype';return r

 def result_metadata(self):
  r=super().result_metadata();r.update(backend='centered residual + orthogonal rotation + sign quality prototype',
   materialized_quality_port=True,lowbit_deployment_candidate=False,center_rotation_statistics=self.center_statistics,
   prototype_bits_scope='actual owner includes FP16 materialization and packed reference; no packed efficiency claim',
   direction_gain_rule=('transformed-space self-inner-product matching' if type(self).settings.get('gain')=='projection' else
                        'transformed residual norm matching' if type(self).settings.get('gain','norm')=='norm' else
                        'least squares for a VQ template' if self.direction_book is not None else 'mean absolute, MSE-optimal fixed-sign amplitude'))
  return r
