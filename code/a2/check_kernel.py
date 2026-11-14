"""Independent dense mixed-representation check, including causal position gaps."""
import argparse,json,sys,itertools,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from check_fused_progressive import residual_r2,pack_planes,unpack_planes,dense_attention,errors
from fused_progressive import attention as prior_attention,decode_last_generic
from fused_masked import attention
def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-scripts',type=Path,required=True);p.add_argument('--kivi-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.legacy_scripts));from p176_kivi_codec import load_official
 codec=load_official(a.kivi_source);torch.manual_seed(20261113);torch.backends.cuda.matmul.allow_tf32=False
 a.output.mkdir(exist_ok=False,parents=True);rows=[]
 with torch.inference_mode():
  for nv,nq in [(256,1),(768,17),(1280,1),(4352,17)]:
   hkv,hq,d=2,8,128;device='cuda'
   vk=torch.randn(1,hkv,nv,d,device=device,dtype=torch.float16);vv=torch.randn_like(vk)
   q=torch.randn(1,hq,nq,d,device=device,dtype=torch.float16)
   # Removed history positions leave gaps; logical length never shrinks.
   tp=torch.cat([torch.arange(1,65,3,device=device),torch.arange(2*nv,2*nv+nq,device=device)])
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device=device),text_pos=tp,
    text_k=torch.randn(1,hkv,len(tp),d,device=device,dtype=torch.float16),text_v=torch.randn(1,hkv,len(tp),d,device=device,dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   for bits in (1,2):
    enc=[codec.triton_quantize_and_pack_along_last_dim(x,128,bits) for x in (vk.transpose(2,3).contiguous(),vv)]
    for axis,rec in zip(('k','v'),enc):
     for suffix,t in zip(('code','scale','min'),rec):setattr(s,axis+'_'+suffix,t)
    bk,bv=[decode_last_generic(*r,128,bits) for r in enc]
    kr,ks,km=residual_r2(vk.transpose(2,3).float()-bk.float());ve=(vv.float()-bv.float()).reshape(1,hkv,nv,1,128)
    va=ve.abs().mean(-1).half();vr=pack_planes((ve>=0).byte(),1);ref=[kr,ks,km,vr,va,va]
    kd=(unpack_planes(kr,2,(1,hkv,d,nv)).reshape(1,hkv,d,-1,128)*ks.float()[...,None]+km.float()[...,None]).reshape(bk.shape)
    vd=((2*unpack_planes(vr,1,(1,hkv,nv,d)).reshape(1,hkv,nv,1,128)-1)*va.float()[...,None]).reshape(bv.shape)
    for page in (128,256):
     for kind in ('empty','full','random'):
      mask=torch.zeros((1,hkv,nv//page),device=device,dtype=torch.uint8)
      if kind=='full':mask.fill_(1)
      elif kind=='random':mask.random_(0,2)
      take=mask.repeat_interleave(page,-1).bool()[...,None]
      for rk,rv in ((True,True),(True,False),(False,True)):
       dk=torch.where(take,(bk.float()+kd).half().transpose(2,3),bk.transpose(2,3)) if rk else bk.transpose(2,3)
       dv=torch.where(take,(bv.float()+vd).half(),bv) if rv else bv
       actual=attention(q,s,refinement=ref,bits=bits,refine_k=rk,refine_v=rv,refinement_mask=mask,page_size=page)
       expected,_=dense_attention(q,dk.contiguous(),dv,s,None)
       e=errors(actual,expected);ok=bool(torch.isfinite(actual).all()) and e['max_abs']<0.002 and e['relative_l2']<0.002
       endpoint=None
       if kind in ('empty','full'):
        prior=prior_attention(q,s,refinement=ref if kind=='full' else None,bits=bits,refine_k=rk,refine_v=rv)
        endpoint=errors(actual,prior);ok=ok and endpoint['max_abs']==0
       rows.append({'nv':nv,'nq':nq,'bits':bits,'page':page,'mask':kind,'refine_k':rk,'refine_v':rv,'dense_error':e,'endpoint_error':endpoint,'pass':ok})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'results':rows,
  'thresholds':{'max_abs':0.002,'relative_l2':0.002,'endpoint_abs':0},'source_sha256':hashlib.sha256(Path(__file__).with_name('fused_masked.py').read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name()}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({k:v for k,v in report.items() if k!='results'}))
 if report['status']!='PASS':raise RuntimeError('Mixed reader fails independent dense correctness')
if __name__=='__main__':main()
