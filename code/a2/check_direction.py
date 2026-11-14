"""Independent dense rotated-space checks for packed sign and VQ key readers."""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'));sys.path.insert(0,str(ROOT/'third_party/reference_ports'))
import torch
from kvarn_reference import _hadamard
from quantizers import pack1,affine_rtn
from fused_direction import attention
from fused_progressive import decode_last_generic
from check_fused_progressive import errors
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(191);torch.backends.cuda.matmul.allow_tf32=False;rows=[];hadamard=_hadamard(128,'cuda:0')
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(1792,1)):
   q=torch.randn(1,nq,28,128,device='cuda',dtype=torch.float16).transpose(1,2);tp=torch.cat((torch.arange(1,64,2,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_v=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq,
    direction_mean=torch.randn(1,4,1,128,device='cuda',dtype=torch.float16),direction_gain=(.5+torch.rand(1,4,nv,1,device='cuda')).half())
   original_v=torch.randn(1,4,nv,128,device='cuda',dtype=torch.float16)
   for dim in (0,4,8):
    book=None if dim==0 else torch.randn(256,dim,device='cuda',dtype=torch.float16)
    if dim==0:
     bits=torch.randint(0,2,(1,4,nv,128),device='cuda',dtype=torch.int32);s.direction_code=pack1(bits);rotated=((2*bits.float()-1)*s.direction_gain.float()).half()
    else:
     s.direction_code=torch.randint(0,256,(1,4,nv,128//dim),device='cuda',dtype=torch.uint8);rotated=(book[s.direction_code.long()].reshape(1,4,nv,128).float()*s.direction_gain.float()).half()
    for vb,channel in ((1,True),(2,False),(2,True),(16,False)):
     if vb==16:s.visual_v=original_v;values=original_v
     else:
      data=original_v.transpose(2,3).contiguous() if channel else original_v;rec=affine_rtn(data,vb,128)
      for name,t in zip(('code','scale','min'),rec):setattr(s,'v_'+name,t)
      values=decode_last_generic(*rec,128,vb);values=values.transpose(2,3).contiguous() if channel else values
     for metric in (False,True):
      scale=(.5+torch.rand(1,4,1,128,device='cuda')).half() if metric else None
      actual=attention(q,s,hadamard,book,scale,vb,channel,128)
      qr=((q.float() if scale is None else q.float()/scale.repeat_interleave(7,1).float())@hadamard).half().float()
      zv=qr@rotated.float().repeat_interleave(7,1).transpose(-1,-2)/128**.5+(q.float()*s.direction_mean.repeat_interleave(7,1).float()).sum(-1,keepdim=True)/128**.5
      zt=q.float()@s.text_k.float().repeat_interleave(7,1).transpose(-1,-2)/128**.5;z=torch.cat((zv,zt),-1)
      pos=torch.cat((s.visual_pos,tp));qp=torch.arange(s.length-nq,s.length,device='cuda');z.masked_fill_(pos[None,None,None,:]>qp[None,None,:,None],-torch.inf)
      vv=torch.cat((values,s.text_v),2).repeat_interleave(7,1);expected=(z.softmax(-1)@vv.float()).half();e=errors(actual,expected)
      rows.append({'nv':nv,'nq':nq,'book_dimension':dim,'value_bits':vb,'value_channel':channel,'query_metric':metric,'error':e,'pass':e['max_abs']<.002 and e['relative_l2']<.002 and bool(torch.isfinite(actual).all())})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'results':rows,'kernel_sha256':hashlib.sha256(Path(__file__).with_name('fused_direction.py').read_bytes()).hexdigest(),
  'scope':'same-representation dense rotated-space arithmetic; original-frame inverse-rotation FP16 rounding is a separate model-level comparison'}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({'status':report['status'],'cases':len(rows)}))
 if report['status']!='PASS':raise RuntimeError('Direction reader mismatch')
if __name__=='__main__':main()
