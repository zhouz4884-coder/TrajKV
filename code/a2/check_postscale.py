"""Independent dense FP32 affine reconstruction, causal gaps and large offsets."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from quantizers import affine_rtn
from fused_postscale import attention
from check_fused_progressive import errors
def decode32(rec,group,bits,shape):
 code,scale,minimum=rec;shift=torch.arange(32//bits,device=code.device)*bits
 values=((code.long()[...,None]>>shift)&((1<<bits)-1)).reshape(shape).float()
 return values*scale.float().repeat_interleave(group,-1)+minimum.float().repeat_interleave(group,-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(181);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(1792,1)):
   for shifted in (False,True):
    bias=torch.randn(1,4,1,128,device='cuda')*25 if shifted else 0.
    k=(torch.randn(1,4,nv,128,device='cuda')*(.05 if shifted else 1.)+bias).half();v=(torch.randn_like(k)*(.05 if shifted else 1.)+bias*.4).half();q=torch.randn(1,28,nq,128,device='cuda',dtype=torch.float16)
    tp=torch.cat((torch.arange(1,64,2,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
    s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=(torch.randn(1,4,len(tp),128,device='cuda')+bias).half(),text_v=(torch.randn(1,4,len(tp),128,device='cuda')+bias*.4).half(),text_length=len(tp),length=2*nv+nq)
    for kb,vb,channel,global_group in ((1,1,True,False),(2,1,True,False),(1,2,False,False),(2,2,False,False),(1,1,True,True),(2,2,True,True)):
     kg=nv if global_group else 128;vg=nv if global_group else 128;kd=k.transpose(2,3).contiguous();vd=v.transpose(2,3).contiguous() if channel else v
     ek=affine_rtn(kd,kb,kg);ev=affine_rtn(vd,vb,vg)
     for axis,rec in zip(('k','v'),(ek,ev)):
      for name,t in zip(('code','scale','min'),rec):setattr(s,axis+'_'+name,t)
     actual=attention(q,s,bits=kb,bits_v=vb,v_channel=channel,k_group_size=kg,v_group_size=vg)
     dk=decode32(ek,kg,kb,kd.shape).transpose(2,3);dv=decode32(ev,vg,vb,vd.shape);dv=dv.transpose(2,3) if channel else dv
     kk=torch.cat((dk,s.text_k.float()),2).repeat_interleave(7,1);vv=torch.cat((dv,s.text_v.float()),2).repeat_interleave(7,1)
     score=q.float()@kk.transpose(-1,-2)/128**.5;pos=torch.cat((s.visual_pos,tp));qp=torch.arange(s.length-nq,s.length,device='cuda')
     score.masked_fill_(pos[None,None,None,:]>qp[None,None,:,None],-torch.inf);expected=(score.softmax(-1)@vv).half();e=errors(actual,expected)
     threshold=max(.003,.003*float(expected.abs().max()));ok=bool(torch.isfinite(actual).all()) and e['max_abs']<=threshold and e['relative_l2']<.003
     rows.append({'nv':nv,'nq':nq,'shifted':shifted,'bits_k':kb,'bits_v':vb,'channel_v':channel,'global_group':global_group,'error':e,'max_abs_threshold':threshold,'pass':ok})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'results':rows,'kernel_sha256':hashlib.sha256(Path(__file__).with_name('fused_postscale.py').read_bytes()).hexdigest(),
  'scope':'FP32 affine decoding reference; FP16 q-scale/TensorCore probability rounding allowed, .003 relative tolerance; large-offset absolute tolerance scales with FP16 output magnitude'}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({'status':report['status'],'cases':len(rows)}))
 if report['status']!='PASS':raise RuntimeError('Postscale reader differs from independent affine reference')
if __name__=='__main__':main()
