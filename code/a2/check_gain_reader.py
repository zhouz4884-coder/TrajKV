"""GPU same-representation tests for grouped/global MAD, shared/per-head gain."""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from quantizers import mean_abs_levels
from fused_progressive import decode_last_generic,attention as frozen_attention
from fused_rates import attention
from fused_masked import attention as raw_attention
from check_fused_progressive import dense_attention,errors
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(768);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(1280,33)):
   k=torch.randn(1,4,nv,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k);q=torch.randn(1,28,nq,128,device='cuda',dtype=torch.float16)
   tp=torch.cat((torch.arange(1,64,3,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_v=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   for group in (128,nv):
    ek=mean_abs_levels(k.transpose(2,3).contiguous(),group_size=group);bk=decode_last_generic(*ek,group,1)
    center=(ek[2].float()+.5*ek[1].float()).repeat_interleave(group,-1)
    for channel in (False,True):
     vg=group if channel else 128;ev=mean_abs_levels(v.transpose(2,3).contiguous() if channel else v,group_size=vg)
     bv=decode_last_generic(*ev,vg,1);bv=bv.transpose(2,3).contiguous() if channel else bv
     for axis,rec in zip(('k','v'),(ek,ev)):
      for name,t in zip(('code','scale','min'),rec):setattr(s,axis+'_'+name,t)
     for mode in ('none','head','shared','origin'):
      gain=None
      if mode!='none':gain=(.5+torch.rand(1,1 if mode=='shared' else 4,nv,device='cuda')).half().contiguous()
      dk=bk if gain is None else ((bk.float()*gain[:,:,None,:]).half() if mode=='origin' else (center+(bk.float()-center)*gain[:,:,None,:]).half())
      actual=attention(q,s,bits=1,bits_v=1,refine_k=False,refine_v=False,v_channel=channel,k_gain=gain,k_group_size=group,v_group_size=vg,gain_centered=mode!='origin')
      expected,_=dense_attention(q,dk.transpose(2,3).contiguous(),bv,s,None);e=errors(actual,expected)
      endpoint=None
      if group==128 and not channel and mode=='none':endpoint=errors(actual,frozen_attention(q,s,bits=1,refine_k=False,refine_v=False))
      ok=bool(torch.isfinite(actual).all()) and e['max_abs']<.002 and e['relative_l2']<.002 and (endpoint is None or endpoint['max_abs']==0)
      rows.append({'nv':nv,'nq':nq,'group':group,'v_channel':channel,'gain':mode,'error':e,'default_endpoint':endpoint,'pass':ok})
   # Segment-specific temperature: independent dense score scaling, raw path.
   view=SimpleNamespace(visual_k=k,visual_v=v,visual_pos=s.visual_pos,text_k=s.text_k,text_v=s.text_v,text_pos=tp,text_length=len(tp),length=s.length)
   for beta in (.26,1.):
    actual=raw_attention(q,view,visual_logit_gain=beta)
    kk=torch.cat((k,s.text_k),2).repeat_interleave(7,1);vv=torch.cat((v,s.text_v),2).repeat_interleave(7,1)
    score=q.float()@kk.float().transpose(-1,-2)/128**.5;score[...,:nv]*=beta
    pos=torch.cat((s.visual_pos,tp));qp=torch.arange(s.length-nq,s.length,device='cuda')
    score.masked_fill_(pos[None,None,None,:]>qp[None,None,:,None],-torch.inf)
    expected=(score.softmax(-1)@vv.float()).half();e=errors(actual,expected)
    rows.append({'nv':nv,'nq':nq,'visual_logit_gain':beta,'error':e,'pass':e['max_abs']<.002 and e['relative_l2']<.002})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'results':rows,'kernel_sha256':hashlib.sha256(Path(__file__).with_name('fused_rates.py').read_bytes()).hexdigest()}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({'status':report['status'],'cases':len(rows)}))
 if report['status']!='PASS':raise RuntimeError('MAD/gain reader numeric mismatch')
if __name__=='__main__':main()
