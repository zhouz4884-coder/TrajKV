"""Dense mixed K2/V1+V-sign reference for the static low-storage prototype."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from fused_rates import attention
from fused_progressive import decode_last_generic
from check_fused_progressive import dense_attention,errors
def main():
 p=argparse.ArgumentParser()
 for k in ('legacy-scripts','kivi-source','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);sys.path.insert(0,str(a.legacy_scripts));from p176_kivi_codec import load_official
 from p202_refinement import pack
 codec=load_official(a.kivi_source);torch.manual_seed(836);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(3328,1)):
   k=torch.randn(1,4,nv,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k);q=torch.randn(1,28,nq,128,device='cuda',dtype=torch.float16)
   enc=[codec.triton_quantize_and_pack_along_last_dim(x,128,b) for x,b in ((k.transpose(2,3).contiguous(),2),(v,1))]
   bk=decode_last_generic(*enc[0],128,2).transpose(2,3).contiguous();bv=decode_last_generic(*enc[1],128,1)
   error=(v.float()-bv.float()).reshape(1,4,nv,1,128);alpha=error.abs().mean(-1).half();sign=(error>=0).byte();vr=[pack(sign,1),alpha,alpha];ref=vr+vr
   dv=(bv.float()+((2*sign.float()-1)*alpha.float()[...,None]).reshape(bv.shape)).half()
   tp=torch.cat((torch.arange(1,64,3,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_v=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   for axis,rec in zip(('k','v'),enc):
    for name,t in zip(('code','scale','min'),rec):setattr(s,axis+'_'+name,t)
   actual=attention(q,s,refinement=ref,bits=2,bits_v=1,refine_k=False,refine_v=True)
   expected,_=dense_attention(q,bk,dv,s,None);e=errors(actual,expected);rows.append({'nv':nv,'nq':nq,'error':e,'pass':e['max_abs']<.002 and e['relative_l2']<.002})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':rows};(a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status']}));print(json.dumps(report))
 if report['status']!='PASS':raise RuntimeError('Mixed-axis V residual fails')
if __name__=='__main__':main()
