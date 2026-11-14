"""Independent dense checks for channel V2 and global-channel V1+residual."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from fused_rates import attention
from quantizers import binary_centroids,mean_abs_levels,affine_rtn
from fused_progressive import decode_last_generic
from check_fused_progressive import dense_attention,errors
def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-scripts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--key-bits',type=int,default=2);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 sys.path.insert(0,str(a.legacy_scripts));from p202_refinement import pack
 torch.manual_seed(1427);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(1792,1)):
   k=torch.randn(1,4,nv,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k);q=torch.randn(1,28,nq,128,device='cuda',dtype=torch.float16)
   ek=affine_rtn(k.transpose(2,3).contiguous(),a.key_bits,128);bk=decode_last_generic(*ek,128,a.key_bits).transpose(2,3).contiguous()
   tp=torch.cat((torch.arange(1,64,3,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_v=torch.randn(1,4,len(tp),128,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   for mode in ('v2_channel','v1_global','v1_global_residual'):
    group=128 if mode=='v2_channel' else nv;bits=2 if mode=='v2_channel' else 1
    ev=affine_rtn(v.transpose(2,3).contiguous(),bits,group) if bits==2 else mean_abs_levels(v.transpose(2,3).contiguous(),group_size=group)
    bv=decode_last_generic(*ev,group,bits).transpose(2,3).contiguous();ref=None;res=mode.endswith('residual');target=bv
    if res:
     error=(v.float()-bv.float()).reshape(1,4,nv,1,128);alpha=error.abs().mean(-1).half();sign=(error>=0).byte();vr=[pack(sign,1),alpha,alpha];ref=vr+vr
     target=(bv.float()+((2*sign.float()-1)*alpha.float()[...,None]).reshape(bv.shape)).half()
    for axis,rec in zip(('k','v'),(ek,ev)):
     for name,t in zip(('code','scale','min'),rec):setattr(s,axis+'_'+name,t)
    actual=attention(q,s,bits=a.key_bits,bits_v=bits,refinement=ref,refine_k=False,refine_v=res,v_channel=True,v_group_size=group)
    expected,_=dense_attention(q,bk,target,s,None);e=errors(actual,expected);rows.append({'nv':nv,'nq':nq,'mode':mode,'error':e,'pass':e['max_abs']<.002 and e['relative_l2']<.002})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'results':rows};(a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({'status':report['status'],'cases':len(rows)}))
 if report['status']!='PASS':raise RuntimeError('Coverage reader mismatch')
if __name__=='__main__':main()
