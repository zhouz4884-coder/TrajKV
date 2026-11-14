"""Independent dense check for the new actual-rate representations."""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from fused_rates import attention
from fused_masked import attention as previous_attention
from fused_progressive import decode_last_generic
from check_fused_progressive import dense_attention,errors,unpack_planes
def main():
 p=argparse.ArgumentParser()
 for k in ('legacy-scripts','kivi-source','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.legacy_scripts))
 from p176_kivi_codec import load_official
 from p201_refinement import quantize
 codec=load_official(a.kivi_source);torch.manual_seed(1337);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(3328,1)):
   b,h,d=1,2,128;vk=torch.randn(b,h,nv,d,device='cuda',dtype=torch.float16);vv=torch.randn_like(vk)
   q=torch.randn(b,8,nq,d,device='cuda',dtype=torch.float16)
   tp=torch.cat((torch.arange(1,40,3,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   state=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(b,h,len(tp),d,device='cuda',dtype=torch.float16),text_v=torch.randn(b,h,len(tp),d,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   for kb,vb in ((1,1),(2,1),(1,2),(2,2),(4,1)):
    enc=[codec.triton_quantize_and_pack_along_last_dim(x,128,bits) for x,bits in ((vk.transpose(2,3).contiguous(),kb),(vv,vb))]
    for axis,rec in zip(('k','v'),enc):
     for name,t in zip(('code','scale','min'),rec):setattr(state,axis+'_'+name,t)
    bk=decode_last_generic(*enc[0],128,kb);bv=decode_last_generic(*enc[1],128,vb)
    for residual_bits in ((0,1,2) if kb in (1,2) else (0,)):
     ref=None;dk=bk
     if residual_bits:
      decoded,encoded=quantize(vk.transpose(2,3).float()-bk.float(),residual_bits)
      ref=list(encoded)+list(encoded);dk=(bk.float()+decoded).half()
     actual=attention(q,state,refinement=ref,bits=kb,bits_v=vb,kr_bits=residual_bits or 2,refine_k=bool(ref),refine_v=False)
     expected,_=dense_attention(q,dk.transpose(2,3).contiguous(),bv,state,None)
     e=errors(actual,expected);ok=e['max_abs']<.002 and e['relative_l2']<.002 and bool(torch.isfinite(actual).all())
     endpoint=None
     if kb==vb and residual_bits in (0,2):
      old=previous_attention(q,state,refinement=ref,bits=kb,refine_k=bool(ref),refine_v=False)
      endpoint=errors(actual,old);ok=ok and endpoint['max_abs']==0
     rows.append({'nv':nv,'nq':nq,'bits_k':kb,'bits_v':vb,'k_residual_bits':residual_bits,'error':e,'previous_endpoint_error':endpoint,'pass':ok})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'thresholds':{'max_abs':.002,'relative_l2':.002},'results':rows,
 'source_sha256':hashlib.sha256(Path(__file__).with_name('fused_rates.py').read_bytes()).hexdigest()}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({k:v for k,v in report.items() if k!='results'}))
 if report['status']!='PASS':raise RuntimeError('Mixed actual rate reader fails')
if __name__=='__main__':main()
