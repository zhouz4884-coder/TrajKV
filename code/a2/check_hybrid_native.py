"""Independent same-representation check for native K2 + clipped channel V1."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from calib_bank import load_official
from fused_hybrid import attention
from fused_progressive import decode_last_generic
from check_hybrid import dense
from check_fused_progressive import errors
def main():
 p=argparse.ArgumentParser()
 for k in ('legacy-scripts','kivi-source','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);sys.path.insert(0,str(a.legacy_scripts));from p176_kivi_codec import load_official as load_kivi
 codec=load_kivi(a.kivi_source);m=load_official(a.legacy_scripts);encode=getattr(m.encode,'_torchdynamo_orig_callable',m.encode)
 torch.manual_seed(417);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(1280,33),(3328,1)):
   hkv,hq,d=4,28,128;x=torch.randn(1,hkv,nv,d,device='cuda',dtype=torch.float16);v=torch.randn_like(x);q=torch.randn(1,hq,nq,d,device='cuda',dtype=torch.float16)
   rec=codec.triton_quantize_and_pack_along_last_dim(x.transpose(2,3).contiguous(),128,2);keys=decode_last_generic(*rec,128,2).transpose(2,3).contiguous()
   vc,vs,vz=m.quant(v.transpose(1,2),1,per_channel=True,scale=.3,post_scale=True);packed=encode(vc.transpose(1,2).long(),bits=1,datum_per_elem=8)
   values=(vs*(vc-vz)).transpose(1,2).contiguous();vr=(packed,vs.transpose(1,2).contiguous(),vz.transpose(1,2).contiguous())
   tp=torch.cat((torch.arange(1,128,3,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,hkv,len(tp),d,device='cuda',dtype=torch.float16),text_v=torch.randn(1,hkv,len(tp),d,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   actual=attention(q,s,rec,vr,native_k=True);expected=dense(q,q,keys,values,s);e=errors(actual,expected)
   rows.append({'nv':nv,'nq':nq,'error':e,'pass':bool(torch.isfinite(actual).all()) and e['max_abs']<.002 and e['relative_l2']<.002})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':rows,'thresholds':{'max_abs':.002,'relative_l2':.002},'kernel_sha256':hashlib.sha256(Path(__file__).with_name('fused_hybrid.py').read_bytes()).hexdigest()}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps(report))
 if report['status']!='PASS':raise RuntimeError('Native2/clip1 kernel fails')
if __name__=='__main__':main()
