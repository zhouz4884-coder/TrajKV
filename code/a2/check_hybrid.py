"""Dense rotated-space and original-frame checks for the packed combination."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'));sys.path.insert(0,str(ROOT/'third_party/reference_ports'))
import torch
from kvarn_reference import encode_kvarn,decode_kvarn,_hadamard,_upstream
from calib_bank import load_official
from fused_hybrid import attention
from check_fused_progressive import errors
def dense(q,qr,kr,values,state,original=None):
 g=q.shape[1]//kr.shape[1];nv=kr.shape[-2]
 z=(qr.float()@kr.repeat_interleave(g,1).float().transpose(-1,-2) if original is None else q.float()@original.repeat_interleave(g,1).float().transpose(-1,-2))
 zt=q.float()@state.text_k.repeat_interleave(g,1).float().transpose(-1,-2)
 scores=torch.cat((z,zt),-1)/q.shape[-1]**.5
 pos=torch.cat((state.visual_pos,state.text_pos));qp=torch.arange(state.length-q.shape[-2],state.length,device=q.device)
 scores.masked_fill_(pos[None,None,None,:]>qp[None,None,:,None],-torch.inf)
 v=torch.cat((values,state.text_v),2).repeat_interleave(g,1)
 return (scores.softmax(-1)@v.float()).half()
def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-scripts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 m=load_official(a.legacy_scripts);encode=getattr(m.encode,'_torchdynamo_orig_callable',m.encode);_,_,decode=_upstream()
 torch.manual_seed(417);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for nv,nq in ((256,1),(768,17),(1280,33),(3328,1)):
   hkv,hq,d=4,28,128;x=torch.randn(1,hkv,nv,d,device='cuda',dtype=torch.float16);v=torch.randn_like(x)
   q=torch.randn(1,hq,nq,d,device='cuda',dtype=torch.float16);r=encode_kvarn(x,'k',2);p0=r.packed
   codes=decode._unpack_lowbit(p0['q_packed_uint8'],128,2).float()
   kr=(((codes*p0['s_col_K'].float()[...,None]+p0['zp_K'].float()[...,None])*p0['s_row_K'].float()[:,None,:]).transpose(-1,-2)).half().reshape(1,hkv,nv,d)
   vc,vs,vz=m.quant(v.transpose(1,2),1,per_channel=True,scale=.3,post_scale=True)
   packed=encode(vc.transpose(1,2).long(),bits=1,datum_per_elem=8);vr=(vs*(vc-vz)).transpose(1,2).contiguous()
   rec=(packed,vs.transpose(1,2).contiguous(),vz.transpose(1,2).contiguous());had=_hadamard(d,'cuda')
   tp=torch.cat((torch.arange(1,128,3,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   s=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,hkv,len(tp),d,device='cuda',dtype=torch.float16),text_v=torch.randn(1,hkv,len(tp),d,device='cuda',dtype=torch.float16),text_length=len(tp),length=2*nv+nq)
   actual=attention(q,s,r,rec,had);qr=(q.float()@had).half();expected=dense(q,qr,kr,vr,s)
   original=dense(q,qr,kr,vr,s,decode_kvarn(r));e=errors(actual,expected);original_error=errors(actual,original)
   ok=bool(torch.isfinite(actual).all()) and e['max_abs']<.002 and e['relative_l2']<.002 and original_error['max_abs']<.02
   rows.append({'nv':nv,'nq':nq,'gqa':7,'rotated_dense_error':e,'original_materialized_frame_error':original_error,'pass':ok})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','thresholds':{'rotated_dense_abs':.002,'rotated_dense_rel':.002,'original_frame_abs':.02},'cases':rows,
  'kernel_sha256':hashlib.sha256(Path(__file__).with_name('fused_hybrid.py').read_bytes()).hexdigest()}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps(report))
 if report['status']!='PASS':raise RuntimeError('Packed hybrid numerical check fails')
if __name__=='__main__':main()
