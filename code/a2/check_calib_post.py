"""Compare post-scale matmuls to the unmodified official packed CUDA operators."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from calib_bank import load_official
def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-scripts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 m=load_official(a.legacy_scripts);encode=getattr(m.encode,'_torchdynamo_orig_callable',m.encode);torch.manual_seed(731);rows=[]
 with torch.inference_mode():
  for nv in (256,768):
   q=torch.randn(1,4,1,128,device='cuda',dtype=torch.float16)
   x=torch.randn(1,nv,2,128,device='cuda',dtype=torch.float16)
   c,scale,zero=m.quant(x,1,per_channel=True,scale=.5,post_scale=True)
   code=c.transpose(1,2).contiguous();packed=encode(code.long(),bits=1,datum_per_elem=8).repeat_interleave(2,1)
   sc=scale.transpose(1,2).repeat_interleave(2,1);z=zero.transpose(1,2).repeat_interleave(2,1)
   qs=q*sc;values=torch.tensor([0.,1.],device='cuda',dtype=torch.float16)
   official=m.fused_kernel(values,qs,packed,BITS=1,DATUM_PER_ELM=8)
   port=qs@code.repeat_interleave(2,1).transpose(-1,-2)
   ek=float((official.float()-port.float()).abs().max())
   w=port.float().softmax(-1).half()
   official_v=m.fused_kernel_wv(values,w,packed,BITS=1,DATUM_PER_ELM=8)
   port_v=w@code.repeat_interleave(2,1)
   ev=float((official_v.float()-port_v.float()).abs().max())
   rows.append({'nv':nv,'official_vs_torch_discrete_qk_max_abs':ek,'official_vs_torch_discrete_wv_max_abs':ev,
    'pass':ek<=.03125 and ev<=.002})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','thresholds':{'qk':.03125,'wv':.002},'cases':rows,
  'scope':'unmodified official packed operators vs same discrete code PyTorch matmuls, not end-to-end original-model reproduction'}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status']}));print(json.dumps(report))
 if report['status']!='PASS':raise RuntimeError('Calib post-scale CUDA parity fails')
if __name__=='__main__':main()
