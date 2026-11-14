"""CUDA parity for the new metadata fold and unchanged packed attention path."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from affine_rate_bank import fold
from fused_rates import attention
from quantizers import binary_centroids
from check_fused_progressive import dense_attention,errors

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(exist_ok=False,parents=True)
 sys.path.insert(0,cfg['legacy_scripts']);from p176_kivi_codec import load_official
 codec=load_official(Path(cfg['kivi_source']));torch.manual_seed(881);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 def decode(record,bits,shape):
  packed,scale,minimum=record;shift=torch.arange(32//bits,device='cuda')*bits
  codes=((packed.long()[...,None]>>shift)&((1<<bits)-1)).reshape(shape).half()
  return ((codes*scale.repeat_interleave(128,-1)).half()+minimum.repeat_interleave(128,-1)).half()
 with torch.inference_mode():
  for nv,nq in [(256,1),(1792,17),(4352,1)]:
   k=torch.randn(1,4,128,nv,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
   original=[codec.triton_quantize_and_pack_along_last_dim(k,128,2),binary_centroids(v)]
   q=torch.randn(1,nq,28,128,device='cuda',dtype=torch.float16).transpose(1,2)
   positions=torch.cat((torch.arange(1,64,2,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')))
   state=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=positions,text_k=torch.randn(1,4,len(positions),128,device='cuda',dtype=torch.float16),
    text_v=torch.randn(1,4,len(positions),128,device='cuda',dtype=torch.float16),text_length=len(positions),length=2*nv+nq)
   for centers in [False,True]:
    records=[]
    for axis,rec,bits in zip(('k','v'),original,(2,1)):
     gain=(torch.rand(1,4,1,1,device='cuda')*1.4-.7).exp();offset=(torch.rand_like(gain)*.7-.35) if centers else None
     adapted=fold(rec,bits,gain,offset);assert adapted[0].data_ptr()==rec[0].data_ptr()
     # Independent spelling of the differentiable calibration's metadata path.
     predicted_scale=(rec[1].float()*gain).half();mid=rec[2].float()+((1<<bits)-1)*rec[1].float()/2
     predicted_min=(mid-((1<<bits)-1)*predicted_scale.float()/2+(0. if offset is None else offset*predicted_scale.float())).half()
     assert torch.equal(adapted[1],predicted_scale)
     # Algebraically equivalent FP32 operation orders may differ before the
     # final FP16 cast by one rounding step; use the exact fitter order too.
     exact_min=(mid+((0. if offset is None else offset)-.5*((1<<bits)-1))*predicted_scale.float()).half()
     assert torch.equal(adapted[2],exact_min)
     for name,tensor in zip(('code','scale','min'),adapted):setattr(state,axis+'_'+name,tensor)
     records.append(adapted)
    actual=attention(q,state,bits=2,bits_v=1,v_channel=True,k_group_size=128,v_group_size=128)
    keys=decode(records[0],2,k.shape).transpose(2,3).contiguous();values=decode(records[1],1,v.shape).transpose(2,3).contiguous()
    expected,_=dense_attention(q,keys,values,state,None);error=errors(actual,expected)
    rows.append({'nv':nv,'nq':nq,'centers':centers,'codes_unchanged':True,'metadata_matches_fitter':True,'error':error,
     'pass':error['max_abs']<.003 and error['relative_l2']<.003 and bool(torch.isfinite(actual).all())})
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','cases':len(rows),'results':rows,
  'scope':'GPU same-representation dense causal/GQA check, no task accuracy','fused_kernel_sha256':hashlib.sha256(Path(__file__).with_name('fused_rates.py').read_bytes()).hexdigest()}
 (a.output/'check.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':report['status'],'cases':len(rows)}));print(json.dumps({'status':report['status'],'cases':len(rows)}))
 if report['status']!='PASS':raise RuntimeError('Affine metadata packed runtime parity failed')
if __name__=='__main__':main()
