"""GPU Lloyd codebooks for normalized rotated residuals, a Gaussian prior.

These constants are a quantizer component, not model training or quality
measurements. Actual model validation is separate. One global book is shared
across layers/heads, avoiding a per-group codebook memory explosion.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(20261113);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 with torch.inference_mode():
  for dim in (4,8):
   x=torch.randn(65536,dim,device='cuda');test=torch.randn(16384,dim,device='cuda');centers=x[torch.randperm(len(x),device='cuda')[:256]].clone();history=[]
   for iteration in range(40):
    distance=x.square().sum(-1,keepdim=True)+centers.square().sum(-1)[None]-2*(x@centers.T);index=distance.argmin(-1)
    total=torch.zeros_like(centers).index_add_(0,index,x);count=torch.bincount(index,minlength=256).float()
    updated=total/count[:,None].clamp_min(1);centers=torch.where(count[:,None]>0,updated,centers)
    history.append(float((x-centers[index]).square().mean()))
   centers=centers.half().float();distance=test.square().sum(-1,keepdim=True)+centers.square().sum(-1)[None]-2*(test@centers.T);index=distance.argmin(-1)
   mse=float((test-centers[index]).square().mean());book={'schema':'trajkv.direction-book.v1','dimension':dim,'centroids':256,'nominal_bits_per_scalar':8/dim,
    'centers':centers.cpu().tolist(),'storage_dtype':'FP16','actual_book_bytes':256*dim*2,'training_distribution':'standard Gaussian prior after normalized rotation; not model data',
    'seed':20261113,'training_vectors':65536,'validation_vectors':16384,'iterations':40,'validation_mse':mse,'training_mse_history':history,
    'scope':'quantizer fit diagnostic; not endpoint model quality'}
   f=a.output/f'gaussian_vq_g{dim}_256.json';f.write_text(json.dumps(book,indent=2));rows.append({'dimension':dim,'validation_mse':mse,'book_bytes':book['actual_book_bytes'],'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
   print(json.dumps(rows[-1]),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','books':rows}))
if __name__=='__main__':main()
