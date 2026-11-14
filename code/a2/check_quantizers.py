"""CPU packing and weighted-centroid objective checks, no model or test data."""
import argparse,json
from pathlib import Path
import torch
from quantizers import binary_centroids,pack1
def decode(rec):
 code,scale,minimum=rec;word=code.long()&0xffffffff;shifts=torch.arange(32)
 c=((word[...,None]>>shifts)&1).reshape(*code.shape[:-1],code.shape[-1]*32)
 g=c.reshape(*c.shape[:-1],-1,128).half()
 return ((g*scale[...,None]).half()+minimum[...,None]).reshape(c.shape)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.manual_seed(18);rows=[]
 for shape in ((1,4,128,256),(1,4,256,128)):
  x=torch.randn(shape).half();x[...,0]=10;x[...,1]=-8
  for weighted in (False,True):
   w=torch.rand(shape).float()+.1 if weighted else torch.ones(shape)
   g=x.float().reshape(*shape[:-1],-1,128);wg=w.reshape(g.shape)
   rec=binary_centroids(x,weights=wg);y=decode(rec).float()
   low,high=g.amin(-1,keepdim=True),g.amax(-1,keepdim=True)
   rtn=torch.where(g>(low+high)/2,high,low).reshape(shape)
   before=float(((rtn-x.float()).square()*w).mean());after=float(((y-x.float()).square()*w).mean())
   rows.append({'shape':shape,'weighted':weighted,'before_mse':before,'after_mse':after,'pass':after<=before+1e-5})
 code=torch.ones((1,1,128,128),dtype=torch.uint8);assert (pack1(code)==-1).all()
 constant=torch.full((1,2,128,128),3.5,dtype=torch.float16);assert torch.equal(decode(binary_centroids(constant)),constant)
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','checks':rows,'signed_word':True,'constant_group':True,'scope':'synthetic CPU encoding only'}
 a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
 if report['status']!='PASS':raise RuntimeError('Centroid objective or packing failed')
if __name__=='__main__':main()
