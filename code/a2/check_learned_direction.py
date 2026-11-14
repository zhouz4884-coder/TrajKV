"""GPU numeric checks for matrix-free rotation and headwise product-code reads."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from fast_wht import wht,torch_wht
from fused_learned_direction import attention
from quantizers import affine_rtn
from fused_progressive import decode_last_generic
from check_fused_progressive import errors

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(125);torch.backends.cuda.matmul.allow_tf32=False;rows=[];rotation=[]
 with torch.inference_mode():
  matrix=torch.ones(1,1,device='cuda',dtype=torch.float32)
  while matrix.shape[0]<128:matrix=torch.cat((torch.cat((matrix,matrix),1),torch.cat((matrix,-matrix),1)),0)
  matrix=matrix/(128**.5)
  for nv,nq in [(256,1),(1792,17),(4352,1)]:
   q=torch.randn(1,nq,28,128,device='cuda',dtype=torch.float16).transpose(1,2)
   actual=wht(q);reference=torch_wht(q);er=errors(actual,reference)
   matrix_error=errors(actual,q.float()@matrix)
   rotation.append({'nv':nv,'nq':nq,'error':er,'matrix_error':matrix_error,'half_equal':bool(torch.equal(wht(q,half=True),reference.half())),'pass':er['max_abs']<.00001 and matrix_error['max_abs']<.00001})
   tp=torch.cat((torch.arange(1,64,2,device='cuda'),torch.arange(2*nv,2*nv+nq,device='cuda')));nt=len(tp)
   state=SimpleNamespace(visual_pos=torch.arange(0,2*nv,2,device='cuda'),text_pos=tp,text_k=torch.randn(1,4,512,128,device='cuda',dtype=torch.float16),text_v=torch.randn(1,4,512,128,device='cuda',dtype=torch.float16),text_length=nt,length=2*nv+nq,
    direction_mean=torch.randn(1,4,1,128,device='cuda',dtype=torch.float16),direction_gain=(.5+torch.rand(1,4,nv,1,device='cuda')).half(),
    direction_code=torch.randint(0,256,(1,4,nv,32),device='cuda',dtype=torch.uint8))
   v=torch.randn(1,4,128,nv,device='cuda',dtype=torch.float16);rec=affine_rtn(v,1,128)
   for name,tensor in zip(('code','scale','min'),rec):setattr(state,'v_'+name,tensor)
   values=decode_last_generic(*rec,128,1).transpose(2,3).contiguous();global_book=torch.randn(256,4,device='cuda',dtype=torch.float16);global_out=None
   for mode in ('global','repeated','headwise'):
    book=global_book if mode=='global' else global_book[None].repeat(4,1,1).contiguous()
    if mode=='headwise':book=(book.float()+.1*torch.randn_like(book.float())).half()
    output=attention(q,state,book,1,True,128)
    if mode=='global':global_out=output.clone()
    if mode=='repeated' and not torch.equal(global_out,output):raise RuntimeError('Repeated identical books changed packed outputs')
    heads=torch.arange(4,device='cuda')[None,:,None,None];template=book[state.direction_code.long()] if mode=='global' else book[heads,state.direction_code.long()]
    rotated=(template.reshape(1,4,nv,128).float()*state.direction_gain.float()).half().repeat_interleave(7,1)
    bias=(q.float()*state.direction_mean.repeat_interleave(7,1).float()).sum(-1,keepdim=True)/(128**.5)
    zv=torch_wht(q).half().float()@rotated.float().transpose(-1,-2)/(128**.5)+bias
    zt=q.float()@state.text_k[:,:,:nt].repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
    z=torch.cat((zv,zt),-1);positions=torch.cat((state.visual_pos,tp));qp=torch.arange(state.length-nq,state.length,device='cuda')
    z.masked_fill_(positions[None,None,None,:]>qp[None,None,:,None],-torch.inf)
    all_values=torch.cat((values,state.text_v[:,:,:nt]),2).repeat_interleave(7,1)
    expected=(z.softmax(-1)@all_values.float()).half();error=errors(output,expected)
    rows.append({'nv':nv,'nq':nq,'book':mode,'error':error,'pass':error['max_abs']<.003 and error['relative_l2']<.003 and bool(torch.isfinite(output).all())})
 passed=all(r['pass'] for r in rows+rotation)
 result={'status':'PASS' if passed else 'FAIL','reader_cases':len(rows),'rotation_cases':len(rotation),'results':rows,'rotation':rotation,
  'scope':'same-representation GPU arithmetic; no task accuracy or performance claim',
  'sources':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('fast_wht.py','fused_learned_direction.py')}}
 (a.output/'check.json').write_text(json.dumps(result,indent=2));(a.output/'complete.json').write_text(json.dumps({'status':result['status'],'reader_cases':len(rows),'rotation_cases':len(rotation)}));print(json.dumps({k:result[k] for k in ('status','reader_cases','rotation_cases')}))
 if not passed:raise RuntimeError('Learned-direction GPU arithmetic check failed')
if __name__=='__main__':main()
