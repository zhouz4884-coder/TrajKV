"""Isolate a headwise lookup reader, retaining the existing attention algorithm."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 source=Path(__file__).with_name('fused_direction.py');text=source.read_text(encoding='utf-8')
 def replace(old,new):
  nonlocal text
  if text.count(old)!=1:raise RuntimeError('Unexpected kernel template: '+old[:80])
  text=text.replace(old,new)
 replace('import triton.language as tl','import triton.language as tl\nfrom fast_wht import wht')
 replace('VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):','VB:tl.constexpr,VCHANNEL:tl.constexpr,VG:tl.constexpr,HEADWISE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):')
 replace('value=tl.load(BOOK+index*KDIM+dims[:,None]%KDIM,valid[None,:],0).to(tl.float32);k=(value*gain[None,:]).to(tl.float16)',
  'book_offset=owner*256*KDIM if HEADWISE else 0\n    value=tl.load(BOOK+book_offset+index*KDIM+dims[:,None]%KDIM,valid[None,:],0).to(tl.float32);k=(value*gain[None,:]).to(tl.float16)')
 replace('def attention(q,state,hadamard,book=None,query_scale=None,vbits=1,v_channel=True,v_group_size=128):','def attention(q,state,book,vbits=1,v_channel=True,v_group_size=128):')
 replace('scaled=q.float() if query_scale is None else q.float()/query_scale.repeat_interleave(g,1).float()\n qr=(scaled@hadamard).half().contiguous();bias=',
  'qr=wht(q,half=True);bias=')
 replace('vbits,v_channel,v_group_size,16,32,','vbits,v_channel,v_group_size,book.ndim==3,16,32,')
 dest=source.with_name('fused_learned_direction.py');compile(text,str(dest),'exec')
 if dest.exists():raise RuntimeError('Learned-direction kernel already exists')
 dest.write_text(text,encoding='utf-8')
 (ROOT/'audit/a2/learned_direction_kernel_build.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'derived_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'changes':['matrix-freeFWHT query rotation','optional per-KV-head decoder table offset'],'status':'not GPU checked yet'},indent=2))
 print(json.dumps({'status':'PREPARED','kernel':dest.name}))
if __name__=='__main__':main()
