"""Diagnostic scalar refinement; decoded views are not deployment memory."""
import torch


def quantize(x, bits):
    g=x.float().reshape(*x.shape[:-1],-1,128)
    minimum=g.amin(-1).half()
    scale=((g.amax(-1)-minimum.float())/((1<<bits)-1)).half()
    codes=((g-minimum.float()[...,None])/scale.float().clamp_min(1e-30)[...,None]).round().clamp(0,(1<<bits)-1).to(torch.uint8)
    # Bit planes provide actual compact storage even for three-bit codes.
    flat=codes.flatten(); assert flat.numel()%8==0
    shifts=torch.arange(8,device=x.device)
    packed=torch.stack([(((flat>>b)&1).reshape(-1,8).long()<<shifts).sum(-1).byte() for b in range(bits)])
    unpacked=sum((((packed[b,:,None].long()>>shifts)&1).flatten()<<b) for b in range(bits)).byte()
    assert torch.equal(unpacked,flat)
    decoded=(unpacked.reshape(g.shape).float()*scale.float()[...,None]+minimum.float()[...,None]).reshape(x.shape)
    return decoded,(packed,scale,minimum)


def augment(tables):
    ledger={};diagnostics=[];payload={}
    for i,t in enumerate(tables):
        for axis in ('k','v'):
            original=t[axis][16];base=t[axis][2];saved=base.clone()
            orient=lambda x:x.transpose(2,3).contiguous() if axis=='k' else x
            for label,bits,residual in (('q3',3,False),('r1',1,True),('r2',2,True)):
                target=original.float()-base.float() if residual else original.float()
                decoded,encoded=quantize(orient(target),bits)
                decoded=orient(decoded)
                result=(base.float()+decoded).half() if residual else decoded.half()
                t[axis][label]=result
                n=sum(x.numel()*x.element_size() for x in encoded)
                ledger[axis+label]=ledger.get(axis+label,0)+n
                payload[f'{i}_{axis}_{label}']=tuple(x.cpu() for x in encoded)
                diagnostics.append(dict(layer=i,axis=axis,format=label,bytes=n,
                    mse=float((result.float()-original.float()).square().mean()),
                    base_mse=float((base.float()-original.float()).square().mean())))
            assert torch.equal(saved,base)
    return ledger,diagnostics,payload


def methods():
    return {'base':((2,2),'prefill'),'fp16_prefill':((16,16),'prefill'),
        'k3v2':(('q3',2),'prefill'),'k3v3':(('q3','q3'),'prefill'),
        'k4v2':((4,2),'prefill'),'k4v4':((4,4),'prefill'),
        'kr1':(('r1',2),'prefill'),'kr2':(('r2',2),'prefill'),
        'kvr1':(('r1','r1'),'prefill'),'kvr2':(('r2','r2'),'prefill')}
