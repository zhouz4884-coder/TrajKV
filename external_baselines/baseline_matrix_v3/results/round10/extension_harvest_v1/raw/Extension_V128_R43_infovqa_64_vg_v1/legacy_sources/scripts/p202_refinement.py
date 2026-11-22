import json
from pathlib import Path
import torch
from p201_refinement import quantize


def pack(c,bits):
    flat=c.flatten().byte();shifts=torch.arange(8,device=c.device)
    assert flat.numel()%8==0
    p=torch.stack([(((flat>>b)&1).reshape(-1,8).long()<<shifts).sum(-1).byte() for b in range(bits)])
    back=sum((((p[b,:,None].long()>>shifts)&1).flatten()<<b) for b in range(bits)).byte()
    assert torch.equal(back,flat)
    return p


def augment(tables):
    learned=json.loads((Path(__file__).resolve().parents[1]/'results/p202/codebook.json').read_text())['centroids']
    ledger={};stats=[];payload={}
    for i,t in enumerate(tables):
        for axis in ('k','v'):
            orig=t[axis][16];base=t[axis][2];saved=base.clone()
            orient=lambda x:x.transpose(2,3).contiguous() if axis=='k' else x
            e=orient(orig.float()-base.float());shape=e.shape;g=e.reshape(*shape[:-1],-1,128)
            scale=t['encoded'][axis,2][1].float()
            u=g/scale[...,None].clamp_min(1e-30)
            for label in ('r2','sign','fixed','learned'):
                if label=='r2':
                    decoded,encoded=quantize(e,2)
                elif label=='sign':
                    alpha=g.abs().mean(-1).half();c=(g>=0).byte()
                    encoded=(pack(c,1),alpha)
                    decoded=((2*c.float()-1)*alpha.float()[...,None]).reshape(shape)
                else:
                    cb=torch.tensor([-.375,-.125,.125,.375] if label=='fixed' else learned[axis],device=e.device,dtype=torch.float32)
                    c=(u[...,None]-cb).abs().argmin(-1).byte()
                    encoded=(pack(c,2),)
                    decoded=(cb[c.long()]*scale[...,None]).reshape(shape)
                result=(base.float()+orient(decoded)).half();t[axis][label]=result
                n=sum(x.numel()*x.element_size() for x in encoded)
                ledger[axis+label]=ledger.get(axis+label,0)+n
                payload[f'{i}_{axis}_{label}']=tuple(x.cpu() for x in encoded)
                stats.append(dict(layer=i,axis=axis,format=label,bytes=n,mse=float((result.float()-orig.float()).square().mean()),
                    base_mse=float((base.float()-orig.float()).square().mean()),normalized_over_half=float((u.abs()>.5).float().mean())))
            assert torch.equal(saved,base)
    return ledger,stats,payload


def methods():
    return {'base':((2,2),'prefill'),'k4v4':((4,4),'prefill'),'p201':(('r2','r2'),'prefill'),
        'asym_sign':(('r2','sign'),'prefill'),'aligned':(('fixed','fixed'),'prefill'),
        'aligned_learned':(('learned','learned'),'prefill')}
