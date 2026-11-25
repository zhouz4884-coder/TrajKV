"""CPU format checks and callable CUDA reader-vs-dense technical acceptance.

CPU mode is explicitly not an end-to-end model result. CUDA mode accepts
already built owners from the real runner, avoiding an alternate encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch
import r7_legacy_bridge_v1 as bridge
import round13_exception_bank_v1 as bank


def _synthetic(n, stages=43):
    # Exactly reproduce physical M1 tensor shapes; five header bytes replace
    # the frozen policy/ranking constants solely for this format unit check.
    lookup=torch.full((28,2,4,2),-1,dtype=torch.int64)
    shared={'book':torch.randn(16,4).half(),'global_residual_lookup':lookup,
            'synthetic_header_bytes':torch.zeros(5,dtype=torch.uint8)}
    positions=torch.arange(n,dtype=torch.long)
    ranking=([['k',i//4,i%4,0] for i in range(23)]
             +[['v',i//4,i%4,0] for i in range(20)])
    result=[]
    for layer in range(28):
        entries=[x for x in ranking if x[1]==layer]
        s=SimpleNamespace(visual_pos=positions,
            direction_code=torch.randint(0,256,(1,4,n,16),dtype=torch.uint8),
            direction_gain=torch.randn(1,4,n,1).half(),
            direction_mean=torch.randn(1,4,1,128).half(),
            v_code=torch.randint(-(2**31),2**31-1,(1,4,128,n//32),dtype=torch.int32),
            v_scale=torch.rand(1,4,128,1).half(),v_min=torch.randn(1,4,128,1).half(),
            depth_residual_lookup=lookup[layer])
        if entries:
            s.depth_residual_code=torch.randint(-(2**31),2**31-1,(len(entries),n,4),dtype=torch.int32)
            s.depth_residual_alpha=torch.randn(len(entries),n).half()
        for slot,(axis,_,head,depth) in enumerate(entries):
            lookup[layer,int(axis=='v'),head,depth]=slot
        result.append(bridge.R7LegacyOwner(layer,s,shared,None,n))
    assert bank.unique_capacity(result)==4590*n+89733
    return result,ranking


def run_cpu_checks():
    torch.manual_seed(13)
    checks=[]
    for n,stages,h in ((256,43,0),(256,43,4),(768,43,13),
                       (1280,43,21),(1792,43,30),(1792,41,32),
                       (1792,0,55),(32,43,32)):
        owners,original=_synthetic(n)
        high=torch.randperm(n)[:h].sort().values
        ordinary=torch.ones(n,dtype=torch.bool);ordinary[high]=False
        rows=ordinary.nonzero(as_tuple=False).flatten()
        selected=bank._stage_subset(original,stages)
        shared=owners[0].shared
        flags=(~ordinary).long().reshape(-1,32)
        shift=torch.arange(32,dtype=torch.int64)
        word=(flags<<shift).sum(-1).int()
        count=flags.sum(-1); prefix=(count.cumsum(0)-count).int()
        if h or stages!=43:
            shared.update(round13_exception_mask=word,round13_exception_prefix=prefix,
                round13_exception_rows=high.int(),
                round13_exception_format=torch.tensor([1,stages,1,0],dtype=torch.uint8))
        out=[]
        for layer,old in enumerate(owners):
            s=old.state
            oldk=s.direction_code.clone();oldgain=s.direction_gain.clone()
            oldv=bank._unpack_v1(s.v_code,n).clone()
            if not (h==0 and stages==43):
                bank.compact_fields(s,original,selected,layer,rows)
                assert torch.equal(s.direction_code,oldk[:,:,rows])
                assert torch.equal(s.direction_gain,oldgain[:,:,rows])
                assert torch.equal(bank._unpack_v1(s.v_code,len(rows)),oldv[...,rows])
                s.round13_native_k=torch.randn(1,4,h,128).half()
                s.round13_native_v=torch.randn(1,4,h,128).half()
                out.append(bank.Round13ExceptionOwner(layer,s,shared,None,n))
            else:
                out.append(old)
        actual=bank.unique_capacity(out)
        expected=bank.expected_owner_bytes(n,h,stages)
        assert actual==expected,(n,stages,h,actual,expected)
        # Check the exact original-row -> compact-low mapping the kernel uses.
        mapped=[]
        for i in range(n):
            w=int(word[i//32])&0xffffffff
            ishigh=(w>>(i%32))&1
            rank=int(prefix[i//32])+(w&((1<<(i%32))-1)).bit_count()
            if not ishigh:
                mapped.append((i,i-rank))
        assert [x[0] for x in mapped]==rows.tolist()
        assert [x[1] for x in mapped]==list(range(n-h))
        checks.append({'N':n,'base_stages':stages,'h':h,'actual_owner_bytes':actual,
                       'expected_owner_bytes':expected,'retained_payload_bit_identity':True,
                       'routing_rank_identity':True,'V_padding_rows':(-(n-h))%32})
    plans=[bank.budget_plan(n,stages) for n in (256,768,1280,1792,2304)
           for stages in (43,41,0)]
    assert bank.budget_plan(1792,43)['exception_rows']==30
    assert bank.budget_plan(1792,41)['exception_rows']==32
    assert bank.budget_plan(1792,0)['exception_rows']==55
    return {'scope':'CPU format/capacity/routing checks only; no GPU or model inference',
            'status':'PASS','checks':checks,'budget_plans':plans}


@torch.no_grad()
def compare_cuda_owner(owner,q,*,exact_keys,exact_values,exact_positions,
                       query_positions,atol=0.012,rtol=0.012):
    """Called by Root technical run on a real built exception owner."""
    from round13_exception_attention_v1 import dense_reference
    actual=owner.attend(q,exact_keys=exact_keys,exact_values=exact_values,
                        exact_positions=exact_positions,query_positions=query_positions)
    state=owner.state
    state.text_k,state.text_v=exact_keys,exact_values
    state.text_pos,state.text_length=exact_positions,int(exact_positions.numel())
    state.length=int(query_positions[-1])+1
    try:
        reference=dense_reference(q,state,owner.shared['book'])
    finally:
        owner._clear_text()
    delta=(actual.float()-reference.float()).abs()
    require=torch.allclose(actual,reference,atol=atol,rtol=rtol)
    return {'status':'PASS' if require else 'FAIL','max_abs':float(delta.max()),
            'mean_abs':float(delta.mean()),'atol':atol,'rtol':rtol,
            'scope':'actual packed CUDA reader against transient dense Torch diagnostic'}


@torch.no_grad()
def run_cuda_checks(legacy_a2, visual_tokens, device):
    """Actual CUDA arithmetic controls, no weights or task-answer generation."""
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required for --output mode')
    sys.path.insert(0,str(legacy_a2))
    reader=importlib.import_module('round13_exception_attention_v1')
    legacy=importlib.import_module('round7_key_attention').attention
    import round12_static_codec_v1 as codec
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    gen=torch.Generator(device=device).manual_seed(13002501)
    records=[]
    for n in visual_tokens:
        for h in sorted({0,1,min(5,n),n//2,n}):
            for nq in (1,5,17):
                book=(torch.randn((16,4),device=device,generator=gen)*.5).half()
                source=torch.randn((1,4,n,128),device=device,generator=gen)
                code,gain=codec.encode_key(source,book)
                nt=23
                vp=torch.arange(1,n+1,device=device,dtype=torch.long)
                tp=torch.cat((torch.zeros(1,device=device,dtype=torch.long),
                              torch.arange(n+1,n+nt,device=device,dtype=torch.long)))
                state=SimpleNamespace(direction_code=code,direction_gain=gain,
                    direction_mean=(torch.randn((1,4,1,128),device=device,generator=gen)*.04).half(),
                    v_code=codec.pack_channel_codes(torch.randint(0,2,(1,4,128,n),device=device,generator=gen),1),
                    v_scale=(torch.rand((1,4,128,1),device=device,generator=gen)+.1).half(),
                    v_min=(torch.randn((1,4,128,1),device=device,generator=gen)*.2-.5).half(),
                    visual_pos=vp,text_pos=tp,text_length=nt,length=n+nt,
                    text_k=torch.randn((1,4,nt,128),device=device,generator=gen).half(),
                    text_v=torch.randn((1,4,nt,128),device=device,generator=gen).half())
                state.depth_residual_lookup=torch.full((2,4,2),-1,dtype=torch.int32,device=device)
                state.depth_residual_lookup[0,0,0]=0;state.depth_residual_lookup[0,0,1]=1
                state.depth_residual_lookup[1,1,0]=2;state.depth_residual_lookup[1,1,1]=3
                state.depth_residual_code=codec.pack_channel_codes(torch.randint(0,2,(4,n,128),device=device,generator=gen),1)
                state.depth_residual_alpha=(torch.rand((4,n),device=device,generator=gen)*.03).half()
                q=torch.randn((1,28,nq,128),device=device,generator=gen).half()
                old=legacy(q,state,book,n) if h==0 else None
                high=torch.randperm(n,device=device,generator=gen)[:h].sort().values
                keep=torch.ones(n,device=device,dtype=torch.bool);keep[high]=False
                ordinary=keep.nonzero(as_tuple=False).flatten()
                original=[['k',0,0,0],['k',0,0,1],['v',0,1,0],['v',0,1,1]]
                bank.compact_fields(state,original,original,0,ordinary)
                flags=(~keep).long().reshape(-1,32)
                shifts=torch.arange(32,device=device,dtype=torch.int64)
                state.round13_exception_mask=(flags<<shifts).sum(-1).int()
                count=flags.sum(-1)
                state.round13_exception_prefix=(count.cumsum(0)-count).int()
                state.round13_exception_rows=high.int()
                state.round13_ordinary_rows=n-h
                source_k=torch.randn((1,4,n,128),device=device,generator=gen).half()
                source_v=torch.randn((1,4,n,128),device=device,generator=gen).half()
                state.round13_native_k=source_k.index_select(2,high)
                state.round13_native_v=source_v.index_select(2,high)
                assert torch.equal(state.round13_native_k.view(torch.int16),source_k[:,:,high].view(torch.int16))
                assert torch.equal(state.round13_native_v.view(torch.int16),source_v[:,:,high].view(torch.int16))
                actual=reader.attention(q,state,book,n)
                reference=reader.dense_reference(q,state,book)
                error=(actual.float()-reference.float()).abs()
                assert bool(torch.isfinite(actual).all())
                assert float(error.max())<=.008,(n,h,nq,float(error.max()))
                row={'N':n,'h':h,'nq':nq,'V_padding_rows':(-(n-h))%32,
                     'dense_max_abs':float(error.max()),'dense_tolerance':.008,
                     'native_KV_field_bit_identity':True,
                     'full_128d_all28_query_heads_checked':True}
                if old is not None:
                    row['h0_new_reader_vs_frozen_legacy_bit_identity']=torch.equal(actual.view(torch.int16),old.view(torch.int16))
                    row['h0_max_abs']=float((actual.float()-old.float()).abs().max())
                    assert row['h0_new_reader_vs_frozen_legacy_bit_identity'],row
                if h==n:
                    keys=torch.cat((source_k,state.text_k),2).repeat_interleave(7,1).float()
                    values=torch.cat((source_v,state.text_v),2).repeat_interleave(7,1).float()
                    positions=torch.cat((vp,tp))
                    logits=q.float()@keys.transpose(-1,-2)/(128**.5)
                    qpos=torch.arange(state.length-nq,state.length,device=device)
                    logits.masked_fill_(positions[None,:]>qpos[:,None],-float('inf'))
                    direct=(logits.softmax(-1)@values).half()
                    row['all_native_direct_max_abs']=float((actual.float()-direct.float()).abs().max())
                    assert row['all_native_direct_max_abs']<=.008,row
                records.append(row)
    torch.cuda.synchronize(device)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-output')
    p.add_argument('--output',type=Path);p.add_argument('--legacy-a2',type=Path)
    p.add_argument('--visual-tokens',nargs='+',type=int,default=[32,256,1792])
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); started=time.perf_counter()
    if a.cpu_output:
        result=run_cpu_checks();path=Path(a.cpu_output)
    else:
        if a.output is None or a.legacy_a2 is None:
            p.error('--output and --legacy-a2 required for actual CUDA mode')
        if not all(n>0 and n%32==0 for n in a.visual_tokens):
            p.error('complete original visual uint32 words required')
        records=run_cuda_checks(a.legacy_a2,a.visual_tokens,torch.device(a.device))
        result={'status':'PASS_ACTUAL_CUDA_PACKED_EXCEPTION_QA0',
                'scope':'Actual CUDA format/attention controls; no model weights, no task answers',
                'actual_CUDA':True,'model_calls':0,'task_answer_events':0,
                'GPU':torch.cuda.get_device_name(torch.device(a.device)),
                'elapsed_seconds':time.perf_counter()-started,'checks':records}
        path=a.output
    result['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('round13_exception_bank_v1.py','round13_exception_attention_v1.py','round13_exception_technical_v1.py')}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),
                      'output':str(path)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
