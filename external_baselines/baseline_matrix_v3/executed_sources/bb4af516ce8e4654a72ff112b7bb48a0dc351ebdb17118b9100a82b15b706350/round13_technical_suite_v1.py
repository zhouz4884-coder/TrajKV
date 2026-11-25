"""Sequential Round13 GPU1 QA0 suite; never loads model weights or answers."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode=True


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def pruned_cuda_checks(a2,output):
    import torch
    sys.path.insert(0,str(a2))
    from round13_pruned_cpuqa_v1 import fake_full_a
    from round13_pruned_bank_v1 import compact_full_a_owners
    import round12_static_bank_v1 as full_bank
    import round12_static_codec_v1 as codec
    from round12_static_technical_v1 import dense_reference
    reader=importlib.import_module('round12_static_attention_v1').attention
    wht=importlib.import_module('fast_wht').wht
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_grad_enabled(False)
    device=torch.device('cuda:0');torch.manual_seed(13003201)
    n,m=768,544
    cpu,cvp=fake_full_a(n)
    shared={name:value.to(device) for name,value in cpu[0].shared.items()}
    vp=cvp.to(device)
    owners=[]
    for old in cpu:
        s=SimpleNamespace(**{name:value.to(device) if isinstance(value,torch.Tensor) else value
                             for name,value in vars(old.state).items()
                             if name not in ('visual_pos','depth_residual_lookup','text_k','text_v','text_pos')})
        s.visual_pos=vp;s.depth_residual_lookup=shared['global_residual_lookup'][old.layer_idx]
        if hasattr(s,'depth_residual_code'):
            slots=s.depth_residual_code.shape[0]
            s.depth_residual_lookup[0,0,0]=0
            if slots>1:s.depth_residual_lookup[1,1,0]=1
        owners.append(full_bank.Round12Owner(old.layer_idx,s,shared,reader,old.boundary))
    assert full_bank.unique_capacity(owners)==6382*n+91657
    pointer_before=[{name:value.data_ptr() for name,value in o.named_tensors().items()} for o in owners]
    identity=compact_full_a_owners(owners,torch.arange(n,device=device),vp)
    assert identity is owners
    assert all(value.data_ptr()==pointer_before[i][name]
               for i,o in enumerate(owners) for name,value in o.named_tensors().items())
    owner=owners[0]
    q=torch.randn((1,28,5,128),device=device).half()
    tp=torch.cat((torch.arange(9,device=device),torch.arange(n+9,n+32,device=device)))
    qp=torch.arange(n+27,n+32,device=device)
    tk=torch.randn((1,4,tp.numel(),128),device=device).half()
    tv=torch.randn_like(tk)
    full_output=owner.attend(q,exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp)
    # M=N must leave the actual old CUDA arithmetic and every packed pointer intact.
    full_again=identity[0].attend(q,exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp)
    assert torch.equal(full_output.view(torch.int16),full_again.view(torch.int16))
    previous={name:getattr(owner.state,name).clone() for name in ('direction_code','direction_gain','v_code','v_scale','v_min')}
    available=torch.arange(n-256,device=device)
    rows=torch.cat((available[::2],available[1:64:2],torch.arange(n-256,n,device=device))).sort().values
    assert rows.numel()==m
    compact=compact_full_a_owners(owners,rows,vp)
    state=compact[0].state
    kbefore=codec.decode_key(previous['direction_code'],previous['direction_gain'],shared['book'])
    kafter=codec.decode_key(state.direction_code,state.direction_gain,shared['book'])
    vbefore=codec.half_value(previous['v_code'],previous['v_scale'],previous['v_min'])
    vafter=codec.half_value(state.v_code,state.v_scale,state.v_min)
    assert torch.equal(kafter.view(torch.int16),kbefore[:,:,rows].contiguous().view(torch.int16))
    assert torch.equal(vafter.view(torch.int16),vbefore[...,rows].contiguous().view(torch.int16))
    actual=compact[0].attend(q,exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp)
    state.text_k,state.text_v,state.text_pos=tk,tv,tp
    state.text_length=tp.numel();state.length=int(qp[-1])+1
    try:reference=dense_reference(q,state,shared['book'],wht)
    finally:compact[0]._clear_text()
    error=(actual.float()-reference.float()).abs()
    assert bool(torch.isfinite(actual).all()) and float(error.max())<=.008,float(error.max())
    actual_bytes=full_bank.unique_capacity(compact)
    expected_bytes=6382*m+91657+8*n+16
    assert actual_bytes==expected_bytes
    blocked=False
    try:
        compact[0].attend(q,exact_keys=torch.cat((tk,tk[:,:,:1]),2),
            exact_values=torch.cat((tv,tv[:,:,:1]),2),
            exact_positions=torch.cat((tp,vp[1:2])),query_positions=qp)
    except ValueError:blocked=True
    assert blocked
    torch.cuda.synchronize(device)
    result={'status':'PASS_ACTUAL_CUDA_PRUNED_QA0','actual_CUDA':True,
            'model_calls':0,'task_answer_events':0,'N':n,'M':m,
            'M_equals_N_pointers_and_CUDA_output_bit_identity':True,
            'retained_packed_K_and_V_Half_field_bit_identity':True,
            'dense_max_abs':float(error.max()),'dense_tolerance':.008,
            'actual_owner_bytes':actual_bytes,'expected_owner_bytes':expected_bytes,
            'pruned_visual_in_native_text_rejected':True,
            'GPU':torch.cuda.get_device_name(device)}
    write_new(output/'PRUNED_CUDA_TECHNICAL_v1.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-a2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not str(out).startswith('/media/') or '/baseline_matrix_v3/results/round13' not in str(out):
        raise ValueError('New Round13 project data-disk output required')
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();results=[]
    code=Path(__file__).resolve().parent
    steps=[('exception',[sys.executable,'-B',str(code/'round13_exception_technical_v1.py'),
        '--legacy-a2',str(args.legacy_a2),'--output',str(out/'EXCEPTION_CUDA_TECHNICAL_v1.json'),
        '--visual-tokens','32','256','1792','--device','cuda:0']),
        ('precision',[sys.executable,'-B',str(code/'round13_precision_technical_v1.py'),
        '--legacy-a2',str(args.legacy_a2),'--output',str(out/'precision'),
        '--visual-tokens','32','256','1792','--device','cuda:0'])]
    for name,argv in steps:
        print(json.dumps({'event':'START','step':name,'time':time.time()}),flush=True)
        before=time.perf_counter()
        result=subprocess.run(argv,check=False)
        row={'step':name,'returncode':result.returncode,'seconds':time.perf_counter()-before}
        results.append(row);write_new(out/(name+'_step_receipt.json'),row)
        if result.returncode:
            write_new(out/'SUITE_FAILED_v1.json',{'status':'FAIL','steps':results})
            raise SystemExit(result.returncode)
    print(json.dumps({'event':'START','step':'pruned','time':time.time()}),flush=True)
    try:
        pruned=pruned_cuda_checks(args.legacy_a2,out)
        results.append({'step':'pruned','status':pruned['status']})
    except Exception as exc:
        write_new(out/'SUITE_FAILED_v1.json',{'status':'FAIL','steps':results,
                  'failed_step':'pruned','error_type':type(exc).__name__,'error':str(exc)})
        raise
    report={'status':'PASS_ACTUAL_CUDA_ROUND13_TECHNICAL_SUITE','model_calls':0,
            'task_answer_events':0,'steps':results,'elapsed_seconds':time.perf_counter()-start,
            'suite_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_new(out/'SUITE_COMPLETE_v1.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
