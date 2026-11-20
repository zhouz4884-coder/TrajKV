"""Source-example KIVI g32/R32 policy/codec/owner/packed-reader acceptance.

CPU uses author PyTorch AST fixtures, never substitutes a fake production
encoder. CUDA calls the untouched official Triton encoder and the new reader.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from methods.kivi_example_adapter import KIVIExampleAdapter,KIVIExampleState,prefill_lengths


def source_functions():
    path=ROOT/'third_party/kivi/quant/new_pack.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    wanted={'pack_tensor','unpack_tensor','quant_and_pack_kcache','quant_and_pack_vcache','unpack_and_dequant_kcache','unpack_and_dequant_vcache'}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in funcs}==wanted
    env={'torch':torch};exec(compile(ast.Module(funcs,type_ignores=[]),str(path),'exec'),env)
    return env


def source_policy_gate():
    example=ast.parse((ROOT/'third_party/kivi/example.py').read_text(encoding='utf-8'))
    defaults={}
    for node in example.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='config' and target.attr in ('k_bits','v_bits','group_size','residual_length'):
                    defaults[target.attr]=ast.literal_eval(node.value)
    assert defaults=={'k_bits':2,'v_bits':2,'group_size':32,'residual_length':32}
    # The runtime snapshot intentionally excludes directories named models.
    # This audit-only fixture is a byte-exact licensed copy of the pinned
    # upstream model source; it is parsed, never imported as a model runtime.
    source=ROOT/'audit/upstream/kivi_example_llama_kivi_source_v1.py'
    locked='2e823f68358c35b414f04fd25e5e91ec646eb16af7d2d6869a1fd7b2a30d3e68'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==locked
    proof=json.loads((ROOT/'configs/methods/kivi_example_source_v1.json').read_text(encoding='utf-8'))
    assert proof['source_files']['third_party/kivi/models/llama_kivi.py']['raw_sha256']==locked
    tree=ast.parse(source.read_text(encoding='utf-8'))
    keys=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='key_states.shape[-2] % self.residual_length != 0']
    vals=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='value_states.shape[-2] <= self.residual_length']
    assert len(keys)==len(vals)==2
    cases=[]
    for n in (1,17,31,32,33,63,64,65,257,512,1008,1792,4032):
        tensor=torch.arange(n,dtype=torch.float16).reshape(1,1,n,1)
        for key_node,value_node in zip(keys,vals):
            env={'self':SimpleNamespace(residual_length=32),'key_states':tensor,'value_states':tensor}
            exec(compile(ast.Module([key_node],type_ignores=[]),str(source),'exec'),env)
            # Extract only source V split statements; the following source
            # Triton codec call is tested on CUDA, not replaced by a CPU fake.
            policy=deepcopy(value_node);policy.orelse=policy.orelse[:2]
            exec(compile(ast.Module([policy],type_ignores=[]),str(source),'exec'),env)
            kq=0 if env['key_states_quant'] is None else env['key_states_quant'].shape[-2]
            vq=0 if env['value_states_quant'] is None else env['value_states_quant'].shape[-2]
            assert (kq,vq)==prefill_lengths(n)
        cases.append({'N':n,'source_K_quantized_tokens':prefill_lengths(n)[0],'source_V_quantized_tokens':prefill_lengths(n)[1]})
    return {'status':'PASS','defaults':defaults,'source_prefill_implementations_checked':2,'cases':cases}


def fixture(k,v,positions,env):
    n,h=k.shape[-2],k.shape[1];kq,vq=prefill_lengths(n);g=32;bits=2
    if kq:
        kc,ks,km=env['quant_and_pack_kcache'](k[:,:,:kq].contiguous(),g,bits)
        kc=kc.transpose(-2,-1).contiguous();ks=ks.squeeze(-2).transpose(-2,-1).contiguous();km=km.squeeze(-2).transpose(-2,-1).contiguous()
    else:kc=torch.empty((1,h,128,0),device=k.device,dtype=torch.int32);ks=k.new_empty((1,h,128,0));km=k.new_empty((1,h,128,0))
    if vq:
        vc,vs,vm=env['quant_and_pack_vcache'](v[:,:,:vq].contiguous(),g,bits);vs=vs.squeeze(-1).contiguous();vm=vm.squeeze(-1).contiguous()
    else:vc=torch.empty((1,h,0,8),device=k.device,dtype=torch.int32);vs=k.new_empty((1,h,0,4));vm=k.new_empty((1,h,0,4))
    return KIVIExampleState(kc,ks,km,vc,vs,vm,k[:,:,kq:].clone(),v[:,:,vq:].clone(),positions.clone().contiguous(),bits,n,kq,vq,g,32,'torch_dequant_reference')


def source_decode(state,env):
    if state.nkq:
        k=env['unpack_and_dequant_kcache'](state.k_code.transpose(-2,-1).contiguous(),state.k_scale.transpose(-2,-1).unsqueeze(-2),state.k_min.transpose(-2,-1).unsqueeze(-2),32,2)
    else:k=state.k_tail[:,:,:0]
    if state.nvq:v=env['unpack_and_dequant_vcache'](state.v_code,state.v_scale.unsqueeze(-1),state.v_min.unsqueeze(-1),32,2)
    else:v=state.v_tail[:,:,:0]
    return torch.cat((k,state.k_tail),-2),torch.cat((v,state.v_tail),-2)


def expected_bytes(n,h,d=128):
    kq,vq=prefill_lengths(n);g=32;bits=2
    payload=h*d*(kq+vq)*bits//8
    metadata=4*(h*d*(kq//g)+h*vq*(d//g))
    tails=2*h*d*((n-kq)+(n-vq));positions=8*n
    return {'payload':payload,'metadata':metadata,'tails':tails,'positions':positions,'total':payload+metadata+tails+positions}


def dense_reference(q,state,tk,tv,tp,qp,env):
    k,v=source_decode(state,env);groups=q.shape[1]//k.shape[1]
    keys=torch.cat((k,tk),-2).repeat_interleave(groups,1);values=torch.cat((v,tv),-2).repeat_interleave(groups,1)
    positions=torch.cat((state.positions,tp));visible=positions[None,:]<=qp[:,None]
    return F.scaled_dot_product_attention(q.float(),keys.float(),values.float(),attn_mask=visible[None,None]).to(q.dtype)


def run(device):
    torch.set_num_threads(4);torch.manual_seed(910332)
    policy=source_policy_gate();env=source_functions()
    result={'status':'RUNNING','scope':'SOURCE_EXAMPLE_B2_G32_R32_VISUAL_PORT_P0','device':str(device),
            'model_quality_measured':False,'actual_gpu_encoder_and_packed_reader':device.type=='cuda',
            'CPU_scope':'author-source PyTorch codec and policy fixtures; production CUDA encoder never substituted',
            'source_policy':policy,'tests':[],'torch_version':torch.__version__}
    if device.type=='cuda':
        assert torch.cuda.get_device_capability(device)==(8,6);result['hardware']=torch.cuda.get_device_name(device);torch.cuda.reset_peak_memory_stats(device)
    for n in (17,31,32,33,63,64,65,257,512,1008,1792,4032):
        for hq,hkv in ((28,4),(32,8)):
            k=torch.randn((1,hkv,n,128),device=device,dtype=torch.float16)*.4
            v=torch.randn_like(k)*.2
            # Four deliberately distinct V channel-group distributions catch
            # an erroneous g128-style one-scale-per-token reader broadcast.
            for group in range(4):v[...,group*32:(group+1)*32]=v[...,group*32:(group+1)*32]*(2**group)+group*.7
            base=torch.arange(3,3+2*n,device=device);positions=base[::2]
            state0=fixture(k,v,positions,env)
            state=KIVIExampleAdapter().encode(k,v,positions) if device.type=='cuda' else state0
            if device.type=='cuda':
                for name in state.named_tensors():torch.testing.assert_close(getattr(state,name),getattr(state0,name),atol=0,rtol=0)
            for actual,expected in zip(state.materialize(),source_decode(state,env)):torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            amounts=expected_bytes(n,hkv);report=state.owner_report()
            assert report['stored_visual_KV_bytes']==amounts['total'] and report['packed_payload_bytes']==amounts['payload']
            assert report['coder_metadata_bytes']==amounts['metadata'] and report['full_precision_visual_tail_bytes']==amounts['tails'] and report['positions_bytes']==amounts['positions']
            assert state.k_scale.shape==(1,hkv,128,state.nkq//32) and state.v_scale.shape==(1,hkv,state.nvq,4)
            assert state.k_tail.shape[-2]==n%32 and state.v_tail.shape[-2]==min(n,32)
            source_ptrs={k.untyped_storage().data_ptr(),v.untyped_storage().data_ptr()}
            assert all(t.untyped_storage().data_ptr() not in source_ptrs for t in state.named_tensors().values() if t.numel())
            prefix_length=int(positions[-1])+2;prefix_text=torch.arange(prefix_length,device=device)
            prefix_text=prefix_text[~torch.isin(prefix_text,positions)]
            for qn in (1,9):
                qp=torch.arange(prefix_length,prefix_length+qn,device=device);tp=torch.cat((prefix_text,qp))
                tk=torch.randn((1,hkv,len(tp),128),device=device,dtype=torch.float16)*.4;tv=torch.randn_like(tk)*.3
                q=torch.randn((1,hq,qn,128),device=device,dtype=torch.float16)*.4
                oracle=dense_reference(q,state,tk,tv,tp,qp,env)
                if device.type=='cuda':
                    def forbidden(*args,**kwargs):raise AssertionError('Packed reader materialized full visual KV')
                    state.materialize=forbidden
                    try:actual=state.attend(q,exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,backend='triton_packed')
                    finally:del state.materialize
                else:actual=state.attend(q,exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,backend='torch_dequant_reference')
                torch.testing.assert_close(actual,oracle,atol=.003,rtol=.005);assert torch.isfinite(actual).all()
                error=(actual.float()-oracle.float()).abs()
                result['tests'].append({'status':'PASS','N':n,'Q':qn,'Hq':hq,'Hkv':hkv,'bits':2,'G':32,'R':32,
                    'source_code_scale_min_exact':True,'V_groups_per_D128_token':4,'K_tail_tokens':n-state.nkq,'V_tail_tokens':n-state.nvq,
                    'owner_bytes':amounts,'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),
                    'packed_materialize_forbidden':device.type=='cuda'})
        print(json.dumps({'N_completed':n,'tests':len(result['tests'])}),flush=True)
    if device.type=='cuda':torch.cuda.synchronize();result['peak_allocated_bytes_with_dense_controls']=torch.cuda.max_memory_allocated(device)
    files=[Path(__file__),ROOT/'code/methods/kivi_example_adapter.py',ROOT/'code/methods/kivi_example_reader.py',
           ROOT/'third_party/kivi/example.py',ROOT/'audit/upstream/kivi_example_llama_kivi_source_v1.py',
           ROOT/'configs/methods/kivi_example_source_v1.json',ROOT/'third_party/kivi/quant/new_pack.py']
    result['source_hashes']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['status']='PASS';return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:result=run(torch.device(args.device))
    except Exception:
        result={'status':'FAILED_RUN','device':args.device,'traceback':traceback.format_exc()}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
        raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'tests':len(result['tests']),'GPU_encoder_reader':result['actual_gpu_encoder_and_packed_reader']}))
