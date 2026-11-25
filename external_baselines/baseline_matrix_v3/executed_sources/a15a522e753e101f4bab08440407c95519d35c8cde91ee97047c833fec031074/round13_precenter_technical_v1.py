"""CPU layout checks and later CUDA QA0 for the precenter prototype.

No model is loaded here. A task-result claim requires the independent whole-
model runner. Root owns any later GPU launch; importing this file does not run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import r7_legacy_bridge_v1 as bridge
import round12_static_bank_v1 as old
import round13_precenter_bank_v1 as bank


def require(value,message):
    if not value:raise ValueError('ROUND13_PRECENTER_TECHNICAL: '+message)


def bits_equal(a,b):
    return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def field_diagnostics(state,expected_s,expected_b,actual_s,actual_b,cos,sin):
    fields={}
    for name,expected,actual in (('s',expected_s,actual_s),('b',expected_b,actual_b)):
        mismatch=expected.contiguous().view(torch.int16)!=actual.contiguous().view(torch.int16)
        numeric=expected!=actual
        error=(expected.float()-actual.float()).abs()
        coords=mismatch.nonzero(as_tuple=False)[:16].cpu().tolist()
        samples=[]
        for _,head,row,dim in coords:
            m=state.round13_precenter_mu[0,head,0,dim]
            mp=state.round13_precenter_mu[0,head,0,(dim+64)%128]
            mp=-mp if dim<64 else mp
            aa=state.direction_mean[0,head,0,dim];gg=state.round13_precenter_gain[head]
            cc=cos[0,0,row,dim];si=sin[0,0,row,dim]
            p0=(m*cc).half();p1=(mp*si).half()
            ss=expected_s[0,head,row,dim]
            delta=ss.float()-aa.float();mul=gg.float()*delta;add=aa.float()+mul
            samples.append({'head':head,'row':row,'dim':dim,'mu':float(m),'mu_pair_signed':float(mp),
                'a':float(aa),'g':float(gg),'cos':float(cc),'sin':float(si),
                'torch_p0_Half':float(p0),'torch_p1_Half':float(p1),
                'torch_sum_FP32_before_Half':float(p0.float()+p1.float()),
                'torch_expected_s':float(ss),'actual_s':float(actual_s[0,head,row,dim]),
                'torch_affine_diff32':float(delta),'torch_affine_mul32':float(mul),
                'torch_affine_add32_before_Half':float(add),
                'expected':float(expected[0,head,row,dim]),'actual':float(actual[0,head,row,dim]),
                'expected_Half_bits':int(expected[0,head,row,dim].view(torch.int16))&65535,
                'actual_Half_bits':int(actual[0,head,row,dim].view(torch.int16))&65535})
        fields[name]={'bit_mismatch_count':int(mismatch.sum()),'numeric_mismatch_count':int(numeric.sum()),
            'max_abs':float(error.max()),'mean_abs':float(error.mean()),
            'signed_zero_only':bool(mismatch.any()) and not bool(numeric.any()),'samples':samples}
    repeated_c,repeated_s=bank.phase_tables(state.visual_pos,state.round13_precenter_freq,
        state.round13_precenter_scaling,rotation_constant=state.round13_precenter_rotation_constant)
    return {'fields':fields,'phase_tables_repeat_bit_identity':bits_equal(cos,repeated_c) and bits_equal(sin,repeated_s),
        'mu_stride':list(state.round13_precenter_mu.stride()),'mean_stride':list(state.direction_mean.stride()),
        'factor_stride':list(state.round13_precenter_gain.stride()),'cos_stride':list(cos.stride()),
        'autocast_enabled':torch.is_autocast_enabled(),
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision()}


def synthetic_R(n):
    """Actual M1-shaped buffers and frozen-format constant sizes, CPU only."""
    stages=bank.r_count(n)
    ranking=[]
    for i in range(112):
        ranking.extend([['k',i//4,i%4,0],['v',i//4,i%4,0]])
    selected=ranking[:stages]
    lookup=torch.full((28,2,4,2),-1,dtype=torch.int32)
    shared={'book':torch.randn(16,4).half(),'global_residual_lookup':lookup,
        'encoder_factor.key_head_factors':torch.rand(28,4)+.5,
        'encoder_factor.value_head_factors':torch.rand(28,4)+.5,
        'encoder_factor.value_center_offsets':torch.randn(28,4),
        'encoder_ranking_bytes':torch.zeros(448,dtype=torch.uint8),
        'encoder_policy_bytes':torch.zeros(5,dtype=torch.uint8),
        'round12_format':torch.tensor([1,4,4,0],dtype=torch.uint8)}
    vp=torch.arange(n,dtype=torch.long);owners=[]
    for layer in range(28):
        entries=old._layer_entries(selected,layer)
        s=SimpleNamespace(visual_pos=vp,round12_vmode=0,
            direction_code=torch.randint(0,256,(1,4,n,16),dtype=torch.uint8),
            direction_gain=torch.randn(1,4,n,1).half(),
            direction_mean=torch.randn(1,4,1,128).half(),
            v_code=torch.randint(-(2**31),2**31-1,(1,4,128,n//32),dtype=torch.int32),
            v_scale=torch.rand(1,4,128,1).half(),v_min=torch.randn(1,4,128,1).half(),
            depth_residual_lookup=lookup[layer])
        if entries:
            s.depth_residual_code=torch.randint(-(2**31),2**31-1,(len(entries),n,4),dtype=torch.int32)
            s.depth_residual_alpha=torch.randn(len(entries),n).half()
        for slot,(axis,head,depth) in enumerate(entries):
            lookup[layer,int(axis=='v'),head,depth]=slot
        owners.append(old.Round12Owner(layer,s,shared,None,n))
    require(old.unique_capacity(owners)==bank.budget_plan(n)['R_cap_bytes'],'synthetic R actual bytes')
    return owners,selected


def cpu_checks():
    torch.manual_seed(130065)
    layouts=[]
    for n in (256,768,1280,1792,2304):
        owners,original=synthetic_R(n);shared=owners[0].shared
        plan=bank.budget_plan(n);selected,dropped=bank.donor_subset(original,plan['K_donor_stages'])
        original_V={}
        for o in owners:
            original_V[o.layer_idx]={'code':o.state.v_code.clone(),'scale':o.state.v_scale.clone(),
                'min':o.state.v_min.clone(),'res':{(head,depth):
                    (o.state.depth_residual_code[slot].clone(),o.state.depth_residual_alpha[slot].clone())
                    for slot,(axis,head,depth) in enumerate(old._layer_entries(original,o.layer_idx)) if axis=='v'}}
        lookup=shared['global_residual_lookup'];lookup.fill_(-1)
        for o in owners:
            bank.reduce_residual_pool(o.state,original,selected,o.layer_idx,lookup)
            previous=original_V[o.layer_idx]
            require(bits_equal(o.state.v_code,previous['code']) and bits_equal(o.state.v_scale,previous['scale'])
                    and bits_equal(o.state.v_min,previous['min']),'V base fields changed')
            for slot,(axis,head,depth) in enumerate(old._layer_entries(selected,o.layer_idx)):
                if axis=='v':
                    c,a=previous['res'][(head,depth)]
                    require(bits_equal(o.state.depth_residual_code[slot],c) and
                            bits_equal(o.state.depth_residual_alpha[slot],a),'logical V residual changed')
        donor_actual=old.unique_capacity(owners)
        require(donor_actual==plan['donor_only_actual_bytes'],'actual donor capacity')
        freq_input=torch.randn(64,dtype=torch.float32)
        freq=freq_input.clone();mu=torch.zeros(28,4,1,128,dtype=torch.float16)
        shared.update(round13_precenter_freq=freq,round13_precenter_mu=mu,
            round13_precenter_scaling=torch.tensor([1.],dtype=torch.float32),
            round13_precenter_format=torch.tensor([13,1,0,0],dtype=torch.uint8))
        result=[]
        for o in owners:
            s=o.state;s.round13_precenter_mu=mu[o.layer_idx].unsqueeze(0)
            s.round13_precenter_freq=freq
            s.round13_precenter_scaling=shared['round13_precenter_scaling']
            s.round13_precenter_gain=shared['encoder_factor.key_head_factors'][o.layer_idx]
            result.append(bank.PrecenterOwner(o.layer_idx,s,shared,None,n))
        actual=old.unique_capacity(result)
        require(actual==plan['PC_expected_actual_bytes'] and actual<=plan['R_cap_bytes'],'actual PC capacity/cap')
        require(bridge._storage(freq)!=bridge._storage(freq_input),'frequency input alias')
        freq_input.zero_();require(bool((freq!=0).any()),'live frequency changes propagated')
        layouts.append({**plan,'actual_donor_bytes':donor_actual,'actual_PC_bytes':actual,
            'all_R_V_fields_bit_preserved':True,'input_frequency_storage_not_retained':True})
    n=256
    pos=torch.arange(n,dtype=torch.long)
    inv=1/(1000000.**(torch.arange(0,128,2,dtype=torch.float32)/128))
    c,s=bank.phase_tables(pos,inv,1.)
    mu=torch.randn(1,4,1,128).half()
    a=torch.randn_like(mu);g=(torch.rand(1,4,1,1)+.5)
    ss,bb=bank.position_mean_bias(mu,a,g,c,s)
    require(ss.dtype==bb.dtype==torch.float16 and ss.shape==(1,4,n,128),'deployed Half mean/bias')
    c0,s0=bank.phase_tables(pos,inv,1.,rotation_constant=True)
    ss0,bb0=bank.position_mean_bias(a,a,g,c0,s0,rotation_constant=True)
    require(bits_equal(bb0,a.expand_as(bb0)),'s=a did not yield exact original mean')
    zero=torch.zeros_like(mu)
    _,bzero=bank.position_mean_bias(zero,a,g,c,s)
    require(bits_equal(bzero,(a.float()+g*(torch.zeros_like(bzero).float()-a.float())).half()),'mu0 affine bias')
    ls=bank.least_squares_mu(ss,c,s)
    ls_error=float((ls.float()-mu.float()).abs().max())
    require(ls_error<.004,'LS failed on an exactly represented rotating mean')
    require(bridge._storage(ls)!=bridge._storage(ss),'LS output retained full source storage')
    return {'status':'PASS_CPU_FORMAT_MATH_ONLY','actual_CUDA':False,'model_calls':0,'task_answer_events':0,
        'layouts':layouts,'shared_Half_rounding_and_anchored_affine_checked':True,
        'constant_rotation_and_mu_equals_a_bias_bit_identity':True,
        'mu_zero_bias_checked':True,'LS_output_not_full_source_alias':True,'LS_rotating_mean_max_abs':ls_error,
        'note':'CPU format/mathematics checks do not establish packed CUDA or task quality'}


def decode_residual(base,state,axis):
    result=base.clone()
    if not hasattr(state,'depth_residual_code'):return result
    shifts=torch.arange(32,device=base.device,dtype=torch.int64)
    for head in range(4):
        for depth in range(2):
            slot=int(state.depth_residual_lookup[axis,head,depth])
            if slot>=0:
                bits=((state.depth_residual_code[slot].long()[...,None]>>shifts)&1).flatten(-2)
                sign=bits.float()*2-1
                result[0,head]=(result[0,head].float()+sign*state.depth_residual_alpha[slot,:,None].float()).half()
    return result


def dense_reference(q,state,book,wht,codec):
    k=decode_residual(codec.decode_key(state.direction_code,state.direction_gain,book),state,0)
    v=decode_residual(codec.half_value(state.v_code,state.v_scale,state.v_min).transpose(2,3).contiguous(),state,1)
    cos,sin=bank.phase_tables(state.visual_pos,state.round13_precenter_freq,state.round13_precenter_scaling,
        rotation_constant=state.round13_precenter_rotation_constant)
    _,bias=bank.position_mean_bias(state.round13_precenter_mu,state.direction_mean,
        state.round13_precenter_gain[None,:,None,None],cos,sin,
        rotation_constant=state.round13_precenter_rotation_constant)
    qr=wht(q,half=True)
    visual=qr.float()@k.repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
    visual+=q.float()@bias.repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
    text=q.float()@state.text_k.repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
    logits=torch.cat((visual,text),-1)
    positions=torch.cat((state.visual_pos,state.text_pos))
    qp=torch.arange(state.length-q.shape[-2],state.length,device=q.device)
    logits.masked_fill_(positions[None,:]>qp[:,None],-float('inf'))
    values=torch.cat((v,state.text_v),2).repeat_interleave(7,1).float()
    return (logits.softmax(-1)@values).half()


@torch.no_grad()
def cuda_checks(a2,visual_tokens,device,*,diagnostics_output=None,dump_field_ptx=False):
    require(torch.cuda.is_available(),'actual CUDA required')
    sys.path.insert(0,str(a2))
    import round12_static_codec_v1 as codec
    legacy_codec=importlib.import_module('round7_key_codec')
    sign=importlib.import_module('round6_depth_codec').encode_sign_depth
    wht=importlib.import_module('fast_wht').wht
    legacy_reader=importlib.import_module('round12_static_attention_v1').attention
    reader_module=importlib.import_module('round13_precenter_attention_v1')
    reader=reader_module.attention
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(130065);results=[]
    for n in visual_tokens:
        source=torch.randn(1,4,n,128,device=device).half()
        book=(torch.randn(16,4,device=device)*.5).half()
        a=source.float().mean(-2,keepdim=True).half()
        factors=(torch.rand(4,device=device)*.5+.75)
        inv=1/(1000000.**(torch.arange(0,128,2,device=device,dtype=torch.float32)/128))
        vp=torch.arange(1,n+1,device=device,dtype=torch.long)
        g=factors[None,:,None,None]
        raw=wht(source.float()-a.float())
        oldcode,oldgain=legacy_codec.encode_key(raw,book,amplitude='norm',assignment='l2',gain_factor=g)
        for constant,zero in ((True,False),(False,False),(False,True)):
            c,s=bank.phase_tables(vp,inv,1.,rotation_constant=constant)
            mu=a.clone() if constant else bank.least_squares_mu(source,c,s)
            if zero:mu.zero_()
            sm,bias=bank.position_mean_bias(mu,a,g,c,s,rotation_constant=constant)
            base_source=wht(source.float()-sm.float())
            target=wht(source.float()-bias.float())
            code,gain=legacy_codec.encode_key(base_source,book,amplitude='norm',assignment='l2',gain_factor=g)
            lookup=torch.full((2,4,2),-1,device=device,dtype=torch.int32)
            lookup[0,0,0]=0;lookup[0,0,1]=1;lookup[1,1,0]=2;lookup[1,1,1]=3
            base=legacy_codec.decode_key(code,gain,book)
            codes,alphas,_,_=sign(target[0,0],base[0,0],2)
            rc=codec.pack_channel_codes(torch.randint(0,2,(4,n,128),device=device),1)
            ra=(torch.rand(4,n,device=device)*.03).half()
            for depth in (0,1):rc[depth].copy_(codes[depth]);ra[depth].copy_(alphas[depth])
            state=SimpleNamespace(direction_code=code,direction_gain=gain,direction_mean=a,
                v_code=codec.pack_channel_codes(torch.randint(0,2,(1,4,128,n),device=device),1),
                v_scale=(torch.rand(1,4,128,1,device=device)+.1).half(),
                v_min=(torch.randn(1,4,128,1,device=device)*.2-.5).half(),round12_vmode=0,
                depth_residual_code=rc,depth_residual_alpha=ra,depth_residual_lookup=lookup,
                visual_pos=vp,text_pos=torch.cat((torch.zeros(1,device=device,dtype=torch.long),
                    torch.arange(n+1,n+21,device=device))),
                text_k=torch.randn(1,4,21,128,device=device).half(),
                text_v=torch.randn(1,4,21,128,device=device).half(),text_length=21,length=n+21,
                round13_precenter_mu=mu,round13_precenter_freq=inv,
                round13_precenter_scaling=torch.tensor([1.],device=device),
                round13_precenter_gain=factors,round13_precenter_rotation_constant=constant)
            q=torch.randn(1,28,17,128,device=device).half()
            deployed_s,deployed_b,compiled=reader_module.materialize_position_fields(state,return_kernel=True)
            diagnostic=field_diagnostics(state,sm,bias,deployed_s,deployed_b,c,s)
            diagnostic.update(N=n,constant_rotation=constant,mu_zero=zero)
            if diagnostics_output is not None:
                diagnostics_output.mkdir(parents=True,exist_ok=True)
                stem=f'field_N{n}_constant{int(constant)}_zero{int(zero)}'
                with (diagnostics_output/(stem+'.json')).open('x',encoding='utf-8') as stream:
                    json.dump(diagnostic,stream,ensure_ascii=False,allow_nan=False,indent=2)
                if dump_field_ptx:
                    for kind in ('ptx','llir','ttir'):
                        if kind in compiled.asm:
                            with (diagnostics_output/(stem+'.'+kind)).open('x',encoding='utf-8') as stream:
                                stream.write(compiled.asm[kind])
            print(json.dumps({'event':'FIELD_CHECK','N':n,'constant':constant,'mu_zero':zero,
                's_bit_mismatch_count':diagnostic['fields']['s']['bit_mismatch_count'],
                'b_bit_mismatch_count':diagnostic['fields']['b']['bit_mismatch_count'],
                's_max_abs':diagnostic['fields']['s']['max_abs'],
                'b_max_abs':diagnostic['fields']['b']['max_abs'],
                'phase_repeat_equal':diagnostic['phase_tables_repeat_bit_identity']}),flush=True)
            require(bits_equal(deployed_s,sm) and bits_equal(deployed_b,bias),
                    'Triton/PyTorch deployed Half mean/bias fields differ: '+json.dumps({
                        'N':n,'constant':constant,'zero':zero,
                        's':diagnostic['fields']['s']['bit_mismatch_count'],
                        'b':diagnostic['fields']['b']['bit_mismatch_count']}))
            actual=reader(q,state,book,n)
            reference=dense_reference(q,state,book,wht,codec)
            error=(actual.float()-reference.float()).abs()
            require(bool(torch.isfinite(actual).all()) and float(error.max())<=.008,'packed/dense precenter mismatch')
            row={'N':n,'constant_rotation':constant,'mu_zero':zero,
                 'dense_max_abs':float(error.max()),'dense_tolerance':.008,
                 'deployed_mean_and_bias_fields_bit_identity_with_encoder':True,
                 'base_assignment_uses_frozen_legacy_codec':True}
            if constant:
                oldbase=legacy_codec.decode_key(oldcode,oldgain,book)
                oldcodes,oldalphas,_,_=sign(raw[0,0],oldbase[0,0],2)
                row['constant_code_bit_identity']=bits_equal(code,oldcode)
                row['constant_gain_bit_identity']=bits_equal(gain,oldgain)
                row['constant_K_residual_bit_identity']=all(bits_equal(codes[d],oldcodes[d]) and bits_equal(alphas[d],oldalphas[d]) for d in (0,1))
                previous=legacy_reader(q,state,book,n)
                row['constant_reader_bit_identity']=bits_equal(actual,previous)
                row['constant_reader_max_abs']=float((actual.float()-previous.float()).abs().max())
                require(row['constant_code_bit_identity'] and row['constant_gain_bit_identity']
                        and row['constant_K_residual_bit_identity'] and row['constant_reader_bit_identity'],
                        'constant-rotation actual legacy identity failed: '+json.dumps(row))
            results.append(row)
    torch.cuda.synchronize(device)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-output',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--legacy-a2',type=Path)
    p.add_argument('--visual-tokens',nargs='+',type=int,default=[32,256,1792])
    p.add_argument('--diagnostics-output',type=Path)
    p.add_argument('--dump-field-ptx',action='store_true')
    p.add_argument('--device',default='cuda:0');a=p.parse_args();start=time.perf_counter()
    if a.cpu_output:
        result=cpu_checks();path=a.cpu_output
    else:
        require(a.output is not None and a.legacy_a2 is not None,'CUDA output and legacy-a2 required')
        result={'status':'PASS_ACTUAL_CUDA_PRECENTER_QA0','actual_CUDA':True,
            'model_calls':0,'task_answer_events':0,
            'checks':cuda_checks(a.legacy_a2,a.visual_tokens,torch.device(a.device),
                diagnostics_output=a.diagnostics_output or (a.output.parent/'field_diagnostics'),
                dump_field_ptx=a.dump_field_ptx)}
        path=a.output
    result['seconds']=time.perf_counter()-start
    result['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in bank.RUNTIME_FILES}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'path':str(path)}),flush=True)


if __name__=='__main__':main()
