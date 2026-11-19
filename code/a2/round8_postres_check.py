"""CPU integration/guards and opt-in8-case actual unchanged-reader check."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType
from unittest.mock import patch

import torch

from round8_codec import encode_key,decode_key,key_template,key_amplitude,pack_nibbles
from round6_depth_codec import encode_sign_depth,decode_sign_stage,plan_depth_stages
from round8_postres_direction_v2 import recode_key_postres_direction,NORM_REL_TOL
from round8_postres_direction import recode_key_postres_direction as frozen_v1_recode
from round8_postres_runner import (postres_class,validate_variant,validate_config,
    source_fingerprints,sha,scoped_registration,backend_registration,finalize_run)
from round8_check import frozen_reference_functions
from quantizers import binary_centroids


def require(value,message):
    if not value:raise RuntimeError(message)


def rejected(call):
    try:call()
    except (ValueError,RuntimeError):return True
    return False


def physical_stats(z,y):
    zd,yd=z.double(),y.double();norm=yd.square().sum(-1).sqrt()
    cosine=(zd*yd).sum(-1)/(zd.square().sum(-1).sqrt()*norm).clamp_min(torch.finfo(torch.float64).tiny)
    return norm,cosine,(zd-yd).square().sum(-1)


def owned_fields(state):
    return sum(getattr(state,name).numel()*getattr(state,name).element_size() for name in
        ('direction_code','direction_gain','v_code','v_scale','v_min','depth_residual_code','depth_residual_alpha'))


def cpu_controls(output):
    torch.set_num_threads(1);torch.manual_seed(20261118)
    decoder,fold=frozen_reference_functions()
    book=torch.randn(16,4).half();bookpath=output/'synthetic_book.json'
    bookpath.write_text(json.dumps(dict(centers=book.float().tolist())),encoding='utf-8')
    local=[('v',1,0),('k',3,0),('k',1,0),('v',3,0),('v',0,0)]
    first=[[axis,0,head,stage] for axis,head,stage in local]
    all_stages=[[axis,layer,head,0] for layer in range(28) for head in range(4) for axis in ('k','v')]
    # Exactly2 selected K heads with interleaved V slots, within a complete224 rank.
    first += [item for item in all_stages if item[0]=='v' and item not in first][:38]
    priority=first+[item for item in all_stages if item not in first]
    common=dict(method='round8_joint',key_assignment='l2',key_amplitude='norm',key_encoding='reencode',
        value_encoding='binary',v_group_size='all',joint_gain_refit=False,value_joint_refit=False,
        direction_book_path=str(bookpath),metadata_reference_book_path=str(bookpath),metadata_factor_path='CPU-only',
        round8_factor_policy='inherit_from_reference_book',selected_stages=priority,res_stages_limit=43,budget_bits=1.8,cap=8)
    off=dict(common,name='original',postres_direction=False)
    on=dict(common,name='postres',postres_direction=True,postres_row_chunk=4096)
    declarations=validate_config(dict(variants=[dict(name='Full',method='full'),off,on]))
    require(rejected(lambda:validate_variant(dict(on,joint_gain_refit=True))),'Pre-NNLS accepted')
    require(rejected(lambda:validate_variant(dict(on,key_assignment='row_cosine_cd'))),'CD initialization accepted')
    require(rejected(lambda:validate_variant(dict(on,postres_row_chunk=0))),'Invalid row chunk accepted')
    invalid=list(map(list,priority));invalid[0][3]=1
    require(rejected(lambda:validate_variant(dict(on,selected_stages=invalid))),'Depth2 selected ranking accepted')
    n=32;ks=torch.randn(1,4,n,128);vs=torch.randn(1,4,n,128).half();vs[:,:,:,0]=.125
    factors=torch.ones(28,4);factors[0]=torch.tensor([.91,.98,1.04,1.11])
    class Base:
        settings={}
        def _initialize(self,k,v,*,fail=False):
            cfg=type(self).settings;self.book=book;self.encoder_factors={'key_head_factors':factors}
            self.joint_gain_enabled=False;self.requested_residual_stages=[tuple(x) for x in cfg['selected_stages']]
            self.selected_residual_stages=self.requested_residual_stages[:43];self.ranking_constant_bytes=448
            self.depth_distortions=[]
            kc,kg=encode_key(k,book,gain_factor=factors[0][None,:,None,None])
            record=binary_centroids(v.transpose(2,3).contiguous(),group_size=n)
            vc,scale,minimum=fold(record,1,torch.tensor([.86,.95,1.06,1.14])[None,:,None,None],torch.tensor([-.1,.03,.12,-.06])[None,:,None,None])
            self.state=SimpleNamespace(direction_code=kc,direction_gain=kg,v_code=vc,v_scale=scale,v_min=minimum,
                depth_residual_code=torch.empty(len(local),n,4,dtype=torch.int32),depth_residual_alpha=torch.empty(len(local),n,dtype=torch.float16),fixed_signs={})
            self.initial_J=kc.clone();self.initial_gain=kg.clone()
            self.pointers=tuple(getattr(self.state,key).data_ptr() for key in ('direction_code','direction_gain','depth_residual_code','depth_residual_alpha'))
            if fail:raise ValueError('intentional constructor failure')
            self._encode_axis(0,self.state,'k',local,k[0],decode_key(kc,kg,book)[0])
            vb=decoder(vc,scale,minimum,n,1).transpose(2,3)[0]
            self._encode_axis(0,self.state,'v',local,v[0],vb)
        def _encode_axis(self,layer,state,axis,selected,source,base):
            for head in sorted({head for a,head,_ in selected if a==axis}):
                depth=sum(a==axis and h==head for a,h,_ in selected)
                rc,ra,_,errors=encode_sign_depth(source[head],base[head],depth)
                for stage in range(depth):
                    slot=selected.index((axis,head,stage));state.depth_residual_code[slot].copy_(rc[stage]);state.depth_residual_alpha[slot].copy_(ra[stage]);state.fixed_signs[slot]=rc[stage].clone()
                self.depth_distortions.append(dict(axis=axis,layer=layer,head=head,depth=depth,squared_error_by_depth=errors))
        def result_metadata(self):return dict(key_gain_formula='original norm formula',key_gain_rule='norm')
    cls=postres_class(Base);cells=[]
    for variant in (off,on):
        cls.settings=variant;instance=cls();instance._initialize(ks,vs);cells.append(instance)
        require(tuple(getattr(instance.state,key).data_ptr() for key in ('direction_code','direction_gain','depth_residual_code','depth_residual_alpha'))==instance.pointers,'Physical slots were replaced')
        require(all(torch.equal(instance.state.depth_residual_code[slot],value) for slot,value in instance.state.fixed_signs.items()),'Residual sign changed')
        require(len(instance.postres_statistics)==(2 if variant['postres_direction'] else 0),'Selected K coverage mismatch')
    old,new=cells;before_bytes,after_bytes=owned_fields(old.state),owned_fields(new.state)
    require(before_bytes==after_bytes,'Physical field bytes changed')
    for key in ('v_code','v_scale','v_min'):require(torch.equal(getattr(old.state,key),getattr(new.state,key)),'V fields changed')
    for slot,(axis,head,_) in enumerate(local):
        if axis=='v':require(torch.equal(old.state.depth_residual_alpha[slot],new.state.depth_residual_alpha[slot]),'V alpha slot changed')
    for head in (0,2):
        require(torch.equal(old.state.direction_code[0,head],new.state.direction_code[0,head]) and torch.equal(old.state.direction_gain[0,head],new.state.direction_gain[0,head]),'Unselected K head changed')
    for head in (1,3):
        slot=local.index(('k',head,0))
        old_y=decode_sign_stage(decode_key(old.state.direction_code[0,head],old.state.direction_gain[0,head],book),old.state.depth_residual_code[slot],old.state.depth_residual_alpha[slot])
        new_y=decode_sign_stage(decode_key(new.state.direction_code[0,head],new.state.direction_gain[0,head],book),new.state.depth_residual_code[slot],new.state.depth_residual_alpha[slot])
        r0,c0,e0=physical_stats(ks[0,head],old_y);r1,c1,e1=physical_stats(ks[0,head],new_y)
        require(bool((e1<=e0).all()) and bool((c1>=c0).all()) and bool(((r1/r0-1).abs()<=NORM_REL_TOL).all()),'Physical final guards failed')
        entry=next(x for x in new.depth_distortions if x['axis']=='k' and x['head']==head)
        require(entry['squared_error_by_depth'][1]==float(e1.sum()) and entry['postres_direction'],'Final K distortion stale')
    require(sum(s['changed_rows'] for s in new.postres_statistics)>0,'Synthetic integration did not exercise non-noop updates')
    metadata=new.result_metadata()
    require(metadata['key_gain_rule']=='postres-fixed-original-final-norm-effective-coefficients' and metadata['postres_added_retained_tensor_bytes']==0,'Final coefficient metadata incorrect')
    # Given-J verification must not rerun argmin: rebuild norm/MAD around a
    # deliberately different provided J, then require it is accepted as input.
    z=torch.randn(3,128);given=pack_nibbles(torch.full((3,32),15,dtype=torch.uint8));t=key_template(given,book)
    g=(key_amplitude(z,t,'norm').half().float()*1.037).half();rc,ra,_,_=encode_sign_depth(z,decode_key(given,g,book),1)
    out=recode_key_postres_direction(z,book,given,g,rc[0],ra[0],gain_factor=1.037,row_chunk=4096)
    require(out[3]['initial_given_J_not_reassigned'],'Given-J initialization was reassigned')
    require(rejected(lambda:recode_key_postres_direction(z,book,given,g,rc[0],ra[0],gain_factor=None)),'Wrong actual factor accepted')
    require(rejected(lambda:recode_key_postres_direction(z,book,given,g,rc[0],(ra[0].float()+.125).half(),gain_factor=1.037)),'Prechanged residual alpha accepted')
    a=recode_key_postres_direction(z,book,given,g,rc[0],ra[0],gain_factor=1.037,row_chunk=1)
    b=recode_key_postres_direction(z,book,given,g,rc[0],ra[0],gain_factor=1.037,row_chunk=4096)
    require(all(torch.equal(x,y) for x,y in zip(a[:3],b[:3])),'CPU row chunks changed fields')
    slot=local.index(('k',1,0))
    original_args=(ks[0,1],book,old.state.direction_code[0,1],old.state.direction_gain[0,1],old.state.depth_residual_code[slot],old.state.depth_residual_alpha[slot])
    frozen=frozen_v1_recode(*original_args,gain_factor=factors[0,1],row_chunk=4096)
    updated=recode_key_postres_direction(*original_args,gain_factor=factors[0,1],row_chunk=4096)
    require(all(torch.equal(x,y) for x,y in zip(frozen[:3],updated[:3])),'Given-J validation update changed the actual search on original L2 inputs')
    # Explicit None is valid only for an absent factor, using unmodified fields.
    j0,g0=encode_key(z,book);r0,a0,_,_=encode_sign_depth(z,decode_key(j0,g0,book),1)
    require(recode_key_postres_direction(z,book,j0,g0,r0[0],a0[0],gain_factor=None)[3]['initialization_verified_bitwise'],'Absent-factor initialization failed')
    factory,executor=SimpleNamespace(bank_class=object()),SimpleNamespace(bank_class=object())
    pf,pe=factory.bank_class,executor.bank_class
    try:
        with scoped_registration(factory,executor,object()):cls.settings=on;cls()._initialize(ks,vs,fail=True)
    except ValueError:pass
    require(factory.bank_class is pf and executor.bank_class is pe,'Exception did not restore registration')
    # Public registration context uses the real class factory, with only import
    # surfaces stubbed to avoid model/Triton imports on this CPU host.
    mods={name:ModuleType(name) for name in ('round8_runner','round8_bank','factory','run_representation_variants')}
    mods['round8_bank'].Round8Bank=Base;mods['factory'].bank_class=lambda method:('old',method);mods['run_representation_variants'].bank_class=pe
    before=mods['factory'].bank_class
    try:
        with patch.dict(sys.modules,mods):
            with backend_registration('postres') as selected_cls:
                require(issubclass(selected_cls,Base) and mods['factory'].bank_class('round8_joint') is selected_cls,'Wrong public backend class')
                raise ValueError('intentional registration exit')
    except ValueError:pass
    require(mods['factory'].bank_class is before and mods['run_representation_variants'].bank_class is pe,'Public registration did not restore')
    nv,lh=1792,112;base=lh*nv*34+lh*128*6+128+nv*8
    plan=plan_depth_stages(layers=28,heads=4,visual_tokens=nv,visual_full_bytes=lh*nv*512,base_bytes=base,factor_bytes=1344,budget_bits=1.8,selected_stages=priority,stage_limit=43)
    require(plan['predicted_bytes']==8315013 and len(plan['ranking_constant'])==448,'Complete owner/rank differs')
    fixture=output/'postcheck_stub';fixture.mkdir()
    (fixture/'manifest.json').write_text(json.dumps(dict(config=dict(variants=[off,on]),config_sha256='CPU-config',external_codec_inputs={})),encoding='utf-8')
    (fixture/'records.jsonl').write_text('{"CPU_stub":0}\n{"CPU_stub":1}\n',encoding='utf-8')
    (fixture/'complete.json').write_text(json.dumps(dict(status='COMPLETE',records=2,expected=2)),encoding='utf-8')
    final=finalize_run(fixture,{str(bookpath):sha(bookpath)},source_fingerprints(),'CPU-config',declarations)
    require(final['complete_round8_sha256']==sha(fixture/'complete_round8.json') and final['records_sha256']==sha(fixture/'records.jsonl'),'Final postres gate binding failed')
    return dict(passed=True,selected_K_heads=2,same_head_interleaved_KV_slots=True,original_whole_layer_encoding=True,
        given_J_not_reassigned=True,actual_per_head_factor_and_explicit_None=True,pre_NNLS_rejected=True,
        changed_rows=sum(s['changed_rows'] for s in new.postres_statistics),postres_stats=new.postres_statistics,
        stored_field_bytes_before=before_bytes,stored_field_bytes_after=after_bytes,complete_owner_N1792=8315013,
        full_ranking_entries=224,ranking_bytes=448,selected_stages=43,V_signs_bits_unchanged=True,
        zero_extra_tensor_fields=True,CPU_chunk1_4096_bitwise=True,final_distortion_updated=True,
        v1_search_fields_unchanged_for_original_L2_inputs=True,
        registration_exception_restore=True,complete_postres_hash_binding=True,
        scope='CPU local multihead integration plus separate complete-owner algebra; no full model/task/timing measurement')


def gpu_controls():
    if not torch.cuda.is_available():raise RuntimeError('--gpu requires actual CUDA')
    from fast_wht import torch_wht
    from round8_attention import attention
    from round7_key_attention import attention as frozen_attention
    torch.manual_seed(20261118);torch.backends.cuda.matmul.allow_tf32=False
    decoder,fold=frozen_reference_functions();rows=[];initialization=[]
    for n in (64,256,1792):
        hk,hq=4,28
        source_k=torch.randn(1,hk,n,128,device='cuda',dtype=torch.float16);source_v=torch.randn_like(source_k)
        mean=source_k.float().mean(-2,keepdim=True).half();rotated=torch_wht(source_k.float()-mean.float())
        book=torch.randn(16,4,device='cuda',dtype=torch.float16);f=torch.tensor([.91,.98,1.04,1.11],device='cuda')
        kc,kg=encode_key(rotated,book,gain_factor=f[None,:,None,None])
        vc,vs,vm=fold(binary_centroids(source_v.transpose(2,3).contiguous(),group_size=n),1,
            torch.tensor([.86,.95,1.06,1.14],device='cuda')[None,:,None,None],torch.tensor([-.1,.03,.12,-.06],device='cuda')[None,:,None,None])
        bk=decode_key(kc,kg,book);bv=decoder(vc,vs,vm,n,1).transpose(2,3).contiguous()
        kr,ka,_,_=encode_sign_depth(rotated[0,1],bk[0,1],1);vr,va,_,_=encode_sign_depth(source_v[0,1],bv[0,1],1)
        result=recode_key_postres_direction(rotated[0,1],book,kc[0,1],kg[0,1],kr[0],ka[0],gain_factor=f[1],row_chunk=4096)
        if n==64:
            other=recode_key_postres_direction(rotated[0,1],book,kc[0,1],kg[0,1],kr[0],ka[0],gain_factor=f[1],row_chunk=16)
            require(all(torch.equal(a,b) for a,b in zip(result[:3],other[:3])),'CUDA row_chunk16/4096 changed physical fields')
        old_y=decode_sign_stage(bk[0,1],kr[0],ka[0]);new_y=decode_sign_stage(decode_key(result[0],result[1],book),kr[0],result[2])
        r0,c0,e0=physical_stats(rotated[0,1],old_y);r1,c1,e1=physical_stats(rotated[0,1],new_y)
        require(bool((e1<=e0).all()) and bool((c1>=c0).all()) and bool(((r1/r0-1).abs()<=NORM_REL_TOL).all()),'CUDA final norm/cos/SSE guards failed')
        initialization.append(dict(nv=n,whole_layer_to_head_given_J=True,stats=result[3],chunk16_4096_bitwise=True if n==64 else None))
        if n==1792:continue  # Real-N initialization/guards; keep reader grid at8calls.
        for enabled in (False,True):
            lookup=torch.full((2,hk,2),-1,device='cuda',dtype=torch.int32);lookup[1,1,0]=0;lookup[0,1,0]=1
            state=SimpleNamespace(direction_mean=mean,direction_code=kc.clone(),direction_gain=kg.clone(),
                v_code=vc,v_scale=vs,v_min=vm,depth_residual_code=torch.stack((vr[0],kr[0])),
                depth_residual_alpha=torch.stack((va[0],ka[0])),depth_residual_lookup=lookup)
            if enabled:
                state.direction_code[0,1].copy_(result[0]);state.direction_gain[0,1].copy_(result[1]);state.depth_residual_alpha[1].copy_(result[2])
            for nq in (1,11):
                prefix=41;nt=prefix+nq;torch.manual_seed(20261118+n+nq)
                state.visual_pos=torch.arange(0,2*n,2,device='cuda');state.text_pos=torch.cat((torch.arange(1,2*prefix,2,device='cuda'),torch.arange(2*n,2*n+nq,device='cuda')))
                state.text_length,state.length=nt,2*n+nq
                state.text_k=torch.randn(1,hk,nt+16,128,device='cuda',dtype=torch.float16);state.text_v=torch.randn_like(state.text_k)
                q=torch.randn(1,nq,hq,128,device='cuda',dtype=torch.float16).transpose(1,2)
                actual=attention(q,state,book,n)
                require(torch.equal(actual,frozen_attention(q,state,book,n)),'Frozen K4/binary reader identity differs')
                dk=decode_key(state.direction_code,state.direction_gain,book);dv=decoder(vc,vs,vm,n,1).transpose(2,3).contiguous()
                dk[0,1]=decode_sign_stage(dk[0,1],state.depth_residual_code[1],state.depth_residual_alpha[1]);dv[0,1]=decode_sign_stage(dv[0,1],state.depth_residual_code[0],state.depth_residual_alpha[0])
                qr=torch_wht(q).half().float();zv=qr@dk.repeat_interleave(7,1).float().transpose(-1,-2)/128**.5
                zv+=(q.float()*mean.repeat_interleave(7,1).float()).sum(-1,keepdim=True)/128**.5
                zt=q.float()@state.text_k[:,:,:nt].repeat_interleave(7,1).float().transpose(-1,-2)/128**.5
                score=torch.cat((zv,zt),-1);positions=torch.cat((state.visual_pos,state.text_pos));qpos=torch.arange(state.length-nq,state.length,device='cuda')
                score.masked_fill_(positions[None,None,None,:]>qpos[None,None,:,None],-torch.inf)
                values=torch.cat((dv,state.text_v[:,:,:nt]),2).repeat_interleave(7,1);expected=(score.softmax(-1)@values.float()).half()
                delta=actual.float()-expected.float();maximum=float(delta.abs().max());relative=float(delta.norm()/expected.float().norm().clamp_min(1e-20))
                rows.append(dict(nv=n,nq=nq,postres_direction=enabled,query_heads=hq,kv_heads=hk,
                    max_abs=maximum,relative_l2=relative,passed=maximum<.003 and relative<.003,frozen_reader_bitwise_equal=True))
    require(len(rows)==8 and all(row['passed'] for row in rows),'Actual final-field reader/dense check failed')
    return dict(reader_cases=rows,whole_layer_initialization_and_guard_cases=initialization,
        scope='CUDA synthetic current physical fields and unchanged actual reader only; no model/quality/timing result')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sources=source_fingerprints();cpu=cpu_controls(args.output)
    with torch.inference_mode():gpu=gpu_controls() if args.gpu else None
    require(source_fingerprints()==sources,'Sources changed during check')
    report=dict(status='PASS',CPU_controls=cpu,GPU_controls=gpu,GPU_executed=bool(args.gpu),source_sha256=sources,
        checker_sha256=sha(__file__),frozen_v1_helper_sha256=sha(Path(__file__).with_name('round8_postres_direction.py')),
        scope='synthetic arithmetic/integration only; no model/task/timing measurement')
    (args.output/'check.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_executed=bool(args.gpu),GPU_reader_cases=8 if args.gpu else 0,
        report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',GPU_executed=bool(args.gpu),GPU_reader_cases=8 if args.gpu else 0)),flush=True)


if __name__=='__main__':main()
