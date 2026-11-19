"""CPU closed-form/physical-goal/slot/bytes controls; no GPU or model."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType
from unittest.mock import patch

import torch

from round8_codec import encode_key,decode_key,key_template
from round6_depth_codec import encode_sign_depth,plan_depth_stages
from quantizers import binary_centroids
from round8_check import frozen_reference_functions
from round8_final_ray import refit_key_final_ray,physical_ray_scale,metrics,final_decode,signs_from_code
from round8_final_ray_runner import (validate_variant,validate_config,final_ray_class,
    backend_registration,source_fingerprints,sha,finalize_run)


def require(value,message):
    if not value:raise RuntimeError(message)


def rejected(call):
    try:call()
    except (ValueError,RuntimeError):return True
    return False


def tensor_bytes(state):
    return sum(getattr(state,k).numel()*getattr(state,k).element_size() for k in
        ('direction_code','direction_gain','v_code','v_scale','v_min','depth_residual_code','depth_residual_alpha'))


def controls(output):
    torch.set_num_threads(1);torch.manual_seed(20261118)
    z=torch.tensor([1.,0.]).repeat(64)[None];book=torch.ones(16,4).half();f=torch.tensor(.1767955869436264)
    code,gain=encode_key(z,book,gain_factor=f);rc,ra,_,_=encode_sign_depth(z,decode_key(code,gain,book),1)
    alpha,sign=ra[0],signs_from_code(rc[0]);t=key_template(code,book);old_y=final_decode(t,gain,alpha,sign)
    before=metrics(z.double(),old_y);require(float(gain)==.125 and float(alpha)==.5,'Legal original norm/MAD witness changed')
    expected={'mse':40/34,'norm':(64/34)**.5,'projection':64/40};theory=[]
    for mode,c in expected.items():
        ng,na,stats=refit_key_final_ray(z,book,code,gain,rc[0],alpha,mode=mode,gain_factor=f)
        after=metrics(z.double(),final_decode(t,ng,na,sign));key='sse' if mode=='mse' else mode
        require(abs(stats['scale_mean']-c)<1e-12 and stats['accepted_rows']==1,'Wrong theoretical complete-state scale')
        require(bool((after[key]<before[key]).all()),'Own actual objective did not improve')
        ideal=metrics(z.double(),old_y.double()*c)
        if mode=='mse':
            for perturb in (-.01,.01):require(bool((metrics(z.double(),old_y.double()*(c+perturb))['sse']>ideal['sse']).all()),'MSE stationary ray check failed')
        else:require(float(ideal[key].max())<1e-12,'Ideal radial target not met')
        if mode in ('norm','projection'):
            require(stats['sse_after']>stats['sse_before'] and stats['accepted_rows_with_SSE_increase']==1 and stats['extra_SSE_gate'] is False,'Hidden SSE gate in non-MSE arm')
        if mode=='mse':
            def source_cos(y):
                yd=y.double();return float((z.double()*yd).sum()/(z.double().norm()*yd.norm()))
            require(source_cos(final_decode(t,ng,na,sign))<source_cos(old_y),'MSE cosine-down acceptance witness lost')
        require(stats['max_old_to_final_direction_cosine_loss']<1e-6,'Ordinary witness no longer approximately follows original direction')
        require(stats['physical_optimized_bytes_before']==stats['physical_optimized_bytes_after']==4,'Physical coefficient bytes changed')
        theory.append(dict(mode=mode,stats=stats))
    off=refit_key_final_ray(z,book,code,gain,rc[0],alpha,mode='off',gain_factor=f)
    require(torch.equal(off[0],gain) and torch.equal(off[1],alpha),'Off fields changed')
    # Generic degenerate-policy tests are physical stress controls, not a claim
    # these tuples came from the original norm/signMAD initializer.
    fallback=[]
    for name,source,template,g,a,s in (
        ('zero_source',torch.zeros(1,128),torch.ones(1,128),torch.ones(1,1).half(),torch.zeros(1).half(),torch.ones(1,128)),
        ('zero_final_state',torch.ones(1,128),torch.ones(1,128),torch.zeros(1,1).half(),torch.zeros(1).half(),torch.ones(1,128)),
        ('negative_dot',-torch.ones(1,128),torch.ones(1,128),torch.ones(1,1).half(),torch.zeros(1).half(),torch.ones(1,128)),
        ('zero_dot',torch.tensor([1.,-1.]).repeat(64)[None],torch.ones(1,128),torch.ones(1,1).half(),torch.zeros(1).half(),torch.ones(1,128)),
        ('candidate_underflow_zero',torch.full((1,128),1e-12),torch.ones(1,128),torch.ones(1,1).half(),torch.zeros(1).half(),torch.ones(1,128)),
        ('candidate_field_overflow',torch.full((1,128),100.),torch.ones(1,128),torch.ones(1,1).half(),torch.full((1,),.99951171875).half(),-torch.ones(1,128))):
        for mode in ('mse','norm','projection'):
            ng,na,stats=physical_ray_scale(source,template,g,s,a,mode)
            require(torch.equal(ng,g) and torch.equal(na,a),'Degenerate policy failed to preserve old fields')
            fallback.append(dict(name=name,mode=mode,accepted_rows=stats['accepted_rows']))
    # A zero individual coefficient is allowed if the complete state is nonzero.
    for g,a in ((torch.zeros(1,1).half(),torch.ones(1).half()),(torch.ones(1,1).half(),torch.zeros(1).half())):
        ng,na,stats=physical_ray_scale(torch.full((1,128),2.),torch.ones(1,128),g,torch.ones(1,128),a,'norm')
        require(stats['accepted_rows']==1,'A zero coefficient was confused with a zero complete state')
    # Metric guard against accidentally substituting squared-norm mismatch.
    zs=torch.zeros(1,128,dtype=torch.float64);zs[0,0]=1
    yo=torch.zeros_like(zs);yo[0,0]=1.5;yn=torch.zeros_like(zs);yn[0,0]=.4
    require(float(metrics(zs,yo)['norm'])<float(metrics(zs,yn)['norm']) and abs(float(yo.square().sum())-1)>abs(float(yn.square().sum())-1),'Norm vs squared-norm metric control failed')
    # Random original verified tuples obey only their named acceptance metric.
    source=torch.randn(31,128);random_book=torch.randn(16,4).half();factor=1.073
    cj,gg=encode_key(source,random_book,gain_factor=factor);rr,aa,_,_=encode_sign_depth(source,decode_key(cj,gg,random_book),1)
    copies=[v.clone() for v in (source,random_book,cj,gg,rr[0],aa[0])]
    template=key_template(cj,random_book);ss=signs_from_code(rr[0]);baseline=metrics(source.double(),final_decode(template,gg,aa[0],ss))
    for mode in ('mse','norm','projection'):
        ng,na,stats=refit_key_final_ray(source,random_book,cj,gg,rr[0],aa[0],mode=mode,gain_factor=factor)
        actual=metrics(source.double(),final_decode(template,ng,na,ss));key='sse' if mode=='mse' else mode
        require(bool((actual[key]<=baseline[key]).all()),'Random actual own-objective guard failed')
    require(all(torch.equal(a,b) for a,b in zip(copies,(source,random_book,cj,gg,rr[0],aa[0]))),'Input/J/sign mutated')
    require(rejected(lambda:refit_key_final_ray(source,random_book,cj,gg,rr[0],aa[0],mode='mse',gain_factor=None)),'Wrong initialization factor accepted')
    # Tiny real class integration: one selected K head with V in another slot.
    n=32;decoder,_=frozen_reference_functions();bookpath=output/'synthetic_book.json'
    bookpath.write_text(json.dumps(dict(centers=book.float().tolist())),encoding='utf-8')
    local=[('v',1,0),('k',1,0)];first=[['v',0,1,0],['k',0,1,0]]
    all_stages=[[axis,layer,head,0] for layer in range(28) for head in range(4) for axis in ('k','v')]
    first += [item for item in all_stages if item[0]=='v' and item not in first][:41]
    priority=first+[item for item in all_stages if item not in first]
    common=dict(method='round8_joint',key_assignment='l2',key_amplitude='norm',key_encoding='reencode',value_encoding='binary',v_group_size='all',
        joint_gain_refit=False,value_joint_refit=False,direction_book_path=str(bookpath),metadata_reference_book_path=str(bookpath),metadata_factor_path='CPU-only',
        round8_factor_policy='inherit_from_reference_book',selected_stages=priority,res_stages_limit=43,budget_bits=1.8,cap=8)
    variants=[dict(common,name='ray_'+mode,final_ray_mode=mode) for mode in ('off','mse','norm','projection')]
    declarations=validate_config(dict(variants=[dict(name='Full',method='full'),*variants]))
    require(rejected(lambda:validate_variant(dict(variants[1],joint_gain_refit=True))),'Pre-NNLS accepted')
    require(rejected(lambda:validate_variant(dict(variants[1],postres_direction=True))),'Direction recoding accepted in fixed-J ray')
    ks=torch.randn(1,4,n,128);ks[0,1]=z.repeat(n,1);vv=torch.randn_like(ks).half();factors=torch.ones(28,4);factors[0,1]=f
    class Base:
        settings={}
        def _initialize(self,fail=False):
            cfg=type(self).settings;self.book=book;self.encoder_factors={'key_head_factors':factors};self.joint_gain_enabled=False
            self.requested_residual_stages=[tuple(x) for x in cfg['selected_stages']];self.selected_residual_stages=self.requested_residual_stages[:43];self.ranking_constant_bytes=448;self.depth_distortions=[]
            kc,kg=encode_key(ks,book,gain_factor=factors[0][None,:,None,None]);vc,vs,vm=binary_centroids(vv.transpose(2,3).contiguous(),group_size=n)
            self.state=SimpleNamespace(direction_code=kc,direction_gain=kg,v_code=vc,v_scale=vs,v_min=vm,
                depth_residual_code=torch.empty(2,n,4,dtype=torch.int32),depth_residual_alpha=torch.empty(2,n,dtype=torch.float16))
            self.pointers=(kg.data_ptr(),self.state.depth_residual_alpha.data_ptr())
            if fail:raise ValueError('intentional constructor failure')
            self._encode_axis(0,self.state,'k',local,ks[0],decode_key(kc,kg,book)[0])
            self._encode_axis(0,self.state,'v',local,vv[0],decoder(vc,vs,vm,n,1).transpose(2,3)[0])
        def _encode_axis(self,layer,state,axis,selected,source,base):
            head=1;slot=selected.index((axis,head,0));rc,ra,_,errors=encode_sign_depth(source[head],base[head],1)
            state.depth_residual_code[slot].copy_(rc[0]);state.depth_residual_alpha[slot].copy_(ra[0])
            self.depth_distortions.append(dict(axis=axis,layer=layer,head=head,depth=1,squared_error_by_depth=errors))
        def result_metadata(self):return dict(key_gain_rule='norm')
    cls=final_ray_class(Base);instances=[]
    for variant in variants:
        cls.settings=variant;instance=cls();instance._initialize();instances.append(instance)
        require((instance.state.direction_gain.data_ptr(),instance.state.depth_residual_alpha.data_ptr())==instance.pointers,'Physical coefficient slots replaced')
        require(len(instance.final_ray_statistics)==(0 if variant['final_ray_mode']=='off' else 1),'Wrong selected ray coverage')
        if variant['final_ray_mode']!='off':require(instance.result_metadata()['key_gain_rule']=='final-state-common-ray-'+variant['final_ray_mode'],'Final coefficient metadata incorrect')
    base=instances[0].state
    for instance in instances:
        state=instance.state
        require(tensor_bytes(state)==tensor_bytes(base),'Stored field bytes changed')
        for name in ('direction_code','v_code','v_scale','v_min','depth_residual_code'):require(torch.equal(getattr(state,name),getattr(base,name)),'J/sign/V fields changed')
        require(torch.equal(state.depth_residual_alpha[0],base.depth_residual_alpha[0]),'Wrong V alpha slot changed')
    # Public registry restores even when construction fails.
    mods={name:ModuleType(name) for name in ('round8_runner','round8_bank','factory','run_representation_variants')};mods['round8_bank'].Round8Bank=Base
    before_factory=lambda method:('old',method);before_executor=object();mods['factory'].bank_class=before_factory;mods['run_representation_variants'].bank_class=before_executor
    try:
        with patch.dict(sys.modules,mods):
            with backend_registration('final_ray') as selected:
                require(mods['factory'].bank_class('round8_joint') is selected,'Wrong public backend')
                selected.settings=variants[1];selected()._initialize(fail=True)
    except ValueError:pass
    require(mods['factory'].bank_class is before_factory and mods['run_representation_variants'].bank_class is before_executor,'Registration not restored')
    nv,lh=1792,112;physical_base=lh*nv*34+lh*128*6+128+nv*8
    plan=plan_depth_stages(layers=28,heads=4,visual_tokens=nv,visual_full_bytes=lh*nv*512,base_bytes=physical_base,factor_bytes=1344,budget_bits=1.8,selected_stages=priority,stage_limit=43)
    require(plan['predicted_bytes']==8315013 and len(plan['ranking_constant'])==448,'Complete owner/rank differs')
    fixture=output/'postcheck_stub';fixture.mkdir();config=dict(variants=variants)
    (fixture/'manifest.json').write_text(json.dumps(dict(config=config,config_sha256='CPU-config',external_codec_inputs={})),encoding='utf-8')
    (fixture/'complete.json').write_text(json.dumps(dict(status='COMPLETE',records=4,expected=4)),encoding='utf-8')
    (fixture/'records.jsonl').write_text(''.join(json.dumps(dict(CPU_stub=i))+'\n' for i in range(4)),encoding='utf-8')
    final=finalize_run(fixture,{str(bookpath):sha(bookpath)},source_fingerprints(),'CPU-config',declarations)
    require(final['complete_round8_sha256']==sha(fixture/'complete_round8.json') and final['records_sha256']==sha(fixture/'records.jsonl'),'Final gate hash binding failed')
    return dict(status='PASS',schema='trajkv.round8.final-ray-cpu-check.v1',theory_and_own_objective_witnesses=theory,
        degenerate_policy_cases=fallback,zero_individual_coefficient_allowed=True,off_strict_tie_original_fields=True,
        actual_norm_not_squared_norm_objective=True,random_given_J_original_tuple_rows=31,initialization_factor_negative_control=True,
        all_four_modes_class_integration=True,J_sign_V_unchanged=True,same_stored_field_bytes=tensor_bytes(base),
        complete_owner_N1792=8315013,full_ranking=224,ranking_bytes=448,selected_stages=43,added_retained_tensor_bytes=0,
        registration_exception_restore=True,complete_final_ray_hash_binding=True,
        source_sha256=source_fingerprints(),checker_sha256=sha(__file__),GPU_used=False,SSH_used=False,
        scope='CPU synthetic complete-state radial goals and local stub only; no model/task/timing measurement')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=controls(args.output);(args.output/'check.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_used=False,physical_goal_modes=3,report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',physical_goal_modes=3,GPU_used=False)))


if __name__=='__main__':main()
