"""Optional posterior K controls, prepared only; no automatic model/GPU run.

K-Mu changes only the existing mean header, without a future query. K-Book_FZJ
holds the original nibble J under a learned MSE/functional book, then recomputes
its source norm gain and original K residual slots. It retains the old assignment
book explicitly (+128B for a distinct new book): diagnostic, not iso-byte frontier.
"""
from __future__ import annotations
import argparse
from copy import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import r7_legacy_bridge_v1 as bridge
import round9_calibration_v1 as decode

MODES=("K-Mu","K-Book_FZJ")
OLD_ALIAS="K-Func_FZJ"
BOOK_SHA="af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589"
FACTOR_SHA="8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea"
BRIDGE_SHA="b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b"
DECODE_SHA="757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f"
REFERENCE_SHA="e37810a2568ebb21f10ea10488829be0758311985c5dbdeebdd20fa450c5631a"
STATE=bridge._STATE_TENSORS


def require(v,m):
    if not v:raise ValueError("ROUND9_KEY_CONTROL: "+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def _pins():require(sha(bridge.__file__)==BRIDGE_SHA and sha(decode.__file__)==DECODE_SHA,'frozen helper source changed')
def _unique(owners):
    values={}
    for owner in owners:
        for t in owner.named_tensors().values():
            if t.numel():values.setdefault(bridge._storage(t),t.untyped_storage().nbytes())
    return sum(values.values())


def _book(path,digest,original):
    if path is None:
        require(digest is None,'SHA without book path');return original,{'origin':'original_book_identity_control','sha256':BOOK_SHA}
    require(digest is not None and sha(path)==digest,'learned book SHA required')
    data=load(path)
    require(data.get('schema')=='trajkv.direction-book.v1' and data.get('original_book_sha256')==BOOK_SHA and
            data.get('dimension')==4 and data.get('centroids')==16 and data.get('round9_fit',{}).get('objective') in ('mse','functional'),
            'original-source MSE/functional G4_16 book artifact required')
    book=torch.tensor(data['centers'],device=original.device,dtype=torch.float16)
    require(tuple(book.shape)==(16,4) and bool(torch.isfinite(book).all()),'finite same-capacity Half book')
    # A separately stored artifact with identical deployed values can share C0.
    # No hidden second allocation is then retained by the returned owners.
    if torch.equal(book.view(torch.uint8),original.view(torch.uint8)):book=original
    return book,{'origin':'same_learned_K_book','objective':data['round9_fit']['objective'],
                 'sha256':digest,'fit_module_sha256':data['round9_fit'].get('module_sha256')}


def mean_fields(state,sourceK,book,*,wht_fn):
    """Pure return. μ*=mean(Ksource)-H(mean(actual decoded centered K))."""
    require(sourceK.dtype==torch.float16 and tuple(sourceK.shape)==(*state.direction_code.shape[:-1],128),'native FP16 K geometry')
    result=copy(state)
    centered=decode.decode_key_rotated(vars(state),book)
    source_mean=sourceK.float().mean(-2,keepdim=True)
    inverse_decoded_mean=wht_fn(centered.float().mean(-2,keepdim=True))
    target=source_mean-inverse_decoded_mean
    result.direction_mean=target.half().contiguous()
    require(bool(torch.isfinite(result.direction_mean).all()),'mean header overflow/nonfinite')
    before=source_mean-state.direction_mean.float()-inverse_decoded_mean
    after=source_mean-result.direction_mean.float()-inverse_decoded_mean
    for name in STATE:
        if name!='direction_mean':require(getattr(result,name,None) is getattr(state,name,None),'K-Mu changed a fixed field: '+name)
    return result,{'mean_error_before_RMS':float(before.square().mean().sqrt()),
        'mean_error_after_RMS':float(after.square().mean().sqrt()),'mean_error_after_maxabs':float(after.abs().max()),
        'changed_mean_values':int((result.direction_mean!=state.direction_mean).sum()),
        'fixed_query_real_arithmetic':'all visual logits for a query/head shift by q dot delta_mu/sqrt128; within-visual routing unchanged',
        'finite_precision_boundary':'FP16 header and FP32 reader/softmax rounding remain; this is not a task improvement theorem',
        'future_question_or_query_used_by_encoder':False}


def fixed_J_fields(state,source_rotated,book,factors,*,key_amplitude,encode_sign_depth):
    """Pure-return FZJ state. J frozen; norm gain/RE follow the new book."""
    require(source_rotated.dtype==torch.float32 and source_rotated.shape[-1]==128,'actual FP32 centered-WHT source')
    result=copy(state);ids=torch.stack((state.direction_code&15,state.direction_code>>4),-1).flatten(-2).long()
    template=book.float()[ids].flatten(-2)
    require(tuple(template.shape)==tuple(source_rotated.shape),'original J/new book shape')
    require(tuple(factors.shape)==(4,) and bool(torch.isfinite(factors).all()) and bool((factors>0).all()),'original K head factors')
    result.direction_gain=(key_amplitude(source_rotated,template,'norm').half().float()*factors[None,:,None,None]).half().contiguous()
    require(bool(torch.isfinite(result.direction_gain).all()),'gain overflow/nonfinite')
    used=bool((state.depth_residual_lookup[0]>=0).any())
    changed_code=changed_alpha=0
    if used:
        result.depth_residual_code=state.depth_residual_code.detach().clone()
        result.depth_residual_alpha=state.depth_residual_alpha.detach().clone()
        base=(template*result.direction_gain.float()).half()
        for head in range(4):
            slots=[int(s) for s in state.depth_residual_lookup[0,head]]
            require(slots[0]>=0 or slots[1]<0,'K stage parent missing')
            slots=[s for s in slots if s>=0]
            if not slots:continue
            codes,alphas,_,_=encode_sign_depth(source_rotated[0,head],base[0,head],len(slots))
            for stage,slot in enumerate(slots):
                result.depth_residual_code[slot].copy_(codes[stage]);result.depth_residual_alpha[slot].copy_(alphas[stage])
                changed_code+=int((codes[stage]!=state.depth_residual_code[slot]).sum())
                changed_alpha+=int((alphas[stage]!=state.depth_residual_alpha[slot]).sum())
    for name in ('direction_code','direction_mean','v_code','v_scale','v_min','depth_residual_lookup','visual_pos'):
        require(getattr(result,name,None) is getattr(state,name,None),'FZJ changed fixed J/mean/V/positions: '+name)
    for head in range(4):
        for stage in range(2):
            slot=int(state.depth_residual_lookup[1,head,stage])
            if slot>=0:require(torch.equal(result.depth_residual_code[slot],state.depth_residual_code[slot]) and
                               torch.equal(result.depth_residual_alpha[slot],state.depth_residual_alpha[slot]),'V residual changed')
    return result,{'J_changed':False,'changed_gain_values':int((result.direction_gain!=state.direction_gain).sum()),
                   'changed_K_residual_code_elements':changed_code,'changed_K_residual_alpha_elements':changed_alpha,
                   'contrast':'FZJ vs RE holds book/f0/mean/V/coverage fixed; gain and K residual differences are derived consequences of the J intervention'}


@torch.no_grad()
def build_round9_key_control_owners(native,vp,variant,mode,book_path=None,book_sha256=None,**legacy_kwargs):
    _pins();own_source=sha(__file__)
    requested_mode=mode
    if mode==OLD_ALIAS:mode='K-Book_FZJ'
    require(mode in MODES,'unknown posterior key control')
    require(mode!='K-Mu' or book_path is book_sha256 is None,'K-Mu keeps original book')
    sourceK=[layer.keys.index_select(2,vp).detach().contiguous() for layer in native.layers]
    owners,metadata=bridge.build_r7_owners(native,vp,variant,**legacy_kwargs)
    require(metadata['variant']['name']=='r7_joint_k1_gt_cond' and metadata['selected_stage_count']==43 and
            metadata['fixed_assets']['direction_book_path']['sha256']==BOOK_SHA and metadata['fixed_assets']['metadata_factor_path']['sha256']==FACTOR_SHA and
            metadata['fixed_assets']['round7_policy_path']['sha256']==bridge._DEFAULT_ASSETS['round7_policy_path'],'original low gtcond43/f0/book/policy required')
    before=_unique(owners);shared=owners[0].shared;assignment_book=shared['book']
    book,book_meta=_book(book_path,book_sha256,assignment_book)
    stats=[]
    with bridge._legacy_paths(Path(legacy_kwargs['legacy_a2']),Path(legacy_kwargs['legacy_experiments']),Path(legacy_kwargs['legacy_scripts'])):
        from fast_wht import wht
        from round7_key_codec import key_amplitude,encode_sign_depth
        for layer,(owner,source) in enumerate(zip(owners,sourceK)):
            old=owner.state
            if mode=='K-Mu':new,detail=mean_fields(old,source,book,wht_fn=wht)
            else:
                rotated=wht(source.float()-old.direction_mean.float())
                new,detail=fixed_J_fields(old,rotated,book,shared['encoder_factor.key_head_factors'][layer],
                                         key_amplitude=key_amplitude,encode_sign_depth=encode_sign_depth)
                if book_path is None:
                    require(torch.equal(new.direction_gain,old.direction_gain),'original book FZJ gain identity differs')
                    if hasattr(old,'depth_residual_code'):require(torch.equal(new.depth_residual_code,old.depth_residual_code) and torch.equal(new.depth_residual_alpha,old.depth_residual_alpha),'original book FZJ RE identity differs')
            owner.state=new;stats.append({'layer':layer,**detail})
    extra=0
    if mode=='K-Book_FZJ':
        shared['book']=book
        # Needed to reproduce the original J from a fresh image. This is a real
        # named owner tensor even though the online reader only uses C*.
        shared['encoder_assignment_book']=assignment_book
        if bridge._storage(book)!=bridge._storage(assignment_book):extra=assignment_book.untyped_storage().nbytes()
        require(extra in (0,128),'unexpected original assignment-book capacity')
    require(_unique(owners)==before+extra,'posterior owner capacity must include retained assignment book')
    forbidden={bridge._storage(t) for layer in native.layers for t in (layer.keys,layer.values)}
    for owner in owners:
        for t in owner.named_tensors().values():require(not t.requires_grad and t.grad_fn is None and bridge._storage(t) not in forbidden,'retained autograd/native source alias')
    require(all(sha(path)==item['sha256'] for path,item in metadata['verified_legacy_source_pins'].items()),'legacy source changed')
    require(all(sha(item['path'])==item['sha256'] for item in metadata['fixed_assets'].values()),'original asset changed')
    if book_path is not None:require(sha(book_path)==book_sha256,'learned book changed')
    require(sha(__file__)==own_source,'control source changed')
    metadata['round9_key_posterior_control']={'schema':'trajkv.round9.key_control.v2','mode':mode,'requested_mode':requested_mode,'book':book_meta,
        'original_J_and_V_and43_coverage_fixed':True,'before_unique_bytes':before,'after_unique_bytes':_unique(owners),
        'extra_retained_fields':int(mode=='K-Book_FZJ'),'encoder_assignment_book_bytes_incremental':extra,
        'same_complete_budget_as_original':extra==0,'diagnostic_only':True,'frontier_eligible':False,
        'assignment_book_is_named_and_charged':mode=='K-Book_FZJ',
        'layer_stats':stats,'source_sha256':own_source,'fit_performed':False,
        'actual_GPU_identity_or_task_benefit':'NOT_ESTABLISHED_BY_BUILD_METADATA'}
    metadata['returned_owners_unique_tensor_capacity_bytes']=_unique(owners)
    return owners,metadata


def _cpu_wht(x):
    y=x.float();shape=y.shape
    for n in (1,2,4,8,16,32,64):
        z=y.reshape(*shape[:-1],128//(2*n),2,n);a,b=z[...,0,:],z[...,1,:]
        y=torch.cat((a+b,a-b),-1).reshape(shape)
    return y*(128**-.5)


def cpu_check(output,legacy_a2):
    _pins();a2=Path(legacy_a2).resolve()
    require(sha(a2/'round7_key_codec.py')==bridge._DEFAULT_SOURCES['a2/round7_key_codec.py'],'original codec SHA')
    with bridge._legacy_paths(a2,a2.parent/'experiments',a2):
        from round7_key_codec import encode_key,key_amplitude,encode_sign_depth,decode_key
        torch.manual_seed(0);source=torch.randn(1,4,32,128).half()*.15;book=torch.randn(16,4).half()
        mean=source.float().mean(-2,keepdim=True).half();rotated=_cpu_wht(source.float()-mean.float());factors=torch.tensor([.8,1.2,1.1,.9])
        code,gain=encode_key(rotated,book,gain_factor=factors[None,:,None,None])
        lookup=torch.full((2,4,2),-1,dtype=torch.int32);lookup[0,1,0]=0;lookup[0,1,1]=1;lookup[1,0,0]=2
        state=SimpleNamespace(direction_code=code,direction_gain=gain,direction_mean=mean,
            v_code=torch.zeros(1,4,128,1,dtype=torch.int32),v_scale=torch.ones(1,4,128,1,dtype=torch.float16),v_min=torch.zeros(1,4,128,1,dtype=torch.float16),
            visual_pos=torch.arange(32),depth_residual_lookup=lookup,depth_residual_code=torch.zeros(3,32,4,dtype=torch.int32),depth_residual_alpha=torch.zeros(3,32,dtype=torch.float16))
        rc,ra,_,_=encode_sign_depth(rotated[0,1],decode_key(code,gain,book)[0,1],2)
        for i in range(2):state.depth_residual_code[i].copy_(rc[i]);state.depth_residual_alpha[i].copy_(ra[i])
        snapshots={n:v.clone() for n,v in vars(state).items()}
        fixed,_=fixed_J_fields(state,rotated,book,factors,key_amplitude=key_amplitude,encode_sign_depth=encode_sign_depth)
        require(all(torch.equal(getattr(fixed,n),v) for n,v in snapshots.items()),'original book FZJ source-codec identity')
        learned=(book.float()+torch.randn_like(book.float())*.3);learned=(learned*book.float().norm()/learned.norm()).half()
        fzj,detail=fixed_J_fields(state,rotated,learned,factors,key_amplitude=key_amplitude,encode_sign_depth=encode_sign_depth)
        re_code,_=encode_key(rotated,learned,gain_factor=factors[None,:,None,None])
        require(fzj.direction_code is state.direction_code and int((re_code!=code).sum())>0,'nonidentity book fixture must distinguish fixed J from RE')
        mu,mean_stats=mean_fields(state,source,book,wht_fn=_cpu_wht)
        centered=decode.decode_key_rotated(vars(state),book)
        target=source.float().mean(-2,keepdim=True)-_cpu_wht(centered.float().mean(-2,keepdim=True))
        require(torch.equal(mu.direction_mean,target.half()),'exact declared Half header update')
        require(bool(((mu.direction_mean.float()-target).abs()<=(state.direction_mean.float()-target).abs()+1e-7).all()),'nearest header arithmetic')
        # FP64 mathematical control only: common shift and modality mass law.
        H=_cpu_wht(torch.eye(128)).double();oldK=centered.double()@H+state.direction_mean.double();newK=centered.double()@H+mu.direction_mean.double()
        q=torch.randn(1,4,3,128).double();scale=128**-.5
        oldlogits=q@oldK.transpose(-1,-2)*scale;newlogits=q@newK.transpose(-1,-2)*scale
        bias=(q*(mu.direction_mean.double()-state.direction_mean.double())).sum(-1,keepdim=True)*scale
        # Float32 H is an exactly fixed linear matrix in this control.
        require(float((newlogits-oldlogits-bias).abs().max())<1e-10,'common visual-logit shift identity')
        require(float((oldlogits.softmax(-1)-newlogits.softmax(-1)).abs().max())<1e-12,'real-arithmetic within-visual routing identity')
        text=torch.randn(1,4,3,5).double();p=torch.cat((oldlogits,text),-1).softmax(-1)[...,:32].sum(-1)
        pnew=torch.cat((newlogits,text),-1).softmax(-1)[...,:32].sum(-1)
        require(float((pnew-torch.sigmoid(torch.logit(p)+bias.squeeze(-1))).abs().max())<1e-12,'modality mass sigmoid law')
        require(all(torch.equal(getattr(state,n),v) for n,v in snapshots.items()),'original state mutated')
    # CPU-only physical charging control: alias C0/C0 costs0; C*/C0 costs128.
    fake=SimpleNamespace(shared={'book':book})
    fake.named_tensors=lambda:fake.shared
    base_bytes=_unique([fake]);fake.shared['encoder_assignment_book']=book
    require(_unique([fake])==base_bytes,'same-book shared storage must add0 bytes')
    fake.shared['book']=learned
    require(_unique([fake])==base_bytes+128,'distinct-book C0 must add128 bytes')
    result={'status':'PASS_CPU_POSTERIOR_KEY_V2_ARITHMETIC_AND_BOOK_CHARGES_ONLY','source_sha256':sha(__file__),
      'checks':['original_book_FZJ_gain_RE_identity','nonidentity_book_FZJ_keeps_J_while_RE_differs','V_fields_and_Vresidual_fixed','KMu_actual_Half_reconstruction_mean_header_formula',
                'KMu_only_existing_mean_slot_changes','fixed_q_common_logit_shift_within_visual_routing_and_mass_law_FP64','no_old_state_mutation',
                'samebook_assignment_alias_adds0_distinct_book_adds128_actual_storage_bytes'],
      'mean_stats':mean_stats,'FZJ_fixture_stats':detail,'actual_GPU_calls':0,'whole_bank_reader_owner_GPU_positive':False}
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write(out/'CPU_RECEIPT.json',result);print(json.dumps(result));return result


@torch.no_grad()
def gpu_check(config_path,output,book_path=None,book_sha256=None):
    """Explicit Root-only synthetic entry. No models/images/GT/fit."""
    import r7_bridge_gpu_check_v1 as ref
    require(sha(ref.__file__)==REFERENCE_SHA,'frozen reference helper changed');cfg=load(config_path);ref.validate_metadata(cfg);variant=ref.profile(cfg)
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6),'actual SM86 CUDA required')
    from transformers.cache_utils import DynamicCache
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    pins={str(Path(__file__).resolve()):sha(__file__),str(config_path):sha(config_path)}
    if book_path is not None:pins[str(book_path)]=book_sha256;require(sha(book_path)==book_sha256,'learned book SHA')
    write(out/'manifest.json',{'schema':'trajkv.round9.key_control_synthetic.v2','source_hashes':pins,'Nv':[256,1792],
        'model_images_GT_loaded':False,'quality_QA_added':0,'diagnostic_source_storage_not_deployment_savings':True})
    kwargs={n:cfg[n] for n in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source')};kwargs['expected_asset_sha256']=ref.ASSETS
    records=[]
    for nv in (256,1792):
        torch.manual_seed(0);native=DynamicCache();boundary=nv+4
        for layer in range(28):
            k=torch.randn(1,4,boundary,128,device='cuda',dtype=torch.float16)*.15;v=torch.randn_like(k)*.1;native.update(k,v,layer)
        k=v=None;vp=torch.arange(4,boundary,device='cuda',dtype=torch.long)
        before=[(ref.tensor_identity(s.keys),ref.tensor_identity(s.values)) for s in native.layers]
        original,meta=bridge.build_r7_owners(native,vp,variant,**kwargs);baseline_bytes=_unique(original)
        rows=[]
        for mode in MODES:
            owners,info=build_round9_key_control_owners(native,vp,variant,mode,
                book_path=book_path if mode=='K-Book_FZJ' else None,book_sha256=book_sha256 if mode=='K-Book_FZJ' else None,**kwargs)
            control=info['round9_key_posterior_control'];extra=control['encoder_assignment_book_bytes_incremental']
            expected_extra=0
            if mode=='K-Book_FZJ' and book_path is not None:
                expected_book,_=_book(book_path,book_sha256,original[0].shared['book'])
                expected_extra=0 if bridge._storage(expected_book)==bridge._storage(original[0].shared['book']) else 128
            require(extra==expected_extra and _unique(owners)==baseline_bytes+expected_extra and info['selected_stages']==meta['selected_stages'],'owner/coverage/assignment book charge differs')
            if mode=='K-Book_FZJ':
                require('encoder_constants.encoder_assignment_book' in owners[0].named_tensors(),'old assignment book is not exposed')
                require(torch.equal(owners[0].shared['encoder_assignment_book'],original[0].shared['book']),'wrong retained C0')
            compare=[];qp=torch.arange(boundary,boundary+4,device='cuda',dtype=torch.long);tp=torch.cat((torch.arange(4,device='cuda'),qp))
            for layer,(a,b) in enumerate(zip(original,owners)):
                for name in STATE:
                    if name=='direction_mean' and mode=='K-Mu' or mode=='K-Book_FZJ' and name in ('direction_gain','depth_residual_code','depth_residual_alpha'):continue
                    require(ref.compare_tensor(getattr(a.state,name,None),getattr(b.state,name,None))['bitwise_equal'],'fixed field differs: '+name)
                for name in ('v_code','v_scale','v_min'):require(torch.equal(getattr(a.state,name),getattr(b.state,name)),'V differs')
                if hasattr(a.state,'depth_residual_code'):
                    for head in range(4):
                        for stage in range(2):
                            slot=int(a.state.depth_residual_lookup[1,head,stage])
                            if slot>=0:require(torch.equal(a.state.depth_residual_code[slot],b.state.depth_residual_code[slot]) and torch.equal(a.state.depth_residual_alpha[slot],b.state.depth_residual_alpha[slot]),'V residual differs')
                q=torch.randn(1,28,4,128,device='cuda',dtype=torch.float16)*.2
                tk=torch.cat((native.layers[layer].keys[:,:,:4],torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.15),2)
                tv=torch.cat((native.layers[layer].values[:,:,:4],torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.1),2)
                args=dict(exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,scale=128**-.5)
                left,right=a.attend(q,**args),b.attend(q,**args)
                require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()),'nonfinite reader output')
                if mode=='K-Book_FZJ' and book_path is None:require(torch.equal(left,right),'original book FZJ reader identity differs')
                compare.append({'layer':layer,**ref.compare_tensor(left,right)})
            rows.append({'mode':mode,'status':'PASS_MECHANICAL_SYNTHETIC_CONTROL','actual_owner_bytes':_unique(owners),
                         'assignment_book_incremental_bytes':extra,'diagnostic_not_frontier':True,'metadata':info,
                         'reader_differences_observed_not_hidden_by_tolerance':compare,'new_mean_is_not_a_quality_gain_claim':True})
        require(before==[(ref.tensor_identity(s.keys),ref.tensor_identity(s.values)) for s in native.layers],'native source mutated')
        records.append({'Nv':nv,'status':'PASS_MECHANICAL_SYNTHETIC_CONTROL','baseline_owner_bytes':baseline_bytes,'controls':rows})
    require(all(sha(p)==h for p,h in pins.items()),'source/asset changed during check')
    with (out/'records.jsonl').open('x',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    result={'status':'PASS_ALL2_SYNTHETIC_KEY_POSTERIOR_CONTROLS_V2','cases':2,'records_sha256':sha(out/'records.jsonl'),
            'actual_GPU':True,'quality_QA_added':0,'KMu_reader_change_expected_not_equivalence_failure':True}
    write(out/'complete.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--cpu-check',action='store_true');mode.add_argument('--gpu-check',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-a2',type=Path,default=Path('${FROZEN_TRAJKV_ROOT}/code/a2'))
    p.add_argument('--config',type=Path);p.add_argument('--book-path',type=Path);p.add_argument('--book-sha256');a=p.parse_args()
    if a.cpu_check:cpu_check(a.output,a.legacy_a2)
    else:
        require(a.config is not None,'explicit frozen low smoke4 config supplies assets only')
        gpu_check(a.config,a.output,a.book_path,a.book_sha256)
