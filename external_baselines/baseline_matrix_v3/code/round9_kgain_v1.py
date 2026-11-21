"""Optional K112-only same-student functional calibration and deployment.

No import-time fit, model, GPU, transport or queue action. The original book/J,
mean, V0 fields and original43 coverage stay fixed. Only the existing K factor
slots and K gain/residual values change. --cpu-check is arithmetic-only.
"""
from __future__ import annotations
import argparse
from copy import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import torch
import round9_calibration_v1 as local
import round9_value_bank_v1 as value_ops
import r7_legacy_bridge_v1 as bridge

SCHEMA="trajkv.round9.kgain112.v1"
BOOK_SHA="af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589"
FACTOR_SHA="8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea"
CALIBRATION_SHA="757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f"
VALUE_OPS_SHA="14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e"
BRIDGE_SHA="b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b"


def require(v,m):
    if not v:raise ValueError("ROUND9_KGAIN: "+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _pins():
    require(sha(local.__file__)==CALIBRATION_SHA and sha(value_ops.__file__)==VALUE_OPS_SHA and sha(bridge.__file__)==BRIDGE_SHA,
            "frozen helper source changed")
def _factor(g,layer):
    require(isinstance(g,torch.Tensor) and g.dtype==torch.float32,"FP32 factor tensor required")
    if tuple(g.shape)==(28,4):g=g[layer]
    require(tuple(g.shape)==(4,) and bool(torch.isfinite(g).all()) and bool((g>0).all()),"positive finite local4/full28x4 K factors")
    return g[None,:,None,None]


def value_parameter_fingerprint(values):
    """Explicit little-endian FP32[2,28,4] value-table identity, not file SHA."""
    flat=[]
    for name in ("value_head_factors","value_center_offsets"):
        x=values[name]
        x=x.detach().cpu().float() if isinstance(x,torch.Tensor) else torch.tensor(x,dtype=torch.float32)
        require(tuple(x.shape)==(28,4) and bool(torch.isfinite(x).all()),"fixed V224 geometry")
        flat.extend(x.reshape(-1).tolist())
    return hashlib.sha256(b"FP32[2,28,4]\0"+struct.pack("<224f",*flat)).hexdigest()


def load_kgain(path,expected_sha256,*,value_fingerprint,theta_sha256=None,device="cpu"):
    require(path is not None and expected_sha256 is not None and sha(path)==expected_sha256,"K112 file SHA required")
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    require(data.get("schema")==SCHEMA and data.get("parameter_count")==112 and data.get("base_profile")=="r7_joint_k1_gt_cond","K112 schema/profile")
    require(data.get("base_metadata_factor_sha256")==FACTOR_SHA and data.get("base_book_sha256")==BOOK_SHA,"wrong inherited factor/book source")
    require(not ({"value_head_factors","value_center_offsets","centers"}&set(data)),"K112 artifact cannot carry V/book replacements")
    require(data.get("base_value_parameter_fp32_sha256")==value_fingerprint and data.get("base_value_theta_sha256")==theta_sha256,"fixed V0 identity differs")
    g=torch.tensor(data["key_head_factors"],device=device,dtype=torch.float32)
    require(tuple(g.shape)==(28,4),"full K112 required");_factor(g,0)
    require(bool(torch.isfinite(g).all()) and bool((g>0).all()),"finite positive full K112")
    return g,data


def _template(code,book):
    require(tuple(book.shape)==(16,4) and book.dtype==torch.float16,"original Half G4_16 only")
    require(code.dtype==torch.uint8 and code.shape[-1]==16,"fixed original nibble J")
    ids=torch.stack((code&15,code>>4),-1).flatten(-2).long()
    return book.float()[ids].flatten(-2)


def apply_kgain_fields(state,source_rotated,gains,book,selected_stages,layer_idx,*,amplitude_fn=None,encode_sign_depth_fn=None):
    """Pure-return (same-class state copy, stats); no original field mutation.

    Recompute the pre-factor norm amplitude. Scaling the already-rounded stored
    gain by g/g0 would lose the original two-Half-rounding contract. J is fixed.
    Hard RE stops sign/alpha gradients; gain keeps the declared cast gradient.
    """
    require(source_rotated.dtype==torch.float32 and tuple(source_rotated.shape)==(*state.direction_code.shape[:-1],128),"actual centered-WHT FP32 K geometry")
    if amplitude_fn is None:amplitude_fn=importlib.import_module("round7_key_codec").key_amplitude
    if encode_sign_depth_fn is None:encode_sign_depth_fn=importlib.import_module("round6_depth_codec").encode_sign_depth
    result=copy(state);template=_template(state.direction_code,book)
    unit=amplitude_fn(source_rotated,template,"norm").half()
    result.direction_gain=(unit.float()*_factor(gains,layer_idx)).half().contiguous()
    require(bool(torch.isfinite(result.direction_gain).all()),"nonfinite stored gain")
    selected=[(axis,head,stage) for axis,layer,head,stage in selected_stages if layer==layer_idx]
    for slot,(axis,head,stage) in enumerate(selected):
        require(int(state.depth_residual_lookup[0 if axis=="k" else 1,head,stage])==slot,"original stage/lookup mapping differs")
    slots=[(slot,head,stage) for slot,(axis,head,stage) in enumerate(selected) if axis=="k"]
    changed_code=changed_alpha=0
    if slots:
        with torch.no_grad():
            result.depth_residual_code=state.depth_residual_code.detach().clone()
            result.depth_residual_alpha=state.depth_residual_alpha.detach().clone()
            base=(template*result.direction_gain.detach().float()).half()
            for head in sorted({h for _,h,_ in slots}):
                entries=[(slot,stage) for slot,h,stage in slots if h==head]
                require(sorted(stage for _,stage in entries)==list(range(len(entries))),"parent-before-child K depth")
                codes,alphas,_,_=encode_sign_depth_fn(source_rotated[0,head],base[0,head],len(entries))
                for slot,stage in entries:
                    result.depth_residual_code[slot].copy_(codes[stage]);result.depth_residual_alpha[slot].copy_(alphas[stage])
                    changed_code+=int((codes[stage]!=state.depth_residual_code[slot]).sum())
                    changed_alpha+=int((alphas[stage]!=state.depth_residual_alpha[slot]).sum())
    for name in ("direction_code","direction_mean","v_code","v_scale","v_min","depth_residual_lookup"):
        require(getattr(result,name) is getattr(state,name),"fixed J/mean/V field replaced: "+name)
    for slot,(axis,head,stage) in enumerate(selected):
        if axis=="v":
            require(torch.equal(result.depth_residual_code[slot],state.depth_residual_code[slot]) and
                    torch.equal(result.depth_residual_alpha[slot],state.depth_residual_alpha[slot]),"fixed V residual changed")
    return result,{"K_slots":len(slots),"changed_gain_values":int((result.direction_gain.detach()!=state.direction_gain).sum()),
                   "changed_K_code_elements":changed_code,"changed_K_alpha_elements":changed_alpha,
                   "J_changed":False,"V_changed":False,"hard_RE_gradient":"stopped"}


@torch.no_grad()
def build_round9_kgain_owners(native_prefix_cache,visual_positions,variant,*,kgain_path=None,kgain_sha256=None,
                            theta_path=None,theta_sha256=None,**legacy_kwargs):
    """Return owners28+metadata. None/None means original g0 and original V0.

    Optional theta fixes a separately frozen V-RE baseline before the K-only
    intervention. Neither None mode creates a new parameter artifact or fit.
    """
    _pins();before_source=sha(__file__)
    require((kgain_path is None)==(kgain_sha256 is None),"K path/SHA pair required")
    require((theta_path is None)==(theta_sha256 is None),"V path/SHA pair required")
    sources=[layer.keys.index_select(2,visual_positions).detach().contiguous() for layer in native_prefix_cache.layers]
    if theta_path is None:owners,metadata=bridge.build_r7_owners(native_prefix_cache,visual_positions,variant,**legacy_kwargs)
    else:owners,metadata=value_ops.build_round9_value_owners(native_prefix_cache,visual_positions,variant,
                          mode="V-RE",theta_path=theta_path,theta_sha256=theta_sha256,**legacy_kwargs)
    require(metadata['variant']['name']=='r7_joint_k1_gt_cond' and metadata['selected_stage_count']==43 and
            metadata['fixed_assets']['round7_policy_path']['sha256']==bridge._DEFAULT_ASSETS['round7_policy_path'] and
            metadata['fixed_assets']['direction_book_path']['sha256']==BOOK_SHA and
            metadata['fixed_assets']['metadata_factor_path']['sha256']==FACTOR_SHA,"original low43/book/f0 required")
    before=value_ops._unique(owners);shared=owners[0].shared
    vparams={name:shared['encoder_factor.'+name] for name in ('value_head_factors','value_center_offsets')}
    vfp=value_parameter_fingerprint(vparams)
    original_g=shared['encoder_factor.key_head_factors']
    if kgain_path is None:gains=original_g;parameter_metadata={'origin':'original_g0_identity','parameter_count':112}
    else:
        gains,payload=load_kgain(kgain_path,kgain_sha256,value_fingerprint=vfp,theta_sha256=theta_sha256,device=visual_positions.device)
        parameter_metadata={'origin':'frozen_K112_file','parameter_count':112,'sha256':kgain_sha256}
    with bridge._legacy_paths(Path(legacy_kwargs['legacy_a2']),Path(legacy_kwargs['legacy_experiments']),Path(legacy_kwargs['legacy_scripts'])):
        from fast_wht import wht
        from round7_key_codec import key_amplitude,encode_sign_depth
        stats=[]
        for layer,(owner,source) in enumerate(zip(owners,sources)):
            old=owner.state;rotated=wht(source.float()-old.direction_mean.float())
            new,detail=apply_kgain_fields(old,rotated,gains,shared['book'],metadata['selected_stages'],layer,
                                        amplitude_fn=key_amplitude,encode_sign_depth_fn=encode_sign_depth)
            if kgain_path is None:
                require(torch.equal(new.direction_gain,old.direction_gain),"g0 identity gain differs")
                if hasattr(old,'depth_residual_code'):
                    require(torch.equal(new.depth_residual_code,old.depth_residual_code) and torch.equal(new.depth_residual_alpha,old.depth_residual_alpha),"g0 identity residual differs")
            owner.state=new;stats.append({'layer':layer,**detail})
    shared['encoder_factor.key_head_factors']=gains
    require(value_ops._unique(owners)==before,"K112 intervention changed persistent owner capacity")
    require(value_parameter_fingerprint({n:shared['encoder_factor.'+n] for n in vparams})==vfp,"V parameter slots changed")
    forbidden={bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys,layer.values)}
    for owner in owners:
        for t in owner.named_tensors().values():
            require(not t.requires_grad and t.grad_fn is None and bridge._storage(t) not in forbidden,"retained graph/native source")
    require(all(sha(path)==item['sha256'] for path,item in metadata['verified_legacy_source_pins'].items()),"legacy source changed")
    require(all(sha(item['path'])==item['sha256'] for item in metadata['fixed_assets'].values()),"base asset changed")
    if kgain_path is not None:require(sha(kgain_path)==kgain_sha256,"K file changed")
    if theta_path is not None:require(sha(theta_path)==theta_sha256,"fixed V file changed")
    require(sha(__file__)==before_source,"K112 source changed")
    metadata['round9_kgain_candidate']={'schema':SCHEMA,'parameters':parameter_metadata,
        'base_value_theta_sha256':theta_sha256,'base_value_parameter_fp32_sha256':vfp,
        'fixed_book_J_mean_V_and43_coverage':True,'K_residuals':'actual hardRE at original K slots from each new actual Half base',
        'before_owner_unique_bytes':before,'after_owner_unique_bytes':value_ops._unique(owners),'layer_stats':stats,
        'same_capacity_existing_factor_slots':True,'source_sha256':before_source,
        'original_g0_whole_reader_GPU_identity':'PENDING_ROOT_REAL_CONTROL','quality_or_speed_gain_claimed':False}
    metadata['returned_owners_unique_tensor_capacity_bytes']=value_ops._unique(owners)
    return owners,metadata


def key_gain_fields(s,gains,cfg):
    result,_=apply_kgain_fields(SimpleNamespace(**s['r7_fields']),
        s['visual_k_rotated'],gains,s['book'],s['selected_stages'],s['layer_idx'],
        amplitude_fn=cfg.get('key_amplitude'),encode_sign_depth_fn=cfg.get('encode_sign_depth'))
    return vars(result)


def _functional_loss(s,gains,cfg,grad=False):
    fields=key_gain_fields(s,gains,cfg)
    with torch.no_grad():actual=cfg['packed_reader'](s,fields,s['book'])
    require(tuple(actual.shape)==tuple(s['q'].shape) and bool(torch.isfinite(actual).all()),"actual packed output differs")
    proxy=local.dense_heads(s,fields,s['book'])
    heads=local.hard_forward_ste(proxy,actual) if grad else actual
    pred=local.project(heads,local._wo(cfg,s['layer_idx'],s['q'].device));target=local.teacher_projected(s,cfg)
    return (pred.float()-target.float()).square().mean(),float((proxy.detach().float()-actual.float()).abs().max())


def fit(samples,cfg):
    """Optional explicit fit100. Never called automatically by factory/import."""
    _pins();module_sha=sha(__file__);samples,steps,outer=local._prepare(samples,cfg)
    torch.manual_seed(int(cfg.get('seed',9)))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from round7_key_codec import key_amplitude
    from round6_depth_codec import encode_sign_depth
    require(cfg.get('key_amplitude') in (None,key_amplitude) and cfg.get('encode_sign_depth') is encode_sign_depth,
            'unchanged original norm and hardRE functions required')
    numerical_sources={inspect.getsourcefile(key_amplitude):bridge._DEFAULT_SOURCES['a2/round7_key_codec.py'],
                       inspect.getsourcefile(encode_sign_depth):bridge._DEFAULT_SOURCES['a2/round6_depth_codec.py']}
    require(all(path and sha(path)==digest for path,digest in numerical_sources.items()),'numerical callback source differs')
    cfg=dict(cfg,key_amplitude=key_amplitude,encode_sign_depth=encode_sign_depth)
    require(outer==1,"one frozen-student inner problem only")
    require(steps<=100 and (cfg.get('fixture_only',False) or steps==100),"optional real prototype is exactly100 updates")
    device=cfg.get('device','cuda:0');theta_sha=cfg.get('theta_sha256')
    require((cfg.get('theta_path') is None)==(theta_sha is None),"fixed V path/SHA pair")
    base=samples[0]['original_factors']
    if theta_sha is None:vfp=value_parameter_fingerprint(base)
    else:
        value_theta,_=value_ops.load_value_theta(cfg['theta_path'],theta_sha,device='cpu');vfp=value_parameter_fingerprint(value_theta)
    for s in samples:
        require(s['student_snapshot'].get('direction_book_path')==BOOK_SHA and s['student_snapshot'].get('metadata_factor_path')==FACTOR_SHA,"mixed book/f0 sample identity")
        require(s['student_snapshot'].get('round7_policy_path')==bridge._DEFAULT_ASSETS['round7_policy_path'],'wrong original43 policy source')
        require(s['student_snapshot'].get('student_theta_sha256')==theta_sha,"q/text must come from this fixed V0 student snapshot")
        if theta_sha is not None:require(s['student_snapshot'].get('student_value_mode')=='V-RE',"V0 collection mode differs")
        require(s.get('visual_k_rotated') is not None and s['visual_k_rotated'].dtype==torch.float32,"actual old FP32 WHT source required")
        require(s['original_factors']['key_head_factors']==base['key_head_factors'],'mixed original K factors')
        require(torch.equal(s['book'],samples[0]['book']),'mixed frozen original book values')
        require(s['selected_stages']==samples[0]['selected_stages'],'mixed original43 stage coverage')
        require(len(s['selected_stages'])==43 and sum(x[0]=='k' for x in s['selected_stages'])==23,'original43 coverage required')
        if theta_sha is None:require(value_parameter_fingerprint(s['original_factors'])==vfp,"mixed fixed original V0 values")
    if not cfg.get('fixture_only',False):require(len({s['source_id'] for s in samples})==32,"exact32 source IDs")
    g0=torch.tensor(base['key_head_factors'],dtype=torch.float32);require(tuple(g0.shape)==(28,4),'original K112 shape');_factor(g0,0)
    # Zero relative log parameters make the initial effective g exactly g0;
    # exp(log(g0)) need not recover the original FP32 values bit for bit.
    base_g=g0.to(device).detach()
    logg=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(base_g[l])) for l in range(28)])
    optimizer=torch.optim.Adam(logg.parameters(),lr=float(cfg.get('learning_rate',.003)))
    provenance=local._provenance(samples,cfg)
    def at(layer):return base_g[layer]*logg[layer].exp()
    def tables():return torch.stack([at(l).detach().cpu() for l in range(28)])
    def evaluate():
        weights=local.source_balanced_weights(samples);total=0.
        with torch.no_grad():
            for raw in samples:
                s=local.to_device(raw,device);loss,_=_functional_loss(s,at(s['layer_idx']),cfg)
                require(bool(torch.isfinite(loss)),'nonfinite actual functional loss');total+=weights[id(raw)]*float(loss)
        return total
    initial=evaluate();best_loss=initial;best=tables();history=[];groups={}
    for s in samples:groups.setdefault(s['source_id'],{}).setdefault((s['id'],s['step'],s['phase']),[]).append(s)
    order=sorted(groups);seen=set();seen_sources=set()
    for source in groups.values():
        for events in source.values():require(len(events)==28 and {s['layer_idx'] for s in events}==set(range(28)),'one event per28 layer for each sampled phase')
    for step in range(steps):
        sid=order[step%len(order)];keys=sorted(groups[sid]);batch=groups[sid][keys[(step//len(order))%len(keys)]]
        optimizer.zero_grad(set_to_none=True);value=0.;bridge_error=0.
        for raw in batch:
            s=local.to_device(raw,device);layer=s['layer_idx'];loss,delta=_functional_loss(s,at(layer),cfg,True)
            require(bool(torch.isfinite(loss)),'nonfinite loss');(loss/28).backward();value+=float(loss.detach())/28;bridge_error=max(bridge_error,delta);seen.add(layer)
        require(all(g.grad is not None and bool(torch.isfinite(g.grad).all()) for g in logg),'missing/nonfinite K112 gradient')
        optimizer.step();require(all(bool(torch.isfinite(g).all()) for g in logg),'nonfinite log gain update')
        seen_sources.add(sid);history.append({'phase':'K112_actual_forward_proxy_backward','step':step+1,'source_id':sid,'loss_before_update':value,'proxy_vs_packed_max_abs':bridge_error})
        if (step+1)%32==0 or step+1==steps:
            score=evaluate()
            if score<best_loss:best_loss=score;best=tables()
            history.append({'phase':'full32_source_balanced_hardRE_checkpoint','step':step+1,'loss':score,'best_loss':best_loss})
    if steps:require(seen==set(range(28)),'all28 layers must update')
    require(bool(torch.isfinite(best).all()) and bool((best>0).all()),'invalid K112 output')
    after=local._provenance(samples,cfg)
    require(provenance['callback_sources']==after['callback_sources'] and provenance['module_sha256']==after['module_sha256'] and sha(__file__)==module_sha,'fit source changed')
    require(all(sha(p)==h for p,h in cfg.get('legacy_source_hashes',{}).items()),'legacy source changed')
    require(all(sha(path)==digest for path,digest in numerical_sources.items()),'norm/hardRE source changed')
    if theta_sha is not None:require(sha(cfg['theta_path'])==theta_sha,'V0 artifact changed')
    result={'schema':SCHEMA,'base_profile':'r7_joint_k1_gt_cond','parameter_count':112,
        'base_book_sha256':BOOK_SHA,'base_metadata_factor_sha256':FACTOR_SHA,
        'base_value_theta_sha256':theta_sha,'base_value_parameter_fp32_sha256':vfp,
        'key_head_factors':best.tolist(),'fit_provenance':{**after,'kgain_numerical_sources':numerical_sources},
        'fit_summary':{'objective':'same-student q/text restored-native-visual teacher projected through frozen W_O',
            'initial_actual_loss':initial,'best_actual_loss':best_loss,'updates':steps,'layers_updated':sorted(seen),'source_ids_seen':sorted(seen_sources),
            'book_J_mean_and_V_fixed':True,'K_residuals_hardRE':True,'assignment_gradient':'no J update; hard sign/alpha stop gradient; stored gain casts differentiated by declared proxy',
            'existing_K112_slots_replaced':True,'additional_deployment_tensor_fields':0,'student_state_refreshes':0,
            'module_sha256':module_sha,'task_quality_or_GPU_identity_claim':False},'history':history}
    if cfg.get('output'):
        out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
        local.write_new(out/'kgain112.json',{k:v for k,v in result.items() if k!='history'});local.write_new(out/'fit_history.json',history)
    return result


def cpu_check(output,legacy_a2):
    """Actual frozen Torch codec arithmetic only; no whole bank/reader/GPU."""
    _pins();a2=Path(legacy_a2).resolve();require(sha(a2/'round7_key_codec.py')==bridge._DEFAULT_SOURCES['a2/round7_key_codec.py'],'original codec SHA')
    with bridge._legacy_paths(a2,a2.parent/'experiments',a2):
        from round7_key_codec import key_amplitude,decode_key,encode_sign_depth
        torch.manual_seed(9);source=torch.randn(1,4,32,128).float();book=torch.randn(16,4).half()
        code=torch.randint(0,256,(1,4,32,16),dtype=torch.uint8);template=_template(code,book)
        g0=torch.tensor([.7,1.3,1.1,.9],dtype=torch.float32)
        gain=(key_amplitude(source,template,'norm').half().float()*g0[None,:,None,None]).half()
        lookup=torch.full((2,4,2),-1,dtype=torch.int32);lookup[0,1,0]=0;lookup[0,1,1]=1;lookup[1,0,0]=2
        selected=[('k',27,1,0),('k',27,1,1),('v',27,0,0)]
        state=SimpleNamespace(direction_code=code,direction_gain=gain,direction_mean=torch.zeros(1,4,1,128,dtype=torch.float16),
            v_code=torch.zeros(1,4,128,1,dtype=torch.int32),v_scale=torch.ones(1,4,128,1,dtype=torch.float16),v_min=torch.zeros(1,4,128,1,dtype=torch.float16),
            depth_residual_lookup=lookup,depth_residual_code=torch.zeros(3,32,4,dtype=torch.int32),depth_residual_alpha=torch.zeros(3,32,dtype=torch.float16))
        rc,ra,_,_=encode_sign_depth(source[0,1],decode_key(code,gain,book)[0,1],2)
        for i in range(2):state.depth_residual_code[i].copy_(rc[i]);state.depth_residual_alpha[i].copy_(ra[i])
        snapshots={n:v.clone() for n,v in vars(state).items()}
        identity,_=apply_kgain_fields(state,source,g0,book,selected,27,amplitude_fn=key_amplitude,encode_sign_depth_fn=encode_sign_depth)
        require(torch.equal(identity.direction_gain,gain) and torch.equal(identity.depth_residual_code,state.depth_residual_code) and torch.equal(identity.depth_residual_alpha,state.depth_residual_alpha),'original g0 arithmetic identity')
        newg=torch.tensor([.8,1.5,.95,1.05],requires_grad=True)
        result,stats=apply_kgain_fields(state,source,newg,book,selected,27,amplitude_fn=key_amplitude,encode_sign_depth_fn=encode_sign_depth)
        expected=(key_amplitude(source,template,'norm').half().float()*newg[None,:,None,None]).half()
        require(torch.equal(result.direction_gain,expected),'exact double-Half norm/factor gain')
        full=newg.detach().repeat(28,1).requires_grad_(True)
        fullstate,_=apply_kgain_fields(state,source,full,book,selected,27,amplitude_fn=key_amplitude,encode_sign_depth_fn=encode_sign_depth)
        result.direction_gain.float().square().mean().backward();fullstate.direction_gain.float().square().mean().backward()
        require(newg.grad is not None and bool(torch.isfinite(newg.grad).all()) and torch.equal(newg.grad,full.grad[27]) and not bool(full.grad[:27].any()),'local4/full112 gain gradient')
        require(stats['changed_K_code_elements']+stats['changed_K_alpha_elements']>0,'hardRE actually changes K residuals')
        require(torch.equal(result.depth_residual_code[2],state.depth_residual_code[2]) and torch.equal(result.depth_residual_alpha[2],state.depth_residual_alpha[2]),'fixed V residual')
        require(all(torch.equal(getattr(state,n),v) for n,v in snapshots.items()),'original state mutated')
        empty=copy(state);empty.depth_residual_lookup=torch.full_like(lookup,-1);empty.depth_residual_code=torch.empty(0,32,4,dtype=torch.int32);empty.depth_residual_alpha=torch.empty(0,32,dtype=torch.float16)
        zerostate,_=apply_kgain_fields(empty,source,g0,book,[],27,amplitude_fn=key_amplitude,encode_sign_depth_fn=encode_sign_depth)
        require(zerostate.depth_residual_code is empty.depth_residual_code and zerostate.depth_residual_alpha is empty.depth_residual_alpha,'empty pool unexpectedly replaced')
    out=Path(output).resolve();require(out.is_relative_to(Path('${EVIDENCE_ROOT}').resolve()),'CPU output must be on E evidence disk');out.mkdir(parents=True,exist_ok=False)
    result={'status':'PASS_CPU_K112_ARITHMETIC_ONLY_PENDING_WHOLE_BANK_GPU_IDENTITY',
        'checks':['original_g0_doubleHalf_gain_and_depth2_RE_identity','new_gain_matches_original_source_norm_function','J_mean_V_and_Vresidual_fixed','old_input_state_not_mutated',
                  'hardRE_sign_alpha_stopped_and_actual_Kresidual_change','local4_full112_layer27_gradient_equal','empty_Kpool_no_extra_storage'],
        'actual_GPU_calls':0,'whole_bank_factory_or_reader_positive':False,'source_sha256':sha(__file__),
        'source_codec_sha256':sha(a2/'round7_key_codec.py'),'fixed_helper_sha256':{'calibration':CALIBRATION_SHA,'value_ops':VALUE_OPS_SHA,'bridge':BRIDGE_SHA}}
    local.write_new(out/'CPU_RECEIPT.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-a2',type=Path,default=Path('${FROZEN_TRAJKV_ROOT}/code/a2'));args=parser.parse_args()
    require(args.cpu_check,'No automatic fit/GPU entry point; caller must explicitly invoke fit after Root admission')
    cpu_check(args.output,args.legacy_a2)
