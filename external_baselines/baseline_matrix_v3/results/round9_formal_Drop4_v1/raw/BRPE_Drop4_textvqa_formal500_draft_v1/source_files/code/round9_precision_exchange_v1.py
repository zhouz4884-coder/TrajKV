"""M1 whole-layer G4_256 / four-K-residual exchange; no automatic GPU work.

Old 16x4 book, V20/V fields and original ranking/policy remain charged. The
global compact lookup is signed int8 with all28 state views rebound. No reader
kernel is rewritten. Root alone runs the explicit synthetic GPU entry.
"""
from __future__ import annotations
import argparse
from copy import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import weakref
from types import SimpleNamespace
import torch
import r7_legacy_bridge_v1 as bridge

MODES=('Lookup8','Drop4','Exchange')
LOW_SHA='af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589'
FACTOR_SHA='8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea'
BRIDGE_SHA='b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b'
REFERENCE_SHA='e37810a2568ebb21f10ea10488829be0758311985c5dbdeebdd20fa450c5631a'

def require(v,m):
    if not v:raise ValueError('ROUND9_EXCHANGE: '+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def _unique(owners):
    seen={}
    for owner in owners:
        for t in owner.named_tensors().values():
            if t.numel():seen.setdefault(bridge._storage(t),t.untyped_storage().nbytes())
    return sum(seen.values())
def _fields(x):return dict(x) if isinstance(x,dict) else dict(vars(x))


def decode_exchange_key_fields(fields,book):
    """Actual Half base + sequential Half K residuals for16/256-word G4."""
    f=_fields(fields);code=f['direction_code'];words=int(book.shape[0])
    require(tuple(book.shape) in ((16,4),(256,4)) and book.dtype==torch.float16,'actual Half G4 book')
    require(code.dtype==torch.uint8 and code.shape[-1]==(16 if words==16 else 32),'K index format/book mismatch')
    ids=torch.stack((code&15,code>>4),-1).flatten(-2).long() if words==16 else code.long()
    template=book.float()[ids].flatten(-2)
    base=(template*f['direction_gain'].float()).half();heads=[]
    for h in range(4):
        value=base[:,h]
        for stage in range(2):
            slot=int(f['depth_residual_lookup'][0,h,stage])
            if slot>=0:
                rc=f['depth_residual_code'][slot]
                sign=(((rc.long()[...,None]>>torch.arange(32,device=rc.device))&1).flatten(-2).float()*2-1)
                value=(value.float()+sign[None]*f['depth_residual_alpha'][slot].float()[None,:,None]).half()
        heads.append(value)
    return torch.stack(heads,1)


def precision_exchange_fields(fields,source_rotated,original_book,high_book,original_K_head_factors_local4,*,
                              layer_idx,upgraded_layer,dropped_k_units,upgrade,compact=False,
                              encode_key=None,decode_key=None,encode_sign_depth=None):
    """Pure field transform -> (newdict, this-layer book, stats).

    compactFalse is a fixed-q calibration view: dropped slots are masked but
    retained old backing is NOT a storage saving. compactTrue makes fresh compact
    pools and a signed-int8 local lookup, for the deploying factory only.
    """
    f=_fields(fields);result=dict(f);look=f['depth_residual_lookup'].clone()
    require(tuple(look.shape)==(2,4,2),'original K/V lookup shape')
    drops=[tuple(int(v) for v in row) for row in dropped_k_units]
    require(len(drops)==len(set(drops)) and len(drops)<=4 and all(len(x)==3 and 0<=x[0]<28 and 0<=x[1]<4 and x[2] in (0,1) for x in drops),'zero-to-four unique K-only [layer,head,stage] donors; deployment factory requires four')
    for l,h,s in drops:
        if l==layer_idx:
            require(int(look[0,h,s])>=0,'donor K slot not originally selected');look[0,h,s]=-1
    for h in range(4):require(int(look[0,h,0])>=0 or int(look[0,h,1])<0,'cannot keep child after dropping K parent')
    active_upgrade=bool(upgrade and layer_idx==upgraded_layer);book=high_book if active_upgrade else original_book
    recoded=0
    if active_upgrade:
        require(tuple(high_book.shape)==(256,4) and high_book.dtype==torch.float16,'one whole-layer high book')
        require(source_rotated is not None and source_rotated.dtype==torch.float32 and tuple(source_rotated.shape)==(*f['direction_code'].shape[:-1],128),'actual centered-WHT FP32 source')
        module=importlib.import_module('round7_key_codec') if encode_key is None or decode_key is None or encode_sign_depth is None else None
        encode_key=encode_key or module.encode_key;decode_key=decode_key or module.decode_key;encode_sign_depth=encode_sign_depth or module.encode_sign_depth
        factors=original_K_head_factors_local4
        require(tuple(factors.shape)==(4,) and bool(torch.isfinite(factors).all()) and bool((factors>0).all()),'original positive local K factors')
        result['direction_code'],result['direction_gain']=encode_key(source_rotated,book,amplitude='norm',assignment='l2',gain_factor=factors[None,:,None,None])
        base=decode_key(result['direction_code'],result['direction_gain'],book)
        if bool((look[0]>=0).any()):
            result['depth_residual_code']=f['depth_residual_code'].detach().clone()
            result['depth_residual_alpha']=f['depth_residual_alpha'].detach().clone()
            for h in range(4):
                slots=[int(x) for x in look[0,h] if int(x)>=0]
                if not slots:continue
                codes,alphas,_,_=encode_sign_depth(source_rotated[0,h],base[0,h],len(slots))
                for stage,slot in enumerate(slots):
                    result['depth_residual_code'][slot].copy_(codes[stage]);result['depth_residual_alpha'][slot].copy_(alphas[stage]);recoded+=1
    # V pool rows are copied bit for bit even when K and V share one backing.
    for h in range(4):
        for stage in range(2):
            slot=int(f['depth_residual_lookup'][1,h,stage])
            if slot>=0:require(torch.equal(result['depth_residual_code'][slot],f['depth_residual_code'][slot]) and torch.equal(result['depth_residual_alpha'][slot],f['depth_residual_alpha'][slot]),'V residual changed')
    if compact:
        used=sorted(int(x) for x in look.reshape(-1) if int(x)>=0)
        require(len(used)==len(set(used)) and len(used)<128,'unique representable compact slots')
        n=f['direction_code'].shape[-2]
        rc=torch.empty((len(used),n,4),device=look.device,dtype=torch.int32)
        ra=torch.empty((len(used),n),device=look.device,dtype=torch.float16)
        remap={old:new for new,old in enumerate(used)}
        for old,new in remap.items():rc[new].copy_(result['depth_residual_code'][old]);ra[new].copy_(result['depth_residual_alpha'][old])
        mapped=torch.full_like(look,-1,dtype=torch.int8)
        for axis in range(2):
            for h in range(4):
                for s in range(2):
                    old=int(look[axis,h,s])
                    if old>=0:mapped[axis,h,s]=remap[old]
        result['depth_residual_code'],result['depth_residual_alpha'],look=rc,ra,mapped
    result['depth_residual_lookup']=look
    for name in ('direction_mean','v_code','v_scale','v_min','visual_pos'):
        if name in f:require(result[name] is f[name],'fixed mean/V/position field replaced')
    if not active_upgrade:
        require(result['direction_code'] is f['direction_code'] and result['direction_gain'] is f['direction_gain'],'nonupgraded K base changed')
    return result,book,{'layer':layer_idx,'whole_layer_upgraded':active_upgrade,'retained_K_stages_reencoded':recoded,
                       'compact_real_pools':compact,'calibration_mask_is_not_storage_saving':not compact,'retained_slots':int((look>=0).sum())}


@torch.no_grad()
def build_round9_precision_exchange_owners(native,vp,variant,*,mode,candidate=None,**legacy_kwargs):
    require(mode in MODES,'unknown precision exchange control')
    require(sha(bridge.__file__)==BRIDGE_SHA,'frozen bridge changed')
    module_sha=sha(__file__);candidate=dict(candidate or {})
    owners,metadata=bridge.build_r7_owners(native,vp,variant,**legacy_kwargs)
    require(metadata['variant']['name']=='r7_joint_k1_gt_cond' and metadata['selected_stage_count']==43 and
            metadata['fixed_assets']['direction_book_path']['sha256']==LOW_SHA and metadata['fixed_assets']['metadata_factor_path']['sha256']==FACTOR_SHA and
            metadata['fixed_assets']['round7_policy_path']['sha256']==bridge._DEFAULT_ASSETS['round7_policy_path'],'original low gtcond43 required')
    n=int(vp.numel());require(n>=256 and n%32==0,'M1 Nv>=256 complete words only')
    before=_unique(owners);old_shared=owners[0].shared;low=old_shared['book'];original=metadata['selected_stages']
    drops=[];layer=-1;high=None;formats=descriptor=None
    if mode!='Lookup8':
        if candidate.get('schema') is not None:
            require(candidate['schema']=='trajkv.round9.precision_exchange_profile.v1' and candidate['base_profile']=='r7_joint_k1_gt_cond' and
                    candidate['low_book_sha256']==LOW_SHA and candidate['base_metadata_factor_sha256']==FACTOR_SHA,'profile identity differs')
        layer=candidate['upgraded_layer'];require(type(layer) is int and 0<=layer<28,'one complete upgraded layer')
        drops=[tuple(x) for x in candidate['dropped_k_units']]
        require(len(drops)==len(set(drops))==4 and all(len(x)==3 and all(type(v) is int for v in x) for x in drops),'exact four donor descriptors')
        require(all(['k',*x] in [list(z) for z in original] for x in drops),'donor not in original selected K coverage')
        # Both nontrivial profiles retain a real format/selection description.
        formats=torch.zeros(28,device=vp.device,dtype=torch.uint8)
        descriptor=torch.tensor(drops,device=vp.device,dtype=torch.uint8)
        require(formats.untyped_storage().nbytes()==28 and descriptor.untyped_storage().nbytes()==12,'real format/donor table bytes')
        drops=[tuple(x) for x in descriptor.cpu().tolist()]
        if mode=='Exchange':
            require(sha(candidate['high_book_path'])==candidate['high_book_sha256'],'fixed high-book SHA')
            high=torch.tensor(load(candidate['high_book_path'])['centers'],device=vp.device,dtype=torch.float16)
            require(tuple(high.shape)==(256,4) and bool(torch.isfinite(high).all()) and high.untyped_storage().nbytes()==2048,'actual G4_256 high book')
            formats[layer]=1
    old_refs=[weakref.ref(old_shared['global_residual_lookup'])]
    for owner in owners:
        for name in ('depth_residual_lookup','depth_residual_code','depth_residual_alpha'):
            value=getattr(owner.state,name,None)
            if isinstance(value,torch.Tensor):old_refs.append(weakref.ref(value))
    value=None
    lookup8=torch.full((28,2,4,2),-1,device=vp.device,dtype=torch.int8)
    common=dict(old_shared);common['global_residual_lookup']=lookup8
    if mode!='Lookup8':common.update(exchange_layer_format=formats,exchange_donor_descriptor=descriptor)
    if mode=='Exchange':common['exchange_high_book']=high
    common['exchange_low_book']=low
    stats=[]
    with bridge._legacy_paths(Path(legacy_kwargs['legacy_a2']),Path(legacy_kwargs['legacy_experiments']),Path(legacy_kwargs['legacy_scripts'])):
        from fast_wht import wht
        from round7_key_codec import encode_key,decode_key,encode_sign_depth
        for i,owner in enumerate(owners):
            old=owner.state
            upgrade=mode=='Exchange' and int(formats[i])==1
            rotated=wht(native.layers[i].keys.index_select(2,vp).float()-old.direction_mean.float()) if upgrade else None
            if mode=='Lookup8':
                # No dropping, but fresh pools and int8 view mapping are real.
                fields=dict(vars(old));look=old.depth_residual_lookup
                used=sorted(int(x) for x in look.reshape(-1) if int(x)>=0)
                require(len(used)==len(set(used)) and len(used)<128,'lossless int8 lookup range')
                codes=torch.empty((len(used),n,4),device=vp.device,dtype=torch.int32)
                alphas=torch.empty((len(used),n),device=vp.device,dtype=torch.float16)
                mapped=torch.full_like(look,-1,dtype=torch.int8)
                for new,slot in enumerate(used):
                    codes[new].copy_(old.depth_residual_code[slot]);alphas[new].copy_(old.depth_residual_alpha[slot])
                    mapped[look==slot]=new
                fields.update(depth_residual_code=codes,depth_residual_alpha=alphas,depth_residual_lookup=mapped)
                actual_book=low;detail={'layer':i,'whole_layer_upgraded':False,'retained_K_stages_reencoded':0,'compact_real_pools':True}
            else:
                fields,actual_book,detail=precision_exchange_fields(vars(old),rotated,low,high,old_shared['encoder_factor.key_head_factors'][i],
                    layer_idx=i,upgraded_layer=layer,dropped_k_units=drops,upgrade=upgrade,compact=True,
                    encode_key=encode_key,decode_key=decode_key,encode_sign_depth=encode_sign_depth)
            newstate=copy(old)
            for name,value in fields.items():setattr(newstate,name,value)
            lookup8[i].copy_(fields['depth_residual_lookup']);newstate.depth_residual_lookup=lookup8[i]
            owner.state=newstate;owner.shared={**common,'book':actual_book};stats.append(detail)
    old=old_shared=fields=newstate=look=used=codes=alphas=mapped=value=rotated=None
    gc.collect()
    require(all(r() is None for r in old_refs),'old int32 lookup or original residual backing still retained')
    actual=_unique(owners)
    expected_delta=-1344 if mode=='Lookup8' else -72*n-1344+40 if mode=='Drop4' else 744-8*n
    require(actual-before==expected_delta,'actual unique bytes do not match exchange ledger')
    require(lookup8.untyped_storage().nbytes()==448 and all(bridge._storage(o.state.depth_residual_lookup)==bridge._storage(lookup8) for o in owners),'all28 int8 lookup views must share one448B backing')
    new_selected=[x for x in original if not (x[0]=='k' and tuple(x[1:]) in drops)]
    require(len(new_selected)==(43 if mode=='Lookup8' else 39) and sum(x[0]=='v' for x in new_selected)==20,'V20 and total stage count')
    forbidden={bridge._storage(t) for l in native.layers for t in (l.keys,l.values)}
    for owner in owners:
        for t in owner.named_tensors().values():require(not t.requires_grad and t.grad_fn is None and bridge._storage(t) not in forbidden,'retained native prefix/autograd graph')
    require(all(sha(p)==v['sha256'] for p,v in metadata['verified_legacy_source_pins'].items()),'legacy source changed')
    require(all(sha(v['path'])==v['sha256'] for v in metadata['fixed_assets'].values()),'original asset changed')
    if mode=='Exchange':require(sha(candidate['high_book_path'])==candidate['high_book_sha256'],'high book changed')
    require(sha(__file__)==module_sha,'new exchange source changed')
    metadata.update(selected_stage_count=len(new_selected),selected_stages=new_selected,original_selected_stages=original,
        returned_owners_unique_tensor_capacity_bytes=actual)
    metadata['round9_precision_exchange']={'schema':'trajkv.round9.precision_exchange.v1','mode':mode,'candidate':candidate,
        'upgraded_layer':layer if mode=='Exchange' else None,'dropped_k_units':drops,'original_ranking_policy_retained_and_charged':True,
        'lookup_dtype':'torch.int8','lookup_bytes':448,'old_lookup_and_original_pools_released':True,
        'format_bytes':0 if formats is None else 28,'donor_descriptor_bytes':0 if descriptor is None else 12,
        'extra_high_book_bytes':2048 if high is not None else 0,'old_low_book_bytes':128,
        'before_unique_bytes':before,'after_unique_bytes':actual,'expected_delta_bytes':expected_delta,'actual_delta_bytes':actual-before,
        'V20_base_parameters_and_residual_values_unchanged':True,'layer_stats':stats,'source_sha256':module_sha,
        'calibration_or_task_gain_claim':False,'reader_int8_GPU_acceptance':'PENDING_EXPLICIT_ROOT_CONTROL'}
    return owners,metadata


def _cpu_wht(x):
    y=x.float();shape=y.shape
    for width in (1,2,4,8,16,32,64):
        z=y.reshape(*shape[:-1],128//(2*width),2,width);a,b=z[...,0,:],z[...,1,:]
        y=torch.cat((a+b,a-b),-1).reshape(shape)
    return y*(128**-.5)


def cpu_check(output,legacy_a2):
    from unittest.mock import patch
    import types
    a2=Path(legacy_a2);root=Path(__file__).resolve().parents[1]
    cfg=load(root/'configs/r7_research/M1_R7gt43_docvqa_smoke4_v1.json')
    selected=[list(x) for x in cfg['variant']['selected_stages'][:43]]
    units=[x[1:] for x in selected if x[0]=='k'];layer=units[-1][0];drops=[x for x in units if x[0]!=layer][:4]
    low_path=root/'configs/r7_research/frozen_assets_v2/gaussian_vq_g4_16.json'
    high_path=root/'configs/r7_research/frozen_assets_v2/gaussian_vq_g4_256.json'
    low=torch.tensor(load(low_path)['centers'],dtype=torch.float16);high=torch.tensor(load(high_path)['centers'],dtype=torch.float16)
    torch.manual_seed(0);n=256
    native=SimpleNamespace(layers=[SimpleNamespace(keys=torch.randn(1,4,n+4,128).half()*.1,values=torch.randn(1,4,n+4,128).half()*.1) for _ in range(28)])
    vp=torch.arange(4,n+4)
    def fake_original(*args,**kwargs):
        # Explicit structural fixture, not original GPU encoder evidence.
        look=torch.full((28,2,4,2),-1,dtype=torch.int32)
        shared={'book':low,'global_residual_lookup':look,'encoder_ranking_bytes':torch.zeros(448,dtype=torch.uint8),'encoder_policy_bytes':torch.zeros(5,dtype=torch.uint8)}
        for name in ('key_head_factors','value_head_factors','value_center_offsets'):shared['encoder_factor.'+name]=torch.ones(28,4)
        owners=[]
        for i in range(28):
            local=[x for x in selected if x[1]==i]
            state=SimpleNamespace(direction_code=torch.zeros(1,4,n,16,dtype=torch.uint8),direction_gain=torch.ones(1,4,n,1,dtype=torch.float16),direction_mean=torch.zeros(1,4,1,128,dtype=torch.float16),
                v_code=torch.zeros(1,4,128,n//32,dtype=torch.int32),v_scale=torch.ones(1,4,128,1,dtype=torch.float16),v_min=torch.zeros(1,4,128,1,dtype=torch.float16),visual_pos=vp,depth_residual_lookup=look[i])
            if local:
                state.depth_residual_code=torch.zeros(len(local),n,4,dtype=torch.int32);state.depth_residual_alpha=torch.zeros(len(local),n,dtype=torch.float16)
                for slot,(axis,_,head,stage) in enumerate(local):look[i,0 if axis=='k' else 1,head,stage]=slot
            owners.append(bridge.R7LegacyOwner(i,state,shared,None,n+4))
        assets={}
        for name,digest in bridge._DEFAULT_ASSETS.items():
            filename={'direction_book_path':'gaussian_vq_g4_16.json','metadata_factor_path':'factors_for_book16_inherited.json','round7_policy_path':'policy_k1_cond.json'}[name]
            path=low_path.parent/filename;assets[name]={'path':str(path),'sha256':digest}
        return owners,{'variant':{'name':'r7_joint_k1_gt_cond'},'selected_stage_count':43,'selected_stages':selected,
                       'fixed_assets':assets,'verified_legacy_source_pins':{},'legacy_sources':{}}
    candidate={'upgraded_layer':layer,'dropped_k_units':drops,'high_book_path':str(high_path),'high_book_sha256':sha(high_path)}
    stub=types.ModuleType('fast_wht');stub.wht=_cpu_wht
    results=[]
    with bridge._legacy_paths(a2,a2.parent/'experiments',a2):
        from round7_key_codec import encode_key,decode_key,encode_sign_depth
        # Actual source-codec pure-Torch field comparison, including G256.
        fields=dict(vars(fake_original()[0][layer].state));rot=_cpu_wht(native.layers[layer].keys.index_select(2,vp).float())
        masked,book,_=precision_exchange_fields(fields,rot,low,high,torch.ones(4),layer_idx=layer,upgraded_layer=layer,dropped_k_units=drops,
            upgrade=True,compact=False,encode_key=encode_key,decode_key=decode_key,encode_sign_depth=encode_sign_depth)
        compact,book,_=precision_exchange_fields(fields,rot,low,high,torch.ones(4),layer_idx=layer,upgraded_layer=layer,dropped_k_units=drops,
            upgrade=True,compact=True,encode_key=encode_key,decode_key=decode_key,encode_sign_depth=encode_sign_depth)
        require(torch.equal(decode_exchange_key_fields(masked,high),decode_exchange_key_fields(compact,high)),'compact G256 sequential-Half decode differs')
        for mode in MODES:
            with patch.object(bridge,'build_r7_owners',side_effect=fake_original),patch.dict(sys.modules,{'fast_wht':stub}):
                owners,meta=build_round9_precision_exchange_owners(native,vp,{},mode=mode,candidate=candidate,legacy_a2=str(a2),legacy_experiments=str(a2.parent/'experiments'),legacy_scripts=str(a2))
            value=meta['round9_precision_exchange'];results.append({'mode':mode,'actual_delta_bytes':value['actual_delta_bytes'],'expected_delta_bytes':value['expected_delta_bytes'],
                'old_lookup_and_pools_released':value['old_lookup_and_original_pools_released'],'lookup_bytes':value['lookup_bytes'],'stages':meta['selected_stage_count']})
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    result={'status':'PASS_CPU_FIELD_AND_EXPLICIT_FAKE_OWNER_STORAGE_CONTROLS_ONLY','Nv':256,'cases':results,
        'generic_G256_Half_decode_compact_equals_masked':True,'source_sha256':sha(__file__),'actual_GPU_calls':0,
        'production_whole_bank_reader_GPU_positive':False,'fixture_scope':'Storage geometry/weak release and actual pure Torch codec arithmetic; fake owners are not measured model outputs'}
    write(out/'CPU_RECEIPT.json',result);print(json.dumps(result));return result


@torch.no_grad()
def gpu_check(config_path,candidate_path,output):
    import r7_bridge_gpu_check_v1 as ref
    require(sha(ref.__file__)==REFERENCE_SHA,'frozen reference helper changed')
    cfg=load(config_path);ref.validate_metadata(cfg);variant=ref.profile(cfg);candidate=load(candidate_path)
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6),'actual SM86 CUDA required')
    from transformers.cache_utils import DynamicCache
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    pins={str(Path(__file__).resolve()):sha(__file__),str(config_path):sha(config_path),str(candidate_path):sha(candidate_path),candidate['high_book_path']:candidate['high_book_sha256']}
    write(out/'manifest.json',{'schema':'trajkv.round9.exchange_synthetic.v1','sources':pins,'Nv':[256,1792],
        'diagnostic_reference_native_storage_not_deployment_savings':True,'model_images_GT_loaded':False,'quality_QA_added':0})
    kwargs={n:cfg[n] for n in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source')};kwargs['expected_asset_sha256']=ref.ASSETS
    records=[]
    for n in (256,1792):
        torch.manual_seed(0);native=DynamicCache()
        for i in range(28):
            k=torch.randn(1,4,n+4,128,device='cuda',dtype=torch.float16)*.15;v=torch.randn_like(k)*.1;native.update(k,v,i)
        k=v=None;vp=torch.arange(4,n+4,device='cuda',dtype=torch.long)
        original,base=bridge.build_r7_owners(native,vp,variant,**kwargs);before=_unique(original)
        high=torch.tensor(load(candidate['high_book_path'])['centers'],device='cuda',dtype=torch.float16)
        native_hash=[(ref.tensor_identity(x.keys),ref.tensor_identity(x.values)) for x in native.layers]
        source=importlib.import_module('round7_key_codec');wht=importlib.import_module('fast_wht').wht
        modes=[]
        for mode in MODES:
            owners,meta=build_round9_precision_exchange_owners(native,vp,variant,mode=mode,candidate=candidate,**kwargs)
            expected=-1344 if mode=='Lookup8' else -72*n-1304 if mode=='Drop4' else 744-8*n
            require(_unique(owners)-before==expected,'actual complete bytes mismatch')
            checks=[];qp=torch.arange(n+4,n+8,device='cuda');tp=torch.cat((torch.arange(4,device='cuda'),qp))
            for i,(old,owner) in enumerate(zip(original,owners)):
                upgrade=mode=='Exchange' and i==candidate['upgraded_layer']
                if mode=='Lookup8':fields=dict(vars(old.state));actual_book=old.shared['book']
                else:
                    rotated=wht(native.layers[i].keys.index_select(2,vp).float()-old.state.direction_mean.float()) if upgrade else None
                    fields,actual_book,_=precision_exchange_fields(vars(old.state),rotated,old.shared['book'],high,old.shared['encoder_factor.key_head_factors'][i],
                        layer_idx=i,upgraded_layer=candidate['upgraded_layer'],dropped_k_units=candidate['dropped_k_units'],upgrade=upgrade,compact=False,
                        encode_key=source.encode_key,decode_key=source.decode_key,encode_sign_depth=source.encode_sign_depth)
                refstate=copy(old.state)
                for name,value in fields.items():setattr(refstate,name,value)
                reference=bridge.R7LegacyOwner(i,refstate,{**old.shared,'book':actual_book},old._reader,n+4)
                for name in ('v_code','v_scale','v_min','direction_mean'):require(torch.equal(getattr(old.state,name),getattr(owner.state,name)),'V/base mean changed')
                for h in range(4):
                    for stage in range(2):
                        a=int(old.state.depth_residual_lookup[1,h,stage]);b=int(owner.state.depth_residual_lookup[1,h,stage])
                        require((a>=0)==(b>=0),'V coverage changed')
                        if a>=0:require(torch.equal(old.state.depth_residual_code[a],owner.state.depth_residual_code[b]) and torch.equal(old.state.depth_residual_alpha[a],owner.state.depth_residual_alpha[b]),'V residual values changed')
                if not upgrade:require(torch.equal(old.state.direction_code,owner.state.direction_code) and torch.equal(old.state.direction_gain,owner.state.direction_gain),'nonupgraded K base changed')
                require(torch.equal(decode_exchange_key_fields(fields,actual_book),decode_exchange_key_fields(vars(owner.state),actual_book)),'field compact/lookup decode differs')
                q=torch.randn(1,28,4,128,device='cuda',dtype=torch.float16)*.2
                tk=torch.cat((native.layers[i].keys[:,:,:4],torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.15),2)
                tv=torch.cat((native.layers[i].values[:,:,:4],torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.1),2)
                args=dict(exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,scale=128**-.5)
                left,right=reference.attend(q,**args),owner.attend(q,**args)
                require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()) and torch.equal(left,right),'int8/compact vs int32 masked reader is not bitwise')
                checks.append({'layer':i,'field_decode_bitwise_equal':True,'reader_bitwise_equal':True})
            modes.append({'mode':mode,'status':'PASS_FIELDS_READER_BYTES','actual_complete_delta':_unique(owners)-before,'expected_delta':expected,'metadata':meta,'all28_checks':checks})
        require(native_hash==[(ref.tensor_identity(x.keys),ref.tensor_identity(x.values)) for x in native.layers],'native prefix mutated')
        records.append({'Nv':n,'status':'PASS_FIELDS_READER_BYTES','modes':modes})
    require(all(sha(p)==h for p,h in pins.items()),'source/profile/book changed')
    with (out/'records.jsonl').open('x',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    result={'status':'PASS_ALL2_PRECISION_EXCHANGE_FIELDS_READER_ACTUAL_BYTES','cases':2,'quality_QA_added':0,'records_sha256':sha(out/'records.jsonl')}
    write(out/'complete.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--cpu-check',action='store_true');m.add_argument('--gpu-check',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-a2',type=Path,default=Path('${FROZEN_TRAJKV_ROOT}/code/a2'));p.add_argument('--config',type=Path);p.add_argument('--candidate',type=Path)
    a=p.parse_args()
    if a.cpu_check:cpu_check(a.output,a.legacy_a2)
    else:
        require(a.config is not None and a.candidate is not None,'explicit frozen asset config and candidate/profile required')
        gpu_check(a.config,a.candidate,a.output)
