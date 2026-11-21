"""Frozen U3/D11 cal-only precision exchange; numerical fields reuse V1.

No model/GPU/remote work at import. One shared high book; three whole layers;
eleven distinct stage0 K donors; V20 unchanged. V1/source results stay intact.
"""
from __future__ import annotations
import argparse,gc,hashlib,importlib,json,sys,weakref
from copy import copy,deepcopy
from pathlib import Path
import torch
import r7_legacy_bridge_v1 as bridge
import round9_precision_exchange_v1 as single

V1_SHA='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'
CAL_SHA='8b986799062ef8d3fbc67624344d1740cfea4726b64aa3fc4d995ffaebc54438'
COLLECTION_SHA='c838b3a79cb970233a9579f8972ae4276157fad1001fd9b4e9118945bc0dc521'
COST_SHA='28de9d0ecbb30a38eb81d9dbe36fac7213b66334e2af19df55b53936b98fd81a'
HIGH_SHA='9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae'
PROFILE_SHA='349514921518520383c375b9b5ead97ceaab356484f080cb546a91b1e188aa1a'
LAYERS=[14,22,26]
MODES=('Lookup8','DropD','Exchange')
decode_exchange_key_fields=single.decode_exchange_key_fields
precision_exchange_fields=single.precision_exchange_fields
require=single.require;sha=single.sha;load=single.load;write=single.write;_unique=single._unique


def validate_profile(p,*,allow_diagnostic=False):
    require(p.get('schema')=='trajkv.round9.precision_exchange_profile.v2' and p.get('base_profile')=='r7_joint_k1_gt_cond','exact U3D11 profile schema')
    role=p.get('profile_role');require(role=='cal_only_DP_selected_U3D11' or allow_diagnostic and role=='GPU_nonempty_KRE_fixture_only','diagnostic profile cannot be a task candidate')
    require(p.get('upgraded_layers')==LAYERS and p.get('U')==3 and p.get('D')==11,'frozen three whole layers / eleven donors')
    drops=p.get('dropped_k_units');require(isinstance(drops,list) and len(drops)==11 and len({tuple(x) for x in drops})==11 and
        all(isinstance(x,list) and len(x)==3 and all(type(v) is int for v in x) and 0<=x[0]<28 and 0<=x[1]<4 and x[2]==0 for x in drops),'eleven unique stage0 K head donors')
    require(p.get('low_book_sha256')==single.LOW_SHA and p.get('base_metadata_factor_sha256')==single.FACTOR_SHA and p.get('high_book_sha256')==HIGH_SHA,'fixed books/f0')
    q=p['source_provenance'];require(q.get('calibrator')==CAL_SHA and q.get('exchange_helper')==V1_SHA and q.get('collection_complete')==COLLECTION_SHA,'actual cal source epoch must remain8b/c838/15e')
    sel=p['selection_provenance'];require(sel.get('candidate_cost_table_sha256')==COST_SHA and sel.get('task_scores_or_answers_used') is False,'cal-table-only DP provenance')
    return p


@torch.no_grad()
def build_round9_precision_exchange_owners(native,vp,variant,*,mode,candidate=None,**legacy_kwargs):
    require(mode in MODES and sha(single.__file__)==V1_SHA and sha(bridge.__file__)==single.BRIDGE_SHA,'mode/frozen numerical helpers')
    module_sha=sha(__file__);candidate=deepcopy(candidate or {})
    if mode!='Lookup8':validate_profile(candidate,allow_diagnostic=True)
    owners,metadata=bridge.build_r7_owners(native,vp,variant,**legacy_kwargs)
    require(metadata['variant']['name']=='r7_joint_k1_gt_cond' and metadata['selected_stage_count']==43 and
        metadata['fixed_assets']['direction_book_path']['sha256']==single.LOW_SHA and metadata['fixed_assets']['metadata_factor_path']['sha256']==single.FACTOR_SHA and
        metadata['fixed_assets']['round7_policy_path']['sha256']==bridge._DEFAULT_ASSETS['round7_policy_path'],'original low43/f0')
    n=int(vp.numel());require(n>=256 and n%32==0,'M1 complete words and Nv>=256')
    before=_unique(owners);old_shared=owners[0].shared;low=old_shared['book'];original=metadata['selected_stages']
    drops=[];formats=descriptor=high=None
    if mode!='Lookup8':
        drops=[tuple(x) for x in candidate['dropped_k_units']]
        require(all(['k',*x] in [list(v) for v in original] for x in drops),'donor not originally selected')
        formats=torch.zeros(28,device=vp.device,dtype=torch.uint8)
        descriptor=torch.tensor(drops,device=vp.device,dtype=torch.uint8)
        require(formats.untyped_storage().nbytes()==28 and descriptor.untyped_storage().nbytes()==33,'actual28+33 byte policy')
        drops=[tuple(x) for x in descriptor.cpu().tolist()]
        if mode=='Exchange':
            require(sha(candidate['high_book_path'])==HIGH_SHA,'high book changed')
            high=torch.tensor(load(candidate['high_book_path'])['centers'],device=vp.device,dtype=torch.float16)
            require(tuple(high.shape)==(256,4) and bool(torch.isfinite(high).all()) and high.untyped_storage().nbytes()==2048,'one real2048B high book')
            formats[torch.tensor(LAYERS,device=vp.device)]=1
    old_refs=[weakref.ref(old_shared['global_residual_lookup'])]
    for owner in owners:
        for name in ('depth_residual_lookup','depth_residual_code','depth_residual_alpha'):
            value=getattr(owner.state,name,None)
            if isinstance(value,torch.Tensor):old_refs.append(weakref.ref(value))
    value=None;lookup8=torch.full((28,2,4,2),-1,device=vp.device,dtype=torch.int8)
    common={**old_shared,'global_residual_lookup':lookup8,'exchange_low_book':low}
    if mode!='Lookup8':common.update(exchange_layer_format=formats,exchange_donor_descriptor=descriptor)
    if mode=='Exchange':common['exchange_high_book']=high
    stats=[]
    with bridge._legacy_paths(Path(legacy_kwargs['legacy_a2']),Path(legacy_kwargs['legacy_experiments']),Path(legacy_kwargs['legacy_scripts'])):
        from fast_wht import wht
        from round7_key_codec import encode_key,decode_key,encode_sign_depth
        for i,owner in enumerate(owners):
            old=owner.state;up=mode=='Exchange' and int(formats[i])==1
            rotated=wht(native.layers[i].keys.index_select(2,vp).float()-old.direction_mean.float()) if up else None
            local_drops=[x for x in drops if x[0]==i]
            require(len(local_drops)<=4,'original stage0/4head local scope')
            fields,book,detail=precision_exchange_fields(vars(old),rotated,low,high,old_shared['encoder_factor.key_head_factors'][i],
                layer_idx=i,upgraded_layer=i,dropped_k_units=local_drops,upgrade=up,compact=True,
                encode_key=encode_key,decode_key=decode_key,encode_sign_depth=encode_sign_depth)
            new=copy(old)
            for name,value in fields.items():setattr(new,name,value)
            lookup8[i].copy_(fields['depth_residual_lookup']);new.depth_residual_lookup=lookup8[i]
            owner.state=new;owner.shared={**common,'book':book};stats.append(detail)
    old=old_shared=fields=new=value=rotated=None;gc.collect()
    require(all(r() is None for r in old_refs),'old lookup/pool backing retained')
    delta=-1344 if mode=='Lookup8' else -198*n-1283 if mode=='DropD' else 765-6*n
    actual=_unique(owners);require(actual-before==delta,'actual U3D11 unique bytes differ from ledger')
    require(lookup8.untyped_storage().nbytes()==448 and all(bridge._storage(o.state.depth_residual_lookup)==bridge._storage(lookup8) for o in owners),'one448B lookup/all28 views')
    selected=[x for x in original if not(x[0]=='k' and tuple(x[1:]) in drops)]
    require(len(selected)==(43 if mode=='Lookup8' else 32) and sum(x[0]=='v' for x in selected)==20,'stage32/V20')
    forbidden={bridge._storage(t) for l in native.layers for t in (l.keys,l.values)}
    for owner in owners:
        for t in owner.named_tensors().values():require(not t.requires_grad and t.grad_fn is None and bridge._storage(t) not in forbidden,'retained native source/graph')
    require(all(sha(p)==x['sha256'] for p,x in metadata['verified_legacy_source_pins'].items()),'legacy source changed')
    require(all(sha(x['path'])==x['sha256'] for x in metadata['fixed_assets'].values()),'original asset changed')
    if mode=='Exchange':require(sha(candidate['high_book_path'])==HIGH_SHA,'high book changed')
    require(sha(__file__)==module_sha and sha(single.__file__)==V1_SHA,'source changed')
    metadata.update(selected_stage_count=len(selected),selected_stages=selected,original_selected_stages=original,returned_owners_unique_tensor_capacity_bytes=actual)
    metadata['round9_precision_exchange']={'schema':'trajkv.round9.precision_exchange.v2','mode':mode,'candidate':candidate,
        'U':0 if mode=='Lookup8' else 3,'D':0 if mode=='Lookup8' else 11,'upgraded_layers':LAYERS if mode=='Exchange' else [],'dropped_k_units':drops,
        'lookup_dtype':'torch.int8','lookup_bytes':448,'old_lookup_and_original_pools_released':True,
        'format_bytes':0 if formats is None else 28,'donor_descriptor_bytes':0 if descriptor is None else 33,
        'extra_high_book_bytes':2048 if high is not None else 0,'high_book_storage_shared_across_all_upgraded_layers':True,'old_low_book_bytes':128,
        'before_unique_bytes':before,'after_unique_bytes':actual,'expected_delta_bytes':delta,'actual_delta_bytes':actual-before,
        'V20_base_parameters_and_residual_values_unchanged':True,'original_ranking_policy_retained_and_charged':True,
        'layer_stats':stats,'source_sha256':module_sha,'numerical_fields_v1_sha256':V1_SHA,'calibration_or_task_gain_claim':False}
    return owners,metadata


def diagnostic_profile(candidate,original_selected):
    p=deepcopy(candidate);drops=[tuple(x) for x in p['dropped_k_units']];units=[tuple(x[1:]) for x in original_selected if x[0]=='k']
    retained=[u for u in units if u[0] in LAYERS and u not in drops]
    if retained:return None
    restore=next(u for u in drops if u[0] in LAYERS);replacement=next(u for u in units if u[0] not in LAYERS and u not in drops)
    drops.remove(restore);drops.append(replacement)
    p['dropped_k_units']=[list(x) for x in drops];p['profile_role']='GPU_nonempty_KRE_fixture_only'
    p['diagnostic_scope']='No model QA or cal optimum claim; donor substitution only to execute retained-KRE branch'
    for key in ('predicted_cal_route_loss','predicted_cal_change_percent','base_cal_route_loss'):p.pop(key,None)
    return p
