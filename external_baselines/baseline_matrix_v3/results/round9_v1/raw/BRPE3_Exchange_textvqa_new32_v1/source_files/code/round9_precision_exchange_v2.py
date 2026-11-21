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


def cpu_check(output,legacy_a2,profile_path):
    from unittest.mock import patch
    import types
    from types import SimpleNamespace
    a2=Path(legacy_a2);root=Path(single.__file__).resolve().parents[1]
    cfg=load(root/'configs/r7_research/M1_R7gt43_docvqa_smoke4_v1.json');selected=cfg['variant']['selected_stages'][:43]
    p=deepcopy(load(profile_path));validate_profile(p)
    high_path=root/'configs/r7_research/frozen_assets_v2/gaussian_vq_g4_256.json';p['high_book_path']=str(high_path)
    low=torch.tensor(load(root/'configs/r7_research/frozen_assets_v2/gaussian_vq_g4_16.json')['centers'],dtype=torch.float16)
    torch.manual_seed(0);n=256;vp=torch.arange(4,n+4)
    native=SimpleNamespace(layers=[SimpleNamespace(keys=torch.randn(1,4,n+4,128).half()*.1,values=torch.randn(1,4,n+4,128).half()*.1) for _ in range(28)])
    def fake_original(*args,**kwargs):
        look=torch.full((28,2,4,2),-1,dtype=torch.int32);shared={'book':low,'global_residual_lookup':look,
            'encoder_ranking_bytes':torch.zeros(448,dtype=torch.uint8),'encoder_policy_bytes':torch.zeros(5,dtype=torch.uint8)}
        for name in ('key_head_factors','value_head_factors','value_center_offsets'):shared['encoder_factor.'+name]=torch.ones(28,4)
        owners=[]
        for i in range(28):
            rows=[x for x in selected if x[1]==i]
            state=SimpleNamespace(direction_code=torch.zeros(1,4,n,16,dtype=torch.uint8),direction_gain=torch.ones(1,4,n,1,dtype=torch.float16),direction_mean=torch.zeros(1,4,1,128,dtype=torch.float16),
                v_code=torch.zeros(1,4,128,n//32,dtype=torch.int32),v_scale=torch.ones(1,4,128,1,dtype=torch.float16),v_min=torch.zeros(1,4,128,1,dtype=torch.float16),visual_pos=vp,depth_residual_lookup=look[i])
            if rows:
                state.depth_residual_code=torch.zeros(len(rows),n,4,dtype=torch.int32);state.depth_residual_alpha=torch.zeros(len(rows),n,dtype=torch.float16)
                for j,(axis,_,head,stage) in enumerate(rows):look[i,0 if axis=='k' else 1,head,stage]=j
            owners.append(bridge.R7LegacyOwner(i,state,shared,None,n+4))
        directory=root/'configs/r7_research/frozen_assets_v2'
        names={'direction_book_path':'gaussian_vq_g4_16.json','metadata_factor_path':'factors_for_book16_inherited.json','round7_policy_path':'policy_k1_cond.json'}
        assets={key:{'path':str(directory/names[key]),'sha256':digest} for key,digest in bridge._DEFAULT_ASSETS.items()}
        return owners,{'variant':{'name':'r7_joint_k1_gt_cond'},'selected_stage_count':43,'selected_stages':selected,'fixed_assets':assets,'verified_legacy_source_pins':{},'legacy_sources':{}}
    stub=types.ModuleType('fast_wht');stub.wht=single._cpu_wht
    diag=diagnostic_profile(p,selected);require(diag is not None,'actual U3D11 expected to need nonempty-KRE fixture')
    cases=[]
    with bridge._legacy_paths(a2,a2.parent/'experiments',a2):
        for label,mode,candidate in [('actual_profile',m,p) for m in MODES]+[('diagnostic_nonempty_KRE','Exchange',diag)]:
            with patch.object(bridge,'build_r7_owners',side_effect=fake_original),patch.dict(sys.modules,{'fast_wht':stub}):
                owners,meta=build_round9_precision_exchange_owners(native,vp,{},mode=mode,candidate=candidate,
                    legacy_a2=str(a2),legacy_experiments=str(a2.parent/'experiments'),legacy_scripts=str(a2))
            proof=meta['round9_precision_exchange'];recoded=sum(x['retained_K_stages_reencoded'] for x in proof['layer_stats'])
            if label=='diagnostic_nonempty_KRE':require(recoded>0,'nonempty KRE branch not executed')
            cases.append({'scope':label,'mode':mode,'actual_delta_bytes':proof['actual_delta_bytes'],'stage_count':meta['selected_stage_count'],
                'actual_tensor_storage_geometry':True,'retained_K_stages_reencoded':recoded,'old_lookup_pool_released':True})
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    result={'status':'PASS_CPU_U3D11_FAKE_OWNER_STORAGE_AND_SOURCE_CODEC_BRANCH_ONLY','cases':cases,
        'source_sha256':sha(__file__),'numerical_v1_sha256':V1_SHA,'profile_sha256':sha(profile_path),
        'actual_GPU_calls':0,'production_whole_bank_GPU_positive':False,'fixture_not_model_QA':True}
    write(out/'CPU_RECEIPT.json',result);print(json.dumps(result));return result


@torch.no_grad()
def gpu_check(config_path,profile_path,output):
    import r7_bridge_gpu_check_v1 as ref
    require(sha(ref.__file__)==single.REFERENCE_SHA,'frozen numeric reference helper')
    cfg=load(config_path);ref.validate_metadata(cfg);variant=ref.profile(cfg);p=load(profile_path);validate_profile(p)
    require(sha(profile_path)==PROFILE_SHA,'exact cal-only selected profile')
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6),'actual SM86 CUDA required')
    from transformers.cache_utils import DynamicCache
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    pins={str(Path(__file__).resolve()):sha(__file__),str(Path(single.__file__).resolve()):V1_SHA,str(config_path):sha(config_path),str(profile_path):PROFILE_SHA,p['high_book_path']:HIGH_SHA}
    write(out/'manifest.json',{'schema':'trajkv.round9.U3D11_synthetic.v1','sources':pins,'Nv':[256,1792],
        'model_images_GT_loaded':False,'quality_QA_added':0,'diagnostic_references_are_not_deployment_storage_savings':True})
    kwargs={name:cfg[name] for name in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source')};kwargs['expected_asset_sha256']=ref.ASSETS
    records=[];nonempty_observed=0
    for n in (256,1792):
        torch.manual_seed(0);native=DynamicCache()
        for i in range(28):
            k=torch.randn(1,4,n+4,128,device='cuda',dtype=torch.float16)*.15;v=torch.randn_like(k)*.1;native.update(k,v,i)
        k=v=None;vp=torch.arange(4,n+4,device='cuda',dtype=torch.long)
        oldowners,base=bridge.build_r7_owners(native,vp,variant,**kwargs);b0=_unique(oldowners)
        native_hash=[(ref.tensor_identity(x.keys),ref.tensor_identity(x.values)) for x in native.layers]
        diag=diagnostic_profile(p,base['selected_stages']);scopes=[('actual_cal_profile',mode,p) for mode in MODES]
        if diag is not None:
            write(out/('DIAGNOSTIC_NONEMPTY_KRE_N'+str(n)+'.json'),diag);scopes.append(('diagnostic_nonempty_KRE_not_task','Exchange',diag))
        high=torch.tensor(load(p['high_book_path'])['centers'],device='cuda',dtype=torch.float16)
        codec=importlib.import_module('round7_key_codec');wht=importlib.import_module('fast_wht').wht;modes=[]
        for scope,mode,candidate in scopes:
            owners,meta=build_round9_precision_exchange_owners(native,vp,variant,mode=mode,candidate=candidate,**kwargs)
            delta=-1344 if mode=='Lookup8' else -198*n-1283 if mode=='DropD' else 765-6*n
            require(_unique(owners)-b0==delta,'actual complete bytes')
            checks=[];qp=torch.arange(n+4,n+8,device='cuda');tp=torch.cat((torch.arange(4,device='cuda'),qp))
            for i,(old,owner) in enumerate(zip(oldowners,owners)):
                up=mode=='Exchange' and i in LAYERS
                rot=wht(native.layers[i].keys.index_select(2,vp).float()-old.state.direction_mean.float()) if up else None
                local_drops=[] if mode=='Lookup8' else [x for x in candidate['dropped_k_units'] if x[0]==i]
                f,book,_=precision_exchange_fields(vars(old.state),rot,old.shared['book'],high,old.shared['encoder_factor.key_head_factors'][i],
                    layer_idx=i,upgraded_layer=i,dropped_k_units=local_drops,upgrade=up,compact=False,
                    encode_key=codec.encode_key,decode_key=codec.decode_key,encode_sign_depth=codec.encode_sign_depth)
                state=copy(old.state)
                for name,value in f.items():setattr(state,name,value)
                reference=bridge.R7LegacyOwner(i,state,{**old.shared,'book':book},old._reader,n+4)
                require(torch.equal(decode_exchange_key_fields(f,book),decode_exchange_key_fields(vars(owner.state),book)),'compact field decode')
                for name in ('v_code','v_scale','v_min','direction_mean'):require(torch.equal(getattr(old.state,name),getattr(owner.state,name)),'fixed V0/mean')
                if not up:require(torch.equal(old.state.direction_code,owner.state.direction_code) and torch.equal(old.state.direction_gain,owner.state.direction_gain),'nonupgraded K base')
                for h in range(4):
                    for stage in range(2):
                        a=int(old.state.depth_residual_lookup[1,h,stage]);b=int(owner.state.depth_residual_lookup[1,h,stage]);require((a>=0)==(b>=0),'V20 coverage')
                        if a>=0:require(torch.equal(old.state.depth_residual_code[a],owner.state.depth_residual_code[b]) and torch.equal(old.state.depth_residual_alpha[a],owner.state.depth_residual_alpha[b]),'V residual values')
                q=torch.randn(1,28,4,128,device='cuda',dtype=torch.float16)*.2
                tk=torch.cat((native.layers[i].keys[:,:,:4],torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.15),2)
                tv=torch.cat((native.layers[i].values[:,:,:4],torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.1),2)
                args=dict(exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,scale=128**-.5)
                left,right=reference.attend(q,**args),owner.attend(q,**args)
                require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()) and torch.equal(left,right),'actual int8/compact reader not bitwise versus int32 noncompact reference')
                checks.append({'layer':i,'field_and_reader_bitwise_equal':True})
            recoded=sum(x['retained_K_stages_reencoded'] for x in meta['round9_precision_exchange']['layer_stats'])
            if scope.startswith('diagnostic') or diag is None and mode=='Exchange':require(recoded>0,'nonempty retained KRE GPU branch mandatory');nonempty_observed+=recoded
            modes.append({'scope':scope,'mode':mode,'status':'PASS_FIELDS_READER_ACTUAL_BYTES','actual_delta_bytes':_unique(owners)-b0,
                'expected_delta_bytes':delta,'retained_K_stages_reencoded':recoded,'metadata':meta,'all28_checks':checks,'quality_QA_added':0})
        require(native_hash==[(ref.tensor_identity(x.keys),ref.tensor_identity(x.values)) for x in native.layers],'native source mutated')
        records.append({'Nv':n,'status':'PASS_FIELDS_READER_ACTUAL_BYTES','scopes':modes})
    require(nonempty_observed>0 and all(sha(path)==digest for path,digest in pins.items()),'missing KRE branch or changed source/asset')
    with (out/'records.jsonl').open('x',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    result={'status':'PASS_ALL2_U3D11_FIELDS_READER_BYTES_AND_NONEMPTY_KRE_BRANCH','cases':2,'records_sha256':sha(out/'records.jsonl'),
        'nonempty_KRE_GPU_stages_observed':nonempty_observed,'quality_QA_added':0,'diagnostic_fixture_not_task_candidate':True}
    write(out/'complete.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--cpu-check',action='store_true');mode.add_argument('--gpu-check',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--legacy-a2',type=Path,default=Path('${FROZEN_TRAJKV_ROOT}/code/a2'));a=parser.parse_args()
    if a.cpu_check:cpu_check(a.output,a.legacy_a2,a.candidate)
    else:
        require(a.config is not None,'frozen low smoke4 asset config required');gpu_check(a.config,a.candidate,a.output)
