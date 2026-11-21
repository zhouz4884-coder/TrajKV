"""Two synthetic CUDA controls of the frozen whole-model R7 legacy bridge.

No model, dataset, image, GT, scoring, transport, fitting or new algorithm.
Root runs this explicitly. --cpu-only checks metadata without importing CUDA.
Reference storage and all temporary controls are never deployment savings.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace
import weakref

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
SCHEMA = 'trajkv.r7_research.bridge_synthetic_control.v1'
BRIDGE_SHA = 'b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b'
PROFILE_SHA = 'aba87db8c87bd1ca8ca71685baa7a942c0450c8ecc1302c9ccbe183c67703ef4'
ASSETS = {'direction_book_path':'af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589',
          'metadata_factor_path':'8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea',
          'round7_policy_path':'505dbbbc67d50023ec0f60fb31ab9a31b0a27212fe7c3e8a2e72de0307b3f38f'}
CORE = {'code/m1_protocol.py':'4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427',
        'code/cache_bridge.py':'36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025',
        'code/ownership.py':'8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456'}
STATE = ('direction_code','direction_gain','direction_mean','v_code','v_scale','v_min',
         'depth_residual_code','depth_residual_alpha','depth_residual_lookup','visual_pos')
NV = (256, 768)


def require(condition, message):
    if not condition: raise ValueError(message)
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')


def validate_metadata(config):
    require(config.get('schema')=='trajkv.r7_research.frozen_pilot.v1', 'Exact frozen smoke config required')
    require(config.get('model_id')=='M1' and config.get('count')==4 and config.get('seed',0)==0,
            'Only the existing M1 first4 smoke configuration supplies assets; no QA is executed')
    require(config.get('expected_asset_sha256')==ASSETS, 'Exact three original assets required')
    source=config.get('variant_source',{})
    require(source.get('name')=='r7_joint_k1_gt_cond' and source.get('sha256')==PROFILE_SHA,
            'Exact original gt_cond43 profile source required')
    v=config.get('variant',{})
    require(v.get('method')=='round7_key' and v.get('name')=='r7_joint_k1_gt_cond'
            and v.get('res_stages_limit')==43 and v.get('budget_bits')==1.8
            and len(v.get('selected_stages',[]))==224, 'Complete frozen224 ranking and43 cap required')
    require(not any(k in config for k in ('refs','references','answers','labels','scores')), 'No GT or score inputs')
    for key in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source'):
        require(isinstance(config.get(key),str) and config[key], 'Explicit legacy source root required: '+key)


def profile(config):
    validate_metadata(config)
    path=Path(config['variant_source']['path']); require(sha(path)==PROFILE_SHA, 'Original profile bytes changed')
    original=[v for v in load(path)['variants'] if v['name']=='r7_joint_k1_gt_cond']
    require(len(original)==1, 'Unique original profile required')
    v=deepcopy(config['variant']); numeric=lambda x:{k:z for k,z in x.items() if k not in ASSETS}
    require(numeric(v)==numeric(original[0]), 'Only original asset paths may be rebound')
    for key,digest in ASSETS.items(): require(sha(v[key])==digest, 'Original asset bytes changed: '+key)
    return v


def tensor_identity(tensor):
    body=tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':hashlib.sha256(body).hexdigest()}


def compare_tensor(left, right):
    import torch
    if left is None or right is None:
        return {'bitwise_equal':left is None and right is None, 'left_present':left is not None,'right_present':right is not None}
    a,b=tensor_identity(left),tensor_identity(right)
    same=a==b
    result={'bitwise_equal':same,'reference':a,'candidate':b}
    if left.shape==right.shape:
        finite=bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
        if not finite:
            result.update(finite_reference=bool(torch.isfinite(left).all()),finite_candidate=bool(torch.isfinite(right).all()))
            return result
        delta=(left.double()-right.double()).abs()
        result.update(different_values=int((left!=right).sum()),max_abs=float(delta.max()) if delta.numel() else 0.)
        if not same and delta.numel():
            index=int(delta.reshape(-1).argmax())
            result.update(max_abs_flat_index=index,reference_at_max=float(left.reshape(-1)[index]),
                          candidate_at_max=float(right.reshape(-1)[index]))
    return result


def role_map(sources, a2, experiments, scripts, kivi):
    """Full actual import graph, in the bridge's explicit source-role schema."""
    result={}
    roots=(('a2',a2),('experiments',experiments),('scripts',scripts),('kivi',kivi.parent),('code',a2.parent))
    for text,item in sources.items():
        path=Path(text).resolve()
        if path==kivi: name='kivi_source'
        else:
            name=None
            for group,root in roots:
                if path.is_relative_to(root):name=group+'/'+path.relative_to(root).as_posix();break
            require(name is not None, 'Imported legacy source escaped explicit roots')
        result[name]=item['sha256']
    return result


def candidate_named(owners):
    return {f'layer{i}.'+name:t for i,owner in enumerate(owners) for name,t in owner.named_tensors().items()}


def reference_named(bank, ranking, policy):
    names={'direction_code':'k_code','direction_gain':'k_gain','direction_mean':'k_mean',
           'visual_pos':'positions','depth_residual_lookup':'encoder_constants.lookup_view'}
    result={}
    for i,state in enumerate(bank.states):
        for field in STATE:
            value=getattr(state,field,None)
            if value is not None:result[f'layer{i}.'+names.get(field,field)]=value
    result.update({'encoder_constants.book':bank.book,'encoder_constants.global_residual_lookup':bank.residual_lookup,
                   'encoder_constants.encoder_ranking_bytes':ranking,'encoder_constants.encoder_policy_bytes':policy})
    result.update({'encoder_constants.encoder_factor.'+key:value for key,value in bank.encoder_factors.items()})
    return result


def byte_components(report):
    """R7 algorithm fields; do not redefine the frozen generic classifier."""
    result={'base_packed_payload':0,'residual_packed_payload':0,'gain_mean_Vmetadata_and_alpha':0,
            'positions':0,'constants_lookup_book_factors_ranking_policy':0}
    for name,row in report['tensors'].items():
        charge=row['charged_bytes'];leaf=name.rsplit('.',1)[-1]
        key=('constants_lookup_book_factors_ranking_policy' if 'encoder_constants' in name else
             'positions' if leaf=='positions' else 'residual_packed_payload' if leaf=='depth_residual_code' else
             'base_packed_payload' if leaf in ('k_code','v_code') else 'gain_mean_Vmetadata_and_alpha')
        result[key]+=charge
    require(sum(result.values())==report['total_bytes'], 'R7 component sum differs')
    return result


def run_case(nv, config, variant, bridge):
    import torch
    from transformers.cache_utils import DynamicCache
    from m1_protocol import clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from ownership import storage_report, cache_storage, assert_no_full_prefix_alias
    torch.manual_seed(0)
    native=DynamicCache(); boundary=nv+4
    for layer in range(28):
        k=torch.randn(1,4,boundary,128,device='cuda',dtype=torch.float16)*.15
        v=torch.randn_like(k)*.1
        native.update(k,v,layer)
    k=v=None
    vp=torch.arange(4,boundary,device='cuda',dtype=torch.int64)
    before=[{'K':tensor_identity(x.keys),'V':tensor_identity(x.values),'layer_object':id(x)} for x in native.layers]
    native_method_functions={name:getattr(native,name).__func__ for name in ('update','get_seq_length','get_mask_sizes')}
    forbidden={(str(t.device),t.untyped_storage().data_ptr()) for x in native.layers for t in (x.keys,x.values)}
    refcache=clone_m1_prefix_cache(native); original_ref=None; hybrid=None
    stub=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(28)])))
    a2,experiments,scripts=(Path(config[key]).resolve() for key in ('legacy_a2','legacy_experiments','legacy_scripts'))
    kivi=Path(config['kivi_source']).resolve(); roots=(a2.parent,scripts,kivi.parent)
    pins=bridge._source_pin_paths(a2,experiments,scripts,kivi)
    bridge._check_loaded_pinned_modules(pins)
    record={'schema':SCHEMA+'.case','visual_tokens':nv,'seed':0,'status':'RUNNING','model_loaded':False,
            'geometries':{'layers':28,'KV_heads':4,'query_heads':28,'head_dim':128,'queries':4,'prefix_length':boundary},
            'numerical_tolerance_used':False,'quality_QA_added':0}
    try:
        with bridge._legacy_paths(a2,experiments,scripts):
            legacy=importlib.import_module('round7_key_bank')
            cls=type('_IndependentSyntheticReferenceR7', (legacy.KeyBank,), {'settings':deepcopy(variant)})
            original_ref=cls(stub,refcache,vp,'round7_key',scripts,kivi,keep_reference=False)
            sources=bridge._legacy_manifest(roots)
        full_pins=role_map(sources,a2,experiments,scripts,kivi)
        owners,meta=bridge.build_r7_owners(native,vp,variant,legacy_a2=a2,legacy_experiments=experiments,
             legacy_scripts=scripts,kivi_source=kivi,expected_asset_sha256=ASSETS,expected_legacy_source_sha256=full_pins)
        require(meta['all_imported_legacy_sources_pinned'] is True, 'Full actual source graph must be pinned')
        source_after=[{'K':tensor_identity(x.keys),'V':tensor_identity(x.values),'layer_object':id(x)} for x in native.layers]
        require(source_after==before and not hasattr(native,'_single_state_append') and
                all(getattr(native,name).__func__ is fn for name,fn in native_method_functions.items()),
                'Original native cache values/descriptors/methods were mutated')
        require(len(owners)==28 and len(original_ref.selected_residual_stages)==meta['selected_stage_count']==43,
                'All28 owners and actual43-stage prefix required')
        selected=original_ref.selected_residual_stages==meta['selected_stages']
        stage_counts={axis:sum(x[0]==axis for x in meta['selected_stages']) for axis in ('k','v')}
        require(stage_counts=={'k':23,'v':20}, 'The frozen gt_cond43 profile is23K+20V, not inc43')
        state_diffs=[]
        for i,(ref,owner) in enumerate(zip(original_ref.states,owners)):
            state_diffs.append({'layer':i,'fields':{name:compare_tensor(getattr(ref,name,None),getattr(owner.state,name,None)) for name in STATE}})
            require(not hasattr(owner.state,'visual_k') and not hasattr(owner.state,'visual_v') and
                    owner.state.text_k.numel()==owner.state.text_v.numel()==owner.state.text_pos.numel()==0
                    and owner.state.reserve==0, 'Candidate retains source or old512 text reserve')
        shared=owners[0].shared
        reference_ranking=torch.tensor(list(original_ref.encoder_ranking_table),device='cuda',dtype=torch.uint8)
        reference_policy=torch.tensor(list(original_ref.encoder_depth_policy),device='cuda',dtype=torch.uint8)
        constants={'book':compare_tensor(original_ref.book,shared['book']),
                   'lookup':compare_tensor(original_ref.residual_lookup,shared['global_residual_lookup']),
                   'ranking':compare_tensor(reference_ranking,shared['encoder_ranking_bytes']),
                   'policy':compare_tensor(reference_policy,shared['encoder_policy_bytes'])}
        for name,value in original_ref.encoder_factors.items():
            constants['factor.'+name]=compare_tensor(value,shared['encoder_factor.'+name])
        record.update(packed_state_comparisons=state_diffs,shared_constant_comparisons=constants,
                      selected_stages_equal=selected,selected_stages=meta['selected_stages'])
        core=storage_report(candidate_named(owners)); parts=byte_components(core)
        refcore=storage_report(reference_named(original_ref,reference_ranking,reference_policy))
        refparts=byte_components(refcore)
        legacy_bytes=int(original_ref.owned()['complete_visual_owner_bytes'])
        record['owner_core']={'physical_total_bytes':core['total_bytes'],'reference_legacy_complete_bytes':legacy_bytes,
             'R7_algorithm_components':parts,'reference_components':refparts,'storage_table':core,'reference_storage_table':refcore,
             'reference_ranking_policy_tensor_copies':'Only this diagnostic materializes old serialized bytes; not deployment savings'}
        require(core['total_bytes']==refcore['total_bytes']==legacy_bytes==meta['returned_owners_unique_tensor_capacity_bytes']
                and parts==refparts,
                'Candidate physical owner-core and unchanged reference complete bytes differ')
        hybrid_prefix=clone_m1_prefix_cache(native)
        weak=[weakref.ref(t) for x in hybrid_prefix.layers for t in (x.keys,x.values)]
        hybrid=HybridVisualCache(hybrid_prefix,vp,owners,consume_native=True); hybrid_prefix=None
        gc.collect(); require(all(x() is None for x in weak), 'Hybrid retained native prefix tensors')
        assert_no_full_prefix_alias(hybrid,forbidden)
        before_owner=cache_storage(hybrid,vp)
        extra=hybrid.visual_positions.untyped_storage().nbytes()
        require(before_owner['complete_visual_owner_bytes']==core['total_bytes']+extra and extra==8*nv,
                'Extra Hybrid originalVP must be separately and fully charged')
        reader_diffs=[];record['reader_comparisons']=reader_diffs
        qp=torch.arange(boundary,boundary+4,device='cuda',dtype=torch.int64)
        for i,owner in enumerate(owners):
            text_k=torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.15
            text_v=torch.randn_like(text_k)*.1
            q=torch.randn(1,28,4,128,device='cuda',dtype=torch.float16)*.2
            refcache.update(text_k,text_v,i); hybrid.update(text_k,text_v,i,cache_kwargs={'cache_position':qp})
            layer=hybrid.layers[i];state=original_ref.states[i]
            require(torch.equal(state.text_pos[:state.text_length],layer.text_positions) and
                    torch.equal(state.text_k[:,:,:state.text_length],layer.keys) and
                    torch.equal(state.text_v[:,:,:state.text_length],layer.values), 'Same prefix+four text K/V required')
            expected=original_ref.read(q,i)
            actual=owner.attend(q,exact_keys=layer.keys,exact_values=layer.values,
                               exact_positions=layer.text_positions,query_positions=layer.query_positions,scale=128**-.5)
            reader_diffs.append({'layer':i,**compare_tensor(expected,actual)})
            require(bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()), 'Nonfinite synthetic reader output')
            require(owner.state.text_k.numel()==owner.state.text_v.numel()==owner.state.text_pos.numel()==0,
                    'Borrowed exact text survived attend()')
        after_owner=cache_storage(hybrid,vp)
        require(after_owner['complete_visual_owner_bytes']==before_owner['complete_visual_owner_bytes'], 'Visual owner changed during read')
        require([{'K':tensor_identity(x.keys),'V':tensor_identity(x.values),'layer_object':id(x)} for x in native.layers]==before,
                'Original synthetic native prefix changed during reader controls')
        all_states=all(d['bitwise_equal'] for x in state_diffs for d in x['fields'].values())
        all_constants=all(d['bitwise_equal'] for d in constants.values())
        all_readers=all(d['bitwise_equal'] for d in reader_diffs)
        record.update(status='PASS_EXACT_SYNTHETIC_CODEC_READER_OWNER' if all_states and all_constants and all_readers and selected else
                            'FAILED_EXACT_EQUIVALENCE_DIFFERENCES_REPORTED',packed_state_comparisons=state_diffs,
            shared_constant_comparisons=constants,reader_comparisons=reader_diffs,selected_stages_equal=selected,
            selected_stage_count=43,selected_stages=meta['selected_stages'],original_native_prefix_untouched=True,
            selected_stage_counts=stage_counts,
            candidate_has_no_full_visual_or_old_text_reserve=True,bridge_metadata=meta,
            owner_core={'physical_total_bytes':core['total_bytes'],'reference_legacy_complete_bytes':legacy_bytes,
                        'R7_algorithm_components':parts,'reference_components':refparts,'storage_table':core,
                        'reference_storage_table':refcore,
                        'reference_ranking_policy_tensor_copies':'Only this diagnostic materializes old serialized bytes to verify identical capacity; not deployment savings'},
            integrated_Hybrid={'extra_originalVP_bytes':extra,'before':before_owner,'after':after_owner,
                 'generic_payload_classifier_note':'Frozen generic coder_metadata includes depth_residual_code; use separate R7_algorithm_components for payload interpretation.'},
            actual_legacy_sources=sources,explicit_complete_source_pin_map=full_pins,
            diagnostic_reference_storage_is_not_deployment_savings=True)
    except Exception as exc:
        record.update(status='FAILED_SYNTHETIC_CONTROL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        if original_ref is not None:
            try:
                original_ref.close()
                require(all(not hasattr(x.self_attn,'_p185_read') for x in stub.model.language_model.layers), 'Reference stub hooks survived close')
            except Exception as exc:
                record.update(status='FAILED_REFERENCE_CLEANUP',cleanup_error=str(exc))
        record['CUDA_peak_allocated_bytes_including_all_reference_and_diagnostic_copies']=torch.cuda.max_memory_allocated()
        record['peak_is_not_deployment_memory']=True
        native=refcache=original_ref=hybrid=None
        gc.collect();torch.cuda.empty_cache()
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu-only',action='store_true');args=parser.parse_args()
    config=load(args.config);validate_metadata(config)
    require(sha(ROOT/'code/r7_legacy_bridge_v1.py')==BRIDGE_SHA, 'Reviewed bridge source changed')
    require(all(sha(ROOT/p)==h for p,h in CORE.items()), 'Frozen cache/helper/ownership source changed')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.cpu_only:
        write(args.output/'CPU_PREPARATION.json',{'status':'PASS_STATIC_CONFIG_SOURCE_SCOPE_ONLY_NO_GPU_EXECUTED',
              'driver_sha256':sha(__file__),'bridge_sha256':BRIDGE_SHA,'config_sha256':sha(args.config),
              'declared_synthetic_Nv':list(NV),'layers':28,'queries':4,'actual_GPU_calls':0,
              'actual_legacy_encoder_or_reader_executed':False,'runtime_equivalence_established':False})
        return 0
    manifest={'schema':SCHEMA+'.manifest','config_sha256':sha(args.config),'config_asset_source_only':True,
        'source_sha256':{p:sha(ROOT/p) for p in ('code/r7_bridge_gpu_check_v1.py','code/r7_legacy_bridge_v1.py',*CORE)},
        'expected_cases':2,'visual_Nv':list(NV),'seed':0,'TF32':False,'references_read':False,
        'model_or_images_or_dataset_loaded':False,'quality_QA_added':0,'test_scope':'synthetic codec/reader/owner equivalence, not model quality',
        'diagnostic_reference_storage_excluded_from_deployment_claims':True}
    rows=[];error=None;started=time.time()
    try:
        import torch
        import transformers
        require(torch.cuda.is_available(), 'Actual CUDA required')
        torch.cuda.set_device(0);require(torch.cuda.get_device_capability()==(8,6), 'SM86 required')
        torch.set_grad_enabled(False);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import r7_legacy_bridge_v1 as bridge
        variant=profile(config)
        manifest.update(torch=torch.__version__,transformers=transformers.__version__,CUDA=torch.version.cuda,
            hardware=torch.cuda.get_device_name(),visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),python=platform.python_version(),
            variant_sha256=canonical(variant),fixed_asset_sha256=ASSETS)
        write(args.output/'manifest.json',manifest)
        with (args.output/'records.jsonl').open('x',encoding='utf-8',newline='\n') as f:
            for nv in NV:
                torch.cuda.reset_peak_memory_stats();row=run_case(nv,config,variant,bridge);rows.append(row)
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                print(json.dumps({'Nv':nv,'status':row['status']}),flush=True)
        require(all(sha(ROOT/p)==h for p,h in manifest['source_sha256'].items()), 'New/frozen source changed during control')
        require(sha(args.config)==manifest['config_sha256'], 'Config changed during control')
        for key,h in ASSETS.items():require(sha(variant[key])==h, 'Asset changed during control')
        for row in rows:
            for path,item in row.get('actual_legacy_sources',{}).items():require(sha(path)==item['sha256'], 'Legacy source changed during control')
    except Exception as exc:
        error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
    finally:
        if not (args.output/'manifest.json').exists():manifest['initialization_error']=error;write(args.output/'manifest.json',manifest)
        if not (args.output/'records.jsonl').exists():(args.output/'records.jsonl').open('x').close()
        passed=error is None and len(rows)==2 and all(x['status']=='PASS_EXACT_SYNTHETIC_CODEC_READER_OWNER' for x in rows)
        complete={'schema':SCHEMA+'.complete','status':'PASS_TWO_ACTUAL_SYNTHETIC_CODEC_READER_OWNER_CASES' if passed else 'FAILED_OR_INCOMPLETE_SYNTHETIC_CONTROL',
            'expected_cases':2,'recorded_cases':len(rows),'case_statuses':[{ 'Nv':x['visual_tokens'],'status':x['status']} for x in rows],
            'GPU_runtime_attempted':True,'actual_GPU_control':bool(rows),'actual_GPU_cases_recorded':len(rows),
            'model_loaded':False,'quality_QA_added':0,'root_error':error,
            'raw_files':{n:sha(args.output/n) for n in ('manifest.json','records.jsonl')},'wall_seconds':time.time()-started,
            'bitwise_gate_no_random_tolerance':True,'no_TrajKV_quality_or_speed_claim':True}
        write(args.output/'complete.json',complete);print(json.dumps(complete),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
