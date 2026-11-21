"""Four synthetic R7 controls: two original profiles x Nv256/1792.

Reuses immutable V1 comparison/accounting helpers, never its fixed KeyBank
reference branch. High is instantiated directly from original DepthResidualBank
and uses its original reader. No model, image, dataset, GT or method fitting.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
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

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
SCHEMA='trajkv.r7_research.dual_bridge_synthetic_control.v2'
V1_SHA='e37810a2568ebb21f10ea10488829be0758311985c5dbdeebdd20fa450c5631a'
BRIDGE_SHA='608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad'
CALLER_SHA='3ff896405be5ac91067bdbe8d736c0552e46f835dea90a93c9835ceeb8fd3a91'
LOW='r7_joint_k1_gt_cond';HIGH='r7_budget_k2_cond';NVS=(256,1792)
_BASE=None


def require(value,message):
    if not value:raise ValueError(message)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def helpers():
    global _BASE
    import hashlib
    p=ROOT/'code/r7_bridge_gpu_check_v1.py'
    require(hashlib.sha256(p.read_bytes()).hexdigest()==V1_SHA,'Immutable V1 control changed')
    if _BASE is None:_BASE=importlib.import_module('r7_bridge_gpu_check_v1')
    return _BASE


def validate_metadata(c,profile):
    require(c.get('schema')=='trajkv.r7_research.frozen_matrix500.v2' and c.get('model_id')=='M1', 'Exact M1 V2 smoke asset config required')
    require(c.get('count')==4 and c.get('start',0)==0 and c.get('seed',0)==0,'Only existing first4 smoke supplies assets; no QA is run')
    require(c.get('variant_source',{}).get('name')==profile and profile in (LOW,HIGH),'Wrong profile')
    require(c.get('variant_source',{}).get('sha256')==helpers().PROFILE_SHA,'Original full profile config SHA required')
    require(c.get('position_owner_layout')=='single_shared_visual_position_map','Exact declared singleVP layout required')
    require(len(c.get('variant',{}).get('selected_stages',[]))==224,'Complete original ranking required')
    require(not any(x in c for x in ('refs','references','answers','labels','scores')),'No GT/score input')
    require(c['variant'].get('res_stages_limit')==(43 if profile==LOW else None),'Low capped43; high retains original dynamic budget')


def profile(c,name,bridge):
    b=helpers();validate_metadata(c,name);definition=bridge.profile_definition(name)
    require(c['expected_asset_sha256']==definition['fixed_asset_sha256'],'Original profile-specific assets required')
    require(b.sha(c['variant_source']['path'])==b.PROFILE_SHA,'Original profile source changed')
    original=[v for v in b.load(c['variant_source']['path'])['variants'] if v['name']==name]
    require(len(original)==1,'Exactly one original variant required')
    assets=definition['fixed_asset_sha256'];other=lambda v:{k:x for k,x in v.items() if k not in assets}
    require(other(c['variant'])==other(original[0]),'Numeric profile changed; only fixed asset paths may be rebound')
    for k,h in assets.items():require(b.sha(c['variant'][k])==h,'Profile asset SHA changed: '+k)
    return deepcopy(c['variant']),definition


def run_case(nv,c,variant,definition,bridge,share_positions):
    import torch
    from transformers.cache_utils import DynamicCache
    from m1_protocol import clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from ownership import storage_report,cache_storage,assert_no_full_prefix_alias
    b=helpers();v1=bridge._v1;name=definition['name'];boundary=nv+4
    torch.manual_seed(0);native=DynamicCache()
    for i in range(28):
        k=torch.randn(1,4,boundary,128,device='cuda',dtype=torch.float16)*.15
        v=torch.randn_like(k)*.1;native.update(k,v,i)
    k=v=None;vp=torch.arange(4,boundary,device='cuda',dtype=torch.long)
    fingerprint=lambda:[{'K':b.tensor_identity(x.keys),'V':b.tensor_identity(x.values),'descriptor':id(x)} for x in native.layers]
    original=fingerprint();functions={n:getattr(native,n).__func__ for n in ('update','get_seq_length','get_mask_sizes')}
    forbidden={(str(t.device),t.untyped_storage().data_ptr()) for x in native.layers for t in (x.keys,x.values)}
    refcache=clone_m1_prefix_cache(native)
    stub=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(28)])))
    a2,exp,scripts=(Path(c[k]).resolve() for k in ('legacy_a2','legacy_experiments','legacy_scripts'));kivi=Path(c['kivi_source']).resolve()
    roots=(a2.parent,scripts,kivi.parent);ref=hybrid=None
    extra=bridge._extra_source_pins(None) if name==HIGH else None
    pins=v1._source_pin_paths(a2,exp,scripts,kivi,extra);v1._check_loaded_pinned_modules(pins)
    record={'schema':SCHEMA+'.case','profile':name,'Nv':nv,'seed':0,'status':'RUNNING','model_loaded':False,
            'layers':28,'KV_heads':4,'query_heads':28,'head_dim':128,'queries':4,'numerical_tolerance_used':False}
    try:
        with v1._legacy_paths(a2,exp,scripts):
            module=importlib.import_module('round7_key_bank' if name==LOW else 'round6_depth_bank')
            parent=module.KeyBank if name==LOW else module.DepthResidualBank
            cls=type('_IndependentFrozenR7DualReference',(parent,),{'settings':deepcopy(variant)})
            ref=cls(stub,refcache,vp,'round7_key' if name==LOW else 'round6_depth',scripts,kivi,keep_reference=False)
            sources=v1._legacy_manifest(roots)
        fullpins=b.role_map(sources,a2,exp,scripts,kivi)
        owners,meta=bridge.build_r7_owners(native,vp,variant,legacy_a2=a2,legacy_experiments=exp,
             legacy_scripts=scripts,kivi_source=kivi,expected_asset_sha256=definition['fixed_asset_sha256'],expected_legacy_source_sha256=fullpins)
        require(meta['all_imported_legacy_sources_pinned'] and len(owners)==28,'All actual imported sources and28owners required')
        require(fingerprint()==original and not hasattr(native,'_single_state_append') and
                all(getattr(native,n).__func__ is f for n,f in functions.items()),'Original native source/cache mutated')
        expected_stages=43 if name==LOW else {256:26,1792:44}[nv]
        require(len(ref.selected_residual_stages)==meta['selected_stage_count']==expected_stages and
                ref.selected_residual_stages==meta['selected_stages'],'Original capped/dynamic allocation differs')
        statecmp=[{'layer':i,'fields':{k:b.compare_tensor(getattr(s,k,None),getattr(o.state,k,None)) for k in b.STATE}}
                  for i,(s,o) in enumerate(zip(ref.states,owners))]
        factors=(ref.encoder_factors if name==LOW else
                 {k:torch.tensor(b.load(variant['metadata_factor_path'])[k],device='cuda',dtype=torch.float32)
                  for k in ('key_head_factors','value_head_factors','value_center_offsets')})
        ranking=torch.tensor(list(ref.encoder_ranking_table),device='cuda',dtype=torch.uint8)
        policy=torch.tensor(list(ref.encoder_depth_policy),device='cuda',dtype=torch.uint8)
        constants={'book':b.compare_tensor(ref.book,owners[0].shared['book']),
            'lookup':b.compare_tensor(ref.residual_lookup,owners[0].shared['global_residual_lookup']),
            'ranking':b.compare_tensor(ranking,owners[0].shared['encoder_ranking_bytes']),
            'policy':b.compare_tensor(policy,owners[0].shared['encoder_policy_bytes'])}
        for k,v in factors.items():constants['factor.'+k]=b.compare_tensor(v,owners[0].shared['encoder_factor.'+k])
        account=SimpleNamespace(states=ref.states,book=ref.book,residual_lookup=ref.residual_lookup,encoder_factors=factors)
        core=storage_report(b.candidate_named(owners));refcore=storage_report(b.reference_named(account,ranking,policy))
        parts,refparts=b.byte_components(core),b.byte_components(refcore)
        old=ref.owned();legacy=int(old['complete_visual_owner_bytes'] if name==LOW else
            old['visual_base']+old['visual_residual']+old['selector_metadata']+old['visual_position']+ref.encoder_parameter_bytes)
        record.update(packed_state_comparisons=statecmp,constant_comparisons=constants,
            owner_core={'candidate':core,'reference':refcore,'R7_components':parts,'reference_R7_components':refparts,'legacy_complete_bytes':legacy},
            selected_stages=meta['selected_stages'],selected_stage_count=expected_stages,bridge_metadata=meta)
        require(core['total_bytes']==refcore['total_bytes']==legacy==meta['returned_owners_unique_tensor_capacity_bytes'] and
                parts==refparts,'Physical format owner-core components or unique totals differ')
        require(all(not hasattr(o.state,'visual_k') and not hasattr(o.state,'visual_v') and
                    o.state.text_k.numel()==o.state.text_v.numel()==o.state.text_pos.numel()==0 and o.state.reserve==0 for o in owners),
                'Candidate has full source or old text reserve')
        clone=clone_m1_prefix_cache(native);weak=[weakref.ref(t) for x in clone.layers for t in (x.keys,x.values)]
        hybrid=HybridVisualCache(clone,vp,owners,consume_native=True);clone=None;gc.collect()
        require(all(x() is None for x in weak),'Hybrid input clone retained')
        assert_no_full_prefix_alias(hybrid,forbidden)
        initial=cache_storage(hybrid,vp);require(initial['complete_visual_owner_bytes']==legacy+8*nv,'DoubleVP initial actual charge differs')
        qp=torch.arange(boundary,boundary+4,device='cuda',dtype=torch.long)
        queries=[];outputs=[];readcmp=[];record['reader_comparisons']=readcmp
        for i,o in enumerate(owners):
            tk=torch.randn(1,4,4,128,device='cuda',dtype=torch.float16)*.15;tv=torch.randn_like(tk)*.1
            q=torch.randn(1,28,4,128,device='cuda',dtype=torch.float16)*.2
            refcache.update(tk,tv,i);hybrid.update(tk,tv,i,cache_kwargs={'cache_position':qp});layer=hybrid.layers[i];s=ref.states[i]
            require(torch.equal(s.text_k[:,:,:s.text_length],layer.keys) and torch.equal(s.text_v[:,:,:s.text_length],layer.values)
                    and torch.equal(s.text_pos[:s.text_length],layer.text_positions),'Exact source text/position mismatch')
            expected=ref.read(q,i);actual=o.attend(q,exact_keys=layer.keys,exact_values=layer.values,
                exact_positions=layer.text_positions,query_positions=layer.query_positions,scale=128**-.5)
            readcmp.append({'layer':i,**b.compare_tensor(expected,actual)})
            require(bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()),'Nonfinite reader')
            queries.append(q);outputs.append(actual)
        before=cache_storage(hybrid,vp);layout=share_positions(hybrid);after=cache_storage(hybrid,vp)
        require(before['complete_visual_owner_bytes']-after['complete_visual_owner_bytes']==8*nv and
                after['complete_visual_owner_bytes']==legacy,'SingleVP must release exactly8Nv and retain legacy core')
        layoutcmp=[]
        for i,(o,q,prior) in enumerate(zip(owners,queries,outputs)):
            layer=hybrid.layers[i];actual=o.attend(q,exact_keys=layer.keys,exact_values=layer.values,
                exact_positions=layer.text_positions,query_positions=layer.query_positions,scale=128**-.5)
            layoutcmp.append({'layer':i,**b.compare_tensor(prior,actual)})
            require(o.state.text_k.numel()==o.state.text_v.numel()==o.state.text_pos.numel()==0,'Borrowed text survived read')
        require(fingerprint()==original,'Original native source changed during reader/layout test')
        equal=all(x['bitwise_equal'] for d in statecmp for x in d['fields'].values()) and all(x['bitwise_equal'] for x in constants.values())
        equal=equal and all(x['bitwise_equal'] for x in readcmp+layoutcmp)
        record.update(status='PASS_EXACT_DUAL_PROFILE_CODEC_READER_SINGLEVP' if equal else 'FAILED_EXACT_DIFFERENCES_REPORTED',
            singleVP=layout,singleVP_reader_comparisons=layoutcmp,Hybrid_doubleVP_report=before,Hybrid_singleVP_report=after,
            original_native_cache_unmutated=True,no_fit_mean_correction_or_other_algorithm=True,
            explicit_import_source_pins=fullpins,actual_legacy_sources=sources,
            reference_factor_rank_policy_copies_are_diagnostic_accounting_not_refitting=True,
            diagnostic_reference_storage_not_deployment_savings=True,
            generic_classifier_note='depth_residual_code remains generic coder_metadata; separate R7_components exposes true packed payload')
    except Exception as exc:record.update(status='FAILED_SYNTHETIC_DUAL_CONTROL',error=str(exc),traceback=traceback.format_exc())
    finally:
        if ref is not None:
            try:
                ref.close();require(all(not hasattr(x.self_attn,'_p185_read') for x in stub.model.language_model.layers),'Reference stub cleanup failed')
            except Exception as exc:record.update(status='FAILED_REFERENCE_CLEANUP',cleanup_error=str(exc))
        record['diagnostic_peak_including_reference_copies']=torch.cuda.max_memory_allocated();record['peak_is_not_deployment']=True
        native=refcache=ref=hybrid=None;gc.collect();torch.cuda.empty_cache()
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--low-config',type=Path,required=True);p.add_argument('--high-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-only',action='store_true');a=p.parse_args();b=helpers()
    require(b.sha(ROOT/'code/r7_legacy_bridge_v2.py')==BRIDGE_SHA,'Reviewed bridgeV2 changed')
    require(b.sha(ROOT/'code/r7_frozen_pilot_v2.py')==CALLER_SHA,'Reviewed caller helper changed')
    for name,h in b.CORE.items():require(b.sha(ROOT/name)==h,'Frozen core source changed')
    configs={LOW:load(a.low_config),HIGH:load(a.high_config)}
    for name,c in configs.items():validate_metadata(c,name)
    a.output.mkdir(parents=True,exist_ok=False)
    if a.cpu_only:
        b.write(a.output/'CPU_PREPARATION.json',{'status':'PASS_STATIC_DUAL_PROFILE_SCOPE_NO_GPU_EXECUTED','driver_sha256':b.sha(__file__),
            'v1_immutable_control_sha256':V1_SHA,'bridge_v2_sha256':BRIDGE_SHA,'caller_helper_sha256':CALLER_SHA,
            'cases':[[name,n] for name in (LOW,HIGH) for n in NVS],'actual_GPU_calls':0,'actual_encoder_reader_equivalence':False})
        return 0
    manifest={'schema':SCHEMA+'.manifest','expected_cases':4,'cases':[[name,n] for name in (LOW,HIGH) for n in NVS],
        'source_sha256':{x:b.sha(ROOT/x) for x in ('code/r7_bridge_gpu_check_v2.py','code/r7_bridge_gpu_check_v1.py',
              'code/r7_legacy_bridge_v1.py','code/r7_legacy_bridge_v2.py','code/r7_frozen_pilot_v2.py',*b.CORE)},
        'config_sha256':{str(x):b.sha(x) for x in (a.low_config,a.high_config)},'seed':0,'TF32':False,
        'model_or_images_or_dataset_loaded':False,'GT_read':False,'quality_QA_added':0,'reference_copies_not_deployment':True}
    rows=[];error=None;started=time.time()
    try:
        import torch,transformers
        require(torch.cuda.is_available(),'Actual CUDA required');torch.cuda.set_device(0)
        require(torch.cuda.get_device_capability()==(8,6),'SM86 required')
        torch.set_grad_enabled(False);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        bridge=importlib.import_module('r7_legacy_bridge_v2');caller=importlib.import_module('r7_frozen_pilot_v2')
        manifest.update(torch=torch.__version__,transformers=transformers.__version__,CUDA=torch.version.cuda,
            hardware=torch.cuda.get_device_name(),visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),python=platform.python_version())
        prepared_profiles={name:profile(configs[name],name,bridge) for name in (LOW,HIGH)}
        assets={c['variant_source']['path']:b.PROFILE_SHA for c in configs.values()}
        for name,(variant,definition) in prepared_profiles.items():
            assets.update({variant[k]:h for k,h in definition['fixed_asset_sha256'].items()})
        manifest['fixed_profile_asset_sha256']=assets
        b.write(a.output/'manifest.json',manifest)
        with (a.output/'records.jsonl').open('x',encoding='utf-8',newline='\n') as f:
            for name in (LOW,HIGH):
                variant,definition=prepared_profiles[name]
                for nv in NVS:
                    torch.cuda.reset_peak_memory_stats();row=run_case(nv,configs[name],variant,definition,bridge,caller.share_original_visual_position_map)
                    rows.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                    print(json.dumps({'profile':name,'Nv':nv,'status':row['status']}),flush=True)
        require(all(b.sha(ROOT/x)==h for x,h in manifest['source_sha256'].items()),'New/frozen source changed during control')
        require(all(b.sha(x)==h for x,h in manifest['config_sha256'].items()),'Config changed during control')
        require(all(b.sha(x)==h for x,h in manifest['fixed_profile_asset_sha256'].items()),'Frozen profile/asset changed during control')
        for row in rows:
            for path,item in row.get('actual_legacy_sources',{}).items():require(b.sha(path)==item['sha256'],'Legacy source changed')
    except Exception as exc:error={'error':str(exc),'traceback':traceback.format_exc()}
    finally:
        if not (a.output/'manifest.json').exists():manifest['initialization_error']=error;b.write(a.output/'manifest.json',manifest)
        if not (a.output/'records.jsonl').exists():(a.output/'records.jsonl').open('x').close()
        passed=error is None and len(rows)==4 and all(x['status']=='PASS_EXACT_DUAL_PROFILE_CODEC_READER_SINGLEVP' for x in rows)
        result={'schema':SCHEMA+'.complete','status':'PASS_FOUR_ACTUAL_DUAL_PROFILE_SYNTHETIC_CASES' if passed else 'FAILED_OR_INCOMPLETE_DUAL_SYNTHETIC_CONTROL',
            'expected_cases':4,'recorded_cases':len(rows),'case_statuses':[{k:x[k] for k in ('profile','Nv','status')} for x in rows],
            'root_error':error,'GPU_runtime_attempted':True,'actual_GPU_cases_recorded':len(rows),'quality_QA_added':0,
            'raw_files':{n:b.sha(a.output/n) for n in ('manifest.json','records.jsonl')},'wall_seconds':time.time()-started,
            'no_random_tolerance_no_model_or_task_quality_claim':True}
        b.write(a.output/'complete.json',result);print(json.dumps(result),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
