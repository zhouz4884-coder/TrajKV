"""Frozen Round13 candidate on already-observed official fixed500 inputs.

The Round10 complete native generation driver is byte-frozen. Round13 factory,
native tile validation, original/sparse position sharing and raw margin hooks
are reused unchanged. This module adds an explicit benchmark admission role,
INPUT-only old-baseline identity checks, and immutable selection binding.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import round10_runner_v1 as frozen
import round10_inputs_v1 as old_inputs
import round13_runner_v1 as parent
import round13_fixed500_inputs_v1 as data

SCHEMA='trajkv.round13.fixed500_comparison.raw.v1'
ROLE='round13_observed_fixed500_comparison'
MODES=parent.MODES
BASE_RUNNER_SHA=parent.BASE_RUNNER_SHA
RUNTIME_FILES=tuple(dict.fromkeys((*parent.RUNTIME_FILES,
    'round13_fixed500_runner_v1.py','round13_fixed500_inputs_v1.py')))
_WRITE_BASE=frozen.write_new
_CONFIG=None
_IDENTITY=None
require,sha=data.require,data.sha


def runtime_source_manifest(config=None):
    require(sha(ROOT/'code/round10_runner_v1.py')==BASE_RUNNER_SHA,
            'byte-frozen complete native driver changed')
    return {'code/'+name:sha(ROOT/'code'/name) for name in RUNTIME_FILES}


def selected_algorithm(config,identity):
    path,digest=config.get('selected_algorithm_path'),config.get('selected_algorithm_sha256')
    require(isinstance(path,str) and isinstance(digest,str) and len(digest)==64 and sha(path)==digest,
            'locked selected algorithm path/SHA required')
    spec=data.read(path)
    require(spec.get('schema')=='trajkv.round13.selected_P_algorithm.v1',
            'explicit newly frozen P algorithm schema required')
    require(spec==identity['algorithm_selection'],'admission/runner selected algorithm disagreement')
    receipt=identity['selection_receipt']
    require(any(x.get('sha256')==digest for x in receipt.get('method_asset_locks',[])),
            'algorithm asset was not locked by comparison selection')
    require(spec.get('name')==config['selected_method_name'],'selected method identity changed')
    normalize=lambda value:{k:v for k,v in value.items() if not k.endswith('_path')}
    require(normalize(spec['candidate'])==normalize(config['candidate']),
            'frozen encoder/selector numeric candidate changed')
    require(spec['variant_profile']==config['variant_source']['name'] and
            spec['variant_source_sha256']==config['variant_source']['sha256'],
            'selected base lineage changed')
    actual=runtime_source_manifest()
    for group in ('algorithm_source_hashes','base_runtime_source_hashes'):
        require(isinstance(spec.get(group),dict) and bool(spec[group]),
                'explicit selected source group required: '+group)
        for name,expected in spec[group].items():
            require(actual.get(name)==expected,'selected numerical/runtime source changed: '+name)
    return spec


def candidate_assets(config):
    detail,locks=parent.candidate_assets(config)
    locks=dict(locks)
    locks[config['selected_algorithm_path']]=config['selected_algorithm_sha256']
    locks[config['selection_receipt_path']]=config['selection_receipt_sha256']
    return {'schema':'trajkv.round13.fixed500_candidate_assets.v1',**detail,
        'comparison_role':ROLE,'previously_observed_fixed500':True,
        'independent_holdout_claimed':False},locks


def validate_config(config):
    global _CONFIG,_IDENTITY
    require(config.get('schema')==SCHEMA,'distinct Round13 fixed500 runner schema required')
    require(getattr(data,'ROLE',None)==ROLE and getattr(data,'RUNNER_SCHEMA',None)==SCHEMA,
            'input/runner role or schema disagree')
    identity=data.validate_config_admission(config)
    data.validate_execution_selection(config,identity)
    require(config.get('method',{}).get('family')=='trajkv_round13_fixed500_comparison'
            and config['method'].get('information_condition')=='query_unknown',
            'explicit Round13 fixed500 method family/query_unknown required')
    require(config.get('position_owner_layout')=='single_shared_visual_position_map',
            'original position sharing contract changed')
    source=config.get('variant_source',{})
    require(source.get('name')=='r7_joint_k1_gt_cond'
            and source.get('sha256')==frozen.ORIGINAL_CONFIG_SHA,'original low43 source required')
    require(config.get('expected_asset_sha256')==frozen.PROFILES[source['name']]['assets'],
            'original book16/f0/policy identity changed')
    require(config.get('candidate',{}).get('mode') in MODES,'unknown deployable Round13 mode')
    require(config['candidate'].get('technical_only',False) is False,
            'technical variants cannot enter fixed500 quality')
    for key in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source',
                'model_path','model_source_lock','scope'):
        require(isinstance(config.get(key),str) and bool(config[key]),'explicit field required: '+key)
    selected_algorithm(config,identity)
    candidate_assets(config)
    frozen.frozen_variant(config)
    require(all(sha(ROOT/path)==digest for path,digest in frozen.CORE_SOURCES.items()),
            'frozen native model/cache/ownership source changed')
    require(config.get('expected_source_hashes')==runtime_source_manifest(),
            'exact new comparison runtime graph differs')
    _CONFIG,_IDENTITY=config,identity
    return config


def load_runner_inputs(path,manifest_path=None,expected_sha256=None,expected_role=ROLE):
    return data.load_inputs(path,manifest_path,expected_sha256,expected_role)


def build_candidate_owners(cache,vp,variant,config,assets):
    # parent.prepare hook observes real tile geometry before this factory.
    return parent.build_candidate_owners(cache,vp,variant,config,assets)


def physical_budget_guard(mode,n,complete_bytes,metadata,base_profile):
    from round13_exception_bank_v1 import r_budget
    require(mode in MODES and base_profile=='r7_joint_k1_gt_cond','mode/lineage changed')
    stages=[list(x) for x in metadata.get('selected_stages',[])]
    expected=[list(x) if len(x)==4 else [*x,0] for x in _CONFIG['variant']['selected_stages'][:43]]
    require(len(expected)==43 and len({tuple(x) for x in expected})==43,
            'frozen exact low43 ordered stage identity missing')
    if mode in ('P_best','P_random','CAL_AFF'):
        require(stages==expected,'original concrete K23/V20 coverage changed')
    require(metadata.get('selected_stage_count')==len(stages) and
            metadata.get('returned_owners_unique_tensor_capacity_bytes')==complete_bytes,
            'factory and Hybrid actual unique owner disagree')
    require(metadata.get('round13_router',{}).get('mode')==mode,'actual router mode differs')
    low=4590*n+89733
    cap=low if mode in ('P_best','P_random') else 6382*n+89745 if mode=='CAL_AFF' else r_budget(n)
    require(0<complete_bytes<=cap,f'actual owner {complete_bytes} exceeds cap {cap}')
    if mode in ('P_best','P_random'):
        detail=metadata['round13_pruned']
        require(detail['actual_owner_bytes']==complete_bytes,'actual P owner proof differs')
    elif mode=='EX_best':
        require(metadata['round13_exception']['original_R_complete_owner_bytes']==cap,'EX R cap differs')
    elif mode=='TOK_best':
        detail=metadata['round13_precision']
        require(detail['R_cap_bytes']==cap and detail['V_fields_frozen_to_request_R'] is True
                and all(x[0]=='v' for x in stages),'TOK same-request R-V contract differs')
    return {'schema':'trajkv.round13.fixed500_physical_budget.v1','mode':mode,'N':n,
        'actual_complete_visual_bytes':complete_bytes,'strict_cap_bytes':cap,
        'strict_cap_satisfied':True,'original_low43_bytes':low,'original_R_bytes':r_budget(n),
        'selected_stages':stages,'selected_stage_count':len(stages),
        'original_visual_rows_denominator':n,'actual_unique_storage_not_nominal_bits':True,
        'comparison_role':ROLE,'previously_observed_fixed500':True,
        'timing_is_instrumented_quality_run_not_controlled_speed':True}


def write_new(path,value):
    path=Path(path)
    if path.name=='manifest.json':
        value=dict(value)
        value.update(runner_schema=SCHEMA,comparison_role=ROLE,
            phase_is_confirmatory=False,previously_observed_fixed500=True,
            independent_holdout_claimed=False,prediction_cache_used=False,
            selected_method_name=value['config']['selected_method_name'],
            selection_receipt_sha256=value['config']['selection_receipt_sha256'],
            selected_algorithm_sha256=value['config']['selected_algorithm_sha256'],
            runtime_process={'PID':os.getpid(),'parent_PID':os.getppid(),'source_snapshot_root':str(ROOT)},
            source_driver_contract='Round13 factories/map/margins over byte-frozen complete Round10 native M1 driver',
            native_binding_scope='all500 source/image/question/prompt/token/position/model metadata checked across old FP16/R7; INPUT-only worker sidecar',
            frozen_base_lineage_numeric_fields_unchanged=True,parameter_fit_in_runner=False,
            round13_fixed500_candidate_assets=value.pop('round10_candidate_assets',None),
            canonical_SourceRun_admission_claimed=False)
    elif path.parent.name=='records':
        # Mutate the same object subsequently appended to records.jsonl.
        value.update(round13_fixed500_schema=SCHEMA,comparison_role=ROLE,
            previously_observed_fixed500=True,independent_holdout_claimed=False)
        if 'round10_budget_vs_original_R7_low' in value:
            value['round13_physical_budget']=value.pop('round10_budget_vs_original_R7_low')
        if value.get('status')=='PASS':
            proof=data.validate_native_record(value,_IDENTITY['native_bindings'])
            value['round13_fixed500_native_input_binding']=proof
            trace=list(parent._LAST_GENERATION_TRACE)
            complete_trace=len(trace)==value['output_tokens']
            value['round13_free_generation_trace']={'scope':'actual raw pre-HF-processor last-position logits on every generation forward',
                'step_count':len(trace),'steps':trace,'all_output_steps_observed':complete_trace,
                'used_for_encoder_or_decoding_decisions':False}
            if not proof['matched'] or not complete_trace:
                value.update(status='INPUT_MISMATCH',error='Fixed500 native metadata or complete margin trace mismatch',
                    untrusted_output_token_ids=value.get('output_token_ids'),
                    untrusted_raw_prediction=value.get('raw_prediction'),
                    output_token_ids=None,raw_prediction=None)
    elif path.name=='complete.json':
        value={**value,'round13_fixed500_schema':SCHEMA,'comparison_role':ROLE,
            'phase_is_confirmatory':False,'previously_observed_fixed500':True,
            'independent_holdout_claimed':False,'canonical_SourceRun_admission_claimed':False,
            'new_free_generation_answer_events':value['recorded_QA'],'parameter_fit_in_runner':False,
            'old_prediction_cache_used_by_new_candidate':False}
    return _WRITE_BASE(path,value)


@contextmanager
def driver_overrides():
    # Keep parent's native tile capture, sparse original-map sharing, and raw
    # generation margin hook. Override only benchmark admission/reporting.
    with parent.driver_overrides():
        changes={'SCHEMA':SCHEMA,'MODES':MODES,'candidate_assets':candidate_assets,
            'validate_config':validate_config,'build_candidate_owners':build_candidate_owners,
            'round10_budget_guard':physical_budget_guard,'runtime_source_manifest':runtime_source_manifest,
            'write_new':write_new}
        saved={key:getattr(frozen,key) for key in changes}
        loader=old_inputs.load_round10_inputs
        try:
            for key,value in changes.items():setattr(frozen,key,value)
            old_inputs.load_round10_inputs=load_runner_inputs
            yield
        finally:
            old_inputs.load_round10_inputs=loader
            for key,value in saved.items():setattr(frozen,key,value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--validate-config',action='store_true')
    args=p.parse_args()
    validate_config(data.read(args.config))
    if args.validate_config:
        print(json.dumps({'status':'PASS_ROUND13_OBSERVED_FIXED500_CONFIG_ONLY',
            'role':ROLE,'dataset':_IDENTITY['dataset'],'slice_count':len(_IDENTITY['rows']),
            'full_cohort':500,'model_or_GPU_calls':0,'independent_holdout_claimed':False}))
        return 0
    require(args.output is not None,'new output path required')
    with driver_overrides():return frozen.main()


if __name__=='__main__':raise SystemExit(main())
