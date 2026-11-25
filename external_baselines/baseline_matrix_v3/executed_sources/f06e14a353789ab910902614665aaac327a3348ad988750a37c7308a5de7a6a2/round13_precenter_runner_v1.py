"""Round13 optional position-conditioned center probe, same full native M1 driver.

This is a separate runtime graph. Existing Round13 queued snapshots are never
changed. Actual prefix frequencies are copied before the question is processed.
"""
from __future__ import annotations
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import round13_runner_v1 as parent

MODES=('PC','PC_DONOR')
EXTRA=('round13_precenter_runner_v1.py','round13_precenter_bank_v1.py','round13_precenter_attention_v1.py',
       'round13_precenter_technical_v1.py')
_FREQUENCIES=None
_ATTENTION_SCALING=None

def runtime_source_manifest(config=None):
    parent.require(parent.sha(ROOT/'code/round10_runner_v1.py')==parent.BASE_RUNNER_SHA,'frozen base changed')
    return {'code/'+name:parent.sha(ROOT/'code'/name) for name in dict.fromkeys((*parent.RUNTIME_FILES,*EXTRA))}

def candidate_assets(config):
    c=config['candidate']
    parent.require(set(c)<={'mode','format_version','technical_only'} and c.get('mode') in MODES,
                   'INPUT_CONTRACT: explicit position-conditioned center candidate required')
    parent.require(c.get('technical_only',False) is False,'INPUT_CONTRACT: technical variant forbidden')
    return {'schema':'trajkv.round13.precenter.assets.v1','mode':c['mode'],
        'test_question_or_answers_used_by_encoder':False,'prefix_frequencies_copied_before_question':True},{}

def build_candidate_owners(cache,vp,variant,config,assets):
    global _FREQUENCIES,_ATTENTION_SCALING
    from round13_precenter_bank_v1 import build_round13_precenter_owners
    parent.require(_FREQUENCIES is not None and _ATTENTION_SCALING is not None,
                   'INPUT_MISMATCH: native question-free prefix frequencies missing')
    freq,scale=_FREQUENCIES,_ATTENTION_SCALING
    _FREQUENCIES=_ATTENTION_SCALING=None
    try:
        candidate={'mode':'PC' if config['candidate']['mode']=='PC' else 'donor','technical_only':False}
        owners,meta=build_round13_precenter_owners(cache,vp,variant,candidate=candidate,
            frozen_prefix_inv_freq=freq,frozen_prefix_attention_scaling=scale,
            **{k:config[k] for k in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source')},
            expected_asset_sha256=assets)
        meta['round13_router']={'mode':config['candidate']['mode'],'query_unknown':True,
            'native_question_free_prefix_only':True,'selection_uses_test_question_or_reference':False,
            'original_visual_count':int(vp.numel()),'technical_only':False}
        return owners,meta
    finally:
        # No extra transient frequency copy survives the ordinary owner audit.
        freq=None

def physical_budget_guard(mode,n,complete_bytes,metadata,base_profile):
    from round13_precenter_bank_v1 import budget_plan
    parent.require(mode in MODES and base_profile=='r7_joint_k1_gt_cond','OWNER_AUDIT_FAILED: PC lineage')
    detail=metadata['round13_precenter']; plan=budget_plan(n)
    parent.require(metadata['returned_owners_unique_tensor_capacity_bytes']==complete_bytes,
                   'OWNER_AUDIT_FAILED: PC factory and Hybrid capacity differ')
    parent.require(detail['actual_owner_bytes']==complete_bytes and 0<complete_bytes<=plan['R_cap_bytes'],
                   'OWNER_AUDIT_FAILED: PC actual bytes exceed R')
    return {'schema':'trajkv.round13.physical_budget.v1','mode':mode,'N':n,
        'actual_complete_visual_bytes':complete_bytes,'strict_cap_bytes':plan['R_cap_bytes'],
        'strict_cap_satisfied':True,'original_low43_bytes':4590*n+89733,
        'original_R_bytes':plan['R_cap_bytes'],'selected_stages':metadata['selected_stages'],
        'selected_stage_count':metadata['selected_stage_count'],
        'actual_unique_storage_not_nominal_bits':True,'original_visual_rows_denominator':n,
        'timing_is_instrumented_quality_run_not_controlled_speed':True}

@contextmanager
def overrides():
    global _FREQUENCIES,_ATTENTION_SCALING
    import torch
    protocol=importlib.import_module('m1_protocol'); prefix0=protocol.m1_prefix_forward
    def prefix(model,batch):
        global _FREQUENCIES,_ATTENTION_SCALING
        result=prefix0(model,batch)
        rotary=model.model.language_model.rotary_emb
        freq=rotary.inv_freq
        parent.require(freq.dtype==torch.float32 and tuple(freq.shape)==(64,),
                       'INPUT_MISMATCH: expected actual M1 native64 FP32 frequencies')
        _FREQUENCIES=freq.detach().clone().contiguous()
        _ATTENTION_SCALING=float(rotary.attention_scaling)
        return result
    changes={'MODES':MODES,'runtime_source_manifest':runtime_source_manifest,
        'candidate_assets':candidate_assets,'build_candidate_owners':build_candidate_owners,
        'physical_budget_guard':physical_budget_guard}
    saved={k:getattr(parent,k) for k in changes}
    try:
        for k,v in changes.items():setattr(parent,k,v)
        protocol.m1_prefix_forward=prefix
        with parent.driver_overrides():yield
    finally:
        protocol.m1_prefix_forward=prefix0
        for k,v in saved.items():setattr(parent,k,v)
        _FREQUENCIES=_ATTENTION_SCALING=None

def main():
    with overrides():return parent.frozen.main()
if __name__=='__main__':raise SystemExit(main())
