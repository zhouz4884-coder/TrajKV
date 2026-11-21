"""Round9 real packed candidate answers on the aligned M1 protocol.

Derived from immutable Round9 runner v4; adds frozen cal-only U3/D11 controls.
No reference answers are loaded during inference.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import weakref

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
SCHEMA = 'trajkv.round9_research.raw.v1'
SUFFIX = '\nAnswer the question using a single word or phrase.'
ORIGINAL_CONFIG_SHA = 'aba87db8c87bd1ca8ca71685baa7a942c0450c8ecc1302c9ccbe183c67703ef4'
PROFILES = {'r7_joint_k1_gt_cond': {'method_id': 'TrajKV_R7_K1_gtcond43_v2', 'assets': {'direction_book_path': 'af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589', 'metadata_factor_path': '8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea', 'round7_policy_path': '505dbbbc67d50023ec0f60fb31ab9a31b0a27212fe7c3e8a2e72de0307b3f38f'}, 'require_43_stages': True}, 'r7_budget_k2_cond': {'method_id': 'TrajKV_R7_K2_cond_v2', 'assets': {'direction_book_path': '9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae', 'metadata_factor_path': 'd60fe4ffcf31e36306ed2c5a4069634627c83b3af2f0b86fdb5c37e456113671', 'round7_policy_path': 'e8a58e602c047adbd4ca6ae253f6ba2eb3961158d7c75444d3e27a41ec47ed6e'}, 'require_43_stages': False}}
CORE_SOURCES = {
    'code/m1_protocol.py':'4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427',
    'code/cache_bridge.py':'36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025',
    'code/attention_bridge.py':'eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876',
    'code/ownership.py':'8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456'}
FORBIDDEN = {'refs','references','reference_answers','reference_path','answers','labels','scores'}
PRIMARY_METRIC_IDS = {
    'textvqa':'textvqa_mmf_ten_vote_leave_one_out_v1',
    'infovqa':'infovqa_anls_qwenvl_vendor_rawlen_ge0p5_v1',
    'docvqa':'docvqa_anls_qwenvl_vendor_rawlen_ge0p5_v1',
    'chartqa':'chartqa_pix2struct_relaxed_5pct_v1'}
BRPE_MODES={'BRPE_Lookup8':'Lookup8','BRPE_Drop4':'Drop4','BRPE_Exchange':'Exchange'}
BRPE_HELPER_SHA='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'
BRPE3_MODES={'BRPE3_DropD':'DropD','BRPE3_Exchange':'Exchange'}
BRPE3_PROFILE_SHA='349514921518520383c375b9b5ead97ceaab356484f080cb546a91b1e188aa1a'
BRPE3_HELPER_SHA='7658d920970a9eba23b4edaed2c110e523ca66326aacafb7031ca188aa37bd99'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''): value.update(part)
    return value.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                         separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def brpe_assets(config):
    """Exact calibrated profile and high book; never alter the base V0 config."""
    c=config['candidate'];require(c['mode'] in BRPE_MODES,'BRPE mode required')
    require(config['variant_source']['name']=='r7_joint_k1_gt_cond' and config['preprocessing']['max_patches']==8,
            'INPUT_CONTRACT: BRPE is the fixed M1 low/cap8 profile only')
    require(not any(c.get(k) for k in ('theta_path','theta_sha256','book_path','book_sha256')) and c.get('value_mode','original')=='original',
            'INPUT_CONTRACT: BRPE retains original calibrated V0 and original K factors')
    require(isinstance(c.get('profile_path'),str) and isinstance(c.get('profile_sha256'),str) and len(c['profile_sha256'])==64,
            'INPUT_CONTRACT: explicit profile_path/profile_sha256 required')
    require(sha(c['profile_path'])==c['profile_sha256'],'INPUT_MISMATCH: calibrated profile SHA differs')
    p=json.loads(Path(c['profile_path']).read_text(encoding='utf-8-sig'))
    low=PROFILES['r7_joint_k1_gt_cond']['assets']
    require(p.get('schema')=='trajkv.round9.precision_exchange_profile.v1' and p.get('base_profile')=='r7_joint_k1_gt_cond' and
            p.get('low_book_sha256')==low['direction_book_path'] and p.get('base_metadata_factor_sha256')==low['metadata_factor_path'],
            'INPUT_CONTRACT: wrong BRPE profile/base source')
    require(type(p.get('upgraded_layer')) is int and 0<=p['upgraded_layer']<28,'INPUT_CONTRACT: one whole upgraded layer')
    drops=p.get('dropped_k_units')
    require(isinstance(drops,list) and len(drops)==4 and all(isinstance(x,list) and len(x)==3 and all(type(v) is int for v in x) for x in drops) and
            len({tuple(x) for x in drops})==4,'INPUT_CONTRACT: four distinct K-only donor triples')
    selected=config['variant']['selected_stages'][:43]
    require(all(['k',*x] in selected for x in drops),'INPUT_CONTRACT: donor outside original43 K coverage')
    require(isinstance(p.get('high_book_path'),str) and isinstance(p.get('high_book_sha256'),str) and len(p['high_book_sha256'])==64 and
            sha(p['high_book_path'])==p['high_book_sha256'],'INPUT_MISMATCH: calibrated high-book SHA differs')
    high=json.loads(Path(p['high_book_path']).read_text(encoding='utf-8-sig'))['centers']
    require(isinstance(high,list) and len(high)==256 and all(isinstance(row,list) and len(row)==4 for row in high),'INPUT_CONTRACT: high book must be256x4')
    require(sha(ROOT/'code/round9_precision_exchange_v1.py')==BRPE_HELPER_SHA,'INPUT_MISMATCH: reviewed BRPE helper changed')
    if 'source_provenance' in p:
        require(p['source_provenance'].get('exchange_helper')==BRPE_HELPER_SHA,'INPUT_MISMATCH: profile selected with another exchange helper')
    return p,{c['profile_path']:c['profile_sha256'],p['high_book_path']:p['high_book_sha256']}


def brpe_budget_guard(mode,n,complete_bytes,metadata):
    """Actual single-VP complete owner must obey the old R7 low B0 envelope."""
    require(mode in BRPE_MODES and type(n) is int and n>=256 and n%32==0,'OWNER_AUDIT_FAILED: BRPE native token scope')
    b0=3816*n+89733+18*n*43
    expected_delta=-1344 if mode=='BRPE_Lookup8' else -72*n-1304 if mode=='BRPE_Drop4' else 744-8*n
    proof=metadata['round9_precision_exchange'];stages=43 if mode=='BRPE_Lookup8' else 39
    require(metadata['selected_stage_count']==stages and proof['mode']==BRPE_MODES[mode] and
            sum(x[0]=='v' for x in metadata['selected_stages'])==20,'OWNER_AUDIT_FAILED: BRPE stage/V20 scope differs')
    require(proof['source_sha256']==BRPE_HELPER_SHA and proof['V20_base_parameters_and_residual_values_unchanged'] is True and
            proof['old_lookup_and_original_pools_released'] is True and proof['lookup_dtype']=='torch.int8' and proof['lookup_bytes']==448,
            'OWNER_AUDIT_FAILED: BRPE field/lookup/release proof differs')
    require(proof['before_unique_bytes']==b0 and proof['expected_delta_bytes']==proof['actual_delta_bytes']==expected_delta and
            proof['after_unique_bytes']==metadata['returned_owners_unique_tensor_capacity_bytes']==complete_bytes==b0+expected_delta<=b0,
            'OWNER_AUDIT_FAILED: actual BRPE complete bytes exceed/differ from old R7 B0 ledger')
    return {'old_R7_low_B0_bytes':b0,'actual_complete_bytes':complete_bytes,'actual_minus_B0_bytes':complete_bytes-b0,
            'exact_expected_delta_bytes':expected_delta,'all_persistent_constants_counted':True,
            'at_or_below_R7_low_B0':True,'strict1p3_budget_is_separate':True}


def brpe3_assets(config):
    c=config['candidate'];require(c['mode'] in BRPE3_MODES,'BRPE3 mode required')
    require(config['variant_source']['name']=='r7_joint_k1_gt_cond' and config['preprocessing']['max_patches']==8,
            'INPUT_CONTRACT: U3D11 fixed low/cap8 only')
    require(not any(c.get(k) for k in ('theta_path','theta_sha256','book_path','book_sha256')) and c.get('value_mode','original')=='original',
            'INPUT_CONTRACT: fixed original V0 and K112')
    require(isinstance(c.get('profile_path'),str) and c.get('profile_sha256')==BRPE3_PROFILE_SHA and sha(c['profile_path'])==BRPE3_PROFILE_SHA,
            'INPUT_MISMATCH: exact cal-only DP U3D11 profile required')
    p=json.loads(Path(c['profile_path']).read_text(encoding='utf-8-sig'))
    import round9_precision_exchange_v2 as multi
    require(sha(multi.__file__)==BRPE3_HELPER_SHA,'INPUT_MISMATCH: U3D11 deployment helper changed')
    multi.validate_profile(p,allow_diagnostic=False)
    require(sha(p['high_book_path'])==p['high_book_sha256']==multi.HIGH_SHA,'INPUT_MISMATCH: shared high book changed')
    return p,{c['profile_path']:BRPE3_PROFILE_SHA,p['high_book_path']:p['high_book_sha256']}


def brpe3_budget_guard(mode,n,complete_bytes,metadata):
    require(mode in BRPE3_MODES and type(n) is int and n>=256 and n%32==0,'OWNER_AUDIT_FAILED: U3D11 token scope')
    b0=3816*n+89733+18*n*43;delta=-198*n-1283 if mode=='BRPE3_DropD' else 765-6*n
    p=metadata['round9_precision_exchange']
    require(metadata['selected_stage_count']==32 and sum(x[0]=='k' for x in metadata['selected_stages'])==12 and sum(x[0]=='v' for x in metadata['selected_stages'])==20,
            'OWNER_AUDIT_FAILED: K12/V20 scope')
    require(p['schema']=='trajkv.round9.precision_exchange.v2' and p['mode']==BRPE3_MODES[mode] and p['U']==3 and p['D']==11 and
            p['source_sha256']==BRPE3_HELPER_SHA and p['numerical_fields_v1_sha256']==BRPE_HELPER_SHA and
            p['V20_base_parameters_and_residual_values_unchanged'] is True and p['old_lookup_and_original_pools_released'] is True and
            p['lookup_dtype']=='torch.int8' and p['lookup_bytes']==448 and p['format_bytes']==28 and p['donor_descriptor_bytes']==33,
            'OWNER_AUDIT_FAILED: U3D11 lookup/format/donor/source differs')
    require(p['extra_high_book_bytes']==(2048 if mode=='BRPE3_Exchange' else 0) and p['high_book_storage_shared_across_all_upgraded_layers'] is True,
            'OWNER_AUDIT_FAILED: high book not shared/fully charged')
    require(p['before_unique_bytes']==b0 and p['expected_delta_bytes']==p['actual_delta_bytes']==delta and
            p['after_unique_bytes']==metadata['returned_owners_unique_tensor_capacity_bytes']==complete_bytes==b0+delta<=b0,
            'OWNER_AUDIT_FAILED: U3D11 actual bytes differ from B0 ledger')
    return {'old_R7_low_B0_bytes':b0,'actual_complete_bytes':complete_bytes,'actual_minus_B0_bytes':delta,
            'exact_expected_delta_bytes':delta,'at_or_below_R7_low_B0':True,'shared_high_book_count':1 if mode=='BRPE3_Exchange' else 0,
            'strict1p3_budget_is_separate':True}


def validate_config(config):
    require(config.get('schema')==SCHEMA,'INPUT_CONTRACT: Round9 research schema required')
    require(not (set(config)&FORBIDDEN),'INPUT_CONTRACT: no inference labels/reference paths')
    require(config.get('model_id')=='M1' and config.get('seed',0)==0,'M1 seed0 only')
    require(config.get('question_suffix')==SUFFIX,'frozen question suffix required')
    require(config.get('preprocessing',{}).get('max_patches') in (8,16),'native cap8/16 required')
    require(config.get('max_new_tokens')==64,'answer cap64 required')
    require(config.get('position_owner_layout')=='single_shared_visual_position_map','shared VP layout')
    require(config.get('evaluation_contract',{}).get('version')=='matrix_v3_metrics_v1','frozen scoring contract')
    require(isinstance(config.get('inputs'),str) and len(config.get('inputs_sha256',''))==64,'locked input file')
    require(type(config.get('start')) is int and config['start']>=0,'explicit nonnegative start')
    require(type(config.get('count')) is int and 0<config['count']<=500,'explicit count1..500')
    require(type(config.get('input_count')) is int and config['start']+config['count']<=config['input_count'],'input slice')
    require(config.get('cohort_kind') in ('development','formal500'),'explicit development/formal500 cohort')
    if config['cohort_kind']=='formal500':
        require(config['start']==0 and config['count']==config['input_count']==500,'formal original500 only')
    source=config.get('variant_source',{})
    require(source.get('name') in PROFILES and source.get('sha256')==ORIGINAL_CONFIG_SHA,'original base profile')
    require(config.get('expected_asset_sha256')==PROFILES[source['name']]['assets'],'original base assets')
    candidate=config.get('candidate',{})
    require(candidate.get('mode') in ('FP16','original','M-V','V-FZ','V-RE','FB0','FB1','K-MSE','K-Func','COMBO','TIER','K-Mu','K-Func_FZJ','K-Book_FZJ',*BRPE_MODES,*BRPE3_MODES),'candidate mode')
    require(config.get('method',{}).get('family')=='trajkv_round9_research','explicit research method family')
    require(config['method'].get('information_condition')=='query_unknown','encode before question')
    for name in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source'):
        require(isinstance(config.get(name),str) and config[name],'explicit legacy dependency '+name)
    if candidate['mode'] in BRPE_MODES:brpe_assets(config)
    if candidate['mode'] in BRPE3_MODES:brpe3_assets(config)
    return config

def build_candidate_owners(cache, vp, variant, config, assets):
    mode=config['candidate']['mode']
    if mode=='FP16':
        from cache_bridge import FP16VisualOwner
        owners=[FP16VisualOwner(layer.keys.index_select(2,vp).contiguous(),
                 layer.values.index_select(2,vp).contiguous(),vp) for layer in cache.layers]
        return owners, {'selected_stage_count':0,'variant_sha256':canonical_sha(variant),
            'original_FP16_visual_storage_retained_by_owners':False,'real_model_or_cache_patched':False,
            'legacy_sources':{},'mode':'FP16','new_parameter_fit':False,
            'scope':'same prefix/suffix protocol; actual separate FP16 visual tensors'}
    kwargs=dict(legacy_a2=config['legacy_a2'],legacy_experiments=config['legacy_experiments'],
                legacy_scripts=config['legacy_scripts'],kivi_source=config['kivi_source'],expected_asset_sha256=assets)
    if mode in BRPE3_MODES:
        from round9_precision_exchange_v2 import build_round9_precision_exchange_owners
        candidate,_=brpe3_assets(config)
        return build_round9_precision_exchange_owners(cache,vp,variant,mode=BRPE3_MODES[mode],candidate=candidate,**kwargs)
    if mode in BRPE_MODES:
        from round9_precision_exchange_v1 import build_round9_precision_exchange_owners
        candidate,_=brpe_assets(config)
        return build_round9_precision_exchange_owners(cache,vp,variant,mode=BRPE_MODES[mode],candidate=candidate,**kwargs)
    if mode=='original':
        from r7_legacy_bridge_v2 import build_r7_owners
        return build_r7_owners(cache,vp,variant,**kwargs)
    if mode in ('M-V','V-FZ','V-RE','FB0','FB1'):
        from round9_value_bank_v1 import build_round9_value_owners
        c=config['candidate']
        return build_round9_value_owners(cache,vp,variant,mode=mode,
                theta_path=c.get('theta_path'),theta_sha256=c.get('theta_sha256'),**kwargs)
    if mode in ('K-Mu','K-Func_FZJ','K-Book_FZJ'):
        from round9_key_controls_v2 import build_round9_key_control_owners
        c=config['candidate']
        return build_round9_key_control_owners(cache,vp,variant,mode=mode,
            book_path=c.get('book_path'),book_sha256=c.get('book_sha256'),**kwargs)
    from round9_extended_bank_v2 import build_round9_extended_owners
    return build_round9_extended_owners(cache,vp,variant,candidate=config['candidate'],**kwargs)



def frozen_variant(config):
    source=config['variant_source'];profile=PROFILES[source['name']];assets=profile['assets'];require(sha(source['path'])==source['sha256'],'INPUT_MISMATCH: original profile file changed')
    original=json.loads(Path(source['path']).read_text(encoding='utf-8-sig'))
    matched=[v for v in original['variants'] if v['name']==source['name']]
    require(len(matched)==1,'INPUT_MISMATCH: exactly one requested original R7 profile required')
    variant=config['variant'];other=lambda v:{key:value for key,value in v.items() if key not in assets}
    require(other(variant)==other(matched[0]),'INPUT_MISMATCH: frozen variant changed outside three asset path rebindings')
    require(all(sha(variant[key])==value for key,value in assets.items()),'INPUT_MISMATCH: frozen codec/factor/policy asset changed')
    return variant


def share_original_visual_position_map(cache):
    """Keep R7's one shared, all-support position owner without changing values."""
    import torch
    from ownership import cache_storage
    require(len(cache.layers)==28, 'OWNER_AUDIT_FAILED: M1 shared position geometry differs')
    target=cache.layers[0].visual_owner.positions
    old=cache.visual_positions
    require(target.ndim==1 and target.dtype==old.dtype==torch.long and target.device==old.device
            and torch.equal(target,old), 'OWNER_AUDIT_FAILED: position maps are not identical')
    identity=lambda value:(str(value.device),value.untyped_storage().data_ptr(),value.untyped_storage().nbytes())
    require(identity(target)!=identity(old), 'OWNER_AUDIT_FAILED: expected distinct initial Hybrid position clone')
    for layer in cache.layers:
        require(torch.equal(layer.original_visual_positions,target) and
                torch.equal(layer.visual_owner.positions,target) and
                identity(layer.visual_owner.positions)==identity(target),
                'OWNER_AUDIT_FAILED: only all-support globally shared R7 positions may be aliased')
    before=cache_storage(cache,target)['complete_visual_owner_bytes']
    expected_saved=old.untyped_storage().nbytes()
    require(expected_saved==8*target.numel(), 'OWNER_AUDIT_FAILED: original map capacity differs')
    previous=weakref.ref(old)
    cache.visual_positions=target
    for layer in cache.layers:
        layer.original_visual_positions=target
    old=None
    gc.collect()
    require(previous() is None, 'OWNER_AUDIT_FAILED: obsolete Hybrid position clone remains retained')
    after=cache_storage(cache,target)['complete_visual_owner_bytes']
    require(before-after==expected_saved, 'OWNER_AUDIT_FAILED: physical shared-position charge differs')
    return {'layout':'single_shared_visual_position_map','same_position_values':True,
            'all_cache_layer_and_owner_fields_share_one_storage':True,'obsolete_clone_released':True,
            'before_complete_visual_bytes':before,'after_complete_visual_bytes':after,
            'actual_released_capacity_bytes':expected_saved,'codec_allocation_or_reader_changed':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=validate_config(json.loads(args.config.read_text(encoding='utf-8-sig')))
    profile=PROFILES[config['variant_source']['name']];assets=profile['assets']
    variant=frozen_variant(config);variant_sha=canonical_sha(variant)
    require(all(sha(ROOT/key)==value for key,value in CORE_SOURCES.items()),'INPUT_MISMATCH: frozen M1/cache/ownership sources changed')
    from round9_inputs_v1 import load_round9_inputs
    from dataset_registry import load_inputs as load_formal_inputs
    from model_registry import get_model_spec
    require(sha(config['inputs'])==config['inputs_sha256'], 'INPUT_MISMATCH: original fixed500 input file changed')
    all_rows=(load_formal_inputs(config['inputs']) if config['cohort_kind']=='formal500' else
              load_round9_inputs(config['inputs'],expected_sha256=config['inputs_sha256']))
    start=config.get('start',0)
    require(len(all_rows)==config['input_count'], 'INPUT_MISMATCH: locked cohort count differs')
    rows=all_rows[start:start+config['count']]
    require(len(rows)==config['count'] and len({row['id'] for row in rows})==len(rows),'INPUT_MISMATCH: incomplete/duplicate fixed input slice')
    require(len({row['dataset'] for row in rows})==1 and rows[0]['dataset'] in ('textvqa','infovqa','docvqa','chartqa'),
            'INPUT_MISMATCH: one of the four fixed official-task datasets per run')
    require(config['evaluation_contract'].get('primary_metric_id')==PRIMARY_METRIC_IDS[rows[0]['dataset']],
            'INPUT_MISMATCH: frozen primary metric identity differs')
    spec=get_model_spec('M1');require(config.get('model_revision',spec['revision'])==spec['revision'],'INPUT_MISMATCH: M1 revision differs')
    args.output.mkdir(parents=True,exist_ok=False)
    expected=[row['id'] for row in rows]
    source_paths=[Path(__file__),*[ROOT/'code'/name for name in ('r7_legacy_bridge_v1.py','r7_legacy_bridge_v2.py','m1_protocol.py',
        'model_registry.py','dataset_registry.py','evaluate.py','cache_bridge.py','attention_bridge.py','ownership.py',
        'round9_value_bank_v1.py','round9_inputs_v1.py','round9_extended_bank_v1.py',
        'round9_extended_bank_v2.py','round9_key_controls_v1.py','round9_key_controls_v2.py','round9_precision_exchange_v1.py','round9_precision_exchange_v2.py')]]
    source_hashes={str(path.relative_to(ROOT)):sha(path) for path in source_paths}
    locked_files={str(args.config):sha(args.config),str(config['inputs']):sha(config['inputs']),
                  config['variant_source']['path']:ORIGINAL_CONFIG_SHA,
                  **{variant[key]:value for key,value in assets.items()},
                  config['kivi_source']:sha(config['kivi_source'])}
    for key in ('theta','book'):
        candidate_path=config['candidate'].get(key+'_path')
        if candidate_path:
            require(sha(candidate_path)==config['candidate'][key+'_sha256'],'candidate asset identity changed')
            locked_files[candidate_path]=config['candidate'][key+'_sha256']
    brpe_profile=None
    if config['candidate']['mode'] in BRPE_MODES:
        brpe_profile,brpe_locks=brpe_assets(config);locked_files.update(brpe_locks)
    brpe3_profile=None
    if config['candidate']['mode'] in BRPE3_MODES:
        brpe3_profile,brpe3_locks=brpe3_assets(config);locked_files.update(brpe3_locks)
    manifest={'schema':'baseline_matrix.raw_inference.v1','runner_schema':SCHEMA,'config':config,
        'config_sha256':sha(args.config),'inputs_sha256':sha(config['inputs']),'expected_ids':expected,'expected_QA':len(rows),
        'model_spec':spec,'references_read':False,'prediction_cache_used':False,'runtime_source_root':str(ROOT),
        'runtime_executable':sys.executable,'started_unix':time.time(),'source_hashes':source_hashes,
        'frozen_variant':variant,'frozen_variant_sha256':variant_sha,'asset_sha256':assets,
        'variant_original_source':config['variant_source'],'base_profile_numeric_fields_unchanged':True,'candidate':config['candidate'],
        'scope':config['scope'],
        'parameter_fit_or_new_algorithm':config['candidate']['mode'] not in ('FP16','original'),'native_FP16_visual_source_retained_for_serving':False,
        'timing_scope':'instrumented quality-run prototype; not controlled speed or HBM traffic benchmark',
        'first_logits_evidence':'one actual raw last-position logits vector before HF logits processors, saved on data disk'}
    if brpe_profile is not None:
        manifest['BRPE_assets']={'profile_path':config['candidate']['profile_path'],'profile_sha256':config['candidate']['profile_sha256'],
            'profile_schema':brpe_profile['schema'],'high_book_path':brpe_profile['high_book_path'],'high_book_sha256':brpe_profile['high_book_sha256'],
            'helper_sha256':BRPE_HELPER_SHA,'V0_retained':True,'stage_count':43 if config['candidate']['mode']=='BRPE_Lookup8' else 39,
            'source_provenance':brpe_profile.get('source_provenance'),'collection_provenance':brpe_profile.get('collection_provenance')}
    if brpe3_profile is not None:
        manifest['BRPE3_assets']={'profile_path':config['candidate']['profile_path'],'profile_sha256':BRPE3_PROFILE_SHA,
            'high_book_path':brpe3_profile['high_book_path'],'high_book_sha256':brpe3_profile['high_book_sha256'],
            'helper_sha256':BRPE3_HELPER_SHA,'numerical_v1_sha256':BRPE_HELPER_SHA,'V0_retained':True,'stage_count':32,
            'source_provenance':brpe3_profile['source_provenance'],'selection_provenance':brpe3_profile['selection_provenance']}
    records=[];load_error=None;manifest_written=False;actual_legacy_sources={};model=processor=None
    record_path=args.output/'records.jsonl'
    def append(record):
        number=len(records);write_new(args.output/'records'/(f'{number:06d}_'+hashlib.sha256(record['id'].encode()).hexdigest()[:16]+'.json'),record)
        with record_path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        records.append(record)
    try:
        import torch
        import transformers
        import tokenizers
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from evaluate import model_identity,tensor_sha,GenerationObserver,observe_generation
        from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_generate_cached
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context,language_attention_modules
        from ownership import model_storage,cache_storage,assert_no_full_prefix_alias
        from r7_legacy_bridge_v2 import build_r7_owners
        torch.set_grad_enabled(False);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.cuda.is_available(),'HARDWARE_UNSUPPORTED: actual CUDA required');torch.cuda.set_device(0)
        require(torch.cuda.get_device_capability()==(8,6),'HARDWARE_UNSUPPORTED: actual RTX3090 SM86 required')
        identity=model_identity(config,spec)
        processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(config['model_path'],dtype=torch.float16,
                device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        require(all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False),
                'INPUT_MISMATCH: original FP16 unquantized weights required')
        require(int(model.generation_config.num_beams or 1)==1,'INPUT_MISMATCH: native greedy beam-one generation required')
        _,modules=language_attention_modules(model);require(len(modules)==28,'INPUT_MISMATCH: M1 layer geometry differs')
        manifest.update(model_identity=identity,torch_version=torch.__version__,transformers_version=transformers.__version__,
            CUDA_version=torch.version.cuda,hardware=torch.cuda.get_device_name(),python_version=platform.python_version(),
            weight_dtype='float16',tf32=False,seed=0,attention_implementation='sdpa',model_residency=model_storage(model),
            processor_id=spec['hf_id'],processor_version=transformers.__version__,
            processor_file_hashes={x['file']:x['sha256'] for x in identity['files'] if not x['file'].endswith('.safetensors')},
            generation_native=model.generation_config.to_dict(),generation_overrides={'do_sample':False,'max_new_tokens':64,
                'use_cache':True,'return_dict_in_generate':False},tokenizer_class=type(processor.tokenizer).__module__+'.'+type(processor.tokenizer).__name__,
            tokenizer_is_fast=bool(getattr(processor.tokenizer,'is_fast',False)),tokenizers_version=tokenizers.__version__,
            decode_overrides={'clean_up_tokenization_spaces':False,'skip_special_tokens':True})
        manifest['runtime_library_source_sha256']={str(Path(inspect.getfile(t))):sha(inspect.getfile(t))
                            for t in (type(model),type(processor),type(processor.tokenizer))}
        write_new(args.output/'manifest.json',manifest);manifest_written=True
        for row in rows:
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            record={**row,'model_id':'M1','model_commit':spec['revision'],'model_arch':spec['architecture'],'dtype':'float16',
                'method':config['method']['id'],'method_contract':config['method'],'preprocessing':config['preprocessing'],
                'status':'RUNNING','output_token_ids':None,'raw_prediction':None,'official_score':None,
                'score_status':'NOT_READ_OR_COMPUTED_BY_RUNNER','timing_scope':manifest['timing_scope']}
            cache=batch=prefix_output=owners=metadata=generated=observer=image=None;first_events=[];first_logits=[]
            pre_handle=post_handle=None
            try:
                require(sha(row['image_path'])==row['original_image_sha256'],'INPUT_MISMATCH: original image SHA differs')
                with Image.open(row['image_path']) as raw:
                    record['original_dimensions']=list(raw.size);image=raw.convert('RGB')
                rendered=row['question']+SUFFIX
                batch=prepare_m1_inputs(processor,image,rendered,image_token_id=spec['image_token_id'],max_patches=config['preprocessing']['max_patches'])
                alternate_question='Describe the objects.' if row['question']=='What is visible in the image?' else 'What is visible in the image?'
                alternate=prepare_m1_inputs(processor,image,alternate_question+SUFFIX,image_token_id=spec['image_token_id'],max_patches=config['preprocessing']['max_patches'])
                require(alternate.boundary==batch.boundary and torch.equal(alternate.ids[:,:batch.boundary],batch.ids[:,:batch.boundary])
                        and torch.equal(alternate.visual_positions,batch.visual_positions)
                        and torch.equal(alternate.positions[:,:batch.boundary],batch.positions[:,:batch.boundary]),
                        'INPUT_MISMATCH: image prefix depends on future question')
                for key in set(batch.inputs)|set(alternate.inputs):
                    if key in ('input_ids','attention_mask','position_ids','token_type_ids'):continue
                    require(key in batch.inputs and key in alternate.inputs,'INPUT_MISMATCH: future question changes visual input keys')
                    a,b=batch.inputs[key],alternate.inputs[key]
                    require(torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b,'INPUT_MISMATCH: native image processing depends on question')
                a=b=None;alternate=None
                template=processor.chat_template
                record.update(raw_question=row['question'],rendered_question=rendered,question_suffix=SUFFIX,
                    raw_question_sha256=hashlib.sha256(row['question'].encode()).hexdigest(),
                    rendered_question_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    full_prompt=batch.prompt,full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest(),
                    chat_template_hash=hashlib.sha256(template.encode()).hexdigest() if isinstance(template,str) else canonical_sha(template),
                    input_tokens=batch.full_length,input_token_ids=batch.ids.detach().cpu().reshape(-1).tolist(),
                    input_ids_sha256=tensor_sha(batch.ids),question_suffix_sha256=tensor_sha(batch.ids[:,batch.boundary:]),
                    compression_boundary=batch.boundary,question_suffix_tokens=batch.suffix_length,
                    visual_positions=batch.visual_positions.detach().cpu().tolist(),actual_visual_token_count=int(batch.visual_positions.numel()),
                    native_position_ids_shape=list(batch.positions.shape),native_position_ids_sha256=tensor_sha(batch.positions),
                    native_position_ids=batch.positions.detach().cpu().reshape(-1).tolist(),question_unknown_prefix_verified=True)
                observer=GenerationObserver(model,started,batch)
                with observe_generation(model,observer):
                    observer.phase='prefix';t0=time.perf_counter();prefix_output=m1_prefix_forward(model,batch)
                    torch.cuda.synchronize();record['prefill_ms']=(time.perf_counter()-t0)*1000
                    cache=prefix_output.past_key_values;prefix_output=None
                    require(int(cache.get_seq_length())==batch.boundary and len(cache.layers)==28,'INPUT_MISMATCH: native prefix geometry differs')
                    vp=batch.visual_positions;record['native_prefix_owner']=cache_storage(cache,vp)
                    refs=[weakref.ref(t) for layer in cache.layers for t in (layer.keys,layer.values)]
                    old_storage={(str(t.device),t.untyped_storage().data_ptr()) for layer in cache.layers for t in (layer.keys,layer.values)}
                    observer.phase='encoding';t0=time.perf_counter()
                    owners,metadata=build_candidate_owners(cache,vp,variant,config,assets)
                    if config['candidate']['mode'] in BRPE3_MODES:
                        require(len(owners)==28 and metadata['selected_stage_count']==32,'OWNER_AUDIT_FAILED: U3D11 stage32')
                    elif config['candidate']['mode'] in BRPE_MODES:
                        require(len(owners)==28 and metadata['selected_stage_count']==(43 if config['candidate']['mode']=='BRPE_Lookup8' else 39),
                                'OWNER_AUDIT_FAILED: BRPE all-layer stage allocation differs')
                    else:
                        require(len(owners)==28 and (metadata['selected_stage_count']==0 if config['candidate']['mode']=='FP16' else metadata['selected_stage_count']==43 if profile['require_43_stages'] and config['candidate']['mode']!='TIER' else 0<=metadata['selected_stage_count']<=224), 'OWNER_AUDIT_FAILED: candidate all-layer allocation differs')
                    require(metadata['variant_sha256']==variant_sha and metadata['original_FP16_visual_storage_retained_by_owners'] is False
                            and metadata['real_model_or_cache_patched'] is False,'OWNER_AUDIT_FAILED: bridge retained source or changed scope')
                    for path,info in metadata['legacy_sources'].items():
                        require(path not in actual_legacy_sources or actual_legacy_sources[path]==info['sha256'],
                                'INPUT_MISMATCH: legacy source changed between QA')
                        actual_legacy_sources[path]=info['sha256']
                    cache=HybridVisualCache(cache,vp,owners,consume_native=True)
                    record['position_owner_layout']=share_original_visual_position_map(cache)
                    record['r7_bridge']=metadata;owners=metadata=None;gc.collect();torch.cuda.synchronize()
                    require(not any(ref() is not None for ref in refs),'OWNER_AUDIT_FAILED: original native prefix tensor retained')
                    assert_no_full_prefix_alias(cache,old_storage)
                    record.update(encode_ms=(time.perf_counter()-t0)*1000,original_full_prefix_released=True,
                                  original_FP16_visual_alias_retained=False)
                    # Only IDs/mask and original logical positions are needed now.
                    keep_ids,keep_mask=batch.ids,batch.mask
                    expected_first_ids=tensor_sha(keep_ids[:,batch.boundary:])
                    expected_first_positions=tensor_sha(batch.positions[:,batch.boundary:])
                    expected_first_position_shape=list(batch.positions[:,batch.boundary:].shape)
                    expected_cache_position=tensor_sha(torch.arange(batch.boundary,batch.full_length,device=keep_ids.device))
                    batch.inputs={'input_ids':keep_ids,'attention_mask':keep_mask};batch.prefix_inputs={}
                    batch.positions=batch.positions.detach().cpu();batch.visual_positions=batch.visual_positions.detach().cpu();vp=batch.visual_positions
                    gc.collect();record['owner_before_suffix']=cache_storage(cache,vp)
                    own=record['owner_before_suffix'];complete_bytes=own['stored_visual_KV_bytes']
                    if config['candidate']['mode'] in BRPE3_MODES:
                        record['BRPE3_budget_vs_original_R7_low']=brpe3_budget_guard(config['candidate']['mode'],len(record['visual_positions']),complete_bytes,record['r7_bridge'])
                    if config['candidate']['mode'] in BRPE_MODES:
                        record['BRPE_budget_vs_original_R7_low']=brpe_budget_guard(config['candidate']['mode'],len(record['visual_positions']),complete_bytes,record['r7_bridge'])
                    for field in ('stored_visual_payload_bytes','coder_metadata_bytes','positions_bytes','encoder_constants_bytes','text_positions_bytes','text_kv_bytes'):
                        record[field]=own[field]
                    denominator=28*2*4*len(record['visual_positions'])*128*2
                    record.update(stored_visual_KV_bytes=complete_bytes,complete_visual_owner_bytes=complete_bytes,
                        full_kv_byte_denominator=denominator,actual_effective_visual_bits=16*complete_bytes/denominator,
                        per_request_strict_budget={'1.3':160*complete_bytes<=13*denominator,'1.8':80*complete_bytes<=9*denominator},
                        request_input_tensor_bytes=keep_ids.untyped_storage().nbytes()+keep_mask.untyped_storage().nbytes())
                    def before(module,positional,kwargs):
                        if first_events:return
                        ids=kwargs.get('input_ids');past=kwargs.get('past_key_values')
                        if ids is None and positional and isinstance(positional[0],torch.Tensor):ids=positional[0]
                        positions=kwargs.get('position_ids');cp=kwargs.get('cache_position')
                        event={'input_tokens':int(ids.shape[-1]) if ids is not None else None,
                            'input_ids_sha256':tensor_sha(ids) if ids is not None else None,
                            'cache_tokens_before':int(past.get_seq_length()) if past is not None else 0,
                            'position_ids_sha256':tensor_sha(positions) if positions is not None else None,
                            'position_ids_shape':list(positions.shape) if positions is not None else None,
                            'cache_position_sha256':tensor_sha(cp) if cp is not None else None,
                            'expected_suffix_ids_sha256':expected_first_ids,'expected_original_positions_sha256':expected_first_positions,
                            'expected_original_cache_position_sha256':expected_cache_position,
                            'native_position_inference_used_if_not_explicit':positions is None,'native_first_logits_injected':False}
                        first_events.append(event)
                        require(ids is not None and event['input_ids_sha256']==expected_first_ids and event['input_tokens']==batch.suffix_length
                            and event['cache_tokens_before']==batch.boundary
                            and (positions is None or (event['position_ids_sha256']==expected_first_positions and event['position_ids_shape']==expected_first_position_shape))
                            and (cp is None or event['cache_position_sha256']==expected_cache_position),
                            'INPUT_MISMATCH: actual first generation forward must recompute the whole question suffix at original positions: '+json.dumps(event))
                    def after(module,positional,output):
                        if first_logits:return
                        require(output.logits is not None and bool(torch.isfinite(output.logits).all()),'NONFINITE_LOGITS: first own compressed logits')
                        first_logits.append(output.logits[0,-1].detach().cpu().contiguous().clone())
                    pre_handle=model.register_forward_pre_hook(before,with_kwargs=True)
                    post_handle=model.register_forward_hook(after)
                    observer.phase='generation'
                    try:
                        with attention_context(model,cache=cache):generated=m1_generate_cached(model,batch,cache,max_new_tokens=64)
                    finally:
                        post_handle.remove();pre_handle.remove();post_handle=pre_handle=None
                    tokens=generated[0,batch.full_length:].detach().cpu().tolist()
                    require(torch.equal(generated[:,:batch.full_length],batch.ids),'INPUT_MISMATCH: generation changed original full prompt history')
                    require(tokens and len(tokens)<=64 and observer.stream_ids==tokens and first_events and first_logits,
                            'INPUT_MISMATCH: generated/streamed token IDs or first-call evidence differ')
                    logits=first_logits[0]
                    first_path=args.output/'first_logits'/(hashlib.sha256(row['id'].encode()).hexdigest()[:24]+'.pt')
                    first_path.parent.mkdir(exist_ok=True);require(not first_path.exists(),'Evidence output must not overwrite')
                    torch.save({'id':row['id'],'raw_last_position_logits':logits,'scope':'actual candidate suffix forward, before native HF logits processors'},first_path)
                    top=torch.topk(logits.float(),5)
                    record.update(first_generation_call=first_events[0],first_logits={'file':str(first_path),'sha256':sha(first_path),
                        'bytes':first_path.stat().st_size,'dtype':str(logits.dtype),'vocabulary_size':logits.numel(),
                        'finite':True,'argmax_id':int(logits.argmax()),'top5_ids':top.indices.tolist(),'top5_logits':top.values.tolist(),
                        'download_policy':'remote_data_disk_only; scalar offline scoring does not require tensor download'},
                        output_token_ids=tokens,output_tokens=len(tokens),
                        raw_prediction=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                        raw_decode_with_special_tokens=processor.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False))
                    record['owner_after_generation']=cache_storage(cache,vp)
                    require(record['owner_after_generation']['stored_visual_KV_bytes']==complete_bytes,
                            'OWNER_AUDIT_FAILED: persistent visual owner changed during generation')
                    record['reference_scope']='native_image_prefix_then_full_question_suffix_recompute'
                    record['status']='PASS'
                record['outer_forward_calls']=observer.calls
                record['ttft_ms']=(observer.token_times[0]-started)*1000 if observer.token_times else None
                record['inter_token_ms']=[(b-a)*1000 for a,b in zip(observer.token_times,observer.token_times[1:])]
                record['question_prefill_ms']=next((x['wall_ms'] for x in observer.calls if x['phase']=='generation'),None)
            except Exception as exc:
                error=str(exc);status=('OOM_AT_DECLARED_INPUT' if 'out of memory' in error.lower() else
                    'INPUT_MISMATCH' if 'INPUT_MISMATCH' in error or 'INPUT_CONTRACT' in error else
                    'OWNER_AUDIT_FAILED' if 'OWNER_AUDIT_FAILED' in error else 'FAILED_RUN')
                record.update(status=status,error_type=type(exc).__name__,error=error,traceback=traceback.format_exc(),
                              output_token_ids=None,raw_prediction=None)
                if observer is not None:record['outer_forward_calls']=observer.calls
                if first_events:record['first_generation_call']=first_events[0]
            finally:
                if post_handle is not None:post_handle.remove()
                if pre_handle is not None:pre_handle.remove()
                record.update(wall_ms=(time.perf_counter()-started)*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),end_allocated_bytes=torch.cuda.memory_allocated())
                append(record);print(json.dumps({'id':row['id'],'status':record['status'],'completed':len(records),'expected':len(rows)}),flush=True)
                cache=batch=prefix_output=owners=metadata=generated=observer=image=None
                keep_ids=keep_mask=a=b=None;first_logits.clear();first_events.clear()
                gc.collect();torch.cuda.empty_cache()
        require(all(sha(ROOT/path)==value for path,value in source_hashes.items()),'INPUT_MISMATCH: new/frozen runtime source changed')
        require(all(sha(path)==value for path,value in locked_files.items()),'INPUT_MISMATCH: input/config/profile/assets changed')
        require(all(sha(path)==value for path,value in actual_legacy_sources.items()),'INPUT_MISMATCH: executed legacy source changed')
        require(canonical_sha(config['variant'])==variant_sha,'INPUT_MISMATCH: frozen variant parameters changed')
        require(all(sha(path)==value for path,value in manifest['runtime_library_source_sha256'].items()),'INPUT_MISMATCH: HF runtime source changed')
    except Exception as exc:
        load_error={'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
    finally:
        if not manifest_written:
            manifest['initialization_error']=load_error;write_new(args.output/'manifest.json',manifest)
        done={row['id'] for row in records}
        for row in rows:
            if row['id'] in done:continue
            append({**row,'model_id':'M1','model_commit':spec['revision'],'method':config['method']['id'],
                'method_contract':config['method'],'status':'FAILED_RUN','output_token_ids':None,'raw_prediction':None,
                'official_score':None,'score_status':'NOT_READ_OR_COMPUTED_BY_RUNNER','error':load_error})
        if not record_path.exists():record_path.open('x',encoding='utf-8').close()
        counts={}
        for row in records:counts[row['status']]=counts.get(row['status'],0)+1
        resources=[r['actual_effective_visual_bits'] for r in records if r['status']=='PASS']
        totalbytes=[r['stored_visual_KV_bytes'] for r in records if r['status']=='PASS']
        complete={'status':'PASS_RAW_COMPLETE' if load_error is None and counts.get('PASS',0)==len(rows) else 'COMPLETE_WITH_FAILURES',
            'expected_QA':len(rows),'recorded_QA':len(records),'counts':counts,'coverage_complete':True,
            'failed_QA_must_remain_in_primary_denominator':True,'quality_scored':False,'load_error':load_error,
            'manifest_sha256':sha(args.output/'manifest.json'),'records_jsonl_sha256':sha(record_path),'completed_unix':time.time(),
            'executed_legacy_source_sha256':actual_legacy_sources,'new_parameter_fit':config['candidate']['mode'] not in ('FP16','original','M-V','K-Mu','TIER','BRPE_Lookup8'),'phase_is_confirmatory':False,
            'resource_summary':{'full_run_available':len(resources)==len(rows) and load_error is None,
                'mean_complete_bits':sum(resources)/len(resources) if len(resources)==len(rows) and load_error is None else None,
                'max_complete_bits':max(resources) if len(resources)==len(rows) and load_error is None else None,
                'mean_complete_visual_bytes':sum(totalbytes)/len(totalbytes) if len(resources)==len(rows) and load_error is None else None,
                'max_complete_visual_bytes':max(totalbytes) if len(resources)==len(rows) and load_error is None else None,
                'no_success_subset_resource_primary':True},
            'frozen_score_py_compatible_raw_schema':True,'prediction_cache_SourceRun_admission_not_claimed':True}
        write_new(args.output/'complete.json',complete);print(json.dumps(complete,ensure_ascii=False),flush=True)
    return 0 if complete['status']=='PASS_RAW_COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
