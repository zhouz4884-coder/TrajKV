"""Fixed additional384 data-only adapter over the unchanged native M1 generation driver.

Only data admission, declared phase, selected-mode admission and factory
dispatch differ. Native prefill/full-suffix/position/ownership/generation
operations remain the byte-pinned Round10 driver's actual operations.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen
import round10_inputs_v1 as original_inputs
import round11_runner_v2 as primary
import round11_wo_runner_v1 as wo
import round11_expand384_data_v1 as data

SCHEMA = 'trajkv.round11.expand384.raw.v1'
ORIGINAL_R128_MANIFEST_SHA = 'acdfb720120d55793a4f6e7dc2b43e5681c0727aff28e8c906ecbc6696578380'
ORIGINAL_SELECTION_SHA = '45409e4adedae77660825c1bf1f9a39cf03d8a5dd14eae34374705b3b823f915'
DESIGN_SHA = 'ebe818bbc67b1d189b4e27b15e2fdd4af59c0a7e5633c5b1125add1ccfac0c19'
DATA_MANIFEST_NAME = 'EXPAND384_INPUT_MANIFEST_v1.json'
MANIFEST_SHA = 'f7a37a254d59135c7f2e4364fd8a1a271e65f681620d7482e0b3e7e9077d1414'
D96_SOURCE_GROUP_SHA = '663076f21fc7df7d00253d06604e9f46da06ffe9d62532cd66d82794c05fa9b4'
COUNTS = dict(textvqa=144, infovqa=144, docvqa=48, chartqa=48)
REFERENCES = ('FP16', 'R7low', 'R7high')
CANDIDATES = ('JointV', 'JointSpatialWO')
MODES = REFERENCES + CANDIDATES
PINS = {'code/round10_runner_v1.py': primary.BASE_RUNNER_SHA,
    'code/round11_runner_v2.py': 'ec87b1270bfae11d4a2d98fcafc544fcf42a3c132c72bc30d41fe4176ee73ce1',
    'code/round11_confirmation_data_v1.py': '44297a3c58a493c26a17084a72c4800303b0b70143543217fe0ede3db02548cf',
    'code/round10_data_builder_v1.py': '914dfa258e6fbcfc00c709df6e78cfba6cba5c385db7c51f1e15ebb5d0f59936',
    'code/round11_expand384_data_v1.py': '1fd95c1515237617c0a1eb26fa3d522d224d7b6218defaca2f125354c70b8778',
    'code/round11_wo_runner_v1.py': 'c83846fce9e41ef4308160e9e48dae78ba143c0ef9843e387e041ec632907517',
    'code/round11_wo_value_v1.py': 'b0f62f1c8ddbad41336b161891cf0943876a0087e9dcdec63e9bc6e312aaf69e'}
RUNTIME_FILES = tuple(dict.fromkeys((*primary.RUNTIME_FILES, *wo.RUNTIME_FILES,
    'round11_expand_runner_v1.py', 'round11_expand384_data_v1.py', 'round11_confirmation_data_v1.py', 'round10_data_builder_v1.py')))
_validate_base = frozen.validate_config
_build_base = frozen.build_candidate_owners
_budget_base = frozen.round10_budget_guard
_write_base = frozen.write_new


def require(value, message):
    frozen.require(value, 'R11_EXPAND384: ' + message)


def sha(path):
    return frozen.sha(path)


def runtime_source_manifest():
    require(all(sha(ROOT / role) == digest for role, digest in PINS.items()), 'frozen executed dependency changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def logical_mode(config):
    mode, profile = config['candidate']['mode'], config['variant_source']['name']
    if mode == 'original':
        require(profile in frozen.PROFILES, 'unknown original R7 profile')
        return 'R7low' if profile == 'r7_joint_k1_gt_cond' else 'R7high'
    require(mode in ('FP16', *CANDIDATES) and profile == 'r7_joint_k1_gt_cond',
            'FP16 and new candidates inherit exact original low43 profile')
    return mode


def selection_identity(config):
    path, digest = config.get('selection_manifest_path'), config.get('selection_manifest_sha256')
    require(isinstance(path, str) and digest == ORIGINAL_SELECTION_SHA,
            'additional384 must retain the exact already-frozen Root selection; no reselection')
    require(sha(path) == digest, 'frozen D96 selection manifest changed')
    selection = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    modes = selection.get('selected_modes')
    require(set(selection) == {'schema', 'selected_modes', 'selection_scope', 'confirmation_results_read',
                              'primary_source_group_sha256', 'development_harvest_summary_sha256'},
            'selection receipt must contain only fixed decision provenance, never scores or input/reference paths')
    require(selection.get('schema') == 'trajkv.round11.confirmation_selection.v2'
            and modes == list(CANDIDATES)
            and selection.get('selection_scope') == 'D96_only_before_R128'
            and selection.get('confirmation_results_read') is False
            and selection.get('primary_source_group_sha256') == D96_SOURCE_GROUP_SHA
            and isinstance(selection.get('development_harvest_summary_sha256'), dict)
            and set(selection['development_harvest_summary_sha256']) == {'primary', 'wo'}
            and all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
                    and value != '0' * 64 for value in selection['development_harvest_summary_sha256'].values()),
            'selection must freeze at most2 modes before reading any R128 outcome')
    require(not (set(selection) & frozen.FORBIDDEN), 'selection receipt must contain no reference text or scores')
    mode = logical_mode(config)
    require(mode in REFERENCES or mode in modes, 'candidate not selected by the frozen D96 decision')
    return {'manifest_sha256': digest, 'selected_modes': modes,
            'selection_scope': selection['selection_scope'],
            'development_harvest_summary_sha256': selection['development_harvest_summary_sha256']}


def load_runner_inputs(path, manifest_path=None, expected_sha256=None, expected_role=None):
    require(expected_role == 'confirmation_expansion_holdout', 'development or old128 loader semantics cannot be used for additional384')
    require(manifest_path is not None and Path(manifest_path).name == DATA_MANIFEST_NAME
            and sha(manifest_path) == MANIFEST_SHA, 'exact frozen additional384 manifest required')
    return data.load_expand_inputs(path, manifest_path, expected_sha256, expected_role='confirmation_expansion_holdout')


def design_identity(config):
    path = config.get('expansion_design_path')
    require(isinstance(path, str) and config.get('expansion_design_sha256') == DESIGN_SHA
            and sha(path) == DESIGN_SHA, 'fixed user-authorized expansion design required')
    design = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(design.get('schema') == 'trajkv.round11.expand384_design.v1'
            and design.get('original_selection_sha256') == ORIGINAL_SELECTION_SHA
            and design.get('original_R128_manifest_sha256') == ORIGINAL_R128_MANIFEST_SHA
            and design.get('selected_modes') == list(CANDIDATES)
            and design.get('references') == list(REFERENCES)
            and design.get('additional_counts') == COUNTS and design.get('additional_sources') == 384
            and design.get('additional_fresh_free_QA') == 1920
            and design.get('interim_R128_observed') is True
            and design.get('freeze_methods_and_size_before_additional_outcomes') is True
            and design.get('no_stopping_on_p_or_favorable_scores') is True
            and design.get('new_method_parameters_or_families') is False,
            'expansion cannot hide interim observation, change methods, or stop on outcomes')
    return {'schema': design['schema'], 'sha256': DESIGN_SHA, 'additional_sources': 384,
            'additional_new_QA': 1920, 'original_selection_sha256': ORIGINAL_SELECTION_SHA,
            'original_R128_manifest_sha256': ORIGINAL_R128_MANIFEST_SHA,
            'original_R128_partially_observed_before_design': True}


def expansion_identity(config, *, selected_ids=None):
    path = config['input_manifest_path']; digest = config.get('input_manifest_sha256')
    require(Path(path).name == DATA_MANIFEST_NAME and digest == MANIFEST_SHA
            and sha(path) == digest, 'actual newly frozen384 manifest SHA required, never a guessed placeholder')
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(manifest.get('schema') == 'trajkv.round11.expand384_inputs.v1'
            and manifest.get('counts') == COUNTS and manifest.get('total_count') == 384
            and manifest.get('answers_in_encoder_inputs') is False
            and manifest.get('original_selection_sha256') == ORIGINAL_SELECTION_SHA
            and manifest.get('base_R128_manifest_sha256') == ORIGINAL_R128_MANIFEST_SHA
            and manifest.get('expansion_design_sha256') == DESIGN_SHA,
            'new384 source manifest must bind original128, fixed methods and expansion design')
    require(manifest.get('builder_source_sha256') == PINS['code/round11_expand384_data_v1.py'],
            'manifest was not built by the frozen additional384 data implementation')
    values = data.load_expand_inputs(config['inputs'], path, config['inputs_sha256'],
                                    expected_role='confirmation_expansion_holdout')
    domains = {row['dataset'] for row in values}
    require(len(domains) == 1, 'one complete additional384 domain per actual child')
    dataset = next(iter(domains)); ids = [row['id'] for row in values]
    require(dataset in COUNTS and config['start'] == 0
            and config['count'] == config['input_count'] == len(values) == COUNTS[dataset],
            'actual144/144/48/48 full-domain coverage required; partial64 cannot pass')
    require(selected_ids is None or selected_ids == ids, 'actual full144/48 runner IDs differ from frozen source file')
    return {'schema': 'trajkv.round11.expand384_identity.v1', 'manifest_sha256': digest,
            'dataset': dataset, 'input_sha256': config['inputs_sha256'], 'count': len(values), 'ids': ids,
            'role': 'confirmation_expansion_holdout', 'source_disjoint_proof_sha256': manifest['disjoint_proof']['sha256'],
            'raw_source_proof_sha256': manifest['raw_source_proof']['sha256'],
            'source_lock_sha256': manifest['source_lock']['sha256'], 'fresh_inference_required': True,
            'separate_from_original128_and_combined512_derived_views': True}


def candidate_assets(config):
    candidate = config['candidate']
    require(set(candidate) <= {'mode', 'technical_only'} and candidate.get('technical_only', False) is False,
            'no technical free answers or fitted/mixed candidate parameters')
    mode = logical_mode(config); selection = selection_identity(config)
    return {'schema': 'trajkv.round11.expand384_deployment.v1', 'logical_mode': mode,
            'base_driver_sha256': primary.BASE_RUNNER_SHA, 'fresh_inference': True,
            'new_parameter_fit': False, 'new_frame_encoding_applied': mode in CANDIDATES,
            'selection': selection}, {config['selection_manifest_path']: config['selection_manifest_sha256'],
                config['expansion_design_path']: config['expansion_design_sha256']}


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('method', {}).get('family') == 'trajkv_round11_expand384',
            'explicit confirmation raw schema and family required')
    require(config.get('input_role') == config.get('cohort_kind') == 'confirmation_expansion_holdout',
            'additional384 must retain its separate expansion identity')
    # Exact full-domain source admission precedes the compatibility-only
    # legacy structure check. A partial64 slice never reaches that check.
    expansion_identity(config)
    design_identity(config)
    shadow = deepcopy(config)
    shadow['method']['family'] = 'trajkv_round10_research'
    shadow['input_role'] = shadow['cohort_kind'] = 'development'
    # Only the legacy QA-slice structural cap is adapted, never answer length
    # or model/codec parameters. The actual returned config remains144/48.
    actual_count = config['count']
    shadow['count'] = min(actual_count, 64)
    with driver_overrides():
        _validate_base(shadow)
    require(config['expected_source_hashes'] == runtime_source_manifest(), 'complete actual confirmation runtime graph differs')
    require(config['count'] == actual_count, 'structural shadow modified actual cohort count')
    candidate_assets(config)
    return config


def build_candidate_owners(cache, vp, variant, config, assets):
    if logical_mode(config) in wo.MODES:
        return wo.build_candidate_owners(cache, vp, variant, config, assets)
    if logical_mode(config) in primary.MODES:
        return primary.build_candidate_owners(cache, vp, variant, config, assets)
    return _build_base(cache, vp, variant, config, assets)


def physical_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    if mode in wo.MODES:
        return wo.round11_budget_guard(mode, n, complete_bytes, metadata, base_profile)
    if mode in primary.MODES:
        return primary.round11_budget_guard(mode, n, complete_bytes, metadata, base_profile)
    return _budget_base(mode, n, complete_bytes, metadata, base_profile)


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value = dict(value); config = value['config']; mode = logical_mode(config)
        value.update(expansion_identity=expansion_identity(config, selected_ids=value['expected_ids']),
            expansion_design=design_identity(config),
            expansion_selection=selection_identity(config),
            runtime_process={'PID': os.getpid(), 'parent_PID': os.getppid(), 'source_snapshot_root': str(ROOT)},
            source_driver_contract='fixed additional384 data-only adapter over frozen M1 packed generation; actual144/48 QA count, no method changes',
            legacy_structural_validator_QA_count_shadow={'legacy_cap': 64, 'actual_count': config['count'],
                'actual_config_and_main_loop_preserved': True, 'max_new_tokens_unchanged': 64},
            base_profile_numeric_fields_unchanged=mode in REFERENCES,
            frozen_base_lineage_numeric_fields_unchanged=True,
            frozen_low43_profile_is_current_deployment_proof=mode == 'R7low',
            round11_expansion_candidate_assets=value.pop('round10_candidate_assets', None),
            expansion_quality_status='NOT_SCORED_BY_INFERENCE_DRIVER',
            all_reference_predictions_must_be_fresh=True)
    elif Path(path).name == 'complete.json':
        value = {**value, 'round11_schema': SCHEMA, 'phase_is_confirmatory': True,
                 'canonical_SourceRun_admission_claimed': False, 'new_parameter_fit': False,
                 'expansion_quality_status': 'NOT_SCORED_BY_INFERENCE_DRIVER',
                 'all_reference_predictions_fresh': True}
    return _write_base(path, value)


@contextmanager
def driver_overrides():
    require(sha(ROOT / 'code/round10_runner_v1.py') == primary.BASE_RUNNER_SHA, 'frozen generation driver changed')
    changes = {'SCHEMA': SCHEMA, 'MODES': ('FP16', 'original', *CANDIDATES), 'RUNTIME_FILES': RUNTIME_FILES,
        'candidate_assets': candidate_assets, 'validate_config': validate_config,
        'build_candidate_owners': build_candidate_owners, 'round10_budget_guard': physical_budget_guard,
        'runtime_source_manifest': runtime_source_manifest, 'write_new': write_new}
    saved = {name: getattr(frozen, name) for name in changes}
    loader = original_inputs.load_round10_inputs
    try:
        for name, value in changes.items():
            setattr(frozen, name, value)
        original_inputs.load_round10_inputs = load_runner_inputs
        yield
    finally:
        original_inputs.load_round10_inputs = loader
        for name, value in saved.items():
            setattr(frozen, name, value)


def main():
    with driver_overrides(), wo.capture_loaded_model():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
