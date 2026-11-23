"""Strict R128 adapter over the unchanged native M1 generation driver.

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
import round11_confirmation_data_v1 as data

SCHEMA = 'trajkv.round11_confirmation.raw.v2'
MANIFEST_SHA = 'acdfb720120d55793a4f6e7dc2b43e5681c0727aff28e8c906ecbc6696578380'
D96_SOURCE_GROUP_SHA = '663076f21fc7df7d00253d06604e9f46da06ffe9d62532cd66d82794c05fa9b4'
COUNTS = dict(textvqa=48, infovqa=48, docvqa=16, chartqa=16)
REFERENCES = ('FP16', 'R7low', 'R7high')
CANDIDATES = primary.MODES + wo.MODES
MODES = REFERENCES + CANDIDATES
PINS = {'code/round10_runner_v1.py': primary.BASE_RUNNER_SHA,
    'code/round11_runner_v2.py': 'ec87b1270bfae11d4a2d98fcafc544fcf42a3c132c72bc30d41fe4176ee73ce1',
    'code/round11_confirmation_data_v1.py': '44297a3c58a493c26a17084a72c4800303b0b70143543217fe0ede3db02548cf',
    'code/round10_data_builder_v1.py': '914dfa258e6fbcfc00c709df6e78cfba6cba5c385db7c51f1e15ebb5d0f59936',
    'code/round11_wo_runner_v1.py': 'c83846fce9e41ef4308160e9e48dae78ba143c0ef9843e387e041ec632907517',
    'code/round11_wo_value_v1.py': 'b0f62f1c8ddbad41336b161891cf0943876a0087e9dcdec63e9bc6e312aaf69e'}
RUNTIME_FILES = tuple(dict.fromkeys((*primary.RUNTIME_FILES, *wo.RUNTIME_FILES,
    'round11_confirmation_runner_v2.py', 'round11_confirmation_data_v1.py', 'round10_data_builder_v1.py')))
_validate_base = frozen.validate_config
_build_base = frozen.build_candidate_owners
_budget_base = frozen.round10_budget_guard
_write_base = frozen.write_new


def require(value, message):
    frozen.require(value, 'R11_CONFIRMATION: ' + message)


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
    require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64,
            'future frozen D96-only selection manifest required; no candidate selected by this adapter')
    require(sha(path) == digest, 'frozen D96 selection manifest changed')
    selection = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    modes = selection.get('selected_modes')
    require(set(selection) == {'schema', 'selected_modes', 'selection_scope', 'confirmation_results_read',
                              'primary_source_group_sha256', 'development_harvest_summary_sha256'},
            'selection receipt must contain only fixed decision provenance, never scores or input/reference paths')
    require(selection.get('schema') == 'trajkv.round11.confirmation_selection.v2'
            and isinstance(modes, list) and len(modes) <= 2 and len(set(modes)) == len(modes)
            and set(modes) <= set(CANDIDATES)
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
    require(expected_role == 'confirmation_holdout', 'development loader semantics cannot be used for R128')
    require(manifest_path is not None and sha(manifest_path) == MANIFEST_SHA, 'exact frozen R128 manifest required')
    return data.load_confirmation_inputs(path, manifest_path, expected_sha256)


def confirmation_identity(config, *, selected_ids=None):
    require(config.get('input_manifest_sha256') == MANIFEST_SHA
            and sha(config['input_manifest_path']) == MANIFEST_SHA, 'R128 manifest source identity differs')
    manifest = json.loads(Path(config['input_manifest_path']).read_text(encoding='utf-8-sig'))
    rows = data.load_confirmation_inputs(config['inputs'], config['input_manifest_path'], config['inputs_sha256'])
    domains = {row['dataset'] for row in rows}
    require(len(domains) == 1, 'one complete R128 domain per cell required')
    dataset = next(iter(domains)); ids = [row['id'] for row in rows]
    require(dataset in COUNTS and config['start'] == 0
            and config['input_count'] == config['count'] == len(rows) == COUNTS[dataset],
            'R128 full-domain48/48/16/16 coverage required; no prefix or source subset')
    require(selected_ids is None or selected_ids == ids, 'actual runner IDs differ from complete frozen R128 domain')
    return {'schema': 'trajkv.round11.confirm128_identity.v1', 'manifest_sha256': MANIFEST_SHA,
            'dataset': dataset, 'input_sha256': config['inputs_sha256'], 'count': len(rows), 'ids': ids,
            'role': 'confirmation_holdout', 'source_disjoint_proof_sha256': manifest['disjoint_proof']['sha256'],
            'raw_source_proof_sha256': manifest['raw_source_proof']['sha256'],
            'source_lock_sha256': manifest['source_lock']['sha256'], 'fresh_inference_required': True}


def candidate_assets(config):
    candidate = config['candidate']
    require(set(candidate) <= {'mode', 'technical_only'} and candidate.get('technical_only', False) is False,
            'no technical free answers or fitted/mixed candidate parameters')
    mode = logical_mode(config); selection = selection_identity(config)
    return {'schema': 'trajkv.round11.confirmation_deployment.v2', 'logical_mode': mode,
            'base_driver_sha256': primary.BASE_RUNNER_SHA, 'fresh_inference': True,
            'new_parameter_fit': False, 'new_frame_encoding_applied': mode in CANDIDATES,
            'selection': selection}, {config['selection_manifest_path']: config['selection_manifest_sha256']}


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('method', {}).get('family') == 'trajkv_round11_confirmation',
            'explicit confirmation raw schema and family required')
    require(config.get('input_role') == config.get('cohort_kind') == 'confirmation_holdout',
            'R128 must retain confirmation identity')
    shadow = deepcopy(config)
    shadow['method']['family'] = 'trajkv_round10_research'
    shadow['input_role'] = shadow['cohort_kind'] = 'development'
    with driver_overrides():
        _validate_base(shadow)
    require(config['expected_source_hashes'] == runtime_source_manifest(), 'complete actual confirmation runtime graph differs')
    confirmation_identity(config)
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
        value.update(confirmation_identity=confirmation_identity(config, selected_ids=value['expected_ids']),
            confirmation_selection=selection_identity(config),
            runtime_process={'PID': os.getpid(), 'parent_PID': os.getppid(), 'source_snapshot_root': str(ROOT)},
            source_driver_contract='R128 confirmation adapter over frozen M1 packed generation; canonical SourceRun admission not claimed',
            base_profile_numeric_fields_unchanged=mode in REFERENCES,
            frozen_base_lineage_numeric_fields_unchanged=True,
            frozen_low43_profile_is_current_deployment_proof=mode == 'R7low',
            round11_confirmation_candidate_assets=value.pop('round10_candidate_assets', None),
            confirmation_quality_status='NOT_SCORED_BY_INFERENCE_DRIVER',
            all_reference_predictions_must_be_fresh=True)
    elif Path(path).name == 'complete.json':
        value = {**value, 'round11_schema': SCHEMA, 'phase_is_confirmatory': True,
                 'canonical_SourceRun_admission_claimed': False, 'new_parameter_fit': False,
                 'confirmation_quality_status': 'NOT_SCORED_BY_INFERENCE_DRIVER',
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
