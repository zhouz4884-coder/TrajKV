"""Round12 static representations over the byte-frozen native M1 driver.

Only data admission, factory dispatch and the physical budget guard are new.
Native question-free prefill, source release, original positions/full suffix,
actual packed attention and streamed free generation remain the Round10 code.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen
import round10_inputs_v1 as old_inputs
import round12_inputs_v1 as data

SCHEMA = 'trajkv.round12_research.raw.v1'
BASE_RUNNER_SHA = '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96'
MODES = ('A', 'V2R', 'B', 'C', 'D', 'G8', 'R')
BASE_RUNTIME_FILES = tuple(dict.fromkeys((*frozen.RUNTIME_FILES, 'round12_runner_v1.py', 'round12_inputs_v1.py')))
_write_base = frozen.write_new
_CONFIG = None
_BINDINGS = {}


def require(value, message):
    frozen.require(value, 'ROUND12: ' + message)


sha = frozen.sha


def runtime_source_manifest(config=None):
    config = config or _CONFIG or {}
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen native generation driver changed')
    additions = config.get('additional_runtime_files')
    if additions is None:
        additions = [p.name for p in sorted((ROOT / 'code').glob('round12_static*_v1.py'))]
    require(isinstance(additions, list) and 'round12_static_bank_v1.py' in additions,
            'INPUT_CONTRACT: explicit implemented static dependency graph required')
    require(all(isinstance(name, str) and Path(name).name == name and name.endswith('.py')
                for name in additions), 'INPUT_CONTRACT: runtime files must be local code basenames')
    files = tuple(dict.fromkeys((*BASE_RUNTIME_FILES, *additions)))
    return {'code/' + name: sha(ROOT / 'code' / name) for name in files}


def candidate_assets(config):
    candidate = config['candidate']
    mode = candidate.get('mode')
    require(mode in MODES, 'INPUT_CONTRACT: unknown static mode')
    allowed = {'mode', 'technical_only', 'key_book_path', 'key_book_sha256',
               'ratio_path', 'ratio_sha256', 'ratio', 'format_version'}
    require(set(candidate) <= allowed, 'INPUT_CONTRACT: unknown candidate fields')
    require(candidate.get('technical_only', False) is False,
            'INPUT_CONTRACT: technical checks are not task-quality inference')
    locks = {}
    for stem in ('key_book', 'ratio'):
        path, expected = candidate.get(stem + '_path'), candidate.get(stem + '_sha256')
        require((path is None) == (expected is None), 'INPUT_CONTRACT: asset path/SHA must appear together')
        if path is not None:
            require(isinstance(path, str) and isinstance(expected, str) and len(expected) == 64
                    and sha(path) == expected, 'INPUT_MISMATCH: candidate asset changed: ' + stem)
            locks[path] = expected
    if mode in ('A', 'B', 'C', 'D', 'G8'):
        require(candidate.get('key_book_path') is not None, 'INPUT_CONTRACT: explicit key book required')
    if mode == 'V2R':
        require(candidate.get('ratio_path') is not None,
                'INPUT_CONTRACT: task runs require frozen calibration ratio asset')
    return {'schema': 'trajkv.round12.static_candidate_assets.v1', 'mode': mode,
            'asset_locks': locks, 'answers_or_future_question_used_by_encoder': False,
            'parameter_fit_in_inference_runner': False}, locks


def load_runner_inputs(path, manifest_path=None, expected_sha256=None, expected_role=None):
    require(manifest_path is not None, 'INPUT_CONTRACT: explicit no-answer manifest required')
    values = data.load_round12_inputs(path, manifest_path, expected_sha256, expected_role)
    manifest = data.read(manifest_path)
    _BINDINGS.update(manifest['native_bindings'])
    return values


def validate_config(config):
    global _CONFIG
    require(config.get('schema') == SCHEMA, 'INPUT_CONTRACT: Round12 schema required')
    require(not (set(config) & frozen.FORBIDDEN), 'INPUT_CONTRACT: inference config contains references/labels')
    require(config.get('model_id') == 'M1' and config.get('seed') == 0, 'INPUT_CONTRACT: M1 seed0 required')
    require(config.get('question_suffix') == frozen.SUFFIX, 'INPUT_CONTRACT: native suffix differs')
    require(config.get('preprocessing') == {'max_patches': 8}, 'INPUT_CONTRACT: fixed cap8 required')
    require(config.get('max_new_tokens') == 64, 'INPUT_CONTRACT: max_new_tokens64 required')
    require(config.get('position_owner_layout') == 'single_shared_visual_position_map', 'INPUT_CONTRACT: VP layout differs')
    require(config.get('evaluation_contract', {}).get('version') == 'matrix_v3_metrics_v1', 'INPUT_CONTRACT: metric version differs')
    require(config.get('input_role') == data.ROLE and config.get('cohort_kind') == 'development',
            'INPUT_CONTRACT: explicit already-observed development role required')
    require(type(config.get('start')) is int and config['start'] >= 0
            and type(config.get('count')) is int and 0 < config['count'] <= 64,
            'INPUT_CONTRACT: explicit count1..64 slice required')
    require(type(config.get('input_count')) is int and config['start'] + config['count'] <= config['input_count'],
            'INPUT_CONTRACT: incomplete input slice')
    require(isinstance(config.get('input_manifest_sha256'), str)
            and sha(config['input_manifest_path']) == config['input_manifest_sha256'], 'INPUT_MISMATCH: manifest SHA differs')
    values = load_runner_inputs(config['inputs'], config['input_manifest_path'], config['inputs_sha256'], config['input_role'])
    require(len(values) == config['input_count'], 'INPUT_MISMATCH: input_count differs')
    chosen = values[config['start']:config['start'] + config['count']]
    require(len({r['dataset'] for r in chosen}) == 1, 'INPUT_CONTRACT: one dataset per actual run')
    method = config.get('method', {})
    require(method.get('family') == 'trajkv_round12_research' and method.get('information_condition') == 'query_unknown',
            'INPUT_CONTRACT: explicit static research family/query_unknown required')
    source = config.get('variant_source', {})
    require(source.get('name') == 'r7_joint_k1_gt_cond' and source.get('sha256') == frozen.ORIGINAL_CONFIG_SHA,
            'INPUT_CONTRACT: original low43 encoding lineage required')
    require(config.get('expected_asset_sha256') == frozen.PROFILES[source['name']]['assets'],
            'INPUT_CONTRACT: original low43 factors/policy/assets differ')
    require(isinstance(config.get('scope'), str) and bool(config['scope']), 'INPUT_CONTRACT: scope required')
    for name in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source'):
        require(isinstance(config.get(name), str) and bool(config[name]), 'INPUT_CONTRACT: missing dependency ' + name)
    candidate_assets(config)
    _CONFIG = config
    actual = runtime_source_manifest(config)
    require(config.get('expected_source_hashes') == actual, 'INPUT_MISMATCH: exact runtime source graph differs')
    return config


def build_candidate_owners(cache, vp, variant, config, assets):
    from round12_static_bank_v1 import build_round12_owners
    return build_round12_owners(cache, vp, variant, candidate=config['candidate'],
        legacy_a2=config['legacy_a2'], legacy_experiments=config['legacy_experiments'],
        legacy_scripts=config['legacy_scripts'], kivi_source=config['kivi_source'],
        expected_asset_sha256=assets)


def physical_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    require(mode in MODES and base_profile == 'r7_joint_k1_gt_cond', 'OWNER_AUDIT_FAILED: mode/lineage differs')
    stages = metadata.get('selected_stages')
    require(isinstance(stages, list) and len(stages) == metadata.get('selected_stage_count'),
            'OWNER_AUDIT_FAILED: actual stage manifest missing')
    stages = [list(stage) for stage in stages]
    expected_stages = next(iter(_BINDINGS.values()))['low_selected_stages']
    require(all(b['low_selected_stages'] == expected_stages for b in _BINDINGS.values())
            and len(set(map(tuple, expected_stages))) == 43,
            'INPUT_CONTRACT: frozen low43 ordered source identities disagree')
    if mode != 'R':
        require(stages == expected_stages,
                'OWNER_AUDIT_FAILED: exact original K23/V20 coverage differs')
    else:
        require(len(stages) >= 43 and stages[:43] == expected_stages,
                'OWNER_AUDIT_FAILED: R does not preserve original43')
    require(metadata.get('returned_owners_unique_tensor_capacity_bytes') == complete_bytes,
            'OWNER_AUDIT_FAILED: factory/Hybrid actual unique bytes disagree')
    detail = metadata.get('round12_static', {})
    require(detail.get('mode') == mode, 'OWNER_AUDIT_FAILED: actual static factory mode missing')
    baseline = 4590 * n + 89733
    return {'schema': 'trajkv.round12.physical_budget.v1', 'mode': mode, 'N': n,
            'actual_complete_visual_bytes': complete_bytes, 'original_low43_bytes': baseline,
            'delta_vs_original_low43_bytes': complete_bytes - baseline,
            'selected_stages': stages, 'selected_stage_count': len(stages),
            'fixed_concrete_low43_coverage': mode != 'R',
            'raw_physical_storage_not_nominal_bit_label': True,
            'strict_cap_not_inferred_from_mode_name': True,
            'timing_is_instrumented_quality_run_not_controlled_speed': True}


def validate_native_record(record):
    expected = _BINDINGS[record['id']]
    mismatches = {key: {'expected': expected[key], 'actual': record.get(key)}
                  for key in data.NATIVE_FIELDS if record.get(key) != expected[key]}
    if record.get('raw_question_sha256') != expected['raw_question_sha256']:
        mismatches['raw_question_sha256'] = {'expected': expected['raw_question_sha256'],
                                            'actual': record.get('raw_question_sha256')}
    if record.get('preprocessing') != expected['preprocessing']:
        mismatches['preprocessing'] = {'expected': expected['preprocessing'], 'actual': record.get('preprocessing')}
    return {'matched': not mismatches, 'mismatches': mismatches,
            'reference': 'previously-observed exact R11 native input identity, not output tokens/scores'}


def write_new(path, value):
    path = Path(path)
    if path.name == 'manifest.json':
        value = dict(value)
        value.update(runner_schema=SCHEMA, source_driver_contract='Round12 static factory over frozen complete native M1 driver',
            runtime_process={'PID': os.getpid(), 'parent_PID': os.getppid(), 'source_snapshot_root': str(ROOT)},
            source_binding='ROUND12_INPUT_MANIFEST_v1.json; development reused from observed R11 sources',
            phase_is_confirmatory=False, base_profile_numeric_fields_unchanged=False,
            frozen_base_lineage_numeric_fields_unchanged=True,
            round12_candidate_assets=value.pop('round10_candidate_assets', None),
            new_static_representation=True, parameter_fit_in_runner=False)
    elif path.parent.name == 'records':
        # Mutate the original record as well: the frozen driver writes this same
        # object into records.jsonl immediately after the immutable row file.
        value['round12_schema'] = SCHEMA
        if 'round10_budget_vs_original_R7_low' in value:
            value['round12_physical_budget'] = value.pop('round10_budget_vs_original_R7_low')
        if value.get('status') == 'PASS':
            proof = validate_native_record(value)
            value['round12_native_input_binding'] = proof
            if not proof['matched']:
                value['status'] = 'INPUT_MISMATCH'
                value['error'] = 'Frozen native input identity mismatch: ' + json.dumps(proof['mismatches'])
                value['untrusted_generated_output_token_ids'] = value.get('output_token_ids')
                value['untrusted_raw_prediction'] = value.get('raw_prediction')
                value['output_token_ids'] = value['raw_prediction'] = None
    elif path.name == 'complete.json':
        value = {**value, 'round12_schema': SCHEMA, 'phase_is_confirmatory': False,
                 'new_parameter_fit': False, 'new_static_representation': True,
                 'new_free_generation_answer_events': value['recorded_QA'],
                 'reused_reference_views_are_not_new_inference': True,
                 'canonical_SourceRun_admission_claimed': False}
    return _write_base(path, value)


@contextmanager
def driver_overrides():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA, 'frozen driver changed')
    # R needs more than43 stages; every mode is checked by the stronger actual
    # coverage guard above, rather than the old profile's stage-count shortcut.
    profiles = deepcopy(frozen.PROFILES)
    profiles['r7_joint_k1_gt_cond']['require_43_stages'] = False
    changes = {'SCHEMA': SCHEMA, 'MODES': MODES, 'PROFILES': profiles,
               'candidate_assets': candidate_assets, 'validate_config': validate_config,
               'build_candidate_owners': build_candidate_owners,
               'round10_budget_guard': physical_budget_guard,
               'runtime_source_manifest': runtime_source_manifest, 'write_new': write_new}
    saved = {key: getattr(frozen, key) for key in changes}
    loader = old_inputs.load_round10_inputs
    try:
        for key, value in changes.items():
            setattr(frozen, key, value)
        old_inputs.load_round10_inputs = load_runner_inputs
        yield
    finally:
        old_inputs.load_round10_inputs = loader
        for key, value in saved.items():
            setattr(frozen, key, value)


def main():
    with driver_overrides():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
