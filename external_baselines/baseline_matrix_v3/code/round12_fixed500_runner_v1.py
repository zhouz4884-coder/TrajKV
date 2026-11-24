"""Selected static candidate on the previously observed matrix fixed500.

This is a distinct benchmark role, not development or fresh confirmation.
Original native prefill/packed generation is the frozen Round10 core, with
explicit new input admission and INPUT-only old metadata comparisons.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen
import round10_inputs_v1 as original_inputs
import round12_runner_v1 as development
import round12_fixed500_inputs_v1 as data

SCHEMA = data.RUNNER_SCHEMA
ROLE = data.ROLE
MODES = development.MODES
BASE_RUNNER_SHA = development.BASE_RUNNER_SHA
RUNTIME_FILES = tuple(dict.fromkeys((*development.BASE_RUNTIME_FILES,
    'round12_fixed500_runner_v1.py', 'round12_fixed500_inputs_v1.py',
    'round12_static_bank_v1.py', 'round12_static_codec_v1.py', 'round12_static_attention_v1.py')))
_WRITE_BASE = frozen.write_new
_CONFIG = None
_IDENTITY = None

require, sha = data.require, data.sha


def runtime_source_manifest():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA, 'frozen complete native M1 driver changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def selected_algorithm(config, identity):
    p, digest = config.get('selected_algorithm_path'), config.get('selected_algorithm_sha256')
    require(isinstance(p, str) and isinstance(digest, str) and len(digest) == 64 and sha(p) == digest,
            'selected static algorithm path/SHA required')
    locks = identity['selection_receipt'].get('method_asset_locks', [])
    require(any(item.get('sha256') == digest for item in locks), 'algorithm not locked by fixed500 selection receipt')
    spec = data.read(p)
    require(spec.get('schema') == 'trajkv.round12.selected_static_algorithm.v1'
            and spec.get('name') == config['selected_method_name'], 'selected algorithm identity differs')
    normalize = lambda value: {k: v for k, v in value.items() if not k.endswith('_path')}
    require(normalize(spec['candidate']) == normalize(config['candidate']), 'selected candidate numeric fields/asset hashes changed')
    require(spec['variant_profile'] == config['variant_source']['name']
            and spec['variant_source_sha256'] == config['variant_source']['sha256'], 'selected encoding lineage changed')
    actual = runtime_source_manifest()
    for path, digest in spec['algorithm_source_hashes'].items():
        require(actual.get(path) == digest, 'selected algorithm code changed: ' + path)
    return spec


def candidate_assets(config):
    detail, locks = development.candidate_assets(config)
    locks = dict(locks)
    locks[config['selected_algorithm_path']] = config['selected_algorithm_sha256']
    locks[config['selection_receipt_path']] = config['selection_receipt_sha256']
    return {'schema': 'trajkv.round12.fixed500_static_assets.v1', **detail,
            'previously_observed_fixed500': True, 'independent_confirmation_claimed': False}, locks


def validate_config(config):
    global _CONFIG, _IDENTITY
    require(config.get('schema') == SCHEMA, 'fixed500 runner schema differs; use module constant')
    identity = data.validate_config_admission(config)
    data.validate_execution_selection(config, identity)
    require(config.get('method', {}).get('family') == 'trajkv_round12_fixed500_comparison'
            and config['method'].get('information_condition') == 'query_unknown', 'explicit static fixed500 family required')
    require(config.get('position_owner_layout') == 'single_shared_visual_position_map', 'original VP layout required')
    source = config.get('variant_source', {})
    require(source.get('name') == 'r7_joint_k1_gt_cond' and source.get('sha256') == frozen.ORIGINAL_CONFIG_SHA,
            'original low43 cond lineage required')
    require(config.get('expected_asset_sha256') == frozen.PROFILES[source['name']]['assets'], 'original f0/policy/base lineage assets changed')
    require(config.get('candidate', {}).get('mode') in MODES, 'unknown selected static mode')
    for name in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source', 'model_path', 'model_source_lock', 'scope'):
        require(isinstance(config.get(name), str) and bool(config[name]), 'explicit field required: ' + name)
    selected_algorithm(config, identity)
    candidate_assets(config)
    frozen.frozen_variant(config)
    require(all(sha(ROOT / path) == digest for path, digest in frozen.CORE_SOURCES.items()),
            'frozen native M1/cache/ownership source changed')
    require(config.get('expected_source_hashes') == runtime_source_manifest(), 'exact runtime source graph differs')
    _CONFIG, _IDENTITY = config, identity
    return config


def load_runner_inputs(path, manifest_path=None, expected_sha256=None, expected_role=ROLE):
    return data.load_inputs(path, manifest_path, expected_sha256, expected_role)


def build_candidate_owners(cache, vp, variant, config, assets):
    return development.build_candidate_owners(cache, vp, variant, config, assets)


def physical_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    require(mode in MODES and base_profile == 'r7_joint_k1_gt_cond', 'mode/lineage changed')
    stages = [list(x) for x in metadata.get('selected_stages', [])]
    expected = [list(x) if len(x) == 4 else [*x, 0] for x in _CONFIG['variant']['selected_stages'][:43]]
    require(stages == expected if mode != 'R' else stages[:43] == expected, 'concrete low43 ordered coverage changed')
    require(metadata['selected_stage_count'] == len(stages)
            and metadata['returned_owners_unique_tensor_capacity_bytes'] == complete_bytes
            and metadata['round12_static']['mode'] == mode, 'actual owner/factory differs')
    return {'schema': 'trajkv.round12.fixed500_physical_budget.v1', 'N': n, 'mode': mode,
            'complete_visual_owner_bytes': complete_bytes, 'selected_stages': stages,
            'selected_stage_count': len(stages), 'strict_cap_not_inferred_from_mode_name': True,
            'fixed500_comparison_not_new_holdout': True}


def write_new(path, value):
    path = Path(path)
    if path.name == 'manifest.json':
        value = dict(value)
        value.update(runner_schema=SCHEMA, phase_is_confirmatory=False,
            comparison_role=ROLE, previously_observed_fixed500=True, independent_holdout_claimed=False,
            selected_method_name=value['config']['selected_method_name'],
            selection_receipt_sha256=value['config']['selection_receipt_sha256'],
            native_binding_scope='INPUT-only metadata verified across all500 FP16/R7 rows; no logits/output reuse',
            source_driver_contract='new fixed500 admission over frozen native M1 packed-generation core',
            frozen_base_lineage_numeric_fields_unchanged=True, base_profile_numeric_fields_unchanged=False,
            round12_fixed500_assets=value.pop('round10_candidate_assets', None),
            prediction_cache_used=False, canonical_SourceRun_admission_claimed=False)
    elif path.parent.name == 'records':
        value.update(round12_fixed500_schema=SCHEMA, comparison_role=ROLE,
                     previously_observed_fixed500=True, independent_holdout_claimed=False)
        if 'round10_budget_vs_original_R7_low' in value:
            value['round12_fixed500_budget'] = value.pop('round10_budget_vs_original_R7_low')
        if value.get('status') == 'PASS':
            proof = data.validate_native_record(value, _IDENTITY['native_bindings'])
            value['round12_fixed500_native_input_binding'] = proof
            if not proof['matched']:
                value.update(status='INPUT_MISMATCH', error='Fixed500 INPUT metadata differs: ' + json.dumps(proof['mismatches']),
                    untrusted_output_token_ids=value.get('output_token_ids'), untrusted_raw_prediction=value.get('raw_prediction'),
                    output_token_ids=None, raw_prediction=None)
    elif path.name == 'complete.json':
        value = {**value, 'round12_fixed500_schema': SCHEMA, 'phase_is_confirmatory': False,
            'comparison_role': ROLE, 'previously_observed_fixed500': True, 'independent_holdout_claimed': False,
            'canonical_SourceRun_admission_claimed': False, 'new_parameter_fit': False,
            'old_prediction_cache_used_by_new_candidate': False}
    return _WRITE_BASE(path, value)


@contextmanager
def driver_overrides():
    profiles = deepcopy(frozen.PROFILES)
    profiles['r7_joint_k1_gt_cond']['require_43_stages'] = False
    changes = {'SCHEMA': SCHEMA, 'MODES': MODES, 'PROFILES': profiles,
        'candidate_assets': candidate_assets, 'validate_config': validate_config,
        'build_candidate_owners': build_candidate_owners, 'round10_budget_guard': physical_budget_guard,
        'runtime_source_manifest': runtime_source_manifest, 'write_new': write_new}
    saved = {key: getattr(frozen, key) for key in changes}
    loader = original_inputs.load_round10_inputs
    try:
        for key, value in changes.items():
            setattr(frozen, key, value)
        original_inputs.load_round10_inputs = load_runner_inputs
        yield
    finally:
        original_inputs.load_round10_inputs = loader
        for key, value in saved.items():
            setattr(frozen, key, value)


def config_template(dataset=None):
    c = data.config_template()
    if dataset is not None:
        require(dataset in data.PRIMARY_METRIC_IDS, 'unknown template dataset')
        c['evaluation_contract']['primary_metric_id'] = data.PRIMARY_METRIC_IDS[dataset]
    c.update(method={'id': None, 'family': 'trajkv_round12_fixed500_comparison', 'information_condition': 'query_unknown'},
        candidate={'mode': None}, variant_source=None, variant=None, expected_asset_sha256=None,
        legacy_a2=None, legacy_experiments=None, legacy_scripts=None, kivi_source=None,
        position_owner_layout='single_shared_visual_position_map',
        scope='previously observed fixed500 benchmark comparison; not independent confirmation')
    return c


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--validate-config', action='store_true')
    p.add_argument('--print-template', action='store_true')
    p.add_argument('--dataset', choices=tuple(data.PRIMARY_METRIC_IDS))
    a = p.parse_args()
    if a.print_template:
        print(json.dumps(config_template(a.dataset), ensure_ascii=False, indent=2))
        return 0
    require(a.config is not None, 'config required')
    config = validate_config(data.read(a.config))
    if a.validate_config:
        print(json.dumps({'status': 'PASS_FIXED500_STATIC_CONFIG_ONLY', 'domain': _IDENTITY['dataset'],
            'slice_count': len(_IDENTITY['rows']), 'full_cohort': 500, 'GPU_or_model_calls': 0,
            'independent_holdout_claimed': False}, ensure_ascii=False))
        return 0
    require(a.output is not None, 'new output required')
    with driver_overrides():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
