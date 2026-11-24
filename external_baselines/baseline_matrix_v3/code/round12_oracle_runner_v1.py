"""Round12 above-budget mechanism-oracle full-model free-generation driver.

Frozen Round10 controls native prefill, question suffix and source release.
No reference answers are opened. Oracle native visual K is explicitly charged.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round12_runner_v1 as base

SCHEMA = 'trajkv.round12_oracle.raw.v1'
MODES = ('mass-only', 'route-only')
frozen = base.frozen


def require(value, message):
    base.require(value, 'ORACLE: ' + message)


def runtime_source_manifest(config=None):
    config = config or base._CONFIG or {}
    require(base.sha(ROOT / 'code/round10_runner_v1.py') == base.BASE_RUNNER_SHA, 'frozen native driver changed')
    additions = config.get('additional_runtime_files', [])
    require(isinstance(additions, list) and all(Path(x).name == x and x.endswith('.py') for x in additions), 'runtime filenames')
    files = tuple(dict.fromkeys((*base.BASE_RUNTIME_FILES, 'round12_oracle_runner_v1.py',
        'round12_oracle_bank_v1.py', 'round12_oracle_attention_v1.py', *additions)))
    return {'code/' + name: base.sha(ROOT / 'code' / name) for name in files}


def candidate_assets(config):
    candidate = config['candidate']
    require(candidate.get('mode') in MODES and set(candidate) <= {'mode', 'technical_only'}, 'declared oracle mode only')
    require(candidate.get('technical_only', False) is False, 'QA0 checks must use separate driver')
    method = config['method']
    require(method.get('mechanism_oracle_only') is True and method.get('production_compression_method') is False,
            'must explicitly exclude oracle from production method table')
    return {'schema': 'trajkv.round12.oracle_assets.v1', 'mode': candidate['mode'],
            'mechanism_oracle_only': True, 'native_K_reference_is_charged': True,
            'GT_or_future_answer_used': False}, {}


def build_candidate_owners(cache, vp, variant, config, assets):
    from round12_oracle_bank_v1 import build_round12_oracle_owners
    return build_round12_oracle_owners(cache, vp, variant, candidate=config['candidate'],
        legacy_a2=config['legacy_a2'], legacy_experiments=config['legacy_experiments'],
        legacy_scripts=config['legacy_scripts'], kivi_source=config['kivi_source'],
        expected_asset_sha256=assets)


def validate_config(config):
    base.validate_config(config)
    manifest = base.data.read(config['input_manifest_path'])
    entry = manifest['input_files'][Path(config['inputs']).name]
    require(entry.get('panel') == 'oracle24', 'only the frozen Info16/Doc8 oracle panel is admitted')
    domain = next(iter(entry['domains']))
    require(domain in ('infovqa', 'docvqa') and entry['count'] == {'infovqa': 16, 'docvqa': 8}[domain],
            'oracle domain/count differs')
    return config


def physical_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    require(mode in MODES and base_profile == 'r7_joint_k1_gt_cond', 'oracle mode/lineage')
    selected = [list(x) for x in metadata['selected_stages']]
    original = next(iter(base._BINDINGS.values()))['low_selected_stages']
    require(selected == original and len(selected) == 43 and metadata['selected_stage_count'] == 43,
            'original concrete low43 coverage changed')
    detail = metadata.get('round12_oracle', {})
    expected = (4590 * n + 89733) + 28672 * n + 2
    require(complete_bytes == expected == metadata['returned_owners_unique_tensor_capacity_bytes']
            == detail.get('actual_complete_visual_owner_bytes'), 'native-K/reference bytes uncharged')
    require(detail.get('mode') == mode and detail.get('mechanism_oracle_only') is True
            and detail.get('production_compression_method') is False
            and detail.get('native_K_reference_bytes') == 28672 * n
            and detail.get('GT_or_future_answer_used') is False, 'oracle declaration missing')
    return {'schema': 'trajkv.round12.oracle_physical_budget.v1', 'mode': mode,
            'N': n, 'actual_complete_visual_bytes': complete_bytes,
            'original_low43_bytes': 4590 * n + 89733, 'extra_native_visual_K_bytes': 28672 * n,
            'mode_descriptor_bytes': 2, 'selected_stages': selected,
            'above_budget_mechanism_oracle_only': True, 'production_low_bit_claimed': False,
            'full_native_K_reference_charged_as_visual_payload': True}


def write_new(path, value):
    path = Path(path)
    if path.name == 'manifest.json':
        value = dict(value)
        value.update(runner_schema=SCHEMA,
            source_driver_contract='Round12 native-K mass/routing oracle over frozen whole-model generator',
            runtime_process={'PID': os.getpid(), 'parent_PID': os.getppid(), 'source_snapshot_root': str(ROOT)},
            phase_is_confirmatory=False, mechanism_oracle_only=True,
            production_compression_method=False, native_K_reference_is_charged=True,
            round12_oracle_assets=value.pop('round10_candidate_assets', None))
    elif path.parent.name == 'records':
        value['round12_schema'] = SCHEMA
        value['mechanism_oracle_only'] = True
        value['production_compression_method'] = False
        if 'round10_budget_vs_original_R7_low' in value:
            value['round12_oracle_physical_budget'] = value.pop('round10_budget_vs_original_R7_low')
        if value.get('status') == 'PASS':
            proof = base.validate_native_record(value)
            value['round12_native_input_binding'] = proof
            if not proof['matched']:
                value['status'] = 'INPUT_MISMATCH'
                value['error'] = 'Original native input binding differs'
                value['untrusted_generated_output_token_ids'] = value.get('output_token_ids')
                value['untrusted_raw_prediction'] = value.get('raw_prediction')
                value['output_token_ids'] = value['raw_prediction'] = None
    elif path.name == 'complete.json':
        value = {**value, 'round12_schema': SCHEMA, 'mechanism_oracle_only': True,
                 'production_compression_method': False,
                 'new_free_generation_answer_events': value['recorded_QA'],
                 'no_additive_trajectory_causal_share_claimed': True}
    return base._write_base(path, value)


@contextmanager
def driver_overrides():
    # Base validation remains unchanged apart from the explicit oracle schema,
    # modes and dependency graph. It still admits only no-answer R12 inputs.
    replacements = {'SCHEMA': SCHEMA, 'MODES': MODES,
                    'candidate_assets': candidate_assets, 'runtime_source_manifest': runtime_source_manifest}
    saved_base = {key: getattr(base, key) for key in replacements}
    changes = {'SCHEMA': SCHEMA, 'MODES': MODES, 'candidate_assets': candidate_assets,
               'validate_config': validate_config, 'build_candidate_owners': build_candidate_owners,
               'round10_budget_guard': physical_budget_guard,
               'runtime_source_manifest': runtime_source_manifest, 'write_new': write_new}
    saved_frozen = {key: getattr(frozen, key) for key in changes}
    loader = base.old_inputs.load_round10_inputs
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        for key, value in changes.items():
            setattr(frozen, key, value)
        base.old_inputs.load_round10_inputs = base.load_runner_inputs
        yield
    finally:
        base.old_inputs.load_round10_inputs = loader
        for key, value in saved_frozen.items():
            setattr(frozen, key, value)
        for key, value in saved_base.items():
            setattr(base, key, value)


def main():
    with driver_overrides():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
