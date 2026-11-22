"""Round10 middle-rate pair using the frozen real M1 generation driver.

The imported driver owns native prefill, question-suffix recomputation, actual
positions, generation, weak-reference release checks and raw evidence. Only
the declared candidate factory, source inventory and physical budget guard
are replaced. The frozen low43 variant is an encoding lineage, not a claim
that either middle deployment still has the original V grouping/stage count.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen

BASE_RUNNER_SHA = '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96'
MODES = ('V128-R43', 'AddOldOrder')
RUNTIME_FILES = tuple(dict.fromkeys((*frozen.RUNTIME_FILES,
    'round10_middle_runner_v1.py', 'round10_middle_bank_v1.py')))
_validate_base = frozen.validate_config
_write_base = frozen.write_new


def require(value, message):
    frozen.require(value, message)


def sha(path):
    return frozen.sha(path)


def runtime_source_manifest():
    """CPU-only executed source inventory, including the imported driver."""
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen generation driver changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def candidate_assets(config):
    candidate = config['candidate']
    require(candidate.get('mode') in MODES, 'INPUT_CONTRACT: middle pair mode required')
    require(not (set(candidate) - {'mode', 'technical_only'}),
            'INPUT_CONTRACT: middle pair has no question-selected parameters')
    technical = candidate.get('technical_only', False)
    require(type(technical) is bool and (not technical or config['count'] <= 4),
            'INPUT_CONTRACT: technical runs are at most four answers')
    return {'mode': candidate['mode'], 'new_parameter_fit': False,
            'base_lineage': 'original_r7_joint_k1_gt_cond_K1_V1_43',
            'deployment_change': ('V_group128_original43_residual_reencode'
                                  if candidate['mode'] == 'V128-R43' else
                                  'original_grouping_original43_plus_priority_order_residuals'),
            'frozen_low43_profile_is_current_deployment_proof': False}, {}


def validate_config(config):
    # The base validator still checks all original M1/input/protocol/assets
    # restrictions. Its global mode/graph declarations are explicitly updated
    # below; no actual variant numeric fields are rebound or modified here.
    if frozen.candidate_assets is not candidate_assets:
        install_driver_overrides()
    result = _validate_base(config)
    require(config['variant_source']['name'] == 'r7_joint_k1_gt_cond',
            'INPUT_CONTRACT: middle pair starts from original low43 only')
    candidate_assets(config)
    return result


def middle_budget(n, mode):
    require(type(n) is int and n > 0, 'OWNER_AUDIT_FAILED: positive actual N required')
    require(mode in MODES, 'OWNER_AUDIT_FAILED: middle mode differs')
    b0 = 4590 * n + 89733
    unit = 18 * n
    q = (448 * n - 57344) // unit
    require(q >= 0, 'OWNER_AUDIT_FAILED: visual N below middle-pair construction domain')
    v128 = 5038 * n + 32389
    additive = b0 + unit * q
    require(0 <= v128 - additive < unit,
            'OWNER_AUDIT_FAILED: dynamic pair floor bound differs')
    return {'N': n, 'original_R7_low_B0_bytes': b0,
            'V128_R43_complete_bytes': v128, 'AddOldOrder_complete_bytes': additive,
            'extra_old_order_stages': q, 'residual_stage_unit_bytes': unit,
            'V128_minus_AddOldOrder_bytes': v128 - additive,
            'candidate_complete_visual_bytes': v128 if mode == 'V128-R43' else additive}


def round10_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    require(base_profile == 'r7_joint_k1_gt_cond',
            'OWNER_AUDIT_FAILED: middle base lineage differs')
    budget = middle_budget(n, mode)
    require(complete_bytes == budget['candidate_complete_visual_bytes'],
            'OWNER_AUDIT_FAILED: physical complete bytes differ from dynamic middle formula')
    detail = metadata['round10_middle_candidate']
    require(detail['mode'] == mode and detail['source_sha256'] == sha(ROOT / 'code/round10_middle_bank_v1.py'),
            'OWNER_AUDIT_FAILED: executed middle factory differs')
    stages = metadata['selected_stages']
    original_stages = metadata['original_selected_stages']
    expected = 43 if mode == 'V128-R43' else 43 + budget['extra_old_order_stages']
    require(metadata['selected_stage_count'] == len(stages) == expected and
            len({tuple(s) for s in stages}) == len(stages) and
            len(original_stages) == 43 and sum(s[0] == 'k' for s in original_stages) == 23 and
            sum(s[0] == 'v' for s in original_stages) == 20 and stages[:43] == original_stages and
            detail['selected_stage_count'] == expected and detail['added_stages'] == expected - 43 and
            detail['K_stages'] == sum(s[0] == 'k' for s in stages) and
            detail['V_stages'] == sum(s[0] == 'v' for s in stages),
            'OWNER_AUDIT_FAILED: actual middle selected stage coverage differs')
    if mode == 'V128-R43':
        require(sum(s[0] == 'k' for s in stages) == 23 and sum(s[0] == 'v' for s in stages) == 20,
                'OWNER_AUDIT_FAILED: V128 original K23/V20 coverage differs')
    require(detail['actual_extra_persistent_bytes'] == complete_bytes - budget['original_R7_low_B0_bytes'] and
            detail['before_unique_capacity_bytes'] == budget['original_R7_low_B0_bytes'] and
            detail['after_unique_capacity_bytes'] == complete_bytes and
            metadata['returned_owners_unique_tensor_capacity_bytes'] == complete_bytes and
            detail['new_persistent_constant_bytes'] == 0 and
            detail['native_FP16_source_or_workspace_retained'] is False and
            detail['V_group_size'] == (128 if mode == 'V128-R43' else n) and
            detail['V_grouping'] == ('128' if mode == 'V128-R43' else 'all'),
            'OWNER_AUDIT_FAILED: physical increment, constants, source release or actual grouping differs')
    return {**budget, 'actual_complete_visual_bytes': complete_bytes,
            'actual_delta_vs_R7_low_bytes': complete_bytes - budget['original_R7_low_B0_bytes'],
            'actual_extra_persistent_bytes': detail['actual_extra_persistent_bytes'],
            'new_persistent_constant_bytes': 0,
            'formula_charges_all_physical_persistent_storages': True,
            'same_complete_bytes_pair': budget['V128_minus_AddOldOrder_bytes'] == 0,
            'pair_gap_less_than_one_18N_residual_unit': True,
            'not_controlled_speed_or_HBM_traffic': True}


def build_candidate_owners(cache, vp, variant, config, assets):
    from round10_middle_bank_v1 import build_round10_middle_owners
    owners, metadata = build_round10_middle_owners(cache, vp, variant,
        mode=config['candidate']['mode'], legacy_a2=config['legacy_a2'],
        legacy_experiments=config['legacy_experiments'], legacy_scripts=config['legacy_scripts'],
        kivi_source=config['kivi_source'], expected_asset_sha256=assets)
    require(metadata['variant_sha256'] == frozen.canonical_sha(variant),
            'OWNER_AUDIT_FAILED: original encoding-lineage variant differs')
    detail = metadata['round10_middle_candidate']
    require(detail['mode'] == config['candidate']['mode'], 'OWNER_AUDIT_FAILED: middle factory mode differs')
    expected = 43 if detail['mode'] == 'V128-R43' else 43 + middle_budget(int(vp.numel()), detail['mode'])['extra_old_order_stages']
    require(metadata['selected_stage_count'] == len(metadata['selected_stages']) == expected,
            'OWNER_AUDIT_FAILED: actual middle stage coverage differs')
    return owners, metadata


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value = dict(value)
        value.update(base_profile_numeric_fields_unchanged=False,
            frozen_base_lineage_numeric_fields_unchanged=True,
            frozen_low43_profile_is_current_deployment_proof=False,
            middle_deployment_contract={'schema': 'trajkv.round10.middle_pair.v1',
                'base_driver_sha256': BASE_RUNNER_SHA,
                'base_variant_is_encoding_lineage_only': True,
                'candidate': candidate_assets(value['config'])[0],
                'actual_group_and_stage_evidence': 'records.jsonl r7_bridge.round10_middle_candidate',
                'physical_budget': 'V128=5038*N+32389; Add=4590*N+89733+18*N*floor((448*N-57344)/(18*N))',
                'pair_exact_byte_equality_claimed': False,
                'new_QA_scope': 'D160 development; two modes, 320 answer events; references reused separately'})
    return _write_base(path, value)


def install_driver_overrides():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen generation driver changed')
    frozen.MODES = MODES
    frozen.RUNTIME_FILES = RUNTIME_FILES
    frozen.PROFILES = deepcopy(frozen.PROFILES)
    # The reused main has an original43 shortcut. Middle stage coverage is
    # instead checked exactly in build_candidate_owners and physical guard.
    frozen.PROFILES['r7_joint_k1_gt_cond']['require_43_stages'] = False
    frozen.candidate_assets = candidate_assets
    frozen.validate_config = validate_config
    frozen.build_candidate_owners = build_candidate_owners
    frozen.round10_budget_guard = round10_budget_guard
    frozen.runtime_source_manifest = runtime_source_manifest
    frozen.write_new = write_new


def main():
    install_driver_overrides()
    return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
