"""KRA1 on the unchanged frozen M1 free-generation driver.

Only original low43 K residual alpha is recalculated. All code assignments,
base K, V and selected stages remain original; the 5980-byte metric is charged.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen

BASE_RUNNER_SHA = '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96'
MODES = ('KRA1',)
RUNTIME_FILES = tuple(dict.fromkeys((*frozen.RUNTIME_FILES,
    'round10_kra_runner_v1.py', 'round10_kra_bank_v1.py', 'round10_kra_metric_v1.py')))
_validate_base = frozen.validate_config
_write_base = frozen.write_new
require, sha = frozen.require, frozen.sha


def runtime_source_manifest():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen generation driver changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def candidate_assets(config):
    from round10_kra_metric_v1 import load_metric
    candidate = config['candidate']
    require(candidate.get('mode') == 'KRA1' and
            not (set(candidate) - {'mode', 'metric_path', 'metric_sha256', 'technical_only'}),
            'INPUT_CONTRACT: KRA1 mode and locked metric only')
    technical = candidate.get('technical_only', False)
    require(type(technical) is bool and (not technical or config['count'] <= 4),
            'INPUT_CONTRACT: technical runs are at most four answers')
    path, digest = candidate.get('metric_path'), candidate.get('metric_sha256')
    require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64,
            'INPUT_CONTRACT: locked KRA1 metric required')
    _, info = load_metric(path, digest, device='cpu')
    import json
    doc = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(technical or doc.get('strength') in (.25, .5, 1.),
            'INPUT_CONTRACT: free-generation KRA1 is a nonzero selected strength')
    return {'mode': 'KRA1', 'metric': info, 'metric_storage_bytes': 5980,
            'only_original_K_residual_alpha_changed': True,
            'V_and_all_codes_and_Kbase_unchanged': True,
            'base_lineage': 'original_r7_joint_k1_gt_cond_K23_V20_43',
            'same_complete_bytes_as_R7low_claimed': False}, {path: digest}


def validate_config(config):
    if frozen.candidate_assets is not candidate_assets:
        install_driver_overrides()
    result = _validate_base(config)
    require(config['variant_source']['name'] == 'r7_joint_k1_gt_cond' and
            config['method'].get('id') == 'Round10_KRA1_v1',
            'INPUT_CONTRACT: original low43 and explicit KRA1 method only')
    return result


def kra_budget(n):
    require(type(n) is int and n > 0, 'OWNER_AUDIT_FAILED: positive actual N required')
    return {'N': n, 'original_R7_low_B0_bytes': 4590*n + 89733,
            'metric_storage_bytes': 5980,
            'candidate_complete_visual_bytes': 4590*n + 89733 + 5980}


def round10_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    require(mode == 'KRA1' and base_profile == 'r7_joint_k1_gt_cond',
            'OWNER_AUDIT_FAILED: KRA1 starts from original low43 only')
    budget = kra_budget(n)
    require(complete_bytes == budget['candidate_complete_visual_bytes'],
            'OWNER_AUDIT_FAILED: physical bytes differ from B0(N)+5980')
    detail = metadata['round10_kra_candidate']
    stages = metadata['selected_stages']
    require(metadata['selected_stage_count'] == len(stages) == 43 and
            len({tuple(s) for s in stages}) == 43 and
            sum(s[0] == 'k' for s in stages) == 23 and
            sum(s[0] == 'v' for s in stages) == 20,
            'OWNER_AUDIT_FAILED: original K23/V20 coverage differs')
    require(detail['mode'] == mode and
            detail['source_sha256'] == sha(ROOT / 'code/round10_kra_bank_v1.py') and
            detail['new_persistent_constant_bytes'] == 5980 and
            detail['actual_extra_persistent_bytes'] == 5980 and
            detail['before_unique_capacity_bytes'] == budget['original_R7_low_B0_bytes'] and
            detail['after_unique_capacity_bytes'] == complete_bytes and
            metadata['returned_owners_unique_tensor_capacity_bytes'] == complete_bytes and
            detail['original_K23_V20_coverage'] is True and
            detail['original_K_base_fields_unchanged'] is True and
            detail['original_V_base_fields_unchanged'] is True and
            detail['all_residual_sign_codes_unchanged'] is True and
            detail['original_V20_alpha_unchanged'] is True and
            detail['original_lookup_ranking_policy_unchanged'] is True and
            detail['reader_unchanged'] is True and
            detail['native_FP16_source_or_workspace_retained'] is False,
            'OWNER_AUDIT_FAILED: KRA1 actual persistent owner/metric/source differs')
    return {**budget, 'actual_complete_visual_bytes': complete_bytes,
            'actual_delta_vs_R7_low_bytes': 5980, 'actual_extra_persistent_bytes': 5980,
            'formula_charges_all_physical_persistent_storages': True,
            'same_complete_bytes_as_R7_low_claimed': False,
            'not_controlled_speed_or_HBM_traffic': True}


def build_candidate_owners(cache, vp, variant, config, assets):
    from round10_kra_bank_v1 import build_round10_kra_owners
    c = config['candidate']
    owners, metadata = build_round10_kra_owners(cache, vp, variant,
        metric_path=c['metric_path'], metric_sha256=c['metric_sha256'],
        legacy_a2=config['legacy_a2'], legacy_experiments=config['legacy_experiments'],
        legacy_scripts=config['legacy_scripts'], kivi_source=config['kivi_source'],
        expected_asset_sha256=assets)
    require(metadata['variant_sha256'] == frozen.canonical_sha(variant),
            'OWNER_AUDIT_FAILED: original variant differs')
    return owners, metadata


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value = dict(value)
        value.update(base_profile_numeric_fields_unchanged=False,
            frozen_base_lineage_numeric_fields_unchanged=True,
            frozen_low43_profile_is_current_deployment_proof=False,
            kra_deployment_contract={'schema': 'trajkv.round10.kra_deployment.v1',
                'base_driver_sha256': BASE_RUNNER_SHA,
                'base_variant_is_encoding_lineage_only': True,
                'candidate': candidate_assets(value['config'])[0],
                'only_K_residual_alpha_recalculated': True,
                'V_all_codes_Kbase_lookup_and_stage_coverage_unchanged': True,
                'physical_budget': 'B0(N)=4590*N+89733; KRA1=B0(N)+5980',
                'metric_storage_bytes': 5980,
                'same_complete_bytes_as_R7low_claimed': False,
                'new_QA_scope': 'D160 development; one KRA1 mode, 160 new answers; 480 references reused'})
    return _write_base(path, value)


def install_driver_overrides():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen generation driver changed')
    frozen.MODES = MODES
    frozen.RUNTIME_FILES = RUNTIME_FILES
    frozen.PROFILES = deepcopy(frozen.PROFILES)
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
