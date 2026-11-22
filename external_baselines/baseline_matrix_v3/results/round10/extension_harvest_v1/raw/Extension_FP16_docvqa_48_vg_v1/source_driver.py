"""Conditional Round10 reserve224 replication using frozen inference parents.

Only cohort role/count validation, candidate dispatch and source declarations
are adapted. The unchanged primary main performs real native prefill, owner
release, question-suffix recomputation, position checks and free generation.
"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_middle_runner_v1 as middle
# Use a separate instance of the unchanged driver so importing this adapter
# cannot change another adapter's module globals in a CPU verifier process.
_spec = importlib.util.spec_from_file_location('round10_extension_frozen_primary', ROOT / 'code/round10_runner_v1.py')
primary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(primary)

PARENT_PINS = {
    'code/round10_runner_v1.py': '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96',
    'code/round10_middle_runner_v1.py': '6dcf3fa5865915271b7c7074bc987aee74625ce41cc9d70ac4c56e268e7f86ac',
    'code/round10_middle_bank_v1.py': '8928db801032c4d104debf1d361fb1275fd12ce957c61df541440c92361c8bbb',
}
INPUT_MANIFEST_SHA = 'b2ba5f54516abf96344e02356c7cee079b4ff48e2a270ea8fc61ea1c78baaecf'
SOURCE_LOCK_SHA = '6e04a59609e915d809304463e501fbe90a166a7d60c7be79a12893d2a5afd1e7'
COUNTS = {'textvqa': 64, 'infovqa': 64, 'docvqa': 48, 'chartqa': 48}
PRIMARY_MODES = tuple(primary.MODES)
CANDIDATE_MODES = ('LS-FZ', 'LS-RE', 'FB0', 'FB-W', *middle.MODES)
MODES = PRIMARY_MODES + middle.MODES
RUNTIME_FILES = tuple(dict.fromkeys((*middle.RUNTIME_FILES, 'round10_extension_runner_v1.py')))
_validate_primary = primary.validate_config
_candidate_primary = primary.candidate_assets
_build_primary = primary.build_candidate_owners
_budget_primary = primary.round10_budget_guard
_write_primary = primary.write_new


def require(value, message):
    primary.require(value, message)


def sha(path):
    return primary.sha(path)


def runtime_source_manifest():
    for path, expected in PARENT_PINS.items():
        require(sha(ROOT / path) == expected, 'INPUT_MISMATCH: frozen extension parent changed: ' + path)
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def extension_domain(config):
    filename = PurePosixPath(config['inputs'].replace('\\', '/')).name
    found = [domain for domain, count in COUNTS.items()
             if filename == f'{domain}_reserve{count}_inputs.jsonl']
    require(len(found) == 1, 'INPUT_CONTRACT: exact declared per-domain reserve input required')
    return found[0]


def parent_config(config):
    lineage = config.get('extension_lineage', {})
    require(set(lineage) == {'parent_config_path', 'parent_config_sha256'} and
            isinstance(lineage['parent_config_path'], str) and
            isinstance(lineage['parent_config_sha256'], str) and len(lineage['parent_config_sha256']) == 64,
            'INPUT_CONTRACT: exact frozen parent config binding required')
    path = lineage['parent_config_path']
    require(sha(path) == lineage['parent_config_sha256'], 'INPUT_MISMATCH: extension parent config changed')
    parent = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_parent_pair(config, parent)
    return parent


def validate_parent_pair(config, parent):
    """Pure lineage validation; usable against independently retrieved bytes."""
    require(parent.get('schema') == primary.SCHEMA and parent.get('input_role') == 'development' and
            parent.get('cohort_kind') == 'development' and parent.get('candidate', {}).get('technical_only', False) is False,
            'INPUT_CONTRACT: extension lineage must be the original development method')
    # Every scientific/runtime setting, asset, method ID, theta and selected W
    # must equal the frozen parent. A new cohort does not create a new method.
    allowed = {'inputs', 'inputs_sha256', 'input_count', 'start', 'count', 'input_role', 'cohort_kind',
               'scope', 'expected_source_hashes', 'extension_lineage', 'input_manifest_sha256'}
    same = lambda item: {key: value for key, value in item.items() if key not in allowed}
    require(same(config) == same(parent), 'INPUT_MISMATCH: replication changed method/parameters/protocol beyond cohort/source bindings')
    ds = extension_domain(config)
    expected_parent_count = 16 if ds in ('docvqa', 'chartqa') else 64
    require(PurePosixPath(parent['inputs']).name == f'{ds}_dev{expected_parent_count}_inputs.jsonl',
            'INPUT_MISMATCH: parent method comes from another domain')
    return parent


def candidate_assets(config):
    parent = parent_config(config)
    function = middle.candidate_assets if config['candidate']['mode'] in middle.MODES else _candidate_primary
    details, locks = function(config)
    lineage = config['extension_lineage']
    locks[lineage['parent_config_path']] = lineage['parent_config_sha256']
    return {**details, 'extension_parent_config_sha256': lineage['parent_config_sha256'],
            'scientific_method_id_unchanged': config['method']['id'] == parent['method']['id'],
            'calibration_parameters_reselected': False}, locks


def validate_config(config):
    if primary.candidate_assets is not candidate_assets:
        install_driver_overrides()
    require(config.get('input_role') == 'reserved_unused_extension' and config.get('cohort_kind') == 'replication',
            'INPUT_CONTRACT: preserve frozen reserve role and label this run replication')
    require(config.get('input_manifest_sha256') == INPUT_MANIFEST_SHA,
            'INPUT_CONTRACT: original input manifest SHA required')
    ds = extension_domain(config)
    require(config.get('start') == 0 and config.get('count') == config.get('input_count') == COUNTS[ds] and
            config.get('candidate', {}).get('technical_only', False) is False,
            'INPUT_CONTRACT: full reserve domain, no technical/development subset')
    require(config['evaluation_contract']['primary_metric_id'] == primary.PRIMARY_METRIC_IDS[ds],
            'INPUT_CONTRACT: reserve domain scoring identity differs')
    # Normalize only a temporary validation view to reuse the frozen M1
    # restrictions. Actual config, loader role, manifest and records retain the
    # real reserved_unused_extension/replication labels without mutation.
    view = deepcopy(config)
    view.update(input_role='development', cohort_kind='development')
    _validate_primary(view)
    parent_config(config)
    return config


def build_candidate_owners(cache, vp, variant, config, assets):
    if config['candidate']['mode'] in middle.MODES:
        return middle.build_candidate_owners(cache, vp, variant, config, assets)
    owners, metadata = _build_primary(cache, vp, variant, config, assets)
    mode = config['candidate']['mode']
    if mode == 'FP16':
        require(metadata['selected_stage_count'] == 0, 'OWNER_AUDIT_FAILED: FP16 stages changed')
    elif config['variant_source']['name'] == 'r7_joint_k1_gt_cond':
        require(metadata['selected_stage_count'] == len(metadata['selected_stages']) == 43,
                'OWNER_AUDIT_FAILED: frozen primary/low43 stages changed')
    return owners, metadata


def round10_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    function = middle.round10_budget_guard if mode in middle.MODES else _budget_primary
    return function(mode, n, complete_bytes, metadata, base_profile)


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value = dict(value)
        config = value['config']
        mode = config['candidate']['mode']
        value['extension_deployment_contract'] = {
            'schema': 'trajkv.round10.reserve224_replication.v1',
            'input_role_preserved': 'reserved_unused_extension', 'run_cohort_kind': 'replication',
            'parent_config_sha256': config['extension_lineage']['parent_config_sha256'],
            'scientific_method_id': config['method']['id'], 'scientific_method_and_candidate_parameters_unchanged': True,
            'calibration_parameters_reselected': False, 'original_input_manifest_sha256': INPUT_MANIFEST_SHA,
            'frozen_source_lock_sha256': SOURCE_LOCK_SHA, 'development_D160_predictions_reused': False,
            'new_method_or_positive_result_claimed': False,
            'numerical_parent': 'middle' if mode in middle.MODES else 'primary',
            'frozen_parent_sources': PARENT_PINS}
        if mode in middle.MODES:
            value.update(base_profile_numeric_fields_unchanged=False,
                frozen_base_lineage_numeric_fields_unchanged=True,
                frozen_low43_profile_is_current_deployment_proof=False)
    return _write_primary(path, value)


def install_driver_overrides():
    runtime_source_manifest()
    primary.MODES = MODES
    primary.RUNTIME_FILES = RUNTIME_FILES
    primary.PROFILES = deepcopy(primary.PROFILES)
    primary.PROFILES['r7_joint_k1_gt_cond']['require_43_stages'] = False
    primary.validate_config = validate_config
    primary.candidate_assets = candidate_assets
    primary.build_candidate_owners = build_candidate_owners
    primary.round10_budget_guard = round10_budget_guard
    primary.runtime_source_manifest = runtime_source_manifest
    primary.write_new = write_new


def main():
    install_driver_overrides()
    return primary.main()


if __name__ == '__main__':
    raise SystemExit(main())
