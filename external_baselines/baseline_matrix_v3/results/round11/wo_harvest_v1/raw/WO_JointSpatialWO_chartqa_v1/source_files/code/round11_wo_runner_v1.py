"""Round11 bounded W_O-geometry SourceGroupD96 packed generation.

The byte-pinned Round10 driver still owns question-free native prefill, full
suffix recomputation, actual positions, source release and real generation.
This module supplies the new value factory and physical deployment contract.
No labels, fitting, remote dispatch or canonical SourceRun admission occur here.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import weakref

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen

BASE_RUNNER_SHA = '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96'
SCHEMA = 'trajkv.round11_wo_research.raw.v1'
MODES = ('SpatialWO', 'JointSpatialWO')
COUNTS = {'textvqa': 32, 'infovqa': 32, 'docvqa': 16, 'chartqa': 16}
RUNTIME_FILES = tuple(dict.fromkeys((*frozen.RUNTIME_FILES,
    'round11_wo_runner_v1.py', 'round11_wo_value_v1.py', 'round11_value_bank_v2.py',
    'round11_joint_value_v2.py', 'round11_joint_value_v1.py', 'round11_attention_v1.py')))
_validate_base = frozen.validate_config
_write_base = frozen.write_new
_MODEL_CAPTURE = None


def require(value, message):
    frozen.require(value, message)


def sha(path):
    return frozen.sha(path)


def parameter_identity(model):
    return tuple((name, tuple(p.shape), str(p.dtype), str(p.device),
                  p.untyped_storage().data_ptr(), p._version)
                 for name, p in model.named_parameters())


def current_model():
    require(_MODEL_CAPTURE is not None and _MODEL_CAPTURE['loader_calls'] == 1,
            'MODEL_CAPTURE: exactly one original model load required')
    model = _MODEL_CAPTURE['model_ref']()
    require(model is not None, 'MODEL_CAPTURE: original model no longer alive')
    return model


def capture_identity():
    model = current_model()
    n = sum(p.numel() for p in model.parameters())
    require(n == 7944373760, 'MODEL_CAPTURE: complete original M1 parameter count differs')
    return {'actual_from_pretrained_calls': _MODEL_CAPTURE['loader_calls'],
            'counted_loader_entry': 'AutoModelForImageTextToText.from_pretrained',
            'actual_model_parameter_count': n,
            'only_weak_reference_retained_by_capture': True,
            'model_weights_borrowed_not_cloned_or_quantized': True}


@contextmanager
def capture_loaded_model(model_class=None):
    """Capture one existing loader result; restore its exact method descriptor.

    A test class is supported for a CPU loader/restoration control. The deployed
    path always uses the original HF class, original arguments and return object.
    No strong model reference is stored outside the frozen native generation code.
    """
    global _MODEL_CAPTURE
    require(_MODEL_CAPTURE is None, 'MODEL_CAPTURE: nested capture is not supported')
    if model_class is None:
        from transformers import AutoModelForImageTextToText
        model_class = AutoModelForImageTextToText
    owned = 'from_pretrained' in model_class.__dict__
    descriptor = model_class.__dict__.get('from_pretrained')
    original = getattr(model_class, 'from_pretrained')
    state = {'loader_calls': 0, 'model_ref': None}
    _MODEL_CAPTURE = state

    def captured(cls, *args, **kwargs):
        require(cls is model_class and state['loader_calls'] == 0,
                'MODEL_CAPTURE: unexpected class or second model load')
        model = original(*args, **kwargs)
        state['loader_calls'] += 1
        state['model_ref'] = weakref.ref(model)
        return model

    setattr(model_class, 'from_pretrained', classmethod(captured))
    try:
        yield state
    finally:
        if owned:
            setattr(model_class, 'from_pretrained', descriptor)
        else:
            delattr(model_class, 'from_pretrained')
        _MODEL_CAPTURE = None


def runtime_source_manifest():
    """CPU-only inventory; an unfinished source graph cannot be launched."""
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen generation driver changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def source_group_identity(config, *, selected_ids=None, dataset=None):
    path = config.get('source_group_manifest_path')
    digest = config.get('source_group_manifest_sha256')
    require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64,
            'INPUT_CONTRACT: exact separately frozen SourceGroupD96 manifest required')
    require(sha(path) == digest, 'INPUT_MISMATCH: SourceGroupD96 manifest changed')
    group = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(group.get('schema') == 'trajkv.round11.source_group.v1',
            'INPUT_CONTRACT: Round11 source-group schema required')
    require(group.get('reference_input_manifest_sha256') == config.get('input_manifest_sha256'),
            'INPUT_MISMATCH: SourceGroupD96 original manifest differs')
    rows = [json.loads(line) for line in Path(config['inputs']).read_text(encoding='utf-8-sig').splitlines()]
    require(len(rows) == config['input_count'] and all(isinstance(row.get('id'), str) for row in rows),
            'INPUT_MISMATCH: original input cohort count or IDs differ')
    chosen = rows[config['start']:config['start'] + config['count']]
    require(len(chosen) == config['count'] and len({r['id'] for r in chosen}) == len(chosen),
            'INPUT_MISMATCH: duplicate/incomplete SourceGroupD96 slice')
    domains = {row['dataset'] for row in chosen}
    require(len(domains) == 1, 'INPUT_MISMATCH: source-group slice crosses domains')
    actual_dataset = next(iter(domains))
    require(dataset is None or dataset == actual_dataset, 'INPUT_MISMATCH: source-group domain differs')
    detail = group.get('datasets', {}).get(actual_dataset, {})
    ids = [row['id'] for row in chosen]
    require(actual_dataset in COUNTS and config['start'] == 0 and config['count'] == COUNTS[actual_dataset]
            and detail.get('input_sha256') == config['inputs_sha256'] == sha(config['inputs'])
            and detail.get('input_count') == config['input_count']
            and detail.get('start') == config['start'] and detail.get('count') == config['count']
            and detail.get('ids') == ids,
            'INPUT_MISMATCH: exact first32/32/16/16 SourceGroupD96 membership differs')
    require(selected_ids is None or selected_ids == ids, 'INPUT_MISMATCH: actual runner slice differs from source group')
    require(not any(set(row) & frozen.FORBIDDEN for row in rows), 'INPUT_CONTRACT: inference inputs contain references')
    return {'schema': group['schema'], 'manifest_sha256': digest, 'dataset': actual_dataset,
            'input_sha256': detail['input_sha256'], 'input_count': config['input_count'],
            'start': config['start'], 'count': config['count'], 'ids': ids,
            'selection_rule': 'first32 Text/Info and first16 Doc/Chart of frozen D160; no answer selection'}


def candidate_assets(config):
    candidate = config['candidate']
    require(candidate.get('mode') in MODES, 'INPUT_CONTRACT: declared Round11 mode required')
    require(set(candidate) <= {'mode', 'technical_only'},
            'INPUT_CONTRACT: no fitted/question-selected Round11 parameters')
    require(candidate.get('technical_only', False) is False,
            'INPUT_CONTRACT: QA0 must use the separate technical driver; no technical free answers')
    return {'schema': 'trajkv.round11.wo_deployment.v1', 'mode': candidate['mode'],
            'encoding_applied': True, 'base_lineage': 'original_r7_joint_k1_gt_cond_K23_V20_43',
            'new_parameter_fit': False, 'fixed_mechanism_tile_tokens': 128,
            'fixed_candidate_chunk': 32, 'same_complete_B0_bytes': True,
            'local_metric': 'sum_GQA7_WO_head_block_Gram',
            'fixed_cyclic_sign_sweeps': 2,
            'joint_base_sign_sweeps': 2 if candidate['mode'] == 'JointSpatialWO' else 0,
            'model_derived_metric_retained_for_serving': False,
            'new_frame_is_deployment_change': True,
            'frozen_low43_profile_is_current_deployment_proof': False}, {
                config['source_group_manifest_path']: config['source_group_manifest_sha256']}


def validate_config(config):
    if frozen.candidate_assets is not candidate_assets:
        install_driver_overrides()
    # Only the family identifier is translated for the byte-frozen validator.
    # No model/input/variant/codec parameter is translated.
    shadow = deepcopy(config)
    require(shadow.get('method', {}).get('family') == 'trajkv_round11_wo_research',
            'INPUT_CONTRACT: explicit Round11 research family required')
    shadow['method']['family'] = 'trajkv_round10_research'
    _validate_base(shadow)
    require(config['variant_source']['name'] == 'r7_joint_k1_gt_cond',
            'INPUT_CONTRACT: Round11 uses original R7 low43 lineage only')
    require(config.get('input_manifest_sha256') is not None,
            'INPUT_CONTRACT: exact original no-answer manifest SHA required')
    source_group_identity(config)
    candidate_assets(config)
    return config


def round11_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    require(type(n) is int and n > 0 and mode in MODES and base_profile == 'r7_joint_k1_gt_cond',
            'OWNER_AUDIT_FAILED: Round11 mode/N/base lineage differs')
    expected = 4590 * n + 89733
    require(complete_bytes == expected, 'OWNER_AUDIT_FAILED: complete physical owner is not B0=4590*N+89733')
    stages = metadata.get('selected_stages', [])
    require(metadata.get('selected_stage_count') == len(stages) == 43
            and len({tuple(stage) for stage in stages}) == 43
            and all(len(stage) == 4 and stage[3] == 0 for stage in stages)
            and sum(stage[0] == 'k' for stage in stages) == 23
            and sum(stage[0] == 'v' for stage in stages) == 20,
            'OWNER_AUDIT_FAILED: original K23/V20 stage coverage differs')
    detail = metadata.get('round11_wo_candidate', {})
    require(detail.get('schema') == 'trajkv.round11.wo_candidate.v1'
            and detail.get('mode') == mode
            and detail.get('encoding_applied') is True and detail.get('control_mode0_only') is False
            and detail.get('source_sha256') == sha(ROOT / 'code/round11_wo_value_v1.py'),
            'OWNER_AUDIT_FAILED: actual new-frame factory/mode differs')
    for key in ('fixed_K', 'original_K23_V20_stage0_coverage', 'old_f0_retained_and_charged',
                'shared_original_constants_unchanged', 'no_new_tensor_fields'):
        require(detail.get(key) is True, 'OWNER_AUDIT_FAILED: missing Round11 invariant ' + key)
    require(detail.get('before_unique_capacity_bytes') == detail.get('after_unique_capacity_bytes') == expected
            and detail.get('actual_extra_persistent_bytes') == 0
            and detail.get('complete_B0_formula') == '4590*N+89733'
            and detail.get('complete_B0_bytes') == expected
            and detail.get('tile_tokens') == 128 and detail.get('candidate_chunk') == 32
            and detail.get('native_FP16_source_or_encoding_workspace_retained') is False
            and metadata.get('returned_owners_unique_tensor_capacity_bytes') == expected
            and metadata.get('reader_unchanged') is False
            and detail.get('reader_is_new_spatial') is True
            and detail.get('persistent_derived_metric_tensor_bytes') == 0,
            'OWNER_AUDIT_FAILED: physical increment or actual reader mode differs')
    return {'schema': 'trajkv.round11.physical_budget.v1', 'N': n,
            'original_R7_low_B0_bytes': expected, 'actual_complete_visual_bytes': complete_bytes,
            'actual_delta_vs_R7_low_bytes': 0, 'same_complete_bytes_as_original': True,
            'K23_V20_stage0_coverage': True, 'fixed_K': True, 'mode': mode,
            'new_frame_encoding_applied': True, 'new_persistent_tensor_bytes': 0,
            'old_f0_retained_and_charged': True, 'formula_charges_all_physical_persistent_storages': True,
            'not_controlled_speed_or_HBM_traffic': True}


def build_candidate_owners(cache, vp, variant, config, assets):
    from round11_wo_value_v1 import build_round11_wo_owners
    model = current_model()
    before = parameter_identity(model)
    owners, metadata = build_round11_wo_owners(cache, vp, variant, model=model,
        mode=config['candidate']['mode'], encode=True,
        legacy_a2=config['legacy_a2'], legacy_experiments=config['legacy_experiments'],
        legacy_scripts=config['legacy_scripts'], kivi_source=config['kivi_source'],
        expected_asset_sha256=assets)
    require(metadata.get('variant_sha256') == frozen.canonical_sha(variant),
            'OWNER_AUDIT_FAILED: original frozen variant changed')
    require(parameter_identity(model) == before, 'MODEL_IDENTITY: W_O encoding changed model parameters')
    metadata['round11_wo_candidate']['actual_model_capture'] = capture_identity()
    metadata['round11_wo_candidate']['model_parameters_unchanged_during_encoding'] = True
    return owners, metadata


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value = dict(value)
        config = value['config']
        identity = source_group_identity(config, selected_ids=value['expected_ids'])
        value.update(source_group_identity=identity,
            runtime_process={'PID': os.getpid(), 'parent_PID': os.getppid(),
                             'source_snapshot_root': str(ROOT)},
            source_driver_contract='Round11 custom packed inference; canonical SourceRun admission not claimed',
            base_profile_numeric_fields_unchanged=False,
            frozen_base_lineage_numeric_fields_unchanged=True,
            frozen_low43_profile_is_current_deployment_proof=False,
            round11_candidate_assets=value.pop('round10_candidate_assets', None),
            round11_deployment_contract={'schema': 'trajkv.round11.wo_deployment.v1',
                'base_driver_sha256': BASE_RUNNER_SHA,
                'base_variant_is_encoding_lineage_only': True,
                'physical_budget': 'B0=4590*N+89733 complete persistent owner bytes',
                'actual_factory_and_flags': 'records.jsonl r7_bridge.round11_wo_candidate',
                'bounded_maximum_new_free_generation_QA': 192,
                'primary_reference_reuse_is_separate_group': True,
                'QA0_is_not_task_quality': True})
    elif Path(path).name == 'complete.json':
        value = {**value, 'round11_schema': SCHEMA, 'canonical_SourceRun_admission_claimed': False,
                 'new_parameter_fit': False, 'new_frame_deployment': True}
    return _write_base(path, value)


def install_driver_overrides():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA,
            'INPUT_MISMATCH: frozen generation driver changed')
    frozen.SCHEMA = SCHEMA
    frozen.MODES = MODES
    frozen.RUNTIME_FILES = RUNTIME_FILES
    frozen.candidate_assets = candidate_assets
    frozen.validate_config = validate_config
    frozen.build_candidate_owners = build_candidate_owners
    frozen.round10_budget_guard = round11_budget_guard
    frozen.runtime_source_manifest = runtime_source_manifest
    frozen.write_new = write_new


def main():
    install_driver_overrides()
    with capture_loaded_model():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
