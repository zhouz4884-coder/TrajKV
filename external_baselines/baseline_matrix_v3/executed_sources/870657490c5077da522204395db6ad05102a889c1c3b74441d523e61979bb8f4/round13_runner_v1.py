"""Round13 production free generation over the byte-frozen Round10 M1 driver.

New code admits the explicit development inputs, routes real owners, preserves
original sparse positions, enforces request-wise physical caps, and observes
raw per-step generation margins. Frozen model/prefill/generation code is not
edited. No diagnostic/oracle or parameter fitting is admitted in this runner.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import weakref

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_runner_v1 as frozen
import round10_inputs_v1 as old_inputs

SCHEMA = 'trajkv.round13_research.raw.v1'
BASE_RUNNER_SHA = '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96'
MODES = ('P_best', 'P_random', 'EX_best', 'TOK_best', 'CAL_AFF')
ADDITIONS = ('round13_runner_v1.py', 'round13_inputs_v1.py', 'round13_bank_v1.py',
    'round12_inputs_v1.py',
    'round13_selector_v1.py', 'round13_pruned_bank_v1.py',
    'round13_exception_bank_v1.py', 'round13_exception_attention_v1.py',
    'round13_precision_bank_v1.py', 'round13_precision_codec_v1.py', 'round13_precision_attention_v1.py',
    'round12_static_bank_v1.py', 'round12_static_codec_v1.py', 'round12_static_attention_v1.py')
RUNTIME_FILES = tuple(dict.fromkeys((*frozen.RUNTIME_FILES, *ADDITIONS)))
_CONFIG = None
_BINDINGS = {}
_LAST_NATIVE_TILE_LAYOUT = None
_LAST_GENERATION_TRACE = []
_write_base = frozen.write_new
sha = frozen.sha


def require(value, message):
    frozen.require(value, 'ROUND13: ' + message)


def runtime_source_manifest(config=None):
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA, 'frozen native generation driver changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def candidate_assets(config):
    candidate = config['candidate']
    mode = candidate.get('mode')
    require(mode in MODES, 'INPUT_CONTRACT: unknown deployable candidate')
    require(type(candidate.get('technical_only', False)) is bool
            and candidate.get('technical_only', False) is False,
            'INPUT_CONTRACT: technical/oracle execution is excluded from production runner')
    allowed = {'mode', 'technical_only', 'format_version'}
    if mode in ('P_best', 'P_random', 'TOK_best'):
        allowed |= {'key_book_path', 'key_book_sha256'}
    if mode in ('P_best', 'P_random', 'EX_best', 'TOK_best'):
        allowed |= {'selector_seed', 'selector_config'}
    if mode in ('P_best', 'P_random'):
        allowed.add('keep_count')
    if mode == 'EX_best':
        allowed |= {'base_stages', 'requested_h', 'small_n_fallback_max_tokens'}
    if mode == 'CAL_AFF':
        allowed |= {'ratio_path', 'ratio_sha256', 'value_fit_iterations'}
    require(set(candidate) <= allowed, 'INPUT_CONTRACT: unknown/mixed candidate fields')
    selector = candidate.get('selector_config', {})
    require(isinstance(selector, dict) and set(selector) <= {'feature_layers', 'max_anchors'},
            'INPUT_CONTRACT: invalid image-only selector configuration')
    require(type(selector.get('max_anchors', 64)) is int and 0 < selector.get('max_anchors', 64) <= 64,
            'INPUT_CONTRACT: selector permits at most64 streaming anchors')
    layers = selector.get('feature_layers', [0, 13, 27])
    require(isinstance(layers, (list, tuple)) and list(layers) == [0, 13, 27],
            'INPUT_CONTRACT: fixed image-only feature layers0/13/27 required')
    require(type(candidate.get('selector_seed', 0)) is int and 0 <= candidate.get('selector_seed', 0) < 2**31,
            'INPUT_CONTRACT: nonnegative int32 selector seed required')
    if mode == 'EX_best':
        require(candidate.get('base_stages', 43) in (0, 41, 43), 'INPUT_CONTRACT: explicit valid donor stage profile')
        require(type(candidate.get('small_n_fallback_max_tokens', 0)) is int
                and candidate.get('small_n_fallback_max_tokens', 0) in (0, 256),
                'INPUT_CONTRACT: explicit absent/thumbnail-only fallback')
        if candidate.get('requested_h') is not None:
            require(type(candidate['requested_h']) is int and candidate['requested_h'] >= 0,
                    'INPUT_CONTRACT: exception dose must be nonnegative integer')
    if mode == 'CAL_AFF':
        require(type(candidate.get('value_fit_iterations', 12)) is int
                and 0 < candidate.get('value_fit_iterations', 12) <= 32,
                'INPUT_CONTRACT: bounded fixed encoding iterations')
    locks = {}
    for stem in ('key_book', 'ratio'):
        path, digest = candidate.get(stem + '_path'), candidate.get(stem + '_sha256')
        require((path is None) == (digest is None), 'INPUT_CONTRACT: asset path/SHA must appear together')
        if path is not None:
            require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
                    'INPUT_MISMATCH: exact candidate asset changed: ' + stem)
            locks[path] = digest
    require(mode not in ('P_best', 'P_random', 'TOK_best') or candidate.get('key_book_path') is not None,
            'INPUT_CONTRACT: locked G4_256 book required')
    require(mode != 'CAL_AFF' or candidate.get('ratio_path') is not None,
            'INPUT_CONTRACT: locked four-level ratio asset required')
    return {'schema': 'trajkv.round13.production_candidate_assets.v1', 'mode': mode,
            'technical_only': False, 'asset_locks': locks,
            'answers_or_future_question_used_by_encoder': False,
            'parameter_fit_in_inference_runner': False}, locks


def load_runner_inputs(path, manifest_path=None, expected_sha256=None, expected_role=None):
    require(manifest_path is not None, 'INPUT_CONTRACT: explicit no-answer manifest required')
    data = importlib.import_module('round13_inputs_v1')
    values = data.load_round13_inputs(path, manifest_path, expected_sha256, expected_role)
    manifest = data.read(manifest_path)
    _BINDINGS.update(manifest['native_bindings'])
    return values


def validate_config(config):
    global _CONFIG
    data = importlib.import_module('round13_inputs_v1')
    require(config.get('schema') == SCHEMA, 'INPUT_CONTRACT: Round13 schema required')
    require(not (set(config) & frozen.FORBIDDEN), 'INPUT_CONTRACT: inference config contains references/labels')
    require(config.get('model_id') == 'M1' and type(config.get('seed')) is int and config['seed'] == 0,
            'INPUT_CONTRACT: M1 seed0 required')
    require(config.get('question_suffix') == frozen.SUFFIX, 'INPUT_CONTRACT: native question suffix differs')
    require(config.get('preprocessing') == {'max_patches': 8}, 'INPUT_CONTRACT: native cap8 required')
    require(config.get('max_new_tokens') == 64, 'INPUT_CONTRACT: max_new_tokens64 required')
    require(config.get('position_owner_layout') == 'single_shared_visual_position_map',
            'INPUT_CONTRACT: shared original visual map contract required')
    require(config.get('evaluation_contract', {}).get('version') == 'matrix_v3_metrics_v1',
            'INPUT_CONTRACT: frozen metric version differs')
    require(config.get('input_role') == data.ROLE and config.get('cohort_kind') == 'development',
            'INPUT_CONTRACT: explicit development role required')
    require(type(config.get('start')) is int and config['start'] >= 0
            and type(config.get('count')) is int and 0 < config['count'] <= 128,
            'INPUT_CONTRACT: explicit bounded count1..128 slice required')
    require(type(config.get('input_count')) is int and config['start'] + config['count'] <= config['input_count'],
            'INPUT_CONTRACT: incomplete input slice')
    require(isinstance(config.get('input_manifest_sha256'), str)
            and sha(config['input_manifest_path']) == config['input_manifest_sha256'],
            'INPUT_MISMATCH: manifest SHA differs')
    values = load_runner_inputs(config['inputs'], config['input_manifest_path'], config['inputs_sha256'], config['input_role'])
    require(len(values) == config['input_count'], 'INPUT_MISMATCH: input count differs')
    chosen = values[config['start']:config['start'] + config['count']]
    require(len({row['dataset'] for row in chosen}) == 1, 'INPUT_CONTRACT: one dataset per actual run')
    require(config.get('method', {}).get('family') == 'trajkv_round13_research'
            and config['method'].get('information_condition') == 'query_unknown',
            'INPUT_CONTRACT: explicit production Round13/query_unknown family required')
    source = config.get('variant_source', {})
    require(source.get('name') == 'r7_joint_k1_gt_cond' and source.get('sha256') == frozen.ORIGINAL_CONFIG_SHA,
            'INPUT_CONTRACT: frozen low43 lineage required')
    require(config.get('expected_asset_sha256') == frozen.PROFILES[source['name']]['assets'],
            'INPUT_CONTRACT: frozen factor/policy assets differ')
    require(isinstance(config.get('scope'), str) and bool(config['scope']), 'INPUT_CONTRACT: explicit scope required')
    for name in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source'):
        require(isinstance(config.get(name), str) and bool(config[name]), 'INPUT_CONTRACT: missing dependency ' + name)
    candidate_assets(config)
    _CONFIG = config
    require(config.get('expected_source_hashes') == runtime_source_manifest(config), 'INPUT_MISMATCH: exact runtime graph differs')
    return config


def build_candidate_owners(cache, vp, variant, config, assets):
    from round13_bank_v1 import build_round13_owners
    require(_LAST_NATIVE_TILE_LAYOUT is not None, 'INPUT_MISMATCH: native processor layout was not observed')
    return build_round13_owners(cache, vp, variant, candidate=config['candidate'],
        legacy_a2=config['legacy_a2'], legacy_experiments=config['legacy_experiments'],
        legacy_scripts=config['legacy_scripts'], kivi_source=config['kivi_source'],
        expected_asset_sha256=assets, native_tile_layout=_LAST_NATIVE_TILE_LAYOUT)


def physical_budget_guard(mode, n, complete_bytes, metadata, base_profile):
    from round13_exception_bank_v1 import r_budget
    require(mode in MODES and base_profile == 'r7_joint_k1_gt_cond', 'OWNER_AUDIT_FAILED: mode/lineage differs')
    stages = metadata.get('selected_stages')
    require(isinstance(stages, list) and len(stages) == metadata.get('selected_stage_count'),
            'OWNER_AUDIT_FAILED: concrete stage manifest missing')
    expected = next(iter(_BINDINGS.values()))['low_selected_stages']
    require(len(expected) == 43 and len(set(map(tuple, expected))) == 43
            and all(binding['low_selected_stages'] == expected for binding in _BINDINGS.values()),
            'INPUT_CONTRACT: frozen low43 source identities disagree')
    if mode in ('P_best', 'P_random', 'CAL_AFF'):
        require([list(x) for x in stages] == [list(x) for x in expected],
                'OWNER_AUDIT_FAILED: original K23/V20 coverage changed')
    require(metadata.get('returned_owners_unique_tensor_capacity_bytes') == complete_bytes,
            'OWNER_AUDIT_FAILED: factory/Hybrid actual unique bytes disagree')
    require(metadata.get('round13_router', {}).get('mode') == mode,
            'OWNER_AUDIT_FAILED: actual router mode differs')
    low = 4590 * n + 89733
    cap = low if mode in ('P_best', 'P_random') else 6382 * n + 89745 if mode == 'CAL_AFF' else r_budget(n)
    require(0 < complete_bytes <= cap, f'OWNER_AUDIT_FAILED: actual owner {complete_bytes} exceeds request cap {cap}')
    if mode == 'TOK_best':
        detail = metadata['round13_precision']
        require(detail['R_cap_bytes'] == cap and detail['V_fields_frozen_to_request_R'] is True,
                'OWNER_AUDIT_FAILED: TOK lost same-request R/V contract')
        require(all(stage[0] == 'v' for stage in stages), 'OWNER_AUDIT_FAILED: TOK retained K residual')
    elif mode == 'EX_best':
        require(metadata['round13_exception']['original_R_complete_owner_bytes'] == cap,
                'OWNER_AUDIT_FAILED: EX cap differs from request R')
    return {'schema': 'trajkv.round13.physical_budget.v1', 'mode': mode, 'N': n,
        'actual_complete_visual_bytes': complete_bytes, 'strict_cap_bytes': cap,
        'strict_cap_satisfied': complete_bytes <= cap, 'original_low43_bytes': low,
        'original_R_bytes': r_budget(n), 'delta_vs_original_low43_bytes': complete_bytes - low,
        'selected_stages': stages, 'selected_stage_count': len(stages),
        'actual_unique_storage_not_nominal_bits': True, 'original_visual_rows_denominator': n,
        'timing_is_instrumented_quality_run_not_controlled_speed': True}


def share_original_visual_position_map(cache):
    """Share the complete original map; P owner positions remain truly sparse."""
    import torch
    from ownership import cache_storage
    require(len(cache.layers) == 28, 'OWNER_AUDIT_FAILED: M1 layer count differs')
    owner0 = cache.layers[0].visual_owner
    target = getattr(owner0, 'original_positions', owner0.positions)
    old = cache.visual_positions
    require(target.dtype == old.dtype == torch.long and target.device == old.device and torch.equal(target, old),
            'OWNER_AUDIT_FAILED: original position values changed')
    storage = lambda value: (str(value.device), value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())
    require(storage(target) != storage(old), 'OWNER_AUDIT_FAILED: initial Hybrid clone unexpectedly aliased')
    for layer in cache.layers:
        owner = layer.visual_owner
        original = getattr(owner, 'original_positions', owner.positions)
        require(storage(original) == storage(target) and torch.equal(original, target)
                and torch.equal(layer.original_visual_positions, target)
                and bool(torch.isin(owner.positions, target).all()),
                'OWNER_AUDIT_FAILED: original/sparse visual maps disagree')
    before = cache_storage(cache, target)['complete_visual_owner_bytes']
    released_bytes = old.untyped_storage().nbytes()
    previous = weakref.ref(old)
    cache.visual_positions = target
    for layer in cache.layers:
        layer.original_visual_positions = target
    old = None
    gc.collect()
    require(previous() is None, 'OWNER_AUDIT_FAILED: obsolete original-map clone retained')
    after = cache_storage(cache, target)['complete_visual_owner_bytes']
    require(before - after == released_bytes == 8 * target.numel(),
            'OWNER_AUDIT_FAILED: actual original-map byte release differs')
    return {'layout': 'shared_original_map_and_optional_sparse_owner_map',
        'same_original_position_values': True, 'obsolete_Hybrid_clone_released': True,
        'original_visual_rows': target.numel(), 'stored_visual_rows': owner0.positions.numel(),
        'sparse_owner_positions_not_replaced_by_original_positions': True,
        'before_complete_visual_bytes': before, 'after_complete_visual_bytes': after,
        'actual_released_capacity_bytes': released_bytes, 'original_logical_length_preserved': True}


def validate_native_record(record):
    data = importlib.import_module('round13_inputs_v1')
    expected = _BINDINGS.get(record['id'])
    if expected is None:
        return {'matched': False, 'mismatches': {'id': 'native binding missing'}}
    fields = tuple(dict.fromkeys((*data.NATIVE_FIELDS, 'raw_question_sha256', 'preprocessing')))
    mismatches = {key: {'expected': expected[key], 'actual': record.get(key)}
                  for key in fields if record.get(key) != expected[key]}
    return {'matched': not mismatches, 'mismatches': mismatches,
            'reference': 'frozen native input identity only; no reference outputs in encoder'}


def write_new(path, value):
    path = Path(path)
    if path.name == 'manifest.json':
        value = dict(value)
        value.update(runner_schema=SCHEMA, source_driver_contract='Round13 production factories over byte-frozen complete native M1 driver',
            runtime_process={'PID': os.getpid(), 'parent_PID': os.getppid(), 'source_snapshot_root': str(ROOT)},
            source_binding='explicit Round13 no-answer INPUT_MANIFEST and frozen native identities',
            phase_is_confirmatory=False, frozen_base_lineage_numeric_fields_unchanged=True,
            round13_candidate_assets=value.pop('round10_candidate_assets', None),
            parameter_fit_in_runner=False, technical_or_oracle_candidate_admitted=False)
    elif path.parent.name == 'records':
        value['round13_schema'] = SCHEMA
        if 'round10_budget_vs_original_R7_low' in value:
            value['round13_physical_budget'] = value.pop('round10_budget_vs_original_R7_low')
        if value.get('status') == 'PASS':
            proof = validate_native_record(value)
            value['round13_native_input_binding'] = proof
            trace = list(_LAST_GENERATION_TRACE)
            value['round13_free_generation_trace'] = {'scope': 'actual raw last-position pre-HF-processor logits on every generation forward',
                'step_count': len(trace), 'steps': trace, 'all_output_steps_observed': len(trace) == value['output_tokens'],
                'used_for_encoder_or_decoding_decisions': False}
            if not proof['matched'] or len(trace) != value['output_tokens']:
                value['status'] = 'INPUT_MISMATCH'
                value['error'] = 'Native input identity or complete generation trace mismatch'
                value['untrusted_generated_output_token_ids'] = value.get('output_token_ids')
                value['untrusted_raw_prediction'] = value.get('raw_prediction')
                value['output_token_ids'] = value['raw_prediction'] = None
    elif path.name == 'complete.json':
        value = {**value, 'round13_schema': SCHEMA, 'phase_is_confirmatory': False,
            'new_free_generation_answer_events': value['recorded_QA'], 'parameter_fit_in_runner': False,
            'technical_or_oracle_candidate_admitted': False, 'canonical_SourceRun_admission_claimed': False}
    return _write_base(path, value)


@contextmanager
def driver_overrides():
    require(sha(ROOT / 'code/round10_runner_v1.py') == BASE_RUNNER_SHA, 'frozen native driver changed')
    protocol = importlib.import_module('m1_protocol')
    prepare0, generate0 = protocol.prepare_m1_inputs, protocol.m1_generate_cached
    def prepare(*args, **kwargs):
        global _LAST_NATIVE_TILE_LAYOUT, _LAST_GENERATION_TRACE
        batch = prepare0(*args, **kwargs)
        layout = frozen.validate_native_tile_layout(batch)
        n = layout['actual_visual_tokens']
        _LAST_NATIVE_TILE_LAYOUT = {**layout, 'thumbnail_visual_rows': [n - 256, n],
            'thumbnail_derived_from_image_processor_native_tile_order': True}
        _LAST_GENERATION_TRACE = []
        return batch
    def generate(model, batch, cache, **kwargs):
        global _LAST_GENERATION_TRACE
        import torch
        _LAST_GENERATION_TRACE = []
        def observe(module, positional, output):
            require(output.logits is not None and bool(torch.isfinite(output.logits[0, -1]).all()),
                    'NONFINITE_LOGITS: actual generation step')
            top = torch.topk(output.logits[0, -1].float(), 2)
            ids, values = top.indices.detach().cpu().tolist(), top.values.detach().cpu().tolist()
            _LAST_GENERATION_TRACE.append({'step': len(_LAST_GENERATION_TRACE), 'argmax_id': ids[0],
                'second_id': ids[1], 'top1_logit': values[0], 'top2_logit': values[1],
                'top1_top2_margin': values[0] - values[1]})
        handle = model.register_forward_hook(observe)
        try:
            return generate0(model, batch, cache, **kwargs)
        finally:
            handle.remove()
    profiles = deepcopy(frozen.PROFILES)
    profiles['r7_joint_k1_gt_cond']['require_43_stages'] = False
    changes = {'SCHEMA': SCHEMA, 'MODES': MODES, 'PROFILES': profiles,
        'candidate_assets': candidate_assets, 'validate_config': validate_config,
        'build_candidate_owners': build_candidate_owners, 'round10_budget_guard': physical_budget_guard,
        'runtime_source_manifest': runtime_source_manifest, 'write_new': write_new,
        'share_original_visual_position_map': share_original_visual_position_map}
    saved = {key: getattr(frozen, key) for key in changes}
    loader = old_inputs.load_round10_inputs
    try:
        for key, value in changes.items():
            setattr(frozen, key, value)
        old_inputs.load_round10_inputs = load_runner_inputs
        protocol.prepare_m1_inputs, protocol.m1_generate_cached = prepare, generate
        yield
    finally:
        protocol.prepare_m1_inputs, protocol.m1_generate_cached = prepare0, generate0
        old_inputs.load_round10_inputs = loader
        for key, value in saved.items():
            setattr(frozen, key, value)


def main():
    with driver_overrides():
        return frozen.main()


if __name__ == '__main__':
    raise SystemExit(main())
