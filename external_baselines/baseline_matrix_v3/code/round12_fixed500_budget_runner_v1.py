"""Fixed500 admission around the frozen R12 budget-v3 numerical action.

Only role/input/asset/source-graph/coverage admission is new. This adapter does
not choose a winner, create selection receipts, read old outputs or launch a
job. Initial input witnesses come from filtered metadata, never old logits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round12_fixed500_inputs_v1 as data
import round9_runner_v1 as frozen_contract

SCHEMA = 'trajkv.round12.fixed500_budget.raw.v1'
ALGORITHM_SCHEMA = 'trajkv.round12.selected_budget_algorithm.v1'
ROLE = data.ROLE
MODES = ('Z-budget-K2', 'Zrandom-budget-K2')
RUNTIME_FILES = (
    'round12_action_runner_v2.py', 'round12_action_probe_v1.py', 'round12_multiimage_v1.py',
    'round12_inputs_v1.py', 'm1_protocol.py', 'cache_bridge.py', 'attention_bridge.py',
    'ownership.py', 'evaluate.py', 'model_registry.py', 'dataset_registry.py',
    'round9_runner_v1.py', 'round9_inputs_v1.py', 'r7_legacy_bridge_v1.py', 'r7_legacy_bridge_v2.py',
    'round12_action_runner_v3.py', 'round12_budget_action_runner_v3.py',
    'round12_budget_multiimage_v1.py', 'round12_static_bank_v1.py',
    'round12_static_codec_v1.py', 'round12_static_attention_v1.py',
    'round12_fixed500_budget_runner_v1.py', 'round12_fixed500_inputs_v1.py',
)
NUMERICAL_PINS = {
    'code/round12_budget_action_runner_v3.py': '88a3e81982bbd01b538ea22ca56b46de2d350657dab21748f2ed283b794c70b0',
    'code/round12_budget_multiimage_v1.py': '5e054068419a6c97dc7cded7a334e59409a98488e757507af0e56182669f15ff',
    'code/round12_action_runner_v2.py': 'fcb29ff52624ab661149768742a2c98f729f4059654b12cf07c8fc56dbf2429e',
}
ALGORITHM_FIELDS = {
    'schema', 'name', 'action_mode', 'a_candidate', 'variant_profile',
    'variant_source_sha256', 'budget_rule', 'ROI_rule', 'view_rule',
    'algorithm_source_hashes', 'selection_scope', 'new_fixed500_outputs_read',
}
BUDGET_RULE = {
    'scope': 'first_suffix_total_persistent_KV',
    'baseline': 'actual_original_low_total_after_complete_question_suffix_before_first_answer_sampling',
    'predictor': '6382N+91657+57568*(full_length-N)',
    'actual_guard': 'cache_storage.total_bytes_before_first_answer_sampling',
    'commit': 'release_original_low_GPU_cache_owners_tensors_before_candidate_CUDA_inputs_and_prefill',
    'on_actual_budget_rejection': 'abort_before_sampling_release_candidate_rebuild_original_low_onsite_all_costs_charged',
    'on_no_CPU_candidate': 'reuse_rolled_back_original_low_cache_then_full_original_suffix',
    'allocator_peak_and_encoding_workspaces_bounded_by_KV_cap': False,
}
ROI_RULE = {
    'area_divisor_each_axis': 4,
    'probe_max_new_tokens': 3,
    'question_tokens': 'last_at_most4_non_special_raw_question_tokens',
    'capture_layers': 'last6',
    'related': 'equal_layer_head_query_conditional_attention_to_actual_low_reconstructed_WHT_K_then_max_mass_window',
    'random_seed': "SHA256('TrajKV-R12-random-ROI-v1|' + row['id'])",
    'related_and_random_bbox_shape_identical': True,
}
VIEW_RULE = {
    'global_cap': 1, 'ROI_caps': list(range(1, 9)),
    'selection': 'maximum_actual_feasible_ROI_tile_count_then_lower_predicted_bytes_then_lower_cap',
    'enumeration': 'native_processor_CPU_only_no_candidate_model_forward',
    'two_view_template': 'same_explicit_two_view_text_as_frozen_Z',
    'no_two_view_fit_fallback': 'ROI_only_native_cap1_A_then_original_low_a0_if_not_fit',
    'candidate_encoder': 'A_K2_V1_original_low_concrete_K23_V20_L43',
    'original_cap8_pixel_preservation_claimed': False,
}
_CONFIG = _IDENTITY = _SPEC = None
require, sha = data.require, data.sha


def runtime_source_manifest():
    actual = {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}
    require(all(actual.get(name) == digest for name, digest in NUMERICAL_PINS.items()),
            'frozen budget-v3 numerical source changed')
    return actual


def normalize_candidate(candidate):
    require(isinstance(candidate, dict) and set(candidate) <= {'mode', 'key_book_path', 'key_book_sha256'}
            and candidate.get('mode') == 'A', 'only declared A K2 book candidate fields required')
    require(isinstance(candidate.get('key_book_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', candidate['key_book_sha256']), 'A book SHA required')
    return {key: value for key, value in candidate.items() if not key.endswith('_path')}


def validate_algorithm_spec(spec, config, identity):
    require(isinstance(spec, dict) and set(spec) == ALGORITHM_FIELDS and spec['schema'] == ALGORITHM_SCHEMA,
            'selected budget algorithm has unknown/missing fields')
    require(isinstance(spec['name'], str) and re.fullmatch('[A-Za-z0-9_.-]{1,128}', spec['name']) is not None
            and spec['name'] == config['selected_method_name'], 'selected budget method name differs')
    require(spec['action_mode'] == config['action_mode'] and spec['action_mode'] in MODES,
            'selected related/random mode changed')
    require(normalize_candidate(spec['a_candidate']) == normalize_candidate(config['a_candidate']),
            'selected A book hash/format changed')
    require(spec['variant_profile'] == config['variant_source']['name'] == 'r7_joint_k1_gt_cond'
            and spec['variant_source_sha256'] == config['variant_source']['sha256'] == frozen_contract.ORIGINAL_CONFIG_SHA,
            'original low cond43 lineage changed')
    require(frozen_contract.canonical_sha(spec['budget_rule']) == frozen_contract.canonical_sha(BUDGET_RULE)
            and frozen_contract.canonical_sha(spec['ROI_rule']) == frozen_contract.canonical_sha(ROI_RULE)
            and frozen_contract.canonical_sha(spec['view_rule']) == frozen_contract.canonical_sha(VIEW_RULE),
            'selected budget/ROI/native-view/fallback rules differ')
    require(spec['selection_scope'] == 'observed_development_before_fixed500_comparison'
            and spec['new_fixed500_outputs_read'] is False, 'selection must precede this candidate fixed500 output')
    require(spec['algorithm_source_hashes'] == runtime_source_manifest(), 'complete selected algorithm source graph changed')
    return spec


def selected_algorithm(config, identity):
    path, digest = config.get('selected_algorithm_path'), config.get('selected_algorithm_sha256')
    require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
            'selected budget algorithm path/SHA required')
    locks = identity['selection_receipt']['method_asset_locks']
    require(any(item.get('sha256') == digest and Path(item['path']).resolve() == Path(path).resolve()
                for item in locks), 'selected budget spec absent from fixed500 receipt asset locks')
    return validate_algorithm_spec(data.read(path), config, identity)


def validate_config(config):
    global _CONFIG, _IDENTITY, _SPEC
    require(config.get('schema') == SCHEMA and config.get('action_mode') in MODES, 'fixed500 budget schema/mode required')
    identity = data.validate_config_admission(config)
    data.validate_execution_selection(config, identity)
    method = config.get('method', {})
    require(method.get('family') == 'trajkv_round12_fixed500_budget_comparison'
            and method.get('information_condition') == 'query_unknown_prefix_then_query_conditional_action',
            'explicit fixed500 budget family/information condition required')
    require(config.get('budget_scope') == BUDGET_RULE['scope']
            and config.get('position_owner_layout') == 'single_shared_visual_position_map',
            'strict actual total-KV scope/original VP layout required')
    source = config.get('variant_source', {})
    require(source.get('name') == 'r7_joint_k1_gt_cond' and source.get('sha256') == frozen_contract.ORIGINAL_CONFIG_SHA,
            'original low profile required')
    require(config.get('expected_asset_sha256') == frozen_contract.PROFILES[source['name']]['assets'],
            'original f0/policy/base assets changed')
    require(set(config.get('a_candidate', {})) == {'mode', 'key_book_path', 'key_book_sha256'},
            'explicit A book asset path/SHA required')
    normalize_candidate(config['a_candidate'])
    require(sha(config['a_candidate']['key_book_path']) == config['a_candidate']['key_book_sha256'], 'A book changed')
    for field in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source', 'model_path', 'model_source_lock', 'scope'):
        require(isinstance(config.get(field), str) and config[field], 'explicit dependency required: ' + field)
    frozen_contract.frozen_variant(config)
    require(config.get('expected_source_hashes') == runtime_source_manifest(), 'complete current runtime graph differs')
    spec = selected_algorithm(config, identity)
    _CONFIG, _IDENTITY, _SPEC = config, identity, spec
    return config


def load_fixed_inputs(path, manifest_path=None, expected_sha256=None):
    return data.load_inputs(path, manifest_path, expected_sha256, ROLE)


def verify_initial_input(processor, image, batch, row, config):
    # The frozen action helper already performs alternate-question prefix and
    # native pixel equality checks using metadata-only native_bindings. No role
    # translation is used: this sidecar is explicitly fixed500 comparison data.
    proof = _ORIGINAL_VERIFY(processor, image, batch, row, config)
    import round12_action_runner_v2 as action
    template = processor.chat_template
    template_sha = hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else action.canonical(template)
    rendered = row['question'] + data.SUFFIX
    actual = {**row, **action.input_evidence(batch), 'model_commit': data.MODEL_REVISION,
        'dtype': 'float16', 'preprocessing': {'max_patches': 8},
        'raw_question_sha256': row['question_sha256'],
        'rendered_question_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
        'chat_template_hash': template_sha}
    fixed = data.validate_native_record(actual, _IDENTITY['native_bindings'])
    require(fixed['matched'], 'initial low fixed500 native INPUT metadata differs: ' + str(fixed['mismatches']))
    proof.update(fixed500_native_input_binding=fixed, comparison_role=ROLE,
        previously_observed_fixed500=True, independent_holdout_claimed=False,
        actual_original_chat_template_hash=template_sha,
        old_predictions_or_logits_read=False)
    return proof


def write_new(path, value):
    path = Path(path)
    if path.name == 'manifest.json':
        value.update(runner_schema=SCHEMA, comparison_role=ROLE,
            previously_observed_fixed500=True, independent_holdout_claimed=False,
            phase_is_confirmatory=False, selected_method_name=value['config']['selected_method_name'],
            selected_algorithm_sha256=value['config']['selected_algorithm_sha256'],
            selection_receipt_sha256=value['config']['selection_receipt_sha256'],
            source_numerical_action_driver='round12_budget_action_runner_v3.py',
            old_prediction_or_logits_used_by_model=False,
            native_binding_scope='filtered fixed500 INPUT metadata only; original low stage validated before altered views')
    elif path.parent.name == 'records':
        value.update(round12_fixed500_budget_schema=SCHEMA, comparison_role=ROLE,
            previously_observed_fixed500=True, independent_holdout_claimed=False)
        proof = value.get('initial_native_input_binding', {})
        if proof.get('actual_original_chat_template_hash'):
            value['chat_template_hash'] = proof['actual_original_chat_template_hash']
        if value.get('status') == 'PASS':
            require(proof.get('fixed500_native_input_binding', {}).get('matched') is True,
                    'PASS candidate missing actual original low input comparison')
            require(value.get('strict_total_cache_budget_satisfied') is True,
                    'PASS candidate missing actual first-suffix total-KV budget acceptance')
    elif path.name == 'complete.json':
        value.update(round12_fixed500_budget_schema=SCHEMA, comparison_role=ROLE,
            previously_observed_fixed500=True, independent_holdout_claimed=False,
            phase_is_confirmatory=False, old_prediction_or_logits_used_by_model=False)
    return _BUDGET_WRITE(path, value)


@contextmanager
def driver_overrides():
    global _ORIGINAL_VERIFY, _BUDGET_WRITE
    import round12_action_runner_v2 as action
    import round12_budget_action_runner_v3 as numerical
    _ORIGINAL_VERIFY = action.verify_low_input_binding
    _BUDGET_WRITE = numerical.write_new
    changes = dict(SCHEMA=SCHEMA, RUNTIME_FILES=RUNTIME_FILES,
        validate_config=validate_config, variants_from_config=numerical.variants_from_config,
        run_action_row=numerical.run_budget_row, write_new=write_new,
        load_round12_inputs=load_fixed_inputs, verify_low_input_binding=verify_initial_input)
    saved = {key: getattr(action, key) for key in changes}
    try:
        for key, value in changes.items():
            setattr(action, key, value)
        yield action
    finally:
        for key, value in saved.items():
            setattr(action, key, value)


def config_template(dataset=None):
    config = data.config_template()
    if dataset is not None:
        require(dataset in data.PRIMARY_METRIC_IDS, 'unknown dataset template')
        config['evaluation_contract']['primary_metric_id'] = data.PRIMARY_METRIC_IDS[dataset]
    config.update(schema=SCHEMA, action_mode=None,
        method={'id': None, 'family': 'trajkv_round12_fixed500_budget_comparison',
                'information_condition': 'query_unknown_prefix_then_query_conditional_action'},
        budget_scope=BUDGET_RULE['scope'], a_candidate={'mode': 'A', 'key_book_path': None, 'key_book_sha256': None},
        variant_source=None, variant=None, expected_asset_sha256=None,
        legacy_a2=None, legacy_experiments=None, legacy_scripts=None, kivi_source=None,
        position_owner_layout='single_shared_visual_position_map',
        scope='selected budget action on previously observed M1 fixed500; no independent holdout claim',
        status='UNCONFIGURED_NO_BUDGET_WINNER_OR_FIXED500_RUN_SELECTED')
    return config


def algorithm_template():
    return dict(schema=ALGORITHM_SCHEMA, name=None, action_mode=None,
        a_candidate={'mode': 'A', 'key_book_sha256': None},
        variant_profile='r7_joint_k1_gt_cond', variant_source_sha256=frozen_contract.ORIGINAL_CONFIG_SHA,
        budget_rule=deepcopy(BUDGET_RULE), ROI_rule=deepcopy(ROI_RULE), view_rule=deepcopy(VIEW_RULE),
        algorithm_source_hashes=None,
        selection_scope='observed_development_before_fixed500_comparison', new_fixed500_outputs_read=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-config', action='store_true')
    parser.add_argument('--print-template', action='store_true')
    parser.add_argument('--print-algorithm-template', action='store_true')
    parser.add_argument('--dataset', choices=tuple(data.PRIMARY_METRIC_IDS))
    args = parser.parse_args()
    if args.print_template or args.print_algorithm_template:
        require(not (args.print_template and args.print_algorithm_template), 'choose one unconfigured template')
        require(args.config is None and args.output is None and not args.validate_config, 'template cannot run a configured method')
        value = algorithm_template() if args.print_algorithm_template else config_template(args.dataset)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    require(args.config is not None, 'actual configured fixed500 budget method required')
    config = validate_config(data.read(args.config))
    if args.validate_config:
        print(json.dumps(dict(status='PASS_FIXED500_BUDGET_CONFIG_ONLY', comparison_role=ROLE,
            domain=_IDENTITY['dataset'], selected_method_name=config['selected_method_name'],
            full_cohort=500, slice_count=len(_IDENTITY['rows']), model_or_GPU_calls=0,
            independent_holdout_claimed=False), ensure_ascii=False))
        return 0
    require(args.output is not None, 'new output directory required for actual model execution')
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], '--config', str(args.config), '--output', str(args.output)]
        with driver_overrides() as action:
            return action.main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
