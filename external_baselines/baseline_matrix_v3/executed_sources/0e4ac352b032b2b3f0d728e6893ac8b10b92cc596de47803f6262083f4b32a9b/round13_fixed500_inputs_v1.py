"""Input-only admission for frozen P on the observed M1 TextVQA fixed500.

No references, raw baseline paths, predictions or scores are opened here.
Offline preparation filters old inputs/native identities into these artifacts.
The P codec/selector/f0 source graph is fixed separately from new admission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import dataset_registry as registry

ROLE = 'round13_observed_fixed500_comparison'
SCHEMA = 'trajkv.round13.observed_fixed500_inputs.v1'
SELECTION_SCHEMA = 'trajkv.round13.fixed500_selection.v1'
ALGORITHM_SCHEMA = 'trajkv.round13.selected_P_algorithm.v1'
RUNNER_SCHEMA = 'trajkv.round13.fixed500_comparison.raw.v1'
SELECTED_NAME = 'TrajKV_R13_Pbest_lowcap_v1'
MODEL_REVISION = '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
SUFFIX = '\nAnswer the question using a single word or phrase.'
PRIMARY_METRIC_ID = 'textvqa_mmf_ten_vote_leave_one_out_v1'
NATIVE_FIELDS = ('id', 'original_image_sha256', 'raw_question_sha256', 'rendered_question_sha256',
    'full_prompt_hash', 'chat_template_hash', 'input_ids_sha256', 'question_suffix_sha256',
    'native_position_ids_sha256', 'compression_boundary', 'actual_visual_token_count', 'model_commit', 'dtype')
FORBIDDEN = {'answers', 'labels', 'scores', 'refs', 'references', 'reference_answers', 'reference_path',
    'old_logits_path', 'old_raw_predictions_path', 'prediction_cache_path', 'raw_prediction', 'output_token_ids'}


def require(value, message):
    if not value:
        raise ValueError('ROUND13_FIXED500_INPUTS: ' + message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def input_only(value):
    if isinstance(value, dict):
        require(not set(value) & FORBIDDEN, 'worker artifact contains forbidden model-output/reference fields')
        for child in value.values():
            input_only(child)
    elif isinstance(value, list):
        for child in value:
            input_only(child)
    elif isinstance(value, str):
        require(not any(word in value.replace('\\', '/') for word in
                ('records.jsonl', 'scored_records', '/r7_matrix500_v2/', '/p2_views/')),
                'old prediction/raw path entered worker artifact')


def load_inputs(path, manifest_path=None, expected_sha256=None, expected_role=ROLE):
    require(expected_role == ROLE and manifest_path is not None, 'explicit observed fixed500 input role/manifest required')
    manifest = read(manifest_path)
    input_only(manifest)
    require(manifest.get('schema') == SCHEMA and manifest.get('role') == ROLE
            and manifest.get('previously_observed') is True and manifest.get('independent_holdout_claimed') is False
            and manifest.get('answers_in_encoder_inputs') is False
            and manifest.get('old_predictions_or_logits_in_inference_artifact') is False,
            'unknown fixed500 filtered input contract')
    path = Path(path)
    require(path.name.endswith('_inputs.jsonl') and path.name in manifest['input_files'], 'unbound no-answer input file')
    entry = manifest['input_files'][path.name]
    digest = sha(path)
    require(digest == entry['sha256'] and path.stat().st_size == entry['bytes']
            and entry['count'] == 500 and entry['role'] == ROLE and entry['answers_included'] is False,
            'fixed500 input bytes/count/role differs')
    require(expected_sha256 is None or expected_sha256 == digest, 'explicit input SHA differs')
    values = registry.load_inputs(path)
    input_only(values)
    require(len(values) == len({row['id'] for row in values}) == len({row['source_cluster_id'] for row in values}) == 500
            and [row['id'] for row in values] == entry['ids'] and all(row['dataset'] == 'textvqa' for row in values),
            'must retain all original500 Text questions/sources in frozen order')
    require(set(manifest['native_bindings']) == set(entry['ids']), 'full500 native identity coverage differs')
    for row in values:
        native = manifest['native_bindings'][row['id']]
        require(set(NATIVE_FIELDS) <= set(native) and native['original_image_sha256'] == row['original_image_sha256']
                and native['raw_question_sha256'] == row['question_sha256']
                and native['model_commit'] == MODEL_REVISION and native['dtype'] == 'float16'
                and native['preprocessing'] == {'max_patches': 8} and native['question_suffix'] == SUFFIX,
                'native input/image/model/prompt/cap identity differs')
    return values


def _pinned(config, stem):
    path, digest = config.get(stem + '_path'), config.get(stem + '_sha256')
    require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
            'fixed path/SHA binding required: ' + stem)
    return read(path)


def _candidate_numeric(candidate):
    return {key: value for key, value in candidate.items() if not key.endswith('_path')}


def validate_config_admission(config):
    input_only(config)
    require(config.get('schema') == RUNNER_SCHEMA and config.get('model_id') == 'M1'
            and config.get('model_revision') == MODEL_REVISION, 'explicit M1 fixed500 runner/revision required')
    require(config.get('input_role') == ROLE and config.get('cohort_kind') == 'fixed500_comparison',
            'observed comparison must not become development/fresh holdout role')
    require(type(config.get('seed')) is int and config['seed'] == 0 and config.get('max_new_tokens') == 64
            and config.get('question_suffix') == SUFFIX and config.get('preprocessing') == {'max_patches': 8},
            'native seed/greedy/prompt/cap contract differs')
    require(config.get('evaluation_contract', {}).get('version') == 'matrix_v3_metrics_v1'
            and config['evaluation_contract'].get('primary_metric_id') == PRIMARY_METRIC_ID,
            'frozen Text primary metric differs')
    manifest = _pinned(config, 'input_manifest')
    require(isinstance(config.get('inputs'), str) and sha(config['inputs']) == config.get('inputs_sha256'), 'input path/SHA required')
    values = load_inputs(config['inputs'], config['input_manifest_path'], config['inputs_sha256'], config['input_role'])
    require(config.get('input_count') == 500 and type(config.get('start')) is int and type(config.get('count')) is int
            and config['start'] >= 0 and 0 < config['count'] <= 125 and config['start'] + config['count'] <= 500,
            'fixed500 explicit legal slice1..125 required')
    receipt = _pinned(config, 'selection_receipt')
    selected = _pinned(config, 'selected_algorithm')
    input_only(receipt); input_only(selected)
    require(receipt.get('schema') == SELECTION_SCHEMA and receipt.get('status') == 'FROZEN_BEFORE_P_FIXED500'
            and receipt.get('fixed500_is_previously_observed') is True and receipt.get('independent_holdout_claimed') is False
            and receipt.get('new_candidate_fixed500_outputs_read_before_selection') is False
            and receipt.get('methods_and_assets_frozen_before_candidate_comparison') is True
            and receipt.get('input_manifest_sha256') == config['input_manifest_sha256'], 'selection must precede new P500 output')
    require(receipt.get('selected_methods') == [SELECTED_NAME] and config.get('selected_method_name') == SELECTED_NAME,
            'only frozen P is a new fixed500 execution; comparators must be reused')
    require(selected.get('schema') == ALGORITHM_SCHEMA and selected.get('name') == SELECTED_NAME
            and _candidate_numeric(selected['candidate']) == _candidate_numeric(config['candidate'])
            and config['candidate']['mode'] == 'P_best', 'selected P algorithm/candidate changed')
    require(any(item.get('name') == SELECTED_NAME and item.get('sha256') == config['selected_algorithm_sha256']
                for item in receipt.get('method_asset_locks', [])), 'selected algorithm absent from receipt locks')
    require(selected['variant_profile'] == config['variant_source']['name']
            and selected['variant_source_sha256'] == config['variant_source']['sha256'], 'original frozen variant differs')
    source_graph = config.get('expected_source_hashes', {})
    require(selected.get('algorithm_source_hashes') and selected.get('base_runtime_source_hashes')
            and all(source_graph.get(key) == digest for key, digest in {
                **selected['algorithm_source_hashes'], **selected['base_runtime_source_hashes']}.items()),
            'selected numeric/base source graph differs from fixed500 runtime')
    chosen = values[config['start']:config['start'] + config['count']]
    return {'schema': 'trajkv.round13.fixed500_admission.v1', 'rows': chosen, 'dataset': 'textvqa',
        'expected_ids': [row['id'] for row in chosen], 'all_input_ids': [row['id'] for row in values],
        'native_bindings': manifest['native_bindings'],
        'selected_native_bindings': {row['id']: manifest['native_bindings'][row['id']] for row in chosen},
        'selection_receipt': receipt, 'algorithm_selection': selected,
        'selection_receipt_sha256': config['selection_receipt_sha256'],
        'input_manifest_sha256': config['input_manifest_sha256'], 'inputs_sha256': config['inputs_sha256'],
        'role': ROLE, 'previously_observed': True, 'independent_holdout_claimed': False,
        'binding_scope': 'filtered original INPUT identities only; no old model outputs',
        'old_prediction_reuse_for_candidate': False}


def validate_execution_selection(config, identity):
    require(config.get('selected_method_name') == SELECTED_NAME
            and identity['algorithm_selection']['name'] == SELECTED_NAME
            and identity['selection_receipt']['selected_methods'] == [SELECTED_NAME], 'execution not frozen selected P')
    return SELECTED_NAME


def validate_native_record(record, bindings):
    require(record.get('id') in bindings, 'actual row outside full500 native identity')
    expected = bindings[record['id']]
    differences = {key: {'expected': expected[key], 'actual': record.get(key)}
                   for key in NATIVE_FIELDS if record.get(key) != expected[key]}
    if record.get('preprocessing') != expected['preprocessing']:
        differences['preprocessing'] = {'expected': expected['preprocessing'], 'actual': record.get('preprocessing')}
    return {'matched': not differences, 'mismatches': differences,
        'scope': 'same observed source/question/image/model/native prompt and position identities',
        'bitexact_logits_claimed': False, 'old_logits_or_output_used_by_model': False}
