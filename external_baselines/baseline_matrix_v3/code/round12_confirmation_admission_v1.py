"""Model-free admission shared by fresh static/policy confirmation runners.

Only newly materialized no-answer inputs and newly computed CPU processor
bindings are admitted. This module never opens refs or old prediction files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import round12_confirmation_data_v1 as data
from model_registry import get_model_spec

ROLE = data.ROLE
ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = data.SCHEMA
CPU_BINDING_SCHEMA = 'trajkv.round12.fresh_cpu_native_bindings.v1'
CPU_BINDING_CONFIG_SCHEMA = 'trajkv.round12.confirmation_native_binding_config.v1'
STATIC_RUNNER_SCHEMA = 'trajkv.round12.fresh_confirmation.raw.v1'
SUFFIX = '\nAnswer the question using a single word or phrase.'
METRIC_VERSION = 'matrix_v3_metrics_v1'
MODEL_ID = 'M1'
MODEL_REVISION = '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
BASELINE_MODES = ('FP16', 'R7low', 'R7high')
CPU_BINDING_SOURCE_FILES = ('round12_confirmation_native_bindings_v1.py',
    'round12_confirmation_admission_v1.py', 'round12_confirmation_data_v1.py',
    'm1_protocol.py', 'model_registry.py')
FORBIDDEN = {'refs', 'references', 'reference_answers', 'reference_path', 'answers', 'labels', 'scores',
             'old_native_bindings_path', 'old_logits_path', 'prediction_cache_path'}
INPUT_BINDING_FIELDS = (
    'id', 'original_image_sha256', 'raw_question_sha256', 'rendered_question_sha256',
    'full_prompt_hash', 'chat_template_hash', 'input_ids_sha256', 'question_suffix_sha256',
    'native_position_ids_sha256', 'compression_boundary', 'actual_visual_token_count',
    'model_commit', 'dtype',
)


def require(condition, message):
    if not condition:
        raise ValueError('ROUND12_CONFIRM_ADMISSION: ' + message)


sha, read = data.sha, data.read


def load_inputs(path, manifest_path=None, expected_sha256=None, expected_role=ROLE):
    require(manifest_path is not None, 'explicit fresh confirmation manifest required')
    return data.load_confirmation_inputs(path, manifest_path, expected_sha256, expected_role)


def validate_data_config(config, *, require_bindings=True):
    require(not (set(config) & FORBIDDEN), 'reference/old-prediction fields forbidden in model config')
    require(config.get('model_id') == MODEL_ID and config.get('model_revision') == MODEL_REVISION,
            'full official M1 revision required')
    require(config.get('seed') == 0 and type(config.get('seed')) is int, 'greedy seed0 required')
    require(config.get('question_suffix') == SUFFIX and config.get('max_new_tokens') == 64,
            'frozen prompt suffix/max-new64 required')
    require(config.get('preprocessing') == {'max_patches': 8}, 'fresh reference cap8 required')
    require(config.get('evaluation_contract', {}).get('version') == METRIC_VERSION, 'frozen metric version required')
    require(config.get('input_role') == ROLE and config.get('cohort_kind') == 'independent_confirmation',
            'explicit fresh role; development role cannot be translated')
    for field in ('input_manifest_path', 'inputs', 'frozen_decision_path', 'model_path', 'model_source_lock'):
        require(isinstance(config.get(field), str) and config[field], 'explicit path required: ' + field)
    for field, digest in (('input_manifest_path', 'input_manifest_sha256'), ('inputs', 'inputs_sha256'),
                          ('frozen_decision_path', 'frozen_decision_sha256')):
        expected = config.get(digest)
        require(isinstance(expected, str) and len(expected) == 64 and sha(config[field]) == expected,
                'immutable source/hash differs: ' + field)
    manifest = read(config['input_manifest_path'])
    require(manifest.get('schema') == INPUT_SCHEMA and manifest['decision']['sha256'] == config['frozen_decision_sha256'],
            'fresh data manifest binds a different decision')
    decision = read(config['frozen_decision_path'])
    require(decision.get('schema') == data.DECISION_SCHEMA and decision.get('confirmation_outputs_read') is False
            and decision.get('methods_and_assets_frozen_before_confirmation') is True
            and decision.get('selection_scope') == 'observed_development_only'
            and decision.get('counts') == manifest['counts'] and decision.get('seed') == manifest['seed'],
            'confirmation decision/size/seed lineage differs')
    values = load_inputs(config['inputs'], config['input_manifest_path'], config['inputs_sha256'], config['input_role'])
    require(config.get('input_count') == len(values), 'full input count differs')
    start, count = config.get('start'), config.get('count')
    require(type(start) is int and type(count) is int and start >= 0 and count > 0 and start + count <= len(values),
            'explicit valid input slice required')
    chosen = values[start:start + count]
    require(len({r['dataset'] for r in chosen}) == 1, 'one domain per run required')
    require(len({r['id'] for r in chosen}) == len(chosen), 'duplicate IDs')
    identity = {'schema': 'trajkv.round12.fresh_confirmation_admission.v1', 'role': ROLE,
        'rows': chosen, 'input_count': len(values), 'expected_ids': [r['id'] for r in chosen],
        'dataset': chosen[0]['dataset'], 'decision': decision,
        'frozen_decision_sha256': config['frozen_decision_sha256'],
        'input_manifest_sha256': config['input_manifest_sha256'], 'inputs_sha256': config['inputs_sha256'],
        'source_selection_native_outputs_read': False, 'old_predictions_admitted': False}
    if require_bindings:
        path, digest = config.get('native_bindings_path'), config.get('native_bindings_sha256')
        require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
                'new CPU native bindings path/SHA required')
        bundle = read(path)
        require(bundle.get('schema') == CPU_BINDING_SCHEMA
                and bundle.get('status') == 'COMPLETE_CPU_PROCESSOR_BINDINGS_NO_MODEL'
                and bundle.get('input_manifest_sha256') == config['input_manifest_sha256']
                and bundle.get('frozen_decision_sha256') == config['frozen_decision_sha256']
                and bundle.get('model_revision') == MODEL_REVISION
                and bundle.get('question_suffix') == SUFFIX and bundle.get('preprocessing') == {'max_patches': 8}
                and bundle.get('model_forward_calls') == 0 and bundle.get('model_loaded') is False
                and bundle.get('old_logits_or_prediction_bindings_read') is False,
                'native bindings are not fresh model-free processor evidence')
        expected_sources = {'code/' + name: sha(ROOT / 'code' / name) for name in CPU_BINDING_SOURCE_FILES}
        require(bundle.get('source_hashes') == expected_sources, 'CPU binding maker/source protocol has changed')
        require(bundle['processor_file_identity']['model_source_lock_sha256'] == sha(config['model_source_lock']),
                'CPU/GPU official model source lock differs')
        for library, library_digest in bundle['processor_library_source_sha256'].items():
            require(Path(library).is_file() and sha(library) == library_digest, 'processor/tokenizer library changed after CPU binding')
        require(bundle['input_files'].get(Path(config['inputs']).name, {}).get('sha256') == config['inputs_sha256'],
                'CPU binding bundle covers a different input cohort')
        native = bundle['bindings']
        for row in chosen:
            require(row['id'] in native and native[row['id']]['original_image_sha256'] == row['original_image_sha256']
                    and native[row['id']]['raw_question_sha256'] == row['question_sha256'], 'fresh native source mismatch')
        identity.update(native_bindings={r['id']: native[r['id']] for r in chosen},
                        native_bindings_sha256=digest, native_binding_source_files=bundle['source_hashes'],
                        native_processor_library_source_sha256=bundle['processor_library_source_sha256'],
                        native_binding_protocol='new CPU processor, no model/logits; planned serial positions')
    return identity


def validate_config_admission(config):
    return validate_data_config(config, require_bindings=True)


def validate_execution_selection(config, identity):
    logical = config.get('confirmation_mode')
    selected = config.get('selected_method_name')
    if logical in BASELINE_MODES:
        require(logical in identity['decision']['comparators'], 'baseline is absent from frozen comparison decision')
        require(selected is None, 'baseline cannot masquerade as selected candidate')
    else:
        require(isinstance(selected, str)
                and selected in (identity['decision']['selected_methods'] + identity['decision']['comparators']),
                'candidate/policy/comparator not declared by frozen development-only decision')
    return logical or selected


def validate_native_record(record, bindings):
    """Compare fresh input plans, never first logits or answer tokens.

Action runners pass the original low-probe input record before adding a crop,
then retain a separately named final action input record. New crops are not
required to equal the original single-image prompt.
"""
    require(record.get('id') in bindings, 'row not covered by fresh CPU binding')
    expected = bindings[record['id']]
    mismatches = {k: {'expected': expected[k], 'actual': record.get(k)}
                  for k in INPUT_BINDING_FIELDS if record.get(k) != expected[k]}
    if record.get('preprocessing') != {'max_patches': 8}:
        mismatches['preprocessing'] = {'expected': {'max_patches': 8}, 'actual': record.get('preprocessing')}
    return {'matched': not mismatches, 'mismatches': mismatches,
            'scope': 'fresh input/prompt/IDs/planned serial positions only',
            'bitexact_logits_or_answers_claimed': False, 'old_native_evidence_reused': False}


def data_config_template():
    return {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_path': None, 'model_source_lock': None,
        'seed': 0, 'max_new_tokens': 64, 'question_suffix': SUFFIX, 'preprocessing': {'max_patches': 8},
        'evaluation_contract': {'version': METRIC_VERSION, 'primary_metric_id': None},
        'input_role': ROLE, 'cohort_kind': 'independent_confirmation',
        'input_manifest_path': None, 'input_manifest_sha256': None,
        'inputs': None, 'inputs_sha256': None, 'input_count': None, 'start': 0, 'count': None,
        'frozen_decision_path': None, 'frozen_decision_sha256': None,
        'native_bindings_path': None, 'native_bindings_sha256': None}
