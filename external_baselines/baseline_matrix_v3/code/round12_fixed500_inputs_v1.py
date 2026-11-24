"""Explicit admission for the already-observed M1 fixed500 comparison.

The preparer extracts INPUT identities from old raw containers and checks all
500 against frozen input bytes and FP16/R7 metadata. Model admission opens
only the no-answer inputs and a filtered metadata sidecar: no predictions,
scores, logit tensors or paths to old raw prediction files enter inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import dataset_registry as registry

ROLE = 'round12_fixed500_comparison'
SCHEMA = 'trajkv.round12.fixed500_inputs.v1'
SELECTION_SCHEMA = 'trajkv.round12.fixed500_selection.v1'
RUNNER_SCHEMA = 'trajkv.round12.fixed500_static.raw.v1'
SUFFIX = '\nAnswer the question using a single word or phrase.'
MODEL_REVISION = '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
METRIC_VERSION = 'matrix_v3_metrics_v1'
EXPECTED_PER_DOMAIN = 500
PRIMARY_METRIC_IDS = {
    'textvqa': 'textvqa_mmf_ten_vote_leave_one_out_v1',
    'infovqa': 'infovqa_anls_qwenvl_vendor_rawlen_ge0p5_v1',
    'docvqa': 'docvqa_anls_qwenvl_vendor_rawlen_ge0p5_v1',
    'chartqa': 'chartqa_pix2struct_relaxed_5pct_v1',
}
FORBIDDEN = {'answers', 'labels', 'scores', 'refs', 'references', 'reference_answers', 'reference_path',
             'old_logits_path', 'old_raw_predictions_path', 'prediction_cache_path'}
NATIVE_FIELDS = ('id', 'original_image_sha256', 'raw_question_sha256', 'rendered_question_sha256',
    'full_prompt_hash', 'chat_template_hash', 'input_ids_sha256', 'question_suffix_sha256',
    'native_position_ids_sha256', 'compression_boundary', 'actual_visual_token_count', 'model_commit', 'dtype')


def require(value, message):
    if not value:
        raise ValueError('ROUND12_FIXED500: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as s:
        return [json.loads(line) for line in s if line.strip()]


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as s:
        json.dump(obj, s, ensure_ascii=False, indent=2, allow_nan=False)
        s.write('\n')


def metadata_only(raw_path):
    """Extraction-stage only; output deliberately excludes predictions/logits."""
    result = {}
    with Path(raw_path).open(encoding='utf-8-sig') as s:
        for line in s:
            if not line.strip():
                continue
            raw = json.loads(line)
            require(raw['id'] not in result, 'duplicate metadata source ID')
            require(raw['status'] == 'PASS' and raw['model_id'] == 'M1'
                    and raw['model_commit'] == MODEL_REVISION and raw['dtype'] == 'float16',
                    'old baseline metadata must be completed official M1 FP16 inference')
            item = {k: raw[k] for k in NATIVE_FIELDS}
            for field in ('source_id', 'source_cluster_id', 'question_id', 'question_sha256', 'dataset',
                          'dataset_revision', 'split', 'preprocessing', 'question_suffix', 'native_position_ids_shape'):
                item[field] = raw[field]
            require(item['preprocessing'] == {'max_patches': 8} and item['question_suffix'] == SUFFIX,
                    'old baseline native input contract differs')
            result[raw['id']] = item
    return result


def prepare(registry_root, r7_root, fp16_score_root, output, datasets=('infovqa', 'docvqa')):
    registry_root, r7_root, fp16_score_root, output = map(Path, (registry_root, r7_root, fp16_score_root, output))
    require(set(datasets) <= set(registry.DATASETS) and len(set(datasets)) == len(datasets), 'unknown/duplicate dataset')
    lock_path = registry_root / 'source_lock.json'
    lock = read(lock_path)
    manifest = {'schema': SCHEMA, 'role': ROLE, 'status': 'READY_FILTERED_INPUT_METADATA_ONLY',
        'fixed_subset': 'original matrix_v3 p2 fixed500, no reselection', 'previously_observed': True,
        'independent_holdout_claimed': False, 'answers_in_encoder_inputs': False,
        'old_predictions_or_logits_in_inference_artifact': False,
        'source_lock_sha256': sha(lock_path), 'input_files': {}, 'native_bindings': {},
        'builder_source_sha256': sha(__file__)}
    proof = {'schema': 'trajkv.round12.fixed500_metadata_preparation_proof.v1', 'domains': {},
             'model_or_GPU_calls': 0, 'source_question_image_model_prompt_position_checks': True,
             'old_raw_containers_read_for_metadata_extraction_only': True,
             'predictions_scores_and_logit_paths_discarded': True}
    for dataset in datasets:
        name = dataset + '_p2_inputs.jsonl'
        p = registry_root / name
        require(sha(p) == lock['files'][name]['sha256'] and p.stat().st_size == lock['files'][name]['bytes'],
                'frozen original fixed500 input bytes changed')
        inputs = registry.load_inputs(p)
        require(len(inputs) == EXPECTED_PER_DOMAIN and len({r['source_cluster_id'] for r in inputs}) == EXPECTED_PER_DOMAIN,
                'fixed500 must retain all500 original unique source rows')
        refs_name = dataset + '_p2_refs.jsonl'
        rp = registry_root / refs_name
        require(sha(rp) == lock['files'][refs_name]['sha256'] and rp.stat().st_size == lock['files'][refs_name]['bytes'],
                'frozen reference bytes changed')
        low_path = r7_root / ('K1_' + dataset + '_fixed500') / 'records.jsonl'
        high_path = r7_root / ('K2_' + dataset + '_fixed500') / 'records.jsonl'
        fp_summary_path = fp16_score_root / ('M1_F_' + dataset + '_p2_500_v1') / 'summary.json'
        fp_summary = read(fp_summary_path)
        fp_path = Path(fp_summary['integrity']['raw_dir']) / 'records.jsonl'
        require(fp_summary['integrity']['status'] == 'PASS_ALL_SHA_AND_EXPECTED_ID_CHECKS'
                and fp_summary['metric_contract_version'] == METRIC_VERSION
                and fp_summary['integrity']['raw_files']['records.jsonl'] == sha(fp_path),
                'existing FP16 score/raw identity differs')
        variants = {'R7low': metadata_only(low_path), 'R7high': metadata_only(high_path), 'FP16': metadata_only(fp_path)}
        expected_ids = [r['id'] for r in inputs]
        require(all(set(v) == set(expected_ids) for v in variants.values()), 'baseline metadata is not the identical full500')
        for row in inputs:
            current = variants['R7low'][row['id']]
            for mode, lookup in variants.items():
                require(lookup[row['id']] == current, 'FP16/R7 native metadata differ: ' + dataset + ' ' + row['id'] + ' ' + mode)
            for field in ('dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id', 'question_id',
                          'question_sha256', 'original_image_sha256'):
                require(current[field] == row[field], 'fixed input/native metadata differ: ' + field)
            require(current['raw_question_sha256'] == row['question_sha256'], 'raw question changed')
            manifest['native_bindings'][row['id']] = current
        manifest['input_files'][name] = {'sha256': sha(p), 'bytes': p.stat().st_size, 'count': 500,
            'dataset': dataset, 'ids': expected_ids, 'role': ROLE, 'answers_included': False}
        proof['domains'][dataset] = {'inputs': {'path': str(p), 'sha256': sha(p)},
            'refs': {'path': str(rp), 'sha256': sha(rp)},
            'metadata_raw_sources': {k: {'path': str(v), 'sha256': sha(v)} for k, v in
                [('R7low', low_path), ('R7high', high_path), ('FP16', fp_path)]},
            'all500_input_identities_match': True, 'questions': 500,
            'source_clusters': len({r['source_cluster_id'] for r in inputs})}
    output.mkdir(parents=True, exist_ok=False)
    mf = output / 'ROUND12_FIXED500_INPUT_MANIFEST_v1.json'
    write_new(mf, manifest)
    proof['input_manifest_sha256'] = sha(mf)
    write_new(output / 'PREPARATION_PROOF_v1.json', proof)
    return {'manifest': str(mf), 'sha256': sha(mf), 'datasets': list(datasets),
            'question_rows': len(manifest['native_bindings']), 'model_or_GPU_calls': 0,
            'role': ROLE, 'previously_observed_not_independent': True}


def load_inputs(path, manifest_path=None, expected_sha256=None, expected_role=ROLE):
    require(expected_role == ROLE and manifest_path is not None, 'explicit fixed500 comparison role/manifest required')
    manifest = read(manifest_path)
    require(manifest.get('schema') == SCHEMA and manifest.get('role') == ROLE
            and manifest.get('previously_observed') is True and manifest.get('independent_holdout_claimed') is False
            and manifest.get('answers_in_encoder_inputs') is False
            and manifest.get('old_predictions_or_logits_in_inference_artifact') is False, 'unknown fixed500 metadata contract')
    path = Path(path)
    entry = manifest['input_files'].get(path.name)
    require(entry is not None and sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
            and entry['count'] == 500 and entry['role'] == ROLE, 'fixed500 bytes/role changed')
    require(expected_sha256 is None or sha(path) == expected_sha256, 'explicit input SHA differs')
    values = registry.load_inputs(path)
    require(len(values) == 500 and [r['id'] for r in values] == entry['ids'], 'all500 identity/order changed')
    return values


def validate_config_admission(config):
    require(not (set(config) & FORBIDDEN), 'old outputs/logits or answers cannot enter model config')
    require(config.get('model_id') == 'M1' and config.get('model_revision') == MODEL_REVISION, 'full M1 revision required')
    require(config.get('input_role') == ROLE and config.get('cohort_kind') == 'fixed500_comparison',
            'cannot translate validation fixed500 into development or fresh holdout role')
    require(config.get('seed') == 0 and type(config.get('seed')) is int
            and config.get('max_new_tokens') == 64 and config.get('question_suffix') == SUFFIX
            and config.get('preprocessing') == {'max_patches': 8}, 'native greedy/prompt/cap contract changed')
    require(config.get('evaluation_contract', {}).get('version') == METRIC_VERSION, 'frozen metric version required')
    for path_key, sha_key in [('input_manifest_path', 'input_manifest_sha256'), ('inputs', 'inputs_sha256'),
                              ('selection_receipt_path', 'selection_receipt_sha256')]:
        require(isinstance(config.get(path_key), str) and isinstance(config.get(sha_key), str)
                and len(config[sha_key]) == 64 and sha(config[path_key]) == config[sha_key], 'fixed immutable path/SHA required: ' + path_key)
    manifest = read(config['input_manifest_path'])
    values = load_inputs(config['inputs'], config['input_manifest_path'], config['inputs_sha256'], config['input_role'])
    require(config.get('input_count') == 500 and type(config.get('start')) is int and type(config.get('count')) is int
            and config['start'] >= 0 and 0 < config['count'] <= 64 and config['start'] + config['count'] <= 500,
            'explicit fixed500 slices of1..64 required')
    chosen = values[config['start']:config['start'] + config['count']]
    require(config['evaluation_contract'].get('primary_metric_id') == PRIMARY_METRIC_IDS[chosen[0]['dataset']],
            'frozen per-domain primary metric ID differs')
    receipt = read(config['selection_receipt_path'])
    require(receipt.get('schema') == SELECTION_SCHEMA and receipt.get('fixed500_is_previously_observed') is True
            and receipt.get('independent_holdout_claimed') is False
            and receipt.get('new_candidate_fixed500_outputs_read_before_selection') is False
            and receipt.get('methods_and_assets_frozen_before_candidate_comparison') is True
            and receipt.get('input_manifest_sha256') == config['input_manifest_sha256'],
            'candidate decision must precede its new fixed500 result; old benchmark exposure disclosed')
    for key in ('selected_methods', 'comparators'):
        require(isinstance(receipt.get(key), list) and all(isinstance(x, str) and x for x in receipt[key])
                and len(set(receipt[key])) == len(receipt[key]), 'explicit unique method/comparator names required')
    require(bool(receipt['selected_methods']) and isinstance(receipt.get('method_asset_locks'), list)
            and bool(receipt['method_asset_locks']), 'frozen selected algorithms/policy asset locks required')
    require(config.get('selected_method_name') in receipt['selected_methods'] + receipt['comparators'], 'method absent from fixed comparison decision')
    return {'schema': 'trajkv.round12.fixed500_admission.v1', 'rows': chosen, 'expected_ids': [r['id'] for r in chosen],
        'dataset': chosen[0]['dataset'], 'native_bindings': {r['id']: manifest['native_bindings'][r['id']] for r in values},
        'selected_native_bindings': {r['id']: manifest['native_bindings'][r['id']] for r in chosen},
        'all_input_ids': [r['id'] for r in values],
        'selection_receipt': receipt, 'selection_receipt_sha256': config['selection_receipt_sha256'],
        'input_manifest_sha256': config['input_manifest_sha256'], 'inputs_sha256': config['inputs_sha256'],
        'role': ROLE, 'previously_observed': True, 'independent_holdout_claimed': False,
        'binding_scope': 'old INPUT metadata only; no model outputs/logit reuse', 'old_prediction_reuse_for_candidate': False}


def validate_execution_selection(config, identity):
    name = config.get('selected_method_name')
    receipt = identity['selection_receipt']
    require(name in receipt['selected_methods'] + receipt['comparators'], 'execution not frozen for this comparison')
    return name


def validate_native_record(record, bindings):
    require(record.get('id') in bindings, 'actual row outside fixed500 native metadata')
    expected = bindings[record['id']]
    mismatches = {k: {'expected': expected[k], 'actual': record.get(k)} for k in NATIVE_FIELDS if record.get(k) != expected[k]}
    if record.get('preprocessing') != expected['preprocessing']:
        mismatches['preprocessing'] = {'expected': expected['preprocessing'], 'actual': record.get('preprocessing')}
    return {'matched': not mismatches, 'mismatches': mismatches,
            'scope': 'same previously observed source/question/image/model/prompt/planned positions',
            'bitexact_logits_claimed': False, 'old_logits_or_output_used_by_model': False}


def config_template():
    return {'schema': RUNNER_SCHEMA, 'model_id': 'M1', 'model_revision': MODEL_REVISION,
        'model_path': None, 'model_source_lock': None, 'seed': 0, 'max_new_tokens': 64,
        'question_suffix': SUFFIX, 'preprocessing': {'max_patches': 8},
        'input_role': ROLE, 'cohort_kind': 'fixed500_comparison',
        'input_manifest_path': None, 'input_manifest_sha256': None, 'inputs': None, 'inputs_sha256': None,
        'input_count': 500, 'start': 0, 'count': None,
        'selected_method_name': None, 'selection_receipt_path': None, 'selection_receipt_sha256': None,
        'selected_algorithm_path': None, 'selected_algorithm_sha256': None,
        'evaluation_contract': {'version': METRIC_VERSION, 'primary_metric_id': None},
        'expected_source_hashes': None, 'status': 'UNCONFIGURED_NO_WINNER_FIXED500_RUN_AUTHORED'}


def selection_receipt_template():
    return {'schema': SELECTION_SCHEMA, 'input_manifest_sha256': None,
        'selected_methods': [], 'comparators': ['FP16', 'R7low', 'R7high'], 'method_asset_locks': [],
        'fixed500_is_previously_observed': True, 'independent_holdout_claimed': False,
        'new_candidate_fixed500_outputs_read_before_selection': False,
        'methods_and_assets_frozen_before_candidate_comparison': False,
        'development_selection_report_sha256': None,
        'status': 'UNCONFIGURED_NOT_A_FROZEN_WINNER_SELECTION'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--registry-root', type=Path)
    p.add_argument('--r7-root', type=Path)
    p.add_argument('--fp16-score-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--datasets', nargs='+', default=['infovqa', 'docvqa'])
    p.add_argument('--print-template', action='store_true')
    a = p.parse_args()
    if a.print_template:
        print(json.dumps(config_template(), ensure_ascii=False, indent=2))
        return
    require(a.prepare and all(x is not None for x in (a.registry_root, a.r7_root, a.fp16_score_root, a.output)), 'explicit metadata preparation paths required')
    print(json.dumps(prepare(a.registry_root, a.r7_root, a.fp16_score_root, a.output, a.datasets), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
