"""Round13 reused-development admission and separate no-answer S0 inputs.

Inference opens only input JSONL and its INPUT_MANIFEST. References remain in
the data-disk bootstrap directory and are never opened by either loader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import round12_inputs_v1 as source

SCHEMA = 'trajkv.round13.inputs.v1'
ROLE = 'round13_development'
CALIBRATION_ROLE = 'round13_calibration_diagnostic'
SOURCE_MANIFEST_SHA = 'c2e235b396179dbc6af4303d2a4053fb5ba3868181660de42e90f5a52e964f29'
NATIVE_FIELDS = source.NATIVE_FIELDS
FIELDS = source.FIELDS
read = source.read
rows = source.rows
sha = source.sha
validate_rows = source.validate_rows


def require(value, message):
    if not value:
        raise ValueError('ROUND13_INPUTS: ' + message)


def _manifest(path):
    manifest = read(path)
    require(manifest.get('schema') == SCHEMA and manifest.get('role') == ROLE and
            manifest.get('answers_in_encoder_inputs') is False and
            manifest.get('previously_observed_sources') is True and
            manifest.get('phase_is_confirmatory') is False and
            manifest.get('source_round12_manifest_sha256') == SOURCE_MANIFEST_SHA,
            'unknown reused-development input manifest')
    return manifest


def _load(path, manifest, group, expected_sha256):
    path = Path(path)
    require(path.name.endswith('_inputs.jsonl'), 'only no-answer input JSONL admitted')
    entry = manifest[group].get(path.name)
    require(isinstance(entry, dict) and entry['answers_included'] is False, 'unbound no-answer input')
    digest = sha(path)
    require(digest == entry['sha256'] and path.stat().st_size == entry['bytes'], 'input bytes/SHA differ')
    require(expected_sha256 is None or expected_sha256 == digest, 'explicit SHA differs')
    values = validate_rows(rows(path), entry['count'])
    require([row['id'] for row in values] == entry['ids'] and
            dict(Counter(row['dataset'] for row in values)) == entry['domains'], 'input order/domain counts differ')
    return values, entry


def load_round13_inputs(path, manifest_path, expected_sha256=None, expected_role=ROLE):
    """Main free-answer loader; no refs or scoring manifests are opened."""
    require(expected_role == ROLE, 'reused sources must explicitly keep development role')
    manifest = _manifest(manifest_path)
    values, entry = _load(path, manifest, 'input_files', expected_sha256)
    require(entry['role'] == ROLE and entry['native_binding_expected'] is True, 'wrong main-panel role')
    for row in values:
        native = manifest['native_bindings'][row['id']]
        require(set(NATIVE_FIELDS) <= set(native) and native['original_image_sha256'] == row['original_image_sha256']
                and native['raw_question_sha256'] == row['question_sha256'], 'native source binding differs')
        require(native['model_commit'] == '259a3b64a14623c0ec91a045cb43f7c5af5fa6af' and
                native['dtype'] == 'float16' and native['preprocessing'] == {'max_patches': 8},
                'native model/preprocessing changed')
    return values


def load_stage0_inputs(path, manifest_path, expected_sha256=None):
    """No-answer calibration sources, whose native bindings are captured now."""
    manifest = _manifest(manifest_path)
    values, entry = _load(path, manifest, 'calibration_input_files', expected_sha256)
    require(entry['role'] == CALIBRATION_ROLE and entry['native_binding_expected'] is False,
            'S0 native capture must be declared, never fabricated')
    return values


def _copy_new(source_path, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open('xb') as stream:
        stream.write(Path(source_path).read_bytes())


def _intersection(a, b):
    return {field: sorted({row[field] for row in a} & {row[field] for row in b})
            for field in ('id', 'source_cluster_id', 'original_image_sha256')}


def prepare(source_inputs, source_references, calibration_fit, calibration_select, output_inputs, output_refs):
    """Pure materialization: original JSONL/native identities remain unchanged."""
    source_inputs, source_references = Path(source_inputs), Path(source_references)
    output_inputs, output_refs = Path(output_inputs), Path(output_refs)
    source_manifest_path = source_inputs / 'ROUND12_INPUT_MANIFEST_v1.json'
    require(sha(source_manifest_path) == SOURCE_MANIFEST_SHA, 'frozen R12 source manifest changed')
    original = read(source_manifest_path)
    scoring_path = source_references / 'ROUND12_SCORING_MANIFEST_v1.json'
    original_scoring = read(scoring_path)
    manifest = {'schema': SCHEMA, 'role': ROLE, 'answers_in_encoder_inputs': False,
        'previously_observed_sources': True, 'phase_is_confirmatory': False,
        'source_round12_manifest_sha256': SOURCE_MANIFEST_SHA,
        'source_panel_sha256': original['proposed_panel_sha256'],
        'builder_sha256': sha(__file__), 'input_files': {}, 'native_bindings': {},
        'calibration_input_files': {}, 'scope': 'R11/R12 observed development sources; not independent confirmation'}
    scoring = {'schema': 'trajkv.round13.offline_scoring_inputs.v1',
        'source_round12_scoring_manifest_sha256': sha(scoring_path), 'reference_files': {}}
    all_main, files = [], []
    for panel in ('middle128', 'middle64'):
        for dataset in ('infovqa', 'docvqa', 'textvqa'):
            name = f'{panel}_{dataset}_inputs.jsonl'
            values = source.load_round12_inputs(source_inputs / name, source_manifest_path)
            _copy_new(source_inputs / name, output_inputs / name)
            require(sha(output_inputs / name) == original['input_files'][name]['sha256'], 'original JSONL bytes changed')
            entry = deepcopy(original['input_files'][name])
            entry.update(role=ROLE, native_binding_expected=True,
                         copied_source_sha256=original['input_files'][name]['sha256'])
            manifest['input_files'][name] = entry
            for row in values:
                binding = deepcopy(original['native_bindings'][row['id']])
                if row['id'] in manifest['native_bindings']:
                    require(manifest['native_bindings'][row['id']] == binding, 'nested native binding differs')
                manifest['native_bindings'][row['id']] = binding
            if panel == 'middle128':
                all_main.extend(values)
            ref = original_scoring['reference_files'][name]
            ref_source = source_references / ref['file']
            require(sha(ref_source) == ref['sha256'] and ref_source.stat().st_size == ref['bytes'], 'source refs differ')
            references = rows(ref_source)
            require([r['id'] for r in references] == [r['id'] for r in values], 'source refs order differs')
            for row, reference in zip(values, references):
                require(reference['source_cluster_id'] == row['source_cluster_id'] and reference['answers'], 'ref identity differs')
            _copy_new(ref_source, output_refs / ref['file'])
            scoring['reference_files'][name] = deepcopy(ref)
            files.append({'name': name, 'sha256': entry['sha256'], 'count': len(values), 'role': ROLE})
    validate_rows(all_main, 128)
    for dataset in ('infovqa', 'docvqa', 'textvqa'):
        large = rows(output_inputs / f'middle128_{dataset}_inputs.jsonl')
        small = rows(output_inputs / f'middle64_{dataset}_inputs.jsonl')
        require(small == large[:len(small)], 'middle64 is not the exact original prefix subset')
    cal_provenance, cal = [], []
    for path in (Path(calibration_fit), Path(calibration_select)):
        original_rows = rows(path)
        # Each file has four domains; S0 uses only Info and Doc, in source order.
        cal.extend(row for row in original_rows if row['dataset'] in ('infovqa', 'docvqa'))
        cal_provenance.append({'name': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                               'domains': dict(Counter(row['dataset'] for row in original_rows))})
    validate_rows(cal, 32)
    require(dict(Counter(row['dataset'] for row in cal)) == {'infovqa': 16, 'docvqa': 16}, 'S0 must have Info16 Doc16')
    overlap = _intersection(all_main, cal)
    require(all(not v for v in overlap.values()), 'S0 and reused development sources overlap')
    s0name = 'stage0_cal32_inputs.jsonl'
    source.write_rows(output_inputs / s0name, cal)
    manifest['calibration_input_files'][s0name] = {'count': 32, 'domains': {'infovqa': 16, 'docvqa': 16},
        'ids': [row['id'] for row in cal], 'sha256': sha(output_inputs / s0name),
        'bytes': (output_inputs / s0name).stat().st_size, 'role': CALIBRATION_ROLE,
        'answers_included': False, 'native_binding_expected': False,
        'source_files': cal_provenance, 'scope': 'Previously used calibration sources, diagnostics only; no free task answers'}
    manifest_path = output_inputs / 'ROUND13_INPUT_MANIFEST_v1.json'
    source.write_new(manifest_path, manifest)
    scoring_manifest_path = output_refs / 'ROUND13_SCORING_MANIFEST_v1.json'
    source.write_new(scoring_manifest_path, scoring)
    for name in manifest['input_files']:
        load_round13_inputs(output_inputs / name, manifest_path)
    load_stage0_inputs(output_inputs / s0name, manifest_path)
    receipt = {'schema': 'trajkv.round13.materialized_inputs_receipt.v1', 'status': 'PASS_REUSED_DEVELOPMENT_INPUTS',
        'input_manifest_sha256': sha(manifest_path), 'scoring_manifest_sha256': sha(scoring_manifest_path),
        'input_files': files, 'unique_main_sources': len(all_main), 'S0_sources': len(cal),
        'S0_development_overlap': overlap, 'native_bindings_copied_without_changes': True,
        'model_inference_events': 0, 'images_copied': 0, 'answers_in_encoder_inputs': False,
        'references_only_in_data_disk_bootstrap': True, 'phase_is_confirmatory': False}
    source.write_new(output_refs / 'ROUND13_INPUTS_COMPLETE_v1.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for key in ('source-inputs', 'source-references', 'calibration-fit', 'calibration-select', 'output-inputs', 'output-refs'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
