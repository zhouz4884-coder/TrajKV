"""Round12 development materialization and no-answer model input admission.

The preparer reads already-frozen R11 inputs and separate references. The
inference loader opens only input JSONL and INPUT_MANIFEST, never references.
No image is copied and no model/torch/GPU code is imported here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = 'trajkv.round12.inputs.v1'
ROLE = 'round12_development'
PANEL_SHA = '9aa48795b508b94e671b7b1c9cf22ab97b85a4c32099285c782d6636fde18870'
FIELDS = {'id', 'dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id',
          'question_id', 'image_path', 'original_image_sha256', 'width', 'height',
          'question', 'group', 'image_format', 'image_bytes', 'original_split_row',
          'question_sha256'}
ALLOWED = {
    'textvqa': ('9c0699cd19768ac5ab97568f6b3cbac4c0062884', {'validation'}),
    'docvqa': ('539088ef8a8ada01ac8e2e6d4e372586748a265e', {'train', 'validation'}),
    'infovqa': ('539088ef8a8ada01ac8e2e6d4e372586748a265e', {'train'}),
}
NATIVE_FIELDS = ('id', 'original_image_sha256', 'full_prompt_hash', 'input_ids_sha256',
                 'native_position_ids_sha256', 'compression_boundary',
                 'actual_visual_token_count', 'model_commit', 'dtype')


def require(condition, message):
    if not condition:
        raise ValueError('ROUND12_INPUTS: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def validate_rows(values, expected_count=None):
    require(bool(values) and (expected_count is None or len(values) == expected_count), 'wrong input count')
    for row in values:
        require(FIELDS - {'group'} <= set(row) <= FIELDS, 'missing fields or labels/unknown fields')
        ds = row['dataset']
        require(ds in ALLOWED and row['dataset_revision'] == ALLOWED[ds][0]
                and row['split'] in ALLOWED[ds][1], 'source revision/split mismatch')
        require(row['id'] == f"{ds}:{row['split']}:{row['question_id']}", 'source ID mismatch')
        require(isinstance(row['question'], str) and row['question'].strip()
                and hashlib.sha256(row['question'].encode()).hexdigest() == row['question_sha256'],
                'question SHA mismatch')
        cluster = ('docvqa:ucsf:' + row['source_id'].rsplit('/', 1)[0]
                   if ds == 'docvqa' else ds + ':image:' + row['source_id'])
        require(row['source_cluster_id'] == cluster, 'cluster identity mismatch')
        require(re.fullmatch('[0-9a-f]{64}', row['original_image_sha256']) is not None, 'bad image SHA')
        require(all(type(row[k]) is int and row[k] > 0 for k in ('width', 'height', 'image_bytes')), 'bad image size')
        require(type(row['original_split_row']) is int and row['original_split_row'] >= 0, 'bad source row')
        p = PurePosixPath(row['image_path'])
        require(p.is_absolute() and '..' not in p.parts and '\\' not in row['image_path']
                and '/baseline_matrix_v3/' in row['image_path'] and p.stem == row['original_image_sha256'],
                'image is not a content-addressed project data-disk path')
    for field in ('id', 'source_cluster_id', 'original_image_sha256'):
        require(len({r[field] for r in values}) == len(values), 'duplicate source/content: ' + field)
    return values


def load_round12_inputs(path, manifest_path, expected_sha256=None, expected_role=ROLE):
    """Inference-only loader. It never opens SCORING_MANIFEST or *_refs files."""
    path = Path(path)
    require(path.name.endswith('_inputs.jsonl'), 'only no-answer input JSONL is admitted')
    require(expected_role == ROLE, 'role must explicitly state reused development data')
    manifest = read(manifest_path)
    require(manifest.get('schema') == SCHEMA and manifest.get('role') == ROLE
            and manifest.get('answers_in_encoder_inputs') is False
            and manifest.get('proposed_panel_sha256') == PANEL_SHA,
            'unknown or unbound no-answer manifest')
    entry = manifest['input_files'].get(path.name)
    require(isinstance(entry, dict), 'input file not bound by manifest')
    digest = sha(path)
    require(entry['sha256'] == digest and entry['bytes'] == path.stat().st_size
            and entry['answers_included'] is False and entry['role'] == ROLE, 'input binding mismatch')
    require(expected_sha256 is None or digest == expected_sha256, 'explicit input SHA differs')
    values = validate_rows(rows(path), entry['count'])
    require([r['id'] for r in values] == entry['ids'], 'input order differs')
    require(dict(Counter(r['dataset'] for r in values)) == entry['domains'], 'domain counts differ')
    for row in values:
        native = manifest['native_bindings'][row['id']]
        require(native['original_image_sha256'] == row['original_image_sha256']
                and native['raw_question_sha256'] == row['question_sha256'], 'native input binding differs')
    return values


def prepare(panels_path, source_plans, output):
    require(sha(panels_path) == PANEL_SHA, 'frozen proposed panels changed')
    panels = read(panels_path)
    require(panels['schema'] == 'trajkv.round12.proposed_development_panels.v1'
            and panels['usage'].startswith('Previously observed R11'), 'development panel identity mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    input_index, ref_index, provenance = {}, {}, []
    for plan_path in source_plans:
        plan = read(plan_path)
        for dataset in ALLOWED:
            info = plan['domains'][dataset]
            require(sha(info['inputs']) == info['inputs_sha256'] and sha(info['refs']) == info['refs_sha256'],
                    'frozen R11 input/reference file differs')
            iv, rv = rows(info['inputs']), rows(info['refs'])
            references = {r['id']: r for r in rv}
            require(len(references) == len(rv), 'duplicate original reference IDs')
            for row in iv:
                require(row['id'] not in input_index, 'R11 source batches overlap')
                require(row['id'] in references, 'original reference missing')
                validate_rows([row], 1)
                ref = references[row['id']]
                require(ref['source_cluster_id'] == row['source_cluster_id'] and isinstance(ref['answers'], list)
                        and ref['answers'], 'reference/source identity differs')
                input_index[row['id']] = row
                # Offline-only data, written into separate files below.
                ref_index[row['id']] = {k: ref[k] for k in ('id', 'source_cluster_id', 'answers')}
            provenance.append({'plan': str(plan_path), 'plan_sha256': sha(plan_path), 'dataset': dataset,
                               'input_sha256': info['inputs_sha256'], 'reference_sha256': info['refs_sha256']})
    manifest = dict(schema=SCHEMA, role=ROLE, proposed_panel_sha256=PANEL_SHA,
                    answers_in_encoder_inputs=False, previously_observed_sources=True,
                    phase_is_confirmatory=False, builder_sha256=sha(__file__), input_files={}, native_bindings={})
    scoring = dict(schema='trajkv.round12.offline_scoring_inputs.v1', reference_files={}, provenance=provenance)
    for panel, members in panels['panels'].items():
        for dataset in ALLOWED:
            selected = [m for m in members if m['dataset'] == dataset]
            if not selected:
                continue
            values, references = [], []
            for member in selected:
                row = input_index[member['id']]
                for key in ('source_id', 'source_cluster_id', 'original_image_sha256', 'question_sha256'):
                    require(row[key] == member[key], 'proposed/source input differs: ' + key)
                require(member['low_profile'] == 'r7_joint_k1_gt_cond'
                        and len(member['low_selected_stages']) == 43, 'wrong original encoding lineage')
                binding = {k: member[k] for k in NATIVE_FIELDS}
                binding.update(raw_question_sha256=member['question_sha256'],
                               low_selected_stages=member['low_selected_stages'],
                               preprocessing=member['preprocessing'])
                require(member['preprocessing'] == {'max_patches': 8}, 'native cap mismatch')
                require(member['model_commit'] == '259a3b64a14623c0ec91a045cb43f7c5af5fa6af', 'model revision mismatch')
                if row['id'] in manifest['native_bindings']:
                    require(manifest['native_bindings'][row['id']] == binding, 'duplicate inconsistent binding')
                manifest['native_bindings'][row['id']] = binding
                values.append(row)
                references.append(ref_index[row['id']])
            name = f'{panel}_{dataset}_inputs.jsonl'
            path = output / name
            write_rows(path, values)
            manifest['input_files'][name] = dict(panel=panel, count=len(values), domains={dataset: len(values)},
                ids=[r['id'] for r in values], sha256=sha(path), bytes=path.stat().st_size,
                role=ROLE, answers_included=False)
            ref_name = f'{panel}_{dataset}_refs.jsonl'
            ref_path = output / ref_name
            write_rows(ref_path, references)
            scoring['reference_files'][name] = dict(file=ref_name, count=len(references),
                sha256=sha(ref_path), bytes=ref_path.stat().st_size)
    manifest_path = output / 'ROUND12_INPUT_MANIFEST_v1.json'
    write_new(manifest_path, manifest)
    write_new(output / 'ROUND12_SCORING_MANIFEST_v1.json', scoring)
    for name in manifest['input_files']:
        load_round12_inputs(output / name, manifest_path, manifest['input_files'][name]['sha256'])
    receipt = dict(schema='trajkv.round12.materialized_inputs_receipt.v1',
        input_manifest_sha256=sha(manifest_path), input_files=len(manifest['input_files']),
        unique_development_sources=len(manifest['native_bindings']), model_inference_events=0,
        images_copied=0, answers_in_encoder_inputs=False, score_files_separate=True,
        output=str(output), source_plan_provenance=provenance)
    write_new(output / 'COMPLETE.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panels', type=Path, required=True)
    p.add_argument('--source-plans', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.panels, a.source_plans, a.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
