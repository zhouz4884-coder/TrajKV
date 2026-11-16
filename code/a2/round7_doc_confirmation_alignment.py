"""CPU-only full Doc128/412-question alignment and independent ANLS audit.

Reads fixed official parquet bytes, frozen inputs and completed primary records.
Never launches inference, changes a sample, or rewrites a result.
"""
import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SUFFIX = '\nAnswer the question using a single word or phrase.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local(value):
    value = str(value).replace('\\', '/')
    return ROOT / value.split('/TrajKV/', 1)[1] if '/TrajKV/' in value else Path(value)


def independent_score(prediction, references):
    """Full DP matrix, independent of the production scorer implementation."""
    pred = prediction.lower().strip()
    gold = [value.lower().strip() for value in references]
    best = 0.
    if pred and pred != 'none':
        for target in gold:
            matrix = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]
            for i in range(len(pred) + 1):
                matrix[i][0] = i
            for j in range(len(target) + 1):
                matrix[0][j] = j
            for i in range(1, len(pred) + 1):
                for j in range(1, len(target) + 1):
                    matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                                       matrix[i - 1][j - 1] + (pred[i - 1] != target[j - 1]))
            best = max(best, 1 - matrix[-1][-1] / max(len(pred), len(target)))
    return (best if best >= .5 else 0.), float(pred != 'none' and pred in gold), best == .5


def forbidden_fields(value, prefix=''):
    names = {'answer', 'answers', 'reference', 'references', 'gold', 'gt',
             'ground_truth', 'target', 'targets', 'forced', 'forced_tokens'}
    found = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = prefix + '/' + key
            if key.lower() in names:
                found.append(path)
            found.extend(forbidden_fields(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(forbidden_fields(nested, prefix + '/' + str(index)))
    elif isinstance(value, str) and any(marker in value.lower() for marker in ('/configs/data/', '_references.json', '_references.jsonl')):
        found.append(prefix + ':reference-path')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'audit/a2/round7_doc_confirmation_alignment_v1.json')
    args = parser.parse_args()
    pool_path = ROOT / 'audit/a2/round7_doc_confirm128_candidate_pool_v1.json'
    samples_path = ROOT / 'configs/a2/round7_v1/doc_confirm128.json'
    refs_path = ROOT / 'configs/data/round7_doc_confirm128_references.jsonl'
    stage_path = ROOT / 'audit/a2/round7_doc_confirm128_staged_v1.json'
    pool, samples, staged = read(pool_path), read(samples_path), read(stage_path)
    references = {row['id']: row for row in map(json.loads, refs_path.read_text(encoding='utf-8').splitlines())}
    initial_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                      for path in (pool_path, samples_path, refs_path, stage_path)}
    assert sha(pool_path) == staged['candidate_pool_sha256']
    assert sha(samples_path) == staged['samples_sha256'] and sha(refs_path) == staged['references_sha256']
    assert len(samples) == len(references) == len(pool['candidates128']) == 128
    assert not forbidden_fields(samples)
    chosen = {item['source_page_id']: item for item in pool['candidates128']}
    selected = {(row['source_page_id'], str(qid)) for row in samples for qid in row['source_question_ids']}
    assert len(selected) == 412
    official, shard_checks, all_types = {}, [], collections.Counter()
    for item in pool['shard_identities']:
        path = ROOT / 'data/docvqa_round7_train_pinned' / Path(item['path']).name
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        initial_hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
        table = pq.read_table(path, columns=['docId', 'ucsf_document_id', 'questionId', 'question',
                                            'question_types', 'answers', 'data_split'])
        rows = table.to_pylist()
        retained = 0
        for row in rows:
            all_types.update(row['question_types'])
            key = (row['docId'], str(row['questionId']))
            if key in selected:
                assert key not in official
                assert row['data_split'] == 'train'
                official[key] = dict(row, shard=path.name)
                retained += 1
        shard_checks.append({'file': str(path.relative_to(ROOT)).replace('\\', '/'),
            'sha256': item['sha256'], 'bytes': item['bytes'], 'rows': len(rows),
            'selected_questions': retained})
    assert set(official) == selected
    official_image_checks = 0
    for item in pool['shard_identities']:
        path = ROOT / 'data/docvqa_round7_train_pinned' / Path(item['path']).name
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=['docId', 'questionId', 'image']):
            for row in batch.to_pylist():
                if (row['docId'], str(row['questionId'])) in selected:
                    assert hashlib.sha256(row['image']['bytes']).hexdigest() == chosen[row['docId']]['sha256']
                    official_image_checks += 1
    assert official_image_checks == 412
    alignment, image_checks, qtypes = [], [], collections.Counter()
    official_targets, row_by_sid = {}, {row['id']: row for row in samples}
    for row in samples:
        page, source = row['source_page_id'], row['source_document_id']
        candidate = chosen[page]
        assert source == candidate['source_document_id']
        assert row['source_question_ids'] == candidate['source_question_ids']
        assert row['turns'] == len(row['questions']) == len(row['source_question_ids'])
        assert row['content'] == [{'type': 'image'}, {'type': 'text', 'text': row['questions'][0]}]
        assert row['hashes'] == [candidate['sha256']]
        assert row['image_id'] == row['source_image_id'] == 'sha256:' + candidate['sha256']
        image = local(row['images'][0])
        assert sha(image) == candidate['sha256']
        image_checks.append({'source_document_id': source, 'source_page_id': page,
            'original_image_sha256': candidate['sha256'], 'staged_image_sha256': sha(image),
            'raw_original_image_bytes': image.stat().st_size})
        ref = references[row['id']]
        assert ref['source_document_id'] == source and len(ref['references']) == row['turns']
        for index, qid in enumerate(row['source_question_ids']):
            original = official[(page, str(qid))]
            target = ref['references'][index]
            assert str(original['ucsf_document_id']) == source
            assert row['questions'][index] == original['question'] + SUFFIX
            assert target['turn'] == index + 1 and str(target['question_id']) == str(qid)
            assert target['answer'] == original['answers'] and target['question_types'] == original['question_types']
            official_targets[(row['id'], index + 1)] = original['answers']
            qtypes.update(original['question_types'])
            alignment.append({'session_id': row['id'], 'source_document_id': source,
                'source_page_id': page, 'question_id': str(qid), 'turn': index + 1,
                'shard': original['shard'], 'question_exact_plus_fixed_suffix': True,
                'official_answers_exact_including_order': True, 'question_types_exact': True,
                'question_sha256': hashlib.sha256(original['question'].encode()).hexdigest(),
                'answers_sha256': hashlib.sha256(json.dumps(original['answers'], ensure_ascii=False).encode()).hexdigest()})
    # Replay the fixed preselection source/content registry, not today's registry
    # (which legitimately includes this now-completed confirmation itself).
    seed = read(ROOT / 'audit/a2/next_docvqa_source_split_v1.json')
    prior_sources, prior_hashes = set(seed['excluded_source_documents']), set()
    def reserve(rows):
        for row in rows:
            if row.get('dataset') == 'docvqa_grouped':
                if row.get('source_document_id'):
                    prior_sources.add(str(row['source_document_id']))
                prior_hashes.update(row.get('hashes', []))
                for key in ('image_id', 'source_image_id'):
                    if str(row.get(key, '')).startswith('sha256:'):
                        prior_hashes.add(row[key][7:])
    checked_registry = 0
    for item in pool['prior_registry_inputs']:
        path = local(item['path'])
        if not path.is_absolute():
            path = ROOT / path
        if item['kind'] == 'registered subset reservation':
            assert sha(path) == item['sha256']
            reserve(read(path)); checked_registry += 1
        elif item['kind'].startswith('manifest selected-ID'):
            assert sha(path) == item['sha256']
            manifest = read(path)
            source_path = local(manifest['config']['arguments']['samples'])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            lookup = {row['id']: row for row in read(source_path)}
            reserve([lookup[sid] for sid in manifest.get('selected_ids', [])]); checked_registry += 1
    assert not prior_sources.intersection(row['source_document_id'] for row in samples)
    assert not prior_hashes.intersection(row['hashes'][0] for row in samples)
    preview_path = ROOT / 'outputs/round7_primary_preview_v1/doc'
    preview = read(preview_path / 'summary.json')
    csv_rows = list(csv.DictReader((preview_path / 'scores.csv').open(encoding='utf-8-sig')))
    scored = {(row['session_id'], row['variant'], int(row['turn'])): row for row in csv_rows}
    assert len(scored) == 412 * 9
    independent, manifests, actual_ids, raw_record_count = collections.defaultdict(list), [], set(), 0
    for split in ('s0', 's1'):
        folder = ROOT / ('results/round7_v1/confirm_' + split + '/doc')
        manifest = read(folder / 'manifest.json')
        assert read(folder / 'complete.json')['status'] == 'COMPLETE'
        assert manifest['samples_sha256'] == sha(samples_path)
        assert manifest['answers_visible_to_runner'] is False
        assert not forbidden_fields(manifest['config'])
        code_checks = {}
        for name in ('session_boundary.py', 'run_representation_variants.py', 'round7_runner.py'):
            relative = 'code/a2/' + name
            assert manifest['sources'][relative] == sha(ROOT / relative)
            code_checks[relative] = manifest['sources'][relative]
        manifests.append({'split': split, 'config_has_answer_or_reference_fields_or_paths': False,
            'selected_sessions': len(manifest['selected_ids']), 'sample_sha256_matches': True,
            'critical_executed_source_hashes_match_current_preserved_code': code_checks})
        actual_ids.update(manifest['selected_ids'])
        records_path = folder / 'records.jsonl'
        initial_hashes[str(records_path.relative_to(ROOT)).replace('\\', '/')] = sha(records_path)
        for record in map(json.loads, records_path.read_text(encoding='utf-8').splitlines()):
            raw_record_count += 1
            assert record['protocol'] == 'image_boundary_free_running' and record['policy'] == 'base'
            assert len(record['turns']) == row_by_sid[record['session_id']]['turns']
            for turn in record['turns']:
                sid, variant, number = record['session_id'], record['variant'], turn['turn']
                targets = official_targets[(sid, number)]
                score, exact, boundary = independent_score(turn['answer'], targets)
                prior = scored[(sid, variant, number)]
                assert prior['prediction'] == turn['answer'] and json.loads(prior['reference']) == targets
                assert abs(score - float(prior['score'])) <= 1e-12
                assert exact == float(prior['exact'])
                independent[variant].append({'session_id': sid, 'turn': number, 'score': score,
                    'exact': exact, 'threshold_boundary': boundary})
    assert actual_ids == set(row_by_sid)
    rescored = []
    for row in preview['table']:
        values = independent[row['variant']]
        assert len(values) == 412
        mean, exact = statistics.mean(x['score'] for x in values), statistics.mean(x['exact'] for x in values)
        assert abs(mean - row['score']) <= 1e-12 and abs(exact - row['exact']) <= 1e-12
        rescored.append({'variant': row['variant'], 'questions': 412, 'independent_anls': mean,
            'independent_exact': exact, 'matches_preview': True,
            'half_threshold_boundary_questions': sum(x['threshold_boundary'] for x in values)})
    full = independent['full8']
    result = {'schema': 'trajkv.round7.doc-confirmation-alignment.v1', 'status': 'PASS',
        'scope': 'Complete frozen Doc128/412 official alignment and all 3708 completed primary answers; CPU only, no sample/result mutation',
        'upstream_repo': pool['repo'], 'upstream_revision': pool['revision'], 'official_split': 'train',
        'project_fresh_not_backbone_unseen': True, 'shards': shard_checks,
        'source_count': 128, 'question_count': 412, 'unique_source_documents': len({r['source_document_id'] for r in samples}),
        'unique_original_image_hashes': len({r['hashes'][0] for r in samples}),
        'official_question_image_byte_comparisons': official_image_checks,
        'staged_images_verified': len(image_checks), 'image_alignment': image_checks,
        'question_answer_alignment': alignment, 'sample_answer_or_ref_fields_or_paths': [],
        'prior_frozen_registry_items_replayed': checked_registry, 'prior_source_overlap': 0, 'prior_content_overlap': 0,
        'runner_input_audit': manifests, 'raw_sessions': raw_record_count,
        'raw_questions_all_methods_independently_rescored': len(scored),
        'scorer_definition': 'lowercase and outer strip only; character Levenshtein; max reference similarity; >=0.5 inclusive; empty/none prediction zero',
        'independent_implementation': 'Full dynamic-programming matrix; production scorer not imported',
        'rescored_methods': rescored, 'question_type_multilabel_counts': dict(qtypes),
        'full_native_score_counts': {'score_one': sum(r['score'] == 1 for r in full),
            'partial': sum(0 < r['score'] < 1 for r in full), 'zero': sum(r['score'] == 0 for r in full)},
        'full_native_by_turn': {str(t): {'questions': len(v := [r for r in full if r['turn'] == t]),
            'anls': statistics.mean(r['score'] for r in v), 'exact': statistics.mean(r['exact'] for r in v)}
            for t in (1, 2, 3, 4)},
        'interpretation': 'No mapping, image, answer-in-prompt, or score-recomputation error found. High ANLS includes partial credit; train-shard source and form-heavy composition limit external benchmark generalization. Backbone training overlap is not established or excluded by this audit.',
        'selection_unchanged': True, 'selection_used_outcomes': False,
        'preserved_input_sha256': initial_hashes, 'audit_source_sha256': sha(__file__)}
    for relative, digest in initial_hashes.items():
        assert sha(ROOT / relative) == digest
    if args.output.exists():
        assert read(args.output) == result, 'Existing audit differs; preserve it and select a new output'
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'questions_aligned': 412, 'images_aligned': 128,
        'questions_rescored_all_methods': len(scored), 'full_counts': result['full_native_score_counts'],
        'output': str(args.output)}))


if __name__ == '__main__':
    main()
