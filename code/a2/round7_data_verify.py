"""Independent byte/identity/label-alignment checks for staged R7 development.

Confirmation answers are not opened. Does not run a model or infer task scores.
"""
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source(row):
    return str(row.get('source_document_id', row.get('source_original_image_id')))


def main():
    member = read(ROOT / 'audit/a2/round7_development_membership_v1.json')
    assert sha(ROOT / member['selection']) == member['selection_sha256']
    selections, counts = {}, {}
    image_paths = {}
    for label, spec in member['datasets'].items():
        sample_path, ref_path = ROOT / spec['samples'], ROOT / spec['references']
        assert sha(sample_path) == spec['samples_sha256'], label
        assert sha(ref_path) == spec['references_sha256'], label
        rows = read(sample_path)
        refs = ([json.loads(line) for line in ref_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                if ref_path.suffix == '.jsonl' else read(ref_path))
        assert len(rows) == len(refs) == spec['rows']
        assert [x['id'] for x in rows] == [x['id'] for x in refs]
        assert len({x['id'] for x in rows}) == len(rows)
        assert sum(x['turns'] for x in rows) == spec['questions']
        assert len({source(x) for x in rows}) == spec['source_count']
        for row, ref in zip(rows, refs):
            assert not any(key in row for key in ('answer', 'answers', 'references'))
            assert len(row['questions']) == row['turns']
            assert row['content'][-1]['text'] == row['questions'][0]
            if label.startswith('doc'):
                assert len(ref['references']) == row['turns']
                assert [x['turn'] for x in ref['references']] == list(range(1, row['turns'] + 1))
                if 'calibration_question_index' in row and row['turns'] == 1:
                    assert str(ref['references'][0]['question_id']) == str(row['source_question_ids'][row['calibration_question_index'] - 1])
            else:
                assert ref['image_id'] == row['image_id']
                assert len(ref['answers']) == 10
            for path, digest in zip(row['images'], row['hashes']):
                image_paths[digest] = Path(path)
        selections[label] = rows
        grouped = collections.Counter(source(row) for row in rows)
        counts[label] = dict(source_count=len(grouped), rows=len(rows), questions=spec['questions'],
            independent_questions_per_source=dict(sorted(collections.Counter(grouped.values()).items())))
    for dataset, train in [('doc', 'doc_train16_independent'), ('text', 'text_train32_independent')]:
        left, right = selections[train], selections[dataset + '_select16_independent']
        assert not ({source(x) for x in left} & {source(x) for x in right})
        assert not ({x['image_id'] for x in left} & {x['image_id'] for x in right})
    doc_pool = read(ROOT / 'audit/a2/round7_doc_confirm128_candidate_pool_v1.json')
    doc_assets = read(ROOT / 'audit/a2/round7_doc_confirm128_image_assets_v1.json')
    text_pool = read(ROOT / 'audit/a2/round7_text_confirm128_membership_v1.json')
    doc_chosen = doc_pool['candidates128']
    assert len(doc_chosen) == len({x['source_document_id'] for x in doc_chosen}) == len({x['sha256'] for x in doc_chosen}) == 128
    assert sum(len(x['source_question_ids']) for x in doc_chosen) == 412
    assert all(0 < len(x['source_question_ids']) <= 4 for x in doc_chosen)
    assert sha(ROOT / 'audit/a2/round7_doc_confirm128_candidate_pool_v1.json') == doc_assets['candidate_pool_sha256']
    text_chosen = text_pool['images_manifest']
    assert len(text_chosen) == len({x['source_image_id'] for x in text_chosen}) == len({x['sha256'] for x in text_chosen}) == 128
    text_dev = selections['text_train32_independent'] + selections['text_select16_independent']
    assert not ({str(x['source_original_image_id']) for x in text_dev} & {x['source_image_id'] for x in text_chosen})
    dev_hashes = {x['hashes'][0] for rows in selections.values() for x in rows}
    confirm_doc_hashes = {x['sha256'] for x in doc_chosen}
    confirm_text_hashes = {x['sha256'] for x in text_chosen}
    assert not (dev_hashes & confirm_doc_hashes or dev_hashes & confirm_text_hashes or confirm_doc_hashes & confirm_text_hashes)
    for item in doc_assets['images_manifest'] + text_chosen:
        image_paths[item['sha256']] = Path(item['local'])
    for digest, path in image_paths.items():
        assert sha(path) == digest, str(path)
    report = dict(schema='trajkv.round7.data-independent-check.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), status='PASS', GPU_used=False,
        checked_unique_images=len(image_paths), development=counts,
        source_and_content_overlap_train_select=0, source_and_content_overlap_development_confirmation=0,
        confirmation_doc_sources=128, confirmation_doc_questions=412,
        confirmation_text_sources=128, confirmation_text_questions=128,
        confirmation_reference_files_opened=False, task_scores_computed=False,
        checked_membership_sha256=sha(ROOT / 'audit/a2/round7_development_membership_v1.json'),
        source_sha256=sha(__file__))
    target = ROOT / 'audit/a2/round7_data_independent_check_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(status='PASS', unique_images=len(image_paths), development_rows=sum(x['rows'] for x in counts.values()),
        confirmation_doc_sources=128, confirmation_text_sources=128, future_references_opened=False, GPU_used=False)))


if __name__ == '__main__':
    main()
