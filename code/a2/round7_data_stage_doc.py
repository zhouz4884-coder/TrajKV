"""Stage fixed Doc128 raw images; read selected QA only with method freeze."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from round7_data_doc_extra import ROOT, POOL, DATA, sha

ASSETS = ROOT / 'audit/a2/round7_doc_confirm128_image_assets_v1.json'
IMAGE_DIR = ROOT / 'data/docvqa_round7_confirm128/images'


def new(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            json.dump(value, stream, indent=2, ensure_ascii=False)


def remote_root():
    old = json.loads((ROOT / 'configs/a2/round6_v1/second_llava15_smoke8.json').read_text(encoding='utf-8'))
    return old['arguments']['samples'].split('/configs/', 1)[0]


def stage_images(pool):
    if ASSETS.exists():
        result = json.loads(ASSETS.read_text(encoding='utf-8'))
        if sha(POOL) != result['candidate_pool_sha256']:
            raise ValueError('Frozen candidate pool changed')
        for row in result['images_manifest']:
            if sha(Path(row['local'])) != row['sha256']:
                raise ValueError('Staged candidate image changed')
        return result
    targets = {item['source_page_id']: item for item in pool['candidates128']}
    staged = {}
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for item in pool['shard_identities']:
        path = DATA / Path(item['path']).name
        if sha(path) != item['sha256']:
            raise ValueError('Pinned source parquet changed')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4, columns=['docId', 'image']):
            for row in batch.to_pylist():
                chosen = targets.get(row['docId'])
                if chosen is None or row['docId'] in staged:
                    continue
                raw = row['image']['bytes']
                if hashlib.sha256(raw).hexdigest() != chosen['sha256']:
                    raise ValueError('Frozen selected image content changed')
                suffix = {'PNG': '.png', 'JPEG': '.jpg', 'TIFF': '.tif'}[chosen['format']]
                target = IMAGE_DIR / (chosen['sha256'] + suffix)
                if target.exists():
                    if sha(target) != chosen['sha256']:
                        raise ValueError('Existing staged bytes conflict')
                else:
                    with target.open('xb') as stream:
                        stream.write(raw)
                staged[row['docId']] = dict(local=str(target), relative=target.relative_to(ROOT).as_posix(),
                    remote=remote_root() + '/' + target.relative_to(ROOT).as_posix(), sha256=chosen['sha256'],
                    bytes=len(raw), source_document_id=chosen['source_document_id'], source_page_id=row['docId'])
    if len(staged) != 128:
        raise ValueError('Incomplete source image staging')
    result = dict(schema='trajkv.round7.doc-confirm128-image-assets.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), candidate_pool_sha256=sha(POOL),
        images=128, image_bytes=sum(r['bytes'] for r in staged.values()),
        images_manifest=[staged[x['source_page_id']] for x in pool['candidates128']],
        question_text_read=False, answers_read=False, uploaded=False, GPU_used=False)
    new(ASSETS, result)
    return result


def stage_questions(pool, assets, method):
    if method is None or not json.loads(method.read_text(encoding='utf-8-sig')):
        raise ValueError('Nonempty caller method-freeze manifest required before questions/labels')
    method_sha = sha(method)
    selected_ids = {str(qid) for row in pool['candidates128'] for qid in row['source_question_ids']}
    questions = {}
    for item in pool['shard_identities']:
        path = DATA / Path(item['path']).name
        table = pq.read_table(path, columns=['docId', 'ucsf_document_id', 'questionId', 'question', 'question_types', 'answers'],
            filters=[('questionId', 'in', sorted(selected_ids))])
        for row in table.to_pylist():
            key = (row['docId'], int(row['questionId']))
            if key in questions:
                raise ValueError('Duplicate selected question identity')
            questions[key] = row
    by_page = {item['source_page_id']: item for item in assets['images_manifest']}
    rows, refs = [], []
    for chosen in pool['candidates128']:
        page = chosen['source_page_id']
        data = [questions[(page, int(qid))] for qid in chosen['source_question_ids']]
        if any(str(row['ucsf_document_id']) != chosen['source_document_id'] for row in data):
            raise ValueError('Frozen source document identity changed')
        image = by_page[page]
        sid = 'docvqa_round7_confirm_' + str(page)
        text = [row['question'] + '\nAnswer the question using a single word or phrase.' for row in data]
        rows.append(dict(schema='trajkv.session.v1', dataset='docvqa_grouped', id=sid, session_id=sid,
            source_document_id=chosen['source_document_id'], source_page_id=page,
            source_question_ids=chosen['source_question_ids'], image_id='sha256:' + chosen['sha256'],
            source_image_id='sha256:' + chosen['sha256'], images=[image['local']], remote_images=[image['remote']],
            hashes=[chosen['sha256']], questions=text, turns=len(text),
            content=[{'type': 'image'}, {'type': 'text', 'text': text[0]}],
            split='round7_fresh_source_confirm128', protocol='derived_same_document_independent_qa_grouped',
            history_dependence='not_assumed_requires_qualification', historical_result_reused=False,
            answer_format_protocol='internvl_official_short_phrase_prompt_v1', original_dataset_split='train'))
        refs.append(dict(id=sid, source_document_id=chosen['source_document_id'],
            references=[dict(turn=i + 1, question_id=row['questionId'], answer=row['answers'], question_types=row['question_types'])
                        for i, row in enumerate(data)]))
    samples = ROOT / 'configs/a2/round7_v1/doc_confirm128.json'
    references = ROOT / 'configs/data/round7_doc_confirm128_references.jsonl'
    new(samples, rows)
    new(references, refs, jsonl=True)
    if sha(method) != method_sha:
        raise RuntimeError('Method freeze changed during materialization')
    new(ROOT / 'audit/a2/round7_doc_confirm128_staged_v1.json', dict(
        schema='trajkv.round7.doc-confirm128-staged.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), images=128, questions=sum(r['turns'] for r in rows),
        candidate_pool_sha256=sha(POOL), image_assets_sha256=sha(ASSETS), method_freeze=str(method), method_freeze_sha256=method_sha,
        samples=samples.relative_to(ROOT).as_posix(), samples_sha256=sha(samples),
        references=references.relative_to(ROOT).as_posix(), references_sha256=sha(references),
        answers_uploaded=False, uploaded=False, source_content_overlap=0,
        original_dataset_split='train', scope='project-fresh source holdout; not claimed unseen in backbone pretraining'))
    print(json.dumps(dict(status='STAGED_LOCAL_ONLY', images=128, questions=sum(r['turns'] for r in rows), answers_uploaded=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage-questions', action='store_true')
    p.add_argument('--method-freeze', type=Path)
    a = p.parse_args()
    pool = json.loads(POOL.read_text(encoding='utf-8'))
    assets = stage_images(pool)
    if a.stage_questions:
        stage_questions(pool, assets, a.method_freeze)
    else:
        print(json.dumps(dict(status='IMAGES_STAGED_LOCAL_ONLY', images=assets['images'],
            bytes=assets['image_bytes'], questions_read=False, answers_read=False)))
