"""Stage a preinspected fresh cohort locally after the caller's method freeze.

Never selects by answer or outcome. Reads only the fixed IDs in the pool audit,
then writes answer-free runners and separate local references. No SSH/upload.
"""
import argparse
import copy
import hashlib
import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'audit/a2/round6_unseen_pool_inspection_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in value)
               if jsonl else json.dumps(value, indent=2, ensure_ascii=False))
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def verify_pool(pool):
    if pool['staged'] or pool['model_scores_read'] or pool['annotation_files_read']:
        raise ValueError('Expected identity-only preinspection audit')
    for item in pool['registry_inputs']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != item['sha256']:
            raise ValueError('Previous exposure registry changed: ' + str(path))


def prepare_doc(pool, image_dir, remote_root):
    spec = pool['doc']
    candidates = spec['candidates']
    if len(candidates) != 54 or len({x['source_document_id'] for x in candidates}) != 54:
        raise ValueError('Expected the audited 54-source Doc pool')
    by_page = {row['source_page_id']: row for row in candidates}
    questions, raw_images = {}, {}
    shards = sorted((ROOT / 'data/docvqa_validation_pinned').glob('*.parquet'))
    for path in shards:
        if sha(path) != spec['shards_sha256'][path.name]:
            raise ValueError('Pinned Doc bytes changed')
        for batch in pq.ParquetFile(path).iter_batches(columns=[
            'docId', 'ucsf_document_id', 'questionId', 'question', 'question_types', 'answers']):
            for row in batch.to_pylist():
                chosen = by_page.get(row['docId'])
                if chosen and int(row['questionId']) in chosen['source_question_ids']:
                    if str(row['ucsf_document_id']) != chosen['source_document_id']:
                        raise ValueError('Doc page/source identity changed')
                    key = (row['docId'], int(row['questionId']))
                    if key in questions:
                        raise ValueError('Duplicate Doc page/question identity')
                    questions[key] = row
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=['docId', 'image']):
            for row in batch.to_pylist():
                page = row['docId']
                if page in by_page and page not in raw_images:
                    raw = row['image']['bytes']
                    if hashlib.sha256(raw).hexdigest() != by_page[page]['sha256']:
                        raise ValueError('Preinspected Doc image changed')
                    raw_images[page] = raw
    output, refs, images = [], [], []
    for candidate in candidates:
        page = candidate['source_page_id']
        rows = [questions[(page, int(qid))] for qid in candidate['source_question_ids']]
        raw = raw_images[page]
        with Image.open(io.BytesIO(raw)) as im:
            fmt = im.format
            im.verify()
        suffix = {'PNG': '.png', 'JPEG': '.jpg', 'TIFF': '.tif'}[fmt]
        target = image_dir / (candidate['sha256'] + suffix)
        target.write_bytes(raw)
        if sha(target) != candidate['sha256']:
            raise ValueError('Staged Doc image byte mismatch')
        remote = remote_root + '/' + target.relative_to(ROOT).as_posix()
        sid = 'docvqa_round6_' + str(page)
        text = [row['question'] + '\nAnswer the question using a single word or phrase.' for row in rows]
        sample = dict(schema='trajkv.session.v1', dataset='docvqa_grouped', id=sid, session_id=sid,
            image_id='sha256:' + candidate['sha256'], source_image_id='sha256:' + candidate['sha256'],
            source_document_id=candidate['source_document_id'], source_page_id=page,
            source_question_ids=candidate['source_question_ids'], images=[str(target)], remote_images=[remote],
            hashes=[candidate['sha256']], questions=text, turns=len(rows),
            content=[{'type': 'image'}, {'type': 'text', 'text': text[0]}],
            protocol='derived_same_document_independent_qa_grouped',
            history_dependence='not_assumed_requires_qualification',
            split='round6_fresh_source_confirm54', historical_result_reused=False,
            answer_format_protocol='internvl_official_short_phrase_prompt_v1')
        output.append(sample)
        refs.append(dict(id=sid, source_document_id=candidate['source_document_id'],
            references=[dict(turn=i + 1, answer=row['answers'], question_id=row['questionId'],
                             question_types=row['question_types']) for i, row in enumerate(rows)]))
        images.append(dict(local=str(target), relative=target.relative_to(ROOT).as_posix(), remote=remote,
            source_document_id=candidate['source_document_id'], source_page_id=page,
            sha256=candidate['sha256'], bytes=target.stat().st_size))
    if sum(row['turns'] for row in output) != spec['chosen_page_questions']:
        raise ValueError('Audited Doc question count changed')
    return output, refs, images


def prepare_text(pool, count, image_dir, remote_root):
    spec = pool['text']
    source_path = ROOT / spec['source']
    official = read(ROOT / 'audit/a2/textvqa_standard5000_membership.json')
    refs_path = ROOT / 'configs/data/textvqa_standard5000_references.json'
    if sha(source_path) != spec['source_sha256'] or sha(refs_path) != official['references_sha256']:
        raise ValueError('Pinned complete Text source/reference bytes changed')
    source = {row['id']: row for row in read(source_path)}
    references = {row['id']: row for row in read(refs_path)}
    output, refs, images = [], [], []
    for chosen in spec['candidates256'][:count]:
        row = source[chosen['source_session_id']]
        if (str(row['source_original_image_id']) != chosen['source_image_id']
                or row['source_question_id'] != chosen['source_question_id']
                or row['hashes'] != [chosen['sha256']]):
            raise ValueError('Audited Text identity differs from pinned runner')
        src = Path(row['images'][0])
        if sha(src) != chosen['sha256']:
            raise ValueError('Audited Text raw bytes changed')
        target = image_dir / (chosen['sha256'] + src.suffix)
        try:
            os.link(src, target)
            mode = 'hardlink original raw bytes'
        except OSError:
            shutil.copyfile(src, target)
            mode = 'raw byte copy'
        if sha(target) != chosen['sha256']:
            raise ValueError('Staged Text raw bytes changed')
        remote = remote_root + '/' + target.relative_to(ROOT).as_posix()
        sid = 'textvqa_round6_' + str(chosen['source_question_id'])
        sample = copy.deepcopy(row)
        sample.update(id=sid, session_id=sid, images=[str(target)], remote_images=[remote],
            split='round6_source_disjoint_text_transfer' + str(count),
            historical_use='fresh against preinspection registered source/content union',
            selection_hash=chosen['selection_hash'])
        if any(key in sample for key in ('answer', 'answers', 'references')):
            raise ValueError('Reference annotation in Text runner')
        answer = references[chosen['source_session_id']]
        if answer['image_id'] != sample['image_id'] or len(answer['answers']) != 10:
            raise ValueError('Pinned Text annotations do not match selected identity')
        output.append(sample)
        refs.append(dict(id=sid, image_id=sample['image_id'],
                         source_question_id=chosen['source_question_id'], answers=answer['answers']))
        images.append(dict(local=str(target), relative=target.relative_to(ROOT).as_posix(), remote=remote,
            source_image_id=chosen['source_image_id'], source_question_id=chosen['source_question_id'],
            sha256=chosen['sha256'], bytes=target.stat().st_size, staging=mode))
    return output, refs, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['doc54', 'text192', 'text256'], required=True)
    parser.add_argument('--method-freeze', type=Path, required=True,
                        help='Caller-prepared JSON freeze manifest; record its unchanged SHA256 before reading annotations')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    pool = read(POOL)
    verify_pool(pool)
    method = read(args.method_freeze)
    if not isinstance(method, dict) or not method:
        raise ValueError('Nonempty method freeze JSON object required')
    method_sha = sha(args.method_freeze)
    root = read(ROOT / 'configs/a2/round5_v1/mixed_dev64.json')['arguments']['model'].split('/models/', 1)[0]
    remote_root = root + '/VLAtest/TrajKV'
    is_doc = args.cohort == 'doc54'
    sample_path = ROOT / 'configs/a2/round6_v1' / ('doc_confirm54.json' if is_doc else 'text_transfer' + args.cohort[4:] + '.json')
    ref_path = ROOT / 'configs/data' / ('round6_doc54_references.jsonl' if is_doc else 'round6_text' + args.cohort[4:] + '_references.json')
    # Shared Text membership lock prevents staging both resource alternatives.
    membership = ROOT / 'audit/a2' / ('round6_doc54_membership.json' if is_doc else 'round6_text_membership_v1.json')
    image_dir = ROOT / 'data' / ('docvqa_round6_confirm54' if is_doc else 'round6_' + args.cohort) / 'images'
    for path in (sample_path, ref_path, membership, image_dir):
        if path.exists():
            raise FileExistsError('Refusing to reuse frozen/partial staging path: ' + str(path))
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN', cohort=args.cohort, annotations_read=False,
                              files_written=False, pool_sha256=sha(POOL), method_freeze_sha256=method_sha)))
        return
    image_dir.mkdir(parents=True, exist_ok=False)
    if is_doc:
        samples, references, images = prepare_doc(pool, image_dir, remote_root)
    else:
        samples, references, images = prepare_text(pool, int(args.cohort[4:]), image_dir, remote_root)
    if sha(args.method_freeze) != method_sha:
        raise ValueError('Method freeze changed during staging')
    if any(any(key in row for key in ('answer', 'answers', 'references')) for row in samples):
        raise ValueError('Reference answers in inference runner')
    write_new(sample_path, samples)
    write_new(ref_path, references, jsonl=is_doc)
    audit = dict(schema='trajkv.round6.fresh-cohort.membership.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), cohort=args.cohort,
        images=len(samples), questions=sum(row['turns'] for row in samples),
        pool_inspection_sha256=sha(POOL), method_freeze=str(args.method_freeze.resolve()),
        method_freeze_sha256=method_sha,
        selection_fixed_before_answers_accessed=True, source_content_overlap=0,
        annotations_used_for_selection=False, model_outcomes_used_for_selection=False,
        samples=sample_path.relative_to(ROOT).as_posix(), samples_sha256=sha(sample_path),
        references=ref_path.relative_to(ROOT).as_posix(), references_sha256=sha(ref_path),
        images_manifest=images, answers_uploaded=False, uploaded=False,
        transport='caller uploads images and answer-free sample runner only; references stay local',
        source_sha256=sha(__file__))
    write_new(membership, audit)
    print(json.dumps(dict(status='STAGED_LOCAL_ONLY', cohort=args.cohort, images=len(samples),
        questions=audit['questions'], samples=str(sample_path), references=str(ref_path), uploaded=False)))


if __name__ == '__main__':
    main()
