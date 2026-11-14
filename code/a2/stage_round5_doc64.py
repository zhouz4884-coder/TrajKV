"""Freeze 64 unused DocVQA source documents without consulting model outcomes.

The pinned public validation export has 118 sources left after all registered
TrajKV cohorts. Reuse the existing stable source/page order, extract original
encoded image bytes, and keep scoring references local. --dry-run writes
nothing; --upload can be invoked later against the verified frozen membership.
"""
import argparse
import hashlib
import io
import json
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from remote import connect, mkdirs, REMOTE_ROOT, PYTHON

SAMPLES = ROOT/'configs/a2/round5_v1/doc_confirm64.json'
REFERENCES = ROOT/'configs/data/round5_doc64_references.jsonl'
MEMBERSHIP = ROOT/'audit/a2/round5_doc64_membership.json'
SALT = 'trajkv-next-docvqa-v1:'
EXPECTED_IDS_SHA256 = 'ac1f8af36a6231a1e23eaad86a237db1ec449a767e12b827598d367ec8db01cb'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable(text):
    return hashlib.sha256((SALT+text).encode()).hexdigest()


def dump_new_or_identical(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError('Refusing to overwrite a different frozen input: '+str(path))
    if not path.exists():
        path.write_bytes(payload)


def exclusions():
    previous_path = ROOT/'audit/a2/next_docvqa_source_split_v1.json'
    previous = load(previous_path)
    used = set(previous['excluded_source_documents'])
    hashes = set()
    registered = [previous_path]
    cohorts = list(previous['manifests'].values()) + [
        'configs/a2/docvqa_future128.json', 'configs/a2/docvqa_meta128.json']
    counts = {'legacy1000_sources': len(used)}
    for relative in cohorts:
        path = ROOT/relative
        rows = load(path)
        documents = {row['source_document_id'] for row in rows}
        if used & documents:
            raise ValueError('Registered supposedly disjoint cohorts overlap: '+relative)
        used.update(documents)
        hashes.update(x for row in rows for x in row['hashes'])
        registered.append(path)
        counts[relative] = len(documents)
    registered.extend(ROOT/relative for relative in [
        'audit/a2/docvqa_validation_download.json',
        'audit/a2/future_doc128_membership.json', 'audit/a2/meta_doc128_membership.json'])
    # The original split already proves old question order and identifies all
    # 212 UCSF sources in legacy1000, including the early dev/confirm cohorts.
    for label, relative in previous['manifests'].items():
        if sha(ROOT/relative) != previous['input_sha256'][relative]:
            raise RuntimeError('Original split manifest changed: '+label)
    for stem, relative in [('future', 'configs/a2/docvqa_future128.json'),
                            ('meta', 'configs/a2/docvqa_meta128.json')]:
        membership = load(ROOT/f'audit/a2/{stem}_doc128_membership.json')
        if sha(ROOT/relative) != membership['samples_sha256']:
            raise RuntimeError('Later cohort manifest changed: '+relative)
    return previous, used, hashes, counts, {
        path.relative_to(ROOT).as_posix(): sha(path) for path in registered}


def selection():
    previous, used, used_hashes, counts, registry = exclusions()
    archive = load(ROOT/'audit/a2/docvqa_validation_download.json')
    expected = {row['file']: row['sha256'] for row in archive['files']}
    shards = sorted((ROOT/'data/docvqa_validation_pinned').glob('*.parquet'))
    if {path.name for path in shards} != set(expected):
        raise RuntimeError('Pinned validation shard inventory differs')
    # Selection needs only source/page IDs. Answers are read after membership
    # has already been fixed by these ID hashes.
    documents = defaultdict(set)
    for path in shards:
        for row in pq.read_table(path, columns=['docId', 'ucsf_document_id']).to_pylist():
            if row['ucsf_document_id'] not in used:
                documents[row['ucsf_document_id']].add(row['docId'])
    ordered = sorted(documents, key=stable)
    if len(ordered) < 64:
        raise RuntimeError('Fewer than 64 unused sources remain')
    chosen = ordered[:64]
    ids_sha = hashlib.sha256(json.dumps(chosen, separators=(',', ':')).encode()).hexdigest()
    if ids_sha != EXPECTED_IDS_SHA256:
        raise RuntimeError('Selection differs from the independently checked round5 source order')
    pages = {name: min(documents[name], key=lambda page: stable(name+':page:'+str(page))) for name in chosen}
    return dict(previous=previous, used=used, used_hashes=used_hashes, counts=counts,
        registry=registry, archive=archive, shards=shards, expected=expected,
        ordered=ordered, chosen=chosen, pages=pages, ids_sha=ids_sha)


def verify_frozen(report):
    for relative, expected in report['registry_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise RuntimeError('Frozen exclusion registry changed: '+relative)
    for relative, expected in report['input_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise RuntimeError('Frozen round5 input changed: '+relative)
    if report['ordered_source_ids_sha256'] != EXPECTED_IDS_SHA256:
        raise RuntimeError('Frozen source IDs differ')
    return report


def prepare():
    if MEMBERSHIP.exists():
        return verify_frozen(load(MEMBERSHIP))
    data = selection()
    for path in data['shards']:
        if sha(path) != data['expected'][path.name]:
            raise RuntimeError('Pinned public shard changed: '+path.name)
    selected = {page: {'source': name, 'rows': []} for name, page in data['pages'].items()}
    columns = ['questionId', 'question', 'question_types', 'docId', 'ucsf_document_id', 'answers']
    for path in data['shards']:
        for row in pq.read_table(path, columns=columns).to_pylist():
            if row['docId'] in selected:
                entry = selected[row['docId']]
                if row['ucsf_document_id'] != entry['source']:
                    raise RuntimeError('Page/source identity collision')
                entry['rows'].append(row)
    image_dir = ROOT/'data/docvqa_round5_confirm64/images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for path in data['shards']:
        for row in pq.read_table(path, columns=['docId', 'image']).to_pylist():
            entry = selected.get(row['docId'])
            if entry is None or 'image_path' in entry:
                continue
            raw = row['image']['bytes']
            digest = hashlib.sha256(raw).hexdigest()
            if digest in data['used_hashes']:
                raise RuntimeError('Selected source reuses a registered encoded image')
            with Image.open(io.BytesIO(raw)) as image:
                fmt, size = image.format, list(image.size)
                image.verify()
            suffix = {'PNG': '.png', 'JPEG': '.jpg', 'TIFF': '.tif'}[fmt]
            target = image_dir/(digest+suffix)
            if target.exists() and sha(target) != digest:
                raise RuntimeError('Existing extracted image has different bytes')
            if not target.exists():
                target.write_bytes(raw)
            entry.update(image_path=target, digest=digest, size=size)
    samples, references, images = [], [], []
    for name in data['chosen']:
        page = data['pages'][name]
        entry = selected[page]
        rows = sorted(entry['rows'], key=lambda row: int(row['questionId']))[:6]
        if not rows or 'image_path' not in entry:
            raise RuntimeError('Missing selected questions/image')
        path, digest = entry['image_path'], entry['digest']
        relative = path.relative_to(ROOT).as_posix()
        remote = REMOTE_ROOT+'/'+relative
        sid = 'docvqa_round5_'+str(page)
        questions = [row['question']+'\nAnswer the question using a single word or phrase.' for row in rows]
        samples.append(dict(schema='trajkv.session.v1', dataset='docvqa_grouped',
            id=sid, session_id=sid, image_id='sha256:'+digest, source_image_id='sha256:'+digest,
            source_document_id=name, source_page_id=page, images=[str(path)], remote_images=[remote],
            hashes=[digest], questions=questions, turns=len(questions),
            content=[{'type': 'image'}, {'type': 'text', 'text': questions[0]}],
            protocol='derived_same_document_independent_qa_grouped',
            history_dependence='not_assumed_requires_qualification',
            split='round5_fresh_source_confirm64',
            answer_format_protocol='internvl_official_short_phrase_prompt_v1',
            historical_result_reused=False, source_question_ids=[row['questionId'] for row in rows]))
        references.append(dict(id=sid, source_document_id=name, references=[dict(turn=i+1,
            answer=row['answers'], question_id=row['questionId'], question_types=row['question_types'])
            for i, row in enumerate(rows)]))
        images.append(dict(relative=relative, local=str(path), remote=remote,
            sha256=digest, bytes=path.stat().st_size, size=entry['size']))
    if len({row['image_id'] for row in samples}) != 64:
        raise RuntimeError('Distinct selected sources reuse image bytes')
    dump_new_or_identical(SAMPLES, samples)
    ref_bytes = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in references).encode('utf-8')
    REFERENCES.parent.mkdir(parents=True, exist_ok=True)
    if REFERENCES.exists() and REFERENCES.read_bytes() != ref_bytes:
        raise RuntimeError('Refusing to overwrite different frozen references')
    if not REFERENCES.exists():
        REFERENCES.write_bytes(ref_bytes)
    inputs = {SAMPLES.relative_to(ROOT).as_posix(): sha(SAMPLES),
              REFERENCES.relative_to(ROOT).as_posix(): sha(REFERENCES)}
    inputs.update({row['relative']: row['sha256'] for row in images})
    report = dict(schema='trajkv.round5.doc64.membership.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), images=64, source_documents=64,
        questions=sum(row['turns'] for row in samples), source_repo=data['archive']['repo'],
        source_revision=data['archive']['revision'], shards_sha256=data['expected'],
        selection='exclude legacy212 + cal32/dev64/confirm256 + future128 + meta128; existing SHA256 source order; one page by independent ID hash; first6 numeric questionIds',
        selection_salt=SALT, ordered_source_ids=data['chosen'],
        ordered_source_ids_sha256=data['ids_sha'], source_documents_excluded=len(data['used']),
        excluded_source_documents=sorted(data['used']), exclusion_cohort_counts=data['counts'],
        remaining_before_selection=len(data['ordered']), remaining_after_selection=len(data['ordered'])-64,
        unselected_remaining_source_ids=data['ordered'][64:],
        samples=SAMPLES.relative_to(ROOT).as_posix(), references=REFERENCES.relative_to(ROOT).as_posix(),
        samples_sha256=sha(SAMPLES), references_sha256=sha(REFERENCES),
        registry_sha256=data['registry'], input_sha256=inputs, images_manifest=images,
        labels_used_for_selection=False, model_outcomes_used_for_selection=False,
        image_scope='original encoded bytes from pinned public validation export; no reencoding',
        scope='fresh relative to registered TrajKV source cohorts; unrelated earlier project exposure not exhaustively audited',
        answers_uploaded=False, source_sha256=sha(__file__))
    dump_new_or_identical(MEMBERSHIP, report)
    return report


def upload(report):
    verify_frozen(report)
    client = connect()
    stream = client.open_sftp()
    try:
        checks = {}
        for item in report['images_manifest']:
            mkdirs(stream, item['remote'].rsplit('/', 1)[0])
            stream.put(item['local'], item['remote'])
            checks[item['remote']] = item['sha256']
        remote = REMOTE_ROOT+'/'+report['samples']
        mkdirs(stream, remote.rsplit('/', 1)[0])
        stream.put(str(ROOT/report['samples']), remote)
        checks[remote] = report['samples_sha256']
        code = "import json,hashlib;from pathlib import Path;e=%r;bad=[p for p,h in e.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];print(json.dumps({'files':len(e),'mismatches':len(bad)}));assert not bad" % checks
        _, out, err = client.exec_command(shlex.join([PYTHON, '-c', code]), timeout=60)
        result, error = json.loads(out.read()), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError('Remote staged-data verification failed: '+error[-400:])
    finally:
        stream.close()
        client.close()
    transport = dict(status='PASS', **result, answers_uploaded=False,
        membership_sha256=sha(MEMBERSHIP), samples_sha256=report['samples_sha256'])
    dump_new_or_identical(ROOT/'audit/a2/round5_doc64_transport.json', transport)
    print(json.dumps(transport), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upload', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        if a.upload:
            raise ValueError('--dry-run cannot upload')
        data = selection()
        print(json.dumps(dict(status='DRY_RUN', writes=0, excluded=len(data['used']),
            remaining=len(data['ordered']), selected=64, remaining_after_selection=len(data['ordered'])-64,
            ordered_source_ids_sha256=data['ids_sha'], first5=data['chosen'][:5])), flush=True)
        return
    report = prepare()
    print(json.dumps(dict(status='FROZEN', images=64, questions=report['questions'],
        remaining=report['remaining_after_selection'], samples_sha256=report['samples_sha256'],
        ordered_source_ids_sha256=report['ordered_source_ids_sha256'])), flush=True)
    if a.upload:
        upload(report)


if __name__ == '__main__':
    main()
