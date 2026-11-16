"""Read-only identity/content inspection of candidate fresh round-six pools.

The output is a proposed source list, not staged inference input or a method
freeze. Never read annotation/reference files or inference records. Existing
runner lists and manifest selected_ids are conservative exposure reservations.
"""
import argparse
import collections
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SALT_DOC = 'trajkv-next-docvqa-v1:'
SALT_TEXT = 'trajkv-round6-text-transfer-source-v1:'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def content_ids(row):
    hashes = set(row.get('hashes', []))
    for key in ('image_id', 'source_image_id'):
        value = str(row.get(key, ''))
        if value.startswith('sha256:'):
            hashes.add(value[7:])
    return hashes


def stable(salt, value):
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()


def project_path(value):
    text = str(value).replace('\\', '/')
    if '/TrajKV/' in text:
        return ROOT / text.split('/TrajKV/', 1)[1]
    return Path(value)


def registry():
    doc_used = set(read(ROOT / 'audit/a2/next_docvqa_source_split_v1.json')['excluded_source_documents'])
    text_used, doc_hashes, text_hashes, inputs = set(), set(), set(), []
    all_runner_rows = {}
    standard = ROOT / 'configs/a2/textvqa_standard5000.json'

    def add(rows):
        for row in rows:
            if row.get('dataset') == 'docvqa_grouped':
                if row.get('source_document_id'):
                    doc_used.add(str(row['source_document_id']))
                doc_hashes.update(content_ids(row))
            elif row.get('dataset') == 'textvqa':
                if row.get('source_original_image_id') is not None:
                    text_used.add(str(row['source_original_image_id']))
                text_hashes.update(content_ids(row))

    for path in sorted((ROOT / 'configs/a2').rglob('*.json')):
        # Exclude all reference names even though references live separately.
        if any(x in path.name.lower() for x in ('references', 'refs')):
            continue
        value = read(path)
        if not isinstance(value, list) or not value or not isinstance(value[0], dict):
            continue
        if value[0].get('dataset') not in ('docvqa_grouped', 'textvqa'):
            continue
        all_runner_rows[path.resolve()] = {row['id']: row for row in value}
        if path.resolve() == standard.resolve():
            continue
        add(value)
        inputs.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                           kind='registered subset reservation', rows=len(value)))
    # Inspect only metadata and selected identity membership, never records.
    # If a full5000 worker planned a source, reserve it conservatively too.
    for path in sorted((ROOT / 'results').rglob('manifest.json')):
        value = read(path)
        config = value.get('config', {}) if isinstance(value, dict) else {}
        if not isinstance(config, dict):
            continue
        sample_value = config.get('arguments', {}).get('samples')
        if not sample_value:
            continue
        source = project_path(sample_value).resolve()
        if source not in all_runner_rows:
            continue
        selected = value.get('selected_ids', [])
        lookup = all_runner_rows[source]
        if not set(selected) <= set(lookup):
            raise ValueError('Manifest selected identity missing from source: ' + str(path))
        add([lookup[sid] for sid in selected])
        inputs.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                           kind='manifest selected-ID reservation; no record opened', selected=len(selected)))
    # The legacy JSONL has annotations. Decode only identity fields by name.
    legacy = ROOT.parent / 'benchmark_samples/textvqa_samples.jsonl'
    decoder = json.JSONDecoder()
    paths = set()
    for line in legacy.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = {}
        for field in ('image_id', 'image_path'):
            match = re.search(r'"' + field + r'"\s*:\s*', line)
            if not match:
                raise ValueError('Missing legacy identity')
            fields[field] = decoder.raw_decode(line, match.end())[0]
        text_used.add(str(fields['image_id']))
        name = str(fields['image_path']).replace('\\', '/').rsplit('/', 1)[-1]
        paths.add(ROOT.parent / 'benchmark_samples/textvqa_images' / name)
    for path in paths:
        text_hashes.add(sha(path))
    inputs.append(dict(path=str(legacy), sha256=sha(legacy),
                       kind='legacy identities only; image content hashes', images=len(paths)))
    for path in sorted((ROOT / 'audit/a2').glob('*membership*.json')):
        if path.name in ('textvqa_standard5000_membership.json',):
            continue
        value = read(path)
        relevant = False
        for item in value.get('images_manifest', []):
            if 'text' in path.name:
                text_hashes.add(item['sha256'])
                if item.get('source_image_id') is not None:
                    text_used.add(str(item['source_image_id']))
                relevant = True
        if relevant:
            inputs.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                               kind='existing membership identities'))
    return doc_used, text_used, doc_hashes, text_hashes, inputs


def inspect_doc(used, used_hashes):
    archive_path = ROOT / 'audit/a2/docvqa_validation_download.json'
    archive = read(archive_path)
    expected = {item['file']: item['sha256'] for item in archive['files']}
    shards = sorted((ROOT / 'data/docvqa_validation_pinned').glob('*.parquet'))
    if {path.name for path in shards} != set(expected):
        raise ValueError('Pinned Doc shard inventory changed')
    documents = collections.defaultdict(lambda: collections.defaultdict(set))
    all_sources = set()
    for path in shards:
        if sha(path) != expected[path.name]:
            raise ValueError('Pinned Doc shard bytes changed')
        for batch in pq.ParquetFile(path).iter_batches(columns=['docId', 'ucsf_document_id', 'questionId']):
            for row in batch.to_pylist():
                source = str(row['ucsf_document_id'])
                all_sources.add(source)
                if source not in used:
                    documents[source][row['docId']].add(int(row['questionId']))
    ordered = sorted(documents, key=lambda source: stable(SALT_DOC, source))
    previous = read(ROOT / 'audit/a2/round5_doc64_membership.json')
    if ordered != previous['unselected_remaining_source_ids']:
        raise ValueError('Current unused Doc sources differ from registered round5 remaining list')
    pages = {source: min(documents[source], key=lambda page: stable(SALT_DOC, source + ':page:' + str(page)))
             for source in ordered}
    by_page = {page: source for source, page in pages.items()}
    images = {}
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=['docId', 'image']):
            for row in batch.to_pylist():
                page = row['docId']
                if page not in by_page or page in images:
                    continue
                raw = row['image']['bytes']
                digest = hashlib.sha256(raw).hexdigest()
                if digest in used_hashes:
                    raise ValueError('Unused source collides with an earlier encoded image')
                with Image.open(io.BytesIO(raw)) as im:
                    fmt, size = im.format, list(im.size)
                    im.verify()
                images[page] = dict(sha256=digest, bytes=len(raw), image_format=fmt, image_size=size)
    candidates = [dict(source_document_id=source, source_page_id=pages[source],
                       source_question_ids=sorted(documents[source][pages[source]])[:6],
                       **images[pages[source]]) for source in ordered]
    if len({row['sha256'] for row in candidates}) != len(candidates):
        raise ValueError('Unused Doc source pages reuse encoded image content')
    return dict(source_repo=archive['repo'], source_revision=archive['revision'],
        shards_sha256=expected, pinned_total_source_documents=len(all_sources),
        registered_excluded_source_documents=len(used & all_sources),
        eligible_sources=len(candidates), chosen_page_questions=sum(len(row['source_question_ids']) for row in candidates),
        selected_page_bytes=sum(row['bytes'] for row in candidates), selection_salt=SALT_DOC,
        selection='all remaining registered-source-disjoint documents in existing stable source hash order; original independent page hash; first6 numeric question IDs',
        source_and_content_overlap=0, candidates=candidates,
        proposed_cohort='Use all 54 as one final small fresh Doc cohort after method freeze; no tuning on these sources',
        limitation='fresh against enumerated TrajKV identities; unrelated earlier project exposure is not exhaustively known')


def inspect_text(used, used_hashes):
    source_path = ROOT / 'configs/a2/textvqa_standard5000.json'
    official_path = ROOT / 'audit/a2/textvqa_standard5000_membership.json'
    official, source = read(official_path), read(source_path)
    if sha(source_path) != official['samples_sha256'] or len(source) != 5000:
        raise ValueError('Pinned complete Text runner changed')
    grouped = collections.defaultdict(list)
    by_hash = {}
    for row in source:
        grouped[str(row['source_original_image_id'])].append(row)
        hashes = content_ids(row)
        if len(hashes) != 1:
            raise ValueError('Expected one Text encoded content ID')
        by_hash[next(iter(hashes))] = str(row['source_original_image_id'])
    used.update(by_hash[digest] for digest in used_hashes if digest in by_hash)
    candidates = []
    for original, rows in grouped.items():
        row = min(rows, key=lambda item: int(item['source_question_id']))
        digest = next(iter(content_ids(row)))
        if original in used or digest in used_hashes:
            continue
        candidates.append(dict(source_image_id=original, source_question_id=row['source_question_id'],
            source_session_id=row['id'], sha256=digest,
            selection_hash=stable(SALT_TEXT, original), raw_source=row['images'][0]))
    candidates.sort(key=lambda item: (item['selection_hash'], item['source_image_id']))
    if len(candidates) < 256:
        raise ValueError('Fewer than 256 clean Text source images remain')
    # Verify byte identity and image readability of the proposed maximum pool.
    seen = set()
    for row in candidates[:256]:
        path = Path(row['raw_source'])
        if sha(path) != row['sha256'] or row['sha256'] in seen:
            raise ValueError('Proposed Text image missing, duplicated, or changed')
        seen.add(row['sha256'])
        with Image.open(path) as image:
            row['image_format'], row['image_size'] = image.format, list(image.size)
            image.verify()
        row['bytes'] = path.stat().st_size
    return dict(official_repo=official['official_repo'], official_revision=official['official_revision'],
        source=source_path.relative_to(ROOT).as_posix(), source_sha256=sha(source_path),
        official_registry_sha256=sha(official_path), official_source_images=len(grouped),
        excluded_source_images=len(used & set(grouped)), eligible_source_images=len(candidates),
        proposed_maximum=256, proposed_smaller=192, proposed256_bytes=sum(x['bytes'] for x in candidates[:256]),
        proposed192_bytes=sum(x['bytes'] for x in candidates[:192]), selection_salt=SALT_TEXT,
        selection='first256 fixed salted source-ID hashes from audited exclusion union; lowest numeric questionId per image; first192 is a nested fallback',
        source_and_content_overlap=0, candidates256=candidates[:256],
        candidate256_identity_sha256=hashlib.sha256(json.dumps([
            {k:row[k] for k in ('source_image_id', 'source_question_id', 'sha256')}
            for row in candidates[:256]], sort_keys=True).encode()).hexdigest(),
        proposed_cohort='Prefer one fixed Text256 cohort for transfer; Text192 is a prespecified resource fallback, not a score-driven choice',
        freeze_rule='freeze methods/rankings/gains only on Doc development before staging/inference; do not calibrate on Text',
        metric='official ten-annotator TextVQA soft accuracy; annotations remain separate local files')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    du, tu, dh, th, inputs = registry()
    report = dict(schema='trajkv.round6.unseen-pool-inspection.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        staged=False, uploaded=False, annotation_files_read=False,
        inference_record_files_read=False, model_scores_read=False,
        selection_inputs='source identities, pinned bytes, prior registered reservations and manifest selected_ids only',
        doc=inspect_doc(du, dh), text=inspect_text(tu, th), registry_inputs=inputs,
        source_sha256=sha(__file__))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(doc_sources=report['doc']['eligible_sources'],
        doc_questions=report['doc']['chosen_page_questions'],
        doc_bytes=report['doc']['selected_page_bytes'],
        text_eligible=report['text']['eligible_source_images'],
        text256_bytes=report['text']['proposed256_bytes'],
        text192_bytes=report['text']['proposed192_bytes'],
        annotations_read=False, model_records_read=False, staged=False,
        output=str(args.output) if args.output else None)))


if __name__ == '__main__':
    main()
