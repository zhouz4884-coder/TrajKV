"""Download pinned public DocVQA train shards and freeze answer-free candidates.

Only identity columns and original image bytes are decoded before the caller's
method freeze. All caches/download partials stay under this project's data/.
"""
import argparse
import collections
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq
from inspect_round6_unseen_pools import registry

ROOT = Path(__file__).resolve().parents[2]
REPO = 'lmms-lab-encoder/DocVQA'
REVISION = '539088ef8a8ada01ac8e2e6d4e372586748a265e'
SALT = 'trajkv-round7-doc-extra-confirm128-source-v1:'
DATA = ROOT / 'data/docvqa_round7_train_pinned'
META = ROOT / 'audit/a2/round7_doc_extra_official_metadata_v1.json'
POOL = ROOT / 'audit/a2/round7_doc_confirm128_candidate_pool_v1.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256((SALT + str(value)).encode()).hexdigest()


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def metadata():
    if META.exists():
        return json.loads(META.read_text())
    endpoint = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/DocVQA?recursive=false&expand=false&limit=1000'
    items = json.load(urllib.request.urlopen(endpoint, timeout=45))
    files = [dict(path=x['path'], bytes=x['size'], sha256=x['lfs']['oid']) for x in items if '/train-' in x['path']]
    files.sort(key=lambda x: (x['bytes'], x['path']))
    result = dict(schema='trajkv.round7.doc-extra-official-metadata.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), repo=REPO, revision=REVISION,
        metadata_endpoint=endpoint, files=files, chosen_shards=files[:2],
        shard_selection_rule='two smallest public train shard payloads at fixed official revision; metadata-only resource choice before annotations/outcomes',
        annotations_read=False, model_records_read=False,
        note='Official upstream dataset split is train; held out only from this project calibration/development, not guaranteed absent from backbone pretraining.')
    new(META, result)
    return result


def download(meta):
    DATA.mkdir(parents=True, exist_ok=True)
    for item in meta['chosen_shards']:
        path = DATA / Path(item['path']).name
        if path.exists():
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Existing pinned source bytes differ')
            continue
        partial = path.with_suffix(path.suffix + '.partial')
        if partial.exists():
            raise FileExistsError('Partial download retained; use an explicit new run or inspect it: ' + str(partial))
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item["path"]}'
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as stream:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        if partial.stat().st_size != item['bytes'] or sha(partial) != item['sha256']:
            raise ValueError('Downloaded bytes differ from pinned LFS identity')
        partial.rename(path)
        print(json.dumps(dict(downloaded=path.name, bytes=item['bytes'], sha256=item['sha256'])), flush=True)


def inspect(meta):
    if POOL.exists():
        raise FileExistsError('Candidate pool already frozen')
    used, _, used_hashes, _, registry_inputs = registry()
    docs = collections.defaultdict(lambda: collections.defaultdict(set))
    shards = [DATA / Path(item['path']).name for item in meta['chosen_shards']]
    for path, item in zip(shards, meta['chosen_shards']):
        if sha(path) != item['sha256']:
            raise ValueError('Source parquet bytes changed')
        for batch in pq.ParquetFile(path).iter_batches(columns=['docId', 'ucsf_document_id', 'questionId']):
            for row in batch.to_pylist():
                source = str(row['ucsf_document_id'])
                if source not in used:
                    docs[source][row['docId']].add(int(row['questionId']))
    pages = {source: min(pages, key=lambda page: stable(source + ':page:' + str(page))) for source, pages in docs.items()}
    by_page = {page: source for source, page in pages.items()}
    # Read candidate page image bytes; answers and question text remain unopened.
    images = {}
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4, columns=['docId', 'image']):
            for row in batch.to_pylist():
                page = row['docId']
                if page not in by_page or page in images:
                    continue
                raw = row['image']['bytes']
                digest = hashlib.sha256(raw).hexdigest()
                if digest in used_hashes:
                    continue
                with Image.open(io.BytesIO(raw)) as image:
                    fmt, size = image.format, list(image.size)
                    image.verify()
                images[page] = dict(sha256=digest, bytes=len(raw), format=fmt, image_size=size, shard=path.name)
    candidates, seen = [], set()
    for source in sorted(pages, key=lambda x: (stable(x), x)):
        page = pages[source]
        if page not in images or images[page]['sha256'] in seen:
            continue
        seen.add(images[page]['sha256'])
        candidates.append(dict(source_document_id=source, source_page_id=page,
            source_question_ids=sorted(docs[source][page])[:4], selection_hash=stable(source), **images[page]))
    if len(candidates) < 128:
        raise ValueError('Fixed resource shard subset yields fewer than 128 clean source images')
    chosen = candidates[:128]
    result = dict(schema='trajkv.round7.doc-confirm128-candidate-pool.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), repo=REPO, revision=REVISION,
        official_metadata_sha256=sha(META), shard_identities=meta['chosen_shards'], prior_registry_inputs=registry_inputs,
        prior_reserved_doc_sources=len(used), prior_content_hashes=len(used_hashes),
        source_disjoint_candidates=len(candidates), proposed_count=128,
        proposed_question_count=sum(len(r['source_question_ids']) for r in chosen),
        proposed_image_bytes=sum(r['bytes'] for r in chosen), salt=SALT,
        selection_rule='exclude prior source/content union; fixed salted source hash; one independently hashed page/source; first4 numeric question IDs; 128 unique encoded images',
        source_and_content_overlap=0, question_text_read=False, answers_read=False,
        inference_records_read=False, selection_used_model_outcomes=False, method_freeze_required_before_text_or_answers=True,
        candidates128=chosen, remaining_candidate_source_ids=[r['source_document_id'] for r in candidates[128:]],
        original_split='train', scope='new project source holdout; no claim of backbone-pretraining exclusion',
        source_sha256=sha(__file__))
    new(POOL, result)
    print(json.dumps(dict(candidate_pool=str(POOL), available_sources=len(candidates), selected_sources=128,
        questions=result['proposed_question_count'], bytes=result['proposed_image_bytes'], answers_read=False)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true')
    p.add_argument('--inspect', action='store_true')
    args = p.parse_args()
    meta = metadata()
    if args.download:
        download(meta)
    if args.inspect:
        inspect(meta)


if __name__ == '__main__':
    main()
