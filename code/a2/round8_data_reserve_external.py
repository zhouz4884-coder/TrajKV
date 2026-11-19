"""Reserve InfoVQA official-val64 and project-fresh Doc TRAIN64 without GT.

Pinned downloads, partials and original images stay on the project data disk.
No question/answer values, model outcomes, SSH or GPU are used for selection.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from PIL import Image
import pyarrow.parquet as pq
from inspect_round6_unseen_pools import registry, content_ids
from round8_data_prepare import ROOT, read, sha, new

REPO = 'lmms-lab-encoder/DocVQA'
REV = '539088ef8a8ada01ac8e2e6d4e372586748a265e'
INFO_DATA = ROOT / 'data/infovqa_validation_round8_pinned'


def metadata_and_download():
    manifest = ROOT / 'audit/a2/round8_infovqa_official_metadata_v1.json'
    if manifest.exists():
        meta = read(manifest)
    else:
        url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/InfographicVQA?recursive=false&expand=false&limit=1000'
        values = json.load(urllib.request.urlopen(url, timeout=45))
        files = [dict(path=row['path'], bytes=row['size'], sha256=row['lfs']['oid'])
                 for row in values if '/validation-' in row['path']]
        assert len(files) == 4
        meta = dict(schema='trajkv.round8.infovqa-pinned-validation.v1', repo=REPO, revision=REV,
            source_split='official validation', metadata_endpoint=url, files=files,
            original_dataset_homepage='https://site.docvqa.org/datasets/infographicvqa',
            selection_scope='All four official-validation mirror shards, not a smallest-shard convenience subset',
            annotations_decoded=False)
        new(manifest, meta)
    INFO_DATA.mkdir(parents=True, exist_ok=True)
    for item in meta['files']:
        target = INFO_DATA / Path(item['path']).name
        if target.exists():
            assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
            continue
        partial = target.with_suffix('.parquet.partial')
        assert not partial.exists(), 'Partial preserved; use an explicitly versioned repair rather than overwriting: ' + str(partial)
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{item["path"]}'
        with urllib.request.urlopen(url, timeout=90) as response, partial.open('xb') as stream:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                stream.write(block)
        assert partial.stat().st_size == item['bytes'] and sha(partial) == item['sha256']
        partial.rename(target)
        print(json.dumps(dict(downloaded=target.name, bytes=item['bytes'], LFS_SHA='PASS')), flush=True)
    return meta


def exposures():
    doc_used, text_used, dh, th, inputs = registry()
    hashes, info_sources = set(dh) | set(th), set()
    for path in sorted((ROOT / 'configs/a2').rglob('*.json')):
        if path.name == 'textvqa_standard5000.json' or any(word in path.name.lower() for word in ('refs', 'references')):
            continue
        value = read(path)
        if not isinstance(value, list) or not value or not isinstance(value[0], dict):
            continue
        for row in value:
            if row.get('dataset'):
                hashes.update(content_ids(row))
            if row.get('dataset') in ('infovqa', 'infographicvqa'):
                for key in ('source_original_image_id', 'source_image_name', 'source_document_id'):
                    if row.get(key) is not None:
                        info_sources.add(str(row[key]))
    return doc_used, hashes, info_sources, inputs


def inspect_and_stage_images(kind, shards, excluded_sources, excluded_hashes, inputs, metadata):
    output = ROOT / f'audit/a2/round8_{kind}64_source_membership_v1.json'
    assert not output.exists(), 'Source cohort already reserved; use a new version'
    doc = kind == 'doc_train'
    first = pq.ParquetFile(shards[0]).schema_arrow.names
    if doc:
        source_key, page_key, qid_key = 'ucsf_document_id', 'docId', 'questionId'
    else:
        source_key = next((key for key in ('image_local_name', 'image_id', 'imageId', 'docId') if key in first), None)
        assert source_key is not None, 'Original image identity field missing: ' + repr(first)
        page_key = source_key
        qid_key = next(key for key in ('questionId', 'question_id', 'id') if key in first)
    salt = 'trajkv-round8-' + kind + '-source64-v1:'
    stable = lambda text: hashlib.sha256((salt + str(text)).encode()).hexdigest()
    identities = defaultdict(lambda: defaultdict(set))
    for path in shards:
        columns = list(dict.fromkeys([source_key, page_key, qid_key]))
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            for row in batch.to_pylist():
                source, page = str(row[source_key]), row[page_key]
                if source not in excluded_sources:
                    identities[source][str(page)].add(str(row[qid_key]))
    pages = {source: min(values, key=lambda page: stable(source + ':page:' + page)) for source, values in identities.items()}
    desired = {pages[source]: source for source in identities}
    images = {}
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4, columns=list(dict.fromkeys([page_key, 'image']))):
            for row in batch.to_pylist():
                page = str(row[page_key])
                if page not in desired or page in images:
                    continue
                raw = row['image']['bytes']
                digest = hashlib.sha256(raw).hexdigest()
                if digest in excluded_hashes:
                    images[page] = None
                    continue
                with Image.open(io.BytesIO(raw)) as image:
                    fmt, dimensions = image.format, list(image.size); image.verify()
                images[page] = dict(raw=raw, sha256=digest, bytes=len(raw), format=fmt, dimensions=dimensions, shard=path.name)
    selected, seen = [], set()
    image_dir = ROOT / ('data/round8_' + kind + '64/images'); image_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(identities, key=lambda value: (stable(value), value)):
        page = pages[source]; image = images.get(page)
        if image is None or image['sha256'] in seen:
            continue
        seen.add(image['sha256'])
        suffix = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tif'}.get(image['format'])
        assert suffix, 'Unsupported original image format'
        target = image_dir / (image['sha256'] + suffix)
        if target.exists():
            assert sha(target) == image['sha256']
        else:
            with target.open('xb') as stream:
                stream.write(image['raw'])
        qids = sorted(identities[source][page], key=lambda value: (0, int(value)) if value.isdigit() else (1, value))[:4 if doc else 1]
        selected.append(dict(source_document_id=source if doc else None, source_image_id=source,
            source_page_id=page, source_question_ids=qids, sha256=image['sha256'], bytes=image['bytes'],
            image_size=image['dimensions'], image_format=image['format'], original_shard=image['shard'],
            local=str(target), relative=target.relative_to(ROOT).as_posix(), selection_hash=stable(source)))
        if len(selected) == 64:
            break
    assert len(selected) == 64, 'Insufficient source/content-disjoint original images'
    result = dict(schema='trajkv.round8.external-source64-membership.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), dataset='docvqa' if doc else 'infovqa',
        original_split='train' if doc else 'validation', images=64, questions=sum(len(row['source_question_ids']) for row in selected),
        fresh_scope='project-fresh TRAIN source holdout, not validation/test' if doc else 'project-fresh official InfoVQA validation external cohort',
        backbone_pretraining_exclusion_claimed=False, repo=REPO, revision=REV,
        source_identity_field=source_key, question_identity_field=qid_key, page_identity_field=page_key,
        source_shards=[dict(file=path.name, sha256=sha(path)) for path in shards],
        eligible_sources_before_content_check=len(identities), selection_salt=salt,
        selection_rule='source hash; independently hashed page where applicable; first4 numeric question IDs for Doc train, first1 for Info; no question/answer/outcome selection',
        images_manifest=selected, source_content_overlap=0, registry_inputs=inputs,
        identity_columns_decoded=list(dict.fromkeys([source_key, page_key, qid_key])), question_text_decoded=False,
        answer_values_decoded=False, model_outcomes_read=False, original_images_staged_local=True,
        method_freeze_required_before_QA_materialization=True, uploaded=False, GPU_used=False,
        source_metadata=metadata, code_sha256=sha(__file__))
    new(output, result)
    print(json.dumps(dict(status='SOURCE_IMAGES_RESERVED', cohort=kind, sources=64,
        questions=result['questions'], image_bytes=sum(row['bytes'] for row in selected),
        membership=str(output), answers_decoded=False, uploaded=False)), flush=True)


def main():
    meta = metadata_and_download()
    used, hashes, info_used, inputs = exposures()
    info_shards = [INFO_DATA / Path(item['path']).name for item in meta['files']]
    inspect_and_stage_images('infovqa_val', info_shards, info_used, hashes, inputs, meta)
    doc_meta = read(ROOT / 'audit/a2/round7_doc_extra_official_metadata_v1.json')
    doc_shards = [ROOT / 'data/docvqa_round7_train_pinned' / Path(item['path']).name for item in doc_meta['chosen_shards']]
    inspect_and_stage_images('doc_train', doc_shards, used, hashes, inputs, doc_meta)


if __name__ == '__main__':
    main()
