"""Local-only, source-disjoint unstratified TextVQA transfer membership.

Dry-run reads no reference answers and writes no files. Actual staging selects
membership first, then attaches annotations in a separate local references file.
No SSH/upload capability is provided; the caller transports images and the
answer-free runner only after the Doc-selected method freeze.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SALT = 'trajkv-round5-text-transfer64-source-v1:'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def content_ids(row):
    values = set(row.get('hashes', []))
    for name in ('image_id', 'source_image_id'):
        value = str(row.get(name, ''))
        if value.startswith('sha256:'):
            values.add(value[7:])
    return values


def used_membership(source_runner):
    used_ids, used_hashes, sources = set(), set(), []
    legacy = ROOT.parent/'benchmark_samples/textvqa_samples.jsonl'
    rows = []
    decoder = json.JSONDecoder()
    for line in legacy.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        # Decode only the two identity fields of the historical JSONL, not
        # its question/annotation fields.
        identity = {}
        for field in ('image_id', 'image_path'):
            match = re.search(r'"'+field+r'"\s*:\s*', line)
            if match is None:
                raise ValueError('Missing legacy identity field: '+field)
            identity[field] = decoder.raw_decode(line, match.end())[0]
        rows.append(identity)
    unique_paths = set()
    for row in rows:
        used_ids.add(str(row['image_id']))
        name = row['image_path'].replace('\\', '/').rsplit('/', 1)[-1]
        unique_paths.add(ROOT.parent/'benchmark_samples/textvqa_images'/name)
    for path in sorted(unique_paths):
        if not path.is_file():
            raise FileNotFoundError('Legacy asset identity cannot be verified: '+str(path))
        used_hashes.add(sha(path))
    sources.append(dict(path=str(legacy), sha256=sha(legacy), kind='all legacy source assets',
                        rows=len(rows), distinct_source_ids=len(used_ids), distinct_contents=len(used_hashes)))
    # Include all registered small Text runner subsets. A complete staged
    # 5000-QA source is not evidence of an inference exposure.
    for path in sorted((ROOT/'configs/a2').rglob('*text*.json')):
        if path.resolve() == source_runner.resolve() or path.name == 'text_transfer64.json':
            continue
        if any(word in path.name for word in ('refs', 'references')):
            continue
        value = read(path)
        if not isinstance(value, list) or not value or not isinstance(value[0], dict) or value[0].get('dataset') != 'textvqa':
            continue
        before_id, before_hash = len(used_ids), len(used_hashes)
        for row in value:
            used_hashes.update(content_ids(row))
            if row.get('source_original_image_id') is not None:
                used_ids.add(str(row['source_original_image_id']))
        sources.append(dict(path=str(path), sha256=sha(path), kind='reserved or used Text subset', rows=len(value),
                            added_source_ids=len(used_ids)-before_id, added_contents=len(used_hashes)-before_hash))
    for path in sorted((ROOT/'audit/a2').glob('textvqa_*membership.json')):
        if path.name == 'textvqa_standard5000_membership.json':
            continue
        value = read(path)
        before = len(used_hashes)
        for image in value.get('images_manifest', []):
            used_hashes.add(image['sha256'])
            if image.get('source_image_id') is not None:
                used_ids.add(str(image['source_image_id']))
        sources.append(dict(path=str(path), sha256=sha(path), kind='Text subset membership registry',
                            added_contents=len(used_hashes)-before))
    # Read only Text-run records for identity, never Doc result records. This
    # catches partial full5000 runs if any were actually started later.
    for path in sorted((ROOT/'results').rglob('manifest.json')):
        value = read(path)
        config = value.get('config', {}) if isinstance(value, dict) else {}
        if not isinstance(config, dict):
            continue
        samples = str(config.get('arguments', {}).get('samples', '')).lower()
        if 'textvqa' not in samples and 'text_transfer' not in samples:
            continue
        records = path.with_name('records.jsonl')
        if not records.is_file() or not records.stat().st_size:
            continue
        before_hash, before_id, count = len(used_hashes), len(used_ids), 0
        with records.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                used_hashes.update(content_ids(record))
                if record.get('source_original_image_id') is not None:
                    used_ids.add(str(record['source_original_image_id']))
                count += 1
        sources.append(dict(path=str(records), sha256=sha(records), manifest_sha256=sha(path),
                            kind='observed Text inference identities; scores and predictions unused', rows=count,
                            added_contents=len(used_hashes)-before_hash, added_source_ids=len(used_ids)-before_id))
    return used_ids, used_hashes, sources


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT/'configs/a2/textvqa_standard5000.json')
    parser.add_argument('--source-references', type=Path, default=ROOT/'configs/data/textvqa_standard5000_references.json')
    parser.add_argument('--output', type=Path, default=ROOT/'configs/a2/round5_v1/text_transfer64.json')
    parser.add_argument('--references', type=Path, default=ROOT/'configs/data/round5_text64_references.json')
    parser.add_argument('--membership', type=Path, default=ROOT/'audit/a2/round5_text64_membership.json')
    parser.add_argument('--image-dir', type=Path, default=ROOT/'data/round5_text64/images')
    parser.add_argument('--image-mode', choices=('hardlink', 'copy'), default='hardlink')
    args = parser.parse_args()
    args.image_dir = args.image_dir.resolve()
    if not args.image_dir.resolve().is_relative_to((ROOT/'data').resolve()):
        raise ValueError('New image staging must remain on the project data disk')
    for path in (args.output, args.references, args.membership):
        if path.exists():
            raise FileExistsError('Refusing to overwrite frozen membership: '+str(path))
    source = read(args.source)
    official_marker = ROOT/'audit/a2/textvqa_standard5000_membership.json'
    official = read(official_marker)
    if len(source) != 5000 or sha(args.source) != official['samples_sha256']:
        raise ValueError('Complete pinned runner source changed')
    used_ids, used_hashes, registries = used_membership(args.source)
    grouped = collections.defaultdict(list)
    all_ids, all_hashes, by_hash = set(), set(), {}
    for row in source:
        original = str(row['source_original_image_id'])
        hashes = content_ids(row)
        if len(hashes) != 1:
            raise ValueError('One source image content hash required')
        digest = next(iter(hashes))
        all_ids.add(original)
        all_hashes.add(digest)
        by_hash[digest] = original
        grouped[original].append(row)
    # Resolve hash-only historical memberships into official source identities.
    used_ids.update(by_hash[digest] for digest in used_hashes if digest in by_hash)
    eligible = []
    for original, rows in grouped.items():
        row = min(rows, key=lambda r: int(r['source_question_id']))
        digest = next(iter(content_ids(row)))
        if original in used_ids or digest in used_hashes:
            continue
        order = hashlib.sha256((SALT+original).encode()).hexdigest()
        eligible.append((order, original, digest, row))
    eligible.sort(key=lambda item: (item[0], item[1]))
    selected, selected_hashes = [], set()
    for item in eligible:
        if item[2] in selected_hashes:
            continue
        selected.append(item)
        selected_hashes.add(item[2])
        if len(selected) == 64:
            break
    if len(selected) != 64:
        raise RuntimeError('Fewer than 64 source/content-disjoint Text images remain')
    from PIL import Image
    image_rows = []
    for order, original, digest, row in selected:
        path = Path(row['images'][0])
        if sha(path) != digest:
            raise RuntimeError('Original raw image hash mismatch')
        with Image.open(path) as image:
            image.verify()
        target = args.image_dir/(digest+path.suffix)
        remote_root = row['remote_images'][0].split('/data/', 1)[0]
        remote = remote_root+'/'+target.relative_to(ROOT).as_posix()
        image_rows.append(dict(source_image_id=original, source_question_id=row['source_question_id'],
                               source_session_id=row['id'], selection_hash=order,
                               raw_source=str(path), local=str(target), remote=remote,
                               sha256=digest, bytes=path.stat().st_size))
    selected_identity = [dict(source_image_id=x['source_image_id'], source_question_id=x['source_question_id'], sha256=x['sha256']) for x in image_rows]
    selection_sha = hashlib.sha256(json.dumps(selected_identity, sort_keys=True).encode()).hexdigest()
    audit = dict(schema='trajkv.round5.text-transfer64.v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                 images=64, questions=64, one_question_per_image=True, selection_salt=SALT,
                 selection='exclude audited source-ID/content-hash union; first64 fixed salted source-ID hashes; lowest numeric question ID per image; no answer-type or model-outcome filtering',
                 source_sha256=sha(args.source), official_registry_sha256=sha(official_marker),
                 official_repo=official['official_repo'], official_revision=official['official_revision'],
                 source_rows=len(source), official_source_images=len(all_ids), official_unique_contents=len(all_hashes),
                 excluded_source_ids=len(used_ids), excluded_content_hashes=len(used_hashes),
                 excluded_source_ids_in_official=len(used_ids & all_ids), excluded_contents_in_official=len(used_hashes & all_hashes),
                 eligible_source_images=len(eligible), overlap_source_ids=0, overlap_contents=0,
                 union_registry_inputs=registries, selection_sha256=selection_sha,
                 images_manifest=image_rows, annotations_used_for_selection=False,
                 model_outcomes_used_for_selection=False, answers_accessed=False,
                 staged_full5000_is_not_treated_as_evaluated=True,
                 scope='single-model zero-shot cross-domain transfer; fresh against enumerated prior Text source/content exposure union; not a multi-model or full-distribution benchmark',
                 method_rule='freeze candidates/rankings/factors on Doc only before Text inference; no Text calibration',
                 metric='official MMF/TextVQA ten-annotator soft accuracy; ANLS optional secondary',
                 transport='not performed by this local-only stage script; upload images plus answer-free runner, never reference annotations')
    summary = dict(status='DRY_RUN' if args.dry_run else 'STAGED',
                   official_images=len(all_ids), excluded_union_images=len(used_ids & all_ids),
                   eligible_images=len(eligible), selected_images=64,
                   image_bytes=sum(x['bytes'] for x in image_rows), selection_sha256=selection_sha,
                   references_read=False, files_written=False)
    if args.dry_run:
        print(json.dumps(summary))
        return
    if args.image_dir.exists():
        raise FileExistsError('Refusing to reuse partially staged image directory: '+str(args.image_dir))
    # Membership is fixed above before any answer annotations are read.
    if sha(args.source_references) != official['references_sha256']:
        raise ValueError('Pinned source annotations changed')
    references = {r['id']: r for r in read(args.source_references)}
    output, answers = [], []
    args.image_dir.mkdir(parents=True, exist_ok=False)
    for (_, original, digest, row), image in zip(selected, image_rows):
        target = Path(image['local'])
        if args.image_mode == 'hardlink':
            try:
                os.link(image['raw_source'], target)
                image['local_staging'] = 'hardlink original raw bytes'
            except OSError:
                shutil.copyfile(image['raw_source'], target)
                image['local_staging'] = 'raw byte copy after hardlink unsupported'
        else:
            shutil.copyfile(image['raw_source'], target)
            image['local_staging'] = 'raw byte copy'
        if sha(target) != digest:
            raise RuntimeError('Staged image differs from original bytes')
        sample = copy.deepcopy(row)
        session_id = 'textvqa_round5_'+str(row['source_question_id'])
        sample.update(id=session_id, session_id=session_id, images=[str(target)], remote_images=[image['remote']],
                      split='round5_source_disjoint_text_transfer64', historical_use='fresh against audited registry union',
                      selection_hash=image['selection_hash'])
        if any(key in sample for key in ('answer', 'answers', 'references')):
            raise RuntimeError('Answer annotation leaked into inference runner')
        ref = references[row['id']]
        if ref['image_id'] != sample['image_id'] or len(ref['answers']) != 10:
            raise ValueError('Selected annotations do not match the fixed source')
        output.append(sample)
        answers.append(dict(id=session_id, image_id=sample['image_id'], source_question_id=row['source_question_id'], answers=ref['answers']))
    write_new(args.output, output)
    write_new(args.references, answers)
    audit.update(answers_accessed=True, answers_accessed_after_membership_frozen=True,
                 samples_sha256=sha(args.output), references_sha256=sha(args.references),
                 source_references_sha256=sha(args.source_references))
    write_new(args.membership, audit)
    summary.update(references_read=True, files_written=True)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
