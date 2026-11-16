"""Reserve new Text128 identities; materialize labels only after method freeze."""
import argparse
import copy
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inspect_round6_unseen_pools import registry, content_ids, project_path
from round7_data_prepare import selected_json_references

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'audit/a2/round7_text_confirm128_membership_v1.json'
SALT = 'trajkv-round7-text-confirm128-source-v1:'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def reserve():
    if POOL.exists():
        raise FileExistsError('Text128 identities already frozen')
    _, used, _, hashes, inputs = registry()
    source_path = ROOT / 'configs/a2/textvqa_standard5000.json'
    groups = defaultdict(list)
    for row in json.loads(source_path.read_text(encoding='utf-8-sig')):
        source = str(row['source_original_image_id'])
        if source not in used and not (content_ids(row) & hashes):
            groups[source].append(row)
    ordered = sorted(groups, key=lambda x: (hashlib.sha256((SALT + x).encode()).hexdigest(), x))
    chosen, seen = [], set()
    for source in ordered:
        row = min(groups[source], key=lambda r: int(r['source_question_id']))
        digest = next(iter(content_ids(row)))
        if digest in seen:
            continue
        seen.add(digest)
        local = project_path(row['images'][0])
        if sha(local) != digest:
            raise ValueError('Text candidate image bytes changed')
        chosen.append(dict(source_image_id=source, source_question_id=row['source_question_id'],
            source_session_id=row['id'], sha256=digest, local=str(local), remote=row['remote_images'][0],
            bytes=local.stat().st_size, selection_hash=hashlib.sha256((SALT + source).encode()).hexdigest()))
        if len(chosen) == 128:
            break
    if len(chosen) != 128:
        raise ValueError('Insufficient clean source images')
    result = dict(schema='trajkv.round7.text-confirm128-reservation.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), cohort='text_confirm128_reserved',
        source=source_path.relative_to(ROOT).as_posix(), source_sha256=sha(source_path),
        registry_inputs=inputs, prior_reserved_source_images=len(used), eligible_source_images=len(groups),
        images=128, questions=128, selection_salt=SALT, source_and_content_overlap=0,
        selection_rule='exclude prior source/content union including R7 train32/select16; salted source hash order; lowest numeric question ID/image',
        images_manifest=chosen, answers_read=False, answers_uploaded=False,
        model_outcomes_used_for_selection=False, method_freeze_required_before_materialization=True,
        image_bytes=sum(r['bytes'] for r in chosen), source_sha256_script=sha(__file__))
    new(POOL, result)
    print(json.dumps(dict(status='IDENTITIES_RESERVED', sources=128, questions=128,
        bytes=result['image_bytes'], answers_read=False, GPU_used=False)))


def stage(method):
    if method is None or not json.loads(method.read_text(encoding='utf-8-sig')):
        raise ValueError('Nonempty caller method-freeze manifest required')
    method_sha = sha(method)
    pool = json.loads(POOL.read_text(encoding='utf-8'))
    source_path = ROOT / pool['source']
    if sha(source_path) != pool['source_sha256']:
        raise ValueError('Official Text runner source changed')
    source = {row['id']: row for row in json.loads(source_path.read_text(encoding='utf-8-sig'))}
    original_ids = {item['source_session_id'] for item in pool['images_manifest']}
    reference_path = ROOT / 'configs/data/textvqa_standard5000_references.json'
    selected_refs = selected_json_references(reference_path, original_ids)
    rows, refs = [], []
    for item in pool['images_manifest']:
        row = copy.deepcopy(source[item['source_session_id']])
        if str(row['source_original_image_id']) != item['source_image_id'] or row['hashes'] != [item['sha256']]:
            raise ValueError('Text128 frozen identity changed')
        if sha(project_path(row['images'][0])) != item['sha256']:
            raise ValueError('Text128 image bytes changed')
        row['id'] = row['session_id'] = 'textvqa_round7_confirm_' + str(item['source_question_id'])
        row['split'] = 'round7_fresh_source_confirm128'
        row['selection_hash'] = item['selection_hash']
        row['historical_result_reused'] = False
        rows.append(row)
        ref = copy.deepcopy(selected_refs[item['source_session_id']])
        ref['id'] = row['id']
        refs.append(ref)
    samples = ROOT / 'configs/a2/round7_v1/text_confirm128.json'
    references = ROOT / 'configs/data/round7_text_confirm128_references.json'
    new(samples, rows)
    new(references, refs)
    if sha(method) != method_sha:
        raise RuntimeError('Method freeze changed')
    new(ROOT / 'audit/a2/round7_text_confirm128_staged_v1.json', dict(
        schema='trajkv.round7.text-confirm128-staged.v1', images=128, questions=128,
        method_freeze=str(method), method_freeze_sha256=method_sha, reservation_sha256=sha(POOL),
        samples=samples.relative_to(ROOT).as_posix(), samples_sha256=sha(samples),
        references=references.relative_to(ROOT).as_posix(), references_sha256=sha(references),
        answers_uploaded=False, uploaded=False, GPU_used=False))
    print(json.dumps(dict(status='STAGED_LOCAL_ONLY', images=128, questions=128, answers_uploaded=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', action='store_true')
    p.add_argument('--method-freeze', type=Path)
    a = p.parse_args()
    stage(a.method_freeze) if a.stage else reserve()
