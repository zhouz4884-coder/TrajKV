"""Identity-only R8 fresh-pool audit. No proposed confirmation GT is decoded."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from inspect_round6_unseen_pools import registry, content_ids, project_path
from round8_data_prepare import ROOT, read, sha, new

SALT_TEXT = 'trajkv-round8-official-text-val128-source-v1:'


def main():
    output = ROOT / 'audit/a2/round8_fresh_pool_identity_audit_v1.json'
    assert not output.exists(), 'Use a new audit version'
    doc_used, text_used, doc_hashes, text_hashes, inputs = registry()
    # Full official Doc validation identity support; no images/questions/labels.
    official_doc = set()
    for path in sorted((ROOT / 'data/docvqa_validation_pinned').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(columns=['ucsf_document_id']):
            official_doc.update(str(row['ucsf_document_id']) for row in batch.to_pylist())
    doc_remaining = official_doc - doc_used
    # The existing extra shards are official TRAIN. List identity availability
    # separately; do not rename them validation or select by answer/outcome.
    train_sources = set()
    for path in sorted((ROOT / 'data/docvqa_round7_train_pinned').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(columns=['ucsf_document_id']):
            train_sources.update(str(row['ucsf_document_id']) for row in batch.to_pylist())
    source_path = ROOT / 'configs/a2/textvqa_standard5000.json'
    official_text = read(ROOT / 'audit/a2/textvqa_standard5000_membership.json')
    assert sha(source_path) == official_text['samples_sha256']
    grouped = defaultdict(list)
    for row in read(source_path):
        source = str(row['source_original_image_id'])
        if source not in text_used and not (content_ids(row) & text_hashes):
            grouped[source].append(row)
    ordered = sorted(grouped, key=lambda source: (hashlib.sha256((SALT_TEXT + source).encode()).hexdigest(), source))
    chosen, seen = [], set()
    for source in ordered:
        row = min(grouped[source], key=lambda item: int(item['source_question_id']))
        digest = next(iter(content_ids(row)))
        if digest in seen:
            continue
        seen.add(digest)
        path = project_path(row['images'][0])
        assert sha(path) == digest
        chosen.append(dict(source_image_id=source, source_question_id=row['source_question_id'],
            source_session_id=row['id'], sha256=digest, local=str(path), remote=row['remote_images'][0],
            bytes=path.stat().st_size, selection_hash=hashlib.sha256((SALT_TEXT + source).encode()).hexdigest()))
        if len(chosen) == 128:
            break
    assert len(chosen) == 128
    reserve_path = ROOT / 'audit/a2/round8_text_validation128_membership_v1.json'
    reserve = dict(schema='trajkv.round8.text-official-validation128-reservation.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), source=source_path.relative_to(ROOT).as_posix(),
        source_sha256=sha(source_path), official_registry_sha256=sha(ROOT / 'audit/a2/textvqa_standard5000_membership.json'),
        source_split='official validation', fresh_scope='project source/content disjoint against enumerated exposure plus reservations; backbone pretraining exposure not established',
        images=128, questions=128, images_manifest=chosen, selection_salt=SALT_TEXT,
        selection_rule='exclude all registered source/content; fixed salted source hash; lowest numeric question ID/source',
        registry_inputs=inputs, eligible_source_images=len(grouped), prior_registered_source_images=len(text_used),
        source_content_overlap=0, answers_read=False, model_outcomes_read=False,
        method_freeze_required_before_materialization=True, uploaded=False)
    new(reserve_path, reserve)
    result = dict(schema='trajkv.round8.fresh-pool-identity-audit.v1', status='PASS',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        official_doc_validation=dict(original_source_count=len(official_doc), registered_excluded=len(official_doc & doc_used),
            remaining_source_count=len(doc_remaining), remaining_source_ids=sorted(doc_remaining),
            scope='The pinned official validation split is exhausted at original-document source level; unused pages do not create new sources.'),
        official_doc_train_alternative=dict(local_source_count=len(train_sources),
            remaining_source_count=len(train_sources - doc_used),
            remaining_source_ids=sorted(train_sources - doc_used),
            fresh_scope='potential project-fresh official TRAIN source holdout only; content hashes still require candidate audit; not validation/test/backbone-unseen',
            answers_read=False, materialized=False),
        official_text_validation=dict(eligible_source_images=len(grouped), reserved128=reserve_path.relative_to(ROOT).as_posix(),
            reserved128_sha256=sha(reserve_path), bytes=sum(row['bytes'] for row in chosen), answers_read=False),
        registry_inputs=inputs, code_sha256=sha(__file__), source_only=True, GPU_used=False,
        answer_annotation_files_opened=False, inference_record_files_opened=False, uploaded=False)
    new(output, result)
    print(json.dumps(dict(status='PASS', official_doc_validation_fresh_sources=len(doc_remaining),
        local_official_doc_train_potential_fresh_sources=len(train_sources - doc_used),
        official_text_validation_eligible_sources=len(grouped), reserved_text_sources=128,
        confirmation_answers_opened=False, GPU_used=False, uploaded=False)))


if __name__ == '__main__':
    main()
