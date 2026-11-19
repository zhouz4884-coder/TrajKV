"""R8 source-only preflight and explicitly post-freeze QA materialization.

Default/preflight never decodes new confirmation answer values. Materialization
requires the exact SHA of an R8 freeze binding every requested membership.
All files are local, versioned, exclusive-create; no upload or inference.
"""
import argparse
import copy
from datetime import datetime, timezone
import itertools
from pathlib import Path

import pyarrow.parquet as pq

from round8_data_prepare import ROOT, read, sha, new
from round7_data_prepare import selected_json_references
from inspect_round6_unseen_pools import registry

POOLS = {
    'text_validation128': 'audit/a2/round8_text_validation128_membership_v1.json',
    'infovqa_validation64': 'audit/a2/round8_infovqa_val64_source_membership_v1.json',
    'doc_train64': 'audit/a2/round8_doc_train64_source_membership_v1.json',
}
SUFFIX = '\nAnswer the question using a single word or phrase.'


def load_pools(names):
    result = {name: read(ROOT / POOLS[name]) for name in names}
    for pool in result.values():
        entries = pool['images_manifest']
        assert len(entries) == pool['images']
        assert len({item['sha256'] for item in entries}) == len(entries)
        for item in entries:
            path = Path(item['local'])
            assert path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Reserved image changed'
        if 'source_shards' in pool:
            folder = 'data/docvqa_round7_train_pinned' if pool['dataset'] == 'docvqa' else 'data/infovqa_validation_round8_pinned'
            for item in pool['source_shards']:
                assert sha(ROOT / folder / item['file']) == item['sha256'], 'Pinned parquet changed'
        else:
            assert sha(ROOT / pool['source']) == pool['source_sha256'], 'Official Text source changed'
    return result


def preflight(pools):
    doc_used, text_used, doc_hashes, text_hashes, inputs = registry()
    historical_hashes = set(doc_hashes) | set(text_hashes)
    hashes = {name: {item['sha256'] for item in pool['images_manifest']} for name, pool in pools.items()}
    checks = []
    for name, pool in pools.items():
        source = {str(item['source_document_id'] if name == 'doc_train64' else item['source_image_id']) for item in pool['images_manifest']}
        prior_sources = doc_used if name == 'doc_train64' else text_used if name == 'text_validation128' else set()
        # Runtime configs can legitimately include the immutable cohort once
        # root stages it. Compare against the registry snapshotted at reserve
        # time below instead of classifying the cohort's own config as prior.
        inputs_at_reserve = pool['registry_inputs']
        for item in inputs_at_reserve:
            assert sha(ROOT / item['path']) == item['sha256'], 'Historical exclusion source changed'
        checks.append(dict(pool=name, images=len(source), questions=pool['questions'],
            membership_sha256=sha(ROOT / POOLS[name]), image_bytes=sum(item['bytes'] for item in pool['images_manifest']),
            fresh_scope=pool['fresh_scope'], current_registered_source_intersection=len(source & prior_sources),
            current_registered_content_intersection=len(hashes[name] & historical_hashes),
            reservation_claimed_overlap=pool['source_content_overlap'],
            registry_scope='Current registry may include this cohort after staging; original immutable exclusion inputs are SHA checked'))
    overlaps = {a + '__' + b: len(hashes[a] & hashes[b]) for a,b in itertools.combinations(hashes, 2)}
    assert not any(overlaps.values()), 'New confirmation pools share raw content'
    for split in ('train','select'):
        rows = read(ROOT / ('configs/a2/round8_v1/calibration_' + split + '32.json'))
        used = {value for row in rows for value in row['hashes']}
        for name in hashes:
            overlaps[name + '__cal_' + split] = len(hashes[name] & used)
    assert not any(overlaps.values()), 'Confirmation/calibration content overlap'
    return dict(schema='trajkv.round8.confirmation-source-preflight.v1', status='PASS',
        created_at_utc=datetime.now(timezone.utc).isoformat(), pools=checks,
        pairwise_raw_content_overlaps=overlaps, new_confirmation_answer_values_decoded=False,
        new_parquet_question_columns_decoded=False,
        existing_runner_question_text_loaded_by_identity_registry=True,
        registry_input_count=len(inputs), code_sha256=sha(__file__), GPU_used=False, uploaded=False)


def validate_freeze(path, expected, names):
    assert path and expected and sha(path) == expected, 'Exact caller R8 freeze SHA required'
    freeze = read(path)
    assert 'round8' in freeze.get('schema','').lower() and freeze.get('status','').startswith('FROZEN'), 'Expected frozen R8 method receipt'
    bindings = freeze.get('confirmation_membership_sha256', {})
    for name in names:
        assert bindings.get(POOLS[name]) == sha(ROOT / POOLS[name]), 'Method freeze must bind membership path and SHA: ' + POOLS[name]
    assert freeze.get('selected_variants') or freeze.get('candidate_configs') or freeze.get('files'), 'Freeze has no candidate identity bindings'
    return expected


def text_rows(pool):
    source = {row['id']: row for row in read(ROOT / pool['source'])}
    # This is the first new-confirmation answer-value decode, after freeze.
    bank = ROOT / 'configs/data/textvqa_standard5000_references.json'
    refs_by = selected_json_references(bank, {item['source_session_id'] for item in pool['images_manifest']})
    rows, refs = [], []
    for item in pool['images_manifest']:
        row = copy.deepcopy(source[item['source_session_id']])
        assert str(row['source_original_image_id']) == item['source_image_id'] and row['hashes'] == [item['sha256']]
        row['id'] = row['session_id'] = 'textvqa_round8_confirm_' + str(item['source_question_id'])
        row.update(source_id=row['image_id'], split='round8_fresh_official_validation128',
            fresh_scope=pool['fresh_scope'], historical_result_reused=False)
        ref = copy.deepcopy(refs_by[item['source_session_id']]); ref['id'] = row['id']
        assert len(ref['answers']) == 10 and ref['image_id'] == row['image_id']
        rows.append(row); refs.append(ref)
    return rows, refs, {bank.relative_to(ROOT).as_posix():sha(bank)}


def parquet_rows(name, pool, remote_root):
    doc = name == 'doc_train64'
    folder = ROOT / ('data/docvqa_round7_train_pinned' if doc else 'data/infovqa_validation_round8_pinned')
    selected = {str(qid) for item in pool['images_manifest'] for qid in item['source_question_ids']}
    questions = {}
    for item in pool['source_shards']:
        path = folder / item['file']; schema = pq.ParquetFile(path).schema_arrow
        field = pool['question_identity_field']
        import pyarrow as pa
        values = [int(value) for value in selected] if pa.types.is_integer(schema.field(field).type) else sorted(selected)
        cols = [field, 'question', 'answers'] + (['docId','ucsf_document_id'] if doc else ['image.path'])
        table = pq.read_table(path, columns=cols, filters=[(field,'in',values)])
        for row in table.to_pylist():
            qid = str(row[field]); assert qid in selected and qid not in questions
            questions[qid] = row
    assert set(questions) == selected, 'Missing selected official question identity'
    rows, refs = [], []
    for item in pool['images_manifest']:
        data = [questions[str(qid)] for qid in item['source_question_ids']]
        if doc:
            assert all(str(row['ucsf_document_id']) == item['source_document_id'] and str(row['docId']) == str(item['source_page_id']) for row in data)
        else:
            assert all(row['path'] == item['source_image_id'] for row in data)
        image_id = 'sha256:' + item['sha256']
        sid = ('docvqa_round8_train_confirm_' if doc else 'infovqa_round8_val_confirm_') + item['sha256']
        text = [row['question'] + SUFFIX for row in data]
        rows.append(dict(schema='trajkv.session.v1', dataset='docvqa_grouped' if doc else 'infovqa',
            id=sid, session_id=sid, image_id=image_id, source_id=image_id, source_image_id=image_id,
            source_document_id=item['source_document_id'] if doc else None,
            source_original_image_id=item['source_image_id'], source_page_id=item['source_page_id'],
            source_question_ids=item['source_question_ids'], images=[item['local']],
            remote_images=[remote_root.rstrip('/') + '/' + item['relative']], hashes=[item['sha256']],
            questions=text, turns=len(text), content=[{'type':'image'},{'type':'text','text':text[0]}],
            split='round8_' + name, original_dataset_split=pool['original_split'],
            fresh_scope=pool['fresh_scope'], protocol='derived_same_source_independent_qa_grouped',
            history_dependence='not_assumed_requires_qualification', historical_result_reused=False,
            answer_format_protocol='internvl_official_short_phrase_prompt_v1'))
        assert all(isinstance(row['answers'],list) and row['answers'] for row in data)
        refs.append(dict(id=sid, image_id=image_id, source_document_id=item['source_document_id'] if doc else None,
            references=[dict(turn=i+1,question_id=row[pool['question_identity_field']],answer=row['answers']) for i,row in enumerate(data)]))
    return rows, refs, {str((folder/item['file']).relative_to(ROOT)):item['sha256'] for item in pool['source_shards']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pools', nargs='+', choices=list(POOLS), default=list(POOLS))
    p.add_argument('--preflight-output', type=Path)
    p.add_argument('--materialize', action='store_true')
    p.add_argument('--method-freeze', type=Path)
    p.add_argument('--expected-freeze-sha256')
    p.add_argument('--remote-root', help='Existing project data-disk root, not a code-stage root')
    a = p.parse_args(); pools = load_pools(a.pools)
    check = preflight(pools)
    if not a.materialize:
        if a.preflight_output:
            new(a.preflight_output,check)
        print(__import__('json').dumps(dict(status='SOURCE_ONLY_PASS',pools=a.pools,answers_decoded=False,GPU_used=False)))
        return
    method_sha = validate_freeze(a.method_freeze,a.expected_freeze_sha256,a.pools)
    assert a.remote_root and a.remote_root.startswith('/'), 'Explicit project data-disk remote root required'
    targets = [(name,ROOT/f'configs/a2/round8_v1/{name}.json',ROOT/f'configs/data/round8_{name}_references_v1.json',ROOT/f'audit/a2/round8_{name}_materialized_v1.json') for name in a.pools]
    assert not any(path.exists() for _,*paths in targets for path in paths), 'Output exists; preserve it and create an explicit versioned repair'
    for name,samples,references,receipt in targets:
        pool = pools[name]
        rows,refs,inputs = text_rows(pool) if name == 'text_validation128' else parquet_rows(name,pool,a.remote_root)
        assert len(rows) == pool['images'] and sum(row['turns'] for row in rows) == pool['questions']
        assert sha(a.method_freeze) == method_sha, 'Method freeze changed during materialization'
        new(samples,rows); new(references,refs)
        new(receipt,dict(schema='trajkv.round8.post-freeze-materialization.v1',status='STAGED_LOCAL_ONLY',
            method_freeze=str(a.method_freeze),method_freeze_sha256=method_sha,membership=POOLS[name],
            membership_sha256=sha(ROOT/POOLS[name]),samples=str(samples.relative_to(ROOT)),samples_sha256=sha(samples),
            references=str(references.relative_to(ROOT)),references_sha256=sha(references),
            input_sha256=inputs,images=len(rows),questions=sum(row['turns'] for row in rows),fresh_scope=pool['fresh_scope'],
            created_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),
            new_answer_values_decoded_only_after_method_freeze=True,answers_uploaded=False,uploaded=False,GPU_used=False))
        print(__import__('json').dumps(dict(status='STAGED_LOCAL_ONLY',pool=name,images=len(rows),questions=pool['questions'],answers_uploaded=False)))


if __name__ == '__main__':
    main()
