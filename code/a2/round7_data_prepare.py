"""Freeze source-only R7 development selections, then isolate selected labels.

No inference records, scores, or future confirmation annotations are decoded.
Only new R7 files are written. GPU execution and transport belong to the root.
"""
import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inspect_round6_unseen_pools import registry, content_ids, project_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'configs/a2/round7_v1'
SALT_DOC = 'trajkv-round7-doc-cal32-source-split-v1:'
SALT_TEXT = 'trajkv-round7-text-cal48-source-split-v1:'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable(salt, value):
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()


def new(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in value) if jsonl else json.dumps(value, indent=2, ensure_ascii=False)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


def selected_json_references(path, ids):
    """Search identity strings; decode exactly selected flat reference objects.

    The full reference file is never passed to json.loads. Future confirmation
    reference objects are skipped as raw bytes, rather than decoded into rows.
    """
    text = Path(path).read_text(encoding='utf-8-sig')
    decoder = json.JSONDecoder()
    result = {}
    for sid in sorted(ids):
        pattern = r'"id"\s*:\s*' + re.escape(json.dumps(sid))
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError('Expected exactly one selected reference identity: ' + sid)
        start = text.rfind('{', 0, matches[0].start())
        value, _ = decoder.raw_decode(text, start)
        if value['id'] != sid:
            raise ValueError('Selected reference framing differs')
        result[sid] = value
    return result


def sample_copy(row, split):
    result = copy.deepcopy(row)
    result['split'] = split
    result['round7_selection_used_answers'] = False
    result['round7_selection_used_model_outcomes'] = False
    if any(key in result for key in ('answer', 'answers', 'references')):
        raise ValueError('Answers in inference runner')
    for image, digest in zip(result['images'], result['hashes']):
        if sha(project_path(image)) != digest:
            raise ValueError('Selected image content changed')
    return result


def prepare():
    audit_path = ROOT / 'audit/a2/round7_development_selection_v1.json'
    membership = ROOT / 'audit/a2/round7_development_membership_v1.json'
    if audit_path.exists() or membership.exists():
        raise FileExistsError('R7 selection is already frozen; use a new version for repairs')
    du, tu, dh, th, inputs = registry()
    doc_source = ROOT / 'configs/a2/next_cal32_independent_questions.json'
    doc = read(doc_source)
    doc_by_source = defaultdict(list)
    for row in doc:
        doc_by_source[str(row['source_document_id'])].append(row)
    names = sorted(doc_by_source, key=lambda x: (stable(SALT_DOC, x), x))
    if len(names) != 32 or len(doc) != 83:
        raise ValueError('Expected frozen 32-source/83-question independent Doc calibration')
    doc_groups = {'doc_train16': names[:16], 'doc_select16': names[16:]}
    selected = {}
    for label, sources in doc_groups.items():
        selected[label + '_independent'] = [sample_copy(row, 'round7_' + label + '_independent')
            for source in sources for row in sorted(doc_by_source[source], key=lambda x: x['calibration_question_index'])]

    text_source = ROOT / 'configs/a2/textvqa_standard5000.json'
    grouped = defaultdict(list)
    for row in read(text_source):
        source = str(row['source_original_image_id'])
        if source not in tu and not (content_ids(row) & th):
            grouped[source].append(row)
    text_names = sorted(grouped, key=lambda x: (stable(SALT_TEXT, x), x))
    if len(text_names) < 48:
        raise ValueError('Too few Text source-disjoint calibration/development images')
    text_groups = {'text_train32': text_names[:32], 'text_select16': text_names[32:48]}
    for label, sources in text_groups.items():
        rows = [sample_copy(row, 'round7_' + label + '_independent')
            for source in sources for row in sorted(grouped[source], key=lambda x: int(x['source_question_id']))[:3]]
        for row in rows:
            row['round7_original_session_id'] = row['id']
            row['id'] = row['session_id'] = 'r7_' + label + '_' + str(row['source_question_id'])
            row['questions'] = row['questions'][:1]
            row['turns'] = 1
        selected[label + '_independent'] = rows
    selected['text_train16_independent'] = [copy.deepcopy(row) for row in selected['text_train32_independent']
        if str(row['source_original_image_id']) in set(text_names[:16])]
    for row in selected['text_train16_independent']:
        row['split'] = 'round7_text_train16_independent_nested_in_train32'

    # First-question rows are distinct from independent multi-question sampling.
    for label in ['doc_train16', 'doc_select16', 'text_train32', 'text_train16', 'text_select16']:
        rows = selected[label + '_independent']
        identity = 'source_document_id' if label.startswith('doc') else 'source_original_image_id'
        seen = set()
        first = []
        for row in rows:
            source = str(row[identity])
            if source not in seen:
                seen.add(source)
                first.append(copy.deepcopy(row))
        selected[label + '_first_questions'] = first

    # Same-source continuous and long-session diagnostics: fixed held-out Doc
    # sources, with question reuse disclosed rather than counted as new data.
    for label, count, turns in [('doc_select16_continuous', 16, None), ('doc_stress8_turn12', 8, 12)]:
        rows = []
        for source in names[16:16 + count]:
            original = sorted(doc_by_source[source], key=lambda x: x['calibration_question_index'])
            row = copy.deepcopy(original[0])
            indices = list(range(len(original))) if turns is None else [i % len(original) for i in range(turns)]
            row['id'] = row['session_id'] = 'r7_' + label + '_' + str(row['source_page_id'])
            row['split'] = 'round7_' + label
            row['questions'] = [original[i]['questions'][0] for i in indices]
            row['turns'] = len(indices)
            row['round7_independent_reference_ids'] = [original[i]['id'] for i in indices]
            row['round7_question_reuse'] = turns is not None
            row['round7_diagnostic_scope'] = 'same held-out development sources; not fresh confirmation'
            row['content'] = [{'type': 'image'}, {'type': 'text', 'text': row['questions'][0]}]
            rows.append(row)
        selected[label] = rows

    # Selection freeze is durable before any label materialization.
    selection = dict(schema='trajkv.round7.development-selection.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), annotation_files_read=False,
        inference_records_read=False, model_outcomes_used_for_selection=False,
        prior_registry_inputs=inputs, reserved_doc_sources=len(du), reserved_text_sources=len(tu),
        original_doc_validation_source_pool_exhausted=len(du) == 938,
        doc_rule='hash split existing cal32 into train16 and selection16; preserve all 83 independent questions',
        text_rule='exclude source/content union; fixed salted hash order; train32 then selection16; max3 numeric-ID questions per source',
        doc_salt=SALT_DOC, text_salt=SALT_TEXT, doc_sources=doc_groups, text_sources=text_groups,
        text_eligible_sources=len(text_names),
        source_inputs={str(p.relative_to(ROOT)): sha(p) for p in (doc_source, text_source)},
        selected={label: dict(sources=len({str(r.get('source_document_id', r.get('source_original_image_id'))) for r in rows}),
            rows=len(rows), questions=sum(r['turns'] for r in rows),
            ids=[r['id'] for r in rows], images=sorted({r['image_id'] for r in rows})) for label, rows in selected.items()},
        source_sha256=sha(__file__))
    new(audit_path, selection)
    selection_sha = sha(audit_path)
    for label, rows in selected.items():
        new(OUT / (label + '.json'), rows)

    # Decode only source-calibration Doc references, never next confirmation GT.
    doc_ref_source = ROOT / 'configs/data/docvqa_next_references.jsonl'
    cal_parent_ids = {r['id'].rsplit('_cal_q', 1)[0] for r in doc}
    doc_parent = {}
    for line in doc_ref_source.read_text(encoding='utf-8-sig').splitlines():
        identity = re.search(r'"id"\s*:\s*"([^"]+)"', line)
        if identity and identity.group(1) in cal_parent_ids:
            value = json.loads(line)
            doc_parent[value['id']] = value
    if set(doc_parent) != cal_parent_ids:
        raise ValueError('Selected Doc calibration reference identities missing')
    doc_refs = {}
    for row in doc:
        parent = doc_parent[row['id'].rsplit('_cal_q', 1)[0]]
        ref = copy.deepcopy(parent['references'][int(row['calibration_question_index']) - 1])
        ref['turn'] = 1
        doc_refs[row['id']] = dict(id=row['id'], source_document_id=row['source_document_id'],
            answers=copy.deepcopy(ref['answer']), references=[ref])
    text_ref_source = ROOT / 'configs/data/textvqa_standard5000_references.json'
    text_original_ids = {r['round7_original_session_id'] for label, rows in selected.items() if label.startswith('text_') for r in rows}
    text_refs = selected_json_references(text_ref_source, text_original_ids)
    manifest = {}
    for label, rows in selected.items():
        refs = []
        for row in rows:
            if label.startswith('text_'):
                ref = copy.deepcopy(text_refs[row['round7_original_session_id']])
                ref['id'] = row['id']
                ref['source_original_image_id'] = row['source_original_image_id']
            elif 'round7_independent_reference_ids' in row:
                answers = [copy.deepcopy(doc_refs[sid]['references'][0]) for sid in row['round7_independent_reference_ids']]
                for i, answer in enumerate(answers):
                    answer['turn'] = i + 1
                ref = dict(id=row['id'], source_document_id=row['source_document_id'], references=answers)
            else:
                ref = copy.deepcopy(doc_refs[row['id']])
            refs.append(ref)
        ref_path = ROOT / 'configs/data' / ('round7_' + label + '_references.' + ('json' if label.startswith('text_') else 'jsonl'))
        new(ref_path, refs, jsonl=not label.startswith('text_'))
        sample_path = OUT / (label + '.json')
        manifest[label] = dict(samples=sample_path.relative_to(ROOT).as_posix(), samples_sha256=sha(sample_path),
            references=ref_path.relative_to(ROOT).as_posix(), references_sha256=sha(ref_path),
            rows=len(rows), source_count=selection['selected'][label]['sources'], questions=sum(r['turns'] for r in rows))
    if sha(audit_path) != selection_sha:
        raise RuntimeError('Source selection freeze changed')
    new(membership, dict(schema='trajkv.round7.development-membership.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), selection= str(audit_path.relative_to(ROOT)), selection_sha256=selection_sha,
        datasets=manifest, references_uploaded=False, GPU_used=False,
        selected_calibration_references_only=True, future_confirmation_references_decoded=False,
        source_and_content_overlap_train_selection=0, source_and_content_overlap_text_prior_union=0,
        note='Doc selection16 are held out from this new fit, but belong to historically exposed cal32; Text selections are new against enumerated project exposure/reservation union. Neither is blind confirmation.'))
    print(json.dumps(dict(status='STAGED_LOCAL_ONLY', datasets=manifest, GPU_used=False, uploaded=False)))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
