"""Prepare versioned R8 calibration/dev inputs from preserved exposed cohorts.

Main cal32 = 16 Doc + 16 Text sources; selection32 = 16 + 16 different sources.
No confirmation labels are opened. No GPU, SSH, transport or publishing occurs.
"""
import copy
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from textvqa_metric import TextVQAAccuracyEvaluator
from inspect_round6_unseen_pools import project_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'configs/a2/round8_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def rows_and_refs(member, label, inputs):
    spec = member['datasets'][label]
    sample, reference = ROOT / spec['samples'], ROOT / spec['references']
    assert sha(sample) == spec['samples_sha256'] and sha(reference) == spec['references_sha256']
    inputs[sample.relative_to(ROOT).as_posix()] = sha(sample)
    inputs[reference.relative_to(ROOT).as_posix()] = sha(reference)
    refs = ([json.loads(line) for line in reference.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            if reference.suffix == '.jsonl' else read(reference))
    return read(sample), {row['id']: row for row in refs}


def grouped(member, label, role, inputs):
    source_rows, references = rows_and_refs(member, label, inputs)
    groups = defaultdict(list)
    for row in source_rows:
        assert row['turns'] == 1 and not any(key in row for key in ('answers', 'answer', 'references'))
        groups[row['image_id']].append(row)
    metric = TextVQAAccuracyEvaluator()
    output, refs, target_changes = [], [], []
    for image_id, rows in groups.items():
        doc = rows[0]['dataset'] == 'docvqa_grouped'
        rows.sort(key=lambda row: int(row['calibration_question_index'] if doc else row['source_question_id']))
        row = copy.deepcopy(rows[0])
        row['id'] = row['session_id'] = 'r8_cal_' + ('doc_' if doc else 'text_') + image_id.removeprefix('sha256:')
        row['source_id'] = image_id
        row['split'] = 'round8_' + role + '_historically_exposed_sources'
        row['questions'] = [item['questions'][0] for item in rows]
        row['turns'] = len(rows)
        row['content'] = [{'type': 'image'}, {'type': 'text', 'text': row['questions'][0]}]
        row['calibration_only'] = True
        row['source_question_ids'] = []
        row['independent_question_input_ids'] = [item['id'] for item in rows]
        row['protocol'] = 'same_source_independent_questions_grouped_for_calibration'
        row['history_dependence'] = 'not_assumed; source prefix may be collected once before the first question'
        for path, digest in zip(row['images'], row['hashes']):
            assert sha(project_path(path)) == digest
        question_refs = []
        for i, item in enumerate(rows):
            ref = references[item['id']]
            if doc:
                original = ref['references'][0]
                answers, qid = original['answer'], original['question_id']
            else:
                answers, qid = list(ref['answers']), ref['source_question_id']
                assert len(answers) == 10
                votes = Counter(metric.answer_processor(answer) for answer in answers)
                normalized = min(key for key, count in votes.items() if count == max(votes.values()))
                index = next(j for j, answer in enumerate(answers) if metric.answer_processor(answer) == normalized)
                chosen = answers.pop(index); answers.insert(0, chosen)
                assert Counter(answers) == Counter(ref['answers'])
                if index:
                    target_changes.append(dict(id=item['id'], old_first=ref['answers'][0], new_first=chosen, normalized=normalized))
            row['source_question_ids'].append(qid)
            question_refs.append(dict(turn=i + 1, question_id=qid, answer=answers, answers=answers,
                                      original_independent_id=item['id']))
        output.append(row)
        refs.append(dict(id=row['id'], source_id=image_id, image_id=image_id, dataset=row['dataset'],
            source_document_id=row.get('source_document_id'), source_original_image_id=row.get('source_original_image_id'),
            answers=question_refs[0]['answers'], references=question_refs,
            target_policy='Doc original references; Text normalized plurality first, lexicographic tie; original 10-answer multiset preserved; calibration only'))
    return output, refs, target_changes


def main():
    membership_path = ROOT / 'audit/a2/round7_development_membership_v1.json'
    member = read(membership_path)
    lock = OUT / 'calibration_source_splits_v1.json'
    assert not lock.exists(), 'R8 source split already prepared; use a new version for repairs'
    inputs = {membership_path.relative_to(ROOT).as_posix(): sha(membership_path)}
    groups, refs, changes = {}, {}, []
    for label, role in [('doc_train16_independent', 'train'), ('text_train16_independent', 'train'),
                        ('doc_select16_independent', 'select'), ('text_select16_independent', 'select'),
                        ('text_train32_independent', 'train_expansion')]:
        groups[label], refs[label], selected = grouped(member, label, role, inputs)
        changes.extend(selected)
    train = groups['doc_train16_independent'] + groups['text_train16_independent']
    select = groups['doc_select16_independent'] + groups['text_select16_independent']
    train_refs = refs['doc_train16_independent'] + refs['text_train16_independent']
    select_refs = refs['doc_select16_independent'] + refs['text_select16_independent']
    expansion = groups['doc_train16_independent'] + groups['text_train32_independent']
    assert len(train) == len(select) == 32 and len(expansion) == 48
    assert not ({row['source_id'] for row in train} & {row['source_id'] for row in select})
    assert not ({row.get('source_document_id') for row in train if row.get('source_document_id')}
                & {row.get('source_document_id') for row in select if row.get('source_document_id')})
    files = {}
    for name, data in [('calibration_train32', train), ('calibration_select32', select), ('calibration_train48_text_expansion', expansion)]:
        path = OUT / (name + '.json'); new(path, data)
        files[name] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), sources=len(data), questions=sum(row['turns'] for row in data))
    for name, data in [('calibration_train32', train_refs), ('calibration_select32', select_refs)]:
        path = ROOT / 'configs/data' / ('round8_' + name + '_references_v1.json'); new(path, data)
        files[name + '_references'] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    split = dict(schema='trajkv.round8.source-splits.v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_id_field='image_id', train=[row['source_id'] for row in train], select=[row['source_id'] for row in select],
        train_samples=files['calibration_train32']['path'], select_samples=files['calibration_select32']['path'],
        train_references=files['calibration_train32_references']['path'], select_references=files['calibration_select32_references']['path'],
        sources={row['source_id']: dict(dataset=row['dataset'], session_id=row['id'],
            source_document_id=row.get('source_document_id'), source_original_image_id=row.get('source_original_image_id'),
            question_ids=row['source_question_ids'], questions=row['turns']) for row in train + select},
        files=files, input_sha256=inputs,
        source_content_train_select_overlap=0, domain_weighting='equal datasets; equal original sources within dataset; do not let token count substitute for source weight',
        exposure_scope='All calibration/selection sources were already prepared or observed in R7; source-disjoint only for this R8 fit, not pristine/blind',
        confirmation_answers_opened=False, GPU_used=False, uploaded=False)
    new(lock, split)
    # Pilot source selection uses fixed identity hashes, not new outcomes.
    pilots = {}
    for kind, source_name, refs_name in [('doc', 'configs/a2/docvqa_next_dev64.json', 'configs/data/docvqa_next_references.jsonl'),
                                          ('text', 'configs/a2/round5_v1/text_transfer64.json', 'configs/data/round5_text64_references.json')]:
        source = ROOT / source_name; rows = read(source)
        assert len(rows) == 64
        rows.sort(key=lambda row: hashlib.sha256(('trajkv-round8-old-dev-pilot-v1:' + row['image_id']).encode()).hexdigest())
        for count in (16, 64):
            selected = copy.deepcopy(rows[:count])
            for row in selected:
                row['round8_analysis_role'] = 'historically_observed_development_not_blind'
                row['source_id'] = row['image_id']
            path = OUT / (kind + '_pilot' + str(count) + '.json'); new(path, selected)
            pilots[kind + str(count)] = dict(samples=path.relative_to(ROOT).as_posix(), sha256=sha(path), references=refs_name,
                input_samples_sha256=sha(source), sources=count, questions=sum(row['turns'] for row in selected))
    images = {}
    for row in train + select:
        for local, remote, digest in zip(row['images'], row['remote_images'], row['hashes']):
            images[digest] = dict(local=str(project_path(local)), remote=remote, sha256=digest,
                role='existing immutable data-disk asset; root verifies/reuses, this helper does not upload')
    audit = dict(schema='trajkv.round8.data-preparation.v1', status='PASS', split=str(lock.relative_to(ROOT)), split_sha256=sha(lock),
        calibration=files, pilots=pilots, input_sha256=inputs, text_plurality_target_changes=changes,
        text_answer_multisets_preserved=True, official_evaluation_references_modified=False,
        images=list(images.values()), exposure_scope=split['exposure_scope'],
        future_confirmation_answers_opened=False, source_sha256=sha(__file__), GPU_used=False, uploaded=False)
    new(ROOT / 'audit/a2/round8_data_preparation_v1.json', audit)
    print(json.dumps(dict(status='PASS', split=str(lock), train_sources=32, train_questions=sum(row['turns'] for row in train),
        select_sources=32, select_questions=sum(row['turns'] for row in select), pilot_scope='exposed development', GPU_used=False, uploaded=False)))


if __name__ == '__main__':
    main()
