"""Read-only post-hoc paired depth errors; never a confirmation refit."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PAIR, INC = 'r6_depth_v_pair', 'r6_depth_incumbent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    n = len(rows)
    if not n:
        raise ValueError('Empty descriptive stratum')
    delta = sum(row['delta'] for row in rows)
    return dict(questions=n,
        improved=sum(row['delta'] > 1e-12 for row in rows),
        harmed=sum(row['delta'] < -1e-12 for row in rows),
        same_score=sum(abs(row['delta']) <= 1e-12 for row in rows),
        pair_anls=100 * sum(row['pair_score'] for row in rows) / n,
        incumbent_anls=100 * sum(row['incumbent_score'] for row in rows) / n,
        delta_pp=100 * delta / n, total_score_delta=delta,
        total_positive_score_delta=sum(max(row['delta'], 0.) for row in rows),
        total_negative_score_delta=sum(min(row['delta'], 0.) for row in rows))


def compare(scores, cohort):
    indexed = {variant: {} for variant in (PAIR, INC)}
    for row in read_csv(scores):
        if row['variant'] not in indexed:
            continue
        key = (row['session_id'], int(row['turn']))
        if key in indexed[row['variant']]:
            raise ValueError('Duplicate paired variant/question: ' + str(key))
        indexed[row['variant']][key] = row
    if not indexed[PAIR] or set(indexed[PAIR]) != set(indexed[INC]):
        raise ValueError('Incomplete pair/incumbent question matching')
    questions = []
    for key, pair in indexed[PAIR].items():
        inc = indexed[INC][key]
        for name in ('reference', 'image_id', 'source_document_id', 'visual_tokens', 'numeric'):
            if pair[name] != inc[name]:
                raise ValueError('Matched input metadata differs: ' + name)
        refs = json.loads(pair['reference'])
        numeric = any(any(char.isdigit() for char in target) for target in refs)
        if numeric != (pair['numeric'] == 'True'):
            raise ValueError('Frozen numeric category disagrees with reference-digit definition')
        pair_score, inc_score = float(pair['score']), float(inc['score'])
        questions.append(dict(cohort=cohort, analysis_role=pair['analysis_role'],
            session_id=key[0], turn=key[1], source_document_id=pair['source_document_id'],
            image_id=pair['image_id'], visual_tokens=int(pair['visual_tokens']),
            reference_contains_digit=numeric, pair_score=pair_score, incumbent_score=inc_score,
            delta=pair_score - inc_score, pair_prediction=pair['prediction'],
            incumbent_prediction=inc['prediction'], reference=pair['reference']))
    strata = []
    for field in ('visual_tokens', 'reference_contains_digit'):
        for value in sorted({row[field] for row in questions}):
            selected = [row for row in questions if row[field] == value]
            strata.append(dict(cohort=cohort, stratum=field, value=value, **summary(selected)))
    docs, images = defaultdict(list), defaultdict(list)
    for row in questions:
        docs[row['source_document_id']].append(row)
        images[row['image_id']].append(row)
    by_source = []
    for source_id, rows in sorted(docs.items()):
        by_source.append(dict(cohort=cohort, source_document_id=source_id,
            image_count=len({row['image_id'] for row in rows}),
            visual_tokens='|'.join(str(n) for n in sorted({row['visual_tokens'] for row in rows})),
            **summary(rows)))
    source_summary = dict(sources=len(by_source), unique_images=len(images),
        improved_sources=sum(row['total_score_delta'] > 1e-12 for row in by_source),
        harmed_sources=sum(row['total_score_delta'] < -1e-12 for row in by_source),
        same_score_sources=sum(abs(row['total_score_delta']) <= 1e-12 for row in by_source))
    if not math.isclose(sum(row['total_score_delta'] for row in by_source),
                        sum(row['delta'] for row in questions), abs_tol=1e-12):
        raise RuntimeError('Source decomposition failed to preserve paired total')
    return dict(cohort=cohort, **summary(questions), **source_summary), questions, strata, by_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-scores', type=Path, required=True)
    parser.add_argument('--fresh-scores', type=Path, required=True)
    parser.add_argument('--dev-mechanism', type=Path, required=True)
    parser.add_argument('--fresh-mechanism', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    totals, questions, strata, sources = [], [], [], []
    for cohort, path in (('development64', args.dev_scores), ('confirmation54', args.fresh_scores)):
        total, q, st, source = compare(path, cohort)
        totals.append(total)
        questions.extend(q)
        strata.extend(st)
        sources.extend(source)
    mechanism = []
    for cohort, path in (('development64', args.dev_mechanism), ('confirmation54', args.fresh_mechanism)):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        for row in data['per_axis_summary']:
            if row['variant'] == PAIR:
                mechanism.append(dict(cohort=cohort, **row))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('cohort_summary', totals), ('paired_questions', questions),
                       ('posthoc_strata', strata), ('by_source_document', sources), ('source_distortion', mechanism)):
        write_csv(args.output / (name + '.csv'), rows)
    report = dict(status='PASS', primary_pair_preserved=True,
        selected_primary=PAIR, comparator=INC, task_method_modified=False,
        cohort_summaries=totals, posthoc_strata=strata, paired_source_distortion=mechanism,
        scope='Read-only paired question/source/N/reference-digit descriptions. No new significance tests, subgroup selection, causal mechanism or confirmation refit.',
        numeric_definition='Frozen score CSV numeric field: any reference string contains any Unicode digit. Dates/phones/mixed letter-digit answers may qualify; not pure arithmetic answers.',
        same_score_definition='Absolute paired ANLS delta <=1e-12; predictions can differ despite equal scores.',
        source_weighting='Published overall delta remains question-weighted; source rows show within-source means and exact score-delta decomposition.',
        inputs={str(path): sha(path) for path in
            (args.dev_scores, args.fresh_scores, args.dev_mechanism, args.fresh_mechanism)},
        source_sha256=sha(__file__))
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status='PASS', questions=len(questions),
        sources=len(sources), primary_pair_preserved=True)), encoding='utf-8')
    print(json.dumps(dict(status='PASS', questions=len(questions), source_rows=len(sources))), flush=True)


if __name__ == '__main__':
    main()
