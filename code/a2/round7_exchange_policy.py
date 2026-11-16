"""Pure policy and source-balanced statistics for Round7 packed exchanges.

Only first sign stages are exchanged. A complete 224-unit ranking has exactly
the same uint16 ranking charge as the frozen Depth incumbent's unit ranking.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(value):
    if isinstance(value, dict):
        value = [value['axis'], *value.get('unit', [value.get('layer'), value.get('head')])]
    result = (str(value[0]).lower(), int(value[1]), int(value[2]))
    if result[0] not in ('k', 'v'):
        raise ValueError('Exchange unit axis must be k or v')
    return result


def ranking_units(settings, layers, heads):
    ranked = [unit(value) for value in settings['selected_units']]
    universe = {(axis, layer, head) for axis in ('k', 'v')
                for layer in range(layers) for head in range(heads)}
    if len(ranked) != len(universe) or set(ranked) != universe:
        raise ValueError('A complete, unique 2*layers*heads incumbent ranking is required')
    return ranked


def stages(ranking):
    return [[*item, 0] for item in ranking]


def apply_swaps(ranking, pairs):
    result = list(ranking)
    used = set()
    for pair in pairs:
        remove, add = unit(pair['remove']), unit(pair['add'])
        if remove == add or remove in used or add in used:
            raise ValueError('Selected simultaneous exchanges must have disjoint endpoints')
        left, right = result.index(remove), result.index(add)
        result[left], result[right] = result[right], result[left]
        used.update((remove, add))
    if set(result) != set(ranking):
        raise AssertionError('Exchange changed the ranking universe')
    return result


def candidates(config, ranking, active_sets, gradient):
    active = set.intersection(*active_sets)
    inactive = set(ranking).difference(set.union(*active_sets))
    requested = config.get('candidate_pairs')
    if requested is not None:
        result = [{'id': str(value.get('id', 'swap_%02d' % i)),
                   'remove': list(unit(value['remove'])), 'add': list(unit(value['add']))}
                  for i, value in enumerate(requested)]
    else:
        scored = gradient['joint_KV_ranks']
        score = {unit(value): float(value['score_head_scalar']) for value in scored}
        if set(ranking).difference(score):
            raise ValueError('Frozen gradient ranking does not cover the full unit universe')
        removals = sorted(active, key=lambda value: (score[value], value))
        additions = sorted(inactive, key=lambda value: (-score[value], value))
        count = min(int(config.get('candidate_count', 8)), len(removals), len(additions))
        result = [{'id': 'swap_%02d' % i, 'remove': list(removals[i]), 'add': list(additions[i]),
                   'frozen_gradient_remove_score': score[removals[i]],
                   'frozen_gradient_add_score': score[additions[i]]} for i in range(count)]
    endpoints = []
    for pair in result:
        remove, add = unit(pair['remove']), unit(pair['add'])
        if remove not in active or add not in inactive:
            raise ValueError('Each frozen exchange must remove an active and add an inactive unit in every selected case')
        endpoints.extend((remove, add))
    if len(endpoints) != len(set(endpoints)) or len({p['id'] for p in result}) != len(result):
        raise ValueError('Candidate exchanges need unique IDs and disjoint endpoints')
    if not result:
        raise ValueError('No legal candidate exchanges')
    return result


def load_references(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
            if Path(path).suffix.lower() == '.jsonl' else json.loads(text))
    if isinstance(rows, dict):
        rows = rows.get('records', rows.get('references', rows))
        if isinstance(rows, dict):
            rows = [dict(value, id=key) if isinstance(value, dict) else {'id': key, 'answers': value}
                    for key, value in rows.items()]
    return {str(row['id']): row for row in rows}


def answer(reference):
    values = reference.get('answers', reference.get('answer'))
    if values is None:
        first = reference.get('references', [{}])[0]
        values = first.get('answer', first.get('answers'))
    if isinstance(values, str):
        values = [values]
    if not values or not isinstance(values[0], str) or not values[0].strip():
        raise ValueError('Calibration reference must contain a nonempty fixed first answer')
    return values[0].strip()


def source_id(row):
    for key in ('source_document_id', 'source_image_id', 'image_id'):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError('Calibration row needs an independent source identifier')


def choose_rows(rows, source_count, questions_per_source):
    groups = defaultdict(list)
    for row in rows:
        groups[source_id(row)].append(row)
    ordered = sorted(groups, key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name))
    selected = []
    for source in ordered[:source_count]:
        def question_key(row):
            value = str(row.get('source_question_id', row.get('calibration_question_index', row['id'])))
            return (0, int(value)) if value.isdigit() else (1, value)
        selected.extend(sorted(groups[source], key=question_key)[:questions_per_source])
    return selected


def macro_scores(records, target_kind):
    """Question mean -> source mean -> equal-weight dataset mean."""
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record['dataset']][record['source_id']].append(record['scores'][target_kind]['mean_nll'])
    by_dataset, by_source = {}, {}
    for dataset, sources in grouped.items():
        by_source[dataset] = {source: sum(values) / len(values) for source, values in sources.items()}
        by_dataset[dataset] = sum(by_source[dataset].values()) / len(sources)
    return {'source_macro_nll': sum(by_dataset.values()) / len(by_dataset),
            'dataset_macro_nll': by_dataset, 'per_source_mean_nll': by_source,
            'questions': len(records), 'sources': sum(len(values) for values in grouped.values()),
            'aggregation': 'equal datasets; equal independent sources within dataset; equal questions within source'}


def pick_disjoint(rows, score_key, limit):
    chosen, used = [], set()
    for row in sorted(rows, key=lambda value: (-value[score_key], value['pair']['id'])):
        endpoints = {unit(row['pair']['remove']), unit(row['pair']['add'])}
        if row[score_key] > 0 and not used.intersection(endpoints):
            chosen.append(row['pair'])
            used.update(endpoints)
            if len(chosen) == limit:
                break
    return chosen


def self_test():
    ranked = [('k', 0, 0), ('v', 0, 0), ('k', 0, 1), ('v', 0, 1)]
    pair = {'id': 'a', 'remove': ['k', 0, 0], 'add': ['v', 0, 1]}
    changed = apply_swaps(ranked, [pair])
    assert changed == [('v', 0, 1), ('v', 0, 0), ('k', 0, 1), ('k', 0, 0)]
    assert stages(changed)[0] == ['v', 0, 1, 0]
    records = [{'dataset': 'doc', 'source_id': 'one', 'scores': {'gt': {'mean_nll': x}}} for x in (1., 3.)]
    records += [{'dataset': 'doc', 'source_id': 'two', 'scores': {'gt': {'mean_nll': 5.}}},
                {'dataset': 'text', 'source_id': 'three', 'scores': {'gt': {'mean_nll': 9.}}}]
    assert macro_scores(records, 'gt')['source_macro_nll'] == 6.25
    assert len(pick_disjoint([{'pair': pair, 'score': 1.}], 'score', 1)) == 1
    assert not pick_disjoint([{'pair': pair, 'score': -1.}], 'score', 1)
    print(json.dumps({'status': 'PASS', 'checks': ['same-universe exchange', 'source-balanced nested mean', 'positive fixed-objective selection']}))


if __name__ == '__main__':
    self_test()
