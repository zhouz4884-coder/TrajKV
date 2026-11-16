"""CPU post-hoc paired output transitions; official metrics remain unchanged."""
import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [('k1_inc43_to_cond', 'r7_joint_k1_inc43', 'r7_joint_k1_gt_cond'),
         ('k2_inc_to_cond', 'r7_budget_k2_inc', 'r7_budget_k2_cond')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    return value.lower().strip()


def digits(value):
    return ''.join(re.findall(r'\d', value))


def readout_class(targets):
    """Post-hoc string forms only; numerical identifiers are not quantities."""
    value = targets[0].strip()
    low = value.lower()
    if (re.search(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', low)
            or re.search(r'\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}', value)
            or re.search(r'\d{1,2}:\d{2}', value)):
        return 'date_or_time_like'
    if re.fullmatch(r'[+-]?\d+(?:[.,]\d+)*%?', value):
        return 'numeric_string_or_numeric_identifier'
    if digits(value) and re.fullmatch(r'[\d\s().,+/%:$\-]+', value):
        return 'punctuated_numeric_or_phone_code'
    if digits(value) and re.search(r'[A-Za-z]', value) and not re.search(r'\s', value):
        return 'compact_alphanumeric_code_like'
    return 'other_text'


def aggregate(rows):
    return {'questions': len(rows),
        'anls_delta_pp': 100 * statistics.mean(row['anls_delta'] for row in rows),
        'exact_delta_pp': 100 * statistics.mean(row['exact_delta'] for row in rows),
        'answer_changed': sum(row['answer_changed'] for row in rows),
        'normalized_answer_changed': sum(row['normalized_answer_changed'] for row in rows),
        'anls_improved': sum(row['anls_delta'] > 0 for row in rows),
        'anls_worsened': sum(row['anls_delta'] < 0 for row in rows),
        'anls_unchanged': sum(row['anls_delta'] == 0 for row in rows),
        'exact_gained': sum(row['exact_delta'] > 0 for row in rows),
        'exact_lost': sum(row['exact_delta'] < 0 for row in rows),
        'sum_positive_anls_delta': sum(max(0, row['anls_delta']) for row in rows),
        'sum_negative_anls_delta': sum(min(0, row['anls_delta']) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'audit/a2/round7_output_error_transitions_v1.json')
    parser.add_argument('--csv', type=Path, default=ROOT / 'audit/a2/round7_output_error_transitions_v1.csv')
    args = parser.parse_args()
    scores_path = ROOT / 'outputs/round7_primary_preview_v1/doc/scores.csv'
    samples_path = ROOT / 'configs/a2/round7_v1/doc_confirm128.json'
    summary_path = ROOT / 'outputs/round7_primary_preview_v1/doc/summary.json'
    source_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                     for path in (scores_path, samples_path, summary_path)}
    samples = {row['id']: row for row in json.loads(samples_path.read_text())}
    scores = list(csv.DictReader(scores_path.open(encoding='utf-8-sig')))
    by_variant = collections.defaultdict(dict)
    for row in scores:
        key = (row['session_id'], int(row['turn']))
        assert key not in by_variant[row['variant']]
        by_variant[row['variant']][key] = row
    assert all(len(rows) == 412 for rows in by_variant.values())
    pair_rows, results = [], {}
    for name, before_name, after_name in PAIRS:
        rows = []
        for key, before in by_variant[before_name].items():
            after, sample = by_variant[after_name][key], samples[key[0]]
            assert before['reference'] == after['reference']
            targets = json.loads(before['reference'])
            old, new = before['prediction'], after['prediction']
            value = {'pair': name, 'session_id': key[0], 'source_document_id': sample['source_document_id'],
                'source_page_id': sample['source_page_id'], 'question_id': str(sample['source_question_ids'][key[1] - 1]),
                'turn': key[1], 'question': sample['questions'][key[1] - 1].split('\nAnswer the question using')[0],
                'official_reference_answers': targets, 'before_answer': old, 'after_answer': new,
                'before_anls': float(before['score']), 'after_anls': float(after['score']),
                'anls_delta': float(after['score']) - float(before['score']),
                'before_exact': float(before['exact']), 'after_exact': float(after['exact']),
                'exact_delta': float(after['exact']) - float(before['exact']),
                'answer_changed': old != new, 'normalized_answer_changed': normalize(old) != normalize(new),
                'same_alphanumeric_content_ignoring_format': re.sub(r'[\W_]', '', normalize(old)) == re.sub(r'[\W_]', '', normalize(new)),
                'digit_sequence_changed': bool(digits(old) or digits(new)) and digits(old) != digits(new),
                'before_digit_sequence': digits(old), 'after_digit_sequence': digits(new),
                'posthoc_reference_string_class': readout_class(targets)}
            rows.append(value)
        pair_rows.extend(rows)
        results[name] = dict(aggregate(rows), before_variant=before_name, after_variant=after_name,
                            changed_rows=[row for row in rows if row['answer_changed']])
        by_class = {}
        for kind in sorted({row['posthoc_reference_string_class'] for row in rows}):
            by_class[kind] = aggregate([row for row in rows if row['posthoc_reference_string_class'] == kind])
        results[name]['by_reference_string_class'] = by_class
    coarse = {}
    low = by_variant['r7_joint_k1_gt_cond']
    for comparator in ('base2_8', 'native4_8'):
        grouped = collections.defaultdict(list)
        for key, row in low.items():
            other = by_variant[comparator][key]
            grouped[readout_class(json.loads(row['reference']))].append({
                'low_anls': float(row['score']), 'other_anls': float(other['score']),
                'low_exact': float(row['exact']), 'other_exact': float(other['exact'])})
        coarse[comparator] = {kind: {'questions': len(rows),
            'k1_cond_anls_percent': 100 * statistics.mean(row['low_anls'] for row in rows),
            'comparator_anls_percent': 100 * statistics.mean(row['other_anls'] for row in rows),
            'k1_cond_minus_comparator_anls_pp': 100 * statistics.mean(row['low_anls'] - row['other_anls'] for row in rows),
            'k1_cond_minus_comparator_exact_pp': 100 * statistics.mean(row['low_exact'] - row['other_exact'] for row in rows)}
            for kind, rows in grouped.items()}
    result = {'schema': 'trajkv.round7.output-error-transitions.v1', 'status': 'COMPLETE',
        'scope': 'Post-hoc explanation of the same 412 completed frozen Doc questions; no new confirmation, inference, metric, or sample selection',
        'pair_table_rows': len(pair_rows), 'csv': str(args.csv.relative_to(ROOT)).replace('\\', '/'),
        'official_metric_unchanged': True, 'all_questions_included': True,
        'pairs': results, 'k1_conditional_vs_baselines_posthoc_reference_string_classes': coarse,
        'class_definition': 'Priority: date/month/time-like regex; numeric-only/punctuated numeric; compact digit+letter without spaces; other. Numeric strings may be identifiers. These use GT only for post-hoc description, not deployment features or a new quality metric.',
        'input_sha256': source_hashes, 'analysis_source_sha256': sha(__file__)}
    if args.output.exists() or args.csv.exists():
        raise FileExistsError('Preserve existing transition outputs; choose new paths')
    with args.csv.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(dict(row, official_reference_answers=json.dumps(row['official_reference_answers'], ensure_ascii=False)))
    result['csv_sha256'] = sha(args.csv)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for path, expected in source_hashes.items():
        assert sha(ROOT / path) == expected
    print(json.dumps({'status': 'COMPLETE', 'all_question_pair_rows': len(pair_rows),
        'pairs': {name: {k: v for k, v in row.items() if k not in ('changed_rows', 'by_reference_string_class')}
                  for name, row in results.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
