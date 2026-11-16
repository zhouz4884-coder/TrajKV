"""Create calibration-only TextVQA plurality-first references, preserving votes.

Only existing train16 labels are read. No model scores, model outputs, future
confirmation labels, or GPU execution enter the target-selection rule.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from textvqa_metric import TextVQAAccuracyEvaluator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'configs/data/round7_text_train16_independent_references.json'
DEFAULT_OUTPUT = ROOT / 'configs/data/round7_text_train16_plurality_calibration_v1.json'
DEFAULT_AUDIT = ROOT / 'audit/a2/round7_text_train16_plurality_calibration_v1.json'
FIXED_PREDICTIONS = ('', '0', 'two', 'The CAT!', 'coffee', 'coffee sandwich',
                     'bank of america', 'unmatched calibration control')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def score(metric, prediction, answers):
    return float(metric.eval_pred_list([dict(pred_answer=prediction, gt_answers=answers)]))


def normalize_plurality_first(row, processor, normalizer_sha):
    answers = row['answers']
    require(len(answers) == 10 and all(isinstance(answer, str) for answer in answers),
            'Exactly ten original string annotator answers required')
    normalized = [processor(answer) for answer in answers]
    votes = Counter(normalized)
    maximum = max(votes.values())
    tied = sorted(answer for answer, count in votes.items() if count == maximum)
    target = tied[0]
    index = next(i for i, answer in enumerate(normalized) if answer == target)
    result = copy.deepcopy(row)
    reordered = list(answers)
    chosen_raw = reordered.pop(index)
    reordered.insert(0, chosen_raw)
    result['answers'] = reordered
    result['calibration_target_policy'] = dict(
        schema='trajkv.round7.text-plurality-calibration-target.v1',
        selection='official normalized plurality; normalized lexical tie break; first matching original raw string moved to index0',
        normalizer='frozen official TextVQA EvalAIAnswerProcessor', normalizer_source_sha256=normalizer_sha,
        normalized_target=target, plurality_votes=maximum, tied_normalized_targets=tied,
        original_raw_index=index, raw_annotator_multiset_preserved=True,
        scope='calibration target serialization only; original evaluation references unchanged; no official task-score change')
    require(Counter(result['answers']) == Counter(answers), 'Raw annotator multiset changed')
    require(processor(result['answers'][0]) == target, 'First raw answer does not match selected normalized target')
    # Every field except the new explanatory metadata and the answer order is
    # preserved. In particular IDs, source IDs and ten raw strings are retained.
    stripped = {key: value for key, value in result.items() if key not in ('answers', 'calibration_target_policy')}
    require(stripped == {key: value for key, value in row.items() if key != 'answers'}, 'Non-answer identity metadata changed')
    change = dict(id=row['id'], source_question_id=row.get('source_question_id'),
        source_original_image_id=row.get('source_original_image_id'),
        original_first_raw=answers[0], original_first_normalized=normalized[0],
        selected_first_raw=chosen_raw, selected_normalized_target=target, selected_raw_original_index=index,
        normalized_vote_counts=dict(sorted(votes.items())), plurality_votes=maximum,
        tied_normalized_targets=tied, answer_order_changed=index != 0,
        normalized_first_target_changed=normalized[0] != target, raw_multiset_preserved=True)
    return result, change


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    p.add_argument('--expected-questions', type=int, default=25)
    args = p.parse_args()
    require(not args.output.exists() and not args.audit.exists(), 'Use new output/audit paths; frozen originals are never overwritten')
    source_sha = sha(args.input)
    if args.input.resolve() == DEFAULT_INPUT.resolve():
        membership = json.loads((ROOT / 'audit/a2/round7_development_membership_v1.json').read_text(encoding='utf-8-sig'))
        require(source_sha == membership['datasets']['text_train16_independent']['references_sha256'],
                'Original train16 reference bytes differ from frozen development membership')
    rows = json.loads(args.input.read_text(encoding='utf-8-sig'))
    require(isinstance(rows, list) and len(rows) == args.expected_questions
            and len({row['id'] for row in rows}) == len(rows), 'Unexpected/duplicate calibration questions')
    metric = TextVQAAccuracyEvaluator()
    normalizer_sha = sha(Path(__file__).with_name('textvqa_metric.py'))
    output, changes, invariance = [], [], []
    maximum_difference = 0.
    for row in rows:
        result, change = normalize_plurality_first(row, metric.answer_processor, normalizer_sha)
        output.append(result); changes.append(change)
        original_map = metric._compute_answer_scores(row['answers'])
        reordered_map = metric._compute_answer_scores(result['answers'])
        require(set(original_map) == set(reordered_map), 'Official normalized answer-score support changed')
        require(all(math.isclose(original_map[key], reordered_map[key], rel_tol=0., abs_tol=1e-12)
                    for key in original_map), 'Official answer-score map changed after reordering')
        # Fixed predictions are source/model independent. Also check every raw
        # human answer, which covers every normalized answer-score category.
        predictions = list(dict.fromkeys([*FIXED_PREDICTIONS, *row['answers']]))
        per_prediction = []
        for prediction in predictions:
            before, after = score(metric, prediction, row['answers']), score(metric, prediction, result['answers'])
            difference = abs(before - after)
            maximum_difference = max(maximum_difference, difference)
            require(math.isclose(before, after, rel_tol=0., abs_tol=1e-12), 'Official score changed under answer reordering')
            per_prediction.append(dict(prediction=prediction, before=before, after=after, absolute_difference=difference))
        invariance.append(dict(id=row['id'], fixed_predictions=list(FIXED_PREDICTIONS),
            all_original_raw_answer_categories_checked=True, score_map_unchanged=True, predictions=per_prediction))
    require(sha(args.input) == source_sha, 'Original calibration references changed during preparation')
    for before, after in zip(rows, output):
        require(before['id'] == after['id'] and Counter(before['answers']) == Counter(after['answers']),
                'Final multiset/identity check failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    target_sha = sha(args.output)
    audit = dict(schema='trajkv.round7.text-plurality-calibration-audit.v1', status='PASS',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_references=str(args.input.resolve()), original_sha256=source_sha,
        new_calibration_references=str(args.output.resolve()), new_sha256=target_sha,
        questions=len(rows), original_annotator_answers=len(rows) * 10,
        raw_multiset_preserved_all_questions=True, original_reference_file_unchanged=sha(args.input) == source_sha,
        reordered_questions=sum(change['answer_order_changed'] for change in changes),
        normalized_target_changed_questions=sum(change['normalized_first_target_changed'] for change in changes),
        modified_target_list=[change for change in changes if change['answer_order_changed']],
        all_target_selection_details=changes, official_soft_score_invariance=invariance,
        official_soft_score_map_unchanged_all_questions=True, maximum_official_score_absolute_difference=maximum_difference,
        official_normalizer_sha256=normalizer_sha, source_sha256=sha(__file__),
        model_scores_read=False, model_outputs_read=False, selection_used_model_scores=False,
        future_confirmation_references_opened=False, original_evaluation_references_modified=False,
        calibration_only=True, GPU_used=False, uploaded=False,
        scope='Only the first calibration target changes to a normalized plurality. Ten raw annotator answers remain the same multiset; this is not an official evaluation-GT correction or a new performance result.')
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', questions=len(rows), reordered_questions=audit['reordered_questions'],
        normalized_target_changed_questions=audit['normalized_target_changed_questions'],
        source_sha256=source_sha, new_sha256=target_sha, official_score_max_abs_difference=maximum_difference,
        output=str(args.output), audit=str(args.audit), calibration_only=True, GPU_used=False, uploaded=False)))


if __name__ == '__main__':
    main()
