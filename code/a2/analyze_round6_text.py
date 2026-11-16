"""Generic, complete-coverage TextVQA scoring for seen or fresh cohorts.

Reuse the frozen Round5 official normalized, ten-annotator leave-one-out metric.
Do not assume a cohort size, variant count, method family, or confirmation role.
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/analysis'))
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap, write_csv
from analyze import paired
from costs import role, visual_owner
from analyze_round5 import validate_sample_source
from analyze_docvqa_confirm64 import holm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local_path(value):
    value = str(value).replace('\\', '/')
    return ROOT / value.split('/TrajKV/', 1)[1] if '/TrajKV/' in value else Path(value)


def controls():
    metric = TextVQAAccuracyEvaluator()
    for matches, expected in [(0, 0.), (1, .3), (2, .6), (3, .9), (4, 1.), (10, 1.)]:
        score = metric.eval_pred_list([dict(pred_answer='2', gt_answers=['two'] * matches + ['other'] * (10 - matches))])
        if not math.isclose(score, expected, abs_tol=1e-12):
            raise ValueError('Official leave-one-out soft score control failed')
    if metric.eval_pred_list([dict(pred_answer='The CAT!', gt_answers=['cat'] * 10)]) != 1.:
        raise ValueError('Official answer normalization control failed')
    return dict(status='PASS', normalized_answers=True, ten_annotator_leave_one_out=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-variants', type=int, required=True)
    p.add_argument('--candidate')
    p.add_argument('--analysis-role', default='seen_cohort_development_transfer')
    p.add_argument('--bootstrap-repeats', type=int, default=8000)
    p.add_argument('--budget-bits', type=float, default=1.8)
    args = p.parse_args()
    metric_controls = controls()
    samples_list, references_list = load(args.samples), load(args.references)
    samples, references = {r['id']:r for r in samples_list}, {r['id']:r for r in references_list}
    if len(samples) != len(samples_list) or not samples or len(references) != len(references_list):
        raise ValueError('Empty or duplicate sample/reference identities')
    if not set(samples) <= set(references):
        raise ValueError('Missing selected Text references')
    for sid, row in samples.items():
        reference = references[sid]
        if (row.get('dataset') != 'textvqa' or row['turns'] != 1 or len(row['questions']) != 1
                or reference['image_id'] != row['image_id'] or len(reference['answers']) != 10
                or not all(isinstance(answer, str) for answer in reference['answers'])):
            raise ValueError('Text sample/reference one-question/ten-answer identity mismatch')
    seen, variants, records, run_checks = set(), {}, [], []
    for name in args.runs:
        folder = Path(name)
        if not folder.is_absolute():
            folder = ROOT / 'results' / folder
        if load(folder / 'complete.json')['status'] != 'COMPLETE':
            raise ValueError('Incomplete supplied run: ' + str(folder))
        if (folder / 'quality').is_dir():
            folder = folder / 'quality'
        elif (folder / 'transfer32').is_dir():
            folder = folder / 'transfer32'
        manifest = load(folder / 'manifest.json')
        if manifest.get('answers_visible_to_runner') is not False:
            raise ValueError('Manifest does not establish answer-free inference')
        selected = manifest['selected_ids']
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(samples):
            raise ValueError('Manifest selected-ID mismatch')
        source_check = validate_sample_source(manifest, args.samples, selected)
        local_variants = {v['name']:v for v in manifest['config']['variants']}
        if len(local_variants) != len(manifest['config']['variants']):
            raise ValueError('Duplicate manifest variant names')
        for variant, config in local_variants.items():
            if variant in variants and variants[variant] != config:
                raise ValueError('Same variant name has different configs across runs')
            variants[variant] = config
        codec_checks = {}
        for value, expected in manifest.get('external_codec_inputs', {}).items():
            path = local_path(value)
            if not path.is_file() or sha(path) != expected:
                raise ValueError('External codec source identity mismatch: ' + str(path))
            codec_checks[value] = expected
        local_seen = set()
        for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r['session_id'], r['variant'])
            if key in seen or key[0] not in selected or key[1] not in local_variants:
                raise ValueError('Duplicate/unexpected session/variant record')
            if (r['variant_config'] != local_variants[key[1]] or r['protocol'] != 'image_boundary_free_running'
                    or r.get('task_accuracy_eligible') is False or r.get('primary_quality_eligible') is False
                    or not r.get('read_ledger_exact') or not r.get('global_extra_budget_ok', True)):
                raise ValueError('Record config/free-answer/read-budget audit failed')
            if (r['image_id'] != samples[key[0]]['image_id'] or len(r['turns']) != 1
                    or r['turns'][0]['turn'] != 1 or not isinstance(r['turns'][0]['answer'], str)):
                raise ValueError('Wrong image or incomplete first-question output')
            seen.add(key); local_seen.add(key); records.append(r)
        if local_seen != {(sid, variant) for sid in selected for variant in local_variants}:
            raise ValueError('Incomplete per-run sample x variant coverage')
        run_checks.append(dict(run=str(folder), records_sha256=sha(folder / 'records.jsonl'),
            manifest_sha256=sha(folder / 'manifest.json'), config_sha256=manifest['config_sha256'],
            sample_source_check=source_check, selected_sessions=len(selected), external_codec_inputs=codec_checks,
            recorded_inference_source_sha256=manifest.get('sources', {})))
    if len(variants) != args.expected_variants or seen != {(sid, variant) for sid in samples for variant in variants}:
        raise ValueError('Incomplete selected cohort x expected variant coverage')
    metric = TextVQAAccuracyEvaluator()
    scored, by, costs = [], defaultdict(dict), defaultdict(list)
    for r in records:
        sid, variant = r['session_id'], r['variant']
        answer, targets = r['turns'][0]['answer'], references[sid]['answers']
        normalized = metric.answer_processor(answer)
        score = float(metric.eval_pred_list([dict(pred_answer=answer, gt_answers=targets)]))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid official score')
        cluster = str(samples[sid].get('source_original_image_id') or r['image_id'])
        base, _ = visual_owner(r)
        base += int(r['owned_initial'].get('spectrum_experiment_reference', 0))
        encoder = int(r.get('encoder_calibration_parameters_bytes', 0))
        denominator = int(r['owned_initial']['visual_fp16_denominator'])
        complete = base + encoder
        for field in ('depth_residual_complete_bytes',):
            if field in r and int(r[field]) != complete:
                raise ValueError('Declared complete bytes disagree with actual owned bytes')
        for field in ('depth_residual_actual_budget_bits', 'mixed_residual_actual_budget_bits', 'sparse_residual_actual_budget_bits'):
            if field in r and not math.isclose(float(r[field]), 16 * complete / denominator, rel_tol=0, abs_tol=1e-10):
                raise ValueError('Declared actual budget disagrees with full owner')
        scored.append(dict(session_id=sid, image_id=r['image_id'], source_image_id=cluster, variant=variant,
            turn=1, score=score, normalized_exact_any_annotator=int(normalized in [metric.answer_processor(x) for x in targets]),
            prediction=answer, normalized_prediction=normalized, reference=json.dumps(targets, ensure_ascii=False),
            visual_tokens=r['visual_tokens'], complete_visual_bytes=complete, encoder_constant_bytes=encoder,
            complete_visual_bits=16 * complete / denominator, analysis_role=args.analysis_role))
        by[variant][(cluster, sid)] = score
        costs[variant].append((r, base, encoder))
    table = []
    for variant in variants:
        rows = [row for row in scored if row['variant'] == variant]
        boot = cluster_bootstrap([(row['source_image_id'], row['score']) for row in rows], repeats=args.bootstrap_repeats)
        group = costs[variant]
        denominator = sum(r['owned_initial']['visual_fp16_denominator'] for r, _, _ in group)
        base, encoder = sum(b for _, b, _ in group), sum(e for _, _, e in group)
        materialized = any(r.get('materialized_quality_port', False) for r, _, _ in group)
        table.append(dict(variant=variant, images=len({row['source_image_id'] for row in rows}), questions=len(rows),
            accuracy=boot['mean'], ci_low=boot['ci_lower'], ci_high=boot['ci_upper'],
            normalized_exact_any_annotator=float(np.mean([row['normalized_exact_any_annotator'] for row in rows])),
            effective_visual_bits=16 * base / denominator, budget_bits_including_encoder_constants=16 * (base + encoder) / denominator,
            visual_bytes_total=base, encoder_constants_bytes_per_session=encoder / len(group),
            maximum_per_image_complete_bits=max(row['complete_visual_bits'] for row in rows),
            eligible_lowbit_frontier=not materialized and all(role(r) == 'actual packed runtime' and r.get('lowbit_deployment_candidate', True) for r, _, _ in group)
                and all(row['complete_visual_bits'] <= args.budget_bits + 1e-12 for row in rows),
            role=role(group[0][0]), materialized_quality_port=materialized,
            free_session_seconds_median=float(np.median([r['session_ms'] / 1000 for r, _, _ in group])),
            extra_reads_total=sum(r['extra_logical_read_bytes'] for r, _, _ in group), analysis_role=args.analysis_role))
    if args.candidate and args.candidate not in variants:
        raise ValueError('Candidate not present')
    pairs = ([(args.candidate, other) for other in variants if other != args.candidate] if args.candidate
             else list(itertools.combinations(variants, 2)))
    comparisons = [dict(candidate=left, comparator=right, **paired(by[left], by[right], repeats=args.bootstrap_repeats))
                   for left, right in pairs]
    for row, adjusted in zip(comparisons, holm([row['p'] for row in comparisons])):
        row['holm_reported_comparisons'] = adjusted
        row['comparison_family'] = 'requested_candidate_vs_others' if args.candidate else 'exploratory_all_pairs'
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('scores.csv', scored), ('table.csv', table), ('comparisons.csv', comparisons)]:
        write_csv(args.output / name, rows)
    summary = dict(status='COMPLETE', metric='official MMF/TextVQA normalized ten-annotator leave-one-out soft accuracy',
        metric_source_sha256=sha(Path(__file__).with_name('textvqa_metric.py')), controls=metric_controls,
        analysis_role=args.analysis_role, variants=len(variants), sessions=len(records), questions_per_variant=len(samples),
        images=len({row['source_image_id'] for row in scored}), table=table, comparisons=comparisons,
        samples_sha256=sha(args.samples), references_sha256=sha(args.references), analyzer_sha256=sha(__file__),
        audit=dict(status='PASS', complete_coverage=True, sample_and_reference_identities=True, codec_input_sha256=True,
                   declared_complete_owner_checks=True, seed=1212, runs=run_checks),
        scope='cohort role supplied explicitly; this scorer does not turn seen images into blind confirmation',
        statistics='question-weighted source-image paired cluster bootstrap; sign-flip two-sided p; Holm over reported comparisons',
        cost_scope='actual visual base/residual/lookup/positions plus encoder constants; no padding or amortization')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status='COMPLETE', sessions=len(records))), encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE', table=table, comparisons=comparisons)))


if __name__ == '__main__':
    main()
