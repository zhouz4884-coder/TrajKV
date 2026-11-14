"""Frozen-method new-image Text64 transfer, official ten-annotator accuracy.

No Text calibration, candidate reselection, or DocVQA ANLS is performed here.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/analysis'))
import numpy as np
from textvqa_metric import TextVQAAccuracyEvaluator
from analyze_cycle import cluster_bootstrap
from analyze import write_csv
from costs import role
from summarize_round5_confirmation import (
    ROOT, MARKER, sha, load, local_path, frozen_asset_checks,
    run_manifest_checks, representation_check, named_comparisons, format_controls, write_output_hashes)


def metric_controls():
    metric = TextVQAAccuracyEvaluator()
    expected = [(0, 0.), (1, .3), (2, .6), (3, .9), (4, 1.), (10, 1.)]
    for matches, target in expected:
        answers = ['two']*matches+['other']*(10-matches)
        score = metric.eval_pred_list([dict(pred_answer='2', gt_answers=answers)])
        if not math.isclose(score, target, abs_tol=1e-12):
            raise RuntimeError('Official soft-accuracy synthetic control failed')
    if metric.eval_pred_list([dict(pred_answer='The CAT!', gt_answers=['cat']*10)]) != 1.:
        raise RuntimeError('Official answer-normalization control failed')
    return dict(status='PASS', ten_annotator_soft_scores=True, answer_normalization=True,
        format_controls=format_controls(), scope='synthetic scoring only; no live model answers inspected')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+')
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    controls = metric_controls()
    if a.self_test:
        print(json.dumps(controls)); return
    if not a.runs or a.output is None:
        p.error('--runs and --output required')
    if len(a.runs) != 2:
        raise ValueError('Expected the two frozen Text64 shards')
    frozen = load(MARKER)
    assets = frozen_asset_checks(frozen)
    sample_key = 'configs/a2/round5_v1/text_transfer64.json'
    reference_key = 'configs/data/round5_text64_references.json'
    samples = {row['id']: row for row in load(ROOT/sample_key)}
    refs = {row['id']: row for row in load(ROOT/reference_key)}
    if len(samples) != 64 or len(refs) != 64 or set(samples) != set(refs):
        raise ValueError('Text64 sample/reference identity mismatch')
    if len({row['image_id'] for row in samples.values()}) != 64:
        raise ValueError('Text64 must contain distinct image contents')
    for sid, row in samples.items():
        if len(row['questions']) != 1 or row['turns'] != 1 or refs[sid]['image_id'] != row['image_id']:
            raise ValueError('Text64 is not one matched question per image')
        if len(refs[sid]['answers']) != 10 or not all(isinstance(answer, str) for answer in refs[sid]['answers']):
            raise ValueError('Text64 needs ten nonnull annotator answers')
    cohort_path = ROOT/'audit/a2/round5_text64_membership.json'
    cohort = load(cohort_path)
    if (cohort['samples_sha256'] != frozen['input_sha256'][sample_key]
            or cohort['references_sha256'] != frozen['input_sha256'][reference_key]
            or cohort['overlap_source_ids'] != 0 or cohort['overlap_contents'] != 0
            or cohort.get('model_outcomes_used_for_selection') is not False):
        raise ValueError('New-image Text membership check failed')
    names = frozen['text_variants']
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError('Frozen Text variant set differs')
    frozen_lookup = {v['name']: v for v in frozen['variants']}
    configs = [frozen_lookup[name] for name in names]
    base_bytes = 4*load(local_path(next(v['metadata_factor_path'] for v in configs if v.get('metadata_factor_path'))))['parameter_count']
    records, run_checks, checks = [], [], []
    seen = set()
    for name in a.runs:
        folder = Path(name)
        if not folder.is_absolute():
            folder = ROOT/'results'/folder
        if load(folder/'complete.json')['status'] != 'COMPLETE':
            raise ValueError('Text shard/queue is incomplete')
        if (folder/'transfer32').is_dir():
            folder = folder/'transfer32'
        manifest, check = run_manifest_checks(folder, frozen, configs, frozen['input_sha256'][sample_key])
        run_checks.append(check)
        for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r['session_id'], r['variant'])
            if key in seen or key[0] not in samples or key[1] not in names or key[0] not in manifest['selected_ids']:
                raise ValueError('Duplicate/unexpected Text record')
            if len(r['turns']) != 1 or r['turns'][0]['turn'] != 1 or r['image_id'] != samples[key[0]]['image_id']:
                raise ValueError('Text first-question/image check failed')
            checks.append(representation_check(r, frozen_lookup[key[1]], base_bytes))
            seen.add(key); records.append(r)
    if seen != {(sid, name) for sid in samples for name in names}:
        raise ValueError('Incomplete Text64 x seven variants coverage')
    metric = TextVQAAccuracyEvaluator()
    scored, by, costs = [], defaultdict(dict), defaultdict(list)
    for r, audit in zip(records, checks):
        sid, name = r['session_id'], r['variant']
        answer = r['turns'][0]['answer']
        if not isinstance(answer, str):
            raise ValueError('Null/nontext generated answer')
        score = float(metric.eval_pred_list([dict(pred_answer=answer, gt_answers=refs[sid]['answers'])]))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Null/nonfinite Text score')
        image = samples[sid].get('source_original_image_id') or r['image_id']
        scored.append(dict(session_id=sid, image_id=r['image_id'], source_image_id=image,
            variant=name, turn=1, score=score, prediction=answer,
            reference=json.dumps(refs[sid]['answers'], ensure_ascii=False),
            visual_tokens=r['visual_tokens'], mask_sha256=audit['mask_sha256'],
            selected_axis_units=audit['mask_units'], analysis_role='independent_new_image_transfer'))
        by[name][(image, sid)] = score
        costs[name].append((r, audit))
    table = []
    for name in names:
        group = costs[name]
        answers = [row for row in scored if row['variant'] == name]
        boot = cluster_bootstrap([(row['source_image_id'], row['score']) for row in answers], repeats=8000)
        denominator = sum(r['owned_initial']['visual_fp16_denominator'] for r, _ in group)
        complete = sum(check['complete_visual_bytes'] for _, check in group)
        encoder = sum(r.get('encoder_calibration_parameters_bytes', 0) for r, _ in group)
        table.append(dict(variant=name, images=64, questions=64, accuracy=boot['mean'],
            ci_low=boot['ci_lower'], ci_high=boot['ci_upper'],
            effective_visual_bits=16*(complete-encoder)/denominator,
            budget_bits_including_encoder_constants=16*complete/denominator,
            visual_bytes_total=complete-encoder, encoder_constants_bytes_per_session=encoder/64,
            maximum_per_image_complete_bits=max(check['complete_visual_bits'] for _, check in group),
            role=role(group[0][0]), materialized_quality_port=False,
            eligible_lowbit_frontier=role(group[0][0]) == 'actual packed runtime'
                and all(check['complete_visual_bits'] <= frozen['budget_bits']+1e-12 for _, check in group),
            free_session_seconds_median=float(np.median([r['session_ms']/1000 for r, _ in group])),
            extra_reads_total=sum(r['extra_logical_read_bytes'] for r, _ in group),
            analysis_role='independent_new_image_transfer'))
    candidates = list(dict.fromkeys([frozen['primary'], frozen['secondary_low_storage'], frozen['original_finite_v']]))
    pairs = [(candidate, baseline) for candidate in candidates
             for baseline in ['full8', 'native4_8', 'base2_8', 'cold_meta_all_kv'] if candidate != baseline]
    comparisons = named_comparisons(by, pairs, 'secondary_new_image_text_transfer')
    a.output.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('scores.csv', scored), ('table.csv', table), ('comparisons.csv', comparisons)]:
        write_csv(a.output/filename, rows)
    result = dict(status='COMPLETE', analysis_role='independent_new_image_transfer',
        metric='official MMF/TextVQA 10-annotator soft accuracy',
        metric_implementation='TextVQAAccuracyEvaluator.eval_pred_list',
        metric_source_sha256=sha(Path(__file__).with_name('textvqa_metric.py')),
        primary_candidate=frozen['primary'], secondary_low_storage=frozen['secondary_low_storage'],
        original_finite_v=frozen['original_finite_v'],
        images=64, questions_per_variant=64, variants=7, sessions=448, scored_answers=448,
        table=table, comparisons=comparisons, controls=controls,
        samples_sha256=frozen['input_sha256'][sample_key], references_sha256=frozen['input_sha256'][reference_key],
        marker_sha256=sha(MARKER), membership_sha256=sha(cohort_path), analyzer_sha256=sha(__file__),
        audit=dict(status='PASS', assets=assets, runs=run_checks, representation_cases=len(checks),
            seed=1212, mask_bits_valid=True, required_nulls_absent=True,
            source_image_overlap=0, encoded_content_overlap=0),
        scope='independent new-image zero-shot TextVQA transfer on one frozen model; Doc-only method selection/calibration; one question per image; not the full5000 distribution',
        statistics='source-image cluster paired bootstrap; named transfer comparisons corrected as one secondary family, separate from three primary Doc tests',
        cost_scope='actual visual owners, residual sign/alpha, lookup, and encoder constants; bias or scales counted in their actual representation',
        source_results=[dict(run=check['job'], records_sha256=check['records_sha256']) for check in run_checks])
    (a.output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (a.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE', sessions=448, source_marker_sha256=sha(MARKER))))
    write_output_hashes(a.output)
    print(json.dumps(dict(status='COMPLETE', metric=result['metric'], table=table)))


if __name__ == '__main__':
    main()
