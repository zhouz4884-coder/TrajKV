"""Summarize the completed six-job queue without model/GPU/network work."""
from datetime import datetime, timezone
import json

from round8_data_prepare import ROOT, read, sha, new


def main():
    queue = ROOT / 'outputs/round8_start_20261118/natural_subsets_queue_v1'
    plans = read(queue / 'analysis_plan.json')['analyses']
    tables, analyses, jobs, paired = {}, [], [], []
    for plan in plans:
        group, dataset = plan['group'], plan['dataset']
        folder = ROOT / plan['output']
        summary = read(folder / 'summary.json')
        assert summary['status'] == 'COMPLETE' and summary['audit']['status'] == 'PASS'
        assert summary['variants'] == plan['expected_variants']
        tables[(group, dataset)] = {row['variant']: row for row in summary['table']}
        raw = ROOT / f'results/round8_v1/natural_dev64_{group}_v1_{dataset}'
        engine = read(raw / 'complete.json')
        marker = 'complete_round8.json' if group == 'pos1' else 'complete_candidate.json'
        assert engine['records'] == engine['expected'] == (64 if group == 'pos1' else 128)
        name = f'round8_natural_dev64_{group}_v1_{dataset}'
        launch = read(queue / 'receipts' / (name + '.launch.json'))
        jobs.append(dict(group=group, dataset=dataset, job=name, pid=launch['pid'], records=engine['records'],
            raw=raw.relative_to(ROOT).as_posix(), complete_marker=marker, marker_sha256=sha(raw / marker),
            engine_complete_sha256=sha(raw / 'complete.json'), manifest_sha256=sha(raw / 'manifest.json'),
            records_sha256=sha(raw / 'records.jsonl'), final_log_sha256=sha(queue / 'logs' / (name + '.log')),
            launch_receipt_sha256=sha(queue / 'receipts' / (name + '.launch.json')),
            status='COMPLETE_FINAL_MARKER_AND_COVERAGE_PASS', source_snapshot=launch['source_snapshot']))
        analyses.append(dict(group=group, dataset=dataset, path=plan['output'],
            summary_sha256=sha(folder / 'summary.json'), scores_sha256=sha(folder / 'scores.csv'),
            owner_audit_sha256=sha(folder / 'owner_audit.csv'),
            candidate_completion_audit_sha256=sha(folder / 'candidate_completion_audit.json'),
            variants=summary['variants'], sources=64, questions_per_variant=64, analysis_role=summary['analysis_role']))
        for comparison in summary['comparisons']:
            baseline = 'r7_frozen_joint43_control'
            if baseline not in (comparison['candidate'], comparison['comparator']):
                continue
            sign = -1 if comparison['candidate'] == baseline else 1
            candidate = comparison['comparator'] if sign == -1 else comparison['candidate']
            lo, hi = comparison['ci_low'], comparison['ci_high']
            paired.append(dict(group=group, dataset=dataset, candidate=candidate, baseline=baseline,
                delta_pp=100 * sign * comparison['delta'],
                ci_low_pp=100 * (lo if sign == 1 else -hi), ci_high_pp=100 * (hi if sign == 1 else -lo),
                signflip_p=comparison['p'], holm_within_view=comparison['holm_reported_comparisons'],
                comparison_family=comparison['comparison_family']))
    method_rows = []
    order = [('cd2', 'full8'), ('cd2', 'r7_frozen_joint43_control'), ('cd2', 'r8_oldbook_binaryV_jointgain'),
        ('cd2', 'r8_cd1_joint0'), ('cd2', 'r8_cd1_joint1'), ('vfield2', 'r8_kjoint0_vjoint1'),
        ('vfield2', 'r8_kjoint1_vjoint1'), ('pos1', 'r8_k4pos_vbinary')]
    for group, name in order:
        doc, text = tables[(group, 'doc')][name], tables[(group, 'text')][name]
        method_rows.append(dict(variant=name, Doc_ANLS_percent=100 * doc['score'], Doc_exact_percent=100 * doc['exact'],
            Text_official_percent=100 * text['score'], Doc_complete_bits=doc['budget_bits_including_encoder_constants'],
            Text_complete_bits=text['budget_bits_including_encoder_constants'],
            Doc_delta_vs_baseline_pp=100 * (doc['score'] - tables[('cd2', 'doc')]['r7_frozen_joint43_control']['score']),
            Text_delta_vs_baseline_pp=100 * (text['score'] - tables[('cd2', 'text')]['r7_frozen_joint43_control']['score']),
            role=('cap16-native-fitted position book to actual cap8 evaluation transfer' if group == 'pos1'
                  else 'same actual quality protocol; no new offline learned book/parameter')))
    assert len(jobs) == 6 and sum(job['records'] for job in jobs) == 640
    result = dict(schema='trajkv.round8.six-natural-subset-jobs-execution.v1', status='SIX_BOUNDED_JOBS_COMPLETE',
        created_at_utc=datetime.now(timezone.utc).isoformat(), authorized_jobs=6, new_free_running_records=640,
        baseline_records_reused=384, unique_raw_records_including_existing_baseline=1024,
        original_sources_per_domain=64, new_independent_source_count=0, GPU_leases_released=[0, 1],
        GPU0_release_sha256=sha(queue / 'GPU0_LEASE_RELEASED.json'), GPU1_release_sha256=sha(queue / 'GPU1_LEASE_RELEASED.json'),
        jobs=jobs, quality_analyses=analyses, method_table=method_rows, paired_vs_baseline=paired,
        protocol_scope_correction_sha256=sha(queue / 'protocol_scope_correction.json'),
        protocol_scope='Exposed development, stored first question/source. Offline fitted position book is cap16-native to actual cap8 transfer; CD/field interventions add no offline learned book/parameter.',
        statistics_scope='Source-cluster bootstrap and paired sign flips; Holm separately within each reported merged view. Reused controls add no sources.',
        no_failed_runs=True, no_run_or_jobname_reused=True, no_cap_or_old_book_changed=True,
        no_model_source_modified=True, no_tensor_image_or_weight_download=True,
        no_confirmation_GT_materialized=True, no_GitHub_mutation=True, code_sha256=sha(__file__),
        completion_scope='Only these six jobs and CPU scoring completed; parent protocol repair, research stage and publication are separate.')
    new(ROOT / 'audit/a2/round8_natural_subsets_execution_v1.json', result)
    new(queue / 'RESULTS.json', result)
    lines = ['# Six bounded natural-dev64 subset jobs', '',
        'All six jobs completed; both GPU leases returned. New records:640. Each domain reuses64 exposed sources and its stored first question.', '',
        '| Method | Doc ANLS % | Doc exact % | Text official % | Doc delta pp | Text delta pp |',
        '|---|---:|---:|---:|---:|---:|']
    for row in method_rows:
        lines.append(f"| {row['variant']} | {row['Doc_ANLS_percent']:.4f} | {row['Doc_exact_percent']:.4f} | {row['Text_official_percent']:.4f} | {row['Doc_delta_vs_baseline_pp']:+.4f} | {row['Text_delta_vs_baseline_pp']:+.4f} |")
    lines += ['', 'K4pos is the original cap16-native fitted table transferred to actual cap8 evaluation; extra3968B fully charged. CD/field retain the same actual quality protocol and add no offline learned book/parameter. This is exposed development, not independent-source confirmation.', '',
        'KVjoint adds one Doc exact answer and+0.390625pp observed ANLS. Paired CIs and within-view corrections are in RESULTS.json. Native reconstruction acceptance is not a task monotonicity guarantee.']
    with (queue / 'RESULTS.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], new_records=640, GPU_leases_released=[0, 1], table=method_rows,
        KVjoint_Doc_paired=[row for row in paired if row['group'] == 'vfield2' and row['dataset'] == 'doc' and row['candidate'] == 'r8_kjoint1_vjoint1'])))


if __name__ == '__main__':
    main()
