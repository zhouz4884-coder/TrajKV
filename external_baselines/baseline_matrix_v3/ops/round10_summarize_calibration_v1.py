"""Independent scalar/coverage recomputation for a CLOSED Round10 selector.

Local files only. Not an execution/source-retrieval validator or a task scorer.
Private PT/WO are not loaded. Incomplete runs produce readiness only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

CALIBRATOR_SHA = 'abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a'
LAMBDAS = (0., .25, .5, 1.)
WINDOWS = (1, 32, 64, 128, 256, 'global')
DOMAINS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')


def require(value, message):
    if not value:
        raise ValueError('ROUND10_CALIBRATION_SUMMARY: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)


def csv_table(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    names = ('manifest.json', 'records.jsonl', 'complete.json', 'fit_collection_complete.json',
             'value_theta_selected.json', 'window_selected.json')
    missing = [name for name in names if not (run / name).is_file()]
    complete = read(run / 'complete.json') if (run / 'complete.json').is_file() else None
    if missing or complete.get('status') != 'PASS_LS48_AND_ACTUAL_MODEL_SELECT16':
        result = {'status': 'WAIT_FULL_CALIBRATION', 'missing': missing,
                  'producer_status': None if complete is None else complete.get('status'),
                  'theta': None, 'window': None, 'task_quality_claim': False,
                  'no_remote_GPU_or_private_PT_access': True}
        write(output / 'readiness.json', result)
        return result
    manifest, fit = read(run / 'manifest.json'), read(run / 'fit_collection_complete.json')
    theta, window_doc = read(run / 'value_theta_selected.json'), read(run / 'window_selected.json')
    require(complete['actual_GPU'] is True and complete['references_read'] is False and
            complete['fit_sources'] == 48 and complete['select_sources'] == 16 and
            complete['new_free_generation_task_answers'] == 0, 'closed producer scope')
    require(manifest['source_hashes'] == complete['source_hashes'] and
            complete['source_hashes']['code/round10_calibrate_v1.py'] == CALIBRATOR_SHA,
            'frozen producer source identity')
    require(sha(run / 'records.jsonl') == complete['records_sha256'] and
            sha(run / 'value_theta_selected.json') == complete['theta_sha256'] and
            sha(run / 'window_selected.json') == complete['window_sha256'], 'artifact SHA mismatch')
    require(fit['status'] == 'PASS_REAL_MODEL_FIT48_ALL28' and len(fit['sources']) == 48 and
            fit['private_WO_sha256'] == complete['private_WO_sha256'], 'fit collection identity')
    fits, teachers, branches = [], {}, {}
    actual, reused, forwards = 0, 0, 0
    table = []
    with (run / 'records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['phase'] == 'fit_source_captured':
                fits.append(row)
                continue
            key = (row['dataset'], row['source_id'])
            if row['phase'] == 'select_source_teacher':
                require(key not in teachers and 0 < len(row['teacher_token_ids']) <= 32,
                        'unique select source/EOS cap')
                teachers[key] = row
                branches[key] = {}
                continue
            require(row['phase'] == 'select_branch' and key in teachers, 'unknown/unordered record')
            require(row['context'] == teachers[key]['context'] and row['id'] == teachers[key]['id'], 'same source/prompt/positions')
            label = row['branch']
            require(label not in branches[key], 'duplicate branch')
            scores = row['loss']
            n = len(teachers[key]['teacher_token_ids'])
            require(scores['positions'] == len(scores['position_KL']) == n and
                    close(sum(scores['position_KL']) / n, scores['source_mean_KL']) and
                    all(math.isfinite(x) and x >= -1e-10 for x in scores['position_KL']), 'position/source KL')
            alias = row['same_case_exact_field_reuse_from']
            if alias is None:
                require(scores['actual_packed_owner_calls'] == 28 * n and
                        scores['actual_model_TF_source_evaluation'] is True, 'actual all-layer TF use')
                actual += 1
                forwards += n
            else:
                require(alias in branches[key], 'reuse producer not earlier in same source')
                original = branches[key][alias]
                require(row['reader_field_sha256'] == original['reader_field_sha256'] and
                        scores['position_KL'] == original['loss']['position_KL'] and
                        scores['actual_packed_owner_calls'] == 0 and
                        scores['actual_model_TF_source_evaluation'] is False, 'exact-field reuse binding')
                reused += 1
            require(scores['question_native_KV_reused'] is False and scores['question_suffix_recomputed'] is True and
                    scores['current_query_and_text_KV'] is True, 'live question/state contract')
            stages = row['owner_metadata']['selected_stages']
            require(len(stages) == 43 and Counter(x[0] for x in stages) == Counter(k=23, v=20), 'K23/V20 coverage')
            require(not branches[key] or row['K_fields_sha256'] == next(iter(branches[key].values()))['K_fields_sha256'], 'K fixed across source branches')
            branches[key][label] = row
            table.append({'dataset': row['dataset'], 'source_id': row['source_id'], 'id': row['id'],
                'branch': label, 'mode': row['mode'], 'lambda': row['lambda'], 'window': row['window'],
                'answer_positions': n, 'source_mean_KL': scores['source_mean_KL'],
                'actual_TF_source_evaluation': alias is None, 'reused_from': alias,
                'reader_fields_sha256': row['reader_field_sha256']})
    require(len(fits) == 48 and Counter(x['dataset'] for x in fits) == Counter({d: 12 for d in DOMAINS}) and
            len({(x['dataset'], x['source_id']) for x in fits}) == 48, 'fit48 four-domain coverage')
    require([(x['file'], x['sha256']) for x in fits] == [(x['file'], x['sha256']) for x in fit['sources']], 'fit receipt list binding')
    require(len(teachers) == 16 and Counter(d for d, _ in teachers) == Counter({d: 4 for d in DOMAINS}), 'select16 four-domain coverage')
    require(not {x['context']['image_sha256'] for x in fits}.intersection(
        x['context']['image_sha256'] for x in teachers.values()), 'fit/select image overlap')
    lambda_rows, window_rows = [], []
    for coefficient in LAMBDAS:
        means = {}
        for mode in ('LS-FZ', 'LS-RE'):
            values = []
            for group in branches.values():
                found = [r for r in group.values() if r['mode'] == mode and r['lambda'] == coefficient]
                require(len(found) == 1, 'lambda/deployment coverage')
                values.append(found[0]['loss']['source_mean_KL'])
            means[mode] = sum(values) / 16
        objective = .5 * (means['LS-FZ'] + means['LS-RE'])
        declared = complete['lambda_scores'][str(coefficient)]
        require(all(close(means[k], declared[k]) for k in means) and close(objective, declared['common_objective']), 'common lambda mean recomputation')
        lambda_rows.append({'lambda': coefficient, 'FZ_mean_KL': means['LS-FZ'], 'RE_mean_KL': means['LS-RE'],
                            'common_objective': objective, 'identity_control': coefficient == 0})
    chosen = min(lambda_rows[1:], key=lambda x: (x['common_objective'], x['lambda']))['lambda']
    require(chosen == complete['selected_common_lambda'] == theta['fit_provenance']['selected_common_lambda'], 'common nonzero lambda selection')
    for window in WINDOWS:
        values = []
        for group in branches.values():
            found = [r for r in group.values() if r['lambda'] is None and r['window'] == window]
            require(len(found) == 1, 'window coverage')
            values.append(found[0]['loss']['source_mean_KL'])
        mean = sum(values) / 16
        require(close(mean, complete['window_scores'][str(window)]), 'window loss recomputation')
        window_rows.append({'window': window, 'mean_KL': mean, 'W1_control': window == 1})
    chosen_w = min(window_rows[1:], key=lambda x: (x['mean_KL'], math.inf if x['window'] == 'global' else x['window']))['window']
    require(chosen_w == complete['selected_window'] == window_doc['window'], 'nontrivial window selection')
    counts = complete['counts']
    require(actual == counts['TF_source_evaluations_actual'] and reused == counts['TF_source_evaluations_reused_exact_fields'] and
            forwards == counts['TF_student_outer_forwards_actual'] and actual + reused == 224 and
            counts['teacher_generated_sources'] == 64, 'actual/reuse counters')
    csv_table(output / 'lambda_losses.csv', lambda_rows)
    csv_table(output / 'window_losses.csv', window_rows)
    csv_table(output / 'source_branch_losses.csv', table)
    result = {'status': 'PASS_LOCAL_SELECTOR_AND_COUNTER_RECOMPUTATION', 'selected_common_lambda': chosen,
              'selected_window': chosen_w, 'fit_sources': 48, 'select_sources': 16,
              'logical_select_source_branches': actual + reused, 'actual_TF_sources': actual,
              'exact_field_TF_reuses': reused, 'actual_student_outer_forwards': forwards,
              'teacher_generation_source_count_recomputed': 64,
              'teacher_generation_token_total_producer_reported': counts['teacher_generated_tokens'],
              'input_files': {name: {'path': str(run / name), 'sha256': sha(run / name)} for name in names},
              'private_PT_or_WO_rehashed': False, 'independent_remote_execution_or_source_fetch_verified': False,
              'task_quality_or_deployment_win_claim': False, 'no_remote_GPU_or_private_PT_access': True,
              'summary_tool_sha256': sha(__file__)}
    write(output / 'summary.json', result)
    write(output / 'receipt.json', {'summary_sha256': sha(output / 'summary.json'),
        'tool_sha256': sha(__file__), 'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.run, a.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'].startswith('PASS_') else 3


if __name__ == '__main__':
    raise SystemExit(main())
