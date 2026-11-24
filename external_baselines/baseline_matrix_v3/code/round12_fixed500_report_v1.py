"""CPU-only fixed500 reference pack and exact-coverage comparison report.

Historical reference rows are not new inference. Every admitted baseline is
bound to the identical source/question/image/model/prompt/position contract
and rescored with the frozen current metric. Incomplete500 never gets a full
primary mean and no successful subset is substituted for its denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import round12_fixed500_inputs_v1 as data
import round12_report_v1 as paired_report
from metrics import score_question, METRIC_CONTRACT_VERSION

require, read, rows, sha, write_new = data.require, data.read, data.rows, data.sha, data.write_new
RESOURCE_FIELDS = ('complete_visual_owner_bytes', 'stored_visual_KV_bytes', 'actual_effective_visual_bits',
    'full_kv_byte_denominator', 'encode_ms', 'wall_ms', 'ttft_ms', 'peak_allocated_bytes', 'peak_reserved_bytes')
EXTRA_LABELS = {'K2_g32R32': 'KIVI_source_example_g32R32',
    'C2_bias32': 'CalibQuant_FP32Affine_Text32_v2_port',
    'OT_Hmix23': 'OnlyTerp_MSE_QJL0_FP32_reference', 'K2_R0': 'KIVI_R0'}
EXTRA_KEYS = tuple(EXTRA_LABELS)
CORE = ('FP16', 'R7low', 'R7high')


def checked_scored(directory, expected_input_sha=None):
    directory = Path(directory)
    summary = read(directory / 'summary.json')
    require(summary.get('schema') == 'baseline_matrix.offline_score.v1'
            and summary.get('metric_contract_version') == METRIC_CONTRACT_VERSION
            and summary['integrity']['status'] == 'PASS_ALL_SHA_AND_EXPECTED_ID_CHECKS'
            and summary['integrity']['fixture_only'] is False, 'score source is not genuine frozen-contract output')
    require(sha(directory / 'scored_records.jsonl') == summary['derived_file_hashes']['scored_records.jsonl'], 'scored rows changed')
    if expected_input_sha is not None:
        require(summary['integrity']['inputs_sha256'] == expected_input_sha, 'scored source uses different fixed inputs')
    values = rows(directory / 'scored_records.jsonl')
    require(len(values) == summary['expected_run_questions'] and len({r['id'] for r in values}) == len(values), 'scored coverage/ID differs')
    return summary, values


def verify_input_record(raw, expected):
    mismatches = {key: (expected[key], raw.get(key)) for key in data.NATIVE_FIELDS if raw.get(key) != expected[key]}
    for key in ('source_id', 'source_cluster_id', 'question_id', 'question_sha256', 'dataset', 'dataset_revision', 'split', 'preprocessing'):
        if raw.get(key) != expected[key]:
            mismatches[key] = (expected[key], raw.get(key))
    require(not mismatches, 'baseline/candidate INPUT mismatch: ' + raw['id'] + ' ' + json.dumps(mismatches))


def load_baseline(label, directory, dataset, manifest, registry_root, index_entry=None):
    filename = dataset + '_p2_inputs.jsonl'
    input_entry = manifest['input_files'][filename]
    summary, scored = checked_scored(directory, input_entry['sha256'])
    require(summary['dataset'] == dataset and summary['model_id'] == 'M1' and summary['expected_run_questions'] == 500,
            'baseline is not complete M1 fixed500')
    require(summary['status'] == 'PASS_SCORED_COMPLETE', 'baseline has failures, not full500 quality')
    raw_dir = Path(summary['integrity']['raw_dir'])
    for name, digest in summary['integrity']['raw_files'].items():
        require(sha(raw_dir / name) == digest, 'old baseline raw triplet changed: ' + label + '/' + name)
    if index_entry is not None:
        require(sha(Path(directory) / 'summary.json') == index_entry['score_summary_sha256']
                and summary['integrity']['raw_files'] == index_entry['raw_file_sha256'], 'competitive index identity differs')
    references = {r['id']: r for r in rows(Path(registry_root) / (dataset + '_p2_refs.jsonl'))}
    raw_values = rows(raw_dir / 'records.jsonl')
    raw_by_id = {r['id']: r for r in raw_values}
    expected_ids = input_entry['ids']
    require(len(raw_by_id) == 500 and set(raw_by_id) == set(expected_ids)
            and {r['id'] for r in scored} == set(expected_ids), 'baseline IDs are not original fixed500')
    result = {}
    for row in scored:
        raw = raw_by_id[row['id']]
        verify_input_record(raw, manifest['native_bindings'][row['id']])
        require(raw['status'] == 'PASS' and row['score_status'] == 'SCORED_PASS', 'baseline has failed questions')
        metric = score_question(dataset, raw['raw_prediction'], references[row['id']])
        require(metric['metric_id'] == data.PRIMARY_METRIC_IDS[dataset] == row['metric_id']
                and float(metric['score']) == float(row['score']), 'baseline score differs under current metric')
        value = {k: row.get(k) for k in ('id', 'source_cluster_id', 'score', 'score_status', 'metric_id')}
        value.update({k: raw.get(k) for k in RESOURCE_FIELDS})
        result[row['id']] = value
    return {'label': label, 'dataset': dataset, 'rows': result, 'original_inference_count_for_this_stage': 0,
        'scored_dir': str(directory), 'summary_sha256': sha(Path(directory) / 'summary.json'),
        'scored_rows_sha256': sha(Path(directory) / 'scored_records.jsonl'), 'raw_hashes': summary['integrity']['raw_files'],
        'input_sha256': input_entry['sha256'], 'metric_id': data.PRIMARY_METRIC_IDS[dataset],
        'all500_INPUT_metadata_verified': True, 'current_metric_recomputed_all500': True}


def prepare_reference_pack(output, manifest_path, registry_root, r7_root, fp16_score_root, source_index_path, include_extras=True):
    output = Path(output)
    manifest = read(manifest_path)
    registry_root = Path(registry_root)
    source_lock = read(registry_root / 'source_lock.json')
    require(sha(registry_root / 'source_lock.json') == manifest['source_lock_sha256'], 'original fixed data source lock changed')
    for domain in ('infovqa', 'docvqa'):
        for suffix in ('inputs', 'refs'):
            name = domain + '_p2_' + suffix + '.jsonl'
            p = registry_root / name
            require(sha(p) == source_lock['files'][name]['sha256']
                    and p.stat().st_size == source_lock['files'][name]['bytes'], 'frozen fixed data changed: ' + name)
    pack_path, receipt_path = output / 'REFERENCE_PACK_v1.json', output / 'REFERENCE_PACK_RECEIPT_v1.json'
    if pack_path.exists():
        receipt = read(receipt_path)
        require(sha(pack_path) == receipt['pack_sha256'] and receipt['metadata_manifest_sha256'] == sha(manifest_path), 'existing reference pack changed')
        require(receipt['source_index_sha256'] == sha(source_index_path), 'competitive source index changed; prepare a new pack version')
        return pack_path
    output.mkdir(parents=True, exist_ok=False)
    index = read(source_index_path)
    pack = {'schema': 'trajkv.round12.fixed500_reference_pack.v1', 'references': {}, 'unavailable': [],
            'metadata_manifest_sha256': sha(manifest_path), 'source_index_sha256': sha(source_index_path),
            'historical_reference_inference_count_for_current_stage': 0,
            'all_scores_recomputed_current_frozen_metric': True,
            'comparison_is_previously_observed_fixed500': True}
    for dataset in ('infovqa', 'docvqa'):
        sources = {
            'FP16': Path(fp16_score_root) / ('M1_F_' + dataset + '_p2_500_v1'),
            'R7low': Path(r7_root) / 'scored' / ('K1_' + dataset + '_fixed500'),
            'R7high': Path(r7_root) / 'scored' / ('K2_' + dataset + '_fixed500'),
        }
        pack['references'][dataset] = {}
        for label, directory in sources.items():
            pack['references'][dataset][label] = load_baseline(label, directory, dataset, manifest, registry_root)
        if include_extras:
            for key in EXTRA_KEYS:
                candidates = [c for c in index['cells'] if c['model'] == 'M1' and c['dataset'] == dataset and c['method_key'] == key]
                if len(candidates) != 1 or candidates[0]['status'] != 'COMPLETE_FIXED500':
                    pack['unavailable'].append({'dataset': dataset, 'method_key': key,
                        'status': candidates[0]['status'] if len(candidates) == 1 else 'NOT_UNIQUELY_INDEXED'})
                    continue
                cell = candidates[0]
                try:
                    label = EXTRA_LABELS[key]
                    pack['references'][dataset][label] = load_baseline(label, cell['scored'], dataset, manifest, registry_root, cell)
                except Exception as exc:
                    pack['unavailable'].append({'dataset': dataset, 'method_key': key,
                        'status': 'NOT_ADMITTED_TO_MATCHED_TABLE', 'error': str(exc)})
    write_new(pack_path, pack)
    write_new(receipt_path, {'status': 'PASS_MATCHED_FIXED500_REFERENCE_PACK', 'pack_sha256': sha(pack_path),
        'metadata_manifest_sha256': sha(manifest_path), 'source_index_sha256': sha(source_index_path),
        'new_model_or_GPU_calls': 0, 'reference_counts': {d: len(v) for d, v in pack['references'].items()}})
    return pack_path


def resources(values):
    result = {}
    for field in RESOURCE_FIELDS:
        nums = [float(r[field]) for r in values if isinstance(r.get(field), (int, float))]
        result[field] = {'n': len(nums), 'complete_coverage': len(nums) == len(values),
                         'mean': statistics.mean(nums) if len(nums) == len(values) else None,
                         'min': min(nums) if len(nums) == len(values) else None,
                         'max': max(nums) if len(nums) == len(values) else None} if nums else None
    if values and all(isinstance(r.get('complete_visual_owner_bytes'), (int, float))
                      and isinstance(r.get('full_kv_byte_denominator'), (int, float)) for r in values):
        total_b = sum(r['complete_visual_owner_bytes'] for r in values)
        total_f = sum(r['full_kv_byte_denominator'] for r in values)
        result['element_weighted_effective_bits'] = 16 * total_b / total_f
        result['strict_all_rows_1p3'] = all(160 * r['complete_visual_owner_bytes'] <= 13 * r['full_kv_byte_denominator'] for r in values)
        result['strict_all_rows_1p8'] = all(80 * r['complete_visual_owner_bytes'] <= 9 * r['full_kv_byte_denominator'] for r in values)
    return result


def build_report(done_cells, manifest_path, reference_pack_path, output, *, candidate_name='TrajKV_R12_R_G8matched'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest, pack = read(manifest_path), read(reference_pack_path)
    require(pack['metadata_manifest_sha256'] == sha(manifest_path), 'reference pack uses different fixed inputs')
    grouped = defaultdict(dict)
    cell_proofs, physical_answers = [], 0
    for cell in done_cells:
        directory, raw_dir = Path(cell['scored_dir']), Path(cell['local_raw'])
        summary, scored = checked_scored(directory)
        require(all(sha(raw_dir / name) == digest for name, digest in summary['integrity']['raw_files'].items()),
                'new candidate raw changed after scoring')
        raw = rows(raw_dir / 'records.jsonl')
        require(len(raw) == len(scored) == cell['expected_QA'], 'new shard raw/scored coverage differs')
        score_by_id = {r['id']: r for r in scored}
        dataset = cell['domain']
        for item in raw:
            expected_input = manifest['native_bindings'][item['id']]
            for field in ('dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id', 'question_id',
                          'question_sha256', 'original_image_sha256'):
                require(item.get(field) == expected_input[field], 'failed/PASS row source identity differs')
            if item['status'] == 'PASS':
                verify_input_record(item, expected_input)
            require(item['comparison_role'] == data.ROLE and item['independent_holdout_claimed'] is False,
                    'new raw mislabels fixed comparison as another role')
            require(item['id'] not in grouped[dataset], 'duplicate candidate ID across shards')
            value = {**{k: item.get(k) for k in RESOURCE_FIELDS}, **score_by_id[item['id']]}
            grouped[dataset][item['id']] = value
            physical_answers += bool(item.get('output_token_ids') or item.get('untrusted_output_token_ids'))
        cell_proofs.append({'cell': cell['name'], 'dataset': dataset, 'questions': len(raw),
            'raw_hashes': summary['integrity']['raw_files'], 'scored_rows_sha256': sha(directory / 'scored_records.jsonl')})
    tables, pairwise = [], []
    for dataset in ('infovqa', 'docvqa'):
        candidate = grouped[dataset]
        expected = manifest['input_files'][dataset + '_p2_inputs.jsonl']['ids']
        complete = len(candidate) == 500 and set(candidate) == set(expected)
        scored_ok = all(r['score_status'] == 'SCORED_PASS' for r in candidate.values())
        candidate_mean = statistics.mean(r['score'] for r in candidate.values()) if candidate and scored_ok else None
        base_row = {'dataset': dataset, 'method': candidate_name, 'expected_questions': 500,
            'observed_completed_questions': len(candidate), 'complete_fixed500_coverage': complete,
            'primary_full500_mean': candidate_mean if complete else None,
            'partial_completed_cohort_mean': candidate_mean if not complete else None,
            'no_success_subset_primary': True, 'score_status_counts': dict(Counter(r['score_status'] for r in candidate.values())),
            'resources': resources(list(candidate.values())) if candidate else None,
            'new_physical_inference_not_reference_reuse': True}
        tables.append(base_row)
        for label, ref in pack['references'][dataset].items():
            reference = ref['rows']
            ref_mean = statistics.mean(r['score'] for r in reference.values())
            tables.append({'dataset': dataset, 'method': label, 'expected_questions': 500,
                'observed_completed_questions': 500, 'complete_fixed500_coverage': True,
                'primary_full500_mean': ref_mean, 'resources': resources(list(reference.values())),
                'new_physical_inference_not_reference_reuse': False, 'historical_reference_only': True})
            if candidate:
                result = paired_report.paired(candidate, reference, repetitions=2000, seed=20261124)
                pairwise.append({'dataset': dataset, 'candidate': base_row['method'], 'reference': label,
                    **result, 'status': 'DESCRIPTIVE_FIXED500_COMPARISON_NOT_INDEPENDENT',
                    'comparison_scope': 'COMPLETE_FIXED500' if complete else 'PARTIAL_COMPLETED_SHARDS_NOT_FULL500',
                    'independent_holdout_claimed': False})
    all_covered = all(len(grouped[d]) == 500 for d in ('infovqa', 'docvqa'))
    all_scored = all(r['score_status'] == 'SCORED_PASS' for values in grouped.values() for r in values.values())
    result = {'schema': 'trajkv.round12.fixed500_result.v1',
        'status': ('COMPLETE_MATCHED1000' if all_scored else 'COMPLETE1000_WITH_FAILURES') if all_covered else 'PARTIAL_FIXED500_COMPARISON',
        'planned_new_answers': 1000, 'new_raw_recorded_tasks': sum(len(v) for v in grouped.values()),
        'new_physical_terminal_answers_present': physical_answers, 'historical_references_counted_as_new_inference': 0,
        'answer_count_scope': 'completed independently retrieved shards only; running records are separate provisional progress',
        'tables': tables, 'paired': pairwise, 'new_cell_proofs': cell_proofs,
        'reference_pack_sha256': sha(reference_pack_path), 'input_metadata_manifest_sha256': sha(manifest_path),
        'unavailable_reference_methods': pack['unavailable'], 'previously_observed_fixed500': True,
        'external_reference_scope': 'completed matched M1 ports in frozen sourceIndex, not a comprehensive global SOTA catalogue',
        'independent_confirmation_claimed': False, 'no_success_subset_full500_mean': True,
        'reporter_source_sha256': sha(__file__), 'paired_helper_source_sha256': sha(paired_report.__file__),
        'timing_scope': 'instrumented new quality runs and historical prototype timings, not controlled speed comparison'}
    write_new(output / 'summary.json', result)
    with (output / 'quality_bytes.csv').open('x', encoding='utf-8', newline='') as s:
        fields = ['dataset', 'method', 'expected_questions', 'observed_completed_questions', 'complete_fixed500_coverage',
                  'primary_full500_mean', 'partial_completed_cohort_mean', 'mean_visual_bytes', 'mean_effective_bits', 'max_effective_bits', 'historical_reference_only']
        w = csv.DictWriter(s, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for row in tables:
            res = row.get('resources') or {}
            w.writerow({**row, 'mean_visual_bytes': (res.get('complete_visual_owner_bytes') or {}).get('mean'),
                'mean_effective_bits': (res.get('actual_effective_visual_bits') or {}).get('mean'),
                'max_effective_bits': (res.get('actual_effective_visual_bits') or {}).get('max')})
    write_new(output / 'paired.json', pairwise)
    return result
