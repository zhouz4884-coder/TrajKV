"""CPU targets from a COMPLETE declared quality gate; no inference/publication.

Consumes frozen scored caches, complete physical-byte/paired-F receipts and an
optional independently closed speed cohort. Never scans raw prediction banks.
An incomplete114/20 scope produces WAIT with no quality targets or rankings.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from public_phase_contract_v2 import ROOT, EVIDENCE, load, sha, require, resolve
import prepare_remaining_phase_delivery_v1 as gate
sys.path.insert(0, str(ROOT/'code'))
from compare_runs import compare, load_scored

BUDGETS = {'1.3': (13, 10), '1.8': (9, 5)}
GATE_SHA = '621b9a3bf29811e3b11292397c1248b19046154a8924dee87cd2fa1ae10c0650'
OBJECTIVE_DOC_SHA = '50e61c11c75b62e7086e2cf19387161815212cb20363379578927a1e58fdd0fc'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def speed_evidence(directory):
    if directory is None: return {'status': 'NO_CONTROLLED_SPEED_INPUT', 'rows': []}
    directory = resolve(directory); ready = load(directory/'readiness.json')
    require(ready['status'] == 'PASS_CANONICAL_560_AND_ORIGINAL_DEPENDENCY_CLOSURE_CPU' and ready['actual_PASS_trials'] == 560,
            'Only actually closed canonical560 speed evidence is supported')
    inventory = {str(resolve(row['path'])): row['sha256'] for row in load(directory/'timing_support.json')['files']}
    for name in ('domain_timing.json', 'timing_expected_origin_binding.json'):
        require(inventory.get(str(directory/name)) == sha(directory/name), 'Timing scalar/binding receipt changed')
    bindings = load(directory/'timing_expected_origin_binding.json')['rows']; contexts = {}
    for definition in ready['definitions']:
        job = load(ROOT/definition['job']); require(sha(ROOT/definition['job']) == definition['job_sha256'], 'Timing job changed')
        remote = job['argv'][job['argv'].index('--output')+1]
        expected = EVIDENCE/'results'/remote.replace('\\', '/').split('/results/', 1)[1]/'expected_predictions.json'
        require(inventory.get(str(expected)) == sha(expected), 'Timing expected-model evidence changed')
        for row in bindings:
            if resolve(row['expected_file']) != expected: continue
            key = (row['model_id'], definition['dataset'], definition['step'], row['method'])
            require(key not in contexts or contexts[key] == row['effective_contract_sha256'], 'Mixed effective speed contracts')
            contexts[key] = row['effective_contract_sha256']
    rows = []
    for row in load(directory/'domain_timing.json')['rows']:
        key = ('M1', row['dataset'], row['step'], row['method'])
        require(key in contexts, 'Speed scalar lacks exact expected-source contract')
        rows.append(dict(row, model='M1', effective_contract_sha256=contexts[key]))
    return {'status': 'PASS_SEPARATE_CANONICAL560_CONTROLLED_COHORT', 'raw_trials': 560,
            'host_quiet_asserted': False, 'source_readiness_sha256': sha(directory/'readiness.json'),
            'required_speed_support_file_list': str(directory/'timing_support.json'),
            'required_speed_support_file_list_sha256': sha(directory/'timing_support.json'),
            'not_low_budget_speed_by_family_name': True, 'no_cross_dataset_pooling': True,
            'not_head_to_head_with_future_TrajKV': True,
            'full_trial_and_execution_audit_reused_from_frozen_complete_gate': True, 'rows': rows}


def verified_points(directory, matrix):
    support = load(directory/'quality_support.json')
    require(support['schema'] == 'baseline_matrix.public_support_files.v1' and support['completed_evidence_only'] is True and
            support.get('quality_QA_added', 0) == 0, 'Expected completed derived-analysis support inventory')
    inventory = {str(resolve(r['path'])): r['sha256'] for r in support['files']}
    for name in ('quality_bytes.json', 'source_closure.json'):
        require(inventory.get(str(directory/name)) == sha(directory/name), 'Complete quality/physical closure receipt changed')
    values = load(directory/'quality_bytes.json')['rows']; by_cell = {r['cell']: r for r in values}
    require(len(by_cell) == len(matrix['rows']), 'Target quality row count differs')
    points = []
    for target in matrix['rows']:
        cell = Path(target['config']).stem; point = by_cell[cell]
        require((point['model'], point['dataset'], point['method'], point['QA']) ==
                (target['model'], target['dataset'], target['method'], target['count']), 'Quality target identity differs')
        require(resolve(point['raw']) == resolve(target['view']) and resolve(point['scored']) == resolve(target['scored']), 'Quality view/score mapping differs')
        scored = resolve(point['scored']); summary, records = load_scored(scored)
        require(sha(scored/'summary.json') == point['summary_sha256'], 'Scored source changed')
        require(summary['status'] == 'PASS_SCORED_COMPLETE' and summary['expected_run_questions'] == len(records) == point['QA'], 'Incomplete scored cohort')
        require(summary['primary_mean'] == point['quality'] and summary['primary_ci95'] == point['quality_CI95'], 'Source quality/CI differs')
        require(summary['metric_ids'] == point['metric_ids'] and point['no_budget_or_quality_subset_selection'] is True, 'Metric/subset contract differs')
        manifest_path = resolve(point['raw'])/'manifest.json'
        require(sha(manifest_path) == summary['integrity']['raw_files']['manifest.json'], 'Frozen physical manifest changed')
        manifest = load(manifest_path); spec = manifest['model_spec']
        bits = []; total_bytes = 0; total_denominator = 0; over = {b: 0 for b in BUDGETS}
        for row in records.values():
            require(row['score_status'] == 'SCORED_PASS' and row['score'] is not None, 'Do not filter failed questions for a quality target')
            denominator = spec['layers']*2*spec['kv_heads']*row['actual_visual_token_count']*spec['head_dim']*2
            actual_bytes = row['stored_visual_KV_bytes']; bit = 16*actual_bytes/denominator
            require(actual_bytes > 0 and abs(bit-row['actual_effective_visual_bits']) < 1e-12, 'Scored bytes/geometry/bit ratio differs')
            bits.append(bit); total_bytes += actual_bytes; total_denominator += denominator
            for budget, (numerator, divisor) in BUDGETS.items():
                over[budget] += 16*actual_bytes*divisor > numerator*denominator
        require(abs(total_bytes/len(bits)-point['complete_visual_owner_mean_bytes']) < 1e-8 and
                abs(max(bits)-point['actual_effective_bits_max']) < 1e-12, 'Complete physical audit/scored resource summary differs')
        require(all(point['all_QA_iso_budget'][budget] == (count == 0) for budget, count in over.items()), 'Strict all-request budget differs')
        paired = resolve(point['paired_F']); require(sha(paired) == point['paired_F_sha256'] and
                inventory.get(str(paired)) == sha(paired), 'Paired-F source changed')
        pair = load(paired); require(pair['status'] == 'PASS_PAIRED_COMPLETE', 'Paired-F cohort incomplete')
        config = load(ROOT/target['config'])
        require(manifest['config'] == config and config['method']['information_condition'] == 'query_unknown', 'Input/information protocol differs')
        require(pair['candidate_summary_sha256'] == point['summary_sha256'] and
                pair['candidate_scored_rows_sha256'] == summary['derived_file_hashes']['scored_records.jsonl'], 'Paired-F candidate source differs')
        points.append(dict(point, family=config['method']['family'], source_method_id=summary['method'],
            config=target['config'], config_source_sha256=target['config_sha256'], method_config=config['method'],
            native_preprocessing=config['preprocessing'],
            predeclared_five_profile_representative=target['method'] in {'F', 'K2_g32R32', 'C2_bias32', 'E_r0888', 'OT_Hmix23'},
            effective_contract_sha256=manifest.get('effective_contract_sha256'), scored=str(scored),
            exact_speed_contract_fingerprint_available=bool(manifest.get('effective_contract_sha256')),
            ratio_of_sums_actual_bits=16*total_bytes/total_denominator, over_budget_QA=over,
            actual_effective_bits_mean=sum(bits)/len(bits), actual_effective_bits_min=min(bits),
            actual_effective_bits_max=max(bits), paired_F_primary=pair['primary']))
    return points


def derive_targets(points, out, speed):
    """Pure declared-point analysis; caller enforces complete114/20 admission."""
    groups = defaultdict(list); targets = []; pair_cache = {}
    for point in points: groups[(point['model'], point['dataset'], tuple(point['metric_ids']))].append(point)
    def pair(anchor, candidate):
        key = (anchor['cell'], candidate['cell'])
        if key not in pair_cache:
            value = compare(Path(anchor['scored']), Path(candidate['scored']))
            require(value['status'] == 'PASS_PAIRED_COMPLETE', 'Different/incomplete cohort cannot define a target gap')
            filename = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]+'.json'
            path = out/'paired_targets'/filename; write(path, value)
            pair_cache[key] = {'path': str(path), 'sha256': sha(path), 'primary': value['primary']}
        return pair_cache[key]
    for (model, dataset, metric_ids), group in groups.items():
        native = next(p for p in group if p['method'] == 'F')
        for point in group:
            source_pair = load(resolve(point['paired_F']))
            require(source_pair['anchor_summary_sha256'] == native['summary_sha256'], 'Paired-F anchor uses another native cohort')
            require(abs(source_pair['primary']['difference_pp']-100*(point['quality']-native['quality'])) < 1e-10, 'Paired native difference differs from full-cohort means')
        strongest_any = sorted(group, key=lambda p: (-p['quality'], p['cell']))[0]
        families = sorted({p['family'] for p in group})
        for budget in BUDGETS:
            eligible = [p for p in group if p['over_budget_QA'][budget] == 0]
            row = {'model': model, 'dataset': dataset, 'metric_ids': list(metric_ids), 'budget_actual_bits': float(budget),
                   'declared_points': len(group), 'eligible_complete_cohort_points': len(eligible),
                   'ineligible_points_keep_their_entire_cohort_quality': True,
                   'status': 'NO_MEASURED_POINT_AT_STRICT_BUDGET' if not eligible else 'MEASURED_REFERENCE_TARGET',
                   'best_observed_budget_quality': None, 'TrajKV_result': None, 'TrajKV_leading_assessed': False,
                   'speed_for_budget_target_status': 'NO_QUALIFIED_QUALITY_POINT',
                   'family_references': []}
            for family in families:
                all_family = [p for p in group if p['family'] == family]
                low_family = [p for p in eligible if p['family'] == family]
                high = sorted(all_family, key=lambda p: (-p['quality'], p['cell']))[0]
                low = sorted(low_family, key=lambda p: (-p['quality'], p['cell']))[0] if low_family else None
                row['family_references'].append({'family': family, 'best_eligible_cell': low['cell'] if low else None,
                    'all_declared_family_points': [p['cell'] for p in all_family],
                    'predeclared_five_profile_cells': [p['cell'] for p in all_family if p.get('predeclared_five_profile_representative')],
                    'best_eligible_quality': low['quality'] if low else None,
                    'best_measured_any_bytes_cell': high['cell'], 'best_measured_any_bytes_quality': high['quality'],
                    'best_any_bytes_is_budget_eligible': high['over_budget_QA'][budget] == 0,
                    'source_parameters_and_execution_scope_unchanged': True})
            if eligible:
                best = sorted(eligible, key=lambda p: (-p['quality'], p['cell']))[0]
                tied = [p['cell'] for p in eligible if abs(p['quality']-best['quality']) < 1e-12]
                native_pair = best['paired_F_primary']; require(native_pair is not None, 'Native gap unavailable')
                row.update(best_observed_budget_quality=best['quality'], target_quality_percent=100*best['quality'],
                    target_cell=best['cell'], descriptive_tied_best_cells=tied,
                    native_quality_percent=100*native['quality'],
                    native_minus_budget_target_pp=-native_pair['difference_pp'],
                    native_minus_budget_target_paired_CI95_pp=([-native_pair['ci95_pp'][1], -native_pair['ci95_pp'][0]] if native_pair['ci95_pp'] else None),
                    native_pair_reused_from_complete_gate={'path': best['paired_F'], 'sha256': best['paired_F_sha256']},
                    native_paired_precision={'source_cluster_count': native_pair['cluster_count'],
                        'pilot_MDE80_normal_approx_pp': native_pair['pilot_MDE80_normal_approx_pp'],
                        'MDE_is_diagnostic_not_a_guaranteed_margin': True},
                    strongest_any_bytes_cell=strongest_any['cell'], strongest_any_bytes_quality_percent=100*strongest_any['quality'],
                    strongest_any_bytes_minus_budget_target=pair(best, strongest_any),
                    reference_gap_is_not_an_attainability_bound=True,
                    target_ChartQA_subgroups={k: v for k, v in best.get('groups', {}).items() if k in ('human', 'aug')},
                    challengers=[{'cell': p['cell'], 'quality_percent': 100*p['quality'],
                        'quality_gain_to_observed_budget_reference': pair(p, best)} for p in eligible],
                    exact_profile_formal_speed=[r for r in speed['rows'] if r['step'] == 'formal_fixed_cohort' and
                        r['model'] == model and r['dataset'] == dataset and r['method'] == best['source_method_id'] and
                        r['effective_contract_sha256'] == best['effective_contract_sha256']])
                row['speed_for_budget_target_status'] = ('MEASURED_EXACT_PROFILE_DIFFERENT_FIXED_COHORT' if row['exact_profile_formal_speed'] else 'NO_CONTROLLED_SPEED_FOR_THIS_EXACT_BUDGET_POINT')
                outside = [p for p in group if p['over_budget_QA'][budget] > 0]
                if outside:
                    reference = sorted(outside, key=lambda p: (-p['quality'], p['cell']))[0]
                    row['strongest_measured_ineligible_reference'] = {'cell': reference['cell'],
                        'quality_percent': 100*reference['quality'], 'reference_minus_budget_target': pair(best, reference),
                        'role': 'outside_strict_budget_reference_only_not_an_attainable_bound'}
            targets.append(row)
    return targets


def qwen_subset_table(points):
    methods = {'F', 'K2_g32R32', 'C2_bias32', 'E_r0888', 'OT_Hmix23'}
    rows = []
    for point in points:
        if point['model'] not in ('M2', 'M3') or point['dataset'] not in ('textvqa', 'infovqa', 'docvqa') or point['method'] not in methods: continue
        primary = point['paired_F_primary']; ci = primary['ci95_pp']
        rows.append({'model': point['model'], 'dataset': point['dataset'], 'method': point['method'],
            'source_method_id': point['source_method_id'], 'QA': point['QA'], 'metric_ids': point['metric_ids'],
            'quality_percent': 100*point['quality'], 'quality_CI95_percent': [100*x for x in point['quality_CI95']],
            'native_minus_method_pp': -primary['difference_pp'],
            'native_minus_method_paired_CI95_pp': [-ci[1], -ci[0]] if ci else None,
            'source_cluster_count': primary['cluster_count'], 'pilot_MDE80_normal_approx_pp': primary['pilot_MDE80_normal_approx_pp'],
            'actual_bits_mean': point['actual_effective_bits_mean'], 'actual_bits_max': point['actual_effective_bits_max'],
            'role': 'reused_fixed500_pilot_as_declared_formal_subset', 'independent_confirmation': False,
            'summary_sha256': point['summary_sha256'], 'effective_contract_sha256': point['effective_contract_sha256']})
    require(len(rows) == 30 and sum(row['QA'] for row in rows) == 15000, 'Complete P2 must contain exactly the required30 Qwen subset cells')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('p2', 'p3')); parser.add_argument('--quality-dir', type=Path, required=True)
    parser.add_argument('--speed-dir', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); directory = resolve(args.quality_dir); out = resolve(args.output)
    require(out.is_relative_to(EVIDENCE/'results/research_targets'), 'All new outputs must use E results/research_targets')
    require(sha(ROOT/'ops/prepare_remaining_phase_delivery_v1.py') == GATE_SHA, 'Complete physical gate source changed')
    require(sha(ROOT/'docs/INPUT_MATRIX_v3_3090.md') == OBJECTIVE_DOC_SHA, 'Original research objective contract changed')
    ready = load(directory/'readiness.json'); speed = speed_evidence(args.speed_dir)
    out.mkdir(parents=True, exist_ok=False)
    base = {'schema': 'baseline_matrix.research_targets.v1', 'phase': args.phase,
        'complete_quality_receipt': str(directory/'readiness.json'), 'complete_quality_receipt_sha256': sha(directory/'readiness.json'),
        'expected_cells': 114 if args.phase == 'p2' else 20,
        'observed_closed_marker_and_score_cells': ready.get('closed_marker_and_score_cells', ready.get('phase', {}).get('verified_scored_cells', 0)),
        'budgets_actual_complete_bits': [1.3, 1.8], 'whole_external_library_complete': False,
        'TrajKV_tested_or_leading': False, 'controlled_speed_is_separate': speed,
        'no_interpolation_or_budget_subset_scores': True, 'source_gate_sha256': GATE_SHA,
        'original_objective_document': 'TrajKV_SOTA_Matrix_v3_3090.md', 'original_objective_sha256': OBJECTIVE_DOC_SHA,
        'whole_goal_requirements': {'document': 'docs/EXTERNAL_LIBRARY_COMPLETION_REQUIREMENTS_v2.md',
            'sha256': sha(ROOT/'docs/EXTERNAL_LIBRARY_COMPLETION_REQUIREMENTS_v2.md')},
        'quality_scope': ('fixed500 pilot predictions reused as declared subsets; not independent confirmation' if args.phase == 'p2' else 'M1 canonical four-official-full-split scope only'),
        'required_Qwen400_speed_supplement': {'models': ['M2', 'M3'], 'profiles': ['F', 'K32', 'C2', 'E64', 'OnlyTerp-Hmix23'],
            'domain': 'TextVQA', 'declared_trials': 400, 'declared_warmup_trials': 100, 'declared_measured_trials': 300,
            'actual_trials_admitted_by_this_version': None, 'status': 'NOT_PROVIDED_OR_ADMITTED_BY_THIS_VERSION',
            'does_not_establish_Info_Doc_Chart_latency': True},
        'Qwen_formal_subset_rows': [], 'Qwen_subsets_independent_confirmation': False,
        'controlled_performance_whole_goal_completion_assessed': False,
        'no_remote_GPU_queue_or_publication_action': True, 'targets': []}
    if ready.get('status') != 'PASS_COMPLETE_DECLARED_QUALITY_PHASE_CPU' or ready.get('deep_evidence_verified') is not True:
        base.update(status='WAIT_RESEARCH_TARGETS_REQUIRE_COMPLETE_DECLARED_QUALITY', publication_eligible=False)
        write(out/'research_targets.json', base)
        (out/'RESEARCH_TARGETS.md').write_text('# 研发参照待完整质量闭合\n\n'+
            f"当前仅有{base['observed_closed_marker_and_score_cells']}/{base['expected_cells']}个marker/score闭合单元。尚未形成完整质量/物理字节验收产物，目标表与强对手排名保持为空。\n\n"+
            '完整后将分模型/域列出真实1.3/1.8-bit全cohort内实测强参照、各family实际点、FP16保真差距及较高字节质量参照的配对CI；这些是研发比较靶值，不是TrajKV可达性或领先结论。独立canonical560速度仍保留自己的profile/cohort，不能借给未测低bit点。\n', encoding='utf-8', newline='\n')
        print(json.dumps({k: base[k] for k in ('status', 'observed_closed_marker_and_score_cells', 'expected_cells')})); raise SystemExit(3)
    matrix = load(directory/'declared_matrix.json'); expected = gate.quality_matrix(args.phase)
    identity = lambda r: (r['model'], r['dataset'], r['method'], r['config'], r['config_sha256'], r['count'])
    require(len(matrix['rows']) == base['expected_cells'] and
            {identity(r) for r in matrix['rows']} == {identity(r) for r in expected['rows']}, 'Incomplete or different declared matrix')
    require(ready['phase']['declared_matrix_source_sha256'] == sha(directory/'declared_matrix.json') and
            ready['phase']['verified_scored_cells'] == base['expected_cells'], 'Complete gate matrix binding differs')
    points = verified_points(directory, matrix); targets = derive_targets(points, out, speed)
    qwen = qwen_subset_table(points) if args.phase == 'p2' else []
    base.update(status='PASS_COMPLETE_DECLARED_EMPIRICAL_RESEARCH_TARGETS', targets=targets,
        selection_rule='Descriptive strongest observed mean within complete declared legal configurations; no population superiority claim',
        paired_CI_rule='Unadjusted source-cluster paired intervals; selection and comparison families require correction before significance claims',
        target_rule='Observed reference values and measured gaps; unrestricted-best can itself be budget-eligible; no future-algorithm attainability assertion',
        native_is_not_a_mathematical_quality_upper_bound=True,
        physical_raw_audit_reused_from_complete_gate_not_rescanned=True)
    base.update(Qwen_formal_subset_rows=qwen, Qwen_subsets_independent_confirmation=False,
                controlled_performance_whole_goal_completion_assessed=False)
    write(out/'research_targets.json', base); write(out/'all_measured_points.json', {'rows': points})
    if qwen:
        with (out/'Qwen_reused_pilot_formal_subsets.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(qwen[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(qwen)
    fields = ['model', 'dataset', 'budget_actual_bits', 'status', 'eligible_complete_cohort_points', 'target_cell',
              'target_quality_percent', 'native_quality_percent', 'native_minus_budget_target_pp', 'native_minus_budget_target_paired_CI95_pp', 'speed_for_budget_target_status']
    with (out/'research_targets.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in targets)
    lines = ['# 完整声明范围的研发实测参照', '',
        '各域及评分指标分别报告。表中参照是完整预定配置集合内的描述性最高实测均值；配对CI未作选择/多重比较校正，不把CI跨零称等价。预算要求每个请求的完整owner均不超限，不筛题、不插值，也不据此更改全域固定的canonical源参数。TrajKV尚未在本入口测试，其可达性和领先状态均未知。', '',
        '| model | dataset | budget | eligible points | observed reference | quality % | native minus reference pp | paired CI95 pp |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: | --- |']
    for row in targets:
        fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
        lines.append(f"| {row['model']} | {row['dataset']} | {row['budget_actual_bits']} | {row['eligible_complete_cohort_points']} | {row.get('target_cell', 'NO_MEASURED_POINT')} | {fmt(row.get('target_quality_percent'))} | {fmt(row.get('native_minus_budget_target_pp'))} | {row.get('native_minus_budget_target_paired_CI95_pp', 'NA')} |")
    lines += ['', 'research_targets.json列出各family所有实际点与预定五代表、预算内描述性强参照、预算内各点至参照的成对差、不限预算质量参照与明确超严格预算参照的差距。不限预算最优点也可能本身已满足预算，参照差距不是低预算可达上界。canonical速度只按精确profile关联，保留Text16/Info8、自然tokens、TTFT/ITL和host_quiet=false；其它profile/域明确缺测。', '',
              ('Qwen30格Text/Info/Doc固定500正式subset表复用P2 pilot预测，包含source-cluster CI、paired-F loss、cluster_count与原pilot MDE80诊断，不能称独立确认。' if qwen else
               'Qwen30格固定500正式subset表不在本M1 full输入中，须由完整P2另行产出。')+
              '所需Qwen五代表Text新增400受控trials在此版本尚未接收实测产物，单列为缺口；不借M1速度。', '',
              'F也是实测参照，不是数学质量上界；量化点高于F时保留负的native loss。此报告仅完成所声明的质量范围分析，不代表整个外部库或P4方法评测完成。']
    (out/'RESEARCH_TARGETS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    supporting = [out/name for name in ('research_targets.json', 'all_measured_points.json', 'research_targets.csv', 'RESEARCH_TARGETS.md')]
    if qwen: supporting.append(out/'Qwen_reused_pilot_formal_subsets.csv')
    supporting += sorted((out/'paired_targets').glob('*.json'))+[Path(__file__)]
    write(out/'research_targets_support.json', {'schema': 'baseline_matrix.public_support_files.v1',
        'label': args.phase+'_complete_declared_research_targets', 'completed_evidence_only': True,
        'primary_quality_QA_added': 0, 'quality_QA_added': 0,
        'scope': 'Complete-declared empirical budget targets and paired quality gaps; no new inference, quality QA, significance/attainability or TrajKV-leading claim. Controlled speed retains exact source-profile and separate cohort.',
        'required_existing_speed_support_file_list': speed.get('required_speed_support_file_list'),
        'files': [{'path': str(path.resolve()), 'sha256': sha(path), 'role': 'research_target_CPU_analysis'} for path in supporting]})
    print(json.dumps({'status': base['status'], 'scope_cells': len(points), 'domain_budget_targets': len(targets), 'TrajKV_tested': False}))


if __name__ == '__main__': main()
