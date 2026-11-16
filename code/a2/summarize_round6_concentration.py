"""Summarize existing scalar C/Ctilde diagnostics; no new model/score/ranking."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELD_C = 'C_sum_visual_a2'
FIELD_CT = 'Ctilde_sum_visual_a2_value_minus_total_o_norm2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def definition(path):
    source = Path(path).read_text(encoding='utf-8')
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'concentration_diagnostics')
    return ast.get_source_segment(source, node), ast.dump(node, include_attributes=False)


def describe(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    valid = values[finite]
    return dict(count=len(values), finite_count=int(finite.sum()), nonfinite_count=int((~finite).sum()),
        negative_count=int((valid < 0).sum()), zero_count=int((valid == 0).sum()),
        minimum=float(valid.min()), maximum=float(valid.max()),
        median=float(np.median(valid)), q01=float(np.quantile(valid, .01)),
        q25=float(np.quantile(valid, .25)), q75=float(np.quantile(valid, .75)),
        q99=float(np.quantile(valid, .99)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'results/round6_v1/student/calibration16')
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'results/round6_v1/receipts/round6_student_v1.source/code/a2/round6_student_directional.py')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round6_v1/student/concentration_summary_v1.json')
    parser.add_argument('--doc', type=Path, default=ROOT / 'docs/ROUND6_CONCENTRATION_DIAGNOSTIC_20261115.zh.md')
    args = parser.parse_args()
    manifest_path, cases_path = args.input / 'manifest.json', args.input / 'cases.jsonl'
    manifest = load(manifest_path)
    if load(args.input / 'complete.json')['status'] != 'COMPLETE':
        raise ValueError('Calibration run is not complete')
    if sha(args.snapshot) != manifest['sources']['round6_student_directional.py']:
        raise ValueError('Executed collector source does not match recorded run SHA')
    current = ROOT / 'code/a2/round6_student_directional.py'
    snapshot_definition, snapshot_ast = definition(args.snapshot)
    current_definition, current_ast = definition(current)
    if snapshot_definition != current_definition or snapshot_ast != current_ast:
        raise ValueError('Current and actual-run concentration definitions differ')
    cases = [json.loads(line) for line in cases_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(cases) != len({row['id'] for row in cases}) or [row['id'] for row in cases] != manifest['selected_case_ids']:
        raise ValueError('Calibration case membership differs')
    points, sampled = [], []
    for row in cases:
        diagnostics = row.get('C_Ctilde_diagnostics', [])
        if not diagnostics:
            continue
        layer_ids = [int(layer['layer']) for layer in diagnostics]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError('Repeated layer diagnostic')
        heads, queries, batches = set(), set(), set()
        for layer in diagnostics:
            c, ct, mass = [np.asarray(layer[field], dtype=float) for field in (FIELD_C, FIELD_CT, 'visual_mass')]
            if c.ndim != 3 or c.shape != ct.shape or c.shape != mass.shape:
                raise ValueError('Expected identical [batch,Q-head,teacher-answer-query] scalar arrays')
            for batch, head, query in np.ndindex(c.shape):
                points.append(dict(case_id=row['id'], layer=int(layer['layer']), batch=batch, q_head=head,
                    teacher_query_position=query, N=int(row['visual_tokens']),
                    C=float(c[batch, head, query]), Ctilde=float(ct[batch, head, query]),
                    visual_mass=float(mass[batch, head, query])))
                heads.add(head); queries.add(query); batches.add(batch)
        sampled.append(dict(id=row['id'], image_id=row['image_id'], visual_tokens=int(row['visual_tokens']),
            layers=len(layer_ids), layer_ids=sorted(layer_ids), q_heads=len(heads),
            teacher_answer_queries=len(queries), batches=len(batches),
            teacher_answer_token_ids=row['teacher_token_ids']))
    if not points:
        raise ValueError('No existing C/Ctilde observations')
    c = np.array([row['C'] for row in points])
    ct = np.array([row['Ctilde'] for row in points])
    mass = np.array([row['visual_mass'] for row in points])
    ns = np.array([row['N'] for row in points])
    finite = np.isfinite(c) & np.isfinite(ct) & np.isfinite(mass)
    if not finite.all():
        raise ValueError('Nonfinite diagnostic scalar')
    upper, lower = mass ** 2, mass ** 2 / ns
    invariant_checks = dict(total_points=len(points), finite_all=bool(finite.all()),
        mass_between_zero_and_one=bool(((mass >= 0) & (mass <= 1)).all()),
        C_nonnegative=bool((c >= 0).all()), Ctilde_nonnegative=bool((ct >= 0).all()),
        C_le_visual_mass_squared_raw_violations=int((c > upper).sum()),
        C_ge_visual_mass_squared_over_N_raw_violations=int((c < lower).sum()),
        upper_max_absolute_excess=float(np.maximum(c - upper, 0).max()),
        lower_max_absolute_deficit=float(np.maximum(lower - c, 0).max()),
        checked_without_tolerance=True,
        justification='nonnegative visual attention weights: (sum a)^2/N <= sum a^2 <= (sum a)^2; N is the actual number of visual tokens')
    invariant_checks['status'] = 'PASS' if (invariant_checks['mass_between_zero_and_one']
        and invariant_checks['C_nonnegative'] and invariant_checks['Ctilde_nonnegative']
        and invariant_checks['C_le_visual_mass_squared_raw_violations'] == 0
        and invariant_checks['C_ge_visual_mass_squared_over_N_raw_violations'] == 0) else 'FAIL'
    summary = dict(status='PASS' if invariant_checks['status'] == 'PASS' else 'FAIL',
        schema='trajkv.round6.concentration-diagnostic-summary.v1',
        calibration_cases_total=len(cases), sampled_cases=len(sampled),
        unsampled_cases=len(cases) - len(sampled), configured_concentration_cases=manifest['config'].get('concentration_cases', 1),
        sampled_case_details=sampled,
        coverage=dict(case_layer_records=sum(row['layers'] for row in sampled),
            case_layer_Q_head_coordinates=len({(row['case_id'], row['layer'], row['q_head']) for row in points}),
            layer_head_query_points=len(points), scalar_fields_per_point=3, total_saved_scalar_observations=3 * len(points),
            independence='four teacher answer-query positions of one calibration image; layer/head observations are not independent task examples'),
        definitions=dict(state='decoded mixed incumbent Student native-cache replay',
            attention='same Student FP16 q/k/v, score and softmax diagnostic recomputed in FP32 with the complete causal visual+text support',
            visual_mass='sum_{i in visual} a_i with no visual-only renormalization',
            C='sum_{i in visual} a_i^2',
            o_total='sum_{i in all legal visual+text positions} a_i*v_i, per query head, recomputed in FP32',
            Ctilde='sum_{i in visual} a_i^2*||v_i-o_total||_2^2, implemented by algebraic expansion and clamp_min(0)',
            coordinate='pre-W_O single query-head value coordinates, D128; GQA expands each KV head to its query heads',
            C_units='dimensionless', Ctilde_units='squared norm in the value coordinates',
            final_task_loss=False, W_O_projection_applied=False,
            complexity='O(answer_queries*total_cached_tokens) transient diagnostic weights; no added visual N-by-N matrix'),
        distributions=dict(visual_mass=describe(mass), C=describe(c), Ctilde=describe(ct),
            C_over_mass_squared_check_ratio=describe(c[mass > 0] / upper[mass > 0])),
        invariant_checks=invariant_checks,
        limitations=['one sampled image only; no distributional or task-quality inference',
            'C and Ctilde have different units and are not compared to decide K/V allocation',
            'raw attention weights and value tensors are not stored, so the scalar file alone cannot reconstruct full normalization or independently recompute Ctilde',
            'Ctilde is saved after nonnegative clamping; pre-clamp cancellation/roundoff was not recorded'],
        no_new_ranking=True, no_performance_prediction=True, GPU_used=False, new_model_inference=False,
        source_sha256=dict(cases=sha(cases_path), manifest=sha(manifest_path),
            actual_run_snapshot=sha(args.snapshot), current_collector=sha(current),
            actual_and_current_C_function_source=hashlib.sha256(snapshot_definition.encode()).hexdigest(),
            summarizer=sha(__file__)),
        collector_provenance=dict(snapshot_matches_recorded_run=True, C_function_source_identical=True,
            C_function_AST_identical=True, whole_current_file_matches_run=sha(current) == sha(args.snapshot)))
    stats = summary['distributions']
    doc = f'''本轮 C/Ctilde 是旁路校准诊断。真实运行快照的 SHA256 与采集 manifest 一致，其 `concentration_diagnostics` 函数与当前实现的源码和 AST 均一致。16 个校准 case 中仅首个 `docvqa_next_2126_cal_q1` 保存非空诊断，符合 `concentration_cases=1`；覆盖 28 层、每层 28 个 Q-head 和 4 个 teacher 答案查询位置，共 3,136 个 layer–head–query 点、9,408 个标量。该图含 1,792 个真实视觉 token，其余 15 个 case 未采此旁路。

诊断从相同 Student q/k/v 在 FP32 中重算完整因果 softmax，支持集含视觉和文本。视觉 mass 为视觉权重之和；`C=Σ_visual a_i²`；`Ctilde=Σ_visual a_i²‖v_i−o_total‖²`，其中 `o_total=Σ_all a_i v_i`。它们位于每个 Q-head 的 pre-W_O、D128 value 坐标，尚未进行 W_O 投影；该统计不是最终任务损失。C 无量纲，Ctilde 保留 value 坐标平方范数量纲，不能直接比较二者大小判定 K/V 配额。

全部 9,408 个标量有限。视觉 mass 范围为 {stats['visual_mass']['minimum']:.6g}–{stats['visual_mass']['maximum']:.9g}（中位数 {stats['visual_mass']['median']:.6g}）；C 为 {stats['C']['minimum']:.6g}–{stats['C']['maximum']:.9g}（中位数 {stats['C']['median']:.6g}）；Ctilde 为 {stats['Ctilde']['minimum']:.6g}–{stats['Ctilde']['maximum']:.9g}（中位数 {stats['Ctilde']['median']:.6g}）。3,136 点均满足 `0≤mass≤1`、`mass²/N≤C≤mass²` 和 `Ctilde≥0`，按保存数值直接检查，无须容差。Ctilde 实现含 `clamp_min(0)`，保存文件不能评估截断前的数值抵消误差；未保存完整权重/value 张量，因此上述不变量检查也不等于重新计算全部诊断。

这是单图的描述性交付，4 个查询位置及各层/head 不构成独立任务样本。本汇总未生成新排序、配比规则或性能预测，未新增 GPU 采样，未构造 N×N 矩阵。机器汇总位于 `results/round6_v1/student/concentration_summary_v1.json`。
'''
    for path in (args.output, args.doc):
        if path.exists():
            raise FileExistsError('Refusing to overwrite frozen diagnostic delivery: ' + str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with args.doc.open('x', encoding='utf-8') as stream:
        stream.write(doc)
    print(json.dumps(dict(status=summary['status'], sampled_cases=len(sampled), total_cases=len(cases),
        points=len(points), scalar_count=3 * len(points), invariants=invariant_checks,
        snapshot_matches_run=True, function_source_and_AST_identical=True,
        summary=str(args.output), document=str(args.doc),
        artifact_sha256={str(args.output):sha(args.output), str(args.doc):sha(args.doc)})))


if __name__ == '__main__':
    main()
