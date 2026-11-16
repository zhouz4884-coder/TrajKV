"""CPU descriptive summaries of completed metric8/profile2/development axes."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    values = np.asarray(list(values), dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(descriptive_cells=0)
    qs = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return dict(descriptive_cells=len(values), **{key: float(value) for key, value in zip(
        ('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'), qs)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-run', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--axis-run', type=Path)
    parser.add_argument('--axis-analysis', type=Path)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-doc', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_doc.exists():
        raise FileExistsError('Preserve prior summaries; use new output paths')
    metric, profile = args.metric_run, args.profile_run
    if read(metric / 'complete.json')['status'] != 'COMPLETE' or read(profile / 'complete.json')['status'] != 'COMPLETE':
        raise ValueError('Only completed diagnostic inputs may be summarized')
    stats, manifest = read(metric / 'stats.json'), read(metric / 'manifest.json')
    if stats['query_mean_subtracted'] is not False or stats['deployment_budget_eligible'] is not False:
        raise ValueError('Unexpected centered/deployment-eligible metric collector')
    cases = [json.loads(line) for line in (metric / 'cases.jsonl').read_text().splitlines() if line.strip()]
    if len({case['session_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate calibration sources')
    pooled, by_source, bridge = defaultdict(list), defaultdict(dict), defaultdict(list)
    for case in cases:
        cells = defaultdict(list)
        for row in case['actual_query_error_diagnostics']:
            phase = row['phase']
            values = [value for value in row['rho_native_bridge_wht_by_head'] if value is not None]
            pooled[phase].extend(values)
            cells[phase].extend(values)
            bridge[phase].extend(value for value in row['actual_to_native_bridge_energy_ratio_by_head'] if value is not None)
        for phase, values in cells.items():
            by_source[phase][case['session_id']] = float(np.median(values))
    eigen = np.asarray(stats['eigvalsh'], dtype=np.float64)
    top_fraction = eigen[..., -1] / eigen.sum(-1)
    gamma = np.asarray(stats['gamma_uncapped'], dtype=np.float64)
    metric_summary = dict(source_images=len(cases), labels_used=False,
        inference_units='original source images; source/layer/head/query cells are correlated descriptive observations, not independent samples',
        phase_rho_correlated_cell_quantiles={phase: quantiles(values) for phase, values in pooled.items()},
        phase_source_median_quantiles={phase: quantiles(values.values()) for phase, values in by_source.items()},
        phase_source_medians=dict(by_source),
        actual_to_native_bridge_energy_ratio={phase: quantiles(values) for phase, values in bridge.items()},
        rho_definition='128*sum_{g,t}(qr_g dot delta_wht_t)^2 / (sum_g||qr_g||^2 * sum_t||delta_wht_t||^2)',
        delta_definition='WHT(FP16(inverseWHT(actual packed sequential-FP16 Khat_rot)+stored mean)-original FP16 K)',
        query_definition='actual oldDepth student FP16 WHT(q), uncentered, grouped GQA heads; first-question final-prefill query and first two available decode queries',
        not_task_Hessian=True, not_task_loss_rank=True, not_an_execution_gate=True,
        top1_eigenvalue_trace_fraction_by_112_correlated_heads=quantiles(top_fraction),
        gamma_uncapped=quantiles(gamma), gamma_above_cap16_heads=int((gamma > 16).sum()),
        metric_tensor_bytes=29120, gamma_cap16_range=quantiles(np.minimum(gamma, 16)),
        deployed_FP16_u_norm=quantiles(np.asarray(stats['deployed_u_fp16_norm'])),
        q_count_per_head_range=quantiles(np.asarray(stats['q_count'])),
        phase_count_per_head={phase: quantiles(np.asarray(values)) for phase, values in stats['phase_count'].items()},
        diagnostic_owner_bits=quantiles(case['diagnostic_actual_visual_bits_including_encoder'] for case in cases),
        FP16_reference_K_retained_diagnostic_only=True, deployment_budget_eligible=False)
    profile_manifest = read(profile / 'manifest.json')
    profile_records = [json.loads(line) for line in (profile / 'records.jsonl').read_text().splitlines() if line.strip()]
    second_id = profile_manifest['selected_ids'][1]
    second = [row for row in profile_records if row['session_id'] == second_id]
    calls = [json.loads(line) for line in (profile / 'calls.jsonl').read_text().splitlines() if line.strip()]
    outlier = max((row for row in calls if row['component'] == 'encoder_FWHT'), key=lambda row: row['gpu_interval_ms'])
    profile_summary = dict(second_source_id=second_id, second_source_components=[dict(
        variant=row['variant'], visual_tokens=row['visual_tokens'], encoding_wall_ms=row['encoding_wall_ms'],
        instrumented_session_ms=row['session_ms'], generated_tokens=row['generated_tokens'],
        reader_calls=row['reader_calls'], forced_canonical_chat_closure=row['forced_canonical_chat_closure'],
        components=row['component_profile']['components']) for row in second],
        largest_encoder_FWHT_call=outlier,
        interpretation='First-call ~412ms event and nearly equal CPU wall are an isolated cold-start/host-gap outlier; compilation is plausible, not proven. Neither it nor these unquiet free-running instrumented sessions establish steady-state speedup.',
        reader_component_scope='reader_FWHT measures query transform only, not the packed attention/read kernel or whole read cost',
        event_scope='current-stream intervals include host launch gaps; they are not kernel-active time',
        quiet_E2E_speed_eligible=False, same_output_timing_eligible=False, task_accuracy_eligible=False)
    axis_summary = None
    if args.axis_run and args.axis_analysis:
        if read(args.axis_run / 'complete.json')['status'] != 'COMPLETE':
            raise ValueError('Axis development run incomplete')
        axis_records = [json.loads(line) for line in (args.axis_run / 'records.jsonl').read_text().splitlines() if line.strip()]
        with (args.axis_analysis / 'table.csv').open(encoding='utf-8-sig', newline='') as stream:
            table = list(csv.DictReader(stream))
        counts = {}
        for name in sorted({row['variant'] for row in axis_records if row['method'] == 'round7_tail_key'}):
            counts[name] = [dict(N=nv, K_stages=k, V_stages=v) for nv, k, v in sorted({
                (row['visual_tokens'], row['depth_residual_K_stages'], row['depth_residual_V_stages'])
                for row in axis_records if row['variant'] == name})]
        axis_summary = dict(source_images=16, role='observed secondary-backbone first-question development; no blind confirmation',
            score_percent={row['variant']: 100 * float(row['score']) for row in table},
            actual_stage_counts=counts,
            execution_semantics='Named incumbent preserves the old affordable-prefix K subset, then fills V. K2 Vfirst is pure V repair; K1 Vfirst is near-all-V plus K, not pure V. At N2144 it keeps K297/V1023 rather than strict-priority K296/V1024.',
            inference='V priority is a useful exploratory direction on this backbone; Full remains about 12pp higher. This does not establish universal uncalibrated quality preservation.',
            new_independent_source_count=0)
    source_files = [metric / name for name in ('manifest.json', 'complete.json', 'cases.jsonl', 'stats.json')]
    source_files += [profile / name for name in ('manifest.json', 'complete.json', 'records.jsonl', 'calls.jsonl')]
    if axis_summary:
        source_files += [args.axis_run / name for name in ('manifest.json', 'complete.json', 'records.jsonl')]
        source_files += [args.axis_analysis / 'table.csv']
    report = dict(schema='trajkv_round7_metric8_profile2_diagnostic_summary_v1', status='COMPLETE',
        scope='CPU descriptive recount of already completed GPU diagnostics/development only; no new model/metric fit/GPU run',
        metric8=metric_summary, profile2=profile_summary, secondary_axis_development=axis_summary,
        source_sha256={str(path): sha(path) for path in source_files}, summarizer_sha256=sha(__file__))
    lines = ['# Round7 metric8 与 profile2 诊断', '',
        f"metric8 覆盖 {len(cases)} 张校准图，采集实际 oldDepth 学生的 FP16 WHT query，未减去 query 均值。ρ使用真实 native 重构桥误差在 WHT 基中的能量比，不是任务 Hessian 或任务损失有效秩。下列分位数描述相关的 source/layer/head/query 单元；独立来源仍只有八张图。", '',
        '| 阶段 | 相关单元数 | ρ P5 | ρ 中位数 | ρ P95 | 每源ρ中位数的中位数 |',
        '|---|---:|---:|---:|---:|---:|']
    for phase in ('prefill', 'decode'):
        q, sq = metric_summary['phase_rho_correlated_cell_quantiles'][phase], metric_summary['phase_source_median_quantiles'][phase]
        lines.append(f"| {phase} | {q['descriptive_cells']} | {q['p05']:.3f} | {q['median']:.3f} | {q['p95']:.3f} | {sq['median']:.3f} |")
    top, g = metric_summary['top1_eigenvalue_trace_fraction_by_112_correlated_heads'], metric_summary['gamma_uncapped']
    lines += ['', f"112 个相关 head 的 query 二阶矩 top1/trace 中位数为 {top['median']:.3f}，P5–P95 为 {top['p05']:.3f}–{top['p95']:.3f}；uncapped γ 范围 {g['min']:.2f}–{g['max']:.2f}，中位数 {g['median']:.2f}，其中 {metric_summary['gamma_above_cap16_heads']}/112 超过 cap16。γ来自 rank-one 近似，不能解释为可回收的任务收益；两套部署 metric 常量均为 29,120 B。采集器额外保留原始 FP16 K，属于单独计费的诊断 owner。", '',
        f"profile2 第二源 `{second_id}` 的组件 GPU 区间如下（ms）。reader FWHT 仅含 query 变换，不含完整 packed attention 读取。", '',
        '| 方法 | 编码 wall | 编码 FWHT | PQ+gain | V Lloyd | 残差 | reader FWHT |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in second:
        c = row['component_profile']['components']
        vals = [c[name]['gpu_interval_total_ms'] for name in ('encoder_FWHT', 'PQ_and_gain', 'V_Lloyd', 'residual_sign_depth', 'reader_FWHT')]
        lines.append('| ' + row['variant'] + f" | {row['encoding_wall_ms']:.2f} | " + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    lines += ['', f"首源 Norm 的首个编码 FWHT 事件为 {outlier['gpu_interval_ms']:.2f} ms，对应 CPU wall {outlier['cpu_wall_ms']:.2f} ms；这是孤立首调用异常，可能含编译或 host launch 等待。不能把该点或两源插桩 session median 当作稳定速度差。正式 quiet 工具另做 warmup、同八个强制输出 ID 和三次测量，只计成本。"]
    if axis_summary:
        lines += ['', '第二骨干开发点：Full 75.12、K2 Vfirst 62.90、K1 Vfirst 62.66、K1 Kfirst 49.65 ANLS。实际 stage 计数符合冻结 incumbent planner：K2 Vfirst 为 0K/403–410V；K1 Vfirst 约 289–297K/1023–1024V，Kfirst 为 1024K/289–296V。N=2144 的 Vfirst 保留 K297/V1023，是 incumbent K 保留规则的结果，不能写成严格全序 V 优先。V 修复改善了该骨干的探索点，仍比 Full 低约 12pp；这十六个已观察来源不增加独立确认来源数。']
    lines += ['', '可复核数值、逐源统计和原始文件 SHA 见对应 audit JSON。', '']
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_doc.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with args.output_doc.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(status='COMPLETE', source_images=len(cases), rho=metric_summary['phase_rho_correlated_cell_quantiles'],
        top1_fraction=top, gamma=g)), flush=True)


if __name__ == '__main__':
    main()
