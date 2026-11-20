"""Compare two completed CPU closures and prepare reviewed canonical560 metadata.

No model execution, remote access, credentials, queue writes, or publication.
Uses the frozen remaining-phase gate; all reports and support inventories are new.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from public_phase_contract_v2 import ROOT, EVIDENCE, load, sha, require, resolve

GENERATOR_SHA = '621b9a3bf29811e3b11292397c1248b19046154a8924dee87cd2fa1ae10c0650'
B_READINESS_SHA = '7362eac5d992c2ad95e2fa6b21ace88f7751a22e2fde90ea4ae9eba3b4d0d46b'
ALIASES = {'F_native_split': 'F', 'KIVI_source_example_K2V2_g32_R32_visual': 'K32',
           'CalibQuant_visual_FP32bias_2bit': 'C2', 'PyramidKV_visual_drop0p888_unknown': 'E64',
           'OT_Hmix23_OnlyTerp_MSE_QJL0_visual': 'OnlyTerp-Hmix23'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialized', type=Path, required=True)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); own, independent = resolve(args.materialized), resolve(args.independent)
    out = resolve(args.output); require(out.is_relative_to(EVIDENCE/'results/publication'), 'Use a new E publication directory')
    require(sha(ROOT/'ops/prepare_remaining_phase_delivery_v1.py') == GENERATOR_SHA, 'Frozen CPU gate changed')
    require(sha(independent/'readiness.json') == B_READINESS_SHA, 'Independent audit identity differs from B receipt')
    pairs = []
    for directory in (own, independent):
        ready = load(directory/'readiness.json')
        require(ready['status'] == 'PASS_CANONICAL_560_AND_ORIGINAL_DEPENDENCY_CLOSURE_CPU' and
                ready['actual_PASS_trials'] == 560 and not ready['missing'] and
                ready['expected_source_line_checks'] == 140 and ready['unique_expected_original_sources'] == 14,
                'Both independently executed full CPU closures must pass')
        pairs.append(ready)
    require(pairs[0]['ledger_sha256'] == pairs[1]['ledger_sha256'], 'Execution ledger differs')
    a, b = load(own/'timing_delivery_recipe.json'), load(independent/'timing_delivery_recipe.json')
    require(a['expected_source_donors'] == b['expected_source_donors'], 'Original source path/three-raw-SHA sets differ')
    require(a['auxiliary_existing_fixed500_view_QA'] == b['auxiliary_existing_fixed500_view_QA'] == 1500,
            'Expected three required exact500 support views')
    require(load(own/'timing_origin_support_matrix.json')['rows'] == load(independent/'timing_origin_support_matrix.json')['rows'],
            'Large-origin support targets differ')
    bindings = load(own/'timing_expected_origin_binding.json')['rows']
    require(bindings == load(independent/'timing_expected_origin_binding.json')['rows'] and len(bindings) == 140,
            'Exact expected ID/method/contract/original-line bindings differ')
    require(load(own/'domain_timing.json') == load(independent/'domain_timing.json') and
            sha(own/'domain_timing.csv') == sha(independent/'domain_timing.csv'), 'Domain timing tables differ')
    audit_files = sorted((own/'audit').glob('*/audit.json')); require(len(audit_files) == 4, 'Expected four full trial audits')
    for path in audit_files:
        other = independent/path.relative_to(own)
        require(load(path) == load(other), 'Independent per-node audit differs')
        require(sha(path.parent/'per_QA.csv') == sha(other.parent/'per_QA.csv'), 'Per-QA timing table differs')
    out.mkdir(parents=True, exist_ok=False)
    comparison = {'status': 'PASS_TWO_SEPARATE_COMPLETE_CANONICAL560_CPU_CLOSURES_EXACTLY_AGREE',
        'same_frozen_validator_two_independent_invocations': True, 'validator_sha256': GENERATOR_SHA,
        'raw_trials': 560, 'warmup_trials': 140, 'measured_trials': 420,
        'four_node_audits_exact_match': True, 'expected_original_record_bindings': 140,
        'unique_original_prediction_sources': 14, 'original_source_path_and_three_raw_SHA_sets_exact_match': True,
        'all_domain_scalar_and_per_QA_tables_exact_match': True,
        'ledger_sha256': pairs[0]['ledger_sha256'],
        'materialized_readiness': {'path': str(own/'readiness.json'), 'sha256': sha(own/'readiness.json')},
        'independent_readiness': {'path': str(independent/'readiness.json'), 'sha256': sha(independent/'readiness.json')},
        'support_only_existing_fixed500_views': 3, 'support_only_view_QA': 1500,
        'new_GPU_inference_by_CPU_packaging': 0, 'timing_quality_QA_added': 0,
        'P2_complete114': False, 'P3_compressed16_full_complete': False, 'whole_external_library_complete': False,
        'archive_prepared_by_this_script': False, 'GitHub_mutation': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    write(out/'independent_closure_comparison.json', comparison)
    rows = load(own/'domain_timing.json')['rows']; formal = [r for r in rows if r['step'] == 'formal_fixed_cohort']
    require(len(formal) == 10, 'Expected separate five-profile Text16 and Info8 formal tables')
    for dataset in ('textvqa', 'infovqa'):
        group = [r for r in formal if r['dataset'] == dataset]; base = next(r for r in group if ALIASES[r['method']] == 'F')
        require(all(r['complete_visual_owner_mean_bytes'] < base['complete_visual_owner_mean_bytes'] and
                    r['TTFT_host_ms'] > base['TTFT_host_ms'] and r['natural_ITL_host_ms'] > base['natural_ITL_host_ms']
                    for r in group if ALIASES[r['method']] != 'F'), 'Descriptive stage conclusion no longer follows from data')
    lines = ['# Canonical source-profile timing: completed560 trials', '',
        '五个预先固定的M1源配置 F/K32/C2/E64/OnlyTerp-Hmix23 在Text16与Info8正式cohort及各自两题硬件smoke上，共完成560个真实CUDA trials。E统一为source-default W64/drop0.888；配置与固定的四域full-target v3逐项对齐。', '',
        '计数为Text2 smoke40 + Text16 formal320 + Info2 smoke40 + Info8 formal160；每QA/方法warm1、repeat3，合计140 warmup与420 measured。560 trials不等于560独立QA，不增加质量QA。旧E8 smoke40、已superseded的旧520计划、legacy656与Calib32均不计入。', '',
        '两次独立CPU完整验收使用同一冻结校验器，四节点raw/token/context/owner/repeat审计、14个原始预测SourceRun、140条expected记录的原行SHA/有效合同/输出token/上下文/bytes绑定全部一致。额外执行依赖均由独立源证明覆盖。三个大donor通过精确既有P2固定500视图及独立评分的辅助闭包进入归档，原JSONL不切分；1500辅助view QA复用既有GPU预测，不代表新增推理或P2完成。', '',
        '以下为正式cohort的分域观测：TTFT为host median，ITL为自然输出token间隔的pooled mean，完整owner bytes为每请求mean。原始字节值和smoke结果见随包domain_timing.csv。', '',
        '| dataset / formal QA per method | source profile | natural tokens mean | TTFT host ms | natural ITL host ms | complete owner mean bytes |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in formal:
        n = 16 if row['dataset'] == 'textvqa' else 8
        lines.append(f"| {row['dataset']} / {n} | {ALIASES[row['method']]} | {row['natural_output_tokens_mean']:.4g} | {row['TTFT_host_ms']:.3f} | {row['natural_ITL_host_ms']:.3f} | {row['complete_visual_owner_mean_bytes']:.0f} |")
    lines += ['', '在这两个正式cohort中，四个压缩源配置的完整visual-owner bytes均低于F，但其median TTFT和pooled自然ITL均高于F。当前移植prototype因此给出了bytes节省，并未在这两个观测量上出现低于F的数值。', '',
        '所有trial均保留host_quiet=false；每trial的GPU compute独占检查不等于整机安静。请求使用预先读入RAM的原始编码图像字节，磁盘读取在计时外，图像解码/预处理在每次请求内重做。GPU stream interval不是纯kernel时间。自然答案长度与内容随方法变化，因此不按较短答案wall时间排名，不给跨域平均加速比，也不将prototype耗时冒称原论文优化kernel速度。', '',
        '阶段完成仅指canonical560实测及来源闭包。既有native F四官方full split被引用，未重跑模型；P2完整114格与canonical16个压缩full split仍分别待完成，本阶段不为其填写预测分数或缓存命中。归档原件、公衍生文件和Git LF正文SHA分别记录；公开发布须由最终交付收据绑定实际commit/tree、asset SHA及固定content_ref。']
    report = ROOT/'docs/CANONICAL_SOURCE_TIMING560_STAGE_v1.md'
    with report.open('x', encoding='utf-8', newline='\n') as stream: stream.write('\n'.join(lines)+'\n')
    original = load(own/'timing_support.json'); files = {str(resolve(r['path'])).lower(): r for r in original['files']}
    additions = [out/'independent_closure_comparison.json', report, Path(__file__), own/'readiness.json',
                 own/'timing_delivery_recipe.json', independent/'readiness.json', independent/'timing_delivery_recipe.json',
                 independent/'timing_expected_origin_binding.json', independent/'domain_timing.json', independent/'domain_timing.csv']
    additions += sorted((independent/'audit').glob('*/*'))
    for path in additions:
        path = resolve(path); require(path.is_file() and path.stat().st_size <= 64*1024**2, 'Supporting file exceeds unchanged exporter guard')
        files[str(path).lower()] = {'path': str(path), 'sha256': sha(path), 'role': 'canonical560_independent_validation_and_browsable_report'}
    support = dict(original, label='canonical560_complete_two_CPU_closures_v1', files=sorted(files.values(), key=lambda r: r['path'].lower()),
        counts={'new_canonical_trials': 560, 'warmup_trials': 140, 'measured_trials': 420,
                'Text_smoke_trials': 40, 'Text_formal_trials': 320, 'Info_smoke_trials': 40, 'Info_formal_trials': 160,
                'legacy_E8_legacy656_Calib32_trials_added': 0, 'timing_quality_QA_added': 0,
                'existing_support_only_fixed500_view_QA': 1500, 'new_GPU_QA_by_CPU_packaging': 0})
    write(out/'canonical560_combined_support_v1.json', support)
    write(out/'EXPORT_PLAN.json', {'schema': 'baseline_matrix.canonical560_reviewed_local_export_plan.v1',
        'primary_contract': a['primary_contract'], 'auxiliary_contracts': a['auxiliary_contracts'],
        'support_file_lists': [str(out/'canonical560_combined_support_v1.json')], 'require_complete': True,
        'stage': 'canonical560_source_profiles_20261120_v1', 'tag': 'external-baselines-canonical-source560-20261120',
        'expected_public_base': '2056c39d696e335da58c3e9982d71da7691c67e6',
        'published': False, 'independent_comparison_sha256': sha(out/'independent_closure_comparison.json')})
    print(json.dumps({'status': comparison['status'], 'report': str(report), 'support_files': len(files),
                      'support_sha256': sha(out/'canonical560_combined_support_v1.json'), 'output': str(out)}))


if __name__ == '__main__': main()
