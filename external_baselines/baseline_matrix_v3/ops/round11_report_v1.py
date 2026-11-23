"""Render the completed frozen Round11 study; no inference, fitting or selection."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = Path('${EVIDENCE_ROOT}/results/round11')
DS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')
MODES = ('FP16', 'R7low', 'R7high', 'JointV', 'JointSpatialWO')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, text):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f: f.write(text)


def table(points, modes=MODES):
    lookup = {(p['dataset'], p['mode']): p for p in points}
    rows = ['| 方法 | TextVQA | InfoVQA | DocVQA | ChartQA |', '|---|---:|---:|---:|---:|']
    for m in modes:
        rows.append('| '+m+' | '+' | '.join(f"{100*lookup[d,m]['primary_mean']:.4f}" for d in DS)+' |')
    return '\n'.join(rows)


def main():
    specs = [('primary', 'summary_final_primary_v4'), ('wo', 'wo_summary_final_v1'),
             ('confirm128', 'confirmation_summary_final_v3'), ('expand384', 'expand_summary_final_v1')]
    docs = {g: read(E/f/'summary.json') for g, f in specs}
    assert docs['primary']['actual_closed_new_free_generation_QA'] == 384
    assert docs['wo']['actual_closed_new_free_generation_QA'] == 192
    assert docs['confirm128']['actual_closed_fresh_QA'] == 640
    assert docs['expand384']['actual_closed_fresh_QA'] == 1920
    assert sum(len(d['points']) for d in docs.values()) == 88
    combo_path = E/'combined512_prepackaging_v1/combined512.json'
    combo = read(combo_path)['original128_additional384_combined512_separate_report']
    assert combo['combined_quality_computed'] and combo['new_inference_events'] == 0
    dest = E/'final_report_v1'; dest.mkdir(parents=True, exist_ok=False)
    rows = []
    for g, _ in specs:
        for p in docs[g]['points']:
            assert p['primary_mean'] is not None
            rows.append({'group_id': g, 'dataset': p['dataset'], 'mode': p['mode'],
                'primary_mean': p['primary_mean'], 'expected_QA': p['expected_QA'],
                'mean_complete_visual_owner_bytes': p['resources']['complete_bytes']['mean'],
                'ci95': json.dumps(p['ci95']), 'new_inference_events': 0 if g in ('primary','wo') and p['mode'] in ('FP16','R7low','R7high') else p['expected_QA']})
    with (dest/'QUALITY_COMPLETE_BYTES_88.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    with (dest/'COMBINED512_QUALITY.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, ['dataset','mode','sources','primary_mean','mean_complete_visual_owner_bytes','new_inference_events'], lineterminator='\n'); w.writeheader()
        for p in combo['points']:
            w.writerow(dict(dataset=p['dataset'],mode=p['mode'],sources=p['combined_sources'],primary_mean=p['primary_mean'],mean_complete_visual_owner_bytes=p['complete_visual_bytes']['mean'],new_inference_events=0))
    contrasts = []
    for d in DS:
        for m in MODES[3:]:
            p = combo['pairs'][d+'__'+m+'_minus_R7low']
            contrasts.append({'dataset':d,'mode':m,'sources':p['questions'],'delta_pp':100*p['mean'], 'ci95_pp':[100*x for x in p['ci95']]})
    write_new(dest/'COMBINED512_PRIMARY_CONTRASTS.json', json.dumps(contrasts,ensure_ascii=False,indent=2)+'\n')
    cirows = ['| 域（来源数） | JointV Δpp [95% CI] | JointSpatialWO Δpp [95% CI] |', '|---|---:|---:|']
    for d in DS:
        items = [x for x in contrasts if x['dataset'] == d]
        cirows.append('| '+d+f"（{items[0]['sources']}） | "+' | '.join(f"{x['delta_pp']:+.4f} [{x['ci95_pp'][0]:+.4f}, {x['ci95_pp'][1]:+.4f}]" for x in items)+' |')
    resources = read(E/'additional384_per_domain_report_v1/report.json')['resources']
    timerows = ['| 域 | R7low | R7high | JointV | JointSpatialWO |', '|---|---:|---:|---:|---:|']
    for d in DS:
        timerows.append('| '+d+' | '+' | '.join(f"{next(x for x in resources if x['dataset']==d and x['mode']==m)['prototype_encode_median_seconds']:.2f}" for m in MODES[1:])+' |')
    findings = dict(schema='trajkv.round11.final_findings.v1', status='SCIENCE_COMPLETE_PUBLICATION_IDENTITY_IN_SEPARATE_RECEIPT',
        actual_new_free_QA=3136, development_new_QA=576, original128_new_QA=640, additional384_new_QA=1920,
        actual_result_views=88, actual_view_events=3712, reused_reference_view_events=576,
        unique_reused_reference_view_events=288, repeated_reused_reference_view_events=288,
        unique_validation_sources=512, unique_development_sources=96,
        candidate_gain_supported=False, replace_R7=False, new_official500_or_cross_model_runs=False,
        combined512=combo['points'], contrasts=contrasts,
        source_summary_sha256={g:sha(E/f/'summary.json') for g,f in specs},
        combined_derived_sha256=sha(combo_path), report_producer_sha256=sha(__file__))
    write_new(dest/'FINAL_FINDINGS.json',json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    body = f'''# Round11最终研究报告：同预算V联合重编码与读出几何验证

实验始于2026-11-23。完成3136次真实端到端GPU自由回答，并按用户要求扩大到512个确认来源。结论是：**本轮两个冻结候选没有形成稳定优于R7的任务收益，暂不替换R7。** 开发集每域一题的改善没有在独立原128及追加384中稳定复现；局部SSE或W_O目标下降不能代替答案质量。

## 合并512来源：描述性主表

Text/Info各192、Doc/Chart各64。Text为官方ten-vote soft、Info/Doc为冻结ANLS、Chart为relaxed指标，均以百分数列出，不跨异质指标计算总分。原128、追加384下方分别保留；合并表是派生视图，新增推理数为0。

{table(combo['points'])}

相对相同逐题完整预算R7low的配对source-cluster bootstrap区间（2000次、未调整、多域描述性区间）：

{chr(10).join(cirows)}

Text/Info差值的区间跨0；Doc/Chart没有正向收益。该结果不支持“新方法提升”，也不是严格的非劣证明。FP16是参照而非数学上限，压缩分支偶有更高任务分数不能自动视为错误。R7本身在该512来源的Text/Doc/Chart点估计接近FP16；Info仍是明显质量缺口。这里没有复跑新方法的官方固定500或其它模型，不能宣称超过完整SOTA矩阵。

## 两批确认分别报告

原128（48/48/16/16，候选冻结后全部五方法fresh640）：

{table(docs['confirm128']['points'])}

追加384（144/144/48/48，全部五方法fresh1920，新的独立复测）：

{table(docs['expand384']['points'])}

用户在已看到部分原128结果后要求扩样。我们在追加输出前固定seed、规模与原两个候选，没有按p值/效果提前停、换题或换方法；不把合并512称为在最初128之前预注册。新增384的ID/source/Doc parent/原图SHA与原128及已登记history、calibration、development、formal500交集为0，合并唯一来源512。真实split为Text validation、Info train、Doc train、Chart test，既有native-full曝光说明保留，不声称预训练或绝对历史未见。

## 方法与模型

完整InternVL3-8B，实际参数7,944,373,760、原FP16权重、两片RTX3090 24GB；图像cap8，原模板/suffix/position IDs，greedy seed0/max_new64。question_unknown视觉前缀在问题到来前编码，之后释放原FP16视觉cache，实际packed reader生成后缀；不将CPU复核或teacher-forcing当作任务回答。模型/原图/结果及全部新TRITON/CUDA/TMP等缓存位于8TB项目数据盘，Windows大材料在E盘。

JointV固定原V端点，联合重选base bits/sign/alpha：实数内层排序128个阈值、至多129区间求最优alpha；实际Half候选包含原值与round/邻接，严格接受、平局保旧，不声称Half域全局最优。Spatial用原非负Half alpha符号位选择原预测或同packed word两行网格邻域预测，Shuffled是同图共轭邻接对照；JointSpatial先写唯一最终B*再预测。V没有K的16词方向书，原f0只fold一次。

SpatialWO/JointSpatialWO从已有W_O临时构造GQA7 Gram，按实际Half重建的完整局部H目标重选字段。后者先在H下验收原点/MSE提案，再固定2轮局部base/sign更新，冻结B*后执行空间搜索。旧字段为合法路径，编码后H/W拷贝全部释放，K23/f0/原V元数据/残差覆盖不变。H下降只是程序定义局部指标；不是attention或任务损失保证，也不是精确Gram浮点证书。完整读出误差还包含路由、text V与跨层状态变化，详见理论文档及独立数学审阅。

## 开发结果与实际费用

D96只有96来源：四主候选384新回答、两WO候选192新回答。JointV在Text +2.1875pp、Info +1.2153pp；JointSpatialWO又保留Doc +2.0833pp，Chart持平。每域仅一题改善，不能将96×6称576独立来源。这些开发信号促成冻结两版确认，而不是最终成功结论。

全部候选逐请求完整视觉owner费用与R7low严格相同：B0(N)=4590N+89733，增量0。payload、gain/mean、V元数据、书/原f0/策略、lookup与positions均计；典型工作点约1.3bit，不是逐题硬≤1.3。文本KV均FP16，编码workspace不常驻，实际全请求allocator peak单列。

追加384的原型编码中位秒（含诊断，非受控速度或HBM吞吐实验）：

{chr(10).join(timerows)}

JointV约14秒、联合版约31秒，均慢于R7约5.5–5.7秒，没有速度收益。内部joint/spatial wall未分别instrument，不能从总时间相减推断；阶段操作次数、Gram host、命名workspace估算和实际整请求peak分开。WO开发96来源中，SpatialWO局部H下降而L2全部增加；JointSpatialWO H约降14–17%且L2约降8–11%，仍未转成确认质量优势。

## 统计、执行及交付口径

3136新回答=384主开发+192WO开发+640原128+1920追加384。另576旧参照视图中唯一288、重复288；88实际结果视图/3712视图事件，合并512另为0推理派生。teacher/TF/合成kernel均QA0单列，不冒充独立样本。

技术v1因CUDA Half nextafter不支持，在原1+noop4共5次TF之后失败；v2逐位等价邻接修复后实际9TF，WO实际5TF，均未作任务评分。R128首次v2父任务因Root漏传队列配置在模型前退出，两setup为0模型/0GPU forward/0回答；补齐后新v3完成640，原证据保留。发布冷核验发现Windows SOURCE_LOCK大小写alias与D96 raw group/metric canonical group合同差异，修复仅在新PUB5精确验证器，不改变原数值scorer或任何预测。阶段发布身份由commit/tree/归档SHA交付回执记录。

## 下一步判断

保留原R7作为当前方法，不用更多局部loss训练步数把本轮包装为正向提升。优先验证K实际码字/视觉-text分区质量是否是剩余缺口；当前证据只给方向，不证明具体因果。R7的K位于centered WHT码器帧，native对角度量拉回后通常稠密，简单取对角会退化成常数欧氏权重，不能把“模型通道权重”直接套到G4。可在原reader/码本/预算内，用正确full-C的16词块坐标选择，与同搜索器C=I及同残差规则配对；固定有效V底座后再看真实GPU答案。

另一候选是分别保视觉分区log-mass与分子/归一输出，再投影回已有码流。Hydragen的精确分区合并依赖输出与LSE，但它不是有损KV压缩方案；OptR先K后V的目标也不自动补偿既存K误差。可借鉴这些组件，新增效果仍须另冻结实验验证。相关一手阅读与未执行推导见文献备忘；本轮512已被观察，后续方法不能把它重新称为独立确认。

[Hydragen原文](https://arxiv.org/html/2402.05099v2)；[OptR原文](https://arxiv.org/html/2608.02691v4)；[CALFA-KV原文](https://arxiv.org/html/2611.10511v1)。组合已有组件是合理研究方式，论文贡献应落在新增、可执行且验证有效的能力；本轮结果还不足以支撑新方法的高水平论文主性能结论。
'''
    report = ROOT/'docs/REPORT_2026-11-23_ROUND11_FINAL.zh.md'
    write_new(report, body)
    for target in (Path('${FROZEN_TRAJKV_ROOT}/docs')/report.name, dest/report.name): write_new(target, body)
    print(json.dumps({'status':'PASS_FINAL_SCIENTIFIC_REPORT_FROM_COMPLETED_SOURCES','report':str(report),'report_sha256':sha(report),'table':str(dest/'QUALITY_COMPLETE_BYTES_88.csv'),'new_GPU_QA':0,'GitHub_published_by_this_script':False},ensure_ascii=False))


if __name__ == '__main__': main()
