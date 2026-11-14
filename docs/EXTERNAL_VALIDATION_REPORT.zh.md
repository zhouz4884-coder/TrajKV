# TrajKV 外部验证报告：固定参数下的质量、格式与参考边界

日期：2026-11-12，Asia/Hong_Kong。范围：`results/external_suite_gpu1_v1` 中 DocVQA16、GQA16 与 ConvBench4；未读取 replication32 结果，未运行新推理、修改参数或重新选样本。

**当前外部证据没有建立新策略优于强廉价对照的结论。** DocVQA 的 future table 高于 Periodic/Uniform，但对更强且成本更低的 Base2，配对 95% 区间下界为 0；对 Full/Native4 的区间跨 0。Probe 未胜 Base2。GQA 的原样 exact 大量受答案首字母大小写影响，辅助口径下模型表现远高于原样分数，Probe/Future 仍未胜 Periodic。ConvBench 存在输出截断和内部矛盾的参考，只能保留词汇诊断。

## 1. 数据与执行审计

| 数据 | 图像/会话 | 每配置已问问题 | 配置数/完整记录 | 协议 |
|---|---:|---:|---:|---|
| DocVQA | 16 | 64 | 8 / 128 | 同文档独立问题成组，最多 6 轮 |
| GQA | 16 | 128 | 8 / 128 | 同图独立问题成组，8 轮 |
| ConvBench | 4 | 12 | 4 / 16 | 原生三轮自由文本压力诊断 |

三组记录均完整，没有重复 run、缺参考、错图 hash 或轮数不符；同一会话各配置的 native initial-logit hash 一致。36 个选中图片文件的 SHA256 通过检查。DocVQA64、GQA128、ConvBench12 个已问问题及其参考与本地原始整理源逐字匹配；Doc/GQA 与各自 dev32 图片 hash 交集均为 0。图像/文本出处复核不等于对完整官方隐藏标注、所有可接受答案或预训练暴露的重新审计。

DocVQA 有一条原会话包含第七问，依照已冻结的 6 轮上限没有询问；64 是实际问题数，不是丢失结果。两组 grouped QA 均不能冒称自然多轮对话或官方完整 leaderboard。

全部配置使用同一个 InternVL3-8B、FP16 执行器。所有本轮外部记录 `read_budget_ok=true`、`read_ledger_exact=true`。16 MiB 限额约束的是**额外逻辑 tensor 读取**；它不是硬件 DRAM/HBM 流量，也不意味着各配置实际花费相同。Base2 没有残差读取，存储也小于 Asym，不能因为共享一个上限就称为等成本。没有生成速度结论。

## 2. DocVQA：完整主表及强对照

主列锁定组织者代码 ANLS：小写、首尾 strip、保留内部空格与标点、多个参考取最大值，similarity ≥ 0.5 保留，空预测得 0。另报只改变阈值比较符的 strict 辅助列。源码及其固定 commit/hash 见 `docs/external_scoring.md`，不冒称重新复现 RRC 服务端。

数值为百分制；每配置 16 张图、64 题。区间按 source image 配对/聚类重采样，2,000 次，seed 1212。以下均为探索性区间，没有以多重比较校正后的显著性作论文结论。

| 配置 | 主 ANLS | 图像聚类 95% CI | Exact / 64 | strict 辅助 ANLS |
|---|---:|---:|---:|---:|
| Full | 83.40 | [74.10, 91.31] | 43 | 81.84 |
| Base2 | 82.56 | [73.37, 90.71] | 43 | 80.22 |
| Native4 | 83.40 | [74.10, 91.31] | 43 | 81.84 |
| Uniform | 79.99 | [68.85, 89.19] | 41 | 78.43 |
| Periodic | 80.06 | [70.02, 89.15] | 42 | 77.72 |
| Future table | 84.99 | [75.96, 92.79] | 45 | 83.43 |
| Probe l7 / 0.02 | 81.00 | [71.45, 89.75] | 42 | 78.66 |
| Margin write | 82.72 | [72.48, 91.46] | 45 | 81.16 |

各配置有 2 或 3 个问题恰落在 similarity 0.5，strict 辅助列因此下降 1.56 或 2.34pp。不能将两个 ANLS 口径交替用于挑选正例。

| Future table − 对照 | 主 ANLS 差，pp | 配对图像 95% CI，pp | 分数提高/下降题数 |
|---|---:|---:|---:|
| Base2 | +2.43 | [0.00, 6.89] | 4 / 0 |
| Native4 | +1.59 | [−5.56, 10.02] | 5 / 6 |
| Full | +1.59 | [−5.56, 10.02] | 5 / 6 |
| Uniform | +5.00 | [0.56, 10.34] | 4 / 0 |
| Periodic | +4.93 | [0.16, 12.63] | 5 / 0 |

Future table 的均值不只是回到 Base2，但**尚未建立它胜过 Base2**：差值区间包含 0，且 Base2 更省存储/读取。相对 Uniform/Periodic 的区间是一个值得保留的局部信号，不能替代对强廉价对照的检验，也不能抵消主研究中的其他负结果。Probe 对 Periodic 为 +0.94pp，95% CI [0.00, 2.46]；对 Base2 的均值低 1.56pp。

### Future 相对 Base2 的四个正例

以下只描述输出字符串与已给 GT 的关系，不新增语义评分，也不把 ANLS 软分提高算作四道题全部答对。

| 会话/轮次 | GT | Base2 → Future | ANLS | 字面观察 |
|---|---|---|---|---|
| `docvqa_990bfa3f1e47880b` / T2 | `3,39,119` | `3,93,196` → `3,39,16,27` | 0.50 → 0.60 | 两者都没有精确匹配 GT，仅编辑距离软分提高 |
| `docvqa_e2a09399f3287f8d` / T2 | `30.46` | `30.62` → `30.46` | 0.60 → 1.00 | 数值字符串改为 GT |
| 同会话 / T3 | `4012.46` | `4303.47` → `4012.47` | 0 → 0.8571 | 仍与 GT 差 0.01，不能称数值回答正确 |
| `docvqa_f6a49562500657ea` / T1 | `CMRO` | `CMROJ` → `CMRO` | 0.80 → 1.00 | 去掉额外尾字母并精确匹配 GT |

四题来自三张图，其中两题成为 exact 正确，另两题仍未匹配 GT；不是统一删掉冗余解释句带来的收益。

## 3. GQA：固定双口径暴露格式瓶颈

主列保持已固定的原始字符串 exact identity，不事后去标点、抽取答案、映射数字或 CLEVR 同义词。辅助列仅做 lower+strip，用来诊断大小写/首尾空白；不是看到结果后替换主指标。

| 配置 | 原样 exact / 128 | 原样 exact，% | lower+strip 辅助 / 128 | 辅助，% |
|---|---:|---:|---:|---:|
| Full | 19 | 14.84 | 87 | 67.97 |
| Base2 | 18 | 14.06 | 85 | 66.41 |
| Native4 | 19 | 14.84 | 87 | 67.97 |
| Uniform | 18 | 14.06 | 86 | 67.19 |
| Periodic | 19 | 14.84 | 88 | 68.75 |
| Future table | 18 | 14.06 | 85 | 66.41 |
| Probe l7 / 0.02 | 18 | 14.06 | 85 | 66.41 |
| Margin write | 18 | 14.06 | 85 | 66.41 |

例如 `No`/`no`、`Right`/`right`、`Wide`/`wide`、`Closed`/`closed`、`Stairs`/`stairs` 会在原样 exact 中失分。Periodic 有 69 个问题仅经预先固定的 lower+strip 辅助口径就由失配变为匹配。因此 14.84% 不能被解释为模型完全没有图像理解能力。辅助口径仍不证明其他未匹配题的语义正误。

Probe 对 Periodic：原样差 −0.78pp，配对图像 CI [−2.34, 0.00]；辅助口径差 −2.34pp，CI [−6.25, 0.00]。Future table 的整体匹配数也没有超过 Periodic。原样计分的表面脆弱性没有掩盖一个已成立的新策略胜利。

## 4. 截断与 ConvBench 参考有效性

- DocVQA：仅 Uniform 的 1/64 回答触及 32-token cap 并需强制结束；其余配置为 0/64。
- GQA：所有配置 0/128 触及 16-token cap。
- ConvBench：每个配置均有 7/12 回答触及 96-token cap 并需强制结束。它们不能代表不限长度的完整开放回答质量。

ConvBench 只报告 multiset 词汇 token-F1：Full 30.77、Base2 32.42、Periodic 32.10、Probe 32.23。Probe−Periodic 为 +0.13pp，图像聚类 CI [−1.58, 1.60]，仅四张图，**不能称为官方质量或语义正确率提升**。

`convbench_37` 的参考还存在可直接核对的内部矛盾：T2 指定白马从 f3 到 e5 吃黑兵；T3 对“previous move”的参考却变为白象 a4 到 b5（Bxb5）。T1 的参考黑兵列表没有 b5。T3 也仍保留 f3 的马和 e5 的黑兵。该会话不能充当可靠的跨轮语义质量裁判。本报告没有删除该会话或重算筛选后的分数，只保留原样词汇诊断与这条有效性警告。

## 5. 输入编码复核与独立 Doc64 的输入提醒

本次最初的局部问句比对曾因 Windows 分析 helper 没有显式指定 UTF-8 而出现错误解码/显示，已修正 helper 的读取编码并重算全部报告。**冻结的探索输入没有发生所谓“��s→’s”的实际修订。** 例如原源与 prepared 的 ITC 问句都包含同一个 `\u2019`；当前已问问句/参考均逐字回溯通过，没有待应用的引号修订列表。

对 root 新冻结的 DocVQA 确认64，仅做了输入审计，没有读取任何确认输出。64 个文档共有 316 个原问题，按 6 轮限额对应 294 题；294 个实际计划问题、参考和图片均可回溯到本地整理源，没有 U+FFFD 编码字符。

核查的初始确认版本 SHA256 `80dbc2f4fbb80556818e0daa13f11e30132f0f4e4ca0a40000433ea51973f3a9` 在 64/64 首问中重复了 `Question:`，形成 `...\nQuestion: \nQuestion: 原问句`。建议列表只提出首问 marker 去重，不涉及答案、图像或后续问句，保存在 `docvqa_confirm64_suggested_revisions.json`，本报告没有应用修改。若 root 已另行修订/冻结，应以新 receipt 版本为准。确认64是在看到探索16汇总之后选择的独立确认，不是早前预注册。

**输入 v2 验收已完成**：root 在确认推理开始前修订并冻结为 SHA256 `25a9dac82654f75db96feca0388b44ee5c750292436f3b643ea83d955e6024f3`。64 个首问均精确等于探索版 `initial_answer_instruction + original_question`（prefix 本身已含 `Question:`）；后续问句和 content prompt 一致，294 个计划参考/源问题与64图 hash 全通过，不存在实际编码修订。`docvqa_confirm64_input_audit_v2.json` 是当前输入验收；旧建议及80db版保留为被拒预检历史，不再是待处理问题。此次 v2 输入验收没有读取确认质量结果。

## 6. 可复算交付与结论边界

- `code/analysis/report_external_suite.py`：只读三份已完成外部输出，复用固定评分器；不读取 replication32。
- `results/analysis_external_suite_v1/external_review.json`：完整统计、出处、图像/轮数/预算审计。
- 同目录 `external_summary.csv`、`format_diagnostics.csv`、`additional_paired_contrasts.csv`、`changed_question_examples.csv`；各数据子目录另有逐题、逐轮、配对差与 summary。
- `figures/external_validation_v1.png` / `.pdf`：强对照下的 DocVQA 差值与 GQA 固定双口径图，已检查布局。
- `docvqa_confirm64_input_audit.json` / `docvqa_confirm64_suggested_revisions.json`：仅确认输入审计，不含推理结果。

本报告保留 DocVQA 的局部正向信号，同时给出强对照、软评分与小样本边界。它没有建立跨数据集的新策略优势；GQA 和 ConvBench 的格式/参考限制也不应被用来掩盖负结果。KVarN64 单独作为质量机制参考交由主分析，不与物化路径的速度或外部数据分数混合。
