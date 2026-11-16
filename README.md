# TrajKV — 视觉KV低码率表示与覆盖分配

## Round7（2026-11-16）

[完整研究报告](docs/RESEARCH_PROGRESS_2026-11-16_round7.zh.md) · [方法与比较口径](docs/METHODS_ROUND7_2026-11-16.zh.md) · [本轮结果导航](docs/START_HERE_ROUND7_2026-11-16.zh.md) · [目标v8及达成情况](docs/GOAL_2026-11-16_v8_ENCODING_AND_CONDITIONAL_EXCHANGE.md)

两张RTX3090执行真实packed KV及完整FP16模型自由生成，数据与新缓存使用数据盘。主确认固定九方法、Doc128源/412问＋Text128源/128问；第二骨干和固定输出成本单列。

| 工作点 | 完整视觉KV bit/元素（约） | 本轮Doc ANLS | 本轮Text soft acc. |
|---|---:|---:|---:|
| 原覆盖 | 1.800 | 98.144 | 82.344 |
| 新G4_16＋43级分批分配 | 1.295 | 97.680 | 80.938 |
| base2（KIVI式视觉轴） | 2.252 | 96.639 | 81.094 |
| Native4 | 4.252 | 98.414 | 82.109 |
| Full | 16.002 | 98.617 | 83.203 |

新低码率点比原覆盖少约28%视觉KV字节，Doc/Text分别低0.464/1.406pp；相对Native4的两域0.5pp质量目标尚未达到。分批分配可以被相同批内约束下的初态静态分数精确复现，当前结果不支持反馈重估的独立增益。

这是project-fresh派生队列上的结果：Doc来自额外官方train分片，不是完整标准测试集或基础模型训练未见保证。bit包含实际视觉码流、元数据、残差及encoder常量，模型权重和文本KV仍为FP16。全部九方法、配对区间、开发反转和成本都在报告中。

[Round7原始证据、执行源码版本及指纹归档](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round7-20261116)。旧阶段报告、提交和release保留。

Strict `encoder_book_sha256` checks use the exact parameter bytes from the [release archive](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round7-20261116). Git normalizes text to LF; the [fixed content index](evidence/ROUND7_CONTENT_REFS.json) lists archive exact SHA256, Git body SHA256 and Git blob OID separately. Do not replace a frozen expected SHA with the hash of Git-rendered text. The final release receipt binds this index to the verified commit.

---

## Latest: Round 6 — 2026-11-16

[Full research report](docs/RESEARCH_PROGRESS_2026-11-15_round6.zh.md) · [Goal and outcome](docs/GOAL_2026-11-15_v7_DEPTH_AND_COVERAGE.md) · [Evidence navigation](docs/START_HERE_ROUND6_2026-11-15.zh.md) · [Raw execution archive](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round6-20261116)

The two-RTX3090 end-to-end stage is complete: new-source Doc54 / 205 questions and Text256 / 256 questions, eight frozen configurations, **3,688 scored answers**; a LLaVA64 first-question test and 63 matched-output timing measurements are also complete. Model weights and text KV remain FP16; the approximately 1.8-bit budget applies to visual KV and the specified codec metadata/constants.

The new fixed two-stage V candidate reached **88.102 ANLS / 80.469% TextVQA accuracy**, versus **89.429 / 81.094%** for the incumbent and **86.982 / 81.719%** for Native4. Its development gain did not reproduce on new sources. Native4 noninferiority was not established; matched-output latency was about **12.5% higher** than Native4. Gradient/gamma and per-image distortion prototypes retain their negative findings. Next work targets coverage-aware conditional equal-byte exchanges.

[Hash verification notes](evidence/ROUND6_HASH_SCOPE.md) distinguish exact archived bytes from LF-normalized Git content.

Earlier results, commits and releases remain available below. This is a research evidence update, not a claim that the paper's performance objectives are achieved.

---

## Latest: Round 5 — 2026-11-14

[Research report](docs/RESEARCH_PROGRESS_2026-11-14_round5.zh.md) · [Current goal and outcome](docs/GOAL_2026-11-14_v6_REPRESENTATION_AND_REFINEMENT.md) · [Figures and source data](figures/round5_final/figure_manifest.json) · [Incremental raw evidence](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round5-20261114)

On 64 new DocVQA source documents / 256 questions, the frozen primary reached **91.445 ANLS at 1.79942 bits per visual-KV scalar**, versus **87.818 at 2.25223 bits** for Base2: **20.1% fewer visual-KV bytes and +3.63pp ANLS** (paired 95% CI +0.87 to +6.47pp; prespecified Holm p=.0244).

Native4 reached 93.495, so the 0.5pp target remains unmet. New-image TextVQA transfer and matched-output timing are reported separately; the current primary is about 11.1% slower than Base2 in the small matched-output timing control. No complete 5,000 x 8 benchmark restart was performed.

This update preserves the earlier history and release. Numerical budgets include actual visual-cache/position payload and the specified encoder constants; text KV remains FP16. The combination and allocator ports are attributed to their prior work in the report.

---

# TrajKV — 视觉KV极低比特研究记录

本仓库汇总2026-11-12起的研究进展、完整模型GPU实验、代码、结果表、科学图和证据。当前处于方法探索阶段；研究结论与尚未验证的设想分别保留。

先读[完整研究时间线](docs/PROJECT_HISTORY_2026-11-14.zh.md)，再看[最新成果入口](docs/START_HERE_2026-11-14.md)、[数据与代码说明](docs/DATA_AND_CODE_AVAILABILITY.md)和[复跑说明](REPRODUCE.md)。公开版本由原运行材料脱敏导出；原始运行SHA和公开文件SHA各有范围。

## 目前已确认的进展

- **同码率重建规则很重要。** 新64图中，约1.252bit的Lloyd一比特格式，经去中心RMS修复由30.48提升到60.15 ANLS。
- **K方向是重要损伤来源，范数也有作用。** V固定FP16的范数×方向诊断中，真实方向臂比真实范数臂高18.56pp；oracle不列作低比特部署点。
- **方向表示收益在更大新源文档中复制。** 新256源文档/999问中，约1.705bit方向码比同档K2/Lloyd V1高3.35pp，主要校正检验支持该差别。
- **方向修复后，V校准值得投入。** 另一128源文档/501问中，约1.695bit的V-only方案为87.73 ANLS，Base2约2.252bit为87.74；视觉缓存少约24.75%。新256图平衡TextVQA子集上分别81.09%和80.63%，Full81.21%。这些点结果有前景，尚不证明等质量。
- **读量与速度分开实测。** 固定存储的读取方案在Doc256少60.62%额外逻辑读量，ANLS低0.31pp；同输出会话尚未加速。

更小参数化也已完成独立确认：约1.695bit主方案在新128文档485问上87.87 ANLS，Base288.76；约1.579bit整图分组方案86.60。后者相对同价未校准控制追回3.57pp，预定Holm p=0.00372。复用Text256上分别81.45%、81.02%，Base280.63%。[本轮结果与下一步](docs/RESEARCH_PROGRESS_AND_NEXT_PLAN_2026-11-14.zh.md)给出所有代价和区间。

固定强参考Native4的0.5pp质量目标尚未达到；主方案固定输出会话比Base2慢约8.84%。当前结果提供可运行的资源点和机制线索。历史计划中的预测不代表已证实结论。

## 材料结构

| 目录 | 内容 |
|---|---|
| `docs/` | 从最初触发器路线到表示、几何、校准与资源前沿的完整报告、理论和历史计划 |
| `code/` | 实际packed GPU读取、完整8B模型执行、量化校准、统计与绘图 |
| `configs/` | 固定方法、数据成员与复跑配置模板 |
| `results/` | 各阶段表格、统计、参数JSON和报告 |
| `figures/` | 已核查的PNG/PDF/SVG科学图及图注 |
| `audit/` | 原实验范围、来源、哈希与独立核验记录的脱敏副本 |
| `legacy/`, `third_party/` | 所需历史依赖子集、引用端口和上游许可 |
| `evidence/` | GitHub Releases证据压缩包索引及SHA256 |

大体量逐会话记录和实际执行源码快照分批放在Releases，仓库保留可浏览结果与索引。原图、模型权重、凭据和私聊不重打包。

## 运行与边界

主实验使用完整InternVL3-8B-hf、统一FP16，在两张RTX3090 24GB上实际运行；CPU负责调度、统计和绘图。视觉KV实际packed，文本保留FP16。视觉bit/scalar包含规定的payload、元数据、位置与所需常量，encoder校准常量另计入资格；该单位不是整个模型的峰值显存压缩率。逻辑read账不是HBM硬件交易。

开发、独立确认、高存储诊断、teacher-forced代理与固定输出计时分别标注。当前多个问答属于同图派生协议，不假定每题必须依赖历史。完整TextVQA5000问数据已经准备，但8方法共4万回答按探索阶段判断暂缓，没有计作已测结果。

已有的旋转、向量码、误差补偿与输出分布校准组件按来源引用。论文方向以可复现的机制和实际资源—质量结果为依据。[第三方来源与许可](THIRD_PARTY_NOTICES.md)保留各自范围。

仓库地址：https://github.com/zhouz4884-coder/TrajKV
