# Round7导航：从编码能力到完整存储前沿

先读[本轮研究报告](RESEARCH_PROGRESS_2026-11-16_round7.zh.md)和[目标v8](GOAL_2026-11-16_v8_ENCODING_AND_CONDITIONAL_EXCHANGE.md)。原始Doc主确认是412问，不是5000个独立问题；九配置产生3708个Doc回答，Text128问九配置产生1152个回答。第二骨干复用其中64个Doc来源首问，不能增加独立来源数。

| 内容 | 路径 |
|---|---|
| 表示、真实nibble reader、完整资源账 | [方法](METHODS_ROUND7_2026-11-16.zh.md) |
| 两域新源确认 | [Doc表](../results/round7_v1/confirm_doc128_analysis/table.csv) · [Text表](../results/round7_v1/confirm_text128_analysis/table.csv) |
| 原始图像/方法身份冻结 | [主方法](../audit/a2/round7_method_freeze_v1.json) · [传输核验](../audit/a2/round7_confirmation_transport_v1.json) |
| 同reader、全一级、同完整字节控制 | [Doc分析](../results/round7_v1/controls_doc64_analysis/summary.json) · [Text分析](../results/round7_v1/controls_text64_analysis/summary.json) |
| 同格式编码及低码率64源开发 | [Doc](../results/round7_v1/key_dev64_doc_analysis/summary.json) · [Text](../results/round7_v1/key_dev64_text_analysis/summary.json) |
| projection幅度迭代 | [Doc](../results/round7_v1/projection_dev64_doc_analysis/summary.json) · [Text](../results/round7_v1/projection_dev64_text_analysis/summary.json) |
| 校准干预与初态静态分批等价 | [解释](ROUND7_CONDITIONAL_DIAGNOSTICS_2026-11-16.md) · [精确审计](../audit/a2/round7_matched_static_batch_equivalence_v1.json) |
| 独立于本次拟合的选择来源 | [Doc](../results/round7_v1/policy_select_doc_analysis/summary.json) · [Text](../results/round7_v1/policy_select_text_analysis/summary.json) |
| query谱、ρ与component profiling | [说明](ROUND7_METRIC8_PROFILE2_DIAGNOSTICS_2026-11-16.zh.md) |
| 第二骨干首问确认 | [表](../results/round7_v1/second_confirm64_analysis/table.csv) · [独立配置冻结](../audit/a2/round7_secondary_method_freeze_v1.json) |
| 安静GPU、同输出成本 | [计时分析](../results/round7_v1/quiet_timing_analysis/summary.json) |
| 全部实际工作量与类型划分 | [完整producer盘点](../audit/a2/round7_completed_run_inventory_v1.json) |
| 文献及借鉴边界 | [原始来源](LITERATURE_2026-11-16_round7_encoding_and_exchange.md) |
| 公开证据包的复核方式 | [交付运行说明](ROUND7_PUBLICATION_RUNBOOK_2026-11-16.md) |

可浏览Git树放结果表、配置、参考答案和脚本；较大的原始JSONL、不可变执行源码对象及失败日志见[Round7 release](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round7-20261116)。原图、模型权重、认证资料、机器连接信息与私聊不在公开副本中。严格hash比较区分原始source、脱敏归档的精确字节和Git LF正文。

历史继续保留：[截至11月14日的项目历史](PROJECT_HISTORY_2026-11-14.zh.md) · [Round5](RESEARCH_PROGRESS_2026-11-14_round5.zh.md) · [Round6](RESEARCH_PROGRESS_2026-11-15_round6.zh.md) · [初始研究周期](RESEARCH_CYCLE_REPORT_2026-11-12.zh.md)。这些冻结阶段的结果不合并成新方法的独立确认样本。
