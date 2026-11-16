# 最新入口：第六阶段真实端到端研究

[完整报告：方法、正负结果及下一步提升](RESEARCH_PROGRESS_2026-11-15_round6.zh.md) · [目标v7与达成情况](GOAL_2026-11-15_v7_DEPTH_AND_COVERAGE.md)

本轮执行完成。约1.8bit视觉KV仍有竞争力；新增固定两级V的开发增量没有在新源上重现，一次学生梯度全排序与gamma校准也没有稳定增益。后续主攻保覆盖的条件等字节交换，不把开发最高点或校准损失当正式质量结论。

| 内容 | 可复核入口 |
|---|---|
| 方法、比较与参数冻结 | [协议](../audit/a2/round6_confirmation_v1.json) |
| 新源Doc54/205问、8配置 | [ANLS及逐题表目录](../results/round6_v1/doc_confirmation_analysis/summary.json) |
| 新源Text256、8配置 | [官方TextVQA准确率](../results/round6_v1/text_confirmation_analysis/summary.json) |
| 四个预设比较、完整字节节约及非劣标尺 | [统一汇总](../results/round6_v1/frozen_primary_summary/summary.json) |
| 来源、冻结源码、实际字节与计时身份 | [独立总审计](../audit/a2/round6_confirmation_and_timing_check_v1.json) |
| 实际两级失真、同字节/同K集合控制 | [机制核验](ANALYSIS_ROUND6_DEPTH_2026-11-15.md) |
| 开发与新源收益反转 | [逐题与源图异质性](ERROR_HETEROGENEITY_ROUND6_DEPTH_2026-11-15.md) |
| 学生梯度与少量真实干预 | [有限动作分析](../results/round6_v1/student/calibration16_analysis_v1/summary.json) |
| 单图C/Ctilde旁路诊断 | [定义与范围](ROUND6_CONCENTRATION_DIAGNOSTIC_20261115.zh.md) |
| LLaVA64首问、5配置 | [实际packed迁移](../results/round6_v1/second_packed64_analysis/summary.json) |
| 新Nv576/32头算术检查 | [18例真实CUDA检查](../results/round6_v1/second_reader_check/check.json) |
| 相同输出、63次测量 | [安静端到端计时](../results/round6_v1/quiet_timing_analysis/summary.json) |
| 逐图D小开发原型 | [结果及额外编码开销](../results/round6_v1/image_rd_small_development_summary/summary.json) |
| 组件归因及强对照范围 | [原始文献核对](RELATED_WORK_ROUND6_2026-11-15.md) |

本轮准确率来自GPU真实自由生成；CPU负责排序、评分和审计。强制相同token的计时不计入准确率。模型权重、FP16文本KV与视觉KV账分别报告。旧结果、Round5确认与发布快照保持原样。

此前进展见[Round5导航](START_HERE_ROUND5_2026-11-14.zh.md)和[项目历史](PROJECT_HISTORY_2026-11-14.zh.md)。
