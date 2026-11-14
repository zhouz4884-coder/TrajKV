# 最新入口：第五阶段表示与精度分配

[研究进展与结论](RESEARCH_PROGRESS_2026-11-14_round5.zh.md)汇总了本轮GPU探索、独立确认、迁移和计时；[当前目标v6](GOAL_2026-11-14_v6_REPRESENTATION_AND_REFINEMENT.md)说明目标达成情况与下一阶段方向。

主要新结果是：全新64份Doc文档、256问上，冻结主候选以1.79942bit达到91.445 ANLS，相对原生2bit少约20.1%视觉KV字节且高3.63pp。Native4的0.5pp目标尚未达到；跨域与耗时结果也保留。

可复核文件：

- [方法、数据及主要比较冻结协议](../audit/a2/round5_confirmation_v1.json)
- [新Doc确认、完整owner及三项主比较](../results/round5_v1/doc_confirmation_summary/summary.json)
- [新Text64官方准确率](../results/round5_v1/text_confirm64_analysis/summary.json)
- [同输出端到端时间](../results/round5_v1/quiet_timing_analysis/summary.json)
- [正式图与绘图数据](../figures/round5_final/figure_manifest.json)
- [相关研究与组件归因](LITERATURE_2026-11-14_round5_precision_allocation.md)

此前项目全过程见[原有研究时间线](PROJECT_HISTORY_2026-11-14.zh.md)。历史报告与冻结数据保持原样，本入口只指向本轮新增产物。
