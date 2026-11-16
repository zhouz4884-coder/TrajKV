# TrajKV — 多轮视觉 KV 精度分配与误差控制

## 最新进展：Round6（2026-11-15，已完成）

[完整研究报告](docs/RESEARCH_PROGRESS_2026-11-15_round6.zh.md) · [当前目标v7及达成情况](docs/GOAL_2026-11-15_v7_DEPTH_AND_COVERAGE.md) · [全部结果导航](docs/START_HERE_ROUND6_2026-11-15.zh.md)

两张3090已完成真实端到端研究、新源Doc54/205问与Text256/256问的8配置确认、LLaVA64首问、同输出计时及逐图RD开发原型。数据与新运行缓存使用数据盘。

约1.8bit视觉KV底座继续保持竞争力；新增固定两级V在新源Doc/Text为88.10/80.47，未稳定超过旧覆盖89.43/81.09。完整报告保留开发增益反转、梯度/gamma负结果与真实运行代价，并给出保覆盖的条件预算交换路线。

---

## 历史摘要（2026-11-12）

2026-11-12 授权的两周研究工作包已完成真实实现、独立评测与一次预算阈值迭代。最终独立验收为0 ERROR、0 PENDING；保留16条早期开发重算超额度记录并排除相同额度结论。

**本轮结论：当前载体 / Margin / 双读探针组合尚未形成稳定方法优势，不能直接包装为论文核心贡献。** 保留实际 packed 执行器、同文本分支仪器、数据与原始反例，下一轮重新设计收益机制。

先读 [统一研究报告](docs/RESEARCH_CYCLE_REPORT_2026-11-12.zh.md)。

| 实测工作 | 证据 |
|---|---|
| 主64图、17预设配置、10,880回答；主Probe7未胜周期对照 | [主报告](results/analysis_eval64_main_v1/formal_report.md) |
| 补KVarN2/4质量参照，合计19配置、12,160回答 | [19配置报告](results/analysis_eval64_with_kvarn_v1/quality19_report.md) |
| 新阈值恢复选择性，独立32图仍未建立质量优势 | [rep32报告](results/analysis_replication32_v1/replication_report.md) |
| 同16图四档预算，独立样本数没有增加 | [预算曲线](results/analysis_budget_sweep_rep16_v1/sweep_report.md) |
| Doc探索16的收益在新64文档、294题×12配置中未确认 | [确认报告](docs/DOCVQA_CONFIRM64_REPORT.zh.md) |
| 108次等token、预热后、quiet GPU计时 | [成本报告](results/quiet_timing_dev4_gpu0_v1/independent_analysis/TIMING_INDEPENDENT_REPORT.zh.md) |

## 代码与复核

- `code/experiments/run_cycle.py`：实际 packed 视觉KV、追加式文本状态、自由生成与预算策略。
- `bank.py` / `fused_progressive.py`：融合读取、残差、1-bit/独立K/V实验扩展。
- `mechanism*.py` / `carrier_surgery*.py` / `self_probe.py`：四臂、位置/剂量、在线双读探针。
- `code/policies/`：当前/未来收益表、额度阈值拟合、特征可用性与成本账。
- `code/analysis/`：独立评分、图像聚类统计、真实图和交付验证。

详细定义见 [方法实现](docs/METHODS_IMPLEMENTED_2026-11-12.md) 与 [本轮执行范围](docs/RESEARCH_CYCLE_2026-11-12.md)。旧PLAN v2/v3保留为历史，不再作为执行约束。

完整argv在 `configs/jobs/`，实际源码在 `audit/job_sources/` 与每run的 `source_snapshot/`。重跑使用新run名/output并锁定相应immutable源码，原始结果不覆盖。入口接收显式模型、样本、旧资产脚本和编解码器路径；模型/图像沿用原资产，其哈希已登记。

[模型身份](audit/model_identity.json) · [运行时](audit/runtime_identity.json) · [原始盘点](audit/run_inventory.json) · [统一结论](audit/research_conclusions.json) · [最终独立验收](audit/FINAL_INDEPENDENT_VALIDATION.json)

逻辑读取payload、真实owner、CUDA峰值与时间分别报告；硬件侧检失败保留原始错误，DRAM指标为null。KVarN标记算子移植†，物化路径不与packed部署效率混排。此为本轮模型/协议的研究结论。
