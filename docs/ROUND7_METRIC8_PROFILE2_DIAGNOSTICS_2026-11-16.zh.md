# Round7 metric8 与 profile2 诊断

metric8 覆盖 8 张校准图，采集实际 oldDepth 学生的 FP16 WHT query，未减去 query 均值。ρ使用真实 native 重构桥误差在 WHT 基中的能量比，不是任务 Hessian 或任务损失有效秩。下列分位数描述相关的 source/layer/head/query 单元；独立来源仍只有八张图。

| 阶段 | 相关单元数 | ρ P5 | ρ 中位数 | ρ P95 | 每源ρ中位数的中位数 |
|---|---:|---:|---:|---:|---:|
| prefill | 896 | 0.951 | 1.011 | 1.115 | 1.010 |
| decode | 1792 | 0.950 | 1.009 | 1.103 | 1.008 |

112 个相关 head 的 query 二阶矩 top1/trace 中位数为 0.676，P5–P95 为 0.420–0.850；uncapped γ 范围 50.59–953.39，中位数 263.83，其中 112/112 超过 cap16。γ来自 rank-one 近似，不能解释为可回收的任务收益；两套部署 metric 常量均为 29,120 B。采集器额外保留原始 FP16 K，属于单独计费的诊断 owner。

profile2 第二源 `docvqa_next_7822` 的组件 GPU 区间如下（ms）。reader FWHT 仅含 query 变换，不含完整 packed attention 读取。

| 方法 | 编码 wall | 编码 FWHT | PQ+gain | V Lloyd | 残差 | reader FWHT |
|---|---:|---:|---:|---:|---:|---:|
| r7_k2_norm | 170.97 | 2.08 | 70.16 | 22.02 | 29.30 | 29.20 |
| r7_k2_projection | 172.49 | 2.02 | 71.48 | 22.61 | 30.11 | 25.13 |
| r7_k1_projection_fixedK2 | 158.00 | 2.61 | 39.29 | 32.31 | 29.84 | 26.31 |
| r7_k1_fixedK2 | 155.21 | 2.65 | 36.20 | 32.44 | 29.76 | 25.24 |

首源 Norm 的首个编码 FWHT 事件为 411.98 ms，对应 CPU wall 411.88 ms；这是孤立首调用异常，可能含编译或 host launch 等待。不能把该点或两源插桩 session median 当作稳定速度差。正式 quiet 工具另做 warmup、同八个强制输出 ID 和三次测量，只计成本。

第二骨干开发点：Full 75.12、K2 Vfirst 62.90、K1 Vfirst 62.66、K1 Kfirst 49.65 ANLS。实际 stage 计数符合冻结 incumbent planner：K2 Vfirst 为 0K/403–410V；K1 Vfirst 约 289–297K/1023–1024V，Kfirst 为 1024K/289–296V。N=2144 的 Vfirst 保留 K297/V1023，是 incumbent K 保留规则的结果，不能写成严格全序 V 优先。V 修复改善了该骨干的探索点，仍比 Full 低约 12pp；这十六个已观察来源不增加独立确认来源数。

可复核数值、逐源统计和原始文件 SHA 见对应 audit JSON。
