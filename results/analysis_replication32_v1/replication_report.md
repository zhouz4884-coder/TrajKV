# 预算标定后的独立 replication32

冻结合同保留 32 原图、12 配置、每配置 320 题，共 3,840 个回答。阈值只来自原 dev8 的无升级 online δ/cost；没有用本批结果重拟合。

主要新策略 `asym:probe_l7_t0.17655292031960651` 对 Periodic：Δ -0.625pp，95% image-cluster CI [-2.188,+0.938]；修复/新增错 3/5，sign-flip p=0.688062，Holm11 p=1。

预先安排的新旧 Probe7 配对：Δ +0.312pp，CI [-0.938,+1.562]，修复/新增错 4/3，raw paired p=1。这是单列迭代比较，不替代 Periodic 主 11 比较家族。

| 方法 | 正确/320 | Exact | 对 Periodic Δpp [95% CI] | Holm11 p | actual extra MiB/会话 |
|---|---:|---:|---:|---:|---:|
| Full FP16 | 202/320 | 63.125% | -0.938 [-3.438,+1.562] | 1 | 0.0000 |
| Base K2/V2 | 204/320 | 63.750% | -0.312 [-1.875,+1.250] | 1 | 0.0000 |
| Native K4/V4 | 204/320 | 63.750% | -0.312 [-2.812,+2.188] | 1 | 0.0000 |
| Asym base read | 204/320 | 63.750% | -0.312 [-1.875,+1.250] | 1 | 0.0000 |
| Uniform | 206/320 | 64.375% | +0.312 [-1.875,+2.812] | 1 | 11.2126 |
| Periodic | 205/320 | 64.062% | reference | — | 15.9785 |
| Current table | 205/320 | 64.062% | +0.000 [-2.500,+2.500] | 1 | 10.7051 |
| Future table | 203/320 | 63.438% | -0.625 [-2.500,+1.258] | 1 | 10.7051 |
| Probe L7 original .02 | 202/320 | 63.125% | -0.938 [-2.188,+0.000] | 1 | 15.9649 |
| Probe L0 budget-calibrated | 202/320 | 63.125% | -0.938 [-2.500,+0.625] | 1 | 14.1719 |
| Probe L7 budget-calibrated | 203/320 | 63.438% | -0.625 [-2.188,+0.938] | 1 | 14.3393 |
| Probe L14 budget-calibrated | 204/320 | 63.750% | -0.312 [-1.875,+1.250] | 1 | 14.7715 |

## 冻结阈值的真实选择性

| 策略 | phase | 观察/调用机会 | 升级请求/观察 | 预算跳过 |
|---|---|---:|---:|---:|
| Probe L0 budget-calibrated | decode | 671/685 | 24/671 | 14 |
| Probe L0 budget-calibrated | text_prefill | 284/288 | 19/284 | 4 |
| Probe L14 budget-calibrated | decode | 671/684 | 105/671 | 13 |
| Probe L14 budget-calibrated | text_prefill | 285/288 | 11/285 | 3 |
| Probe L7 original .02 | decode | 276/683 | 276/276 | 407 |
| Probe L7 original .02 | text_prefill | 108/288 | 108/108 | 180 |
| Probe L7 budget-calibrated | decode | 663/684 | 68/663 | 21 |
| Probe L7 budget-calibrated | text_prefill | 283/288 | 0/283 | 5 |

预算校准恢复了选择性；这不是质量收益本身。实际 mean extra 由观察到的 probe、升级和硬 cap 共同形成，不能将缺失 probe 当作零信号。原始账全部独立复算，资源与自然生成时间分开保存。

## 固定 probe floor 的作用域

dev8 无升级轨迹中，2-bit Asym mean fixed probe extra 为 12.5488 MiB，1-bit 为 60.4065 MiB。1-bit 的长错误回答增加了探针调用次数；4–32 MiB 无解只针对这些观察轨迹、采集所有探针且仅阈值化后半层升级的预测模型。它不证明所有 1-bit 部署数学不可行。

独立审计见 identity_audit.json；原64评测和其结论保留，Doc确认数据没有用于本分析。
