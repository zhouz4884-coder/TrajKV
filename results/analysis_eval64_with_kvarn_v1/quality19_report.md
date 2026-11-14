# 冻结主评测与 KVarN64 质量参考

19 配置 × 64 图像 × 10 轮，共 12,160 个回答。原 17 配置的报告、图与 Holm16 主结论保持原样。

KVarN2/4 为 †materialized mechanism-port，只作为质量参考；两方法对 Periodic 的结果属于独立 Holm2 secondary 家族。

| 方法 | 正确/640 | Exact [95% image CI] |
|---|---:|---:|
| Full FP16 | 392/640 | 61.250% [57.344, 65.156] |
| Base K2/V2 | 386/640 | 60.312% [56.563, 63.906] |
| Native K4/V4 | 392/640 | 61.250% [57.340, 65.156] |
| Asym base read | 386/640 | 60.312% [56.563, 63.906] |
| Uniform | 388/640 | 60.625% [57.031, 64.375] |
| Random | 392/640 | 61.250% [57.188, 65.156] |
| Periodic | 388/640 | 60.625% [56.563, 64.531] |
| Text prefill | 390/640 | 60.938% [57.031, 64.844] |
| Early | 387/640 | 60.469% [56.406, 64.375] |
| Margin replay | 387/640 | 60.469% [56.563, 64.219] |
| Margin write | 386/640 | 60.312% [56.563, 63.906] |
| Carrier write | 386/640 | 60.312% [56.563, 63.906] |
| Current table | 385/640 | 60.156% [56.250, 64.062] |
| Future table | 389/640 | 60.781% [56.875, 64.531] |
| Probe L0 / .05 | 388/640 | 60.625% [56.719, 64.375] |
| Probe L7 / .02 | 386/640 | 60.312% [56.406, 64.062] |
| Probe L14 / .02 | 382/640 | 59.688% [55.781, 63.594] |
| KVarN2 † mechanism-port | 388/640 | 60.625% [56.875, 64.375] |
| KVarN4 † mechanism-port | 389/640 | 60.781% [56.719, 64.688] |

## KVarN 对 Periodic 的 secondary 比较

| 方法/范围 | Δpp [95% CI] | 修复/新增错误 | MC p | Holm2 p |
|---|---:|---:|---:|---:|
| kvarn2:base / all_scored | +0.000 [-1.562, +1.562] | 17/17 | 1 | 1 |
| kvarn4:base / all_scored | +0.156 [-1.406, +1.562] | 12/11 | 1 | 1 |
| kvarn2:base / later_turns | +0.000 [-1.736, +1.736] | 17/17 | 1 | 1 |
| kvarn4:base / later_turns | +0.174 [-1.562, +1.736] | 12/11 | 1 | 1 |

KVarN2/4 的真实视觉 owner 分别为 FP16 visual 的 1.1484375/1.2734375：其中保留物化 FP16 值和编码参考。不能将 encoded-reference ratio 当作当前运行时显存节省，也不与 packed 方法做部署延迟/吞吐优劣推断。

Periodic/KVarN2/KVarN4 的 640 行同题三方答案、正确性组合及三组配对翻转全部保留；pattern 位顺序为 Periodic/KVarN2/KVarN4。Full/Base2/Native4 的辅助配对另表保存。最终数据仅用于报告，没有参与阈值拟合。

独立身份核对见 kvarn_identity_audit.json。子 manifest 的 preflight/prepared 与原 candidate split 标签保留为已知旧字段，实际完成与同 64 图身份由 complete、运行记录、冻结样本哈希及 initial-logit 对齐另外核实。
