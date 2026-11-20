# M2/M3 八格真实 source 兼容性已完成

M2 Qwen2.5-VL-7B 与 M3 Qwen3-VL-4B 的 C1/C2、KIVI 官方 example 点和 OnlyTerp Hmix23 均完成固定 TextVQA 首两题的真实 GPU 端到端推理。八格全部原始状态 PASS，完整取回原始三文件与独立 source/launch 证据，并通过 v2 的本模型、完整协议、精确输入、finite、原位置与实际 owner 检查。每格只有 2QA 技术验收，不据此作质量排名。

| 模型 | 源方法点 | 两题实际完整 bit/elem 范围 | 状态 |
|---|---|---:|---|
| M2 | C1 FP32bias | 1.028407–1.034329 | 2/2 PASS |
| M2 | C2 FP32bias | 2.028458–2.034392 | 2/2 PASS |
| M2 | KIVI b2/g32/R32 | 3.282285–3.318488 | 2/2 PASS |
| M2 | OnlyTerp Hmix23 MSE/QJL0 | 3.008289–3.009154 | 2/2 PASS |
| M3 | C1 FP32bias | 1.032149–1.042577 | 2/2 PASS |
| M3 | C2 FP32bias | 2.032180–2.042617 | 2/2 PASS |
| M3 | KIVI b2/g32/R32 | 3.235244–3.302953 | 2/2 PASS |
| M3 | OnlyTerp Hmix23 MSE/QJL0 | 3.006276–3.007789 | 2/2 PASS |

完整 bit 包含实际持有的视觉 payload、scale/码本/常量、原始位置及公共视觉状态；逐题 denominator、字节数和全部 source SHA 在 `audit/Source_compat_original_source_fingerprints_v2.json`，原始完整 owner 在 raw 中保留。KIVI 的 32 分组元数据与 FP16 tail 已计费，不能用名义 2bit 替代表中完整账。C1/C2 是冻结的 FP16 矩阵加明确 FP32 bias sum/div 端口，保留官方 packed QK/WV 和校准，未冒充作者 BF16 配置。OnlyTerp 点是既有 practical Hadamard、MSE2/3、QJL0 与 FP32 materialize reference reader，不是原论文精确 production 表或优化 kernel 速度。

M2 原生几何为 28 层、28 query heads/4 KV heads；M3 为 36 层、32/8。两者 head dim=128，均使用原生三轴 position 协议。真实输入、processor/pixel 限制、完整 question suffix、生成参数和 upstream 依赖按各自 source 合同逐条核对，不借 M1 或另一 Qwen 模型的合同。v2 CPU 正负控制和局限说明保留在 `audit/compat_extension/compat_smoke_v2_independent_CPU_review.json`。

每个 family 已基于自身真实两题生成四域固定 500 题的实际 missing 计划：Text 复用 2、待跑 498；Info/Doc/Chart 各复用 0、待跑 500。每模型 16 格合计 8 条实际缓存、7,992 条未来推理；两模型共 32 格、16,000 目标 QA，其中 16 已有、15,984 待跑。计划数不代表已完成的 GPU 量。调度由唯一 operator 接管，CPU 收取阶段没有 stage、launch 或改动 GPU 队列。

终态支持表位于 `results/matrix/Source_compat_M2_M3_eight_smokes_v2.json` 与 `.csv`；两模型聚合 ready 清单位于 E 的 `results/compat_smoke_collection/source_compat_cpu_v2/model_batches/`。正式质量应由同模型同数据集完整固定 500 题评分给出，正式速度用独立受控 timing 阶段；这里没有引入质量门槛、填零或按好坏排除方法。

本次公开支持 cutoff 固定为这八格完成。`configs/publication/Completed_source_compat_and_Calib_cold_support_v2.json` 同时保留先前 M1 K32 两题，共 9 个技术作业、18 条模型/方法回应，仍只复用了相同两道 TextVQA 输入。它们不加入 native 15,650、P2 500 题主表或 OnlyTerp real64 的主质量计数；v1 支持清单和全部原件保留。
