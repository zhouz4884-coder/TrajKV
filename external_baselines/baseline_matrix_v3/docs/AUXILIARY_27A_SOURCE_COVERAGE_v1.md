# 已完成辅助证据的 27a 覆盖核查

固定 commit `27a88288e18032ce88de1fe19392197ff80e656c`；public index 与 release archive SHA 已匹配。仅核已有证据及小型执行源证明，没有运行 GPU、没有新增推理、没有读取 GT，也未下载 PT/原图/权重。

| 分组 | 类型 | operator/gate cases | 输入 QA / freegen arms | raw 已在27a | 新 sourceproof |
| --- | --- | ---: | ---: | ---: | --- |
| native_fixed_query_60 | CPU_REAL_NATIVE_FIXED_QUERY_OPERATOR_CONTROL | 60 | 0 / 0 | 0/3 | PASS_EXECUTED_SOURCE_INSPECTION |
| native_T_R0_R128 | GPU_MODEL_DIAGNOSTIC | None | 4 / 24 | 2/2 | PASS_EXECUTED_SOURCE_INSPECTION |
| C2_source_roundoff_replay | GPU_SOURCE_OPERATOR_REPLAY_NO_MODEL | None | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| C2_forward_roundoff_bound | CPU_FROZEN_CAPTURE_FORWARD_ERROR_AUDIT | 23 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| C2_max_gap_trace | CPU_FROZEN_CAPTURE_LOCAL_TRACE | 1 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| calib_affine | GPU_MODEL_DIAGNOSTIC | None | 2 / 10 | 2/2 | PASS_EXECUTED_SOURCE_INSPECTION |
| late_question_boundary | GPU_MODEL_BOUNDARY_CONTROL_DIFFERENT_INFORMATION_CONDITIONS | None | 2 / 12 | 0/2 | PASS_EXECUTED_SOURCE_INSPECTION |
| OnlyTerp_MSE_CUDA_gate | GPU_SOURCE_OPERATOR_GATE_NO_MODEL | 6 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| OnlyTerp_prod_CUDA_gate | GPU_SOURCE_OPERATOR_GATE_NO_MODEL | 5 | 0 / 0 | 0/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| OnlyTerp_prod_probe | GPU_MODEL_DIAGNOSTIC_NOT_FORMAL_PREDICTION_PRODUCER | None | 2 / 6 | 0/2 | PASS_EXECUTED_SOURCE_INSPECTION |
| KIVI_g32R32_CUDA_gate | GPU_SOURCE_OPERATOR_GATE_NO_MODEL | 48 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| KIVI_g128_CUDA_gate | GPU_SOURCE_OPERATOR_GATE_NO_MODEL | 80 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| CalibQuant_CUDA_gate | GPU_SOURCE_OPERATOR_GATE_NO_MODEL | 12 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |
| TurboQuant_eviction_CUDA_gate | GPU_SOURCE_OPERATOR_GATE_NO_MODEL | 8 | 0 / 0 | 1/1 | PASS_EXECUTED_SOURCE_INSPECTION |

每个文件的精确路径、原始字节 SHA、27a index/member/source-identity 三项检查均在 `audit/auxiliary_27a_source_coverage_v1.json`。新取回的执行源证明在 E 盘 `results/auxiliary_source_proofs_v1/<job>/`，包括原 launch、原 job、原 source inventory、原 driver 与报告声明的源码；独立 remote SHA receipt 与文件字节逐个核对。

CPU60 使用已有 4-source × 5-layer native Q/K/V 捕获做 3 个编码变体，共60个固定查询 operator case，新增 GPU QA 为0。各 GPU 模型诊断的 arms 是同输入上的控制分支，不是新的 benchmark 问数；late-boundary 两条件的信息时刻不同。operator CUDA 门没有加载任务模型，不能计为端到端任务得分。

下一公开包应把 raw 小 JSON/JSONL/CSV 和对应执行源证明作为独立 auxiliary 分组导出；报告已公开不能替代 raw/source 已公开。OnlyTerp12/23 各64 pilot 的原始闭包由独立 phase 合同负责，不与本表重复计数。公开派生路径与 canonical 私有原始 SHA、Git LF 正文身份、归档身份分别记录。
