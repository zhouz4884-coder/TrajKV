# KIVI 源示例默认点的追加 P2 准备

只准备 `K2/V2, group_size=32, residual_length=32, triton_packed`，不新增其它宽度，不修改原 g128/R0 点、生产模块、reader 或 evaluate。该点标为固定 KIVI `example.py` 超参数与 prefill policy 的视觉移植：视觉前缀不再滚动，文本 KV 保持 FP16，不能称原始 whole-KV rolling Llama 系统。

完整预定目标见 `configs/matrix/P2_kivi_example_g32R32_full_targets_v1.json`：三模型 ×TextVQA/InfoVQA/DocVQA/ChartQA，12格 ×500题。沿用各模型既定 checkpoint、processor、pixel/cap、prompt suffix 和生成配置；同一 registry v3 的固定500-source问题不变。这里没有读取 GT 选题，也没有启动 GPU。

当前实际/条件状态见 **`configs/matrix/P2_kivi_example_g32R32_targets_v2.json`**，不要使用早期 v1 的 queue-marker 表述：

| 范围 | 实际证据/状态 | 复用 | 新跑 |
|---|---|---:|---:|
| M1 Text500 | 已核真实同模型 probe2 的原始输出、源快照和有效合同 | 2 | 498 |
| M1 Info/Doc/Chart 各500 | 以同 M1/方法/执行合同作为实际 reference；这些域无相同输入命中 | 各0 | 各500 |
| M2/M3 四域各500 | 尚无该方法的同模型整链 reference，条件性目标 | 当前已知0 | 待 smoke/refresh 后形成实际计划 |

M1 已准备四个可运行 missing-only job，共1998题。源 probe2 中的真实预测属于相同推理协议，只有其实际命中的两个 Text问题被复用。完整缺失 ID、输入锁、源行 SHA 和有效合同都在 E 镜像 `results/cache_plans/kivi_example_p2/v2/`。

M2/M3 分别准备了 TextVQA P2 前两题 technical smoke：

```text
configs/jobs/bm_M2_K2_g32R32_textvqa_probe2_v1.json
configs/jobs/bm_M3_K2_g32R32_textvqa_probe2_v1.json
```

其余八个500任务目前是 `configs/pending_jobs/*conditional_v2.json`，没有顶层 `name/argv`，不能直接交给 launcher。它们要求对应模型 smoke 的2/2 PASS、独立 raw/source SHA 凭据和一次实际 cache-plan refresh。没有借 M1 的合同伪造 M2/M3 的 context 或 cache hit。

母任务安排 smoke 后，先取回原始三文件与源凭据，再刷新：

```text
python -B ops/fetch_raw_outputs.py results/p1_smoke/M2_K2_g32R32_textvqa_probe2_v1
python -B ops/fetch_prediction_sources.py ${EVIDENCE_ROOT}/results/p1_smoke/M2_K2_g32R32_textvqa_probe2_v1
python -B ops/prepare_kivi_example_p2_v1.py --refresh-model M2 --reference-run ${EVIDENCE_ROOT}/results/p1_smoke/M2_K2_g32R32_textvqa_probe2_v1 --run-id M2_after_smoke_v1
```

M3 使用对应路径和新的 run-id。refresh 将依据真实 `SourceRun` 写新的四域计划与 missing-only jobs；Text首2题若合法命中则变为2+498，其它域按实际输入命中计算，不预先写死。失败或不同合同不能当作 cache hit。所有运行目录和新缓存指向远端8TB项目盘；本地新 raw/计划在 E。配置内 GPU编号只是草案，租约与排队仍由 Root 控制，本准备工具不修改 controller。

源示例默认点的真实有效位数随视觉 N、K/V尾部、g32元数据和位置费用变化。M1 probe2 的约3.18/3.23 bits 只是这两条实测，不能填到其它模型或500题上。它是更强质量参照，不能用更高实际字节冒充1.3/1.8 iso-budget。

`audit/kivi_example_p2_preparation_acceptance_v1.json` 验证了 M1 的2/498和三域0/500、原生模型/输入配置一致、条件任务不可直接启动、跨模型 reference 被拒绝及数据盘缓存路径。仅CPU准备验收，不是新的CUDA或模型质量结果。
