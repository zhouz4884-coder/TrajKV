# Raw inference Runner v1

入口：`python -m evaluate --config <immutable_global_job.json> --output <new_directory>`。`--plan` 只验证模型 revision、输入 schema、选择范围和方法 allowlist，不加载 GPU，也不读取参考答案。输出目录必须新建；原始 record、manifest、complete marker 不覆盖。

最小配置：

```json
{
  "schema": "baseline_matrix.eval.v1",
  "model_id": "M1",
  "model_path": "<verified_checkpoint_directory>",
  "model_source_lock": "<frozen_HF_tree_lock.json>",
  "inputs": "<BROOT>/data/registry/v3/textvqa_p1_inputs.jsonl",
  "start": 0,
  "count": 1,
  "max_new_tokens": 64,
  "question_suffix": "\nAnswer the question using a single word or phrase.",
  "preprocessing": {"max_patches": 8},
  "method": {"id": "F_native_split", "family": "fp16"}
}
```

`model_source_lock` 默认是 BROOT/configs/models 下的 M1_hf_reference_lock_v1.json 或 Qwen hf_revision_lock.json；Runner 对原始 metadata Git blob 和每个 safetensors 完整 LFS SHA逐文件核验。FP16权重、eval、SDPA与 native greedy generation固定；保留官方 EOS 和 repetition penalty。Qwen使用 `qwen_protocol.py` 的原生三轴位置/Delta/DeepStack路径，M1使用 `m1_protocol.py`。

方法 family 的保留值是 `fp16`/`native`（主 NativeDynamicSplit）、`fp16_hybrid`（技术桥）、`full_native`（独立完整提示参考）。其它 family 必须配置 allowlist 中的 `adapter_module`、`adapter_class`、`constructor_args`、`reader_backend`。允许 KIVIStyleAdapter、CalibQuantAdapter、明确命名的 CalibQuantFP32BiasAdapter、TurboQuantAdapter 和 PyramidKVVisualAdapter。T默认 constructor seed=42+6*i，再encode(layer_idx=i)，得到源实例42+7*i；E默认 encode_scope=prefix，并核实际层数。query-known 另需独立选择-workload driver，本版拒绝冒用 unknown。

每个 QA 重新读原图并核 encoded SHA，保留 raw question，同时将统一 question_suffix 组成 rendered question；两者分别存 SHA。默认再以另一个问题核 image-prefix token和native视觉处理不变。fresh原生 prefix后只编码视觉 KV；全部问题 suffix由原生 generate按逻辑 cache length重新计算，没有继承未压缩首答案 logits。E捕获的是真实 question-free prefix末W个post-RoPE Q。

编码后 Hybrid消费原 native full-prefix cache，weakref与storage alias同时验证释放。消耗后的 pixel/deepstack输入不保留，完整三轴map移CPU；Qwen只保留native tiny delta状态。全局 storage 去重覆盖所有层/共享矩阵。complete_visual_owner_bytes包括payload、coder metadata、全部视觉positions、必要Qwen delta与encoder constants；LUT是constants子集，不能重复加。exact text KV与text_positions单列。原生F只有分配容量严格等于逻辑K/V且没有alias时，才能按不相交视觉/文本区域分配物理字节。

原始 outer forward logits 每次检查 finite，未做 clamp/nan替换。临时 generate observer只加入streamer和返回类型参数，记录原生生成 token边界，核 streamer IDs与generate返回完全一致；raw prediction不strip。时延包括有限性检查、同步、prefix独立性核验等 instrumentation，属于实际prototype quality run，不是作者canonical kernel速度。正式quiet timing由独立矩阵配置控制。

每QA失败保留status、exception、None预测，继续其它QA；runtime/model加载失败也为所有expected IDs写失败记录。complete.json核完整coverage与records SHA，明确失败必须保留在primary分母。Runner不读取refs、不评分、不实现prediction复用；后续独立scorer/immutable contractcache由raw record与同run manifest合并身份字段。

CPU验收：`results/p0/evaluate_ownership_cpu_v1.json`检查native区域、Hybrid完整位置账与共享常量只计一次；`evaluate_expected_cpu_hardware_failure_v1`确认无CUDA时不会产生假GPU答案，仍保留完整expected coverage。GPU各方法首题smoke由主调度解锁，CPU这些记录不是模型质量结果。
