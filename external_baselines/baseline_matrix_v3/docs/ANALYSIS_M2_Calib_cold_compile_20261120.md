# M2 CalibQuant 首后缀耗时：冷编译证据与同输入热控制

本次只读 CPU 审计支持一个具体解释：M2 首两题的 108–166 秒 question-prefill 主要发生在新的视觉长度触发官方 Triton 候选编译和 autotune 时。它不支持把每道 500 题的稳定耗时都外推为分钟级。尚未用 profiler 分解编译、autotune 与热计算的各自占比。

| 实际分支 | Nv | 后缀 token | encode 秒 | 首后缀秒 | 随后单 token outer-forward 毫秒 |
|---|---:|---:|---:|---:|---|
| M2 C1 FP32bias | 999 | 25 | 0.862 | 136.724 | 111.9 |
| M2 C1 FP32bias | 1225 | 21 | 0.216 | 166.469 | 90.2, 92.2, 87.8, 98.8 |
| M2 C2 FP32bias | 999 | 25 | 0.345 | 108.282 | 110.9 |
| M2 C2 FP32bias | 1225 | 21 | 0.241 | 126.158 | 88.3, 107.1, 87.8, 87.1 |

这些是既有完整 GPU 原始记录中的带审计测量；本审计没有运行新 GPU 推理。其原始文件 SHA 与独立 source proof 已保留。后续单 token 耗时支持冷启动解释，但不能替代相同完整后缀的重复测量。

## 实际编译缓存与源码一致

官方 QK 网格有 `3×3×3×2=54` 个候选，key 为 `K,N`；此处 `K=128,N=Nv`。官方 WV 网格有 `4×3×2=24` 个候选，声明 key 为 `K,ROUNDED_K,N`，此处 reduction `K=Nv`、`ROUNDED_K=next_power_of_2(Nv)`、输出 `N=128`。`BITS`、打包因子与这些维度均进入 JIT 特化。

在 C1 真实启动至完成区间，缓存新写 QK 108、WV 48 个 SM86 编译 metadata；C2 相同。每个分支都恰好是 Nv=999 和 Nv=1225 各 54+24 个。全部 312 个 metadata 的 TTIR 源位置分别指向对应真实 smoke 的不可变 source snapshot，没有借其它作业的编译数作证据。完整既有缓存观察原件位于 `${EVIDENCE_ROOT}/results/diagnostics/calib_compile_static_v1/remote_existing_cache_metadata.json`，SHA 为 `6468e0c9e04a626dac9e56ad9b35563e974598336f810083cd24e6c1585cd7df`。

`_load_upstream` 在进程内缓存一个 module。每层同一视觉 Nv 不随文本追加增长；后缀 token 索引、完整文本 prefix 长度、RoPE position 都不是这两个 kernel 的参数。每个 t 的输入已经 contiguous。因此没有源码依据认为每个 t 或每一层都重新触发全部候选调优。

Triton 3.0 的 autotune winner 只保存在进程内 `self.cache={}`，与磁盘编译二进制缓存不同。新作业即使命中磁盘二进制，也会重建进程内 winner 并 benchmark 候选。Nv 变化会同时改变 QK 输出维度和 WV reduction，可能产生新的编译和调优。

有两处必须按实际运行库解释：WV 的 heuristic `ROUNDED_K` 在外层 autotuner 构造 key 时尚未加入；Triton 3.0 的过滤与下标实现使有效 tuple 包含 `Nv,128,total_matrices` 和 dtypes。Nv 仍然决定新 key，kernel 数学不受这处 key 结构影响。另外，官方 autotune key 没有 BITS：JIT 二进制按 BITS 区分，但同进程 C1/C2 可共享相同 Nv 的 winner。这是冻结来源的行为，本阶段不改它。

## 热路径仍有真实成本

M2 的 28 query heads 对应 4 KV heads。源兼容 GQA 路径先把 packed KV 与通道元数据 repeat 为 28 heads，重复因子为 7。这个过程保持既有分组语义，但读期间有重复 packed 临时量，无法视为原生共享 KV 的最优带宽实现。官方 InternVL patch 也使用 repeat-kv；这里没有偷偷换成展开 FP16 的读取器。

官方 kernel 每次只接受一个 query。长度 T 的完整后缀要进行 `2×28×T` 次 QK/WV 调用：T=25 时 1400 次，T=21 时 1176 次。repeat 在每次 layer read 的循环外；kernel 启动循环、PyTorch 校准/softmax、精确文本注意力则都是热成本。冷编译解释不会把这些成本消掉。

## 已准备的真实控制，尚未运行

Root 指定的控制为 M2 固定首两题、F/C1/C2 三方法，每 QA 每方法 warmup 1 次、measure 3 次，共 24 trials。原生模型只加载一次，每次重新处理原图、计算 native prefix、编码并读取完整后缀、自由生成。使用冻结的 `benchmark_speed.py`，6 条 expected 是经 `SourceRun` 验证的既有 MODEL token IDs，未读 GT。GPU 必须独占，默认 `host_quiet_asserted=false`，不清共享缓存。

- 可调度配置：`configs/jobs/bm_M2_text2_calib_coldwarm_diagnostic_v1.json`，默认 GPU1，需 P 在合法空窗调度。
- 速度配置：`configs/speed/M2_text2_F_C1_C2_coldwarm_diagnostic_v1.json`。
- 原始 expected：`${EVIDENCE_ROOT}/results/diagnostics/calib_compile_static_v1/expected_M2_text2_F_C1_C2_coldwarm_v1.json`；D 的小副本只供 staging。
- CPU 验收与精确 stage 清单：`audit/M2_calib_coldwarm_preparation_v1.json`。
- 静态审计：`audit/calibquant_M2_cold_compile_CPU_v1.json`。

这里 warmup 是相对于新进程 winner 状态的首次试次，不保证磁盘编译冷；三方法交错也继承官方跨 bit winner 共享。应逐条报告 warmup、三个 repeat、TTFT/完整 suffix、生成长度和 MODEL IDs 是否一致。它是两题的热冷诊断，不能作为大样本正式速度结论。方法、Triton 源码、编译配置、容差和 GPU 队列均未修改。

## 优先执行的 v2：各 bit 独立进程

Root 随后授权把各 bit 的进程内 winner 冷热独立出来。冻结 driver 明确要求至少两个方法，因此采用允许的 F+单个 C 对照，不修改 driver：

| 新 job | 方法 | QA | warmup | measure | 合计 |
|---|---|---:|---:|---:|---:|
| `bm_M2_text2_F_C1_process_coldwarm_diagnostic_v2` | F/C1 | 2 | 4 | 12 | 16 |
| `bm_M2_text2_F_C2_process_coldwarm_diagnostic_v2` | F/C2 | 2 | 4 | 12 | 16 |

两 job 是两个 Python 进程，各自只包含一种 Calib bit，不会把 C2 winner 共享给 C1。共 32 trials 和 8 条已核验 expected MODEL 输出。配置位于 `configs/jobs/`，精确 6 文件 stage 清单与 SHA 位于 `audit/M2_calib_process_coldwarm_preparation_v2.json`，原始 expected 位于 E 的 `results/diagnostics/calib_process_coldwarm_v2/`。CPU 源码/计划/expected 检查已通过，尚未运行 GPU。两 job 默认 GPU1 独占，host quiet 仍为 false，需由唯一 operator P 安排。

磁盘二进制缓存继续保留，所以 v2 同样只检验进程 winner 冷热，不声称磁盘完全冷。原 v1 24-trial 计划及其原件保留；为解释各 bit 冷热，优先采用这个独立进程 v2。
