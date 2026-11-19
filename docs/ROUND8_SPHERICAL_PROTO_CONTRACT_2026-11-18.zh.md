# R8 独立完整模板 cosine 坐标原型

本原型只新增 `round8_spherical_codec.py`、`round8_spherical_runner.py`、`round8_spherical_check.py`。不修改现有 R7/R8 encoder、bank 或 reader；只支持原全局 G4_16 书、norm gain、binary/all V、原完整 ranking 的 43-stage 前缀。

给定实际 centered-WHT 源向量 x、32 个 4D 码字拼成的完整 128D 模板 t，维护 A=x·t 与 U=||t||²。逐 slot 枚举 16 个替换，选择 signed A/sqrt(U) 严格提高的单个 proposal，再用完整 128D 重新计算确认。原 J 是 incumbent，平局不动；仅做一轮或两轮，不宣称局部或全局最优。

理想实数 normmatch 下，正 factor f 对应误差 ||x||²(1+f²-2f cos(x,t))，所以 f 不改变该理想方向排序。实际接受条件另算部署字段：先用原函数 normmatch、gain 舍入 FP16、乘冻结 FP32 factor、再舍入 FP16，最后 `(t * gain.float()).half()`。默认仅当此 proposal 的完整 base-SSE 严格降低才更新 J/gain；若不更新，保留原字段逐位。每 slot 只审查 cosine 最好的一个 proposal，不等价于在 16 个 FP16 候选中求实际 SSE 最优。

此严格降低只针对 encoder 的 centered-WHT base 重建。原 bank 随后重算 K/V sign residual，最终带 residual 的 native 误差、attention 以及任务质量均无单调保证。CPU 中已有 cosine 改善但 FP16 base 变差而被拒绝的反例，也有两轮不动但全局另一码序列误差为零的坐标陷阱，不能用 no-op 宣称无容量余地。

接口：`encode_key(source, book, assignment='row_cosine_cd', amplitude='norm', gain_factor=..., sweeps=1, hard_accept=True, row_chunk=8192)` 返回原 `uint8[...,16]` packed J 和 `float16[...,1]` gain。`assignment='l2'` 直接调用原 encoder，包含原 autograd 行为。新分支不增加持久 tensor 常量或 reader 字段；encoder 额外计算与临时内存必须实测计入，不声称速度收益。

每行复杂度为 O(sweeps·32·(16·4+128))，临时 scratch 受 row_chunk 约束，另有原 codes/gain/output。当前严格比较与 SSE 累加使用 FP64，3090 上可能显著增加编码耗时。元数据按 layer-call 保留 proposals/accept/reject/changed rows、base-SSE 前后及 guarantee scope。

独立 runner 通过构造期间有恢复保障的 dispatcher 仅接管 `row_cosine_cd`，其余调用原函数。原 bank 负责全部 residual 重算和预算。要求 direction book 与 metadata-reference book 文件 SHA 相同，书为原全局 16×4。配置应保留完整 224 项 `selected_stages`，另设 `res_stages_limit=43`，实际选中前 43 项；不能把 priority 截成 43 项而丢掉 ranking 常量。N=1792 CPU owner 控制为 448B ranking、完整 8,315,013B。

配置增量字段为 `key_assignment="row_cosine_cd"`、`row_cosine_sweeps=1|2`、`row_cosine_hard_accept=true`、`row_cosine_chunk=8192`；保持 `method="round8_joint"`、`key_encoding="reencode"`、`key_amplitude="norm"`、`value_encoding="binary"`、`v_group_size="all"`、`joint_gain_refit=false` 和原 43-stage mask。先用 `round8_spherical_runner.py --config ... --output ... --plan-only` 核声明，再由 root 调度独立新 run。

CPU v5 证据：`outputs/round8_spherical_cpu_v5/check.json`、`complete.json`，8 个 dense 候选控制及 tie/no-op/WHT/packed/factor/dispatcher/stub/完整 ranking 控制通过。可选真实 CUDA 算术入口是 `round8_spherical_check.py --gpu --output <newdir>`，覆盖 N32/256/1792、单/双轮和两个完整模板 dense witness；未由本 agent 执行 GPU。该检查不替代模型质量或端到端计时。

Cosine 坐标搜索已有相关 PTQ 工作（root 文献复核的 Beacon，arXiv:2508.20293）；这里把它作为组合基底，研究对象是此冻结 KV 表示下的实际 gain/FP16 guard 与下游任务表现，不主张 gain 公式或坐标搜索首创。
