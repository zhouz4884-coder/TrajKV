# Round6：真实两级残差、完整字节审计与 RD 深度基线

日期：2026-11-15。本文只分析现有 `results/round6_v1/depth_doc64_mechanism`、`depth_rd_fit` 与对应自由答案评分；不修改原始记录、不运行新 GPU 实验。

**现有证据支持真实深度—覆盖交换继续推进。** Doc dev64 上，固定 K 子集、完整字节相同的 V 两级方案得到 92.1802 ANLS，宽覆盖方案为 90.8679，差 1.3123 pp。实际第二级能降低已有残差误差：所选 Doc V 单元的 `D2/D1=0.40422`；cal16 全单元原生坐标测量得到 K 为 0.36346、V 为 0.41506。后二者是码流失真比例，不能解释或替代上述 ANLS 差值。RD 已生成完整合法 448-stage 顺序；本说明不对其尚未纳入本文的自由答案结果作结论。

## 1. 两组失真数据的坐标与样本范围

`depth_doc64_mechanism/curves.json` 来自实际 packed 自由答案运行的编码记录：K 误差在中心化 FWHT 域，V 误差在原坐标；每个单元的 SSE 除以 `N×128` 得到源 MSE。只记录实际选中的单元，未选单元仍保留完整 base，但没有在该诊断表中补测其误差。表中 `D2/D1` 只对实际深度为 2 的单元汇总；不以只测到 D1 的单元充当第二级样本。

`depth_rd_fit/curves.json` 使用 cal16 的 16 个唯一图像，各取首问，一图一次 Full native prefill，`teacher_case(target_limit=1)` 不再追加答案 token 前向。每个图像、28 层、4 个 KV-head、K/V 两轴均实际编码至深度 2。K 在 FWHT 域逐级编码后，逆 FWHT、加存储均值、转 FP16，再与原 native K 比较；V 直接与原 native V 比较。先算每图每标量 MSE，再对 16 图等权平均。该 native K 桥接不包含线上 `FWHT(q)` 的 FP16 舍入，不能冒充完整 reader 误差。

以下总体比例均使用相同样本集合中的 MSE 总和相除，不是预设衰减常数，也不是各单元比例的无权平均。

## 2. 实际第二级保留多少误差

### 实际 Doc dev64 选中单元

| 配置/轴 | 图像×单元观测数 | 实际二级观测数 | D1/D0 | D2/D1 | 二级比例的单观测范围 |
|---|---:|---:|---:|---:|---:|
| incumbent / K | 1536 | 0 | 0.351266 | — | — |
| incumbent / V | 1274 | 0 | 0.353838 | — | — |
| v_pair / K | 1536 | 0 | 0.351266 | — | — |
| v_pair / V | 637 | 637 | 0.336586 | 0.404223 | 0.381698–0.474863 |
| v_mixed / K | 1536 | 0 | 0.351266 | — | — |
| v_mixed / V | 957 | 317 | 0.347863 | 0.404398 | 0.383790–0.443409 |

三臂 K 的源误差完全相同，与固定同一组 24 个 K 一级单元相符。`v_pair` 的第二级在已有 D1 上再降低约 **59.58%**；其中每图×单元比例的中位数为 0.410082。`v_mixed` 的对应中位数为 0.406574。两者的第二级总体衰减接近，但自由答案结果不同，说明一个总体源失真比例不足以选择覆盖与加深的组合。

V 轴的 D1/D0 不能跨行直接当作编码器优劣：三个配置选择的 V 单元集合、覆盖数不同。`v_pair` 只覆盖更靠前的候选，宽覆盖控制还编码了后续候选；这里只比较各自实测范围。

### cal16 全部 112 K + 112 V 单元，native 坐标

| 轴 | 平均 D0 | 平均 D1 | 平均 D2 | D1/D0 | D2/D1 | D2/D0 |
|---|---:|---:|---:|---:|---:|---:|
| K | 0.191952 | 0.067845 | 0.024659 | 0.353448 | 0.363459 | 0.128464 |
| V | 0.813670 | 0.297361 | 0.123424 | 0.365457 | 0.415065 | 0.151688 |

K 第一级降低约 64.66% 源 MSE，第二级再降低约 63.65%；V 分别为 63.45% 和 58.49%。真实 FP16 码流中，第二级并未出现统一的 0.363 比例：V 的第二级衰减更弱。K/V 的绝对 MSE 不具备同等任务敏感性，不能据此单独确定预算价格。

跨 112 个单元，K 的 `D2/D1` 为 0.346490–0.431511，V 为 0.319325–0.486844。按层汇总四个 KV-head，K 第二级比例最低为第 1 层 0.349708、最高为第 0 层 0.385604；V 最低为第 24 层 0.363461、最高为第 11 层 0.457799。层号均从 0 起。完整 layer/head 曲线保留在 `depth_rd_fit/curves.csv`，不能把总体比例广播成每个单元的测量值。

## 3. 真实字节、合法前缀和固定 K 集合

`depth_doc64_mechanism/byte_audit.json` 的 **192 条原始记录全部通过**，覆盖三臂各 64 个 session；64 组逐 session 对照均通过相同完整 bytes、相同 stage 总数与相同 K 单元集合检查。

完整 owner 为：

`visual_base + visual_residual + selector_metadata + visual_position + encoder_calibration_parameters_bytes`。

每个实际存储的 stage 恰为 `N×(16+2)=18N` 字节：128 个 sign bit 加一个 FP16 MAD 幅度/视觉 token。compact pool 只包含选中 stage；第二级先有同轴/层/head 的第一级。三臂均计入 `[28,2,4,2]` int32 lookup（1792 B）、224 项 uint16 单元排序（448 B）、真实五字节策略描述符，以及冻结参数；全部 visual base 始终保留，无原始 FP16 视觉 K/V owner。

| 视觉 N | 图像数 | 每臂 stage 数 | K 一级数 | V stage 数 | 每臂完整 bytes | 余量 |
|---:|---:|---:|---:|---:|---:|---:|
| 1280 | 3 | 42 | 24 | 18 | 8,237,573 | 19,963 B |
| 1792 | 57 | 44 | 24 | 20 | 11,560,453 | 97 B |
| 2304 | 4 | 44 | 24 | 20 | 14,837,253 | 26,311 B |

因此“20V×1 对 10V×2”只描述 stage 预算为 44 的图像；3 个 N=1280 图像实际为 18V×1 对 9V×2。预算未被硬写成 44，也没有通过无用 payload 补齐。

评分主表按总体 owner/分母汇总得到三臂相同 **1.799586589 bit/KV scalar**。逐 session 实际码率为 1.795648411–1.799984835，均满足 ≤1.8。逐 session 码率的算术均值为 1.799583363；它与总体比值的微小差别来自 N 的权重，不是字节账不一致。

`logical_reads.json` 记录层事件中的 base、残差、文本、位置及选择暂存字节。它说明了实际读取范围；自由生成长度变化会改变累计读取量，不能把这些逻辑流量或同一顺序下的 session 中位时间直接解释为 reader 加速。

## 4. RD 强对照的分配实际选择了什么

`depth_rd_fit/complete.json` 验证 16 图、28 层、4 KV-head、448 个唯一 stage、合法前缀。权重直接读取旧 `ranking_scores.json` 的 `per_axis.{k,v}.sensitivity_w`，保留共同 CE 尺度，不对 K/V 分别归一化。

每次从当前合法动作中选最大值：

`w[axis,L,H] × mean_cal16(D_native(d) − D_native(d+1))`。

只有自己的 stage 0 已进入顺序，stage 1 才可竞争。两轴均允许第二级；这比固定 K、只交换 V 深度的动作空间更宽。它是 **RateQuant-style strategy-port**：复用既有 pooled answer-CE 权重，在本项目实际 sign 码流与 greedy 前缀约束下分配；不是官方完整复现，也没有用 dev64 ANLS 拟合顺序。

完整 RD 顺序的前 44 个动作分布为：

| 轴 | 第一级 stage 数 | 第二级 stage 数 | 最终 depth=1 单元 | 最终 depth=2 单元 |
|---|---:|---:|---:|---:|
| K | 19 | 10 | 9 | 10 |
| V | 12 | 3 | 9 | 3 |
| 合计 | 31 | 13 | 18 | 13 |

第一个第二级出现在第 4 个动作，为 `K/L24/H3/stage1`，其 stage0 在第 1 个动作；第 7 个动作为 `V/L19/H2/stage1`，其 stage0 在第 2 个动作。前 44 个动作中第 22、19、23 层分别占 9、8、7 个，合计 24/44。其余分布为 L24/L20/L14/L16 各 3，L25/L15/L27 各 2，L21/L26 各 1。它没有把第二级限制在 V，也没有因 V 的绝对源误差较大就自动偏向 V。

**前 44 项是优先序诊断，不是所有图像的实际 stage 数。** RD 与新增 incumbent envelope 都存 448 个 uint16 stage ID，即 896 B 排序常量；五字节 descriptor 与 lookup 不变。envelope 将旧 224 单元的全部 stage0 排序接上对应 stage1，提供同格式、同常量字节的宽覆盖控制。相对 named 三臂，排序多 448 B，必须重新算 budget：

| N | 同格式 explicit 可承担 stage | RD 实际前缀分布（K0/K1/V0/V1） | 同格式 envelope 前缀 |
|---:|---:|---|---|
| 1280 | 42 | 18 / 9 / 12 / 3 | 24K0 + 18V0 |
| 1792 | 43 | 19 / 9 / 12 / 3 | 24K0 + 19V0 |
| 2304 | 44 | 19 / 10 / 12 / 3 | 24K0 + 20V0 |

其中 N=1792 完整 owner 为 11,528,645 B：448 B 常量超过原来仅剩的 97 B，需少存一个 32,256 B stage。这是合法整数码预算的实际结果，不能将 RD 直接与 named 44-stage 配置描述成严格同格式匹配。RD 与 envelope 使用同一 explicit bank 的实际 owner 检查，才是这一强对照的精确匹配。

## 5. 失真改善与 ANLS 分开解释

现有自由答案评分是开发集证据：64 个已用于开发的文档，255 个答案/配置，incumbent 90.8679、v_pair 92.1802、v_mixed 90.8192 ANLS。固定 K 与完整 bytes 排除了新增总残差数、额外视觉基底或遗漏辅助常量这些混杂；它支持“在这组开发样本上，所测 V 深度—覆盖交换改变了实际答案质量”的结论。

源 MSE 表尚未包含同一学生 q、完整视觉/文本 softmax、GQA 汇合、W_O 和后续层的损失传播。加深的单元与失去覆盖的单元还不同；只看被选单元的 D2/D1 不能计算全缓存任务效用。两级源误差下降，也可能通过注意力重分配、非线性、误差相消及生成轨迹变化产生不同答案。因此不能把“第二级减少约 60% 剩余 V 源 MSE”改写成“ANLS 提升由该比例解释”，也不能据此宣称全局最优分配、显著性或未见文档泛化。

RD 的 `ΔD×w` 是更强、可部署的分配竞争者，但仍是可检验代理。它复用的 Full native CE 权重不等于当前压缩学生状态的精确有限修复收益，native K materialization 也省略了线上 query-WHT 的 FP16 舍入。它是否改进自由答案，须由相同完整字节的实际 packed 运行判定。

## 6. 独立 reader/encoder 舍入核对

本次只读复核未发现两级舍入或 compact slot 映射错误。

1. 编码器每一级先计算 `error = source.float() − decoded.float()`，以每 token 128 维 MAD 转 FP16 存 alpha，按 `error>=0` 打包 sign；`decode_sign_stage` 执行 `FP16(float(decoded)+sign×float(alpha))`。第二级 error 明确基于这个已舍入值。
2. Triton K/V 分支各自先完成第一级加法并 `.to(tl.float16)`，随后第二级再 `.to(tl.float32)` 加第二级幅度，最后转 FP16。两级间舍入点与编码器相同；launch 使用 `enable_fp_fusion=False`。K base 同样为 FP16(book×final gain)，V base 的 multiply/add 与 `decode_last_generic` 一致，分别保留 FP16 结果。
3. lookup 为每轴/head 两个独立 int32 slot；reader 以 `owner*2 + stage` 读取，pool 偏移以每层实际 compact index×N 为准。stage0/1 和 K/V 同 head 可同时存在；合法性由编码规划器验证后存储。
4. 两级只在同一视觉块中逐级更新 K/V，随后与文本块共用一个 online-softmax 的 `m/l/acc`；未新增单独的残差 softmax。

现有 CPU 控制 `outputs/round6_depth_cpu_control_20261115_b/check.json` 通过整数 sign/幅度往返与真实级间舍入检查，且 88 个测试 token 的第二级 alpha 与“跳过第一级 FP16 舍入”的错误实现不同。这证明控制确实能识别该类舍入错误；CPU 控制不代替 GPU reader 数值证据。本说明没有启动新 GPU 检查。

## 7. 可复核入口与来源

- 实际源失真和字节审计：`results/round6_v1/depth_doc64_mechanism/{curves.json,curves.csv,byte_audit.json,unit_counts.csv,logical_reads.json,manifest.json}`。
- 实际自由答案评分：`results/round6_v1/depth_doc64_analysis/{table.csv,summary.json}`，标为 development。
- 全单元 native 曲线和冻结顺序：`results/round6_v1/depth_rd_fit/{curves.csv,curves.json,cases.jsonl,policy.json,incumbent_envelope.json,manifest.json,complete.json}`。
- Doc 原始 records SHA256：`17745586b589d0e366252719b56fcb7064c875bd6271111fd649cef7680c2749`。
- RD policy SHA256：`9721b13de75e0d82b98ae77e18c4920ee20f58443d0e0c5014d17934b2a5406e`。
- RD incumbent-envelope SHA256：`f1772c27f74efcb2756e2bd3c103a9882c1511d9cd1cc983a6cdaaa480b4fa2b`。
- RD curves SHA256：`ecbf6914b4cd20d6d885e9e99605f1a5871e5fa864d7e929236052e3881b3a17`。
