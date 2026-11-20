# T / E 源码身份、数学适配与执行门禁

审计日期：2026-11-19。这里的 CPU 合成验收只证明算法和存储接口；CUDA 与模型任务质量需要各自的独立记录。

## 固定来源

| 来源 | 固定 commit | 身份与用途 |
|---|---|---|
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | `31660314b229b8d1c29bfddf8b9f5026b5121095` | 选用的第三方 TurboQuant 数学实现，GPL-3.0；不是 Google 官方实现 |
| [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) | `1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306` | 备选审计材料，不用于当前基线 |
| [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) | `6cf5e324be90ea9cedce66e843699b6dc655c0bd` | NVIDIA 官方框架对 PyramidKV/SnapKV 的公开实现；不是 PyramidKV 原作者实现 |

精确源码 SHA256 和参数在 `configs/methods/turboquant_eviction_lock.json`。third_party 原文件不修改。适配器校验所选 TurboQuant 源文件字节，再直接调用上游；不安装 vLLM 或更换旧项目环境。E 的 score/budget 仅抽离模型框架依赖，保留 NVIDIA 数学公式、窗口、池化、GQA 平均和原预算 fallback。

主要论文为 [TurboQuant, arXiv:2504.19874v1](https://arxiv.org/html/2504.19874v1) 与 [PyramidKV, arXiv:2406.02069v4](https://arxiv.org/html/2406.02069v4)。PyramidKV 论文作者页指向 Zefan-Cai/PyramidKV；输入矩阵的 Linking-ai/PyramidKV 页面自述官方但 GitHub 标为 fork，不能据此替代原作者身份核验。

## T：本次执行的精确定义

`TurboQuant_0xSero_K3V2_visual_port` / `K3V4_visual_port` 保留选定社区实现的算法：

1. K：每个 head-dim 向量归一化，CPU 固定 seed Gaussian QR 生成正交旋转；旋转后的坐标用对应维度 Beta 分布 Lloyd-Max 2-bit 标量码本编码。
2. 对原始空间的 K 残差应用独立 Gaussian QJL：存储一位符号和残差范数。解码为 MSE 近似加 `sqrt(pi/2)/d * residual_norm * signs @ S`。总 K payload 是 2-bit MSE + 1-bit QJL。不能关闭 QJL 后仍使用该方法名称。
3. V：原实现的 per-token、group-size=32 非对称 min/max 量化，存 group scale 与 zero，payload 为 2 或 4 bits。这里没有把 V 偷换成别的旋转量化。
4. 全部视觉 token 被压缩；社区缓存类的 recent buffer 改由视觉作用域外的文本缓存承担，视觉 buffer=0。所有文本 KV 仍为原模型精度。此作用域限制明确标为 visual port，不能叫完整原始系统复现。
5. 编码调用上游 FP32 路径，norm、residual norm、V scale 和 zero 实际都是 FP32 并计费。明确标记数值变体 `UPSTREAM_FP32_ENCODER_AND_FP32_METADATA`，不同于默认模型 FP16 输入的元数据精度。上游直接在 FP16 常数 group 上使用 `clamp(min=1e-10)` 会把 epsilon 下溢到零；FP32 路径避免零 scale 和 NaN，不隐瞒元数据成本。

K3/V2 和 K3/V4 的平均 payload 位数分别是 2.5、3.5。**它们不是论文 mixed-channel 2.5/3.5-bit 实验配置复现。** 论文 §4.3 中 `(32*3+96*2)/128=2.5` 算式实际为 2.25，故不按该文字标签推导预算。OnlyTerp 也采用 mixed subset 并附加 QJL，存储代码有未 pack 的整数索引/符号；不能直接套用其 README 压缩率。所选 0xSero `_pack_indices(bits=3)` 会以 4-bit nibble 存，当前 K3 的 MSE=2bit 不触发该问题；未审计 K4 配置不开放。

首轮冻结适配器的默认随机 seed 是 `42+layer_idx`，QJL 独立 seed 为其再加 1000；同层所有 KV heads 共享旋转和 QJL 常量。Pi、S、码本、边界与 QJL 位操作辅助 tensor 均列入 owner（共享 storage 去重）。`positions` 是独立 int64 owner。报告分别给 payload、coder metadata、positions、常量、实际有效 bits 及完整 FP16 分母。逐层汇总时仍必须按 storage 全局去重共享常量。

上述 `42+layer_idx` 是已冻结适配器默认参数的 **seed schedule 变体**，不是上游 cache 默认。后续追查固定源 `TurboQuantKVCache` 和 `CompressedKVStore` 均每层独立实例化 codec，采用 `42+7*layer_idx`；Pi/S 同层跨heads共享，不跨层共享。新增配置 `configs/methods/turboquant_source_instance_v2.json` 在不改 codec 源码和旧锁的前提下，由 runner 每层设构造参数 `seed=42+6*i`，再 `encode(layer_idx=i)`，得到原始实例 schedule。`results/p0/turboquant_instance_seed_source_cpu_v1.json` 对 layer0/1/13/27/35 的 Pi/S 与上游 cache 实例逐元素严格误差0。10张 Pi/S 为10份独立 storage；只有全局 per-device QJL bit powers 共1份8字节。旧 codec gate 保留原 seed 标签，不追改成 source-cache schedule。

### reader 与实际 GPU 内存

默认 `torch_dequant` 参考 reader 每次从 uint8 packed owner 调用上游 dequantize，再用 Torch grouped attention 读取。无解码 KV 持久副本；每次 reader **会完整临时解压 FP32 K/V**，还产生投影、logits、softmax、文本拼接等临时分配。`temporary_full_dequant_KV_bytes` 是解码 K/V 两张 tensor 的准确字节，不是全部峰值，不作低内存 kernel 峰值或速度保证。端到端必须另测 CUDA peak alloc/reserved、TTFT 和 wall time。该参考 reader 的慢速不能当作 TurboQuant 规范内核的速度。

显式 `upstream_triton_score` backend 直接调用选定仓库 `turboquant_attention_score` 两个 packed-K Triton kernels，逐 query / GQA group 做兼容 dispatch，没有自行改 kernel；K 不完整解压。V weighted-value 仍调用上游完整 FP32 V 解码，因此不能称全 KV fused reader，临时完整 FP32 V 字节为 `4*B*Hkv*N*D`。该 backend 不可用时抛出原错误，不自动切回参考 reader。压缩 owner 相同，backend 必须在每次运行 manifest 明确记录。

另核查上游 `turboquant_fused_decode`：原包装函数先将 packed V 完整解包成 uint8 `[B,H,N,D]`，其后在线 softmax；它不返回 log-sum-exp 且只支持一个 Q 与同数 query/KV heads，不能直接同本协议仍精确的 text/suffix 合并。它作为独立 visual-only kernel gate 验证，不冒充已支持主协议全链。未来速度对照必须同时保留 native FP16 和同参考 reader 的无压缩桥。

所需算子是 Torch FP32/FP16 matmul、searchsorted、位运算；不要求 FP8/SM100。RTX3090 SM86 实跑仍是独立硬门禁，不能从算子列表推定已通过。

## E：NVIDIA PyramidKV 的视觉候选限制适配

端口输入必须是完整视觉前缀 K/V，以及同一原生前向产生的 **post-RoPE 最后 W 个 Q** 或对应真实 attention。不能把 K 当 Q，不能以 norm、uniform、random、topmass 或其它排名替代。模型适配层负责处理原生 q-norm、RoPE、mRoPE 等，再传入 Q；未核对这些步骤不能报适配通过。

保留 NVIDIA `SnapKVPress.score`：先对最后 W 个 Q 与全部 prefix K 做带正确因果 mask 的 softmax，再切去最后 W 个 keys，平均观察窗口，avg_pool1d(kernel=5)，按 GQA group 平均；最后 W 个 token score 设为大于已有最大值。默认 W=64、beta=20，这是固定 NVIDIA 实现的默认值；论文实验采用 alpha=8，不能将 W64 写成论文默认。

保留 NVIDIA `PyramidKVPress.get_layer_budget` 完整计算，包含其说明的短序列 SnapKV fallback。然后施加本实验 **文本全部精度保留，只有视觉候选可移除** 的明确限制：`visual_keep = clamp(original_full_prefix_layer_budget - text_tokens, protected_visual_window_tokens, visual_tokens)`。这可能增加上层实际预算或导致部分层零视觉 token；不重分配、不为命中 iso-byte 改选别的算法。每层保留数与真实 bytes 必须记录。

**单图长度尤其需要暴露源码 fallback。** 例如 W64、beta20、保留比例 0.125 时，`min_num>=W` 要求 prefix N 至少 10240，常见较短单图会进入 NVIDIA 原始 `SnapKV` fallback，层间预算相同。owner 的 `config.budget_regime` 明确输出 `UPSTREAM_EXPLICIT_SNAPKV_FALLBACK` 或 `PYRAMID_LAYER_ALLOCATION`；前者不能宣传已复现 pyramid 分层分配。W8、beta20 同保留率的阈值为 N1280，可作为事先声明的论文窗口配置单独运行；不能静默调 beta、关 fallback 或再造分配以获得强结果。

每个 KV head 可保留不同原位置，输出 compact `[B,Hkv,Nkept,D]` K/V 和 `[B,Hkv,Nkept]` int64 positions；不保留原 full storage view，也不以全长零 KV 冒充驱逐。reader 与仍精确保存的 text/suffix KV 合并，对原位置做因果可见性，已经应用的 RoPE 不重新编号。

### 两个信息条件

- `query_unknown`：只允许 question-free 视觉前缀及其最后 W 个原生 Q，调用者必须显式传入 `prefix_is_question_free=True`。没有未来问题内容，这属于原选择器在视觉前缀观测条件下的作用域适配。它不等同于原论文最后 instruction window 已包含任务问题的场景。
- `query_known`：只有当前问题已参与观测才能使用，必须 `question_in_observation=True`。这需要当前题完整原生 prefill，再编码压缩视觉 cache，再按约定重放问题 suffix，额外选择费用计入总成本。**单列 query-known 表，不与未来问题未知的编码方法混称严格同信息协议。**

无真实 Q/attention 返回 `METHOD_PORT_UNVERIFIED`；未知条件收到当前问题或未验证 question-free 返回 `INPUT_MISMATCH`。仅传一个布尔值不是模型适配核验，端到端 manifest 仍需输入 token IDs/hash 与边界证据。

## 当前门禁与重现

`results/p0_turboquant_eviction_cpu_20261119_01.json` 与后继 `..._02.json`：CPU 合成算术 6 项 PASS；02 加入源码字节校验与显式数值变体。02 数值源文件与验收结果冻结于 `audit/source_snapshots/te_cpu_checked_20261119_02/`，后续源变化通过新 run 验证，不覆盖该快照。两档 T 的 K/V 与所选上游编解码输出逐元素完全一致；QJL dot 恒等式、GQA+causal reader 对独立 SDPA 验证；E 对 AST 隔离执行的 NVIDIA 原方法验证 score 完全一致、48 个层预算配置完全一致、视觉限制 topk、compact owner 和 sparse reader 均通过，query-unknown 拒绝条件通过。

CPU03/04/05 保留为独立后继 run；04 增加 D128 范围门禁与 reader backend 临时字节说明，05 只增加 E 原始预算 fallback 的标注。03 起 gate 的 CUDA 分支另验上游 packed score 与 visual-only fused kernel，未运行该分支时不得将 CPU 标签扩展为 CUDA 通过。

```
cd code
python -m methods.validate_turboquant_eviction --device cpu --output ../results/<new_unique_name>.json
python -m methods.validate_turboquant_eviction --device cuda:0 --output ../results/<new_unique_cuda_name>.json
```

输出使用 exclusive-create，不覆盖旧记录。CUDA gate 要求实际 capability=(8,6)，记录 GPU 名称、Torch/CUDA、峰值和源文件 SHA。**CPU PASS 不等于 CUDA PASS，更不等于模型质量、三模型适配、P1/P2/P3 主表完成。**

### CUDA v2 实测结果

`results/p0/turboquant_eviction_math_cuda_v2.json`：RTX3090、SM86、Torch2.4.1+cu121，8/8 项 PASS。job manifest 和 launch receipt 固定了 GPU1 可见设备、只读 P72 runtime、矩阵数据盘 cache；该短租约已结束，进程已退出。实际 `-m` 模块路径与 PYTHONPATH 均位于不可变 source snapshot，验收中有断言。

- 两档 K/V 与锁定源编解码最大误差 0。
- 原生 Triton packed-K logits 对独立 dequant reference 最大误差 `1.90735e-6`。
- 原生 packed-K 与 exact text 合并后的 reader 最大误差 `2.44141e-4`；原始 visual-only fused reader 最大误差同为 `2.44141e-4`。
- E 的 NVIDIA 原始 score 与 48 个预算 case 完全一致；compact sparse reader 最大误差 `1.22070e-4`。
- 此合成进程峰值 allocated `17,699,840` bytes，包含测试输入和参考副本；**不能当模型峰值、生产 serving 内存或速度成绩**。

结果、日志与 snapshot inventory 已取回，并与远端重读 SHA256 核对一致；收据为 `audit/te_cuda_v2_download_receipt.json`。验收状态及结果 SHA 单独记录在 `audit/support_status/turboquant_eviction_cuda_p0_v2.json`；原始 source lock 从已验收 snapshot 恢复为逐字节相同的冻结版本，SHA256=`4074c7784634a04d66e1b5749aa6826008dcd9cdc8845bf7fbf7294e69861983`，不将可变支持状态回写源锁。当前仍未测量任何模型问答质量；模型 E2E 需要 native prefix Q 捕获、位置/边界核验、同输入 F_bridge 与 suffix replay 门禁。
