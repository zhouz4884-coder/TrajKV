# OnlyTerp 的可执行选择：独立 TurboQuant-MSE 视觉移植

本轮选择是 **`methods.turboquant_onlyterp_mse_adapter.OnlyTerpMSEAdapter`**：直接使用固定上游的 `turboquant_mse_encode/decode`，明确 **QJL=0**，无损打包原始离散符号。它不是 TurboQuant-prod，也不冒称论文“2.5-bit”主表复现。生产路径已实现、六组 CPU 源一致性验收通过；本子任务没有运行 GPU 或模型问答，真实质量和实际 GPU owner 由母任务的 CUDA→2QA→8/64QA 实验决定。

## 论文与源码身份

- 主论文：[arXiv:2504.19874v1](https://arxiv.org/html/2504.19874v1)，对照 Algorithm 1/2 和 §4.3。
- 第三方源码：[OnlyTerp/turboquant 固定 commit](https://github.com/OnlyTerp/turboquant/tree/1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306)，MIT；不是 Google 官方实现。
- 唯一上游运行文件：[src/cache.py](https://github.com/OnlyTerp/turboquant/blob/1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306/src/cache.py)。rotation、Lloyd-Max、variance split、MSE 编解码都定义在该文件内，无其它量化工具模块依赖。
- 生产源码锁：`configs/methods/onlyterp_mse_source_lock_v2.json`。`cache.py` 是 runtime 文件；LICENSE 是保留原文的 provenance 文件。旧 launcher 不复制无扩展名文件，因此不把未执行的 LICENSE 伪列为 worker 的运行依赖。

v1 论文 §4.3 明确使用独立的 outlier/regular 量化实例，但所写 `32×3+96×2` 除以 128 的结果实际为 **2.25**。不能照抄 2.5，也不能自行把差值解释成作者已确认的元数据开销。§4.3 没给出“3.5”实例的完整分组配置。Algorithm 2 的总位数还包含 QJL，不能把源码 `b_mse` 当成 prod 总位宽。论文的流式文本 KV 范围也不同于本库的视觉压缩、文本 FP16 协议。

## 源码实际做了什么

`src/cache.py` 的主路径可核查为：

1. 在原始空间按 K 的跨 token 方差选 32 个通道；其余 96 个是 regular。源码高层 Cache 将 K 得到的同一分组/码本用于 V，本移植保持这一点，每个新视觉前缀重新计算分组，不沿用前一幅图的 `_mixed_configs`。
2. regular/outlier 分别使用独立 seed 的旋转、标量码本和每向量范数。默认 `rotation_mode='hadamard'` 把 96 维 regular 补到 128，outlier 为 32；所以 mixed1/2 的 MSE payload 是 `(128×1+32×2)/128=1.5`，不是 1.25。mixed2/3 对应 2.75，不是 2.25。
3. 源码向量运算是 FP32，范数明确存为 FP16，码本 centroids/boundaries 是 FP32。本实现保持这些源数值精度；没有另做低精度 norm/scale port。
4. 源码默认离线 Lloyd-Max，`use_online_codebook=False`。在线经验码本是第三方可选功能，涉及 CPU histogram 和逐 prefill 拟合；审计到的 v1 论文 §4.1 是 Empirical Validation，未找到该 repo 所声称的 prefill-clustering 描述。本轮不把在线功能当论文默认。
5. 主 Cache 的 mixed MSE 之外再加一个全 128 维 Gaussian QJL；它不等于“两个独立完整 prod 实例”。V 的 QJL 数据会被存储，但高层 `compute_attention()` 的 V 只读 MSE 重构。本轮直接选择公开 Algorithm-1/MSE alias，不生成或存储 QJL。

因此，“原样运行 OnlyTerp 就得到低比特强基线”不成立。其主路径 indices 是逐元素 uint8，QJL signs 是逐元素 int64。CPU 单层/单头、N64 的字面高层 Cache 实际 Tensor storage 为约 107.63 bit/KV 标量，含常量，不能用名义宽度替代。

## 不采用的现成 GPU/缓存路径

`src/kernels.py` 自己说明使用 Rademacher QJL，不能和主路径 Gaussian encode/decode 混用；它仅硬编码 uniform 2-bit 标量码本，没有 96/32 mixed 接口。其编码 wrapper 确实返回 byte-packed uint8，读取时又转 int32：后者是临时费用，不能错误算成其持久 payload。

还有两个可重复的独立问题：

- 缓存 manager 的 `_compressed_scores()` 先调用已经含 QJL 的 `turboquant_decode_single`，再加一次 QJL。CPU 源表达式复现中，对单次 QJL 的最大误差 2.2861，对双 QJL 公式仅 1.67e-6。其 mixed/raw 分支的 V 还使用 full-prod decode，而 compressed-only 主 Cache 使用 MSE V。这不是可直接采用的“128窗口保护”实现。
- experimental Triton butterfly 的 upper 更新为 `b-a`，解码又复用同样公式。CPU 按字面标量语义复现，未量化 round-trip 相对误差约 1.414；正确主路径 FWHT 约 9.86e-8。这里没有运行该 CUDA kernel，结论是源表达式与自逆声明不一致，不是给它虚构 GPU 失败分数。

这两点只描述 OnlyTerp 备选路径，不是对原 0xSero 负结果的事后归因。新生产模块完全不 import experimental kernels、LUT attention、vLLM manager 或 plugin。

## 实际实现和资源口径

新 owner 将 K/V 码字紧凑存为 uint8 bitstream；范数保持源 FP16；channel map 的 int64 值无损存为 uint8，读取恢复 int64；Hadamard 的 FP32 ±1 符号无损存为 1 bit，读取恢复相同 FP32 值。源码本边界没有省略，centroids/boundaries 的 CPU 源对象与真实 GPU 对象都通过 `named_tensors` 计费；相同离线常量只在确实共享同一 storage 时去重。

数据无关 Lloyd-Max 数组被缓存，但图像相关分组从不缓存。CPU 生成的旋转符号再精确上传，是明确的 source-device 选择；源 API 默认也是 CPU。调用相同上游函数验证了每个实际存储字段，避免用“同分布”代替同实例数值校验。

reader 只有显式的 `torch_materialize_fp32_reference`：从真实 packed owner 解码临时 FP32 K/V，和精确文本 KV 的 FP32 算术一起计算同一次 softmax，再转回原 query dtype。GQA 与原位置因果可见性均保留。它不持久保留解码 KV；临时完整视觉 K/V、bit 展开、exact-text FP32 和同步开销都属于该 prototype 的实际峰值/时延，不能称论文优化 kernel 的速度。

CPU storage-only 最小化投影显示，uniform MSE1 与 mixed H1/2 值得分别挑战 1.3/1.8 档；但这些投影不是 GPU owner 实测，也不用于主表。当前生产 owner保留了更多源常量，最终只报真实 `cache_storage` 数字。dense 96/32 的矩阵常量开销也不能隐去；例如 CPU 投影中 dense mix1/2 在 N1280 约 2.51 bits，在 N256 约 6.52，远高于仅看 1.25 payload 的印象。

## 验收和下一步

- CPU：`code/gates/onlyterp_mse_gate.py`，六组覆盖 N1/17/65/129、KV heads1/2、mixed1/2、mixed2/3、uniform1、零值。源 K/V 重构逐值完全一致；categorical tail widths 1/2/3/4/8 无损；joint-GQA-causal reader 对 FP64 参考通过；无 full KV alias、无持久 decoded 数据、实际共享常量去重、原位置 backing-capacity 和拒绝非法 backend 均通过。
- 当前 CPU 证据：`results/p0/onlyterp_mse_adapter_cpu_v2.json`，原始镜像在 E；v1 源快照和 v1→v2 执行锁修复均保留。
- SM86：生产路径只用 Torch FP32/FP16、searchsorted、位运算和 FWHT，不要求 FP8/SM100。**这不代替实际 CUDA 验收。** CUDA 同一 gate 的配置已备好，母任务安排执行。
- 真实 2QA：`configs/jobs/bm_M1_OT_Hmix12_textvqa_probe2_v1.json`；沿用已注册的 `evaluate` driver，相同原生前缀、完整 question suffix、独立首 token 和完整自由生成，无参考分支首 token 复用。
- 8QA：Text4+Info4；64QA：Text32+Info32。输入来自固定 registry 的前几个 source，不读取 GT 选题。全部草案见 `configs/matrix/OnlyTerp_MSE_pilots_v1.json`。Hmix23 是更高字节的强质量参照，Huniform1 是更低字节的独立源配置；不拿它们互换方法标签。

只有真实模型结果才能决定该候选能否进入正式外部库。低质量时保留真实失败；高质量时再独立优化 reader 并复核 owner 与速度，不修改这次源算术来追分。
