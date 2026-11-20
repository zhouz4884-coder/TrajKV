# Cache / SDPA bridge 合同

`code/cache_bridge.py` 与 `code/attention_bridge.py` 仅实现本矩阵所需的 batch1、FP16 文本、non-sliding、append-only 推理。没有修改 HF、模型 forward、RoPE/mRoPE、Q/K norm 或 DeepStack。语言注意力层路径为 `model.model.language_model.layers[*].self_attn`。

## 调用

```python
with attention_context(model, capture_queries=True, window_size=64) as capture:
    native_prefix = model(**question_free_prefix_inputs, use_cache=True)
queries = capture["queries_by_layer"]  # compact native post-RoPE Q only
# codec owners are constructed externally from this same complete native prefix.
hybrid = HybridVisualCache(native_prefix.past_key_values, visual_positions,
                          visual_owners=owners, reader_backend=backend)
# The constructor consumes and clears native_prefix.past_key_values.layers.
# The caller must also delete any other temporary full-KV lists/views it created.
del native_prefix, queries, capture
with attention_context(model, cache=hybrid):
    suffix = model(**native_suffix_inputs, past_key_values=hybrid, use_cache=True)
```

`visual_positions` 为原序列中真实视觉 token 的排序一维位置；中间可以包含非图像分隔文本。传 `visual_owners=None` 建立 FP16VisualOwner 无压缩桥。原始完整前缀用原生 cache 或可消费 list；tuple 等不可变容器只能显式 `consume_native=False`，调用者随后自行释放。

每层持久状态为 compact text K/V、text_positions、visual owner 和整数 logical_length。视觉 token 不因驱逐而从逻辑位置计数中消失。`update(new_post_rope_K,V,layer_idx)` 只追加新文本，并向 HF 返回 text-only K/V；query_positions 是更新前 logical_length 开始的 append 范围。`get_seq_length` 与 `get_mask_sizes` 按完整逻辑长度工作，dynamic max shape=-1，全部 is_sliding=False。crop 仅允许删除 frozen prefix 之后追加的 suffix。batch/beam 扩张、sliding、offload、reset 原前缀均拒绝。

## reader 和 mask

- FP16VisualOwner 将视觉与文本临时重组为原顺序 FP16 K/V，调用进入 context 前的原始 HF SDPA。没有持久重组副本。
- T/E 调用已验收 owner.attend；E 的每个 head 可以保留不同原位置。T 仅在函数明确支持 backend 参数时传显式 reader backend。
- CalibQuant 调用 `state.read(q, softmax_scale=scaling)`，该结果已经包含 scaling 和视觉 logit calibration，不再除 sqrt(D)。exact-text logits 按作者 patch 的 FP16 QK 后 division 保留 FP16 舍入（分母为 `1/scaling`），再转 FP32 合并。随后按原逻辑位置 mask、FP32 softmax、概率转模型 FP16，分别调用源 weighted_value 和精确文本矩阵乘法。
- 接受 None 或覆盖完整逻辑 key 轴的 `[1,1或Hq,Q,Llogical]` bool/浮点 mask。T/E 的现有 attend 仅支持标准因果 mask，其它额外 padding/bias 显式拒绝。Calib reader 按原 positions 提取完整 mask，因此保留这些 4D mask 的语义。2D、BlockMask 或形状不符不静默忽略。
- dropout 必须0、模块必须eval、语言 attention 必须causal。native mask/tag 始终为 `sdpa`，没有私造 tag 或改 mask registry。

## hook 和状态恢复

context 临时替换共享 `ALL_ATTENTION_FUNCTIONS` 实例的本地 `sdpa` 映射，仅拦截当前模型语言 self-attn 的对象身份；vision 和其它模块委托原函数。只临时设置语言 config 的 `_attn_implementation=sdpa`，异常退出也恢复原配置值和 registry 映射。拒绝嵌套 context。

capture 模式始终委托原 attention，只将传入原 attention 函数的真实 post-qnorm/post-RoPE Q 最后 W 项复制到紧凑 owner；不复制完整原生 K/V。是否 question-free 由 runner 的原 prompt、边界及 hashes 核验。capture dict 必须在 E owner 编码完成后释放，不能将其藏在持久 cache 外不计费。

`hybrid.named_tensors()` 含 text K/V/positions、共享原视觉 positions 与各方法全部 owner tensors；root 统一按实际 storage 跨层去重。构造器拒绝 owner 与 native full-KV storage alias。该检查不替代调用者释放其它 native-prefix 变量或 CUDA 峰值核验。

## 合成验收

CPU v1：`results/p0/cache_attention_bridge_cpu_v1.json`，HF5.4/Torch2.11 CPU，7/7 PASS；不冒称目标 HF5.12 的执行结果。覆盖 native Q 窗口捕获、vision 委托、正常/异常恢复、2层、28 query heads/4 KV heads、不规则视觉位置、多 token suffix 和 decode、原始 full-KV 引用消费、owner 无原 storage alias、逻辑长度/mask/crop、四种方法混合读取。FP16 对原 HF SDPA 误差0。

CUDA v1：`results/p0/cache_attention_bridge_cuda_v1.json`，实际 HF5.12/Torch2.4.1+cu121/RTX3090，7/7 PASS。FP16 原 SDPA 误差0，T 上游 packed score 混合 reader 最大误差2.44141e-4，E与C读出误差0。合成峰值269055488 bytes，包含控制和测试副本，不能作模型峰值。完整源码和验收冻结于 `audit/source_snapshots/cache_attention_bridge_cuda_v1/`。

**v1 Calib text-logits 精度缺口已单独识别。** v1 文本 QK 使用 FP32，且合成 oracle 镜像了这一精度，故 v1 不能证明这一项与作者 patch 完全一致。v2 仅修正新桥的文本 QK 精度（不改 frozen C/T/E 方法源码或源锁），并直接 AST 执行作者 `patch_internvl.py` 中 `attn_weights_quant_t` 赋值作为独立 oracle；CPU v2 7/7 PASS，源 hash 在结果中。归约先按 visual/text 两段而 oracle 用原序 full logits，数学等价，FP32 softmax 归约顺序产生的舍入差异按实际误差记录。

目标 CUDA gate 独立使用 P72 HF5.12/Torch2.4 只读 runtime、矩阵数据盘全 cache 与不可变源码 snapshot；使用 T/C 上游实际 Triton reader。合成结果不能代替模型前缀/问答质量门禁或模型 peak。

正式 main_overlay_v1 上的后继 `results/p0/cache_attention_bridge_cuda_v2_main.json` 已实际 PASS 7/7，HF5.12/Torch2.4.1+cu121/RTX3090；运行时为只读冻结基础的独立项目 overlay。新桥源码与 CPU v2 一致，FP16 原 SDPA 误差0，T 与 C 最大误差2.44141e-4，E误差0。C 使用作者 AST text QK 原表达式及全原序 logits oracle；oracle 与桥的 softmax key 归约顺序不同，实测记录该数值差异。原始文档/结果/日志不覆盖；v1、v2 source snapshot 和取回 SHA 收据分别保留。新支持状态在 `audit/support_status/cache_attention_bridge_v2_main.json`，不回写方法源锁。
