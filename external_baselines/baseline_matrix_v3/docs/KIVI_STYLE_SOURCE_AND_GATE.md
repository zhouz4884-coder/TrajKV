# KIVI-style 技术锚：官方编码、真实 packed reader 与独立尾段

采用作者 [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) commit `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`，论文为 [KIVI, ICML2024](https://arxiv.org/abs/2402.02750)。编码直接导入固定 `quant/new_pack.py` 的 `triton_quantize_and_pack_along_last_dim`，未改官方源码。MIT 原文附于 `audit/upstream/kivi/LICENSE.txt`。源码锁 `configs/methods/kivi_style_source_v1.json` 分别记录工作副本 raw SHA、LF-normalized SHA 与固定 Git blob，避免将换行当算法变动。

## 合法配置与作用域

仅 batch1、FP16、D128、group128、2/4bits，R=0或128。K 沿视觉 token 轴分组、V 沿每个 token 的 channel 轴分组。文本 KV 全部保留原精度。所有视觉位置原样保留，group 中的视觉顺序不因文本分隔符改变。

- R0：KIVI-style 技术锚。K 的不完整128组留 FP16，V 全量量化。它不是原 rolling KIVI 的 R0 配置；原系统 R0 会触发不合法的窗口取模。
- R128：固定视觉前缀应用上游 prefill 策略，K 留 N%128个尾 token，V 留 min(128,N)个尾 token；两种尾段独立。

| N | K FP16 tail | V FP16 tail R0 | V FP16 tail R128 |
|---:|---:|---:|---:|
| 17 | 17 | 0 | 17 |
| 257 | 1 | 0 | 128 |
| 512 | 0 | 0 | 128 |
| 1008 | 112 | 0 | 128 |
| 1792 | 0 | 0 | 128 |

整个视觉前缀编码后不变，新 suffix/生成 token 进入 exact text cache。这是视觉轴作用域适配，没有声称复现原 Llama 的整个文本/图像 rolling KV 系统，也没有新增1bit KIVI。

## reader 与物理字节

`triton_packed` 使用项目新建的 tiled online-softmax reader，按原始位置做因果可见性；KV tiles 直接读 int32 packed codes、FP16 scale/min 和各自尾段，不持久展开视觉 K/V。其布局参考只读冻结 `TrajKV/code/experiments/fused_progressive.py`，完整 fingerprint 记于 `audit/upstream/kivi/reader_provenance_v1.json`。它是新的视觉适配 reader，没有称为作者原 CUDA bmm 内核。

`torch_dequant_reference` 是显式完整临时解压参考变体，不可静默 fallback。输入视觉 positions 复制为 compact int64，text positions 在 kernel dispatch 前保证连续；不重编号其实际值。源 hash guard、dtype/head/batch/位宽/尾段配置不满足时拒绝运行。

Owner 仅包括 K/V codes、四张 scale/min、K/V FP16 tails 和原 positions。gate 以 H、D、N、bits、Kq、Vq 的独立标量公式逐项推导 payload、metadata、tails 和 positions 字节，再与实际 underlying storage 去重结果和 `owner_report` 比较。FP16 tail 全额计费；N17/R128可能比不含 positions 的原 FP16更大，不能仅以名义2bit宣称压缩率。全局文本在外层 Cache owner 单独计费。

## 验收证据

- CPU：`results/p0/kivi_style_cpu_v1.json`，80项通过。只执行 AST 隔离的作者 PyTorch quant/pack/decode 构造数学 fixture，没有用伪 CPU encoder 替换生产 adapter。
- CUDA：`results/p0/kivi_style_cuda_v1.json`，MAIN runtime、RTX3090 SM86，80/80通过。N17/257/512/1008/1792，Q1/9，Hq:KVH=28:4和32:8，bits2/4，R0/R128全部覆盖。
- 生产 encoder 的 codes/scale/min/tails/positions 与官方独立 PyTorch源码逐元素完全一致。
- 实际 packed reader 调用期间将 full materialize 替换为抛错函数，确认未走完整解压 fallback；输出对独立官方 decode + FP32 SDPA 最大绝对差 `2.44140625e-4`。
- 原视觉位置输入故意为非连续 view、含文本孔；text cache 同时包含 prefix文本孔和新的多 token suffix，检查了未来 suffix key 的因果屏蔽。
- 无持久输入 FP16 storage alias；各类 owner 字节与独立公式完全一致。合成进程峰值218743808 bytes包含 dense控制副本，不是模型或生产内存成绩。

取回结果/日志/inventory后与远端重读 SHA一致，结果SHA为 `40c9e937ea226fe3e869fed53af56da8701b136e823c2ca7120c6e1ae94c8022`。CPU与CUDA数值源SHA相同，成功源码冻结于 `audit/source_snapshots/kivi_style_cuda_v1/`；支持状态另写 `audit/support_status/kivi_style_cuda_v1.json`，不覆盖原 source lock。

首次 GPU 验收前对 root 新 port 的输入/位置契约修正保留 before源码及SHA：`audit/source_snapshots/kivi_port_before_gate_v1/`。没有修改旧 Round1–8 的代码、环境、参数或证据。接下来还需模型 NativeSplit suffix 重放与质量评测；这些合成结果不提供模型质量分数。
