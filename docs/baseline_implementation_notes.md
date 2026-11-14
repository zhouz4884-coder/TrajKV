# 强基线与事件读取：来源、可运行接口与新颖性边界

核验日期：2026-11-12（Asia/Hong_Kong）。本次交付是代码准备和 CPU 数值核对；尚未测模型质量、GPU 兼容性、GPU 时延或物理 DRAM 流量。来源台账：`audit/baseline_sources.json`。

## 1. 当前决定

- **优先接入 KVarN 的低层量化机制**：有锁定的官方源码，纯 PyTorch 数值函数可复用。表中必须写 `KVarN low-level reference [mechanism-port]`。
- **TriAxialKV 做显式标注的机制对照**：本次未定位到可验证的官方代码仓库。已提供按 tag 的 2/4-bit 分配器；它不含 tagger、校准或 SGLang 内核，不能称为完整复现。
- **ReadKV 是固定存储、按查询改变读取精度的直接先行工作**。该概念不得作为 TrajKV 的原创贡献。若继续研究长期状态代价，应在同一表示上加入 query-local reader 对照，再检验历史写入后果是否提供额外收益。
- **Don't Waste Bits 是存储决策基线，不是固定存储的读取精化**。官方仓库仅含 README、图和 LICENSE，未发布控制器代码/权重；本轮不能声称复现。

## 2. KVarN：现在能调用什么

官方：[论文](https://arxiv.org/html/2606.03458v1)、[仓库](https://github.com/huawei-csl/KVarN)。本地 `third_party/KVarN`，commit `7586257f1c632e63187bfacbbe21ccb51540f7b3`，Apache-2.0，clone 后工作树干净。

| 官方文件 | 可独立复用的入口 | 边界 |
|---|---|---|
| `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py` | `variance_normalize_batched` | 纯 torch；交替双轴方差归一化，默认 16 次，保留最低 imbalance 状态 |
| `vllm/v1/attention/ops/kvarn_store.py` | `kvarn_store_tile_k_batch_from_sinkhorn`、`kvarn_store_tile_v_batch_from_sinkhorn` | 2/4-bit packed records；元数据 fp16。单 tile 旧入口仍限制 4-bit，不用于 K2/V2 |
| `vllm/v1/attention/ops/kvarn_decode.py` | `kvarn_dequant_tile_k`、`kvarn_dequant_tile_v`、`_unpack_lowbit` | 恢复旋转坐标中的 tile，外部再逆 Hadamard |

K 输入旋转后朝向为 `[head_dim, group]`，V 为 `[group, head_dim]`。K 沿 token 轴做每 channel RTN，V 沿 channel 轴做每 token RTN。scale/zero 吸收到双轴尺度中，不能只算名义 data bits。

提供的 wrapper：`third_party/reference_ports/kvarn_reference.py`。它不导入整个 vLLM，不安装包；通过文件 loader 调用锁定的官方函数，只去掉 store 文件中指向 vLLM 包路径的 Sinkhorn import，并绑定同一个官方函数。运行时校验三份源码 SHA256，防止源码漂移。

```python
import sys
sys.path.insert(0, r"/path/to/TrajKV\third_party\reference_ports")
from kvarn_reference import encode_kvarn, decode_kvarn, quant_dequant_kvarn

# x: [..., tokens, head_dim], typically BF16/FP16 KV from the shared harness
record = encode_kvarn(x, axis="k", bits=2, group=128, iterations=16)
x_hat = decode_kvarn(record)
actual_storage = record.storage_tensors()  # 交给 code/bytes.py 唯一口径计数
v_hat = quant_dequant_kvarn(v, axis="v", bits=2, group=128)
```

尾部不足 group 的 token 原精度保留，不用补零参与方差归一化。默认 `rotation_dtype="float32"`，避免额外 FP16 GEMM 舍入污染 codec 对照；可显式设 `float16`。输入 BF16 时最终回到 BF16。默认旋转、tail 策略、同执行器读写方式必须进入实验 manifest；它们与官方 serving 路径存在区别。

本机 `torch==2.11.0+cpu` 已完成 8 个 axis×bits×rotation 配置，以及 FP16/BF16、空/短尾、常数/零、非连续输入检查。packing roundtrip 完全一致；与官方单 tile dequant 加逆旋转相比，最大绝对差约 `1.43e-6`（FP32 批量 GEMM 减法顺序），FP16 配置在本次 fixture 为 0。结果 `third_party/reference_ports/kvarn_cpu_smoke.json`。

复核命令：

```powershell
python /path/to/TrajKV\third_party\reference_ports\check_kvarn_cpu.py
```

**3090 结论**：该机制移植只用 torch 张量运算、2/4-bit integer packing 和矩阵乘，源码未引入 FP8/FP4 等新硬件指令依赖；接到现有 PyTorch CUDA harness 在技术上可行，仍需真实 GPU smoke。CPU PASS 不能写成 3090 已验证。官方全栈当前为 vLLM 0.23.0 fork，其依赖声明 torch 2.11.0；不要替换现有环境。完整 serving 兼容性、吞吐与 FP16/BF16 转换另行核验。

## 3. TriAxialKV：能复现到哪一层

[原文](https://arxiv.org/html/2605.17170v1) §3.1–3.2：按 recency×modality×role 给 token 打 tag，真实 capture 上逐 tag 量化、测 attention-output distortion；heads 取 max，requests 取 mean，layers 求和。预算约束下分配 INT2/INT4，生成 token 写 INT4 池。它已经含实测敏感性，不能简写成“仅结构标签”。

本次在论文、作者主页和精确标题/GitHub 搜索中未定位到可验证官方仓库，状态 `official_code_not_located_in_bounded_search`，不是“作者一定没开源”。

提供 `third_party/reference_ports/triaxial_allocator.py`：输入每 tag 的实测 `D(2),D(4)` 与由 canonical bytes 模块得到的部署字节；小 active tag 集穷举求解，较大集合用边际收益/成本贪心并标近似。不能用它替代全套 tagger、校准数据或 SGLang serving。已通过一个不等成本的精确预算 fixture；未测下游质量。

## 4. ReadKV：直接覆盖范围

[Read What Matters](https://arxiv.org/html/2610.11245v1) §2–3 已将 retained/read budgets 分开：渐进标量码固定保存在缓存，每个 query 决定 K channel 前缀深度，重建 attention 再决定 V token 前缀深度。未读部分为后续查询保留；有显式读预算和收益分配规则。

因此“存得细、每次少读”“同一存储、按事件精化”均不是足够的新点。其质量实验与单层 GPU 内核也应区别：后者是已有缓存上的单层计时，尚未建立端到端 serving 优势。未在本次有限查阅找到官方代码链接，不能杜撰仓库或把均匀残差模拟称作 ReadKV 官方复现。

对 TrajKV 有价值的可检验差别是：reader 除当前局部输出误差外，还估计该读取会写入何种持久文字状态；在相同存储和累计读取预算下，这个附加信号能否赢 query-local reader。需要因果 crossover 和自由生成的应用收益共同支持。

ReadKV 自身还引用 [Xie et al. 2025](https://arxiv.org/abs/2503.18869) 的分层精度与 bit-plane 选择读取。此处仅登记为进一步核验项，未准备其硬件实现。

## 5. Don't Waste Bits：写入决策而非读取精化

[CVPR Workshops 2026 正式页](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/html/Boroujeni_Dont_Waste_Bits_Adaptive_KV-Cache_Quantization_for_Lightweight_On-Device_LLMs_CVPRW_2026_paper.html)、[原文](https://arxiv.org/html/2604.04722v1) §3.5–3.8。它用 entropy、rarity、attention variance、confidence 训练 3-layer MLP，输出 2/4/8/16 位，在新 K/V append 前量化；后续复用该存储。它覆盖 uncertainty-guided precision allocation，但不是同一高精度存储的可变深度读取。

官方仓库已 clone 到 `third_party/Dont-Waste-Bits`，commit `622ae139437e29ed090b53c265a7781d3ff790b9`。仅三份 tracked files：LICENSE、Overview.png、README.md，README 仍写未来发布代码。因此无官方控制器可运行。独立开发 entropy 规则只能标 `entropy heuristic control`，不能写作作者方法复现。

## 6. 交付口径

以上代码检查只验证 CPU 数值接口和资源约束求解。所有 method-port 与官方复现分开；名义 payload、实际常驻张量、临时 workspace、逻辑读取和物理 DRAM 流量分别报告。新颖性核验的结果是研究约束，不是算法效果证据。
