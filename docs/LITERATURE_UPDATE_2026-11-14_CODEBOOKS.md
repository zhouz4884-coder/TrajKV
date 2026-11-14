# 方向码本与行为校准：补查可借鉴的部件

本次补查的用途是改进下一步实验配方。已有旋转、VQ、可学习量化和端到端缓存codec都可以借鉴；新增贡献应由我们的条件、组合及消融确定，引用应覆盖直接来源。

| 工作与核对来源 | 本次吸取的点 | 对我们实验的影响 |
|---|---|---|
| [VecInfer，ACL2026正式页](https://aclanthology.org/2026.acl-long.1454/) | 平滑与Hadamard可改善VQ面对的离群分布，部署需核对解码开销 | 旋转与VQ本身有明确前人基础；我们的新问题是如何在视觉前缀使用过程中校准重建，以及共享常量的实际价格 |
| [Beyond，EMNLP2026正式页](https://aclanthology.org/2026.emnlp-main.526/) | 冻结backbone，通过模型损失学习非均匀重建电平和阈值，训练与推理均用hard quantizer | 支持按行为损失学习量化参数。我们当前只调decoder书及可选V元数据，编码分配固定，必须用同格式未学习对照分开收益 |
| [DCKV，arXiv:2610.36950v1](https://arxiv.org/html/2610.36950v1) | 以冻结模型输出变化和码长联合训练缓存codec | 提醒把共享codec权重、存储态、工作态和峰值分开核算。它的hybrid模型stored-prompt设置，与我们每步直接读packed视觉KV不同，不能直接抄压缩比 |
| [CommVQ，ICML2025正式页](https://proceedings.mlr.press/v267/li25du.html) | 用RoPE交换结构组织和学习码本，兼顾读出成本 | 编码几何、码本与位置机制应共同评估；不能将一个普通码本表叫作首次VQ缓存压缩 |
| [More Tokens, Lower Precision，Findings2025正式页](https://aclanthology.org/2025.findings-emnlp.429/) | token数量与精度存在联合资源取舍 | 我们的分辨率对照要报告总字节与匹配输入，不能只拿bit/scalar排序。图像encoder分辨率变化还会改变视觉内容，与纯KV剪枝有区别 |

Beyond的正式摘要也报告了量化优于Full的结果。这支持将Full视为质量参考，而不是数学上的准确率上界；我们仍按事先固定的强参考和0.5pp研发容差评价。[正式摘要](https://aclanthology.org/2026.emnlp-main.526/)

另有一处文献版本信息更新：此前读到的`KV Cache Quantization for Diffusion Language Models` arXiv版本使用TRACE/dLLM-KVQ命名。现在已核对[EMNLP2026 Findings正式页](https://aclanthology.org/2026.findings-emnlp.799/)，正式名称为TraceKV/KProtect。旧阅读记录保留其当时版本，新写作使用正式条目；它研究双向去噪缓存重用，我们研究因果视觉前缀后的问题状态形成，条件不同。

## 当前组合的可检验部分

当前原型将固定Gaussian encoder、FWHT、每层/KV-head共享decoder书以及冻结模型行为目标组合起来。数据仍只用cal32的83个问题输入；比较cold/warm状态条件与是否联合V元数据。114688个decoder参数和最多224个encoder-only参数均有明确价格，原始预训练权重不更新。

GPU底层9个reader和3个旋转检查已经通过，完整模型校准和新图自由生成尚未运行。预期约1.73bit是计费推算，不是质量结论，也不是最优码率。需要继续比较同成本未学习书、旧matrix-VQ、Base2、Native4和Full。

若这一支不足，另一个可借鉴的部件是[HeadQ的query子空间残差score校正](https://arxiv.org/html/2605.03562v1)。其吸引力是只保存实际query方向上可见的残差，不要求K误差矩阵低秩。我们尚未实现它；后续若采用，需计入basis、系数、量化元数据，并用压缩状态采样与原生状态采样作同价对照。
