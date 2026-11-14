# 状态匹配校准：本轮补查与可用组合

检索工作流为 multi-source-search。当前没有可调用的 academic-search MCP；按 skill 的独立脚本回退，PubMed/CrossRef/arXiv 接口预检3/3可达，但 OpenAlex 发现接口返回429。随后用网页搜索发现线索、直接打开 arXiv 原文和 PMLR 正式页面核对。以下按 arXiv ID/正式标题去重，仅列与下一步设计直接有关的四项，不按是否正式发表排斥可借鉴方法。

| 工作 | 核对的来源与状态 | 可借鉴部件 | 本项目要补的实验条件 |
|---|---|---|---|
| HeadQ: Model-Visible Distortion and Score-Space Correction for KV-Cache Quantization | [arXiv:2605.03562v1](https://arxiv.org/html/2605.03562v1)，预印本 | query-PCA残差侧码、对score误差进行校正、同预算及错误方向对照 | 用真实压缩前缀形成的query而非只用FP16 query；所有basis/侧码字节计入，最后用VLM自由答案评估 |
| KV Cache Quantization for Diffusion Language Models | [arXiv:2610.39853v1](https://arxiv.org/html/2610.39853v1)，预印本 | 按部署中的缓存重用过程构造校准轨迹 | 我们是因果AR视觉前缀，不是双向去噪中的stale-cache误差；应测“问题状态形成条件×答案轨迹”而非照搬reveal schedule |
| High-accuracy Low-Bit KV-Cache Quantization via Local Distribution Restoration | [arXiv:2607.16248v1](https://arxiv.org/html/2607.16248v1)，预印本 | 用FP16/低比特成对解码轨迹监督局部候选分布恢复 | 当前先把行为目标用于编码元数据校准，不引入新的在线触发器；多token比单token的增益需独立消融 |
| PatternKV: Flattening KV Representation Expands Quantization Headroom | [ICML2026/PMLR正式页](https://proceedings.mlr.press/v306/zhang26ci.html) | 先对齐少量共享pattern，再量化残差 | 后备方向：按真实存储预算比较pattern和逐轴码，不能把pattern表、assignment、残差元数据当免费 |

HeadQ的主要下游证据是K-only、V保持dense的PPL，辅助full-KV并非packed延迟结论。它支持我们关注实际query看到的误差，而不是把原始MSE当任务损伤；不能直接用其数字替代视觉KV实验。[原文§3–5](https://arxiv.org/html/2605.03562v1)

另一个直接影响实现的细节：对**所有**key加同一个向量，所有logit同移，softmax不变；只对视觉key同移、文本key不移，则改变视觉段相对文本段的质量。因此我们只压视觉时，中心化后的`q·μ_visual`必须加回。这个区别也说明，学center带来的收益可能来自视觉/文本质量分配，而非只来自视觉内部方向恢复。[HeadQ几何讨论](https://arxiv.org/html/2605.03562v1)

dLLM-KVQ的轨迹校准针对双向去噪中旧缓存与全重算状态不一致；我们的图像因果前缀可以相同，而图像之后的问题状态因读取不同精度而不同。两者共同启发是“按实际状态生成过程采样校准”，具体误差来源和协议要分开验证。[dLLM-KVQ原文§3–4](https://arxiv.org/html/2610.39853v1)

DGAP从成对FP16/压缩解码轨迹学习对局部词表候选的修正，说明答案分布可作为训练信号。我们先测试更小的编码侧可调量，把修复折入已有元数据；若多token校准仍失败，再判断是否需要额外解码器。[DGAP原文§4](https://arxiv.org/html/2607.16248v1)

PatternKV提供了另一个可实际组合的部件：去除可共享结构后编码残差。我们已经尝试的单均值/方向码不等于该方法；下一步若使用多个pattern，应与同成本的VQ或混合位宽比较。[正式摘要](https://proceedings.mlr.press/v306/zhang26ci.html)

## 当前执行决定

继续已经锁定的32新源文档、83个独立问题输入上的6臂校准。首token与最多8个teacher自生token，分别在压缩问题状态与原生问题历史上拟合；额外center臂单独增加参数，不将其收益混入“只改校准条件”。先通过梯度检查，再做真实packed自由生成。

如果head级缩放/偏移不足以修复方向，后备是小组通道的仿射参数或query子空间侧码。先用新开发64测收益和成本，方法冻结后再进入256新源文档。现有Doc64和Text128已观察，仅作既有复制证据，不再给新候选提供“未见确认”标签。
