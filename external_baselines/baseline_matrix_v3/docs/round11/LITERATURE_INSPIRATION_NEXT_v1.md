# 下一轮机制启发：五项一手来源，只读、未运行

核查日期：2026-11-23。本文不读取原 R128 或追加 384 的结果，不改变当前冻结算法、样本、队列或选择。它是后续设计材料，不是实验授权。先读 Root 的 `NEXT_IDEAS_NOT_RUN_v3.md` 与 `NEXT_UNTESTED_KERNEL_COMPOSITION_v1.md`，再在约十五分钟内核查一手全文。检索使用 nature-academic-search 工作流；本会话未挂载学术数据库工具，网页工具出现传输错误，故以本地 curl 只读获取 arXiv 与 PMLR 原站。未将摘要检索结果当作全文公式证据。

## 已核实的来源与身份

| 来源 | 作者 | 日期、状态与可复核入口 |
|---|---|---|
| **Hydragen: High-Throughput LLM Inference with Shared Prefixes** | Jordan Juravsky; Bradley Brown; Ryan Ehrlich; Daniel Y. Fu; Christopher Ré; Azalia Mirhoseini | arXiv:2402.05099，首版 2024-02-07，本次读 v2（2024-05-13）。本次未核实正式会议录，不补写录用状态。[原站](https://arxiv.org/abs/2402.05099) · [全文 §3.1、式4–5、附录A](https://arxiv.org/html/2402.05099v2) · [作者代码](https://github.com/jordan-benjamin/hydragen) |
| **Output-Aware Rotation for INT2 KV-Cache Quantization（OptR）** | Vincent-Daniel Yun; Woosang Lim; Minsoo Cheong; Sunwoo Lee; Murali Annavaram; Sai Praneeth Karimireddy; Sungjoo Yoo | arXiv:2608.02691，首版 2026-08-03，本次读 v4（2026-09-23）；本次所见为预印本，未核实正式录用。[原站](https://arxiv.org/abs/2608.02691) · [全文 §3.2、§4.1–4.5](https://arxiv.org/html/2608.02691v4) |
| **CALFA-KV: Joint Calibration and Fisher-Guided Anchors for 1-Bit KV Cache Compression** | Xiancai Tian; Leichao Zhang; Shenhui Zhang; Yi Shan; Hailong Li; Feng Lin; Xiaotian Yu; Aijun Li | arXiv:2611.10511，v1，2026-11-07；本次所见为预印本，未核实正式录用。[原站](https://arxiv.org/abs/2611.10511) · [全文 §3.1–3.3](https://arxiv.org/html/2611.10511v1) · [作者代码](https://github.com/XiancaiTian/CALFA-KV) |
| **GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache** | Soosung Kim; Minjae Park; Eui-Young Chung; Jaeyong Chung | arXiv:2607.01065v1，2026-07-01；正式 **ICML 2026，PMLR 306:58743–58756**，PMLR 元数据发布日期 2026-09-29。[会议录](https://proceedings.mlr.press/v306/kim26bf.html) · [全文 §5.2–5.3](https://arxiv.org/html/2607.01065v1) |
| **CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs** | Insu Han; Zeliang Zhang; Zhiyuan Wang; Yifan Zhu; Susan Liang; Jiani Liu; Haiting Lin; Mingjie Zhao; Chenliang Xu; Kun Wan; Wentian Zhao | arXiv:2502.14882，首版 2025-02-15，本次读 v2（2025-03-24）；本次未核实正式录用。[原站](https://arxiv.org/abs/2502.14882) · [全文 §4.1–4.3](https://arxiv.org/html/2502.14882v2) · [作者代码](https://github.com/insuhan/calibquant) |

以上名称均已找到相应一手正文；OptR 是方法简称，不是论文题名。这里只归纳机制，不把各论文的异模型、异预算成绩当成本项目证据。

## 1. 功能目标应保留分区质量与分子

**论文明确支持。** Hydragen 在同一实际 query 上分别计算两个 attention 分区的归一化输出与 log-sum-exp，再按分区的 softmax 分母重新加权，精确合并。它是精确 attention 的实现方法，不是未知 query 下的有损 KV 压缩定理。CalibQuant §4.2 对量化后的 pre-softmax logits 使用仿射校准，参数以网格选择后跨输入、层与 head 固定；这支持检查 logit 分布失真，但没有宣称分别精确保留视觉分区质量与分子。[Hydragen](https://arxiv.org/html/2402.05099v2), [CalibQuant](https://arxiv.org/html/2502.14882v2)

**我们的推导。** 对视觉部分定义

\[
M(q)=\sum_{i\in I}e^{q^Tk_i/\sqrt d},\qquad
N(q)=\sum_{i\in I}e^{q^Tk_i/\sqrt d}v_i.
\]

若后续文本部分为 \((M_T,N_T)\)，总输出是 \((N+N_T)/(M+M_T)\)。只匹配视觉归一化输出 \(N/M\) 不够：将视觉的 \((M,N)\) 同时乘以 2，视觉单独输出不变，与相同非零文本分区合并通常改变。更精确地，令量化误差为 \((\Delta M,\Delta N)\)，原总输出为 \(y\)，则

\[
\widehat y-y=\frac{\Delta N-y\Delta M}{M+M_T+\Delta M}.
\]

因此后续代理至少应分别记录 log-mass 误差、视觉归一化输出误差和合并误差；只改善其中之一不能推导未知未来文本下的整体改善。这里没有找到在本项目同一 B0 下对所有未知 query 保证同时保留二者的现成方案。

OptR §4.1 的减去共同 key 中心只在**参与整个 softmax 的所有 keys 同移**时保持分布，正文也对 BF16 保留窗口使用同样中心。本项目如果仅改变视觉 key 的共同均值而文本不变，视觉内部比例可不变，视觉相对文本的分区质量却会改变。这正可成为待检验的质量校正自由度，不能套用“attention 完全等价”表述。[OptR §4.1](https://arxiv.org/html/2608.02691v4)

## 2. 敏感度可以借鉴，但必须保持坐标与目标清楚

**论文明确支持。** GSRQ §5.2 将 codeword 写成非负 gain 与单位 shape 的乘积，§5.3 用经过对数平滑的**每个样本标量梯度范数权重**训练残差聚类。CALFA-KV 将联合输出校准与后续 anchor 选择分开：前者用冻结模型的投影后 attention 输出；后者用参考 next-token CE 的平方梯度估计通道 Fisher 型敏感度，并在与误差一致的坐标内评分。它还明确计入额外高精度 anchors；名义 1-bit index rate 不是全部字节。[GSRQ](https://arxiv.org/html/2607.01065v1), [CALFA-KV](https://arxiv.org/html/2611.10511v1)

**我们的推导，不是上述论文已证明的本项目算法。** 若目标只是预 softmax logit 的平均平方误差，key 误差 \(e\) 的二次型为 \(e^T\mathbb E[qq^T]e/d\)。这里是非中心二阶矩，不是协方差。它也不是整个 attention 或任务损失的精确 Hessian：softmax 的曲率还耦合不同缓存 token，后续 value 与 W_O 进一步改变目标。

若原 reader 的行向量关系是 \(x_{codec}=x_{native}U\)，且 U 为当前正交 WHT，则 codec 误差的矩阵为 \(C=U^T H_{native}U\)；等价列向量约定可写成 \(U H_{native}U^T\)。当前对称 WHT 两式相同。native 对角 H 的 pullback 通常是 dense，codec 对角元在等幅 WHT 下全等于 trace(H)/128。只拿它的对角来改变 G4 assignment 会退化为普通欧氏 assignment。若采用 pre-RoPE 坐标，还须对每个位置使用对应 RoPE pullback；缓存已经 post-RoPE 时不得重复旋转。动态 RoPE 的 attention_scaling 仍须核对实际函数，不能从 factor=2 猜定。

因此可借鉴“敏感度引导离散选择”，但 GSRQ 的样本权重、CALFA 的 Fisher 对角、query 二阶矩、W_O Gram 是四种不同对象；本次不把它们混称成相同 Hessian，也未找到证明“native 对角权重直接用于现有 WHT G4”正确的来源。

## 3. 固定量化 K 后再处理 V，有直接相关工作与更强的未测推导

**论文明确支持。** OptR 先用 BF16 V 校准 K rotation，再固定选定的量化 K attention 路径优化 V rotation；其 §4.4 的 V 目标是量化 K 权重下的 value-induced、投影后输出误差。CALFA-KV §3.2 则联合拟合中心、径向参数与 K/V codebooks，每次候选还要用部署 encoder 重编码、重算投影输出后再接受。两者都支持通过实际 attention 读出评价，而非只看各自 raw SSE。[OptR §4.3–4.5](https://arxiv.org/html/2608.02691v4), [CALFA-KV §3.2](https://arxiv.org/html/2611.10511v1)

**关键区别。** OptR 的 V-only 目标聚合 \(\widehat A_K(\widehat V-V)\)，不自动消除已存在的 key-induced error。我们的条件补偿提议要直接针对整体残差 \(AV-\widehat A_KV\)，不能宣称该闭式补偿就是 OptR 已实现的方法。

**我们的未测推导。** 对合法开发 query 集的 attention 矩阵 A 与固定量化 K 的 \(\widehat A\)，不受限的 ridge 补偿为

\[
E^*=\widehat A^T(\widehat A\widehat A^T+\lambda I)^{-1}
       (AV-\widehat AV).
\]

它只是在拟合 query 上解 \(\|\widehat A E-(AV-\widehat AV)\|_F^2+\lambda\|E\|_F^2\)。随后必须投影回原 V20 的真实 Half、base/sign/alpha 合法字段，并以原点为候选重算整体目标；不得把 dense E、额外 anchors、query 历史表或新常量偷偷留在 serving owner。投影可能失去未约束解的改善，未知 query 与生成历史变化更无保证。W_O 加权读出可作为另一拟合目标，但不能把局部读出拟合说成任务损失下降。

## 后续最多两组可冻结的试验提议（全部 NOT_RUN）

1. **分区质量—条件 V 因子对照。** 保持原 K/V 覆盖、码本、WHT、reader、payload 与 B0；仅用已有可写 mean/合法字段构造质量校正，再将条件 V 提议投影到原字段。冻结一个 2×2 因子表：质量校正关/开 × 条件 V 关/开。开发数据、query 先验、λ、候选次数与严格接受目标先锁定，再看独立数据；报告 mass、numerator/合并读出、实际字段容量、临时拟合开销与任务结果。不得使用本轮 R128/追加384 的预测或标签做拟合、选参数或选方法。先核实 mean 改动不会被重新居中/重拟 gain 抵消。
2. **固定 G4 的 full-C assignment 对照。** 保持所有书、slot、覆盖与 reader，用固定次数的 16-codeword block coordinate search，显式保留 cross-block 项，比较 full-C 与同计算框架 C=I；另报告对角捷径退化这一解析控制。每次 actual decode 重算目标、ties 保旧字段。新增查询矩阵仅允许临时工作区，返回 owner 前释放，任何新持久常量均须收费。先冻结数学/CPU 对照再申请实际模型任务，不能拿 proxy 最优替代质量结果。

两组都只是未来组合方向。与相关工作有组件重叠不构成否决理由；需要明确借用的组件、原研究没有覆盖的预算/坐标/泛化条件，以及随后真正验证出来的新增效果。当前实验保持原样。

## 核查留痕与限度

身份元数据及网页字节 SHA 存于相邻 `literature_inspiration_next_v1/PRIMARY_SOURCE_METADATA_v1.json`；全文版本、网页 SHA 与定位片段在 `PRIMARY_METHOD_SNIPPETS_v1.json`。五篇 HTML 均实际读取；只作机制阅读，未执行作者代码、复现实验或核对全部附录。原站全文为时间点快照，arXiv 预印本状态不能替代同行评议。本文不把 GSRQ 的 gain-shape、CALFA 的付费 anchors、OptR 的学习 rotation 或 CalibQuant 的 logits 变换当成当前冻结 reader 中免费的可插拔项。
