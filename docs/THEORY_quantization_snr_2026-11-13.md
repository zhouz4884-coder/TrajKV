> 历史研究草案，含尚未验证或已被后续实验修正的假设。当前结论见项目时间线与本轮收尾报告。

# 一比特 KV 的解析理论：为什么 K 是悬崖、V 是缓坡，以及为什么 1.25 bit 可能就够

> 2026-11-13。本文只做一件事：把"K1 崩到 0、K2 恢复、V1 只掉 1–3pp"这一组经验事实，从量化论与 attention 的代数结构推导出来，并由此得到三个**零成本或极低成本**的干涉项。
> 所有结论都写成可证伪的形式，配一条当天能跑完的验证。假设与反例单列在 §8，不藏。

---

## 1. 记号与既成事实

单层单 KV-head，head_dim $d=128$，视觉 token 数 $N\in\{256,768,1280\}$，GQA 组内 $G$ 个 query head 共享一份 KV。

量化采用 KIVI 的**非对称均匀**方案（`triton_quantize_and_pack_along_last_dim`），$b$ bit 下每个分组存 $(\text{scale},\text{min})$ 两个 fp16，重建为

$$\hat{x} = \text{code}\cdot\text{scale} + \text{min},\qquad \text{scale}=\frac{\max-\min}{2^b-1}.$$

分组 128 个元素 → 元数据成本 $2\times16/128 = 0.25$ bit/elem（与轴无关）。所以：

| 配置 | payload | 元数据 | 合计 bit/elem | 占 fp16 |
|---|---:|---:|---:|---:|
| K1/V1 | 1.00 | 0.25 | **1.25** | 7.81% |
| K2/V1 | 2.00 | 0.25 | **1.75** | 10.94% |
| K2/V2 | 2.00 | 0.25 | **2.25** | 14.06% |
| K4/V4 | 4.00 | 0.25 | **4.25** | 26.56% |

轴：K 走 **per-channel**（沿序列分组，`bank.py:73` 先 `transpose(2,3)`），V 走 **per-token**（沿通道分组）。这一层已由卷二补充核实，不再讨论。

经验锚点（`results/a2_analysis_value/summary.json`，20 题 9 图，K 或 V 单侧压、另一侧 FP16）：

| 变体 | ANLS | 相对 fp16_both(93.04) |
|---|---:|---:|
| K 1-bit per-channel | **0.00** | −93.04 |
| K 2-bit per-channel | 94.29 | +1.25 |
| V 1-bit per-channel | 91.96 | −1.08 |
| V 1-bit per-token | 90.24 | −2.80 |
| V 2-bit per-token | 93.04 | 0.00 |

**待解释的核心事实只有一条**：同一个 per-channel 分组、同一套 min/max 编码，K 在 1 bit 是灾难、在 2 bit 完全恢复；V 在 1 bit 只掉 1–3pp。下文给出一个统一判据，它同时给出这两个结论，并预测一个新工作点。

---

## 2. 命题一：min/max 的一比特量化，MSE 比最优差约 14 倍

### 2.1 最优两级量化器（精确解）

设分组内标量 $x$ 零均值、方差 $\sigma^2$，一比特只能用两个重建电平 $\pm a$。失真

$$D(a)=\mathbb{E}\big[(x-a\,\text{sign}(x))^2\big]=\mathbb{E}[x^2]-2a\,\mathbb{E}|x|+a^2 .$$

对 $a$ 求极值：

$$\boxed{a^\*=\mathbb{E}|x|\quad\Longrightarrow\quad D_{\min}=\sigma^2-(\mathbb{E}|x|)^2 .}$$

这是两级 Lloyd–Max 的闭式解（阈值取在均值/中位处，电平取条件均值，对称分布下两者一致）。三条推论：

1. **最优电平由平均绝对偏差 $\mathbb{E}|x|$（MAD）决定，与极值无关。**
2. $D_{\min}/\sigma^2 = 1-k^2$，$k=\mathbb{E}|x|/\sigma\in(0,1]$。高斯 $k=\sqrt{2/\pi}=0.7979\Rightarrow D_{\min}=0.3634\sigma^2$。
3. 只要**存 MAD 而不是半极差**，就能在字节完全不变的前提下拿到这个最优——因为 scale/min 两个 fp16 槽位本来就在，只换填进去的数。

### 2.2 min/max 的失真

现在实现取 $a=(\max-\min)/2\approx R$。128 个高斯样本的半极差 $R\approx\frac12(\mathbb{E}\max-\mathbb{E}\min)\approx 3.0\sigma$（$\sqrt{2\ln 128}\approx3.1$）。代入 $D(a)$：

$$D_{\text{min/max}}=\sigma^2+R^2-2R(\mathbb{E}|x|)=\sigma^2+9\sigma^2-2\cdot3\sigma\cdot0.7979\sigma=5.21\,\sigma^2 .$$

$$\frac{D_{\text{min/max}}}{D_{\min}}=\frac{5.21}{0.3634}\approx \mathbf{14.3\times}.$$

同样的 1 bit，MSE 差 14 倍，**纯编码常数造成的**。

### 2.3 二比特也有同样的病，只是轻

均匀 4 电平、范围 $\pm c\sigma$：步长 $\Delta=2c\sigma/3$，$D=\Delta^2/12=c^2\sigma^2/27$。

| $b$ | 现实现（$c=3$） | 最优 $c$ | 最优 $D/\sigma^2$ | 比现实现好 |
|---|---:|---:|---:|---:|
| 1 | 5.21 | 0.80 | 0.363 | 14.3× |
| 2 | 0.333 | 2.2 | 0.117 | 2.8× |
| 3 | 0.061 | 2.7 | 0.0345 | 1.8× |

两条读法：
- $b=1$ 的"最优 $c$"落在 0.80，意味着**一比特的正确做法根本不是"均匀切一段范围"**，而是"以 $\pm\mathbb{E}|x|$ 为两级"。用 min/max 做一比特，等于把重建点放在离数据 3.75 倍标准差的地方。
- $b\ge2$ 时 uniform 已经接近 Lloyd–Max，只需把范围从 $\pm3\sigma$ 收到 $\pm2.2\sigma$（即 percentile clipping），**免费**再省 2.8 倍。

---

## 3. 命题二：一比特 K 为什么是悬崖——一个 SNR 判据

### 3.1 推导

$\text{score}=q\cdot k/\sqrt d$。设量化误差 $\varepsilon=k-k$ 与 $k$ 独立、每元素方差 $\sigma_\varepsilon^2$。则 logit 扰动

$$\delta z = \frac{q\cdot\varepsilon}{\sqrt d},\qquad
\operatorname{Var}(\delta z)=\frac{1}{d}\sum_c q_c^2\sigma_\varepsilon^2=\frac{\|q\|^2\sigma_\varepsilon^2}{d}.$$

若 $q$ 元素近似 i.i.d.、每元素方差 $\sigma_q^2$，则 $\|q\|^2=d\sigma_q^2$，于是 $\operatorname{std}(\delta z)=\sigma_q\sigma_\varepsilon$。

同时真实 logit 的跨位置标准差（同一 $q$、不同 $k_i$）：

$$\sigma_z^2=\operatorname{Var}_i\!\Big(\frac{q\cdot k_i}{\sqrt d}\Big)=\frac{\|q\|^2\sigma_k^2}{d}=\sigma_q^2\sigma_k^2
\quad\Longrightarrow\quad \sigma_z=\sigma_q\sigma_k .$$

两者之比把 $\sigma_q$ 消掉了：

$$\boxed{\ \text{SNR}_{\text{attn}}\equiv\frac{\sigma_z}{\operatorname{std}(\delta z)}=\frac{\sigma_k}{\sigma_\varepsilon}\ }$$

**判据：当 $\sigma_\varepsilon<\sigma_k$ 时，logit 的相对次序基本保持；当 $\sigma_\varepsilon\gtrsim\sigma_k$，$\hat z$ 与 $z$ 的次序近乎无关，argmax 退化为在 $N$ 个位置里均匀随机抽。**

代入 §2 的数：

| K 的配置 | $D/\sigma_k^2$ | $\sigma_\varepsilon/\sigma_k$ | SNR | 预测 |
|---|---:|---:|---:|---|
| 1-bit min/max（现实现） | 5.21 | **2.28** | 0.44 | 崩溃 |
| 1-bit 最优电平（$\pm$MAD） | 0.363 | 0.60 | 1.65 | 可用 |
| 2-bit min/max（现实现） | 0.333 | 0.58 | 1.73 | 可用 |
| 2-bit 最优（$c=2.2$） | 0.117 | 0.34 | 2.92 | 宽裕 |

**这张表同时解释了经验事实的三个细节：**

1. **K1 为什么是 0 而不是"掉一半"**：$\sigma_\varepsilon/\sigma_k=2.28>1$，attention 已经不是在"看错地方"，而是在"随机看"。随机 attention 的视觉信息贡献趋于 0，模型退化到"只用文本 + 一个视觉平均色块"，对读数型题就是 0 分。
2. **K2 为什么完全恢复**：2.28 → 0.58，SNR 从 0.44 跳到 1.73，越过阈值。这不是"多了 1 bit 所以好一点"，是**跨过了一个相变点**。
3. **为什么阈值是 1 附近而不是别的数**：$\sigma_\varepsilon=\sigma_k$ 恰好是"误差与信号等能量"。超过它，$k$ 的估计与真值的相关性 $\rho=1/(1+\sigma_\varepsilon^2/\sigma_k^2)^{1/2}$ 掉到 $1/\sqrt2$ 以下。

### 3.2 由此得到第一个强预测

> **把 1-bit 的 min/max 电平换成 $\pm\mathbb{E}|x|$（MAD），K1 的 SNR 从 0.44 升到 1.65，已优于现在能用的 2-bit（1.73 的 95%）。因此 K1\* 的 ANLS 应当接近 K2（94.29），而不是 0。**

成本：编码器改两行常数（`min ← μ−MAD`，`scale ← 2·MAD`），**打包格式不变、Triton 内核零改动、字节零变化**。半天可验。

若成立，则 K 的字节从 2.25 → 1.25 bit/elem，**−44%**；K2/V1（1.75）→ K1\*/V1（1.25），**−28.6%**，已经越过"−20%"这条线，且是在同一个质量点上。

---

## 4. 命题三：V 为什么是缓坡——attention averaging 与"可吸收性"两个机制

### 4.1 机制一：输出误差被 $\sqrt{N_{\text{eff}}}$ 摊薄

$o=\sum_i a_i v_i$，$a=\text{softmax}(z)$。$\hat v_i=v_i+\varepsilon_i$：

$$\delta o=\sum_i a_i\varepsilon_i,\qquad
\mathbb{E}\|\delta o\|^2=\sigma_{\varepsilon,V}^2\sum_i a_i^2=\frac{\sigma_{\varepsilon,V}^2}{N_{\text{eff}}},\quad N_{\text{eff}}=1/\textstyle\sum_i a_i^2 .$$

所以判据从"$\sigma_\varepsilon<\sigma_v$"放宽成

$$\boxed{\ \sigma_{\varepsilon,V}<\sigma_v\sqrt{N_{\text{eff}}}\ }$$

V 比 K 多了一个 $\sqrt{N_{\text{eff}}}$ 的宽容度，这正是"V 是缓坡"的第一层原因。实测 attention 稀疏度 84.3%（老项目的数），$N_{\text{eff}}$ 量级 5–15，对应 $2.2$–$3.9$ 倍宽容。

代入 V 一比特 min/max（$\sigma_\varepsilon/\sigma_v=2.28$）：$2.28<3.2$ 时勉强通过 → 预测"掉几个点但不崩"。与实测 90.24/93.04（−2.8pp）一致。

**注意这条同时给出一个可测的层间预测**：$N_{\text{eff}}$ 越小（attention 越尖）的层，能容忍的 $\sigma_\varepsilon$ 越小。所以"哪些层必须 V2"不该靠搜索，而该算 $\sigma_{\varepsilon,V}/\big(\sigma_v\sqrt{N_{\text{eff}}}\big)$ 哪几层 $>1$。这支instrumentation只需在现有内核里多存一个 $\sum a_i^2$。

### 4.2 机制二：per-channel 的失真是一个可吸收的对角变换

这是 per-channel V（91.96）胜 per-token V（90.24）的原因，也是我认为更本质的一层。

一比特 per-channel 重建：$\hat v_{ic}=c_c\,\text{sign}(v_{ic})$，$c_c$ 只依赖通道、不依赖 token。于是

$$o_c=\sum_i a_i\hat v_{ic}=c_c\sum_i a_i\,\text{sign}(v_{ic}).$$

**$c_c$ 是一个固定的对角线性算子**，对整层的所有 token 一视同仁。这种扰动在数学上等价于把 $W_V$（或 $W_O$）的对应列乘一个常数——模型下游可以吸收，不产生"输入依赖的"错误。

一比特 per-token 重建：$\hat v_{ic}=3\sigma_i\,\text{sign}(v_{ic})$，$\sigma_i$ 是**该 token 自己的**元素标准差。于是

$$o_c=3\sum_i a_i\sigma_i\,\text{sign}(v_{ic}),$$

等价于把注意力权重从 $a_i$ 改成 $a_i\sigma_i$——**attention 被一个与内容无关的增益污染了**，而且这个污染随输入变，下游无法用一个固定算子吸收。

两种失真的区别是结构性的，不是程度上的：

| | 扰动形式 | 是否输入依赖 | 下游能否吸收 |
|---|---|---|---|
| per-channel | $c_c$（通道常数）乘在原输出上 | 否 | **能**（等价于改 $W_V$ 的列尺度） |
| per-token | $a_i\to a_i\sigma_i$ | 是 | **不能** |

这解释了实测的 91.96 与 90.24：两者逐元素 MSE 相同（都是 min/max 一比特），差别只在"扰动落在哪个子空间"。**逐元素重建误差相同、质量差 1.72pp，说明 elementwise MSE 根本不是 V 的正确目标函数。**

**推论**：per-token 与 per-channel 的差距，应该随"每 token 幅度 $\sigma_i$ 的离散度"增大而增大。这是一条可算的相关性检验（$\text{CV}(\log\sigma_i)$ vs 逐题 ANLS 差），成本为零。

### 4.3 由此给出 V 的两个免费/极便宜改进

- **免费**：per-channel V 的 1-bit 电平同样从 $\pm$半极差 改成 $\pm$MAD。$\sigma_\varepsilon/\sigma_v$ 从 2.28 降到 0.60，直接把 91.96 推向 93.04。这与 §3 是**同一个改动**。
- **更便宜**：V 的错误按 $a_i^2$ 加权，所以最优电平不是按均匀 MSE 拟合，而是按 $\sum_i a_i^2$ 加权的最小二乘——离线用平均注意力权重算一次即可。但收益小于上一条，排在后面。

---

## 5. 命题四：K 与 V 的容忍度差几个 bit——闭式分配规则

把 §3、§4 合成一句话：

$$\text{K 可用}\iff \frac{\sigma_{\varepsilon,K}}{\sigma_k}<1,\qquad
\text{V 可用}\iff \frac{\sigma_{\varepsilon,V}}{\sigma_v}<\sqrt{N_{\text{eff}}}.$$

用 min/max 编码（$\sigma_\varepsilon/\sigma_x = 2.28$ at $b=1$，$0.58$ at $b=2$，误差随 $b$ 大致按 $2^{-b}$ 衰减）：

| 轴 | 门槛 | $b=1$ | $b=2$ | 结论 |
|---|---|---:|---:|---|
| K | $<1$ | 2.28 ✗ | 0.58 ✓ | **K 必须 2 bit** |
| V | $<2.2\sim3.9$ | 2.28 ✓（勉强） | 0.58 ✓✓ | **V 1 bit 够** |

**这就是 SOL 用网格搜出来的 K2/V1=1.75 的那个点，现在它是被推导出来的，不是被搜出来的。** 这条本身有独立价值：它把"比特怎么分"从搜索问题（反事实干预 + 成对交换，需要 $O(\#\text{groups}^2)$ 次干预）变成一次除法。

而换成 MAD 电平之后（$\sigma_\varepsilon/\sigma_x=0.60$ at $b=1$）：

| 轴 | $b=1$（MAD 电平） | 结论 |
|---|---:|---|
| K | 0.60 ✓ | **K 只需 1 bit** |
| V | 0.60 ✓✓ | V 1 bit |

于是工作点从 K2/V1 = 1.75 移到 K1\*/V1\* = **1.25 bit/elem（fp16 的 7.8%）**，质量预测持平或更好。

### 5.1 这条推论给"比特交换"一个反直觉的结论

现在的交换逻辑是：K 贵、V 便宜，所以把 V 的位让给 K。这在当前编码下**正确**（边际收益 2.5 倍是真实的）。

但一旦 MAD 电平成立，K 的 1→2 bit 边际收益从 $\Delta D = 5.21-0.333=4.88\sigma^2$ 掉到 $\Delta D = 0.363-0.117=0.246\sigma^2$，**降了 20 倍**。此时"用 1 bit 去买 K 的可用性"等于**花一整个 bit 买一个免费的东西**。

> **所以顺序必须是：先免费修编码器，再谈置换。** 否则所有置换实验测的都是"补偿编码缺陷"的价值，不是"分配"的价值。

---

## 6. 命题五：per-channel 一比特 K 丢掉的是"每 token 的幅度"，而这一项便宜到 0.125 bit

§3 的 SNR 判据用的是元素级 MSE，它其实低估了 K 的问题。K 的分数对**每 token 的整体幅度**极其敏感：

$$k_{ic}\approx\alpha_i\,\sigma_c\,u_{ic},\qquad \alpha_i=\frac{\|k_i\|}{\sqrt d\,\bar\sigma}\ (\text{token 幅度}),\ \ u_{ic}\sim O(1).$$

$$\Rightarrow\quad z_i\approx\frac{\alpha_i}{\sqrt d}\sum_c q_c\sigma_c u_{ic}\ \propto\ \alpha_i .$$

也就是说，**跨 token 的 score 次序主要由 $\alpha_i$ 决定**。而 per-channel 一比特（无论 min/max 还是 MAD）都用"通道的典型幅度"$\sigma_c$ 或 $c_c$ 去替代了 $\alpha_i\sigma_c$，于是 $\alpha_i$ 这个因子整体丢失——所有 key 的范数被压成同一个数。

这正是 §3 里 SNR 表的物理含义：$\sigma_\varepsilon/\sigma_k=2.28$ 里，绝大部分能量来自 $\alpha_i$ 的丢失，而不是逐元素噪声。

### 6.1 因此有一个更强的编码：符号位 + 通道尺度 + **每 token 一个增益**

$$\hat k_{ic}=\alpha_i\,c_c\,\text{sign}(k_{ic}),\qquad
\alpha_i^\*=\arg\min_\alpha\sum_c\big(\alpha c_c\,\text{sign}(k_{ic})-k_{ic}\big)^2=\frac{\sum_c c_c|k_{ic}|}{\sum_c c_c^2}.$$

成本：每 token、每 KV-head、每层 1 个 fp16 → $16/128=\mathbf{0.125}$ bit/elem（用 fp8 则 0.0625）。于是

$$\text{K}:\ 1+0.125+0.25=1.375\ \text{bit/elem},\qquad
\text{K/V 平均}=(1.375+1.25)/2=\mathbf{1.3125}\ \text{bit/elem}\ (8.2\%\ \text{of fp16}).$$

**这个方案的理论性质**：$\alpha_i$ 精确恢复 token 范数，$c_c$ 恢复通道结构，只剩 $u_{ic}$ 的逐元素形状误差——而这一项的量级就是 §2 的 $0.363\sigma^2$ 下限。也就是说：

> **1.375 bit 的 K 在元素级 MSE 上与 1.25 bit 的 MAD 电平持平，但恢复了跨 token 的 score 次序；而 score 次序才是 attention 真正需要的东西。**

代价是内核要多一次 load + 一次乘（$k=(\text{code}\cdot\text{scale}+\text{min})\cdot\alpha_i$）。这是真改动，但不涉及数值稳定性，也不改打包格式。

### 6.2 分层预测（可证伪）

- **只做 MAD 电平（1.25 bit），不做 $\alpha_i$**：预测 K1 从 0 升到 **30–60**，但**不到 94**。因为元素 MSE 达标了，score 次序没完全恢复。
- **MAD 电平 + $\alpha_i$（1.375 bit）**：预测升到 **90 以上**，与 K2（94.29）在 CI 内不可分。

两条一起跑，就能把"K1 崩"拆成"编码常数 60%"+"token 幅度 40%"。这个归因如果成立，本身就是论文的一节——**它说明"1-bit K 不可用"这个流行结论是编码器造成的，不是信息论造成的。**

---

## 7. 三条对现有诊断的修正

### 7.1 温度 $\tau$ 扫描必须**分段**，否则会误判为"尺度不是原因"

§2 的 min/max 一比特把每个元素推到 $\pm3.0\sigma$，等于把 $k$ 的范数放大约 $3.0/0.798\approx3.8$ 倍。若只对**全部** logits 乘一个 $\tau$，视觉段与文本段被同比例缩放，而破坏恰恰是**视觉段相对文本段的尺度失衡**（文本 KV 是 fp16，未膨胀）。全局 $\tau$ 修不了相对量。

> **正确做法：$\tau_{\text{vis}}$ 与 $\tau_{\text{txt}}$ 分开扫，且 $\tau_{\text{vis}}$ 的候选区间应包含 $1/3.8\approx0.26$，而不是围绕 1.0。** 建议网格 $\{0.15,0.2,0.26,0.35,0.5\}$（log 间隔）。若只在 0.5–1.0 扫，会得到"$\tau$ 无效"的假结论。

### 7.2 三值 K 的性价比低于免费修，但也不该直接砍

三电平 Lloyd–Max 高斯失真 $\approx0.1905\sigma^2$（+0.625 bit，相对最优两级 1.9×）。与"直接上 4 电平"（0.117，+1.0 bit，3.1×）相比，**每 bit 的收益几乎相同**（3.0×/bit vs 3.1×/bit）。所以：

- 三值不是"更聪明"，只是"更便宜买半个 2-bit"；
- 而 MAD 电平是 **0 bit 换 14.3×**，性价比高一个数量级；
- 结论：**先做 MAD（半天），再决定要不要三值**。若 MAD 已把 K1 拉到 90+，三值的 10% 预算应该直接收回。

### 7.3 材料 39 的 37 题不能当"读轴判决"

`a2_all` 0.8514 vs `base2` 0.8205，差 0.0309；37 题下单题权重 $1/37=0.027$。**这个差值是 1.14 道题。** `a2_mass` 0.8137 比 `a2_none` 0.8205 低 0.68pp，是 **0.25 道题**。所有选择器之间的排序都在 1–2 道题之内，不能作判决，只能作方向。

读轴要出结论，最小样本是 294 题的 confirm64 级别（0.0309 对应 9 道题），或者更大。建议：**读轴第一枪用 294 题跑，不要用 37 题。**

---

## 8. 假设、反例与自我攻击

| # | 假设 | 若不成立会怎样 | 怎么提前知道 |
|---|---|---|---|
| 1 | 分组内近似高斯、独立 | 重尾时 min/max 更差、MAD 优势更大（方向不变） | 算每组 $\max/\sigma$ 与 kurtosis，预测增益 |
| 2 | 通道内 $u_{ic}$ 的离散度小，所以 $\alpha_i$ 是一阶项 | 若某通道跨 token 重尾（边界 token 极大），$\alpha_i$ 会压掉这些 token | 直接测 $\text{Var}_i(\|k_{ic}\|/\alpha_i)$ |
| 3 | 层间误差不累积（单层分析） | 28 层累积会把 SNR 判据变严 | 逐层注入实验（已有四臂能力） |
| 4 | $N_{\text{eff}}$ 在 5–15 | 若 attention 极度尖（$N_{\text{eff}}\to1$），V 也只剩 1 bit 的宽容度 | 内核多存 $\sum a_i^2$，一行 |
| 5 | ANLS 是阶梯函数（数字答案一个字错就 0） | "K1=0"里混着"指标台阶"，真实信息损失可能只有 60–80% | **同时报一个连续指标**（参考答案的 token 平均 log-prob，或与 FP16 的 logit KL），把"信息损失"和"指标台阶"分开 |

第 5 条我认为最容易被忽略也最容易致命：**"悬崖"可能是指标的形状而不是信息的形状。** 若连续指标显示 K1 只是平缓退化，那么"1-bit K 崩溃"这个说法本身就要改写，而 MAD 电平的收益也会比预测的小。**所以连续指标必须和 ANLS 一起报，不能只报 ANLS。**

---

## 9. 当天可跑的三条（按杠杆排序）

| # | 改动 | 成本 | 预测 | 判据（跑前写死） |
|---|---|---|---|---|
| **A** | 1-bit 电平 $\pm$半极差 → $\pm\text{MAD}$（K 与 V 同时改），字节不变 | 半天，编码器两行 | K1 从 0 → 30–60；V1 从 91.96 → ≥92.5 | K1 < 20 → 我的 SNR 判据错，回退 |
| **B** | 加每 token 增益 $\alpha_i$（+0.125 bit） | 1 天，含内核一次乘 | K1 → ≥90，与 K2 不可分；总比特 1.375 | K1 < 60 → "token 幅度"不是主因，改写 §6 |
| **C** | 分段 $\tau$（$\tau_{\text{vis}}\in[0.15,0.5]$）作诊断，不做方法 | 2 小时 | A 的前置诊断：若单靠 $\tau$ 就能把 K1 拉到 60+，说明主因是尺度 | 与 A 对照，判断"尺度 vs 次序"占比 |

三条都在 20 题探针上跑，当天出数；成立后再上 294 题。

**若 A 和 B 都成立**，论文的主线就变成一句可以推导、可以证伪、且成本极低的话：

> 视觉 KV 的一比特失效不是信息论极限，而是编码常数（min/max）与丢失 token 幅度造成的；给出 SNR 判据 $\sigma_\varepsilon<\sigma_k$，据此把比特分配从搜索问题化为闭式规则，并在 1.31 bit/elem（fp16 的 8.2%）上达到 2-bit min/max 的质量。

这句话比"我们做了一个非对称分配 + 预算化精化"强得多，因为它有一个**可独立检验的判据**，而后者只有一张表。

---

## 10. 与已有工作的边界（按"组合创新"口径）

| 已有的单点 | 我们借用什么 | 我们新加什么 |
|---|---|---|
| KIVI：K per-channel、V per-token、min/max | 轴的选择（已核实正确） | **证明 min/max 在一比特下比最优差 14 倍**，并给出闭式替代（MAD） |
| CalibQuant：post-scaling + pre-softmax 校准 | 校准思想 | 校准是全局的；我们指出**丢失的是每 token 的幅度**，需要 per-token 增益而非全局 scale |
| KVarN：Hadamard + 双轴方差归一化 | 数值对齐的底座 | 它治的是 token-scale 异常；我们给的是**"该给哪个轴几个 bit"的闭式判据** |
| ReadKV：存储/读取双预算、渐进前缀 | 二维预算框架 | 我们把"比特怎么分"从搜索降为除法，并指出**先修编码再置换**的顺序 |

**不能声称的**：首次发现量化误差会传播、首次做视觉 1-bit、首次做非对称分配、首次做存储/读取双轴。**可以声称的**：一个把 K/V 比特分配从搜索变成闭式规则的 SNR 判据，以及"一比特失效主要来自编码常数与丢失 token 幅度"这一可分离归因。
