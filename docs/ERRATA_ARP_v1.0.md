# ERRATA：ARP v1.0 的五处实质性问题与修正

**日期** 2026-11-12 ｜ **对象** `VisContract_P14_2/docs/PLAN_2026-11-12_研究方案_非对称读取精度.md`
**性质** 该方案未进入执行阶段，以下修正替代原文对应章节。v1.0 保留存档，不覆盖。
**核实方式** 逐条回溯到 `VisContract_P14_2/scripts/` 的实现与 `results/p201|p202/results.json` 的原始字段。

---

## E1｜注意力读出误差分解写错（严重，影响方法立论）

### 原文（v1.0 §4.2）

$$
\Delta Y = \sum_i a_i(\varepsilon_i-\bar\varepsilon) + \sum_i a_i(\Delta z_i-\overline{\Delta z})(v_i-\bar y)
$$

并据此声称："V 的共同偏移被消掉，所以 V 只需保留方向与相对幅度。"

### 问题

两处错误：

1. **代数错误**。$\sum_i a_i(\varepsilon_i-\bar\varepsilon)=\sum_i a_i\varepsilon_i-\bar\varepsilon$（因 $\sum a_i=1$）。只有当无权均值 $\bar\varepsilon=0$ 时才等于 $\sum_i a_i\varepsilon_i$，而量化误差的组内均值一般不为零，所以这不是恒等变形。
2. **直觉与反例冲突**。若所有 $v_i$ 同加常数 $c$，则 $y$ 也加 $c$，不是不变。softmax 的平移不变性只作用于 **logit**（$z_i+c$ 不改变 $a$），不作用于 **value**。

### 修正

令 $\tilde k_i=k_i+\delta k_i$、$\tilde v_i=v_i+\varepsilon_i$、$\Delta z_i=q^\top\delta k_i/\sqrt d$、$\mathbb E_a[\Delta z]=\sum_j a_j\Delta z_j$：

$$
\boxed{\;\Delta y \;\approx\; \underbrace{\sum_i a_i\,\varepsilon_i}_{\text{(I) V 直接进入}}\;+\;\underbrace{\sum_i a_i\big(\Delta z_i-\mathbb E_a[\Delta z]\big)(v_i-y)}_{\text{(II) K 改变注意力分配}}\;}
$$

二阶余项含 $\delta^2$ 与 $\delta\varepsilon$ 交叉项。

### 这条修正带来的新内容（不是纯认错）

原式给出的是"V 只要方向"，修正后给出的是**可检验的方差预言**：

- (I) 的方差为 $a^\top\Sigma_\varepsilon a$。若各 token 的量化误差独立同方差 $\sigma_V^2$，则 $=\sigma_V^2\sum_i a_i^2$。
- 于是 V 误差的衰减来自**注意力集中度**，不是中心化：注意力越尖锐（$\sum a_i^2\to1$），V 误差几乎原样进入输出；越平坦（$\sum a_i^2\to1/n$），越被平均掉。
- 这与"decode 注意力尖锐 ⇒ V 不重要"的朴素直觉**相反**，是一个可以直接证伪的预言，列入 P1-C。
- (II) 保留中心化是合法的（logit 共同平移确实被 softmax 消掉），权重是 $(v_i-y)$，即值相对输出的偏差。

### 影响

| 原结论 | 修正后 |
|---|---|
| "V 只需保留方向"是**由公式推出**的 | 降级为**编码假设**，须由 P1-C 的实验支持 |
| "V 的共同偏移可抵消" | **删除**，不再出现 |
| "K 需要残差保序" | **保留**，但理由改写成：(II) 是 $\Delta z$ 与 $(v_i-y)$ 的协方差，单纯保 top-k 幅值不够，需保 mass 内部相对次序 |

---

## E2｜字节口径：我的配置写错，且原表混了两种记账方式

### 原文（v1.0 §4.4/§4.7）

K ≈ 5.5 bit/elem、V ≈ 3.5 bit/elem，合计 9/32 = 28.125%，同时把 24.61% 当作同一个量的实测值。

### 核实结果（逐行回溯到代码）

`scripts/p201_refinement.py:5-17` 的 `quantize()`：

- 组大小 **128**（不是 32）；
- 每组存 **`scale` 与 `minimum` 两个 fp16**（不是单个 scale），即 32 bit / 128 elem = 0.25 bit/elem。

按真实配置重算：

| 分量 | 数据位 | 元数据 | 合计 bit/elem |
|---|---:|---:|---:|
| K base 2bit | 2 | +0.25 | **2.25** |
| K + 残差 2bit | 2 | +0.25 | **+2.25 → 4.50** |
| V base 2bit | 2 | +0.25 | **2.25** |
| V + sign + α | 1（sign）+ 0.125（α，fp16/128） | 复用 base | **+1.125 → 3.375** |
| **合计** | | | **7.875** |

$7.875/32 = 24.609\%$ → 与 `results/p202/results.json` 的 `payload_ratio=0.24609375` **逐位一致**。

独立 4-bit（`aligned`）：$4.25+4.25=8.5$，$8.5/32=26.5625\%$ → 与 `0.265625` 一致。

**结论：24.61% 与 26.56% 都是对的；错的是 v1.0 里 5.5/3.5 那组配置。** 用户按 5.5+3.5 算出的 28.125% 是对该错误配置的正确反驳。

### 但顺带查出更危险的一条

同一个 `results.json` 里 `k4v4` 行的 `payload_ratio = 0.40625`（**40.6%**），不是 26.56%。原因在 `scripts/analyze_p202.py:39` 的 `scope` 字段自述：

> `payload includes base plus all prefill-only encodings, not runtime memory`

即 payload 统计的是"base + 变体叠加"的**物化产物大小**，不是稳态驻留。因此：

| 行 | 语义 | 可部署字节 |
|---|---|---|
| `base` | K2V2 本身 | 14.06% |
| `asym_sign` | base + K 残差 + V 符号 | 24.61%（base 是残差的解码前提，必须保留）✓ |
| `aligned` | 独立 4-bit | 26.56% ✓ |
| `k4v4` | base **加** K4V4（4bit 不需要 base） | 报 40.6%，**虚高**，不可用 |

**含义**：残差类和独立量化类可以放在同一张表比，但 `k4v4` 这一行必须重算成独立部署口径（8.5 bits）或直接删掉。任何从旧表直接搬数字的表格都会被这一条打穿。

### 修正后的记账规则（写进新项目代码强制约束）

1. 每个配置必须输出 `bytes_breakdown.json`：逐层、逐张量列出 `data / scale / zero / residual / padding / index`。
2. 分母固定为该模型**视觉 token 的 FP16 K+V** 字节，不用全模型。
3. 禁止"base + 变体"累加口径；前缀 `materialized_` 只允许出现在调试产物里。
4. 表头必须写明 `deployed` 还是 `materialized`。

---

## E3｜阶段读取的定义不完整，且混淆了存储与带宽

### 原文（v1.0 §4.5）

| 阶段 | K | V |
|---|---|---|
| prefill | base + residual | base + sign |
| decode | base + residual | base + sign + α |

### 问题（三条，均成立）

1. **`sign` 离开 `α` 无法重建**。残差是 $\alpha\cdot\mathrm{sign}$，只读 1 bit 符号而丢 $\alpha$，要么得到幅度为 1 的伪残差，要么等于不读。原文没有定义"不读 α 时的重建规则"。
2. **阶段切换不改变驻留字节**。base/residual/sign/α 若全部常驻显存，prefill 读哪些分量只影响**读带宽**，不影响存储。原文把"阶段分离"和"省字节"混在一起，会导致 E3 的因子不干净。
3. **prefill 有三个不同阶段被混为一谈**：
   - ① 首轮视觉前缀 prefill（发生在量化之前，此时还没有压缩缓存）；
   - ② 后续轮新增**文本** token 的 prefill（视觉前缀不重算）；
   - ③ 逐 token decode。
   阶段策略必须指明作用于 ②/③ 中的哪一个。

### 修正

- 把主张拆成两个互不替代的量：**$B_{\text{stored}}$（常驻字节）** 与 **$B_{\text{read}}(s)$（阶段 $s$ 的读字节）**。阶段策略只允许声称降低 $B_{\text{read}}$，不得声称降低 $B_{\text{stored}}$。
- 若要做"不读 α"，必须定义显式的降级重建 $\hat v = \mathrm{dequant}(v_{\text{base}}) + \bar\alpha\cdot\mathrm{sign}$，其中 $\bar\alpha$ 是**预先写入的全局常数**（额外 2 字节），并在消融里作为独立因子，不能隐含。
- 阶段因子固定作用于 ② 与 ③，②与③分开测；①不参与（量化尚未发生）。
- 实验设计改为：**固定存储表示，只变读取掩码**，比较读字节 / 输出差异 / 时延。原文"为凑等字节而同时换编码"会破坏因子独立性。

---

## E4｜验收目标选在了接近饱和的区间

### 原文（v1.0 §6.1/§6.4）

"相对最强低比特基线 ΔEM ≥ +2.0pp"，主战场定为 GQA grouped。

### 问题

GQA grouped 现有实测：FP16 0.332、KIVI2 0.324，**差距只有 0.8pp**。在这个区间要求"比最强基线高 2pp"，等价于要求超过 FP16 约 1.2pp，物理上不可能是方法本身的稳定收益，只可能是噪声。这是**天花板效应**，不是方法难度问题。

### 修正

**先用"可改进空间"筛战场，再定效应量。** 定义：

$$
G = S_{\text{FP16}} - S_{\text{base}},\qquad
\mathrm{GC} = \frac{S_{\text{method}}-S_{\text{base}}}{G}
$$

- 准入条件：该 (dataset, budget, model) 组合必须满足 $G \ge 5$ pp，否则不作为主战场（只能作"不劣"证据）。
- 主指标从"绝对 Δ"改为 **GC（gap closure）**，辅以绝对 Δ：要求 $\mathrm{GC}\ge 30\%$ 且 CI 下界 > 0。
- 候选战场（需 P0 实测确认 $G$）：DocVQA 在 1-bit、GQA 在 1–1.5 bit、跨图事实关联、6.6k token 长上下文、困难 OCR 子集。

**理由**：GC 是尺度无关的，能在不同 $G$ 的区间之间比较，也直接回答"你恢复了多少本来会丢的东西"，比"高 2pp"更难被 ceiling 打掉。

---

## E5｜统计功效与算力估算不一致

### 原文（v1.0 §6.4）

"CI 半宽 ±1.5pp、80% power MDE ≈ 1.6pp"。

### 问题

双侧 $\alpha=0.05$、power 0.8 时，$\mathrm{MDE}=(z_{0.975}+z_{0.8})\,\mathrm{SE}=2.80\,\mathrm{SE}$，而 CI 半宽 $=1.96\,\mathrm{SE}$，故 $\mathrm{MDE}\approx1.43\times$ 半宽。半宽 1.5pp 对应 MDE ≈ 2.15pp，不是 1.6pp。两者不自洽（用户指出，成立）。

另外，用独立比例方差估 SE 会高估配对设计的功效；真实 SE 取决于**配对不一致率**与**图像级相关**，必须从 pilot 的配对差向量估计。

### 修正

1. 删除硬编码的 MDE 数字。
2. 改为**基于 pilot 的功效模拟**：取 60–100 会话的配对差向量，按图像 cluster 做 bootstrap，重采样得到 $\mathrm{SE}_{\text{paired}}$，再解出达到 80% power 所需会话数。
3. 报告时同时给：绝对 Δ、GC、cluster bootstrap 95% CI、Holm 校正后的 $p$、以及"该样本量能检测到的最小效应"。
4. 算力重估：

| 阶段 | 内容 | GPU-days |
|---|---|---:|
| P0 审计 | 复算编码、字节、FP16/KIVI2/native4 同语义 | 2–3 |
| P1 机制 | 注入实验 + 敏感性漂移 + 历史分支 | 3–4 |
| P2 方法 | 6 层基线 × 3 预算 × 2 模型 | 10–15 |
| P3 系统 | 读取策略与内核 | 3–5 |
| 合计 | | **18–27** |

v1.0 的 5–7 GPU-days 明显偏低，按 579 会话 × 7 方法 × 95s ≈ 107 单卡小时（≈4.5 GPU-days）只是 E1 一项，双模型即 9 GPU-days，尚未含消融与重复。

---

## 修正后的"可保留 / 须撤回"清单

| v1.0 内容 | 处置 |
|---|---|
| §4.2 的 (I) 项及"V 只需方向"的推导 | **撤回**，改为 E1 的编码假设 + P1-C 检验 |
| §4.2 的 (II) 项（K 重分布） | **保留**，中心化合法，理由重写 |
| §4.4 的 5.5/3.5 bit 配置 | **撤回**，改为 4.50/3.375（g=128，scale+min） |
| §4.4 的编码结构（K base+r2，V base+sign+α） | **保留**为候选算子（P202 已验证可实现） |
| §4.5 阶段分离 | **降级**为"读带宽策略"，且须固定存储表示 |
| §4.7 开销表 | **重写**，区分 stored / read |
| §6.1 的 +2.0pp 验收线 | **撤回**，改为 GC ≥ 30% + 战场准入 $G\ge5$pp |
| §6.4 的功效数字 | **撤回**，改为 pilot 驱动的模拟 |
| §5.5 的 6.5 GPU-days | **撤回**，改为 18–27 |
| P202 的 24/36 种子 | **保留**为效应量先验，不作证据（n=36，CI 约 ±12pp） |

---

## 附：本次核实顺手发现的一个反例

`results/p202/results.json` 里，`aligned` 的 `attention_error=0.0176`（比 `asym_sign` 的 0.0307 低 43%）但 `correct=23`（比 `asym_sign` 的 24 低）。`k4v4` 的 attention_error 最低（0.0119），correct 也只有 24。

**含义**：注意力输出 MSE 的排序与答案正确率的排序在 n=36 上不一致。这不能证明"MSE 是坏的代理"（样本太小），但足以说明**不能把注意力 MSE 当作唯一优化目标**，分配目标必须在 P1 阶段与答案级指标做相关性验证后再定。这一条直接影响 v2 的方法设计。
