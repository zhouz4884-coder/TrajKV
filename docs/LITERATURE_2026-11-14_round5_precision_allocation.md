# Round5：静态精度分配与符号残差的直接文献参照

日期：2026-11-14。检索范围：layer×KV-head静态混合精度、模型输出/损失决定分配、低比特残差修复。本文仅查公开原文与本轮方法接口，未读取新确认集。已知16份开发文档上的改善属于探索信号，本文不将其写成最终效果。

## 检索与核实记录

- 使用 `nature-academic-search` 的 `multi-source-search` 工作流，读取manifest、core、wf1和dedup规则；专用academic MCP未挂载。
- skill `preflight.py`：PubMed、CrossRef、arXiv API均可达（3/3）。
- skill `academic_search.py`：主题检索limit8、缩窄KVTuner检索limit3，两次均返回OpenAlex HTTP429。按fallback使用公开web发现，再读arXiv HTML/摘要页原文；按arXiv ID与题名去重。
- OpenReview的CalibQuant/KVTuner PDF地址本次均转到浏览器验证页，未据此认定版本或录用状态；改用其arXiv原文核实。没有绕过验证、没有引用搜索摘要替代下面的技术依据。
- 部分猜测的旧HTML版本号返回404，已用可访问v1/v2/v4正文核实；链接均指实际读取的版本。此笔记是针对当前实现的定向检索，不是穷尽综述。

## 六篇最直接的原文

| 原文与身份 | 实际做法及与当前工作的重合 | 可借组件与应有对照 |
|---|---|---|
| **RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory**，Fei Zuo等，2026，arXiv:2605.06675v2。 | per-layer×KV-head分配；用损失对K/V缓存的梯度估计敏感性，拟合具体量化器的失真曲线，逆水填充及整数边际收益分配；明确讨论K/V分开校准与预算。静态head分配、闭式水位、校准失真均已有直接先例。[原文§3、附录E/H](https://arxiv.org/html/2605.06675v2) | **最近的分配算法参照。** 优先移植“梯度敏感性×本码实际残差收益”到相同112单元、同一packed residual格式，与当前局部输出误差排序竞争；应标注为allocation strategy port，不冒称复现其完整pipeline。 |
| **KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference**，Xing Li等，2025，arXiv:2502.04420v1。 | 用relative attention-output error与精度成本作层内Pareto剪枝和层间聚类，再用最终校准任务accuracy与memory进行多目标搜索。**它没有主张只看局部误差就足以替代端到端评测。**[原文§5.3、附录A.3–A.4](https://arxiv.org/html/2502.04420v1) | 借其层级候选剪枝；同预算增加“整层优先、层内均匀”的粗粒度对照，判断head粒度是否带来额外质量。若只移植分配规则，明确范围；无需在当前探索期重启大规模Optuna搜索。 |
| **HeadQ: Model-Visible Distortion and Score-Space Correction for KV-Cache Quantization**，Jorge L. Ruiz Williams，2026，arXiv:2605.03562v1。 | K侧用校准query基存低秩残差并加到logits；V侧给固定attention下的精确读出误差及弱相关条件下的A²代理，另有full-KV组合实验。因此“元素MSE不是模型可见失真”“K残差side code”“A²分配”都需归因。[原文§3–5](https://arxiv.org/html/2605.03562v1) | 用A²加权残差作一个便宜排序对照；比较我们经GQA及W_O汇合的实际误差/收益能否更好指导同价单元选择。当前直接MAD符号残差不等于其低秩query-basis码。 |
| **CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs**，Susan Liang等，2025，arXiv:2502.14882v2。 | 直接面向视觉KV；channel-wise量化K/V，post-scale结合后续乘法，且用固定校准参数修正pre-softmax分数分布。视觉低比特、轴选择和分数校准是其明确内容。[原文§4](https://arxiv.org/html/2502.14882v2) | 继续用其算子与校准口径作视觉直接参照；论文级比较需同模型、视觉/文本压缩范围、总字节。旧移植失败结果不能代替有效基线，更不能据此宣布方法失效。 |
| **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**，Amir Zandieh等，2025，arXiv:2504.19874v1。 | 随机旋转后scalar Lloyd-Max；prod版本在较低位宽MSE码后加1bit QJL残差，针对无偏inner-product估计。**base量化＋符号残差**本身已有强先例。[原文§1.3、§3.2](https://arxiv.org/html/2504.19874v1) | 借旋转、码本与残差的分项消融。我们的MAD符号直接重构是确定的近似，不具QJL无偏保证；应比较原坐标/旋转坐标、K/V用途与实际完整成本，不能把sign1命名成新的普遍原理。 |
| **GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM**，Hao Kang等，2024，arXiv:2403.05527v1。 | 低位宽backbone、低秩量化残差与稀疏outlier三部分组合；保存/恢复残差的思想已存在。其低秩对象和稀疏对象不同于我们的整头符号码。[原文§3](https://arxiv.org/html/2403.05527v1) | 借“先测残差结构，再选择修复形式”。当前低rank-2解释力弱只是本底座/样本的线索；若比较GEAR式低秩修复，必须计基矩阵、系数、稀疏索引与解码，不以不匹配工作点的4bit结果当等字节结论。 |

## 其余被提到的方法，分别在哪个位置

| 原文 | 与本轮的实际关系 |
|---|---|
| [KVQuant，2401.18079v4](https://arxiv.org/html/2401.18079v4) | 有Fisher敏感性加权非均匀码本、pre-RoPE K及dense-and-sparse outlier处理，是“损失权重参与编码”的先例。它不只是轴选择；当前后RoPE框架不能直接套pre-RoPE路径而省略位置处理。 |
| [KIVI，2402.02750v1](https://arxiv.org/html/2402.02750v1) | 核心是K per-channel、V per-token及近期高精度buffer。它支持K/V读出结构不同，不等于证明所有任务都应给K更多payload。比较时注明视觉是否有FP16窗口、文本是否压缩，并计窗口字节。 |
| [MiKV，2402.18096v1](https://arxiv.org/html/2402.18096v1) | 重要token保较高精度，原本会驱逐的token保低精度，适合说明“全支持＋局部高精度”已有基础。其分配单元是token重要性缓存，不能直接当静态layer×head的完全相同算法。 |
| [Palu，2407.21118v2](https://arxiv.org/abs/2407.21118) | 分解线性层、缓存低秩中间状态并恢复KV，附rank搜索、量化适配与融合kernel。可借低秩与量化组合；当前事后对缓存做投影的机制臂不能称完整Palu复现。 |
| [TriAxialKV，2605.17170v1](https://arxiv.org/html/2605.17170v1) | 离线量attention-output MSE，按时间、模态、语义角色tag分INT2/INT4，使用混合池与融合decode。按输出损伤配置静态精度不是空白；它的tag粒度与当前视觉层×head粒度不同，可借分类/预算选择作为粗粒度参照。 |
| [Read What Matters / ReadKV，2610.11245v1](https://arxiv.org/html/2610.11245v1) | 分离存储与每query读取预算，progressive prefix码；先按q分K-channel读取，再按attention分V-token读取。边际收益分配及条件性最优已有明确讨论。当前静态少量残差根本不存、存下来的每次都读，属于存储分配；不要将其写成首次两个资源轴或直接宣称超越ReadKV的读取模型。 |
| [QJL，2406.03482v1](https://arxiv.org/html/2406.03482v1) | JL随机投影＋sign量化、非对称inner-product估计，是TurboQuant符号残差的重要组成。我们的MAD幅度与直接加回原空间误差使用不同估计器，需分清保证。 |

## 对下一轮最有价值的组合与实验

当前真正值得验证的组合是：**视觉V1底座的功能校准＋codec-conditioned的局部修复收益估计＋静态紧凑符号残差＋真实每图预算。** 这是待证明的整体能力，不预先认领上述单组件的首创。

最接近现有实现、应优先加入同预算比较的是以下两种分配规则：

1. **RateQuant风格的梯度权重×实际失真降低。** 保持同一底座、同一sign/MAD residual、相同可选单元和字节；仅改变冻结排序。梯度用校准teacher自生token的CE或明示的代理损失，具体定义记录。这个策略移植用来区分“任意敏感性分配都行”与当前输出读出量的新增价值；不称完整RateQuant复现。
2. **KVTuner风格的层级粗分配。** 将同码残差的局部读出收益先按层聚合，同预算给整层/层内均匀单元，与逐head排序竞争。若没有运行原论文的Pareto＋MOO搜索，表中称layer-level allocation control即可。

随机、均匀和同价K残差对照继续保留。所有排序都用同一校准池，冻结后评估；读取时不看参考答案。比较必须使用含lookup、参数、码本、残差及必要位置索引的完整账，不能只把平均payload写成总bit。

此外，当前排序若仅用“base已有输出误差”，下一步可直接挑战它：计算每单元加入**本码实际残差**后的局部误差降低

`G(g)=E[ ||delta_y_base||² − ||delta_y_after_actual_residual_g||² ]`。

这是把误差大小改成这笔预算真正能买回多少，属于对当前原型的设计建议。跨head/跨层交互会让收益非可加，所以即使同价排序优化了一个局部代理，也不能宣称端到端最优。若实测收益排序、梯度排序、A²排序都相近，论文重点应放在已证实的可执行组合；若新收益量在同码同预算下更好，才有理由强化估计器贡献。

本轮无需因为这些相关工作而停止探索，也不必一次重建所有完整系统。先用兼容本码的强分配规则竞争，冻结候选后再补必要原算法/质量前沿。本文没有使用新确认结果选择方法，也没有为未核实的出版状态作判断。

## RateQuant策略移植的实现口径

主文Algorithm1/§3.2的权重是

`w_lh^A = E_x[(1/T) * sum_t || dL/dA_lh,t ||_2²]，A∈{K,V}`。

向量范数沿head维，时间位置与校准序列取平均；L是causal-LM loss。原实验用WikiText-2文本，附录W.8明确CE，并另讨论逐target-token梯度能量的平均。因此“mean CE求一次梯度再平方”和“每target单独求梯度再平均”要区分。[主文§3.2与附录W.8](https://arxiv.org/html/2605.06675v2)

我们的strategy-port限定为：只对视觉缓存求梯度，以Full自生答案token的pooled CE为loss，明确与原论文文本next-token校准不同；K/V保留公共loss尺度，不擅自各归一化到单位均值。原文K/V各自拟合失真、共享总预算；本次以同一码的实际残差MSE降低代替指数拟合，再乘权重排序。[原文§3.4](https://arxiv.org/html/2605.06675v2)

Full自生token是伪标签；若在相同Full状态直接计算teacher-vs-self KL，零梯度不能当作敏感性。pooled CE通常有非零梯度，但是否能指导本工作点，仍由与局部有限干预排序、随机和均匀的同预算实验回答。
