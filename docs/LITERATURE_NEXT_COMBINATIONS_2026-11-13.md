# 下一轮可借鉴的组合

现有100图已支持“编码方式与任务质量存在强交互”，但通用的norm/angle、query-aware quantization本身不是空白。下面列的是可组合部件与本项目可测的新增条件。

| 来源 | 值得借鉴 | 本项目先做什么 |
|---|---|---|
| [SageAttention](https://arxiv.org/html/2410.02367v1) | 沿token轴去除K共同均值，显露token差异信号 | 视觉独立压缩时保留q·视觉均值偏置；验证中心化与旋转的交互 |
| [TaDA](https://arxiv.org/html/2506.04642v1) | 跨head均值与偏差，以及逐层精度选择 | 区分平均轴，不将其误称token均值法 |
| [NOVA-KV](https://arxiv.org/html/2608.04074v1) | 用query统计定义注意力乘积的失真，再设计变换与VQ | 从廉价的query对角二阶矩开始，与数据无关旋转同成本比较 |
| [KVQuant](https://arxiv.org/html/2401.18079v1) | pre-RoPE、非均匀datatypes、独立处理outliers | 若post-RoPE方向码仍差，明确用pre-RoPE对照定位位置旋转影响 |
| [QJL](https://arxiv.org/html/2406.03482v1) / [TurboQuant](https://arxiv.org/html/2504.19874v1) | sign sketch、内积校正、变换与Lloyd | 先测数据无关Hadamard和幅度法，再考虑校准变换或小向量码本 |

一个低成本可推导的候选：令旧开发数据上的query二阶矩近似为对角M，取正对角S≈sqrt(M)。编码残差(K−μ)S后旋转；读取query用qS^{-1}作相同旋转，并加回q·μ。未量化时恒等保留内积；量化后的误差在对角query度量下评价。这是已知变换编码思路的廉价近似，不冒称完整NOVA-KV或普适最优。

只用旧20图的无标签FP16 query做校准，参数在新图前冻结；每层/KV-head的128个FP16尺度共28,672字节，另计Hadamard、均值和token幅度。后续真实packed实现要将这些常量和query变换读/计算费用计入。

对照需保留无中心/有中心、无旋转/有旋转、无query缩放/有query缩放，以及相同payload的现有K2/V1和同格式单标量方法；先真实GPU模型验证质量，出现收益后再实现packed系统评估。
