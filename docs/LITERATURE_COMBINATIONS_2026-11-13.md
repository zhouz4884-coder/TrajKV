# 本轮组合与直接对照

本轮把相关工作当作可组合的技术来源，以实测新增能力决定主线。预印本状态不作为否定研究的理由；归因和直接比较按技术重合程度安排。

| 来源 | 本轮使用的原料 | 当前可检验的组合变化 |
|---|---|---|
| [Quest](https://arxiv.org/abs/2406.10774)，ICML 2024 | 页级 K min/max 与当前 q 估计重要性 | 用于选择残差读取；所有未选页仍以 base 参与 attention，mass-only 进入直接对照 |
| [SparQ](https://arxiv.org/abs/2312.04985) | query 条件的选择性读取 | 区分选支持与选保真度，显式计选择输入/临时量及实际耗时 |
| [ReadKV](https://arxiv.org/html/2610.11245v1) | 渐进码、读取预算和 query 条件分配 | 直接相关；不把“存储与读取分开”单独当作本轮创新。当前实验考察视觉页残差的全支持执行及与历史释放的资源关系 |
| [AATC](https://arxiv.org/html/2608.14191v1) | attention-aware 的误差权重 | raw MSE、q 投影、mass 与 value 对比作为不同可测量，不把原始误差集中当成必要条件 |
| [CalibQuant](https://arxiv.org/html/2502.14882v2) | 通道量化、post-scale 与 pre-softmax 校准 | 接官方 quant/encode 数值函数，1-bit clipped K/V + decode 校准作质量参考；与轴残差按真实存储比较 |
| [TriAxialKV](https://arxiv.org/html/2605.17170v1) | modality、recency、role 三轴分配 | C 在同 user role 内混合图像可再得事实与会话私有绑定，并反转顺序，检验 role/recency 之外的条件价值 |

## 已核正的源码事实

CalibQuant 原论文 DocVQA 表包含 1/2/4/8-bit；旧 patch 的固定 `BITS_KV=1` 不能推出该方法只支持 1-bit。该文件的 `calibquant` 与 `qjl` 是两个独立 forward，默认 `patch_model(..., quant_method='calibquant')`。本轮接前者的量化/校准原理，不能将其标为 CalibQuant-QJL。

本轮 Calib 端口替换固定视觉 span，保留现代 HF/GQA 的原位置；其 packed 记录与解码后的 FP16 所有者同时计入内存。使用原码的 quant/encode 与独立 unpack 验证；运行路径为数值质量参考，尚非原论文 serving 复现。单 query decode 的校准偏移沿用 -3，多 query 后续文本预填使用量化视觉而不做该 decode 校准，范围在每条记录标注。

A2 当前 selector 是 Quest 式页上界加可在线存储的残差范数/value 对比的启发式，不能称为 attention 误差的严格上界。CS 诊断只检验视觉 score 的 top-1 排序证书，不推导最终答案安全性。

## 论文主张随数据确定

若页风险策略在隔离文档、同存储/读额度下超过随机和 mass/全局均摊，再围绕“视觉全支持、事件级保真度分配”组织主张。若收益主要来自表示，围绕实测轴不对称与校准的组合继续；若 C 显示可再得事实能真实释放内存并支付视觉保真度升级，则联合预算成为候选主线。本轮完整报告同时保留正例、新错和成本。
