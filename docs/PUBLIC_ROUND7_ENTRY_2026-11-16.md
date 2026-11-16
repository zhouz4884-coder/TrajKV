# TrajKV — 视觉KV低码率表示与覆盖分配

## Round7（2026-11-16）

[完整研究报告](docs/RESEARCH_PROGRESS_2026-11-16_round7.zh.md) · [方法与比较口径](docs/METHODS_ROUND7_2026-11-16.zh.md) · [本轮结果导航](docs/START_HERE_ROUND7_2026-11-16.zh.md) · [目标v8及达成情况](docs/GOAL_2026-11-16_v8_ENCODING_AND_CONDITIONAL_EXCHANGE.md)

两张RTX3090执行真实packed KV及完整FP16模型自由生成，数据与新缓存使用数据盘。主确认固定九方法、Doc128源/412问＋Text128源/128问；第二骨干和固定输出成本单列。

| 工作点 | 完整视觉KV bit/元素（约） | 本轮Doc ANLS | 本轮Text soft acc. |
|---|---:|---:|---:|
| 原覆盖 | 1.800 | 98.144 | 82.344 |
| 新G4_16＋43级分批分配 | 1.295 | 97.680 | 80.938 |
| base2（KIVI式视觉轴） | 2.252 | 96.639 | 81.094 |
| Native4 | 4.252 | 98.414 | 82.109 |
| Full | 16.002 | 98.617 | 83.203 |

新低码率点比原覆盖少约28%视觉KV字节，Doc/Text分别低0.464/1.406pp；相对Native4的两域0.5pp质量目标尚未达到。分批分配可以被相同批内约束下的初态静态分数精确复现，当前结果不支持反馈重估的独立增益。

这是project-fresh派生队列上的结果：Doc来自额外官方train分片，不是完整标准测试集或基础模型训练未见保证。bit包含实际视觉码流、元数据、残差及encoder常量，模型权重和文本KV仍为FP16。全部九方法、配对区间、开发反转和成本都在报告中。

[Round7原始证据、执行源码版本及指纹归档](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round7-20261116)。旧阶段报告、提交和release保留。
