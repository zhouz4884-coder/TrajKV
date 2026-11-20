# 首个完成的 M1 FP16 ChartQA 完整 split 参考银行

在固定的 InternVL3-8B-hf、FP16、max_patches=8、图像前缀＋完整问题 suffix 重算协议下，ChartQA test **2500/2500题全部完成**，Google Pix2Struct 5% relaxed accuracy 为 **86.24%**。这是本项目该协议的无压缩参考，不是压缩算法结果，也不表示整个 P2/P3 或外部 SOTA 库完成。

| 分组 | 题数 | relaxed正确 | 准确率 | source-cluster bootstrap 95%CI |
|---|---:|---:|---:|---:|
| Overall | 2500 | 2156 | 86.24% | 84.8374–87.6533% |
| Human | 1250 | 971 | 77.68% | 75.3580–80.1600% |
| Augmented | 1250 | 1185 | 94.80% | 93.4972–96.0599% |

均值按问题加权；整体1509个原图来源聚类，2000次bootstrap，seed20261119。human和aug均保留，不能只取较高的aug组。评分使用固定的 `chartqa_pix2struct_relaxed_5pct_v1`，原始解码文本不额外清理，具体零值/百分号规则见既定metric合同。

完整银行由**新增2000条真实推理＋复用500条既有同合同P2推理**组成，derived view不是新的一次2500条GPU运行。两个原始run的三文件、独立取回SHA、原producer源码/inventory/launch凭据，以及逐条source line、input/execution fingerprint和fresh/reused角色全部归档；GT仅进入离线评分。完整参考答案、官方test的human1250/aug1250与输入清单均锁定。

模型revision为 `259a3b64a14623c0ec91a045cb43f7c5af5fa6af`；相同原始图像、原生processor与chat模板，greedy、max_new_tokens=64，suffix为单词/短语回答提示。后续压缩方法只有在模型/图像密度/提示/生成/边界等合同完全相同时才能逐题复用该参考。原始instrumented timing随各producer行保留，不能把复用500条解释为本次吞吐，也不能代替受控速度实验。

已闭合的基础包括三模型30/30原生技术门、四个官方split共15650题的来源锁、已公开P1和当前单目标完整银行。g32/R32源示例默认点的CUDA门及M1probe已过；低比特OnlyTerp-MSE的64题结果是独立source变体测量。较高位MSE、mixed-prod、其它模型/压缩方法及剩余完整split仍分别按其实际证据标记pending或ongoing，不能把计划配置数量当作已运行结果。

本次公开副本只含代码/配置、可浏览表、原始派生证据与完整source闭包；原图、模型权重、张量dump、机器连接信息及认证信息不上传。完成仅指这个明确的单目标参考银行。
