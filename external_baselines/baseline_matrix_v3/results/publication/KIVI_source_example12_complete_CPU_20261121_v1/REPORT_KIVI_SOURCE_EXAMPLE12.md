# KIVI source-example:12 fixed500 cells completed

本阶段覆盖原先预声明的M1/M2/M3各四域K2/V2、group32、residual32视觉前缀source-example点，共12格、6000条视图问答。所有预测来自已有真实RTX3090运行；本次CPU核验与打包不新增GPU推理。固定500为官方数据集子集，不是四域全split，也不完成P2-114、M1压缩full16或Qwen400速度补充。

保留原生模型、processor、checkpoint与各自输入视觉密度，压缩视觉KV后重算完整问题suffix，text KV保持FP16，未来问题未知。此点是固定源码示例的视觉prefill移植，不能标作原始rolling whole-LLM系统。较早g128/R0点及历史失败/诊断均保留。

旧phase validator、SourceRun闭包和独立物理helper已逐条核查6000行；15个原始donor与12个derived view保留原始三文件、来源/检索证明、行指纹、raw输出与token IDs、锁定refs及离线分数。物理核查包含各tensor形状/容量、共享storage唯一charge、visual/text分区与native几何分母。旧score_matrix直接复用精确已有score cache产出完整质量与bytes表。

| model | dataset | QA | quality % [cluster CI95] | complete owner mean bytes | actual bits mean [min,max] |
| --- | --- | ---: | --- | ---: | --- |
| M1 | textvqa | 500 | 79.040 [75.560, 82.221] | 19129700.352 | 3.19455 [3.18080, 3.33557] |
| M1 | infovqa | 500 | 73.632 [69.753, 77.165] | 18246762.496 | 3.21252 [3.15501, 3.87723] |
| M1 | docvqa | 500 | 91.038 [88.449, 93.368] | 20395057.152 | 3.18222 [3.15501, 3.33557] |
| M1 | chartqa | 500 | 89.800 [87.000, 92.400] | 19517743.104 | 3.20940 [3.15501, 3.87723] |
| M2 | textvqa | 500 | 80.680 [77.260, 83.880] | 11649155.328 | 3.38901 [3.24041, 3.94312] |
| M2 | infovqa | 500 | 80.838 [77.495, 83.939] | 13667382.048 | 3.34744 [3.22723, 4.47383] |
| M2 | docvqa | 500 | 95.879 [94.254, 97.381] | 14753787.664 | 3.34414 [3.22723, 3.80474] |
| M2 | chartqa | 500 | 87.000 [84.200, 90.000] | 7534909.504 | 3.54901 [3.23140, 4.45597] |
| M3 | textvqa | 500 | 81.440 [78.260, 84.540] | 23773506.288 | 3.32067 [3.04880, 3.68212] |
| M3 | infovqa | 500 | 80.418 [76.964, 83.733] | 96108524.352 | 3.24499 [3.04481, 4.55605] |
| M3 | docvqa | 500 | 96.196 [94.738, 97.521] | 103156386.416 | 3.14154 [3.05133, 4.01697] |
| M3 | chartqa | 500 | 87.400 [84.400, 90.400] | 14951507.408 | 3.77746 [3.20892, 7.24641] |

TextVQA采用十票soft评分，Info/Doc采用原冻结vendor ANLS合同，ChartQA采用relaxed accuracy；原官方服务器identity边界保留。各域/模型独立报告，不混合平均。上述CI按source_cluster_id重采样；本组每格500个source clusters。

所有6000请求的实测complete bits均高于2.25，跨本组范围为3.04481–7.24641；因此此source-example点不属于1.3/1.8/2.25的严格iso-budget对手。名义2-bit只描述payload构造，完整owner保留真实量化metadata、FP16残留和位置等费用。原全cohort质量完整保留，不剔除超预算题。

| model | ChartQA group | QA | quality % [cluster CI95] |
| --- | --- | ---: | --- |
| M1 | human | 250 | 86.400 [82.400, 90.400] |
| M1 | aug | 250 | 93.200 [90.000, 96.000] |
| M2 | human | 250 | 80.800 [76.000, 85.600] |
| M2 | aug | 250 | 93.200 [90.000, 96.000] |
| M3 | human | 250 | 80.400 [75.200, 84.810] |
| M3 | aug | 250 | 94.400 [91.200, 97.200] |

默认候选发布不输出相对F的数值。部分同模型/同域/同500 cohort的native-F配对仍缺；其状态见PAIRED_REFERENCE_SCOPE.json，不使用其它模型、128题或不同full cohort补值。已有pair的数值也留待连同对应native500原始闭包另行发表。当前绝对候选score可独立复核。

完整表tables/matrix.csv另报ratio-of-sums与mean bits、逐QA超预算数、payload/metadata/positions/encoder常数、视觉tokens和实际CUDA峰值。该表中的quality wall/TTFT/encode字段是带审计的仪器化测量，不当作受控speed或原论文优化kernel速度。本阶段不产生TrajKV评测、领先或总体SOTA结论。

公开副本仅包含可浏览代码/配置/refs/表、派生raw证据与完整source闭包；不含原图、模型权重、PT dump、认证和机器连接信息。source原字节、public/ZIP字节和Git-LF正文身份分别记录，最终发布由固定commit/tree、下载asset SHA和历史保留收据绑定。
