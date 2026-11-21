# OnlyTerp Hmix23 MSE/QJL0:12 fixed500 cells completed

本阶段完成既有预声明的三模型×Text/Info/Doc/Chart四域Hmix23，共12格、6000条视图问答。固定500为各官方数据集的预定子集；不代表full split、P2-114或整个外部库完成。本次CPU核验与打包不新增GPU推理。

身份为OnlyTerp第三方源码Algorithm1-MSE/QJL0 practical Hadamard变体，regular2/outlier3、mixed_precision=true、query_unknown视觉前缀压缩及torch_materialize_fp32_reference reader。原生checkpoint/processor/输入视觉密度保持模型内一致，压缩后重算完整问题suffix，text KV保持FP16。此结果不混Hmix12、旧QJL/prod诊断，不冒称Google原论文完整prod、精确3.5-bit表或优化kernel速度。

旧phase validator及SourceRun闭包核验19个原始donor与12个derived view；冻结物理helper独立逐6000行检查形状、storage容量/唯一charge、CPU/GPU编码常数、位置/映射/范数/padding及native几何分母。旧score_matrix复用精确原score cache产出全部质量、CI、Chart分组及完整bytes。

| model | dataset | QA | quality % [source-cluster CI95] | complete owner mean bytes | actual bits mean [min,max] |
| --- | --- | ---: | --- | ---: | --- |
| M1 | textvqa | 500 | 74.880 [71.459, 78.160] | 18039487.104 | 3.00739 [3.00708, 3.01056] |
| M1 | infovqa | 500 | 71.377 [67.823, 75.048] | 17173912.192 | 3.00779 [3.00650, 3.02276] |
| M1 | docvqa | 500 | 87.661 [85.165, 90.162] | 19279960.704 | 3.00711 [3.00650, 3.01056] |
| M1 | chartqa | 500 | 84.400 [81.200, 87.600] | 18419899.008 | 3.00772 [3.00650, 3.02276] |
| M2 | textvqa | 500 | 60.020 [55.860, 64.180] | 10372248.384 | 3.00943 [3.00818, 3.01598] |
| M2 | infovqa | 500 | 52.151 [48.032, 56.084] | 12350329.984 | 3.00885 [3.00812, 3.02182] |
| M2 | docvqa | 500 | 60.649 [56.794, 64.641] | 13277476.320 | 3.00831 [3.00812, 3.01888] |
| M2 | chartqa | 500 | 43.000 [38.600, 47.400] | 6471734.208 | 3.01295 [3.00814, 3.02248] |
| M3 | textvqa | 500 | 80.060 [76.739, 83.280] | 21672934.304 | 3.00816 [3.00208, 3.01626] |
| M3 | infovqa | 500 | 79.959 [76.459, 83.370] | 92508764.096 | 3.00512 [3.00202, 3.02408] |
| M3 | docvqa | 500 | 95.099 [93.488, 96.594] | 99441528.480 | 3.00341 [3.00202, 3.01934] |
| M3 | chartqa | 500 | 86.800 [84.000, 89.800] | 12382588.512 | 3.01443 [3.00512, 3.06630] |

TextVQA采用十票soft评分，Info/Doc采用原冻结vendor ANLS合同，Chart采用relaxed accuracy；原官方服务器身份边界保留。所有模型/域分别报告，不混合质量平均。每格500 source clusters；CI保留原source-cluster重采样协议。

完整实测位宽跨本组全部请求为3.00201983–3.06629774 bits；6000/6000均超过2.25。因此此Hmix23实测点不具备1.3/1.8/2.25严格同预算资格。所有问题及真实低分均保留，不以超预算或输出长度筛题。

| model | ChartQA group | QA | quality % [source-cluster CI95] |
| --- | --- | ---: | --- |
| M1 | human | 250 | 76.000 [70.800, 81.200] |
| M1 | aug | 250 | 92.800 [89.590, 95.600] |
| M2 | human | 250 | 43.200 [37.200, 49.200] |
| M2 | aug | 250 | 42.800 [36.800, 48.800] |
| M3 | human | 250 | 80.400 [75.600, 85.200] |
| M3 | aug | 250 | 93.200 [90.000, 96.000] |

默认不输出相对F数值，未含对应native500原source闭包的pair不承担相对质量声明。不能用其它模型、128题或不同full cohort补值。表格中的quality wall/TTFT/encode字段为仪器化原测量，不作为受控speed或原论文优化实现速度；没有TrajKV评测或领先结论。

当前公开源码封套包含evaluate dbb556、prediction_cache e8ada8及reviewed policy v3；新Calib FP32AffineText模块仅为当前注册源码，不是本阶段Calib实验结果。所有OnlyTerp预测仍绑定其各自真实执行snapshot，原source_evaluate、source inventory、launch/retrieval proofs及原始行指纹完整保留。v1/v2 policy原件继续保留，v3仅按复核规则维持非Calib旧语义身份。

原始donor始终来自保留的E/results；未使用或重检已回收archive-only records的旧public tree。新归档保存本阶段完整原预测/分数/refs/source闭包与public派生逐行映射，保留source原SHA、public/ZIP SHA和Git-LF身份。排除原图、权重、PT dump、私聊与连接认证信息，最终由Root固定commit/tree和下载assets核验。主114、新Calib24、M1压缩full16及Qwen400要求保持独立、尚不由本组完成。
