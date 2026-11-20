# P2外部默认设置与容量补点

冻结表为`configs/matrix/P2_budget_extension_targets_v1.json`。它只声明待执行目标，不表示新增格已经运行或获准进入强基线主表。先补M1 Text/Info各500源的必要9档，共18格；M2/M3仅各补Text的K2/R128一格。初始上限20格、10,000问；Doc/Chart不自动乘九档，避免重新膨胀成114格。

|方法|既有锚|新增必要档|来源与解释|
|---|---|---|---|
|KIVI-style|K2/g128/R0|K2/R128、K4/R0、K4/R128|两种合法bitwidth及源近窗策略；R0保留低成本锚；是视觉prefill端口，非全系统rolling KIVI|
|CalibQuant|C1、clip K=.5/V=.3|C2、clip K=V=1|源bitwidth改变时clip也改变；保留原校准map，bias SUM/DIV的FP32精度移植单独标识|
|PyramidKV视觉端口|W64、drop=.888|W64/.91875、W8/.91875、W8/.888|两个window与两个容量参数构成最小2×2；其中W64/.888已有，不重复新增|
|TurboQuant第三方|T0 K3/V2/R0诊断|K3/V2/R128及K3/V4/R128|恢复选中源码prefill近窗；仍不是原论文混合outlier/nonoutlier位宽配置；须新CUDA/整模型诊断审查|

PyramidKV [论文v2 §5.4](https://arxiv.org/html/2406.02069v2)使用`beta=20`和`alpha=8`。选中[NVIDIA实现](https://github.com/NVIDIA/kvpress/blob/6cf5e324be90ea9cedce66e843699b6dc655c0bd/kvpress/presses/pyramidkv_press.py)默认window64；两者应分别标“论文实验设置”和“选定实现默认”。保留window64不等于完整覆盖论文设置，新增window8也不改原选择器。

`.91875=1−1.3/16`、`.888=1−1.792/16`仅把**FP16 payload容量**换成合法drop参数。后者沿用既有参数，使window对比严格保持同一drop；不能把它写成实际完整1.8bit。源层间预算、保留窗、精确text预算扣除、索引、整数rounding共同决定最终bytes；逐问记录实测有效bit，并同时报告均值、ratio-of-sums、范围和超过目标的题数。少量问题因保护窗超过目标时，整cohort质量仍保留，该配置只是不具备逐问严格iso-budget资格。禁止去掉这些题后另报更好看的平均质量。

E继续采用query-unknown图像前缀中的原生post-RoPE Q，不读取未来问题。短前缀若触发源代码SnapKV fallback，逐格和逐问注明，不能称这些格实际执行了金字塔层预算。不同window可能改变是否fallback，这是源逻辑的一部分。

C2在单题→固定128源→500源路径上，按有限执行、同源operator/数值合同和owner核查解锁；不能以C1低分断言C2也弱。T的新近窗端口先比较packed与明确dense参考、原生多层q/K/V和完整自由生成；旧T0近0不能充当强SOTA证据。全部配置来自源码默认、论文预设或容量代数，不按已见的P1质量寻优clip、seed或beta。

所有新增问答仍执行完整suffix及各自首token。只在输入、模型、processor、prompt、实际数值后端合同完全一致时复用既有预测；受控速度另行交错测量，不把质量试验的prototype wall当成速度排名。本阶段不混入TrajKV本方算法改动。

## 已观察形状上的容量核查

`ops/forecast_eviction_budgets.py`将完整FP16 retained KV、每head位置、共享视觉位置和Qwen原生delta逐项记账，对P1四组共512条既有W64/.888的实际owner bytes完全匹配。其余window/drop点仅作**已有原生Nv形状上的解析预测**，不产生新GPU或质量结果。原W64/.888在这512条都触发源SnapKV fallback；M1换成W8/.888时，Text有104/128、Info有88/128条将使用真正的pyramid层分配。因此两种window在本输入长度下还涉及源分支切换，不能将旧W64分数直接视为论文W8设置的结果。

M1现有128源形状上，W8/.91875预测完整有效bit均值为Text1.2857、Info1.2830，所有题低于1.3；W8/.888分别为1.7867、1.7835，所有题低于1.8。该预测不保证P2/P3所有新形状都满足同一上限，正式资格仍逐问检查实际bytes。完整范围和fallback数量见`results/analysis/E_budget_forecast_P1_shapes_v1/`。预声明超参没有根据这些数或已测质量修改。
