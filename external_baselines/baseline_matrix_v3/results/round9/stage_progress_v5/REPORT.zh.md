# Round9 阶段进展 v5：14个已闭端到端source runs

共576个真实回答事件：FP16/R7参考256，K-MSE/K-Func首32筛选128，MSE后32开发复核64，Mu/FZJ诊断128。独立开发问题仍为Info64＋Text64；不是576个独立来源，也不是576条新候选质量验证。Info为真实TRAIN，Text为validation，不能拿其高分直接外推官方测试泛化。

全部14格为已闭完整source slice，原raw3/校正后producer source proof/scorer/metric/逐行SHA均核对。此报告CPU生成，没有新GPU、远端、Git或旧证据改写。当前新保留确认32＋32未运行、未被本表使用。

| 处理 | Text首32 Δpp | Info首32 Δpp | 解释 |
|---|---:|---:|---|
| K-Func，实际functional书 | +0.0000 | +0.6250 | 同容量64标量；首32开发筛选 |
| K-MSE，实际mse书＋K硬RE | +5.0000 | +0.6250 | 首32信号未直接当成功确认 |
| K-Mu | +0.0000 | +0.0000 | diagnostic_only，不进方法前沿 |
| MSE书的generic K-Book_FZJ | +0.0000 | +3.1250 | 来源是MSE书，不能因旧Func命名误写functional |

Δ均对相同实际ID的R7原低点，按原连续任务分数；改善/退化不是二元语义审判。细表含每run质量、exact-ID参考、修复/新增得分损失、实际mean/max完整bit与source driver SHA。

## 后32与真实union64

冻结MSE书后在未用于首轮选择的后32上，Text +0.00000pp，Info -4.07986pp。后32现在也已被观察，只是开发复核，不能为下一种交换candidate重新命名为盲确认。全局64码本重拟合路线据此冻结，不再继续扫描它。

| 域 | 真来源runs | QA | union相对R7 Δpp |
|---|---:|---:|---:|
| textvqa | 2×32 | 64 | +2.50000 |
| infovqa | 2×32 | 64 | -1.72743 |

union64由两次真实32原件组成，不改原manifest、不生成假单次fresh64，也不将两域不同指标平均。配对CI保留原尺度和未校正限制。

## 当前主动作：Base–Residual Precision Exchange

新动作保持原V20与因子，整层4个KV-head升为原G4_256，卖掉4个原K stage0残差，全部lookup重打包为signed int8；完整物理delta为744−8Nv。Lookup8、Drop4与Exchange是用于分清索引重排、丢残差和加base精度的控制。它不是继续训练全局64词典。

calibrator v1在旧cal32固定q的KL计算中遇到技术数值失败。Root的最小实际GPU/CPU对照在Torch2.4.1+cu121／RTX3090、宽度797观察到标准F64 log_softmax与CPU最大差29.1883、exp行和最低约0.000281；手工F64路径对CPU约7e−15。该事实属于特定环境/输入的数值问题，不能推出所有F64或所有宽度797都错误。它不是BRPE任务分变差，也没有产生可采用profile或端到端BRPE答案。

Root正在修复calibrator v2并安排真实profile选择/整4删重放，然后才能执行Drop4／Exchange各两域64的真实自由回答。此报告不预测其涨分、不把CPUfake owner或局部KL当任务结果。

## 交付边界

R7 anchor8已独立完成公开发布（commit63281…），与本R9开发阶段分开。R9新publisher/source扩展还需按实际diagdriver v3和未来BRPE driver v4另版本化，不能让现有只知435b/cebb的v2直接接收新来源。本文及CSV只汇总已有真实结果，未发布R9，也未改变保留确认池或当前cal/dev。
