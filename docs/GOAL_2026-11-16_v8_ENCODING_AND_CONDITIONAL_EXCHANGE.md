# 阶段目标v8：同格式编码能力与条件预算交换

用户2026-11-16明确授权按chat.md第二十二节完整执行，自主迭代，不在子任务询问；完成后自动同步并核验GitHub。

本轮目标是把新的编码自由度做成实际packed方法，在完整视觉KV资源前沿上同时检验：同字节提高质量、预定质量容忍度内减少字节。ρ、谱、连续损失用于解释和开发，不作为开工或关闭方法线的单标量门槛。

## 研究范围

- 编码：G4_256的norm/LS幅度与实际FP16重构下的块选码；学生query近似度量计入encoder常量；G4_16实际nibble packing、16B/K-token。先隔离固定覆盖，再比较完整≤1.8bit预算重新分配。
- 预算：同一depth bank、全一级、同lookup/ranking/policy的incumbent/spread/random；有限候选下比较独立加减分数、完整成对干预、接受后重估学生状态。最终为冻结静态部署policy。
- 连续诊断：真正packed teacher forcing的GT＋EOS与Full自目标，源/问题/数据集宏平均；stress只作开发工具，不替代自然分布与独立源推断。
- 最终：少量候选冻结、未用来源确认、同输出安静计时、方法及限制报告、GitHub增量交付和完整远端核验。

## 资源与证据

2×RTX3090 24GB，完整InternVL3-8B-hf FP16；两个独立CUDA worker并行，不是CPU模型或离线预测准确率。预检两卡各24MiB/0%，数据盘约6TB空闲。新run统一round7_*与results/round7_v1。

所有新数据、模型访问、结果、TRITON/TMPDIR/Inductor/CUDA与HF缓存使用数据盘。通过launch_remote_data_disk.py建立不可变源码快照；remote.py、旧launcher和冻结参数/结果保持不变。失败保留旧run，修复使用新名称。

诊断为取参考KV而增加FP16 owner时单独标注，不放入低比特部署前沿。校准缓存前缀加速与真正自由答案端到端评分、计时严格区分。

## 数据

旧Doc64/255问及已观察Text64用于开发；新算法拟合采用历史独立Doc cal32中按source固定拆分的16训练/16选择（45/38独立问题runner），明确选择源曾用于旧基码校准，仅对本次新拟合holdout。Text新增训练32源/52问、选择16源/22问，source/content排重。

新的确认预留为Doc128源/412问、Text128源/128问；Doc从官方额外train分片抽取，称project-fresh holdout，不声称基础模型训练绝对未见。当前只固定身份与原始图像，方法冻结前不物化确认答案或按结果选样。旧Doc54/Text256保持已观察身份。

## 完成定义

完整做完实现、适当GPU算术检查、真实自由回答、依据结果的迭代、适当未用源确认、成本、总结及Github同步后结束。完成执行不等于全部性能假设成立；不能用开发最高点、训练损失或数学反例代替确认增益。


## Round7执行结果（2026-11-16）

研发执行已完成：4684次真实自由会话/9752回答；主新源Doc128/412问、Text128/128问的九方法确认；第二骨干64源首问；72次同输出安静trial。校准、性能诊断、失败run单列，数据和新远端缓存使用数据盘，冻结结果保留。

新G4_16真nibble＋43级残差在Doc/Text完整加权1.29466/1.29579bit得到97.680 ANLS/80.938 soft accuracy，较原约1.8bit少28%视觉KV字节，质量差0.464/1.406pp。相对Native4两域0.5pp标尺未达到；静态同批分数可精确复现条件policy，未建立反馈重估独立贡献；第二骨干质量保持与整体提速未建立。

完成研发执行不等于全部性能假设成立。[方法、结果、限制与下一步](RESEARCH_PROGRESS_2026-11-16_round7.zh.md)。自动GitHub同步与远端核验以[Round7 release](https://github.com/zhouz4884-coder/TrajKV/releases/tag/research-round7-20261116)的独立交付receipt为准，根目标在远端核验成功后完成。
