# 另一个待测目标：保视觉前缀作为attention算子，而非只保native V

未执行、不是当前方法结论或新实验授权。写于R128已冻结之后；JointV在首两个确认域出现退化，只能说明开发集的少量收益暂未复现，不能单凭这些结果证明具体机制。

## 精确的局部组合关系

固定层、head、query和共同text状态，定义视觉部分M(q)=sum_i exp(q^T k_i/sqrt(d))，N(q)=sum_i exp(q^T k_i/sqrt(d))*v_i。加入任意text部分M_t,N_t后的attention为(N+N_t)/(M+M_t)。因此只匹配视觉内归一化输出N/M不足以保证未知text suffix下的混合：还须保留M。稳定实现可用logM与o_vis=N/M两组量。

R11的W_O度量控制native V重构的一个直接分量，不控制compressed K的路由和log-partition误差。若原R7的V/f0已部分补偿K误差，native V更近仍可能使任务变差。这是待验证解释，必须用同学生query/history下的替换/恢复对照确认，不能把它写成现有因果结论。

## 可以利用已有字段的动作

1. 视觉mass：若现有K mean槽是在解码时对所有视觉key加同一个向量，增加delta_mu会给全部视觉logit加q^T delta_mu/sqrt(d)。它精确保持visual内相对权重，却改变visual/text质量。对一组合法的、非未来实际问题的query先验Q，拟合logM_native-logM_compressed到Q delta_mu/sqrt(d)（ridge），可能用现有128个Half mean槽修复mass而不加字节。WHT码器必须在正确query/mean坐标中做，原codes/gain不因新均值重新center，否则会把补偿抵消；实际Half需要验收。该动作尚未核完整reader槽语义，更没跑答案。
2. 条件V目标：在固定compressed K下，拟合A_hat V_tilde约等于A_native V，再把V_tilde约束回真实低比特码流。可以用正规化最小二乘/低秩Woodbury形成编码提案；不保留dense V作为serving后备。合法query先验可以来自模型算子或明确的校准来源，不能偷看未来问题。其GT任务收益未知，需防只在局部queries上误差抵消。
3. K真实索引：另一条是保持原reader/WHT/G4码本，按正确拉回的功能度量重选实际索引，见NEXT_IDEAS_NOT_RUN_v3。这样从源头减路由误差，而不把所有责任压给V。

## 干净的小实验应回答什么

固定一个已验证V底座，先做原K/仅mean-mass/仅索引重选/两者合并的2x2，同实际字节与残差规则。mean-mass与within-visual路由在实数形式上可分，仍要实际GPU测。若再适配V，需要同预算旧K侧V重训对照。若为了诊断使用native K或V oracle，必须单列为超预算oracle，不能进方法竞争表。

查询先验与真实未来分布不同是主要风险；局部算子误差也不保证答案。当前R128已用于确认/观察，若从其错误中改方法，下次须另冻结新确认来源。该备忘不触发任何新GPU、不改变当前1216目标或已冻结候选。
