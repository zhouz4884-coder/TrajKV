# 下一阶段候选v3：在真实WHT码器坐标中优化K索引（未执行）

本文件纠正v2的一处重要推论，不增加Round11实验。依据静态审阅 NEXT_IDEAS_V2_STATIC_MATH_REVIEW_v1.json（SHA f622391bff43e3fe7b818997607127aa381ba153772eb9fd490ed521616d8017）。当前方法/候选/确认数据与码流均不改。

## 已核模型前提

当前M1的pinned配置、Qwen2源码及index支持：输入RMSNorm gamma/eps1e-6→有bias的q_proj→RoPE，无q_norm；W_Q应取每head的128×3584行块，增广bias后A为128×3585。RoPE pair为(c,c+64)。dynamic RoPE factor2的初始化函数与attention_scaling尚未核实，不能说是default/幅度必为1。v2的RMSNorm增广输入Cauchy界与2×2块范数row-sum旋转包络在相应实数/单位旋转前提下成立；Sigma_q是E[qq^T]，不是漏掉均值的中心协方差。

## 被否定的便宜推论

R7 K先在centered WHT帧编码，reader也变换query。native post-RoPE度量T必须拉回码器坐标C=U T U^T（正交WHT约定）。原native对角T不能直接作为原G4码器的通道权重。归一化WHT各元素平方都为1/128，因此diag(U T U^T)=trace(T)/128，是常数；仅取变换后对角会退化为欧氏选择。

具体反例：让native T_jj按bit2(j)取1或2，仍保持j与j+64的pair相同权重；变换后C=1.5I−0.5P_xor4。重要性完全以跨G4块的耦合出现。故“保留原WHT/reader、保证native界、廉价逐通道nearest”三者不能直接兼得。v2中这个推论视为已被静态证伪，不作为执行设计。

## 更值得试的实际动作

保持原U、G4_16字典、gain/mean、K23残差覆盖、已验证V底座与reader不变。对全部112个K单元的实际4bit索引做模型度量引导的G4 block coordinate search，而非只调全局64参数码本。目标在实际码器坐标中计算；原R7索引为合法起点，每改一个4维块，16个候选并行评价 deltaL=-2 delta_b^T(Cr)_b+delta_b^T C_bb delta_b，再更新Cr。固定2轮，与同搜索器/同残差重编码规则下的C=I对照；实际Half字段与旧点必须参与验收。

若dense搜索代价不合适，可以比较保守4×4 block-diagonal包络B_bb=C_bb+sum_(a!=b)||C_ba|| I4，它允许逐G4块选择但不是单通道加权，界可能很松。以上均未实现、未跑GPU、未获质量；不把源代码静态推导当作K已经有效。

模型先验可用RMSNorm/W_Q/bias和位置不变包络，也可用合法prefix的非中心query二阶矩。必须明确pre/post-RoPE坐标、动态RoPE幅度与具体native/学生state来源。当前Reader实际WHT/Half的舍入也不能自动继承精确实数界。

## 为什么这是下一步候选

R11的V联合重编码只触及20个有残差的V单元，而K基础索引覆盖112个K单元。增加这个已有码字的选择自由度，在同一字节下可能比继续给V加幅度参数更直接地修复路由；“可能”仍由固定V、真正端到端小集验证。原f0/覆盖/预算与新残差规则的匹配控制必须保留，不用V再校准掩盖新K是否有独立收益。

v2的V正对角加权129区间解仍成立，可作为简化储备。更丰富metadata预测模式仍只是储备。当前首要工作是完成已冻结JointV/JointSpatialWO的R128确认和完整阶段交付，再依据实际结果决定下一阶段，不新增分支。
