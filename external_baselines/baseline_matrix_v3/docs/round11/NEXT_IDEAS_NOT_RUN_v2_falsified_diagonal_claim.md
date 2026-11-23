# 后续候选v2：算子度量、RMSNorm与RoPE；本轮不新增实验

这是数学设计备忘，不是已验证方法或新执行合同。Round11仍只完成既定WO192与最多两版R128，然后报告和GitHub交付。

## 更有潜力的K改进

V的联合编码只改原覆盖20/112个V单元；K1基础G4_16索引存在于全部112个K单元。若保持原4维16词码本、gain/mean、残差槽和packed reader，只改变实际写入的4bit码字索引，再对原K23重新编码残差，就能在同字节下获得比全局64参数码本训练更大的、逐图逐token的路由自由度。V固定为已验证底座，先测K独立增量。

原始logit误差的均方为 e_K^T M_q e_K，其中M_q:=E[q q^T]是非中心二阶矩。若使用Cov(q)，必须另加mean(q)mean(q)^T，不能默认query均值为0。一个无未来问题的先验可以来自W_Q与合法前缀hidden二阶矩；但post-RoPE需要R_t M_q R_t^T，t未知，不能把pre-RoPE度量当作实际最优。

## 一个可推导的鲁棒先验（模型前提需先查代码）

假设实际LM在Q投影前使用RMSNorm，理想实数模型中其未乘learned gain的u满足||u||^2<=d_hidden；设Gamma为gain，q=R_t(W_Q Gamma u+b_Q)。若无后续Q归一化等非线性，这个表达可用；有QK norm、bias或多维RoPE时必须改推导，不能照套。

无bias时，(q^T e)^2/d_key <= (d_hidden/d_key) e^T R_t S R_t^T e，S=W_Q Gamma^2 W_Q^T。含bias可用A=[W_Q Gamma,b_Q]、u'=[u;1]，则||u'||^2<=d_hidden+1、S=A A^T，得到因子(d_hidden+1)/d_key的相同形式。它是query无关的直接visual logit扰动界，不是答案/整模型误差界；实际Half与Gram舍入另计。

为了避免偷看未来位置，将S按实际RoPE二维旋转pair分块。令c_ab=||S_ab||_2，r_a=||e_a||，则对任意这些pair的旋转：e^T R S R^T e <= sum_ab c_ab r_a r_b <= sum_a(sum_b c_ab)r_a^2。由此得到RoPE不变的对角pair上包络T=blockdiag((sum_b c_ab) I_2)。也可用Frobenius块范数作更松但易算的上界。Qwen式rotate_half的pair不一定相邻维，必须按真实实现配对。

T是对角度量，因此G4_16的最近码字仍可逐4维块并行枚举16项，计算量接近原量化，decoder不变。原码字作为合法候选；重新编码原覆盖K残差后用实际Half解码评价。它不等于最优query分布，但保留模型通道异质性、覆盖所有未来RoPE相位，给出可证伪且无未来数据的设计。最便宜的两臂是原R7欧氏选码与这个算子pair度量选码，同一V底座、相同覆盖/位宽/更新预算；先看真实答案，再讨论文献定位与更精确query矩。

## V侧可保留的简化路线

若全H搜索无额外任务收益，试diag(H_V)的联合编码比继续增加sweeps更清晰。正通道权重不改变128个(u+v)/2切换阈值，每区间的最优alpha由加权距离均值给出；前缀和扫描仍至多129区间，实数内层全局解、Half候选实际验收。它可能保留模型感知而避免昂贵逐坐标搜索/新reader，但当前未实现或跑分。若要主张WO目标的独立作用，需要同搜索器H=I的匹配消融，不能仅凭更多计算下更好就归因模型几何。

## 性能与下一步约束

当前原型的编码延迟明显，若质量胜出，优先把逐token/逐channelTorch循环批量化或融合GPU kernel；H本身可从原模型即时生成并释放，不必增加常驻资产。本文所有性能改善都是待测目标，不能由上界或低MSE推出ANLS。未选参数、未开新GPU任务、未读R128结果。
