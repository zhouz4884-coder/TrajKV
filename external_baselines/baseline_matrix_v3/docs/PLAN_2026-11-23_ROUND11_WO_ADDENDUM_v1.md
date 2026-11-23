# Round11有界读出几何分支补充稿v1

原执行稿v1及384条主实验不改。本补充稿在R128任何结果产生之前形成，只细化原已授权至多两版W_O读出度量候选，仍最多192条新开发回答；是否追加GPU自由回答待完整D96判断。当前准备依据：Chart四候选打平；Doc小集Spatial一处ANLS改善，JointV/JointSpatial未保留该改善；Text已完成JointV/JointSpatial总分相同，较R7改善1题。它们不是最终统计结论，也不用于重抽来源。

## 模型度量与部署资源

对第l层第g个KV head，原生GQA将其value复用到7个query heads。取原W_O相应连续128输入列块W_j，构造H_lg=sum_j W_j^T W_j。原M1的H为128x128，Gram源有7*3584=25088行。H是各head独立读出误差之和的模型几何，也是逐token跨head Cauchy界的几何部分；不是包含真实attention权重/跨token交叉项的完整attention输出损失，更不是任务Hessian。

只给原覆盖V20逐unit临时生成H，从已有完整FP16模型借用W_O；构造/转换/编码临时峰值与耗时单记。H及任何W拷贝不挂owner/shared/global，不留在serving cache。模型不训练、不改weight；无需新问题、答案或校准数据。最终仍完整B0(N)=4590N+89733，与原R7逐ID精确相等，K23、V端点/原f0、覆盖槽及级数不改，读取仍用已验reader1。

## 两个候选

SpatialWO保持原R7 base。固定P0原base与P1已冻结邻域base预测器，实际Half重建q=Half(p+alpha*s)，按L=(v-q)^T H(v-q)选择sign/幅度与mode；保原合法点，严格改进才替换，tie旧字段。

JointSpatialWO不再无条件先写MSE JointV。先将原R7完整点与MSE JointV提案在实际full-H目标下逐token严格比较。随后固定两次cyclic sweep，对每channel的四个Half电平(l0+-alpha,l1+-alpha)并行枚举，用精确单坐标差更新basebit和sign；每个sweep后提出full-H条件幅度候选。完成所有token后冻结唯一最终B*，再执行与SpatialWO相同的空间步骤。MSE提案、joint重选和spatial改进的费用及指标分开记录。该候选同时改变码字的离散选择与残差，不增加基础码本或索引位宽。

## 数学与保证范围

固定p与s，忽略最后Half舍入时，alpha*=max(0,s^T H(v-p)/(s^T Hs))；分母为0则保留原候选，不以epsilon冒充同一个目标。实际只把此值及其Half舍入/相邻值作提案，仍比较实际Half解码，并保留旧alpha。

令r=v-q。只改第c维为q'_c，delta=q'_c-q_c，则精确二次差为 -2*delta*(Hr)_c+delta^2*H_cc。接受后Hr更新为Hr-delta*H[:,c]；跨token向量化，避免逐channel CPU同步。完整目标在阶段结束再算并严格验收/回滚，防浮点增量漂移。JointSpatialWO的joint与spatial各做固定2个sweeps，不按任务结果增加训练步数。

原R7点在初始full-H比较中保留；每一步的实际full-H验收给出局部目标相对原R7不增。最终不需要保存第二份旧base。它不是联合离散问题的全局最优，也不保证无权MSE、KL或答案不下降。H的trace/对角范围/非对角Frobenius占比/对称与finite检查单列，以检验这种模型几何是否接近各向同性。

## GPU与对照

先完成源码/CPU小维精确差值与字段控制；若完整D96支持追加，则新run先做真实8B的原R7+2no-op+2encoded共5次TF source评估（QA0），再在同一D96上两卡各96自由回答。参照仍为原同题FP16/R7低/高，不重跑开发参照，不另开家族。主实验与WO组的同一288条参照是重复视图，不能计576条独立来源或新推理。

候选选定后才启动冻结R128最多两版加3个新参照；不从R128选参数/版本。若没有可复核的任务及成本收益，明确NOT_RUN或负结果，不把局部full-H改进包装成方法胜出。原384主实验/source/配置/结果及最初执行稿保留。
