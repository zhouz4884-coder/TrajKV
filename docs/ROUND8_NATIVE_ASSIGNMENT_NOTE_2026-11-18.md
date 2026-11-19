# R8 原坐标目标与 assignment 的边界

当前 K 重编码有意保留旧 baseline 的 assignment 语义；这不是漏乘 head factor 的实现 bug。但该 assignment 不是拟合器原坐标 hard loss 的精确 E-step，三档提议全拒不能解释为码本或编码能力已达最优。

令 `x` 为实际减去存储 FP16 mean 后的 128 维 WHT 源向量，`t(J)` 为 FP16 码本按索引拼出的完整模板，`f>0` 为冻结 head factor，`Q16` 表示 FP16 舍入。当前路径为

\[
a=\|x\|/\sqrt{128},\qquad
J_b=\arg\min_j\|x_b/a-c_j\|^2,
\]
\[
\hat g(J)=Q_{16}\!\left(f\,Q_{16}(\|x\|/\|t(J)\|)\right),\qquad
\hat x_0=Q_{16}(\hat g(J)t(J)).
\]

所以初始 assignment 确实不使用 `f` 或最终 gain；随后 `centroid_proposal` 使用的 `fields.gain` 已包含 factor，原坐标加权正规方程没有忘记它。

在**无舍入、无残差的单向量 norm-match 模型**中，设 `Z=||x||²>0`、`A(J)=||t(J)||²>0`、`U(J)=xᵀt(J)`，则

\[
D_{\rm norm}(J;f)
=(1+f^2)Z-2f\sqrt Z\,U(J)/\sqrt{A(J)}
=Z(1+f^2-2f\cos\theta_J).
\]

固定正 `f` 不改变**完整模板的有符号 cosine 排序**。因此，不能仅凭 factor 没进入逐块归一化最近邻就判定 bug。若用 `U²/A` 替代有符号排序，必须处理 `U` 的符号；零源、零模板及无正投影情形不能直接套用 LS 与 norm 的等价结论。FP16 舍入与残差可能使真实候选排序依赖 `f`；不能把理想关系当成实际排序恒等式，也不能说 factor 对幅度或模型行为无关。

若把**实际最终 gain 固定**为 `ḡ>0`，理想连续条件子问题才是

\[
\arg\min_j\|x_b-\bar g c_j\|^2
=\arg\min_j\|x_b/\bar g-c_j\|^2.
\]

真实实现应枚举 `||x_b-Q16(ḡc_j)||²`；若 sign/alpha 固定，还要完整执行各级 FP16 加法。`ḡ=0` 时直接处理解码后的并列，不能除零。**一旦更换索引后重新 norm-match，gain 又依赖整个 128 维模板，逐块 `x/ḡ` 的条件保证便不能继续沿用。**

当前拟合先按固定字段生成 centroid LS 提议，随后在 reencode 模式重新计算 assignment、norm/factor gain 和所选 sign/MAD 残差，再以完整 hard loss 接受或拒绝。它保证接受端点的训练目标不升；不保证每次提议有下降方向，也不是同一目标下的精确交替最小化。decoder-only 移动模板却保留旧 gain，可以改变实际重构半径；reencode 强制重置 norm/factor gain，并不包含 decoder-only 的全部可达表示。冻结 factor 来自功能校准，其原坐标 MSE 最优性也未被假定。

下一步最便宜的独立对照应保持现有运行不变，待本阶段完整结束后，在**一份已完成的 reencode 端点或冻结 Gaussian16 书，以及少量预登记 native 行**上比较：

1. 保留 legacy assignment 作为原点，不重新拟合码本。
2. 每次替换一个块时，用完整模板更新 `A`、`U`，而不是只看局部归一化距离：`A′=A−||c_old||²+||c_new||²`、`U′=U−x_bᵀc_old+x_bᵀc_new`。K4_16 可枚举每块16个候选。
3. 以完整 128 维模板的有符号 cosine作理想参考；实际判定重新计算规定的 norm/factor 双舍入 gain，并执行 `Q16(g*t)`。有残差的行再按原策略重建同一 mask 下的 sign/alpha，逐级 FP16 解码计分；base-only 与完整表示分列。
4. 保留原索引候选，只接受声明目标下的严格改善。一次坐标扫描只是局部候选搜索，不称全局最优。固定最终 gain 的 hard 枚举可另作诊断，不能悄悄替换 norm-mode 契约。

这样检验的是“旧 assignment 启发式留下多少可达改进”，而不是盲目把 `f` 塞入最近邻。索引与已有字段宽度不变，但编码计算仍需计时；原坐标误差改善也不能代替固定 teacher-history 诊断和自然自由回答证据。当前结果不支持关闭该方向或宣布瓶颈已解决。

本备忘录仅作源码与数学审查，未修改实现、运行模型/拟合或控制任务。对应源码：`round8_codec.py` SHA256 `b30dd4b13745d45b46436b470110ccd5174cf4a26eb88ee9dd00d02cbace5e64`；`round8_calibration.py` SHA256 `5a8f1115551aec9d6f22f5b8d155caae8384317f2dfcba8cc6bc549d358a0469`。
