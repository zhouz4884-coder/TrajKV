# 固定最终 norm 的 K post-residual direction 原型

本轮只新增 `round8_postres_direction.py` 与 CPU checker，不接 bank/runner，不修改任何既有实现，也不运行 GPU/SSH。目的是把已有字段的方向搜索与普通 NNLS 的长度收缩分开，不能从此 CPU 原型推断模型任务收益。

单个已选一级 K head 的源为实际 centered-WHT FP32 z[N,128]，书为旧全局 FP16 G4_16。原字段是 packed J[N,16]、FP16 g[N,1]、I32 sign[N,4]、FP16 a[N]。先用原 encoder 及显式 frozen factor 重算 l2/norm→FP16→factor→FP16，再重算实际 base 的 signMAD，逐位核对全部初始字段；不接受预先普通 NNLS 后的字段。factor 必须显式传入，None 明确表示无 factor。r0 取自此原 tuple 的实际最终解码 y0=Q16(Q16(g*tJ)+a*s)，在所有后续步骤中保持同一个锚，不按当前 norm 滚动更新。

固定 J/s 时，记 T=t·t、B=t·s、D=s·s=128，u=z·t、v=z·s。实数 NNLS interior 候选为 g=(D*u-B*v)/(T*D-B²)、a=(T*v-B*u)/(T*D-B²)，仅在 det>0 且两系数非负时使用；另提议两条非负坐标轴。零方向不产生新候选。这些方向由本模块直接在实数中产生，未调用原 hard-SSE refit 来先缩长度。

对每个非零方向 p=g*t+a*s，令 lambda=r0/||p||，随后分别把 lambda*g、lambda*a 舍入为 FP16，再按原 reader 的 nested-half 次序重建完整 128D y。旧物理 tuple 始终保留。每一步只有实际 cosine 严格改善、实际 ||y|| 对原 r0 的相对偏差不超过 2^-9、对给定 z 的 row SSE 不升时才接受。零源或零目标 norm 为 no-op；无法在 FP16 中满足 norm 容差的候选被拒绝，不强迫归一化旧字段。

之后仅做一轮 32 slot 搜索。固定当前物理 g/a，比的是完整实数 d=g*t+a*s 的 signed cosine proposal，不能把 hard y 的 A/U 代入实数更新公式。替换一个 4D slot 为 c，delta=c-t_slot，使用 A'=A+g*(z_slot·delta)，U'=U+2*g*(d_slot·delta)+g²||delta||²，包含残差交叉项。枚举 16 词，保留当前词的严格 tie 优先，只提议 real cosine 最好且严格改善的一个词。选中后重算完整方向、按原 r0 共同缩放 g/a，再重解全部 128D 实际 y 做同样三重 guard。共同缩放会改变 slot 外的值，不能只用局部 4D hard 误差验收。

此搜索是 single-real-best-proposal heuristic。CPU checker 独立枚举完整 128D real 条件候选，验证增量公式和算法输出；还另枚举同一条件下 15 个改词的实际 FP16 可行候选作为更强诊断 oracle。后者在合成控制中确有 1 个 slot 找到本算法未选的更好候选，因此没有实际 FP16 条件最优、局部或全局最优声明。Cosine PTQ 坐标搜索可借已有 Beacon 工作作为组合基底；本原型检验冻结 KV 字段下的最终重建方向与长度约束。

接口：`recode_key_postres_direction(source, book, initial_code, initial_gain, sign_code, initial_alpha, gain_factor=..., row_chunk=256)`，返回同格式 packed J、g、a 和标量 stats。源码不会更改任何输入；sign 固定且不重编码。J/g/a 仍为每 row 20B，原 16B sign pool、书、mean、lookup、ranking、positions 不变，0 新持久 tensor 字段。重算初始 encoder、FP64 reduction 和临时 scratch 都是额外 encoder 成本，不能称免费或已有速度收益。

未来最小集成需另建 bank 子类，显式要求 key_assignment=l2、joint_gain_refit=False、原 G4_16、同一完整 ranking/43 mask。先调用原 `_encode_axis` 生成一级 K signMAD，再仅在该 head 恰有一级 K stage 时调用 helper，传实际 frozen head factor，把返回 J/g/a copy 回原槽。不要再调用普通 K NNLS、不要再生 sign，也不改变 mean 或 V。slot 用 `selected.index(('k',head,0))`，初始化阶段不能依赖尚未赋值的 state lookup。更新同一 K distortion entry：helper 已返回 base SSE 前后和最终 SSE，保证仅覆盖最终重建；base SSE 本身不受单调约束。

CPU v3 回执 `outputs/round8_postres_direction_cpu_v3/check.json`：6 个完整 dense 协议对照逐位一致，240 个全词 real 条件分数匹配，原始初始化 3 个负控、zero/tie 原位、norm 不可行、FP16 overflow、cos 上升且 norm 可行但 SSE 上升的拒绝 witness、chunk1/17 字段一致与同 bytes 通过。6 个合成 row 中 5 个改变、1 个 no-op；最大实际 norm 相对偏差约 3.30e-4，小于 2^-9。这些不是模型数据或质量结论。统计里的 source SSE 是调用者所给 centered-WHT 坐标的误差，不自动等于含 FP16 mean/native bridge 的原 K 误差，也不保证 attention 或任务质量改善。
