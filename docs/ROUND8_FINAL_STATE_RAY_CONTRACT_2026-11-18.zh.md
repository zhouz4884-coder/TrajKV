# 最终残差合成态的径向目标对照

新增 `round8_final_ray.py`、`round8_final_ray_runner.py`、`round8_final_ray_check.py`，不修改任何既有源码、不启动 GPU/SSH。这个对照固定原 J、sign 和舍入前 g:alpha 比例，只共同缩放两个已有物理字段，以区分长度调整与方向重编码。

先从原 given-J norm→FP16→frozen factor→FP16 gain 及实际 base 的 signMAD，计算原实际最终 y0=Q16(Q16(g*tJ)+alpha*s)。helper 按给定 J 重建并核验 gain/sign/alpha，不重新 argmin。rays 的三个提议系数均由 y0 计算：mse 的 c=(z·y0)/||y0||²；norm 的 c=||z||/||y0||；projection 的 c=||z||²/(z·y0)。这几个式子分别是理想向量 ray c*y0 的目标点，不是物理 FP16 字段的最优性证书。

实际候选为 g'=Q16(c*g)、alpha'=Q16(c*alpha)，随后按原两级 half 解码。三模式只接受各自目标严格改善：mse 看 row SSE；norm 看绝对范数误差 | ||y||-||z|| |；projection 看自身 source projection 误差 |z·y-||z||²|。norm/projection 不添加 SSE 门，任何模式都不添加 cosine 门；旧 tuple 始终先列并赢得 ties。source/原完整态为零、dot<=0、无效 c、溢出或实际候选全零时保留原字段。统一非正 dot fallback 是明确的保守政策，norm 公式本身不需要 dot 正，负 dot 的非负 MSE ray 最优也可能为零，因此不称此策略全局最优。

共同缩放只保持舍入前系数比例。独立 FP16 字段舍入与 base/residual 舍入会改变最终方向，尤其近相消时没有统一小角度界。stats 实测原到新 final direction cosine drift。projection 仅约束给定 z 的自身投影，不能叫作任意 q 无偏。它操作完整残差合成态且保持原 sign，与旧 R7 base projection 后再编码 residual 是不同干预。

runner 支持任意 `final_ray_mode=off|mse|norm|projection` 子集和 Full/旧 R7。R8 arm 必须是 old global16×4、L2/norm/reencode、binary/all V、joint_gain_refit=false、value_joint_refit=false、postres_direction=false，保持完整224 ranking 与 limit43，同一候选文件内 book/factors/mask/budget/cap 相同。继承原 `_encode_axis` 后，仅对已选一级 K head 调用 helper，明确传原 per-head frozen factor。只 copy g/alpha 回原槽，J/sign/mean/V 不变，不再 fold、不再生 residual；同步更新同一 K distortion 记录。

三个模式都不增加持久 tensor、码本或 cache 字段；临时 native/FP64 arrays 与额外编码计算仍需真实计费和测量。N1792 的原完整 owner 仍为 8,315,013B，448B ranking、43 stages。元数据把 norm 标为初始化、把最终系数标为 complete-state common ray，并记录模式自己的目标和可能的 SSE 上升。

CLI 为 `round8_final_ray_runner.py --config ... --output ... [--plan-only]`，可显式声明 backend_family=final_ray。最后完成门槛 `complete_final_ray.json` 绑定 complete_round8、manifest、records、config、runner/source；不要求四 arm 齐全，不声明完整四格。新诊断可用 `backend_registration('final_ray')`，context 内设真实 cls.settings/ProgressiveBank，退出还原注册指针。

CPU 回执 `outputs/round8_final_ray_cpu_v1/check.json`：合法原初始化 z=(1,0)×64、全1模板、F32 factor=.17679558694，得到 g=.125、alpha=.5、SSE18。mse c=1.17647 后 SSE16.95221，source cosine 可略降；norm c=1.37199 后 norm error2.16905→.000141，但 SSE18.25226；projection c=1.6 后自身 projection error24→0，但 SSE23.01001。后两者确实接受，证明没有隐藏 SSE 门。这是算术 witness，不是模型质量证据。另有18退化策略控制、非零单系数状态、norm目标反控、31原 tuple 随机 rows、四模式实际 class stub、字段 bytes/J/sign/V 不变、异常恢复和最后 marker 哈希检查通过。
