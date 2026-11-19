# Postres 独立集成契约

新入口 `round8_postres_runner.py` 复用冻结 `Round8Bank`；数值 helper 使用新增 `round8_postres_direction_v2.py`。旧 v1 helper ecef512a... 及所有旧 runner/bank/codec 保持原字节。v2 只调整初始化校验：给定原 J，重建原 norm→FP16→实际 frozen factor→FP16，再验证原实际 base 的 signMAD；不重新 argmin，避免 whole-layer 与 head/chunk 形状改变使边界 J 被误拒。J 的 L2 来源由冻结 bank/config/source 负责。原 L2 输入下 v1/v2 搜索输出已经 CPU 逐位对照。

任意单/多候选及 Full/旧 R7 controls 都可运行，不要求完整四格。R8 候选要求 `method=round8_joint`、`postres_direction=true`、`key_assignment=l2`、`key_amplitude=norm`、`key_encoding=reencode`、旧全局 16×4 书、`value_encoding=binary`、`v_group_size=all`、`joint_gain_refit=false`、`value_joint_refit=false`，保留完整 224 ranking 和 `res_stages_limit=43`。同一文件中的这些 R8 候选必须共享 book/factors/mask/budget/cap。可显式声明 `backend_family=postres`，其它 family 会被拒绝。

`postres_row_chunk` 默认 4096，可设 8192。它控制暂态内存与 launch/sync 数量，不增加持久 cache 字段；FP64 proposal/guard、计数同步与原系数验证都是额外 encoder 开销，尚无速度测量。

子类 `_encode_axis` 先调用原方法。只有 K 且该 head 恰有一个现存 stage0 时，取 `selected.index(('k',head,0))`，把原 J/g/该 slot sign/alpha 和 `encoder_factors['key_head_factors'][layer,head]` 显式交给 helper；f 不存在时才传 None。普通 joint NNLS 必须关闭，r0 来自原 norm/signMAD 的实际最终解码。返回 J/g/alpha 只 copy 回原槽；sign、mean、V、positions、lookup 和 mask 不变。不再 fold、不重编码 sign。原同 layer/head 的 K distortion entry 被更新为最终 base/final SSE，并保留 pre-postres 值。

元数据把 norm 明确限定为初始化。选中 K 的最终 g/alpha 是保原最终 norm 的物理系数，按 Q16(Q16(g*tJ)+alpha*s) 解码；未选中的 K 保持原 norm/fold。目标与 guards 覆盖最终 centered-WHT reconstruction，不把它称为含 mean/native bridge 的 K SSE，也不保证 attention 或任务改善。原 packed reader 未改。

CLI 为 `python code/a2/round8_postres_runner.py --config ... --output ... [--plan-only]`。最后完成门槛是 `complete_postres.json`，绑定最终 complete_round8、manifest、records、config、runner 和完整 source 指纹；单/多候选均标 `candidate_subset=true`、`factorial_complete=false`。旧 raw records 不重写。

连续诊断/quiet 的新薄 wrapper 可用 `with round8_postres_runner.backend_registration('postres') as cls`。它在加载原 R8 registration 后临时注册真实 postres subclass，finally 还原 factory/executor 指针。直接 session 调用者还要设 `cls.settings=v` 及该诊断 executor 的 `ProgressiveBank=cls`，并把执行/metadata 收集留在 context 内；不修改旧诊断工具。

CPU v2 证据为 `outputs/round8_postres_integration_cpu_v2/check.json`：两 K heads、同 K/V head 的交错 slot、whole-layer 编码切 head、非恒定真实 factor、缺 factor 的显式 None、给定非最近 J 不重新 argmin、拒绝 pre-NNLS/坏 alpha、64 行非 no-op、原物理槽/bytes/sign/V 不变、最终误差更新、chunk1/4096 一致与完整 postcheck 通过。N1792 的单独 owner 代数为 8,315,013B，448B ranking，仍为 43 stages。

GPU 准备入口 `round8_postres_check.py --gpu --output <新目录>` 尚未由本 agent 执行：Nv64/256、q1/11、off/on 共 8 次 actual unchanged-reader 对最终物理字段 dense 对照，另有 Nv1792 whole-layer→head 初始化与 guards、Nv64 chunk16/4096 字段一致；同时核 R7 K4/binary reader 身份。CPU/GPU 合成检查不是模型质量或时延证据。所有新 GPU/SSH 调度仍由 root 统一执行。
