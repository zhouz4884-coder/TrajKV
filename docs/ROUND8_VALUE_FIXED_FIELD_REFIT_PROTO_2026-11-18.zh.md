# V binary/all 已有字段联合回拟合原型

新模块 `code/a2/round8_value_joint_refit.py` 只准备数学与 CPU 算术原型，不接 bank/runner，也不新增 reader/kernel。固定原 channel bit b、一级残差 sign s、all-frame 分组、positions 和选中 head，只更新原 FP16 m[128,1]、S[128,1] 与 alpha[N]。输出为已 fold 的物理字段，不能再次乘 factor 或加 offset。

目标是原 native V 对实际重建 `Q16(Q16(Q16(S*b)+m)+alpha*s)` 的 SSE。固定 alpha，令 y=v-alpha*s，逐 channel 求两个类均值。如果两个非空类均值有序，就分别取均值；若次序反转，用 n0/n1 加权 pool 到全体 y 的均值，令 S=0。空类不识别唯一中心：本版只提议两个中心都等于已观测类均值，S=0。旧字段始终作为 incumbent，所有候选都须物理 FP16 评分，ties 不改，因此不会强制改写一个未使用却等价的旧 level。

把候选 m/S 舍入到 FP16，并按当前 alpha 和真实嵌套解码评分。每 channel 全 N 的 SSE 严格降低才一起接受该 channel 的 m/S。随后固定新 frame，先算实际 FP16 base，再提议 `alpha=max(mean_c(s*(v-base)),0)`，舍入 FP16 后按每 row 的 128D 实际最终 SSE 严格降低才接受。sign 固定时不能用 MAD 代替 signed mean。恰做 2 或 3 轮，不重编码 sign。

两个 phase 分别操作不相交的 column/row blocks，因而各自 hard-SSE 接受条件控制总重构 SSE 不升。Frame phase 可增加某个 row 的误差，alpha phase 可增加某个 channel 的误差，不能宣称每 row/channel 跨两个 phase 都单调。连续闭式解仅产生候选，也不等于 FP16 或全局最优；没有 attention、任务质量或速度保证。拒绝非有限 native 输入/物理字段、负初始 S/alpha；m 允许为负。

接口是 `joint_refit_value(sourceNative, bit_codeI32, old_scaleF16, old_minF16, sign_codeI32, old_alphaF16, rounds=2)`。单个 head 的 shape 分别是 FP16/FP32[N,128]、I32[128,N/32]、F16[128,1]、F16[128,1]、I32[N,4]、F16[N]，返回同 shape/dtype 的新 S、m、alpha 和标量 stats。当前 N 正且整除 32，与 Round8Bank 一致。所有输入不变，输出不包含源、解包 bit/sign 或新常量；临时 scratch 与额外 encoder 计算需要另行测量。

后续最小集成可在新子类的 `_encode_axis` 中先运行原方法，随后仅当 axis='v'、value_encoding='binary'、vbook=None、vgroup=N 且该 head 恰有一级现存 V residual 时调用本函数。传 `source[head]`、`state.v_code[0,head]`、`state.v_scale[0,head]`、`state.v_min[0,head]`，以及 `slot=selected.index(('v',head,0))` 对应的原 packed sign/alpha。只把返回 S/m/alpha copy 回原槽，不修改 bit/sign，不在之后再次 fold 或再生 sign。二级/缺失 residual 或其它 V 格式应显式拒绝此 intervention。现有 K 路径和未选中 V heads 保持原方法。需要同步重算该 head 的 base/final distortion 记录，并把 refit stats 标记为 native fixed-code coefficient refit；原 pre-refit distortion 不可冒充最终值。

完整 owner 不变：每 head 的优化字段始终是 2*(256+N)B，bit words、residual sign words、lookup/ranking/positions 也不变。此原型没有新增码本、row gain、字段或选中 stage。实际部署预算仍须由原 owner 与独立 analyzer 复核。

CPU 回执 `outputs/round8_value_joint_cpu_v1/check.json`：12 cases、65 个独立 scalar physical-decode 对照通过，覆盖 N32/256/1792 的 2/3 轮随机控制、严格接受/输入不变/同 bytes、不等簇 weighted pool、两种 empty class/tie、S 与 alpha 约束以及旧物理 decode overflow 修复。一个舍入 witness 中 `S=1,m=alpha=2^-11,b=s=1` 的真实两次加法结果为 1，而跳过中间 base 舍入的结果是 1+2^-10。合成可表示例的 SSE 降至零仅是算术正确性证据；本轮无真实 V/model 数据、GPU 或任务质量测量。
