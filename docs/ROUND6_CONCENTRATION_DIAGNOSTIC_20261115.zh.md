本轮 C/Ctilde 是旁路校准诊断。真实运行快照的 SHA256 与采集 manifest 一致，其 `concentration_diagnostics` 函数与当前实现的源码和 AST 均一致。16 个校准 case 中仅首个 `docvqa_next_2126_cal_q1` 保存非空诊断，符合 `concentration_cases=1`；覆盖 28 层、每层 28 个 Q-head 和 4 个 teacher 答案查询位置，共 3,136 个 layer–head–query 点、9,408 个标量。该图含 1,792 个真实视觉 token，其余 15 个 case 未采此旁路。

诊断从相同 Student q/k/v 在 FP32 中重算完整因果 softmax，支持集含视觉和文本。视觉 mass 为视觉权重之和；`C=Σ_visual a_i²`；`Ctilde=Σ_visual a_i²‖v_i−o_total‖²`，其中 `o_total=Σ_all a_i v_i`。它们位于每个 Q-head 的 pre-W_O、D128 value 坐标，尚未进行 W_O 投影；该统计不是最终任务损失。C 无量纲，Ctilde 保留 value 坐标平方范数量纲，不能直接比较二者大小判定 K/V 配额。

全部 9,408 个标量有限。视觉 mass 范围为 9.61532e-13–0.999754786（中位数 0.206117）；C 为 8.15546e-25–0.818020701（中位数 0.000380363）；Ctilde 为 1.05897e-22–119.27079（中位数 0.0147817）。3,136 点均满足 `0≤mass≤1`、`mass²/N≤C≤mass²` 和 `Ctilde≥0`，按保存数值直接检查，无须容差。Ctilde 实现含 `clamp_min(0)`，保存文件不能评估截断前的数值抵消误差；未保存完整权重/value 张量，因此上述不变量检查也不等于重新计算全部诊断。

这是单图的描述性交付，4 个查询位置及各层/head 不构成独立任务样本。本汇总未生成新排序、配比规则或性能预测，未新增 GPU 采样，未构造 N×N 矩阵。机器汇总位于 `results/round6_v1/student/concentration_summary_v1.json`。
