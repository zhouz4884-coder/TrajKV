# Round8：最终解码视觉 K 均值补偿 select 结果

本批完成 Doc 38 题、Text 22 题的 mean off/on，共 120 条真实模型回答。两域各 16 个旧 select source，全部已历史暴露；本结果不属于新源确认。

| 方法 | Doc ANLS（38 题） | Text accuracy（22 题） |
|---|---:|---:|
| mean off | 92.2289% | 90.4545% |
| mean on | 92.7553% | 90.4545% |
| on − off | +0.5263 pp | 0 pp |

Doc 仅 1 题的答案/token 改变；paired source-cluster exact sign-flip p=1、Holm=1，差值 bootstrap 95% CI 为 [0, 1.7647] pp。Text 22 题的答案和 token 完全相同。两域全部 60 题的 initial-logit SHA 均改变，所以该操作确实进入模型计算，但当前 select 证据不足以确立稳健任务收益。

方法保持 old Gaussian16、L2 assignment、norm gain/frozen factor、原 signMAD、binary/all V、完整 224 排序及实际 43 阶段，且禁用 K joint、V joint、CD、postres 与 final-ray。每层编码结束、原视觉 FP16 K/V 已释放后，临时按实际最终 packed 字段解码全部真实 Nv（包括每一级 FP16 residual），只向现有 `direction_mean` 写入 `FP16(mu0 − H_FP32(mean(final_Zhat)))`。实际 chunk4096、每层恰好一次；J/g/sign/alpha/V/text/book 的 SHA 和存储指针不变，实际 owner 无新增 tensor 字节。

目标是已有的 FP16 source-mean header mu0。Doc 的平均逐头目标残留 L2 从 0.329670 降至 0.0067566，Text 从 0.325091 降至 0.0068778；共 6,720 个逐头观测均降低。仍存在 header FP16 舍入、原 source mean 量化与 H(q) 舍入，不能表述为真实均值或部署 reader bias 的位级精确消除。

CUDA 门禁的 8 组 Nv64/256 × q1/11 × 无文本/混合文本均通过。实际 R8 与旧 R7 reader 输出位级一致；相对同物理字段 dense decode，最大绝对误差 0.00048828125、最大相对 L2 0.0002878073；混合组 log-odds 平移公式最大误差 8.7182e-8。chunk37/4096 的最终均值位级一致。无文本输出变化处于 FP16 reader 容差内。这些是合成算术检查，不是任务质量或时延证据。

关闭候选分别对 postres-off、joint-CD L2-off、主链 R7 frozen43、主链 R8 oldK4/Vbinary 做了同题对照：每条基线均 60/60 answer、token、initial-logit SHA 与完整 `owned_initial` 精确一致。on/off owner 同样一致，真实 Nv 为 1280–1792，两个参数的有效 max_patches/cap 均为 8。

有效 producer 是 `mean_select2_v2_doc/text`，须以最后的 `complete_mean.json` 为准，并通过新增独立 `analyze_round8_final_quality.py` 再评分。v1 的 120 条推理完成后，收尾误用 `id` 而真实记录只有 `session_id`，因此没有最终 marker、未评分且原 run/source 全部保留；v2 仅修复 schema/runner identity，数值初始化路径 AST 不变。独立 gate 的 owner 类型检查也用真实 `owned()` mapping 修正，未改有效 producer、记录或最终 marker。

证据入口为 `results/round8_v1/mean_select2_v2_summary_v1.json`，SHA256 `3253512553f3f25f5a7f42ade7e52696d20a549336dd382f70d2cc17692f997d`；其中绑定 CUDA、CPU、source stage、120 条 producer、正式评分、四条基线一致性与双卡释放的 SHA。实际 argv 从隔离 sourceRoot 执行，执行源以 producer 编码前后 SHA 与 stage/runner identity 绑定；launcher 的 global-code snapshot 仅作为快照盘点，不能当作新 Mean 入口实际由该副本执行的证明。

GPU0 PID509810、GPU1 PID509811 均已退出；释放观察中两卡各约 24 MiB、compute apps 为空。未下载原图、权重或 tensor，未打开新确认 GT，未执行发布。该方法暂无 serving 时延结论。
