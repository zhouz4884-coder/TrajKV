# 原生F正式参考银行收数

`ops/watch_native_F_full.py`是CPU收数控制器，隐藏后台运行。它不调用GPU启动、停止、重试或stage命令；只读取远端已关闭的结果、独立核验raw和源码SHA、形成有逐行来源的完整cohort视图，并离线评分。大raw和详细scored行均放E镜像，权威原始记录仍在远端8TB。

当前实例`native_F_v1`的状态和日志位于`${EVIDENCE_ROOT}/results/harvest/native_F_v1/`。启动PID见`audit/native_F_watcher_launch_v1.json`，实际argv已核验；CPU子进程使用空CUDA可见设备。每30秒检查闭源marker，已有core收数器在处理同一raw时优先等待其物证，避免并发写同名文件。

Doc/Chart严格沿用已冻结的P2源计划：500原始PASS分别与4,849、2,000条新原始行组合成5,349、2,500条完整view，之后才使用full refs评分。Text/Info须先由实际128条P1和372条新P2原始行形成完整500、验证源码和coverage；各自满足后才准备full的4,500/2,301缺失ID工作。写配置不代表启动；调度仍由Root负责。原Doc/Chart计划与原full目标配置不覆盖。

每个F单元按对应P2完成状态解锁，不等待其他压缩方法。新四target矩阵准备在`configs/matrix/P3_M1_F_all4_missing_v2.json`；尚未完成的单元不会被填造缓存命中。该流程不会解锁压缩方法full、本方O或P4。

四个完整视图最终关闭并评分后，控制器会准备`results/native_F_bank/native_F_v1/PHASE_CONTRACT.json`、`summary.csv`及相应短报告供Root公开同步。记录完整题数、逐源SHA和fresh/reused；Chart human/aug分列留在groups中，不把四种不同指标合成一个总平均。速度仍是原型质量运行中的instrumented记录，不当作可重用的受控速度。

任何source/coverage/评分问题写入state错误记录并保留证据；失败行继续计入完整cohort，主质量按现合同保持null，不报成功子集均值冒称full。派生view从未冒称新增GPU推理。控制器本身不会上传GitHub，公开同步仍需Root完成脱敏、远端commit/tree和证据归档核验。
