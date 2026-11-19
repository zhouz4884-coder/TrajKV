# Round8 自然 dev64：Ray norm 与 Mean on 最终验证

四个独立 job 完成256条新真实回答：Doc/Text各64题，分别运行Ray norm与Mean on。沿用旧自然dev64首问样本和原Full/R7/jointgain三臂；每个新候选各做独立四方法正式分析。128个source/问题均属于历史暴露开发集，不是新源确认。

| 方法 | Doc ANLS（64题） | Text accuracy（64题） |
|---|---:|---:|
| Full | 92.0723% | 78.5938% |
| 旧 R7 frozen43 | 92.3993% | 70.9375% |
| 旧 jointgain | 92.3212% | 68.4375% |
| Ray norm | 92.3993% | 68.4375% |
| Mean on | 92.3993% | 70.9375% |

Ray norm相对旧R7的Doc差值为0，Text为−2.5pp；Text paired source-cluster sign-flip p=0.500685、Holm=1，bootstrap差值95% CI为[−6.5625,0]pp。Mean on两域差值均为0、paired p/Holm均为1。此批未得到相对旧R7的任务收益，不能把先前select上的小幅正点估计推广为稳定提升。

Ray norm相对R7改变Doc1题、Text3题答案/token；Mean on改变Doc0题、Text1题，后者评分不变。四个新候选的initial-logit SHA均64/64改变。Mean的14,336个逐头观测，其已有FP16 mean目标的残留均降低（Doc平均L2 0.330062→0.00684257，Text 0.335172→0.00690025），但该坐标指标没有在本批转化为任务增益；它不等于真实source mean或reader bias的位级消除。

全部候选保持old Gaussian16/f0、L2、init norm、binary/all V、完整224 ranking和23K+20V的43阶段，禁用其它干预。四批各64条的image ID、真实Nv、cap8、完整`owned_initial`与首问suffix重放长度均逐同题精确匹配旧R7，新增持久tensor字节为0。Doc实际Nv1280–2304，Text768–1792；`max_patches=8`可能另含thumbnail，不能据8×256假定token上界。本次远端重新哈希128张原图，全部匹配冻结sample SHA；没有下载图片。

本批实际使用专用`code/launch_round8_data_disk.py`，已读取四个live PID的`/proc`命令行并逐项匹配launch receipt，入口真正位于`SR/jobs/<name>.source/code/a2/`。方法依赖在snapshot中的字节与库存SHA一致，另逐文件核对实际snapshot的executor、session_boundary和factory。globalR配置/样本/结果、8TB P72 worker和项目shared cache保持声明路径；没有改动旧launch或旧源。

四个producer的最后`complete_final_ray.json`或`complete_mean.json`、中间R8/engine、canonical/source/config/input/64-record覆盖图及独立最终门禁全部通过后才评分。正式分析分别位于`natural_dev64_raynorm1_v1_{doc,text}_merged4_analysis_v1`和`natural_dev64_mean1_v1_{doc,text}_merged4_analysis_v1`。四视图共1,024条评分行复用了相同基线；实际新增回答仍为256，含旧三臂的唯一producer回答总计640，不能重复计算独立样本。

证据入口：`results/round8_v1/natural_dev64_ray_mean_final_v1_summary.json`，SHA256 `0ff7a5e6543735682007426b8c22c9f8b62581221780393361c5dea9db7b789b`。它绑定四个producer和正式评分SHA、四份独立匹配owner/geometry审计、remote图像SHA、实际snapshot/runtime JSON及释放观察。

Ray Doc/Text PID521197/521198、Mean Doc/Text PID522548/522272均已退出；分卡释放时compute apps为空、显存约24MiB。该批不再启动GPU任务，未打开新确认GT、未下载权重或tensor、未发布，也没有时延/吞吐结论。
