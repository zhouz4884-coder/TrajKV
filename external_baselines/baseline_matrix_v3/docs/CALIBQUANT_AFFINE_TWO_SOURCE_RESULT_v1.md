# FP32 affine两源真实诊断结果

M1真实3090诊断完成2源×5臂，全部完整suffix和独立首token/freegen通过。原生F对native suffix逐元素一致；两种bitwidth的新旧端口在28层的编码、metadata及完整owner bytes均相同。四组新旧生成token IDs全部相同。

|来源|F输出|C1 bias/affine输出|C2 bias/affine输出|
|---|---|---|---|
|Text首源|stop|stop / stop|stop / stop|
|Info首源|11.93|15.85 / 15.85|8.42 / 8.42|

没有读取GT或计算这两题的任务分数；可直接确认的事实是新旧输出未改变，不能据此推出全任务有收益或没有收益。暂不注册新class并大规模扩表。

三个native-q层0/14/27、两种bits、两个来源共12组coded-math检查中，FP32 logits相对“同一实际Half原kernel输出边界”的FP64仿射/校准参考，relative L2为约6.59e-8–1.23e-7。weighted输出恢复Half后对同参考的relative L2约1.82e-4–2.46e-4；对FP64结果再round Half，relative L2约3.94e-6–1.54e-5。它们验证临时算术路径，不代表保持原生FP16 KV的attention或答案。原kernel自身的Half输出边界仍存在。

源码BF16模型与硬Half kernel输出会自动产生FP32仿射/校准，而共同Half模型不会，这项dtype发现仍成立。新prototype采用更统一的FP32临时量，并保留公共bridge提前Half返回的接口，始终标为独立数值port；没有称其为作者BF16位级复现或本方算法创新。旧23项误差闭环、旧C2 Text128和全部源数据保持原样。

`audit/Calib_affine_diagnostic_fetch_v1.json`独立核对了原始JSON，报告SHA为`981e4c8ff9ce4af21e65725dceaee6394a824d2035b2fce8ffcab08eac24346f`。`results/analysis/Calib_affine_diagnostic_v1/`保存10行分支结果、12行native-q算术记录和4组owner/生成一致性。诊断保留原生副本及FP64临时量，不作内存峰值或速度基准。
