# C2 TextVQA固定128源试评

C2_fp32_bias在真实M1上完成128/128问，原始manifest、records和complete经独立远端SHA核验，源码snapshot物证也已取回。按冻结MMF十票soft评分得到 **74.53125%**，source-cluster bootstrap95%CI为[67.96875,81.328125]%。这是预定TextVQA validation子集，不是官方full-split成绩。

同题F为81.5625%；C2对F配对差−7.03125pp，95%CI [−12.1875,−2.18555]pp。C1同cohort为55.625%，因此第二个合法bitwidth的质量明显不同，不能由C1结果代替C2。多格比较中的未校正p值不单独解释为最终显著性结论。

完整owner平均2.022206bit/elem，范围[2.020159,2.044062]，native视觉bytes比为12.6339%。128题均不满足1.8bit；在2.25bit预算下均满足。计费包含packed payload、量化scale/zero、LUT及原生位置元数据，正式比较不得换成名义2bit。原型质量运行平均wall约4.294秒/问，包含审计/同步/streamer，未作受控速度排名。

方法ID `CalibQuant_visual_FP32bias_2bit`与已准备的500问目标配置完全一致；只按完全相同合同复用128条PASS原始预测，补跑372个missing ID，不重新生成这128题。Info500与Text的输入key不同，不能按同一个method就假称命中。

小汇总在`results/scored/p1extra_v1/M1_C2_bias32_textvqa_p1extra_128_v1/`，详细raw/scored行在E镜像、权威raw在远端8TB。配对与完整字节表在`results/analysis/C2_roundoff_closure_v2/`。本次没有改变clip、−3校准偏移、模型精度或原始FP16失败记录。
