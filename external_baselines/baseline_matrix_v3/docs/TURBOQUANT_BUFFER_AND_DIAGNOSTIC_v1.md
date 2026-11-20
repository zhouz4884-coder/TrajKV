# TurboQuant近窗与整模型诊断

P1旧T0记录保持原样。它表示选定0xSero实现的uniform K3/V2、视觉recent buffer=0端口；近0质量是需解释的真实移植结果，不能当作原论文或Google官方方法的成绩。原论文LongBench配置另含outlier/nonoutlier混合位宽，本端口没有声称复现该配置。选中源码的cache类也没有实现header提到的outlier组件。

源码`third_party/turboquant_0xsero/turboquant/kv_cache.py`的`TurboQuantKVCache`默认`buffer_size=128`，prefill只量化`seq_len-128`之前的K/V，其后保持FP16。新模块`code/methods/turboquant_buffered_adapter.py`恢复这一prefill近窗策略：以**原始逻辑位置**`prefix_length-128`切分视觉候选，文本继续遵循矩阵公共协议保持FP16。不是机械地取最近128个视觉候选，因此有位置间隙时保护的视觉token可以少于128。当前M1/Qwen协议的prefix boundary等于最后视觉位置加一。

这是共同视觉前缀协议下的**静态prefill近窗端口**；后续文本全部精确保存，视觉前缀不随新文本滚动重编码，不能标原方法完整rolling cache系统。FP16视觉tail及其位置均真实持有并计费，不能仍以名义2.5bit给default近窗点记账。tail和原有精确text被并入同一次softmax，与量化视觉头部共享归一化；并非各自softmax后相加。packed分支委托冻结原始packed-key reader，无全视觉materialize fallback。全前缀都落在近窗内时是源策略允许的显式全FP16情形。

CPU验收`audit/turboquant_buffered_cpu_gate_v1.json`覆盖16个组合：GQA28:4/32:8，N17/129/257，逻辑间隔1/2/3，R0/R128，V2/V4。与源类prefill的整数码逐项一致；全prefix与选中行的FP32 GEMM形状不同导致最大residual-norm舍入差4.768e-7，单独记录，不宣称浮点位级相等。未来suffix value变化未影响更早query；owner不引用原生full-prefix storage，物理字节核对通过。此门只在CPU运行，尚不构成目标3090 packed验收。

新诊断驱动`code/gates/turboquant_m1_diagnostic.py`默认取预定Text/Info各前2题，无GT访问。一次加载M1，使用同一个原生图像prefix原型，每臂克隆并独立重算整个问题suffix和首个自由token：F、T0 packed、T0 dense参考、T128 packed、T128 dense参考、K3V4/R128 packed。默认最多生成16个token，仅作诊断，不拼进P1正式成绩。原始native q和FP16 visual K/V在0/7/14/21/27层保存到远端8TB的pt，禁止下载或公开上传；JSON保存每层角度/范数/居中logit误差/attention输出误差、packed-vs-dense差以及实际压缩分支的q漂移。故意同时持有原生原型和临时解码状态，时间和GPU峰值均不作压缩效率结论。

已有CPU代理probe使用的是Calib2重建视觉K/V及C2分支q，不能替代native状态。layer26的该代理中，K误差约0.426、V误差约0.353、logit相关约0.920；mean-key能量仅37.5%，不支持预先断言“共同均值巨大必为主因”。须以新native多层probe检验。

QJL本身也不能按“更多bit必降低向量MSE”判断。令残差r、m个独立高斯投影g，源码估计为`r_hat = sqrt(pi/2)*||r||/m * Σ g sign(g·r)`。它的期望为r，而向量均方误差为`E||r_hat-r||² = ((pi*d/2-1)/m)||r||²`。当m=d=128，系数约1.563；无偏inner-product修正可以提高向量MSE，不能据此认定实现错误。真正需要测量的是原始query方向上的误差、相对跨位置logit差及联合视觉/text softmax后的输出。该数学式对随机投影取期望，不保证单个固定seed或argmax保真，更不直接给出任务ANLS。

若R128仍表现异常，先区分packed实现错误、codec失真与VLM分布迁移，再决定是否评估另一公开实现或原论文近似配置。任何额外中心化或本方算法修补都应另立方法名，不能改完仍称原始TurboQuant。
