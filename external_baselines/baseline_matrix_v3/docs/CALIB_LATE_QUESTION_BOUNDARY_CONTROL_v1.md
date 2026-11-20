# 独立execution-boundary对照

源码证据支持增加一个单列控制：若选择作者Calib分支，完整prompt prefill（q_len>1）虽然编码视觉KV，却使用未压缩重建的原始Q/K/V走`_flash_attention_forward`产生输出；同时保存nonvisual原生K/V，其中已包含问题文本状态。packed visual reader只在q_len==1分支调用。fresh完整prompt的第一答案logits因此来自这次native路径。`infer.py`示例选择qjl，这份证据是对单独存在的Calib分支的源码判断，不是假称运行示例就已测Calib。

`audit/calib_execution_boundary_source_v1.json`记录作者AST分支、变量和行号，并在当前服务器Transformers5.12.0上只用CPU检查generation源码：prefill输出→最后位置logits→logits processors→greedy argmax→追加token→stopping criteria。没有加载模型权重或初始化CUDA。

|条件|压缩时刻|问题text KV|第一答案token|
|---|---|---|---|
|A 主benchmark协议|图像prefix后、问题计算前|在压缩视觉上下文下重算|各方法自行计算|
|B 独立late/query-known控制|完整prompt native prefill后|已经在native上下文中计算|明确使用native y1|

B已经支付了当前问题的高精度视觉读取，还通过native问题KV保存了部分读出状态。这与未来问题未知时先压图像的A不同。B若质量更高，不能单归为codec更好，也不能直接补入A主表；此对照同时改变问题状态形成与第一token来源，不能只归因于y1一个因素。短答案尤其可能主要由已受保护的native y1承载，但须真实实验确认。

prototype为`code/gates/late_question_compression_m1.py`，默认固定Text/Info首各1题，可选各4题；模型、像素、suffix、64-token总上限和F/C1bias32/C2bias32参数相同。A保留现有imageprefix→压→完整suffix及首token。B用原生HF生成的停止条件只走一次native prefill，保持原总上限以免max_new_tokens=1意外触发forced-EOS；取y1和prompt cache后释放native logits。再仅压visual，使用完整P+y1历史与长度P的cache继续y2，不重复处理question或y1。

原生y1若为EOS或总cap只有1，不启动压缩读取。续生cap为总cap−1，min_new_tokens按已完成1个调整，首次专用suppression不在y2重复应用。默认M1的其他设置保留；prototype不支持的主动stop_strings等配置会明确拒绝。每臂记录实际forward输入长度，以验证A首call为完整suffix，B首续生call仅为未处理y1。F全native控制可观察full/split执行形状差异。

报告显式分开native token与压缩状态生成的token，B标`query_known_late_control`，不使用主benchmark raw schema，不进入原预测缓存，不读取GT或输出量化任务成绩。原生cache副本为诊断对照保留，时间和GPU峰值不作部署比较。它使用共同M1/FP16数值端口，不声称作者InternVL2.5/BF16原patch位级复现。

`audit/late_question_boundary_cpu_gate_v1.json`通过10项CPU fake-model控制流/schema检查，包括完整history、y1不在cache中、EOS、总cap、processor相对计数及原配置不变；这些假输出不是模型实验。该prototype尚未启动GPU，正式F全量银行和A协议均未修改。
