本分析只用D160 development：primary1120＋middle新增320＋KRA新增160＝1600次实际free回答，40个run；两个追加视图各复用480次primary参照。合计2560个view事件不等于2560次独立推理。

V128-R43相对R7low（Text/Info/Doc/Chart）为+2.6563/−0.9549/+2.0833/0 pp；配对95%CI分别为[0,6.8750]/[−4.6875,1.8403]/[0,6.2500]/[0,0]。相对近字节AddOldOrder为+4.8438/−1.0938/+6.2500/+12.5000 pp。Text的未校正区间[0.6250,10.1563]仍是选择用development证据。

LS-FZ四域净分均与R7low相同；Text实际2题改善、1题变差，gross gain/loss各1.5625 pp，净0，不能写成逐题恒等或等价。

新增KRA1独立复核：160/160的raw_prediction和output_token_ids都与R7low相同，四域score变化均为0，每题完整owner额外5980B。零差异bootstrap CI=[0,0]仅反映这160个已观察样本，不证明模型或分布等价。这是有实际alpha坐标动作但没有观察到任务收益的有界负结果。

Root已在selection_freeze_v1/DECISION.json固定仅V128-R43进入224复测，不扩展KRA；本分析没有读取任何reserved224预测/评分。D160用于候选选择，不作为独立确认；各域指标不混合平均。

完整owner字节/bit逐题核查，V128与AddOldOrder差额小于一个18N stage而非精确同字节。combined表保留native prefill、encode、question prefill、TTFT、自然长度decode间隔之和、wall及输出长度；均为含同步诊断的仪器化host时间，不是受控加速/HBM结论。
