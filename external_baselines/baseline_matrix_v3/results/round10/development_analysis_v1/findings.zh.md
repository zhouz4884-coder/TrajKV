本次仅复核D160 development：36个实际run、1440次原始free回答；middle视图另复用primary的480次参照，合并视图为1920次，不能当作1920次独立推理。未读取reserved224或KRA预测/评分。

V128-R43相对R7low的各域变化（pp，配对source bootstrap 95%区间）：
- textvqa: 79.6875% → 82.3438%，Δ=+2.6562 pp，CI [0.0000, 6.8750]；score改善/变差 2/0题。
- infovqa: 82.9908% → 82.0360%，Δ=-0.9549 pp，CI [-4.6875, 1.8403]；score改善/变差 2/2题。
- docvqa: 97.9167% → 100.0000%，Δ=+2.0833 pp，CI [0.0000, 6.2500]；score改善/变差 1/0题。
- chartqa: 81.2500% → 81.2500%，Δ=+0.0000 pp，CI [0.0000, 0.0000]；score改善/变差 0/0题。

与近同字节AddOldOrder比较，V128-R43的变化：
- textvqa: Δ=+4.8437 pp，CI [0.6250, 10.1562]。
- infovqa: Δ=-1.0938 pp，CI [-5.4688, 3.5937]。
- docvqa: Δ=+6.2500 pp，CI [0.0000, 18.7500]。
- chartqa: Δ=+12.5000 pp，CI [0.0000, 31.2500]。

LS-FZ四域净分均与R7low相同。TextVQA不是逐题恒等：2题score改善、1题变差，gross gain/loss各1.5625 pp，净0；配对CI [-4.2188, 3.7500]。其余三域逐题score相同，预测字符串差异另见flips表。

V128在Text/Doc/Chart没有观察到相对R7low的score下降题，但Info略降；相对近字节AddOldOrder的Text区间不跨0也只是同一development面板上的未校正描述，不能升级为独立确认、等价或算法胜者结论。不同数据集指标不作混合平均。

每题实测V128与AddOldOrder的容量差严格小于一个18N残差stage，非精确同字节。完整owner含原position map、codebook、factors、lookup/ranking/policy；LS额外896B保留计费。详见quality_complete_owner.csv和analysis.json逐题容量。

计时为真实quality-run的仪器化host统计，保留native prefill、encode、question prefill、TTFT、自然长度decode间隔之和及wall。encode含同步诊断，输出长度与初始化状态不同；这些数据不支持受控加速或HBM流量结论。

配对bootstrap固定seed=20261119、2000次source抽样；84个既有配对区间独立重算吻合。1440条原始预测均已用冻结metric与dev参考答案重算，raw行SHA/评分派生文件/物理storage表及来源SHA均核对，输入证据执行前后不变。
