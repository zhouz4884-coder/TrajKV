# 第五阶段：模型可见失真与可执行编码

2026-11-14，香港时间。由 chat.md 第九至十一节达成执行共识后开始，当前研究目标已经建立为 active。旧阶段和公开快照保持不变。

## 研究目标

在真实视觉缓存、位置及规定辅助/encoder常量不超过1.8bit/KV scalar的格式中提高质量，固定Native4作为质量参照。理论平方失真、材料化机制原型、真正packed候选分别报告。用实际模型干预检验“模型可见失真能否指导编码”，不把Gaussian率失真参照当作ANLS预言。

当前K是2bit/元素的256×4 PQ，V为1bit；1.579/1.695是含元数据的K/V平均码率，不能代入K payload率失真公式。保留现有K每token/KV-head范数。文本KV保持FP16。

## 固定数据与启动

- T0：从 next_cal32_independent_questions.json 固定前16个唯一图像，每图一个问题，真实Full FP16 cache的完整question/decode前向。校准来源32图83问，与已有dev64图像不重合。
- 首轮校准：上述校准池前16图共39个问题、四臂各80步，同初始化、同K表示、相同teacher上下文，最多8个teacher自生token；无参考答案。
- 开发首轮：docvqa_next_dev64.json固定前16图62问，所有预定variant完整覆盖。之后有效候选扩到完整64图255问。它是既有开发池，不称新确认。
- 新确认在候选定型后另冻结源文档和参数；旧4万回答不重启。

所有模型、数据、run及TRITON/TMP/Inductor/CUDA缓存均在远端数据盘。新任务经 launch_remote_data_disk.py 启动并执行不可变代码快照。自有2×RTX3090，学校V100未用。

## 已启动的独立任务

1. round5_vcheck_v1：12项GPU同表示测试，已PASS。
2. round5_t0_v1：原始完整16图采集，保留；其重构诊断码本先FP32构建，与实际encoder的FP16码本存在差异，因此不作最终同codec误差证据。协方差来源不受该重构支路影响。
3. round5_t0_v2：复现FP16码本与FP16 query-WHT/rotated K乘法路径，16图、2632个layer/query局部样本，COMPLETE。输出 results/round5_v1/t0_v2/moments.pt、local_metrics.json。
4. round5_v_queue_v1：2步梯度smoke及G1/G8×bias关/开四臂各80步，已完成，参数在 results/round5_v1/v_queue/。
5. round5_spectrum_analysis_v1：完整二阶矩、稳定白化、Gaussian参考与变换artifact，COMPLETE。
6. round5_spectrum_queue_v1：2图满秩数值控制，再16图15配置端到端自由回答。
7. round5_v_eval_v1：16图8配置端到端自由回答，含Full、Native4、Base2、冻结整图底座与四个新臂。

配置在 configs/a2/round5_v1/。每个队列按依赖完成与成功状态顺序运行，不重叠占用同张GPU。状态以远端complete.json/当前进程/日志为准，本文件不是持续心跳。

## T0首批发现（尚非质量结果）

112个layer×KV-head平均，mixed query加权K谱前4/8/16/32/64维能量约77.15/87.02/94.02/97.74/99.41%。Gaussian理想K payload B1/B2的归一化参考失真约0.01234/0.002736。前缀每KV-head仅112条q样本，存在样本秩限制；decode相邻query相关。上述数不等于新图上的编码误差或答案预测。

V残差在未attention加权的W_O Gram代理下，前两维平均覆盖7.11%，暂不支持rank2万能修补。实际同q/同text cache下经完整W_O投影的误差能量总和：K-only104740、V-only254411、joint332993；参考输出能量5050323。交互与非线性余项同时保存，不能直接把K分数谱和V输出谱拼成“共同最优水位”。

## 之后的实验动作

先等已排队的真实开发结果，不按谱阈值取消另一条路线。围绕实测发现准备三个便宜候选：

- K现有mean槽的重构偏差修正，FP16字节不变，FP32增量8/N bit；固定原code/gain/V，改变视觉/文本竞争。
- FWHT域4D块的query metric加权PQ选码，保持现有书/格式/reader，计encoder metric常量；与Euclidean选码直接比较。
- 整图V1底座的稀疏layer×KV-head V符号残差，真实只存选中头的1bit+per-token FP16幅度，按每图完整owner预算取静态排序前缀。包含同价random/uniform对照。

仅在完成机制/数值控制后启动新真实推理；按效果推进，最终输出配对质量、实际bytes、耗时与机制解释。Gaussian参考不要求先成立才允许做实验。

## 17:25阶段更新

完整功能校准开发64图255问、640个真实会话已完成：Native4 92.079、Base2 88.966、整图底座87.230；G1无/有bias为87.492/87.481，G8为87.251/87.240。mean FP16/F32均87.230。当前没有明显的功能分组或mean修复质量收益。

稀疏真实packed原型首批16图62问：整图底座92.579，V干预收益排序、预算1.7993bit为95.668；同预算random92.756、spread94.009、K干预排序92.779；V前10/20单元分别92.634/94.247（约1.629/1.679bit）。这是开发pilot，完整64图两卡分片评测已启动，排序和参数保持冻结。

谱路线16图：r16/r32/r64高精度投影ANLS24.82/83.42/91.22；相同rank恢复每token范数后r16/r32为36.90/89.90。首问question-prefill与后续decode/text-prefill、不同student q必须区分；不把前缀MSE与全会话ANLS错配后宣称所有分数代理被否定。固定Full q/text局部对照已实现并排队，真实参考答案teacherforcing已完成（明确为连续诊断，不计入自由答案）。

common输出Jacobian近似已计算，两轴统一到局部W_O输出平方误差。Gaussian共同水位平均给出K0.598/V2.402 payload的方向参考；它依赖线性化、token独立及采样Jacobian，不是可执行码，不证明可丢K尾部。实际符号残差收益采集显示V当前有更大边际空间，遂用于冻结静态分配排序。

新的独立Doc确认已冻结为64源文档256问，从排除820个已用源后剩余118源中按旧固定hash顺序抽取；剩余54保留。ordered64 ID hash为ac1f8af36a6231a1e23eaad86a237db1ec449a767e12b827598d367ec8db01cb。图像及无答案runner清单已上传，65文件hash全部通过，参考答案留本地；尚未开始这批确认推理。

执行修正：common_queue中的collector成功，但随后GPU数值检查缺少新bank源文件，导致队列和等待它的functional48_v1退出，原目录保留。补齐源文件后functional48_v2检查PASS并完成；最终两轴残差24个数值case也PASS。T0和评测数据未被覆盖。

文献检索同步核对KVTuner/RateQuant等现有敏感度分配工作；准备同一符号残差码、同预算的梯度加权排序和MSE排序对照。它们是strategy-port，若与官方训练目标不同明确标注，不称完整官方复现。组合创新以新的实际能力和受控比较为依据。
