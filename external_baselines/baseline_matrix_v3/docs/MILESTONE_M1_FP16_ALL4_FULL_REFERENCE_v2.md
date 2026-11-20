# M1 FP16 四个官方完整 split 参考银行

已完成 `OpenGVLab/InternVL3-8B-hf`（revision `259a3b64a14623c0ec91a045cb43f7c5af5fa6af`）在四个完整公开 split 的真实端到端评测：15,650 题全部执行成功。该阶段新增13,650条预测，另2,000条沿用同模型、提示、预处理、生成和压缩边界合同下的既有预测，保留10个原始GPU run及逐行来源。派生组合没有新增模型回答。

|数据集|完整题数|质量 %|来源簇95% CI %|新增 / 复用|
|---|---:|---:|---|---:|
|textvqa|5000|81.9400|[80.9335, 82.9331]|4500 / 500|
|docvqa|5349|91.4037|[90.5345, 92.2611]|4849 / 500|
|infovqa|2801|72.1018|[69.6174, 74.4529]|2301 / 500|
|chartqa|2500|86.2400|[84.8374, 87.6533]|2000 / 500|

TextVQA采用锁定的十票soft指标；DocVQA/InfographicVQA采用锁定vendor ANLS；ChartQA采用锁定5% relaxed指标，human/aug各1,250题，分别为77.68% / 94.80%。不同数据集的质量指标分开报告，不计算跨数据集单一均分。完整分数与固定500题分数保留各自口径。

所有原始GPU推理均使用项目数据盘的模型、数据与缓存。该M1协议使用FP16、固定单词/短语后缀、最多64个新token和max_patches=8，并在问题未知的视觉前缀边界执行原生读取；这些设置和真实输入、模型文件及processor身份均随证据固化。质量run中的instrumented wall/TTFT不作为控制速度排名。

独立CPU冷验已重新核对原始/派生SHA、逐行预测来源、锁定refs、逐题分数和来源簇bootstrap区间；完整验收见 `audit/native_F_cold/all4_cold_v1.json`。公开包还会再次在清洗后内容上重算全部指标与区间。

本次公开附加证据分别声明：OnlyTerp-MSE/QJL0 Hmix12、Hmix23各Text32+Info32=64个source-point预测，以及共享FP16 matched64支持视图。两MSE工作点使用相同固定题目，不能合称128个不同测试问题；FP16 matched视图完全复用既有预测，新增GPU回答为0。Hmix23采用实际regular2/outlier3与Hadamard padding，完整计费约3.007bit/elem，Hmix12完整计费约1.757bit/elem（含norm与其余元数据），均以逐行owner实测为准；这些是源码variant，不冒称论文精确prod表或优化kernel。临时FP32解码的峰值和时间代价保留。

独立技术支持组保留14组20份原始小结果、来源/launch/driver及远端SHA物证；CPU60是4个既有captured query ×5层×3codec的operator控制，新增GPU问答为0。另附已完成M1 K32与M2/M3各四方法的9个两题兼容smoke及v2核验/Calib编译观察源物证：18条模型/方法技术预测只来自相同的2个Text输入，不计入native或固定500质量题数。所有GPU诊断的cases/arms与正式质量题数分开。原始PT、图像、权重和认证/连接信息不公开；需要原捕获才能重跑的控制明确保留该限制。

另以独立计时支持域附上已闭合legacy656 trials（4个正式cohort576，加3个硬件smoke80）、两个独立进程Calib冷/热诊断32以及五个源profile候选的Text2硬件smoke40；本次新增计时728 trials，旧Text2前置32另作既有支持，均不增加正式质量题数。这些候选profile其余520 trials仍为conditional。每项保留token/context/owner/repeat核验与warm/measure逐条记录；host_quiet=false，不按更短错误答案的wall排名。Calib同输入重复显示首用与热路径差异，磁盘binary cache保留，进程内winner首次使用与全盘冷编译区分；两题控制不外推全500性能。

该里程碑只完成M1原生参考银行。P2跨模型固定500、压缩方法P3完整评测及各profile专属控制速度仍按自己的合同推进；这里没有声明整个外部SOTA库或本方算法研究已完成。公开上传与远端核验由独立交付receipt确认，本报告本身不表示已上传。
