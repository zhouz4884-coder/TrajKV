# 最终独立验收简报

**核心完整性通过；0 ERROR、0 PENDING、0未解释源码缺口。** `verify_delivery --remote-sources --strict-integrity` 返回0。该通过是执行/证据完整性验收，不是方法优势、论文录用或部署收益认证。

## 核心通过

- 核验36个artifact容器：30个模型/机制run、3个composite suite、1个四预算sweep、1个teacher-forced timing、1个非模型硬件counter侧检；25份分析summary与原输出/计数相符。
- inventory原8个current-copy mismatch与19个hash-only均由9个真实小文件的精确SHA解析；当前258/258声明复制品可验证，统计总数与原inventory完全相同，未改raw或manifest。
- 31个主raw JSONL文件、3,550条主记录中的524,069个登记浮点量全部finite，0坏JSON、0非有限数，读取内容SHA与inventory相符。方法/轮次/预算/重复并非独立样本；全局唯一图像并集212。
- 1,383个本地图像文件核查通过；主64、rep32、外部探索16与Doc确认64的冻结文件hash/ID及图像隔离通过；gain-table训练ID/图像与冻结评测无重叠。
- Doc确认64修订时间22:40:49早于两suite创建22:50:49/22:50:50；两suite均绑定修订后25a9…SHA。768方法-session、每法294题、3,528回答的来源/初始logit/预算/登记owner账通过。
- 11个已做完整SHA的模型文件，最终size与mtime均未变；其中4个完整权重分片共15,888,846,496字节。最终nvidia-smi compute列表为空，无本任务GPU进程。本终检只做stat，不把stat等同于再次全文件哈希。
- 统一报告conclusions的5个输入SHA全部仍匹配当前机器汇总。主64Probe−Periodic −0.3125pp、Holm16=1；rep32新Probe−Periodic −0.625pp、Holm11=1；Doc确认Future−Base2 +0.1367pp/Holm2=.88334、Future−Current −0.6868pp/Holm2=.64747，与原summary及报告一致。未重生成统一报告。

## 保留的历史排除

旧`write_dev16_gpu0_v1`的16条`repair_once`记录真实存在且预算违规，原样保留，排除等预算结论。修复后的新主批不继承该排除。旧partial19分析的精确原始输入byte-prefix已恢复，SHA逐字匹配；原summary与其PARTIAL历史不变，不能冒称为完整批。

## 辅助限制

- Ncu侧检在CUDA初始化Error36失败，0 kernel、物理DRAM读写为null。它是可选硬件证据不可用，不是模型推理0题/准确率0，也不阻断其余通过项；没有伪造硬件数据或重试GPU。
- KVarN为显式materialized机制移植质量参考，不能用于该路径的packed serving速度/显存优势主张。逻辑tensor读取不等同物理HBM流量。
- Teacher-forced quiet timing只计measurement trials与相同token日程，不算应用质量；warmup、allocator audit与测量分开。
- Doc/GQA grouped协议、GQA大小写格式瓶颈、ConvBench参考内部矛盾/长度上限、单模型/有限轮数、历史曝光未完全审计等限制仍保留。
- 当前完整模型身份与已登记metadata/index一致，不反向证明所有更早运行期间权重/包版本从未变化。数值finite核查限已保存主raw字段，不声称检查未保存GPU隐状态。

## 冻结验收文件

- `audit/FINAL_INDEPENDENT_VALIDATION.json`：完整validator输出；findings仅上述16条排除。
- `audit/final_numeric_validation.json`：有限数值与raw SHA。
- `audit/final_model_process_validation.json`：模型stat与GPU进程终检。
- `audit/inventory_source_resolution.json`：27处标记的精确来源解析；原inventory备份保留。
- 本简报。输出验收器的精确版本按FINAL JSON的`validator_sha256`保存于content-hash目录，后续分析代码的元数据标签改进不被冒称原版。

本验收之后不再追加实验、参数、重新评分或改写统一报告；最终打包只需引用冻结文件。
