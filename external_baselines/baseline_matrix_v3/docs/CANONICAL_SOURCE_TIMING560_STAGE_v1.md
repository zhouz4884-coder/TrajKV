# Canonical source-profile timing: completed560 trials

五个预先固定的M1源配置 F/K32/C2/E64/OnlyTerp-Hmix23 在Text16与Info8正式cohort及各自两题硬件smoke上，共完成560个真实CUDA trials。E统一为source-default W64/drop0.888；配置与固定的四域full-target v3逐项对齐。

计数为Text2 smoke40 + Text16 formal320 + Info2 smoke40 + Info8 formal160；每QA/方法warm1、repeat3，合计140 warmup与420 measured。560 trials不等于560独立QA，不增加质量QA。旧E8 smoke40、已superseded的旧520计划、legacy656与Calib32均不计入。

两次独立CPU完整验收使用同一冻结校验器，四节点raw/token/context/owner/repeat审计、14个原始预测SourceRun、140条expected记录的原行SHA/有效合同/输出token/上下文/bytes绑定全部一致。额外执行依赖均由独立源证明覆盖。三个大donor通过精确既有P2固定500视图及独立评分的辅助闭包进入归档，原JSONL不切分；1500辅助view QA复用既有GPU预测，不代表新增推理或P2完成。

以下为正式cohort的分域观测：TTFT为host median，ITL为自然输出token间隔的pooled mean，完整owner bytes为每请求mean。原始字节值和smoke结果见随包domain_timing.csv。

| dataset / formal QA per method | source profile | natural tokens mean | TTFT host ms | natural ITL host ms | complete owner mean bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| textvqa / 16 | F | 4.375 | 620.044 | 38.799 | 97255424 |
| textvqa / 16 | K32 | 4.438 | 683.823 | 48.150 | 19374336 |
| textvqa / 16 | C2 | 4.312 | 912.103 | 84.749 | 12285632 |
| textvqa / 16 | E64 | 4.375 | 685.342 | 54.428 | 10864408 |
| textvqa / 16 | OnlyTerp-Hmix23 | 4 | 2173.159 | 802.484 | 18279312 |
| infovqa / 8 | F | 4.75 | 681.266 | 39.150 | 97255424 |
| infovqa / 8 | K32 | 4.75 | 734.404 | 48.872 | 19374336 |
| infovqa / 8 | C2 | 4.375 | 1042.004 | 86.420 | 12285632 |
| infovqa / 8 | E64 | 5 | 731.571 | 53.525 | 10860768 |
| infovqa / 8 | OnlyTerp-Hmix23 | 4.5 | 2208.226 | 783.008 | 18279312 |

在这两个正式cohort中，四个压缩源配置的完整visual-owner bytes均低于F，但其median TTFT和pooled自然ITL均高于F。当前移植prototype因此给出了bytes节省，并未在这两个观测量上出现低于F的数值。

所有trial均保留host_quiet=false；每trial的GPU compute独占检查不等于整机安静。请求使用预先读入RAM的原始编码图像字节，磁盘读取在计时外，图像解码/预处理在每次请求内重做。GPU stream interval不是纯kernel时间。自然答案长度与内容随方法变化，因此不按较短答案wall时间排名，不给跨域平均加速比，也不将prototype耗时冒称原论文优化kernel速度。

阶段完成仅指canonical560实测及来源闭包。既有native F四官方full split被引用，未重跑模型；P2完整114格与canonical16个压缩full split仍分别待完成，本阶段不为其填写预测分数或缓存命中。归档原件、公衍生文件和Git LF正文SHA分别记录；公开发布须由最终交付收据绑定实际commit/tree、asset SHA及固定content_ref。
