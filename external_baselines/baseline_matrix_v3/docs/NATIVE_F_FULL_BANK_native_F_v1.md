# M1原生FP16正式参考银行

四个固定公开split已按完整cohort关闭并评分。每条预测都来自原始真实CUDA问答，重用行与新推理行均保留逐行来源；该银行的派生组合没有新增GPU回答。

|数据集|全split题数|指标|质量 %|fresh / reused|
|---|---:|---|---:|---|
|textvqa|5000|textvqa_mmf_ten_vote_leave_one_out_v1|81.9400|4500 / 500|
|infovqa|2801|infovqa_anls_qwenvl_vendor_rawlen_ge0p5_v1|72.1018|2301 / 500|
|docvqa|5349|docvqa_anls_qwenvl_vendor_rawlen_ge0p5_v1|91.4037|4849 / 500|
|chartqa|2500|chartqa_pix2struct_relaxed_5pct_v1|86.2400|2000 / 500|

ChartQA human/aug分别保存在summary的groups中。Doc/Info指标采用已锁定vendor ANLS及独立诊断版本，未将其冒称今日RRC服务器代码。各数据集指标不同，不合并成单一平均质量。
所有输入、模型、processor、提示、生成上限及视觉前缀边界保持相同合同。仅可比的质量预测可重用；速度仍需同环境交错重跑。
本银行不解锁压缩方法full或本方O/P4。公开同步由Root完成并核验，本文档产生时未声称已经上传。
