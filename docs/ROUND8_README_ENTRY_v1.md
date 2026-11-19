## Round8：分布拟合改善，任务与成本目标未达成

研究执行已完成。固定 336 参数的后残差校准使训练 KL 下降 75.03%，训练来源外八个已暴露选择来源的 Full-self KL 下降 33.22%；对应 teacher-winner flip 汇总率未降低。

| 新来源确认 | R7 % | learned R42 % |
|---|---:|---:|
| Doc 官方 TRAIN 来源 holdout64，ANLS | 99.2299 | 98.6273 |
| Text 官方 validation128，十票官方准确率 | 76.3281 | 75.6250 |
| InfoVQA 官方 validation64，ANLS | 72.2325 | 71.8419 |

三个来源池共 256 个首问、四方法、1,024 条真实回答，分别评分。新方法均值均略低于 R7，配对区间均跨零；三来源固定八 token quiet 比较中，整轮 wall 中位数增加约 3.4%，编码增加 44.30%。本轮未建立稳健的质量—成本前沿提升。

全轮去重保留 5,714 条有效模型答案，失败 Mean v1 的 120 条 records、训练、诊断和 CPU fixtures 不计入该数。方法及 40 个输入/协议文件在新确认答案值解码前冻结。研究执行完成与性能目标成功分别记录，发布完成状态以远端交付回执核验。

详见[中文研究报告](docs/REPORT_ROUND8_2026-11-18.md)、[三域确认](results/round8_v1/confirmation_firstq4_v1_summary_v1.json)、[受控计时](results/round8_v1/final_quiet6_v1/summary.json)与[执行回执](audit/a2/round8_stage_execution_complete_v1.json)。下一阶段的密度交换及读出导向码本属于未验证建议。
