# 开发集 margin 诊断与 V2 保真收益观察

本次只分析既有 8 张开发图像，没有读取或调节最终测试运行。margin 来源为自由生成时记录的低精度试算 logits；答案独立使用选定开发 session 的 CLEVR 参考评分。各方法保留自己的历史，因此这里描述应用层关联；没有将后续分歧 token 按位置强行配对。

## 1. 第一生成 token 的 margin 能识别部分一般错误，仍有漏检

“首内容 token”在无 tokenizer 解码的诊断中明确操作化为首个不属于显式 termination ID 的生成 token 候选。它可能只是词的第一个 subword，并不保证已经承载完整事实。阈值固定为 0.1、0.5、1、2、4、8，触发规则为 `margin < threshold`。每个比率和 AUROC 的区间按原图像重采样。

### Base2

后续 T2–T10 共 72 题，31 个规范化 exact 错误。

- 一般答案错误 AUROC = 0.7592，95% image-cluster CI [0.6373, 0.8590]。
- 阈值 2：触发 28/72，命中 20/31 个错误；precision 71.43%，recall 64.52%，仍漏掉 11/31 = 35.48% 的错误。
- 阈值 4：仍有 8/31 = 25.81% 的错误处于阈值以上。
- 阈值 8：仍有 3/31 = 9.68% 的错误处于阈值以上。
- 相对于同题 Full 没有 Full-correct/Base2-wrong 负翻转，因此压缩负翻转 AUROC 与 recall 为 N/A。这组数据只能支持 margin 与一般任务错误的关联。

### Base1

后续 72 题全部错误；其中 1 题没有非终止生成 token，可用首 token margin 为 71 题。一般错误只有一个类别，AUROC 为 N/A。Precision=1 来自样本全错，不能解释成区分能力强。

相对 Full 的后续负翻转为 38 题，其中 37 题具有 margin。阈值 2 触发 62 个可观测回答，命中 32 个负翻转：可观测条件 recall 为 32/37=86.49%；保留空回答漏检后，operational recall 为 32/38=84.21%；precision 为 32/62=51.61%。可观测负翻转 AUROC = 0.4869，95% CI [0.3806, 0.5901]。

全部阈值、trigger rate、precision/recall、错误高 margin 比例、各自 CI 与缺失 margin 计数位于 `results/analysis_margin_dev8_v1/`。Base1 和 Base2 没有混合成一个分类样本。与 Full 配对的 initial-logit hash 检查没有发现不一致。

## 2. V2 的当前与未来排序使用同一 KL 保真目标

V2 的参照是完整 FP16 视觉读取历史在同一 canonical base token 串上的轨迹。正收益定义为：

`KL(p_base || p_ref) - KL(p_action || p_ref)`。

因此正值表示更接近同 token 参考分布；它不奖励 base token 本身，也不直接等价于答案答对。该参照只用于离线标定，在线方法读取冻结收益表。

8 个开发图像、192 个写入干预、12 类动作的当前/未来收益排序 Spearman 为 **0.2634**。当前正未来负有 19 个干预，当前负未来正有 17 个。

| 动作 | 当前平均保真收益 | 未来平均保真收益 | 当前排名 → 未来排名 |
|---|---:|---:|---:|
| layer0:text_prefill | +0.013029 | +0.023103 | 2 → 1 |
| layer1:text_prefill | +0.014576 | +0.013150 | 1 → 2 |
| layer2:text_prefill | −0.020631 | +0.007708 | 11 → 3 |
| layer3:text_prefill | −0.022628 | +0.000240 | 12 → 6 |

Layer0 和 layer1 prefill 的未来均值 95% 图像聚类区间分别为 [0.002754, 0.057852]、[0.002179, 0.029934]。Layer2 prefill 的未来区间为 [-0.030458, 0.062642]，体现目前样本中的不确定性；仍可用其排序反转作为下一轮原型的具体探索案例。

V2 的散点单独输出到 `figures/mechanisms_dev_v2/mechanism_development_fidelity_v2.*`，没有与 V1 的 base-target logp 散点混成同一指标。完整数据、每动作 CI 与局部反例位于 `results/analysis_mechanisms_dev_v2/`。

## 3. 对下一轮的启发

保留 margin 作为便宜信号，并研究它漏掉的高置信错误，以及中层状态/读出差异能否提供补充信息。Base1 的强退化样本提示，单看“低 margin 触发后 precision 高”会受到错误基率影响。V2 的具体动作反转提供了联合比较当前信号与未来保真收益的实测起点。
