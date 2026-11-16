# Round7 Doc128 确认集对齐与高分核查

结论：全量核对没有发现图像、问题、答案或评分对齐错误，也没有发现把 GT 拼入 runner 输入的路径。Full 的 98.6169 ANLS 可由保存的真实回答和官方参考独立重算得到；它不是 98.6% 的题目严格完全正确。该批数据来自固定 DocVQA **train** 镜像分片，是本项目未用来源的确认，不是 backbone-unseen 或官方 validation/test 基准成绩。

## 全量对齐

使用 `lmms-lab-encoder/DocVQA` 的固定 revision `539088ef8a8ada01ac8e2e6d4e372586748a265e`，核对保留的两个 parquet 原文件，SHA 与冻结 LFS 身份相同。审查覆盖全部 **128 来源、128 原图、412 问**，没有只抽取高分题。

按冻结 pool 的 `source_page_id` 和 `questionId` 定位后，412 问的 `ucsf_document_id`、原始 question、answers 列表及其顺序、question_types 均与 runner/ref 文件精确一致。每道选中问题所在 parquet 的原始 image bytes SHA 与 pool 相同，128 个本地 staged 原图也与其相同。原图未因这次核查重编码或修改。

runner 问题严格等于 parquet 原文加固定后缀 `Answer the question using a single word or phrase.`，没有附加答案。样本没有 answer/reference/GT 字段或路径。两个已完成执行分片的 config 也没有此类字段或路径，manifest 样本 SHA 一致；实际记录全部为 `image_boundary_free_running`。已记录的关键 runner/session 源码 SHA 与保留代码相同；生成入口没有传入 forced 目标。既有 transport 审计记载确认 reference 文件没有上传。

此外，对候选池冻结时保存的旧 registry 项目重新汇总来源和图像内容，没有发现确认来源或内容与该旧登记集重叠。这里重放的是当时的登记清单，不使用现已包含确认集本身的动态 registry。

## 独立评分

核查脚本没有导入生产 `score_docvqa`。它用独立的完整二维动态规划矩阵计算字符编辑距离，采用相同声明口径：仅 lowercase 与首尾 strip；取全部官方参考的最大归一化相似度；相似度至少 0.5 才计分；空预测和 `none` 预测为零。没有删内部空格、标点或数字字符，也没有数字语义容错。

对九个方法的 **3,708 个实际自由回答**，独立评分逐题匹配 preview 的 prediction、官方参考、score 和 exact；各方法均值也匹配。Full 的分解为：

| Full 描述 | 数值 |
|---|---:|
| ANLS | 98.6169% |
| 规范化 exact match | 92.9612%（383/412） |
| 部分得分 | 27 题 |
| 零分 | 2 题 |
| 第一次提问 ANLS | 98.0406%（128 问） |
| 第二/三/四次提问 ANLS | 98.9730 / 98.4587 / 99.2322% |

部分分数符合 ANLS 的字符定义。例如 `4873` 对参考 `4373` 得 0.75，`6000` 对 `6,000` 得 0.8。数值字符串有实质错误仍可能获得较高字符相似度，这是已声明指标的语义，不是本轮评分脚本把错误答案改成正确答案。首问已经很高，也说明高分不能全部解释为后续会话历史帮助。

## 高分的解释边界

该便利子集首先按资源元数据选择两个最小 payload 的 train 分片，再排除旧来源/内容、按固定来源 hash 选页，并取最小的至多四个题号；这些规则在读取答案和模型结果之前冻结。它不是从完整 DocVQA validation/test 随机抽样，不能把这 128 来源的点估计推广为标准 DocVQA 成绩。

本批问题标签中，form 有 223/412（54.1%），layout 有 168/412（40.8%），table/list 有 60/412（14.6%）；标签可重叠。保存结果表明，这个 train 分片组合上的 Full 回答本身接近 ANLS 天花板。train 来源和题型组成提供了合理的适用范围解释，但本次对齐审计不能因果证明高分来自 backbone 训练记忆，也不能排除其训练暴露。因此保留 `project-fresh train source holdout` 的表述，不写 backbone-unseen。

本审查没有按结果删样、换集、改变 reference、改变方法或覆盖预测。确认结果应按冻结范围照实报告，同时保留 train 来源、样本构造和 ANLS/exact 的区别。

审计：`audit/a2/round7_doc_confirmation_alignment_v1.json`。可复核命令：

```text
python code/a2/round7_doc_confirmation_alignment.py
```

脚本只执行本地 CPU 读取与评分；审计末尾重新核对原输入和原记录 SHA，确认未修改。
