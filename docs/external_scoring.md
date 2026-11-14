# 外部数据离线评分口径

核验：2026-11-12。入口 `code/analysis/analyze_external.py`，不加载模型，不调用外部评审 API。答案只在生成完成后的离线分析阶段读取。

## DocVQA：锁定组织者代码口径并暴露边界差异

[DocVQA 2020 官方说明](https://www.docvqa.org/challenges/2020)规定 ANLS，不区分大小写、区分空格。本实现的主列逐项适配组织者 Rubèn Pérez Tito 的 [MP-DocVQA evaluator](https://raw.githubusercontent.com/rubenpt91/MP-DocVQA-Framework/7939f521fe5a82a442f54831b1c0e96daab331d9/metrics.py)：

- commit `7939f521fe5a82a442f54831b1c0e96daab331d9`，文件 SHA256 `481fe17e8d443a6dcf852b8348fcf06b7973ac1e99336f230a749008e992bf70`。
- 小写、去首尾空白；保留内部空格、标点和数字写法。
- 字符编辑距离除以两个字符串的最大长度，多参考取最大 similarity。
- **主列保留 similarity ≥ 0.5；空 prediction 得 0。** 保留组织者代码对 answerable 项的 `none` 特例。

原始 ANLS 论文体系的公式写 normalized distance < 0.5，即 similarity > 0.5；该边界与上述组织者代码不完全相同。因此同时输出 `anls_strict_threshold_aux` 和边界命中数，不混用两个口径。该辅助列只替换阈值比较符，保持主列的标准化和空答案规则。旧 P173 的“空空=1”也不符合这份组织者代码。

RRC 页面本次访问超时，未核验服务端 evaluator。输出明确称 `organizer-code ANLS`，不声称与 RRC 服务端逐字相同，也不使用 DocVQA 2026 新任务的特殊日期/数字格式规则。当前适配仅支持原单页 DocVQA 的 answerable string references，不扩展 not-answerable 类型。

## GQA：原始短答案匹配

[Stanford 官方 evaluation 页面](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)定义预测答案与标准答案相同计 1，否则计 0。主列使用原始字符串 exact identity；不删除标点、不截取答案、不映射数字或 CLEVR 同义词。另报 `case_whitespace_exact_aux` 仅用于格式诊断，不替换主结果。

这里评测的是已冻结的 grouped 子集及多轮协议，不是官方 balanced-test leaderboard；没有生成 consistency、grounding、plausibility 等完整 GQA 指标。

## ConvBench

仅输出清楚标记的词频 multiset token-F1 诊断，`official_quality_metric=false`。它不能判断语义、幻觉、指代或创造性答案是否正确，不作为官方质量或方法门的依据。

## 统计与配对

方法比较按同模型、执行器、数据协议和 generation settings 分组，按 session+reference turn 配对；置信区间重采样 source image，保留同图多个对话/轮次的聚类。报告 all/later turns、逐轮结果、连续分数增减以及 exact 追回/新增错。不把 DocVQA 的软分数翻转冒称二值正确率。

仅自由生成记录可进入应用质量；fixed-token、`quality_scoring_allowed=false` 被显式排除。缺参考、重复 run/turn、配对图像不一致均报错。部分记录集带 `partial` 标记；区间属于开发阶段探索性分析，未做多重比较校正，不据此给显著性结论。

```powershell
python code/analysis/analyze_external.py --dataset docvqa --records results/<run>/records.jsonl --references configs/data/docvqa_references.jsonl --samples configs/data/docvqa_dev32_runner.json --output results/<analysis>
python code/analysis/analyze_external.py --dataset gqa --records results/<run>/records.jsonl --references configs/data/gqa_references.jsonl --samples configs/data/gqa_dev32_runner.json --output results/<analysis>
python -m unittest discover -s code/analysis/tests -p test_external.py
```

生成 `summary.json`、逐题、逐轮、方法汇总、配对差和翻转 CSV；所有输入与脚本均记录 SHA256。所有分数使用 0–1 单位。
