# Round7 数据与独立诊断准备

本阶段未启动 GPU。所有选择依据来源身份、固定盐哈希、题目 ID 和原始图像字节；未使用模型结果或答案筛选来源。旧结果、旧来源清单与旧 launcher/session 源文件保持不变。

## 校准与选择集

|子集|来源数|独立首问数|用途|
|---|---:|---:|---|
|`doc_train16_independent`|16|45|新拟合的 Doc 校准|
|`doc_select16_independent`|16|38|与本次拟合来源分离的选择/开发|
|`text_train32_independent`|32|52|新增 Text 校准|
|`text_train16_independent`|16|25|上述 train32 的预先固定嵌套资源控制|
|`text_select16_independent`|16|22|新增 Text 选择/开发|

样本位于 `configs/a2/round7_v1/<子集>.json`。对应评分/校准答案位于 `configs/data/round7_<子集>_references.jsonl`（Doc）或 `.json`（Text）。每个独立样本仍是单问、单图、`turns=1`，多问以同一来源多条独立样本表达，不借用前问答案。

Doc 从现有 `next_cal32_independent_questions.json` 的 32 来源、83 问按 `trajkv-round7-doc-cal32-source-split-v1:` 哈希划分 16/16。Doc 选择集对本次新拟合保持来源分离，但这 32 来源此前属于历史校准集，不能称为新盲确认。

Text 从官方标准 5000 问 runner 中按已使用与保留来源、已编码图像内容哈希的 union 排除后选择。首次选择时剩余 1949 个合格来源，按 `trajkv-round7-text-cal48-source-split-v1:` 固定排序，前 32 为训练、后 16 为选择，每源最多前三个数字题目 ID。实际这些来源每源最多两问：train32 有 20 源两问、12 源一问；select16 有 6 源两问、10 源一问。校准目标应先按来源平均，再按数据集等权，不能把多问或多次干预当作独立来源。

每个子集另有 `<子集前缀>_first_questions.json`，各来源只取首问，可直接用于原 Depth 的 16 图或 32 图同格式控制。Doc 独立样本的实际题目 ID 应使用 `calibration_question_index`（1-based）对应原 `source_question_ids`，或使用其独立 reference 的 `question_id`。

## 连续与长会话诊断配额

`doc_select16_continuous.json` 使用同一 16 个选择来源的 38 问，按每来源已有独立问的固定顺序组成连续会话。`doc_stress8_turn12.json` 固定取选择集哈希顺序的前 8 来源，每来源循环已有问题至 12 轮，共 96 轮。后者用于检查长会话累计误差、位置处理与读账本，不增加独立数据量，也不构成新确认集。两者 references 单独保存在本地，可与同源独立问进行配对比较。

## 新确认候选

旧验证集 938 个来源均已使用或保留。新的 Doc 确认候选来自同一公开仓库 `lmms-lab-encoder/DocVQA` 的固定 revision `539088ef8a8ada01ac8e2e6d4e372586748a265e`，另下载两份最小训练分片，总计 188,237,923 字节，两个原始文件均与官方 LFS SHA256 相同。下载及 partial 均位于项目数据盘。

两份新分片按旧来源与旧内容 union 排除后得到 1055 个合格来源。按 `trajkv-round7-doc-extra-confirm128-source-v1:` 来源哈希、独立页哈希和数字题目 ID 固定 128 个来源、128 张唯一图，每源最多 4 问，共 412 问。原始图像 11,124,190 字节已经提取至 `data/docvqa_round7_confirm128/images`；身份冻结和图像准备没有读取问题文本或答案。上游数据 split 是 train，所以只能称本项目新来源 holdout，不能保证骨干预训练未见过。

Text 另按 `trajkv-round7-text-confirm128-source-v1:` 预留 128 个新来源，与旧 union、新 train32、新 select16 的来源和内容均无重叠。每来源固定最小数字题目 ID，共 128 问、38,311,045 原图字节。未来确认答案尚未物化。

方法冻结后，由 root 运行：

```powershell
python code/a2/round7_data_stage_doc.py --stage-questions --method-freeze <freeze.json>
python code/a2/round7_data_text_confirm.py --stage --method-freeze <freeze.json>
```

这会生成新的 `doc_confirm128.json`、`text_confirm128.json`、本地 refs 和冻结 SHA 收据。原 Doc54、Text256 与原冻结 refs 不会被覆盖。只上传确认推理 runner 与原图，确认 refs 留本地评分。

## 第二骨干首问适配

`round7_second_smoke.py` 与 `second_llava16_native_full4.json` 只比较真实 FP16 LLaVA-1.6 的 native HF 和 Full bank，固定 4 张 Doc 选择来源图，首问自由生成。官方模板固定为 `llava-hf/llava-v1.6-vicuna-7b-hf@c916e6cdcd760b4cecd1dd4907f84ac649f93b23/chat_template.json`，SHA256 为 `2eabebdd61f13f04c29ce6c6ebf70e76ea49b7503fa7f0ac36ccbb4d1023ed6e`；仅在 processor 没有模板时填入。只读 SSH 核验显示远端已经有同 SHA 的模板。

远端模型 config SHA 与旧 inventory 一致，`model_type=llava_next`、`quantization_config=null`、`torch_dtype=float16`，三权重文件共 14,126,954,312 字节。该复核只检查配置、索引及文件长度，没有重新计算完整权重 payload SHA。实际视觉 token 数及模 32 余数由 smoke 记录，禁止 padding，不预先假定能直接运行 packed codec。

## 复核与交付入口

`round7_data_verify.py` 已通过：336 张唯一原图 SHA、所有新开发样本与 refs 的 ID/轮次/题目对应、训练与选择的来源/内容分离、开发与确认来源/内容分离。没有计算任务分数，也没有打开未来确认 refs。

关键收据：

- `audit/a2/round7_development_selection_v1.json`：先于 selected-label 物化的身份冻结。
- `audit/a2/round7_development_membership_v1.json`：所有新开发 runner、refs 的原始字节 SHA。
- `audit/a2/round7_development_transport_plan_v1.json`：80 张开发原图、runner、训练标签及强骨干代码/配置的 local/remote/SHA；其中 upload 仅是待传计划，尚不代表传输完成。
- `audit/a2/round7_doc_confirm128_candidate_pool_v1.json` 与 `round7_doc_confirm128_image_assets_v1.json`：新 Doc 身份及原图。
- `audit/a2/round7_text_confirm128_membership_v1.json`：Text 新确认预留。
- `audit/a2/round7_data_independent_check_v1.json`：独立字节与对齐检查 PASS。
- `audit/a2/round7_existing_assets_readonly_v1.json`、`round7_llava16_official_template_provenance_v1.json`：实时远端资产与官方模板来源。

阶段的 GPU 调度、推理、最终方法冻结、传输收据和公开 GitHub 增量同步由 root 统一完成。本文件只报告已完成的数据准备与静态适配，不能被解释为实验或上传已完成。
