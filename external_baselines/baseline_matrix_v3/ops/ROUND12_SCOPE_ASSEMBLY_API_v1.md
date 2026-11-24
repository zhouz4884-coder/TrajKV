# Round12实际公开scope组装

本工具只读原始结果并写入新的E盘派生目录，不启动GPU、修改raw、封口或访问GitHub。它覆盖1200次开发自由回答、2次排除计分的协议pilot、选中R在Info/Doc各500题，以及兼容的旧FP16/R7/SOTA原始证据。旧固定500已被观察，不能称独立确认。

## 运行中准备

```text
python -X utf8 -B ops/round12_collect_producer_evidence_v4.py
python -X utf8 -B ops/round12_assemble_public_scope_v1.py --output ${EVIDENCE_ROOT}/results/round12/NEW_DRAFT --draft --check-closed-bindings
```

collector仅使用既有授权连接helper；新v4增加有界SFTP预取和已哈希本地对象复用，独立远端SHA仍在传输前后核对。`--job bm_R12_...`可选定已知job。旧collector/runtime/scorer不修改。

draft记录缺失cell/尚未闭合parent，写`SCOPE_DRAFT_NOT_SEALABLE.json`。可选closed-binding检查只验证已闭合producer子集，明确`actual_stage_sealed=false`、`publication_eligible=false`。它不是阶段完成或发布。

## 实际阶段结束后

等待Info/Doc完整500官方计分、父队列complete、Root自然语言报告定稿，再执行新的producer刷新与组装：

```text
python -X utf8 -B ops/round12_collect_producer_evidence_v4.py
python -X utf8 -B ops/round12_assemble_public_scope_v1.py --report ${EVIDENCE_ROOT}/results/round12/REPORT_ROUND12_FINAL_v1.zh.md --output ${EVIDENCE_ROOT}/results/round12/NEW_FINAL_SCOPE
```

成功输出`R12_SCOPE_UNSEALED.json`，stage为`READY_FOR_ROOT_SEAL`。仍需Root调用既有seal/export/independent-cold，以及publisher的实际上传和远端核验。任何blocker均拒绝生成可封口结果。

默认报告source位于上述E盘；公开相对路径固定为`docs/REPORT_ROUND12_FINAL_v1.zh.md`，不依赖D盘保存报告或大派生材料。

统一`summary.json`、`quality_complete_bytes.csv`、`paired_comparisons.csv`由全部声明的raw+refs重新计算。DEV/ORACLE/FIXED500使用不同label，不能合并均值。严格核账2200生产自由回答、2pilot；探针token、校准/TF、旧参考行不计新回答。若Root正式改变阶段执行范围，需要新版本修改明确预期核账，不能通过忽略失败/未运行项达标。

旧producer保留原`source_retrieval_receipt`、`retrieval_receipt`及其全部选定原文件。旧派生500视图保留provenance、逐行原source exact SHA与完整原始producer；不虚构另一次GPU运行。早期native外部端口保留其原始entry-source+完整hash inventory证据范围，不宣称重放未归档的所有依赖。

原生FP16的实际56个K/V分配可按同一完整allocation中的不相交视觉/文本区间记账；压缩bank仍按唯一tensor storage逐项计费。此适配仅用于明确标记的历史native参考。

`check_round12_legacy_public_v1.py`提供真实历史source/derived-view准入及篡改拒绝的CPU检查，0新GPU/模型/网络调用。公开包始终排除原图、模型权重、私聊、认证及连接helper和PT/logits二进制。
