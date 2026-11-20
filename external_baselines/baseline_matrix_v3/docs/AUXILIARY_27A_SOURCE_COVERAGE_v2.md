# 辅助证据闭包补充 v2

`audit/auxiliary_27a_source_coverage_v2.json` 在 v1 的 27a index/ZIP 成员核查和14组原执行源证明之上，追加20个 raw JSON/JSONL/CSV 的独立远端 SHA 与大小重核；全部与现存 canonical 镜像一致。原 v1 清单、所有原始报告和 source receipts 都保持不变。

统一 raw receipt 位于 E 盘 `results/auxiliary_source_proofs_v1/raw_remote_rehash_receipt_v2.json`。各组原 launch/job/source inventory/driver/声明源码位于同目录的 `<job>/` 子目录。公开发布可按 v2 `groups[].artifacts` 精确选文件，另附统一 raw receipt；私有路径需要公开派生模板与独立指纹，不能把 raw/source SHA 改写成公开正文 SHA。

27a 已覆盖12/20个 raw；缺失的是 fixed-q CPU60 的3个文件、late-boundary 的2个文件、OnlyTerp-prod CUDA gate 1个文件、prod probe 2个文件。完整逐文件覆盖与14组类型/计数见 v1表格及v2 JSON。CPU60、CPU数值控制均不新增GPU推理；CUDA算子门与诊断分支不加入正式质量问数。没有读取GT，没有下载PT、原图、权重，没有运行GPU。
