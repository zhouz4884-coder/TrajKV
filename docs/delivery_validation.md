# 独立交付核验器

入口：`code/analysis/verify_delivery.py`。默认输出 `audit/delivery_validation.json`。它只读取证据并写核验报告，不删结果、不改 runner、不启动或停止实验。

```powershell
python code/analysis/verify_delivery.py --remote-sources
python -m unittest discover -s code/analysis/tests -p test_verify_delivery.py
```

`--remote-sources` 仅通过现有授权 SSH 路线读取本地尚未下载的 run/source_snapshot 文件；不访问 GPU、不读环境变量或凭据内容。离线核验可省略该参数，此时缺历史源码标 PENDING。`--strict-integrity` 只让已证实的 ERROR 返回非零；PENDING 和资源超限的记录排除不触发研究停止。

核验范围：

- run 的 manifest、complete receipt、预期 session/method/policy 组合及重复记录。
- manifest 声明的源码 hash 优先与该 run 的归档比对，不能拿后来修改的当前文件推断旧源码错误。
- 所有本地图像 SHA256，冻结 final64/replication32 的文件 hash、ID 集及与全部 dev/smoke/pool 的图像交集。
- gain table 的训练 session/source image 与冻结评测隔离，及校准原始输入 hash。
- 从 read_events 独立重算额外逻辑读取，包含 replay/repair 再读 base/text；超预算记录保留并明确排除等预算主张。
- summary 与自己的 scored/resource CSV 计数，以及输入仍匹配时与原始输出的条数和逐题 prediction 一致。
- 当前模型完整 shard 身份，与历史 run 中已登记的 metadata/index hash 比对。

核验器递归识别 composite suite 与 budget sweep；父 receipt/冻结 plan/子 receipt hash 分开检查，父容器不额外算一次模型运行。Sweep 另核对子预算、方法、策略、cohort 合同。不可把只有父 receipt 的容器解释成零条模型结果。

Teacher-forced quiet timing 单独计量 measurement trials，暖身不进入计数或应用质量。检查固定 token 身份、quiet 标记、嵌套 runner 账本和 timing summary，不将同一强制文本的 answer 当作真实应用正确率。计数器侧检属于 `hardware_counter_probe`；Ncu 的0 kernel / CUDA-init失败会保留为硬件证据不可用，`model_inference_records=null`，不会冒充模型推理了0题或准确率为0。

运行时白名单另存 `audit/runtime_identity.json`：仅 Python、torch/transformers/triton/accelerate/safetensors/PIL/numpy、编译 CUDA/cuDNN、GPU driver。没有 pip freeze、全环境枚举或环境变量转储；CUDA context 初始化状态查询前后均为 false。

状态含义：`ERROR` 是已证实的完整性矛盾；`EXCLUSION` 是不能进入某类主张的真实记录；`WARNING` 包括保留的旧分析快照；`PENDING` 包括仍在运行、未下载完或缺历史依赖快照。未完成不能被解释成失败。

历史 partial summary 若指向后来增长的同一 records 文件，输入 hash 会过时；核验器把它标为历史快照，当前结论应重新分析。旧 write16-v1 的 repair_once 预算违规原始记录不会被删除或改写。

缺失源码可在 `audit/recovered-by-content-hash/<expected-sha256>/<filename>` 保留显式恢复对象。核验器再次计算完整 hash 才采纳，并将来源标为 `RECOVERED_BY_EXACT_CONTENT_HASH`；不会把它冒称为原 run 当时已归档的文件。`audit/source_recovery.json` 记录本轮三处缺口的两份精确对象、原始来源和可复算 diff；原 manifest 未修改。

模型身份文件 `audit/model_identity.json` 由授权远端只读全文件 SHA256 得到：配置/tokenizer 与 index 所列全部四个 safetensors。哈希期间检查 size/mtime 稳定，并以低优先级和读速节流运行，不占 GPU。当前完整权重 hash 不能反向证明旧实验期间文件从未变化；旧 manifest 仅锁定配置/index 的局限单独保留。
