# 四数据源冻结 registry 与原图入口

状态：`PASS_ALL_FOUR_OFFICIAL_SPLITS_RAW_IMAGES_AND_SEPARATE_REFERENCES_READY`。本阶段仅准备数据，没有模型评分、训练、GPU 作业或 GitHub 写入。

| 数据集 / split | 原始问题 | 原图 encoded SHA 数 | source cluster | P1 | P2 |
|---|---:|---:|---:|---:|---:|
| TextVQA validation | 5000 | 3166 | 3166 image_id | 128源首问 | 500源首问 |
| DocVQA validation | 5349 | 1285 | 938 ucsf_document_id（PDF） | 128源首问 | 500源首问 |
| InfoVQA validation | 2801 | 500 | 500 image.path | 128源首问 | 500源首问 |
| ChartQA test | 2500 | 1509 | 1509 imgname | human64 + augmented64 | human250 + augmented250，500个不同源 |

总计15,650个官方 split 问题、6,460个不同原始 encoded 图像；原图逻辑字节总量2,019,302,688。原始旧图片仅在 SHA 严格匹配时 hardlink，不修改旧文件字节。全量 staged 后各预定子集的全部图片都已存在并验证。

## 冻结来源

`configs/datasets/v3/source_lock.json` 的 SHA256 为 `d3d48f16e788ed7e62123bcdfad0d16a6e182215e2814792a1014db5b55d1f2f`，包括每个 shard 的完整 LFS SHA、文件尺寸、冻结镜像 revision、所有输出文件 SHA 与抽样规则。

- [TextVQA 冻结公开镜像](https://huggingface.co/datasets/lmms-lab-encoder/textvqa/tree/9c0699cd19768ac5ab97568f6b3cbac4c0062884)：3个 validation shards，完整 SHA 全部匹配。
- [DocVQA / InfographicVQA 冻结公开镜像](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/tree/539088ef8a8ada01ac8e2e6d4e372586748a265e)：DocVQA validation6个、InfographicVQA validation4个 shards，完整 SHA 全部匹配。
- [ChartQA 冻结来源](https://huggingface.co/datasets/ahmed-masry/ChartQA/tree/af8b6f5c08c95085271561c2a3f9d15f2b5a9031)：test1个 shard，68,916,402 bytes，SHA256 `84889bae30aaf6e24ff899e1604221606c0b65e658b6c70a7d02c79f50fc87df`。

前三源使用现有只读 Parquet，没有复制或修改旧 source。ChartQA 下载到 E 数据盘 mirror，先完整核 SHA 再解析；直连失败与配置网络下载成功均有独立身份记录，不保存重定向签名 URL。

## 接口与数据分离

本地小元数据在 `configs/datasets/v3/`。远端以项目 `BROOT` 为根：

```
BROOT/data/registry/v3/{textvqa|docvqa|infovqa|chartqa}_{p1|p2|full}_inputs.jsonl
BROOT/data/refs/v3/{textvqa|docvqa|infovqa|chartqa}_{p1|p2|full}_refs.jsonl
BROOT/data/registry/v3/source_lock.json
BROOT/data/registry/v3/STAGE_IDENTITY.json
BROOT/data/images/<original_encoded_sha256>.<original_format>
```

`code/dataset_registry.py` 提供 `load_inputs(path)`、`load_references(path)`、`select_subset(rows,stage)`。输入行含全局 id、数据集 revision/split、source/cluster/question 身份、remote image_path、原 encoded SHA、宽高、原问题和原 split 行号。模型输入不含 answers、OCR、reference 或 label 字段；loader 对混入字段拒绝。refs 单独按 id join，TextVQA 保留原始10票及其重复；Doc/Info 保留全部原始答案变体；Chart 保留原 label 字符串。

Chart 的原始 type 实际为 `human`1250题、`augmented`1250题。两种字符串原样保留在输入和 refs，评分库显式将 augmented 映射到 aug。没有因为问题重复而删行；没有原 question ID 的 Chart 用原 test 行号作为身份。

图像保留原 encoded bytes，没有 RGB 重编码、JPEG 转换、调整分辨率或按质量筛选。TextVQA 中5个问题引用有效 MPO 内容，格式明确保留为 mpo；native runner 后续按统一原生 RGB/EXIF 合同处理。宽高来自同一 encoded 图像的 header，不参与筛选。

## 抽样与后续复用

每个完整 split 的 rawQA/refs 先冻结，再按 `SHA256("20261119" + NUL + dataset + NUL + source_cluster_id)` 排序 source，取每源在原始 split 顺序的首问。Text/Doc/Info 的 P1 是同一 P2 的子集；Info P2 恰好覆盖全部500源，但仍只有500题，不冒充2801题 full。

Chart 按 human/aug 各自排名，P2每组250源首问；后一组优先避开已取 source，本次实际500个不同源。额外的 Chart P1 manifest 每组64；它不用于替代本轮要求的 Text/Info P1。

抽样只读数据身份，不读模型 predictions/scores。P2/M2/M3 可复用这些输入，但复用样本不能称作新的独立确认集。只有输入、processor、prompt、模型、dtype、边界和解码合同都一致时，后续新算法才能与缓存预测逐项配对。

## 验收与失败保留

- `audit/dataset_staging/registry_v3_complete.json`：所有4源的 full StageIdentity、P1/P2/full 图片覆盖与输入/ref SHA，远端同内容的 `STAGE_IDENTITY.json` SHA256 为 `8444ba1ac49248c20b2b15900b3e8284699936eb18227d0f0a7dd08ae67e1362`。
- `audit/dataset_sources/registry_v3_integrity.json`：全量数目、groups、原图格式、ID与输入无标签检查。
- `audit/dataset_sources/registry_v3_independent_qa_replay.json` 与 `registry_v3_info_sourcepath_replay.json`：独立重读原 Parquet，核原问题、全部答案/10票、原 split 行号、真实 PDF/image source 聚类；全部通过。
- `audit/source_snapshots/dataset_registry_v3/`：成功执行的两份数据脚本与 source lock 冻结副本。

初始 v1 缺少 MPO 格式，v2 将 Chart group 误假为 aug，均在读取数据时失败；失败目录、失败记录与对应源码快照全部保留。v3 明确支持实际格式与原始 group，没有删样本、变答案或覆盖失败版本。

复用命令（新 registry 版本/新 staging run-id；已有完整原图只做 SHA 核实）：

```
python ops/materialize_dataset.py build --version <new_version>
python ops/materialize_dataset.py stage --version <version> --dataset <dataset> --stage p1 --run-id <new_unique_run>
```
