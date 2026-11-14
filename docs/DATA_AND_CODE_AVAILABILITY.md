# Data and code availability

This is a research-progress archive. No target journal or final manuscript has been fixed.

## Data Availability

The generated result tables, figure source data, calibration parameter artifacts and experiment metadata are provided in the project repository at `https://github.com/zhouz4884-coder/TrajKV`. Versioned evidence archives contain derived, path-redacted execution records and source snapshots. Original-run hashes and public-export hashes have distinct scopes and are recorded separately; transformed files are not presented as byte-identical originals.

The underlying benchmark images are reused public data obtained from the documented upstream sources. They and the pretrained model weights are not repackaged here. Current source-disjoint DocVQA and TextVQA cohorts use pinned validation exports and preserve original image bytes. Membership files record source IDs, selection rules and image hashes. Historical legacy cohorts also retain their input hashes and provenance notes; some historical asset encodings require the original legacy assets for exact replay.

The complete TextVQA5000 input set was staged but its eight-method,40000-answer evaluation was deferred before GPU inference. It must not be counted as an evaluated result. The256-image balanced transfer cohort is not an estimate of the official overall question distribution.

## Code Availability

Method implementations, analysis/plotting programs, fixed configurations and the required legacy helper subset are included. Third-party code retains its upstream notices. The tested model environment uses Python3.10, PyTorch2.4.1+CUDA12.1, Triton3.0, Transformers5.12.0 and NumPy1.26.4. GPU execution used the full FP16 InternVL3-8B-hf model; teacher-forced diagnostics and materialized/oracle controls are labeled separately from packed deployment candidates.

The public copy replaces private machine paths and connection information. A new reproduction should generate a new configuration/manifest for its own paths and environment, using the frozen method parameters and cohort IDs. Original-host source hashes document the historical executions, while `PUBLIC_EXPORT_MANIFEST.json` describes the derived public files.

## Reused data records

| Resource | Version / record | Access route |
|---|---|---|
| DocVQA validation export | `lmms-lab-encoder/DocVQA`, revision`539088ef8a8ada01ac8e2e6d4e372586748a265e` | [Pinned public export](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/tree/539088ef8a8ada01ac8e2e6d4e372586748a265e/DocVQA) |
| TextVQA validation export | `lmms-lab-encoder/textvqa`, revision`9c0699cd19768ac5ab97568f6b3cbac4c0062884` | [Pinned public export](https://huggingface.co/datasets/lmms-lab-encoder/textvqa/tree/9c0699cd19768ac5ab97568f6b3cbac4c0062884/data) |
| InternVL3-8B-hf | Exact local checkpoint file hashes in the model identity record | [Upstream model card](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) |
| Legacy CLEVR-Dialog/GQA/ConvBench subsets | IDs, hashes and provenance in the corresponding archived cohort manifests | Reused sources documented per cohort |

These entries identify the redistribution/export provider separately from the original dataset creators. Upstream images, annotations, model weights and third-party software retain their original terms; this archive does not assign them a new license. No DOI, accession or ethics approval is asserted.

## 中文核对

数据分为生成结果/参数、复用数据、模型权重和代码。公开版本提供结果与来源，原图和权重从上游取得。新源文档确认、已观察子集复用、开发和准备但未运行任务均分别标注。当前是研究归档，正式投稿时再按目标期刊确定最终许可与长期归档标识。
