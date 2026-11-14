# Inspect or reproduce

The CSV/JSON tables and scientific figures can be inspected without a GPU. Start with`docs/PROJECT_HISTORY_2026-11-14.zh.md`, the latest result summary, and`PUBLIC_EXPORT_MANIFEST.json`. GitHub Releases provide derived raw execution records and immutable source snapshots, with original/public hash scopes identified.

## A new GPU run

The verified inference environment is Linux, Python3.10, CUDA12.1, PyTorch2.4.1, Triton3.0 and Transformers5.12.0. Use the full local InternVL3-8B-hf checkpoint and compare its files with the model identity record. Every tested main model run used FP16. One full model fits a24GB RTX3090; other accelerators are not verified here.

Install the dependencies, obtain the model from its upstream model card, and use a data volume with enough scratch space. Set cache directories before starting Python:

```bash
mkdir -p .runtime-cache/triton .runtime-cache/tmp .runtime-cache/cuda
export TRITON_CACHE_DIR="$PWD/.runtime-cache/triton"
export TMPDIR="$PWD/.runtime-cache/tmp"
export CUDA_CACHE_PATH="$PWD/.runtime-cache/cuda"
```

For example, restore original pinned DocVQA image bytes by their hashes and render a new local config:

```bash
python code/reproduction/prepare_run.py --config configs/a2/metadata_confirm_doc_s0_v1.json --model /your/model/InternVL3-8B-hf --variants cold_meta_g128_kv --output repro_inputs/metadata_doc
CUDA_VISIBLE_DEVICES=0 python code/a2/run_representation_variants.py --config repro_inputs/metadata_doc/config.json --output results/reproduced_metadata_doc
python code/a2/analyze_next_docvqa.py --runs reproduced_metadata_doc --samples repro_inputs/metadata_doc/samples.json --references configs/data/docvqa_meta128_references.jsonl --expected-variants 1 --output results/reproduced_metadata_doc_analysis
```

The restoration helper downloads only the documented validation exports through the public Hugging Face route, with pinned revisions. It preserves image bytes and original cohort IDs, and creates new files rather than overwriting frozen inputs. Model weights are not downloaded by this helper. Historical legacy image encodings have their own provenance and may require their original assets for exact replay.

The new reproduction produces new input/source/runtime hashes. Specialized original-host audit scripts describe the original frozen execution and are not a substitute for checking the public-export manifest. Where path redaction changes a file, its public SHA differs from its original SHA. Parameter JSON and numerical tables are kept bytewise when no redaction is needed.

## Interpretation

- ANLS, TextVQA soft accuracy, exact match, teacher-forced KL and timing are different measurements.
- Same-image question groups are a derived workload; historical dependence is not assumed.
- Budget bits refer to the declared visual-cache account, not total model peak memory. Text KV remainsFP16.
- Logical payload reads are not physical HBM transactions.
- A confidence interval crossing zero does not establish equivalence. The fixed strong-reference0.5pp target and secondary comparisons stay separate.
- The prepared5000-question benchmark was deferred before inference and must not be counted as a result.
