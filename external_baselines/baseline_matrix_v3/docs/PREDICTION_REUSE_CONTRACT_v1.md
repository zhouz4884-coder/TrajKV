# Exact-contract prediction reuse

This is a quality-evidence cache. It does not substitute cached answers for a new algorithm, and it does not claim cached timestamps are replay speed. The original image-prefix prefill, packed encoding and complete question-suffix generation were executed by the original GPU workers. Each original raw run remains unchanged.

`code/prediction_cache.py` uses only Python standard-library CPU operations. It imports no model, reference answer, metric or CUDA package. Indexing is streamed: retain source offsets and small identity fingerprints rather than all large records in RAM.

## Effective contract

A reusable prediction requires the same verified checkpoint files and revision, model geometry, FP16 weights, processor and generation configuration, pixel/crop settings, answer suffix, greedy generation cap, native protocol, relevant adapter/reader/upstream source, runtime versions and executable. The per-question identity also includes dataset revision/split, question ID/text/hash, original image SHA/dimensions and source-cluster identity. Original prompt/input/position fingerprints and physical-owner bytes detect conflicts between repeated same-contract records.

Phase, start/count, expected ID collection, input-list file identity and descriptive scope are not inference behavior. They may change when expanding P1 to P2 or full official splits. Method-specific dependency lists exclude the runner's glob of unrelated method source files. Unknown configuration behavior or an unknown adapter dependency map is rejected.

Only a `PASS` record can be a cache hit. Failed records are listed as missing; they are not assigned fabricated zero scores. Repeated identical answers are explicitly recorded as aliases and retained in supplied source order. A repeated same-contract input with a different answer or execution fingerprint raises `CacheConflictError`; it cannot silently select the better answer.

## Source and runtime evidence

Each raw source needs `manifest.json`, `records.jsonl`, `complete.json` and the independent remote-hash `retrieval_receipt.json` produced by `ops/fetch_raw_outputs.py`. It also needs three tiny source proofs, fetched with `ops/fetch_prediction_sources.py`: the original source inventory, launch receipt and exact snapshot `evaluate.py`, plus `source_retrieval_receipt_v2.json`. Original launch SHA binds the inventory; the inventory binds all relevant executed source. The launch runtime must match the raw manifest.

The expected execution context for a new missing-only job comes from an already verified real reference run. The actual fresh run must independently reproduce that effective contract before composition. A changed runtime, model/processor, source or prompt is rejected rather than assumed equivalent. With no fresh records, the view describes the original frozen context, not a newly observed runtime.

The sole reviewed source equivalence is the I/O optimization in `evaluate.py::model_identity`: full weight SHA reads versus previously verified immutable stat bindings with full-hash fallback. `configs/cache/evaluate_identity_equivalence_v1.json` admits exactly two entire-source hashes. Their `model_identity` AST hashes are separately locked, and the complete remaining numerical AST must match. A change to any other numerical statement, or an unknown identity function, is rejected. The AST canonical form removes source coordinates and the empty `type_params` field introduced after Python 3.10; nonempty syntax is retained. The model identity's actual filename/size/content SHA triplets still match exactly. This is not permission to ignore arbitrary validation code changes.

## Preparing and completing a phase

Use the existing remote-authorized fetch route to mirror only result files onto the named E drive. Canonical images, weights, original results, temporary files and compilation caches remain on the project data volume. Do not copy weights or images to E/D/C for cache preparation.

```powershell
python ops/fetch_raw_outputs.py results/p1/M1_F_textvqa_p1_128_v1
python ops/fetch_prediction_sources.py ${EVIDENCE_ROOT}/results/p1/M1_F_textvqa_p1_128_v1

python ops/compose_predictions.py prepare `
  --config configs/eval/M1_F_textvqa_p2_500_v1.json `
  --inputs configs/datasets/v3/textvqa_p2_inputs.jsonl `
  --reference-run ${EVIDENCE_ROOT}/results/p1/M1_F_textvqa_p1_128_v1 `
  --cached-runs ${EVIDENCE_ROOT}/results/p1/M1_F_textvqa_p1_128_v1 `
  --output ${EVIDENCE_ROOT}/results/cache_plans/NEW_PLAN.json `
  --missing-config configs/eval/NEW_MISSING_ONLY.json
```

Only the missing config is sent to the GPU runner. The full target config specifies the final cohort and is not an extra inference job. After the missing run completes and its raw/source proofs are fetched, call `compose` with that full target, previous `--cached-runs` and the actual `--fresh-runs`, using a new output directory. `ops/compose_predictions.py verify NEW_VIEW` independently rechecks the closure and original sources. Every output is created exclusively; no frozen run or view is overwritten.

The derived manifest has schema `baseline_matrix.derived_prediction_view.v1`, `inference_run=false`, `prediction_cache_used=true`, complete target IDs and explicit fresh/reused counts. `records.jsonl` contains exact byte copies of selected original source lines. `provenance.jsonl` binds every line to its source run, line number, raw hashes, effective contract and line SHA. The manifest also records the complete sizes of supplied fresh source runs: a derived selected-row count is not claimed as the amount of GPU work those sources produced. The separate offline scorer validates the entire derived source chain before applying the locked target references.

## Current acceptance and P2 preparation

`audit/prediction_cache_acceptance_v2.json` passes 13 CPU checks. It uses two real, previously completed FP16 runs to test exact reuse and a derived 128-question view. Mock conflicting-answer and failure cases stay explicitly in CPU memory and are not written as fabricated remote evidence. The tests include changed cap/prompt/generation/checkpoint/protocol/processor invalidation, bounded old/new identity-I/O source equivalence, duplicate conflict refusal and failed-record exclusion. This gate generated zero new model answers.

The fixed P2 Text/Info preparation is `configs/matrix/P2_text_info_missing_v1.json`: 20 non-quarantined cells, 10,000 target questions, 1,792 exact P1 hits and 8,208 missing questions. Fourteen prior P1 cells each supply 128 hits. Six M2/M3 Info cells have no prior data and require 500 new answers each. Rows distinguish `full_target_config` and `planned_full_job_not_to_run` from the actual missing-only `job`. `count`/`expected_QA` describe the full 500-question target; `fresh_QA` is the actual missing worker's expected count. This preparation launched no GPU worker and did not select inputs from scores.

CPU fixtures, detailed plans, mirrored raw records and derived records are on E. D contains only small source/config/summary artifacts. Subsequent public delivery must preserve source/public identity mappings and exclude private connection material; a derived view is never relabeled as a newly executed full cohort.
