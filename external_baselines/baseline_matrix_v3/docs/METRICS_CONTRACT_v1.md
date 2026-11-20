# Frozen offline scoring contract

`configs/metrics_v1.json` declares the primary metrics before public-task quality runs. `code/metrics.py::score_question(dataset,prediction,reference)` receives reference answers **only in the offline scorer**, preserves the original prediction, and returns a score in [0,1], normalized text, exact-match diagnostics, `metric_id`, ChartQA group and named diagnostic scores. No reference is added to a model prompt, calibration objective or method-selection input. The existing record field `official_score` is a schema alias for the explicitly identified primary score; it does not certify an inaccessible evaluation server.

## TextVQA

The scorer uses a byte-identical copy of the frozen standalone Meta MMF evaluator, with its original BSD license and source snapshots retained. Its processor/evaluator source bytes match [MMF commit 72f898e](https://github.com/facebookresearch/mmf/tree/72f898ec35af86567423c7a019dd1dc5175d1a5a), independently locked by Git blob and SHA256. Exactly ten annotator answers are required. The evaluator averages the nine-answer comparison over each held-out annotator, giving 0.3/0.6/0.9/1.0 for 1/2/3/4 matching answers. Replacing this with `min(votes/3,1)` changes the metric. Case, punctuation, articles, number words and contractions follow the original processor.

## DocVQA and InfographicVQA

The [DocVQA site](https://www.docvqa.org/datasets/docvqa) and [InfographicVQA site](https://www.docvqa.org/datasets/infographicvqa) both direct downloads to RRC challenge 17. Direct RRC task/download retrieval timed out during this audit. A current server-side original was **not obtained**. The locked [Qwen-VL vendor](https://github.com/QwenLM/Qwen-VL/blob/aa00ed04091eea5fcdd32985e7915f1c53e7d599/eval_mm/infographicsvqa_eval.py) states official-site provenance; this header alone does not authenticate it as today's unique official server implementation. The [DocVQA author's framework](https://github.com/rubenpt91/MP-DocVQA-Framework/blob/7939f521fe5a82a442f54831b1c0e96daab331d9/metrics.py) is also retained and tested as a distinct implementation.

The primary protocol is explicitly named `docvqa_anls_qwenvl_vendor_rawlen_ge0p5_v1` or `infovqa_anls_qwenvl_vendor_rawlen_ge0p5_v1`. Both use the same **declared local** vendor semantics: strip/lower/collapse whitespace for edit distance; divide by the maximum length of the original uppercase strings; choose the best reference; retain similarity >=0.5. This is a reproducible fixed protocol, not a claim that the two current challenge servers necessarily share code.

Every row also reports normalized-length ANLS with the same whitespace processing and threshold, plus the author's strip-only framework score as named diagnostics. Primary and diagnostic scores must remain separate in tables and caches. The contract is fixed across all models and methods; it must not be selected according to which variant produces higher scores.

Boundary fixtures expose material disagreements: `1970` versus `1985` scores 0.5 under the declared primary and 0 under Pix2Struct's strict `NL<0.5` variant; padded prediction `  ax  ` versus `ab` scores 5/6 under the vendor raw-length rule and 0.5 under normalized-length scoring. The vendor denominator can reward padding; this behavior is preserved and disclosed rather than silently changed. Existing R7 or other predictions require **offline rescoring under this contract**; old scalar ANLS values are not interchangeable.

## ChartQA

The selected implementation is [Google Pix2Struct relaxed_correctness at 110d666](https://github.com/google-research/pix2struct/blob/110d6666e8813936185fa0e9e6c38f35b387012c/pix2struct/metrics.py), a primary code implementation of the ChartQA paper's 5% relative tolerance. The [ChartQA author repository at 044eabf](https://github.com/vis-nlp/ChartQA/tree/044eabfc306abfe9340c5741f0093aefc5973d06) currently releases an exact-match evaluator, so it is retained as provenance evidence rather than described as the selected relaxed scorer.

Numeric targets use relative error <=0.05; a final `%` suffix divides by 100. A numeric target of zero takes the canonical function's original-string case-insensitive equality fallback: `0` versus `0.0` is false, while `0` versus `0` is true. Yes/no and other strings match case-insensitively with their original punctuation and whitespace. No extra comma, currency, sentence-prefix or punctuation cleanup is applied. Percent conversion, negative numbers, zero, numerical thresholds and text rules are covered by source-function crosschecks.

The registry must explicitly label each test question `group="human"` or `group="aug"` (the input alias `augmented` is accepted). The scorer preserves these labels; `aggregate_scores` reports both groups and the question-weighted overall mean. Missing groups are exposed as `ungrouped`, never silently assigned.

## Verification and provenance

`audit/upstream/metrics_source_lock_v1.json` records five pinned repositories, 13 small source/license files, Git blob SHA1 and byte SHA256. It also records the frozen TextVQA port SHA256 `b1a8e90abbf35e32e1e6c146c36c4a47ad72b71e73011f97cc3729aa1402b4c5` and source provenance limitations.

`audit/metrics_gate_v2.json` passes 465 synthetic boundary/source comparisons, including the full locked vendor evaluator on persisted JSON fixtures, AST-extracted unmodified Pix2Struct relaxed scoring, the MMF classes and the separate author's ANLS. Human/aug aggregation, raw-string preservation and invalid-input rejection also pass. The gate's technical status label includes `OFFICIAL_SOURCE_CONTRACTS`; it describes these comparisons and does not authenticate RRC server identity. No dataset quality was measured by this CPU gate. The earlier Windows GBK source-read failure is preserved in `audit/metrics_gate_v1_failed.json` with its original gate source; the fix only made UTF-8 parsing explicit.
