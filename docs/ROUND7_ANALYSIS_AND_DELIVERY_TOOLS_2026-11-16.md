# Round7 analysis, freeze, and local delivery tools

These additions perform CPU/local work. They do not launch inference, stage confirmation answers automatically, invoke Git, or publish externally.

## Completed-run quality and cost analysis

```powershell
python code/a2/analyze_round7_quality.py --runs results/round7_v1/RUN_A results/round7_v1/RUN_B --samples SAMPLE_JSON --references REFERENCE_JSON_OR_JSONL --dataset doc --output NEW_ANALYSIS_DIRECTORY --expected-variants N --analysis-role development_SELECTION_ROLE
```

Use `--dataset text` for official ten-annotator TextVQA scoring. One method is valid; its comparison file is empty. The supplied samples may be a complete source bank: the analysis uses the exact union of completed manifests' selected identities. Incomplete runs return `NOT_COMPLETE_NO_SCORING` before opening references and do not create an output directory.

`--variants NAME_A NAME_B ...` applies an analysis-only filter after validating each original run's complete records, configuration and owner evidence. Original producer counts, variant names and fingerprints remain in the summary; no replacement execution manifest is created. This allows the same five methods from an observed nine-method pilot on 16 sources to be combined with new five-method runs on the remaining 48 sources. The resulting development choice does not become blind confirmation. `--quiet` prints only compact completion counts; the complete tables and comparisons remain in files. Default stdout prints a short quality/bit-rate table and omits verbose comparisons.

The analyzer reports question-weighted source-cluster bootstrap intervals, paired source-cluster sign flips, Holm adjustment over the reported comparison family, and a descriptive source-macro mean. First-question second-model results remain separate in scope from primary multi-turn results. It does not infer blind status from filenames or significance from a development smoke.

For new Key records, it independently reconstructs the actual K nibble/byte indices, row gains, centered means, V bit words and metadata, global book, selected residual stages, lookup, positions, and encoder factor/metric/ranking/policy payload. It ignores `owned.total_unique` as a visual-cache addend. The primary rank-one metric costs 29,120 bytes: FP16 `u[28,4,128]` and FP32 `gamma[28,4]`. K2-reference layouts preserve the reference residual mask while charging actual K1/K2 storage. Second-model V tails charge `ceil(real_N/32)` physical bit words; they never add semantic tokens, centroid-fit samples, residuals, or positions.

Outputs are `scores.csv`, `table.csv`, `comparisons.csv`, `owner_audit.csv`, and SHA-linked `summary.json`. The full precision reference includes its real position owner, so its complete bit rate is slightly above 16.

Real parser/owner checks completed on `key_smoke2` and `second_llava16_smoke4_v1`, with new results under `key_smoke2_quality_v2` and `second_llava16_smoke4_quality_v2`. `check_round7_quality_analyzer.py` passed eight CPU acceptance controls. Those controls create no empirical model observations.

## Concrete method freeze before confirmation labels

```powershell
python code/a2/freeze_round7_confirmation.py --configs SELECTED_CONFIGS --analyses COMPLETED_DEV_SUMMARIES --artifacts CALIBRATION_POLICY_AND_PARAMETERS --selected-variants METHOD_NAMES --output audit/a2/round7_method_freeze_v1.json
python code/a2/freeze_round7_confirmation.py --verify audit/a2/round7_method_freeze_v1.json
```

`--inference-sources` adds any source files beyond the automatic narrow dependency list. The freeze pins concrete variant objects, config bytes, deployed codec inputs, caller-selected calibration artifacts, development receipts, source-only calibration splits, and both new confirmation identity pools. A selected variant cannot silently differ from the same named variant in its supplied analysis receipt. Creation refuses if either new confirmation reference file already exists. The freeze establishes unchanged declared inputs; it does not establish that a method is superior or absent from backbone pretraining.

Only after this freeze, root explicitly runs:

```powershell
python code/a2/round7_data_stage_doc.py --stage-questions --method-freeze audit/a2/round7_method_freeze_v1.json
python code/a2/round7_data_text_confirm.py --stage --method-freeze audit/a2/round7_method_freeze_v1.json
```

## Independent local evidence recount

```powershell
python code/publication/verify_round7_evidence.py --analyses FINAL_ANALYSIS_DIRECTORIES --calibration-runs COMPLETED_CACHED_PREFIX_RUNS --failed-runs PRESERVED_FAILED_SMOKE_DIRECTORIES --freeze METHOD_FREEZE_JSON --output NEW_LOCAL_VERIFICATION_JSON
```

The verifier recounts each raw prediction with the frozen original selected references, checks all raw/scored identities and means, and recomputes complete owners including new metadata/tail storage. Cached-prefix teacher-forced calibration remains a diagnostic and is excluded from E2E quality and timing. Failed or partial smokes remain individually fingerprinted and cannot enter the quality run list. The real-smoke recount passed for 56 scored answers; this small parser check does not establish a new research result.

## Exact archive content and Git LF bodies

After root prepares the final derived payload/archive and its `files` source index:

```powershell
python code/publication/round7_content_index.py --source-index ROUND7_SOURCE_INDEX_JSON --archive EXACT_ARCHIVE_ZIP --payload DERIVED_PAYLOAD_DIRECTORY --output ROUND7_CONTENT_REFS_JSON
python code/publication/round7_content_index.py --verify-index ROUND7_CONTENT_REFS_JSON --archive EXACT_ARCHIVE_ZIP --payload DERIVED_PAYLOAD_DIRECTORY --git-tree-map ROOT_SUPPLIED_ACTUAL_TREE_JSON
```

Each fixed `content_ref=sha256:<exact-public-content>` points to exact archive bytes. Original `source_sha256`, archive exact SHA256, LF-normalized Git body SHA256, and predicted SHA1 Git blob OID remain distinct. UTF-8 BOM and all non-newline bytes are retained. Root must generate the actual tree JSON from the intended commit; this helper never invokes Git itself.

The index emits a staging plan with explicit `force_paths` for every expected `configs/data/` file. Root's publisher must run `git add -f -- <each verified force path>` and verify the resulting actual tree. A file's presence in the working directory alone does not establish Git coverage. Raw JSONL and runtime source blobs can be marked archive-only, while browsable files must match their exact archived counterparts.

A CPU fixture verified that CRLF archive bytes differ from normalized LF body hashes, that both identities remain valid, and that ignored `configs/data/` metadata appears in the force list. The fixture used a simulated tree map and made no Git calls; it is not a real publication receipt.

Final Round7 publication preflight and the selected public payload should be wired to the final completed summaries. Existing releases, frozen evidence, original media, model files, and private routing/credentials remain outside this tool's publication actions. Root performs the final incremental push, archive upload, remote commit/tree verification, release preservation checks, and consolidated report.
