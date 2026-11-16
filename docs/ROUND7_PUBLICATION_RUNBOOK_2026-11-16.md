# Round7 local preparation and root-controlled publication

The three new entry points are `prepare_round7_delivery.py`, `verify_round7_increment.py`, and `publish_round7_update.py`. They preserve prior scripts, commits, releases and raw evidence. Only root runs the publisher with `--execute` after final research validation.

## Inputs root completes

Copy `configs/publication/round7_delivery_plan.template.json` to a new concrete plan, for example `configs/publication/round7_delivery_plan_v1.json`. Set the final report, README entry and receipt paths. The required fields are:

|Field|Meaning|
|---|---|
|`execution_receipt`|Explicit research-completion receipt; not a claim GitHub delivery already succeeded.|
|`evidence_verification`|Passing local raw-score/owner recount produced by `verify_round7_evidence.py`.|
|`final_report`|Final Round7 research report.|
|`readme_entry_file`|Concrete new README top entry; the publisher appends the prior README in full.|
|`method_freeze`|Existing immutable primary method/source freeze.|
|`additional_checks`|Final quiet/profiling and secondary-freeze validation files, each with `path` and optional `accepted_statuses`.|
|`failed_evidence_roots`|Preserved failure logs whose contents must remain outside primary performance.|
|`tag_name`, `release_title`, `release_body`|New Round7 release identity and author-reviewed wording.|

The supported prepublication receipt state is `status="RESEARCH_COMPLETE_READY_FOR_PUBLICATION"`, `research_execution_completed=true`, `execution_goal_completed=false`, `github_sync_pending=true`. Publication preparation preserves that distinction. Successful remote verification later creates separate private/source and derived public delivery receipts; root completes the overarching goal last.

Primary final analyses are expected under `confirm_doc128_analysis` and `confirm_text128_analysis`; the separately frozen second-model first-question analysis is expected under `second_confirm64_analysis`. Secondary sources overlap the prescribed primary Doc subset and do not increase independent source counts. Completed development analyses with complete run/score/reference fingerprints are also discovered and numerically recounted; overlapping analysis views are explicitly not new observations.

The runtime source collection may use either the old `COLLECTION.json` layout or the new `runtime_index.json` plus `objects/<original_sha256>.py` layout. Both recurse into original source objects and preserve per-job/narrow mappings. Recollect after the final jobs so the archive includes their immutable versions too.

## Exact commands after the final analyses complete

```powershell
python code/publication/verify_round7_evidence.py --analyses results/round7_v1/confirm_doc128_analysis results/round7_v1/confirm_text128_analysis results/round7_v1/second_confirm64_analysis --calibration-runs FINAL_COMPLETED_CALIBRATION_DIRECTORIES --failed-runs PRESERVED_FAILED_RUN_DIRECTORIES --freeze audit/a2/round7_method_freeze_v1.json --output audit/publication/round7_final_numeric_verification.json

python code/publication/prepare_round7_delivery.py --plan configs/publication/round7_delivery_plan_v1.json --preflight-only

python code/publication/prepare_round7_delivery.py --plan configs/publication/round7_delivery_plan_v1.json --output artifacts/round7_publication_assets_v1 --runtime-sources artifacts/round7_private_runtime_sources_v1

python code/publication/verify_round7_increment.py --assets artifacts/round7_publication_assets_v1 --output artifacts/round7_publication_assets_v1/VERIFICATION.json

python code/publication/publish_round7_update.py --assets artifacts/round7_publication_assets_v1

python code/publication/publish_round7_update.py --assets artifacts/round7_publication_assets_v1 --execute
```

Omit the optional calibration/failed-run arguments only if there are no such runs. Use actual preserved directories, including the empty interrupted runs linked to failure logs. The publisher's default checkout remains the existing public checkout; `--checkout` and `--receipt` can select explicit existing/local destinations. A call without `--execute` performs local package verification and reports readiness without Git/network operations.

## Contents and privacy

The preparation includes changed research code/configs/docs/figures/audits, all safe Round7 results, browsable summaries and CSVs, explicitly indexed original JSONL evidence, failures, and supplied source snapshots. The method and literature documents are included through the normal changed-document inventory. All required scoring references and consumed codec inputs are included even when unchanged from the public baseline.

Chats, DS original conversations, checkpoints, authentication/connection helpers, original images, model files and every `.pt`/tensor/checkpoint payload are excluded. Machine roots, hosts and connection metadata are redacted in the derived copy. Known private roots are also redacted in protected metadata strings, while ordinary human-answer IP/numeric content is retained. Failed partial JSONL tails are preserved as explicitly unparsed evidence, rather than converted into fabricated completed rows. UTF-16 failure traces are decoded, redacted and exported as UTF-8 with distinct original/public hashes.

Cached-prefix teacher-forced calibration remains diagnostic, never E2E accuracy or timing. Alias/equivalence files remain metadata; identical policy aliases do not acquire duplicate free-running observations. The final report must describe the matched static batch equivalence and cannot attribute a conditional-policy gain uniquely to re-estimation when a matched static policy reproduces it.

## Byte identity and publication checks

`ROUND7_SOURCE_INDEX.json` separates preserved original `source_sha256` from exact derived archive `public_sha256`. `ROUND7_CONTENT_REFS.json` fixes exact archive bytes with `content_ref=sha256:...`, and separately lists LF Git body SHA256 and predicted Git blob OID. The strict frozen `encoder_book_sha256` contract requires the release's exact parameter bytes; a Git text rendering may have normalized line endings. Expected frozen hashes are never silently changed to fit a Git rendering.

All verified expected Git files are force-staged explicitly, including every `configs/data/` reference/source list. The publisher checks staged paths, committed tree OIDs, actual committed blob-body SHA256, and each expected path in the remote recursive tree. README, source/content indices, public delivery metadata and generated manifest/ready files participate in that expected set. The final public receipt binds the content index to the verified commit and verifies its anonymous fixed-commit Git body.

All prior releases are enumerated with pagination, including Round6. Prior tags, release identities and asset IDs are retained; same-name assets with different size/SHA are rejected rather than overwritten. A journal permits safe continuation after installation/commit/push/upload interruption. The original local remote-verification receipt and its derived public receipt remain separate; only the redacted version is uploaded, with distinct source/public fingerprints.

The implementation was compiled and checked locally using preserved completed smoke evidence: a 22-file derived archive independently recounted 56 existing answers and owners, without Git/network/GPU operations. Additional controls passed for pending GitHub goal state, exact-vs-LF identities, ignored reference coverage, bare-mount/protected-metadata redaction and failed partial/UTF-16 traces. These are publication-tool acceptance checks, not a claim that the full Round7 package or upload has completed.
