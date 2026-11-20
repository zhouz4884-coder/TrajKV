# Declared-matrix publication and original-donor closure

These v2 tools are separate files. The P1 exporter/publisher, prepared stages, exact-byte archives, receipt, commit `9171b357eb610387c09ac50014510a8b9349da8d`, previous Round8 commit and releases remain unchanged.

## Scope and readiness

`ops/export_public_v2.py` accepts a frozen `--contract` or explicit `--matrix`, `--phase-label`, `--view-root`, `--score-root` and optional `--coverage-scope`. It detects original raw runs and derived views from their real schema; directory names such as `p1`, `p2_missing` or `p3` do not determine their identity. Matrix row counts, model/dataset/configuration, frozen inputs/references, exact expected QA order, original raw/source proofs and existing offline means/cluster CIs are checked. ChartQA human/aug groups and their separate means/CIs are also checked.

The current contract is `configs/publication/P2_core_nonT_40_v1.json`: 40 non-T cells ×500 questions. Completing that declared matrix is not the same as completing every external SOTA method, later default configuration or P3 full split. The exported status is `PASS_DECLARED_MATRIX_COMPLETE` only when its actual matrix closes. Missing cells remain explicit. `--require-complete` rejects an incomplete phase before creating a publication stage. Progress preview packages are explicitly ineligible for publication. Validator fixtures remain ineligible even when their small test matrix is complete.

## Evidence closure

Every derived view contributes its own three raw files and `provenance.jsonl`. Each `source_runs` descriptor is followed recursively to include every original donor, including earlier P1 donors and `p2_missing` runs. Each original donor includes the original manifest/records/complete, independent retrieval receipt, source inventory, launch receipt, original producer evaluate source and independent source-retrieval receipt. The view's per-row original line SHA, input/execution fingerprint, effective contract, source line number and fresh/reused counts are all checked. A derived view is never relabeled as new GPU inference.

The public verifier rebuilds the normalized effective contract using the archived producer source/inventory and the verified public cache-policy code. Original SHA fields retain their source meaning after path templating; source/public mappings translate them. No original hashes are silently replaced to hide sanitization. Only unchanged predictions are eligible for reuse. Future differences in model, pixels, prompt, generation, method or relevant source remain protocol differences rather than apparent extra results.

Small technical JSON/JSONL evidence is collected from both the project and the authorized E mirror's `results/p0` tree. PT/tensor dumps, images, weights, compiler caches, private routes and private execution state are not selected. Data/archives/staging remain on E; canonical GPU assets remain on the remote project data volume.

## Large result sets and verification

The v2 exporter streams JSONL rather than retaining an entire full-split result set in memory. Per-line source/public identity maps are separate archived sidecars. The main content index pins those sidecars, keeping the Git index compact as P3 grows. Every selected file still has original SHA256, public exact-byte SHA256, LF-normalized Git SHA256 and Git blob OID. Code/configuration/tables/small manifests are browsable in Git; bulky raw/scored rows and sidecars are in the exact archive.

`ops/verify_public_export_v2.py` streams public/archive identities, validates the complete donor closure, rechecks the actual declared matrix, and optionally recomputes fixed-reference scores and source-cluster CIs. It loads verified public cache/metric modules even when called from a process with workspace modules already imported. Its generated public copy is `code/verify_public_export.py` in the exported namespace.

Preparation commands, using a new stage and tag:

```text
python -B ops/export_public_v2.py --contract configs/publication/P2_core_nonT_40_v1.json --stage NEW_STAGE --tag NEW_TAG --require-complete
python -B ops/verify_public_export_v2.py --root ${EVIDENCE_ROOT}/public_staging/NEW_STAGE/tree/external_baselines/baseline_matrix_v3 --archive ${EVIDENCE_ROOT}/public_staging/NEW_STAGE/NEW_STAGE.zip --recompute-scores --receipt ${EVIDENCE_ROOT}/public_staging/NEW_STAGE/VERIFICATION.json
python -B ops/publish_public_increment_v2.py --stage-dir ${EVIDENCE_ROOT}/public_staging/NEW_STAGE
```

The publisher defaults to a dry run. Actual authorized execution adds `--execute` only after reviewing the concrete stage. It rejects previews and fixtures before opening a credential or network route. Commit/release text comes from `DELIVERY`'s actual phase label and coverage scope. The initial expected base is `9171b357...`; later increments explicitly supply their reviewed expected base. Ordinary fast-forward history is used, with both P1 and Round8 ancestry protected. Previous result/config paths cannot change; use new versioned paths. Existing tags/releases/assets and all earlier namespaces remain protected.

The actual staged/committed Git tree and blob bodies, independent server-fetched expected blobs, release asset SHA256/digests, exact archive/index identities and old release/tag preservation are required for a successful receipt. That receipt closes only the declared phase; it does not infer whole-library completion from a core subset. No P2/P3 GitHub upload was performed while developing these tools.
