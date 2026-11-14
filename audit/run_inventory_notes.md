# Raw run inventory scope

Run `python code/analysis/collect_run_inventory.py` to refresh
`audit/run_inventory.json`. The collector reads raw JSONL, manifests,
completion receipts and hardware CSV structure. It skips every `analysis*`
directory and every `summary.json`; no answer scores or timing statistics are
computed. The inventory is a timestamped filesystem snapshot.

Each leaf retains raw-byte SHA256, parse/stability checks, primary and derived
record counts, turn-item counts, source/model/protocol metadata and an explicit
image set. Free-running answers, fixed-token generation, state diagnostics,
native/fused diagnostics and teacher-forced timing have different categories.
Control, scatter, canonical timing and warmup files are separately retained;
they do not add primary application samples. Unscored description turns are
counted separately in the raw-file entry.

Composite suite/sweep entries are containers with child-completion checks,
not extra observations. Repeated budgets retain the same image cohort. Global
image counts are unions of exact image hashes when available, with explicitly
labeled source-ID fallback; output records are not independent-sample counts.

Historical source hashes are checked against run snapshots, retained job
source bundles, exact content-hash recoveries and current files. A historical
copy that matches the declared bytes is sufficient even when current code has
changed. Missing copies and available-copy mismatches remain visible; a valid
completion receipt is not a claim that every historical source was recovered.

The legacy partial analysis input was recovered independently under
`results/analysis_dev_baselines_gpu0_v1_partial/recovered_inputs/`. The first
19 unmodified lines of the now-complete 72-line raw exactly reproduce the old
input SHA256 `3a42a9f4a675dd0c15c98cfb79cd2119a437d4957761b6f7f8629e89af929331`.
The original raw and analysis summary were unchanged, and the old analysis
remains a partial analysis. The recovery receipt links both byte identities;
it does not relabel the partial computation as a complete one.

`audit/run_inventory_cpu_validation.json` records ten structural fixture
checks. These controls do not constitute GPU, quality or performance evidence.
