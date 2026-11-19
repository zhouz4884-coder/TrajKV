# Existing K joint-gain geometry: CPU diagnostic

This is an offline diagnostic on the existing 64-source native dump. It performs no model forward, label reading, dictionary fitting or GPU operation. It does not change model-quality source files or retain another tensor bank.

Root uploads/hash-verifies the new `code/a2/round8_joint_geometry_diagnostic.py` in the isolated Round8 stage, then may run:

```powershell
python code/launch_round8_data_disk.py configs/jobs/r8_joint_geometry_native_cpu_v1.json
```

This document and job are preparation only; no remote process has been launched here. The job uses the actual 8TB P72 interpreter, four Torch/OMP/MKL threads, empty `CUDA_VISIBLE_DEVICES`, and data-disk cache/tmp directories. The native `.pt` shards stay in their existing global project data-disk directory. The new result path is `results/round8_v1/joint_gain_geometry_native_v1/`.

For each source and head, the script reconstructs the original K4_16 template and verifies its FP16 base against the recorded old base. For the 23 selected first-stage K heads it calls the existing `round8_joint_gain.joint_refit_key` using the original physical gain, sign bytes and alpha; it does not apply the functional factor again. Reconstruction is the same `FP16(FP16(g*t)+alpha*s)` operation, using the shared sign decoder. All other 89 heads retain their original base and exactly zero intervention delta.

Float64 measurement uses

\[
\mathrm{SSE}=(r-a)^2+2ra(1-\cos\theta),\qquad
r=\|z\|_2,\quad a=\|\widehat z\|_2.
\]

The implementation checks this identity independently on every sampled row. Cosine is undefined when either norm is zero; those rows are excluded from cosine means and have zero angular contribution. It also checks native SSE nonincrease against the shared joint-refit statistics and the all-head dilution identity for each source. Radial/angular terms are geometric components: the angular term depends on reconstruction norm as well as cosine, so its reduction alone does not establish improved directional alignment or a causal task mechanism.

Outputs are small `head_metrics.csv` (64×112 heads), `source_metrics.csv` (64×2 scopes), `summary.json`, `toy_check.json` and hash-pinned `complete.json`. Each sampled row contributes to strict SSE-improvement counts, cosine improvement/worsening counts, radial/angular counts, cases where SSE improves while cosine/angular worsens, and delta quantiles. A large per-row table and all tensors are intentionally omitted. The private original shards plus their pinned hashes remain the reproducible inputs.

Train32 and select32 are labelled independently. Each has two scopes: protected23 and all112. Reported means give every source equal total weight and every native head/token row equal weight within its source. Pooled-row counts and quantiles are labelled separately and are not a source-bootstrap confidence interval. The local toy audit covers pure radial error, pure angular error, zero norms, and the original repeated `(2,1)` / `(1,3)` example where the shared interface returns `g=.75, alpha=1.25`.

This diagnostic can characterize whether the observed reconstruction change accompanies norm matching, direction alignment or both. Model/task quality, causality, encoding throughput and serving speed remain separate measurements.
