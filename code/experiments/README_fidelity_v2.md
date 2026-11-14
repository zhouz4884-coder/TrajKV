# Mechanism fidelity v2

`mechanism_fidelity.py` is an independent fork; `mechanism.py` v1 remains
unchanged. The CLI is identical. No bank/runtime changes are introduced.

The base-generated token sequence is frozen. Before action probes, v2
teacher-forces the complete same-token all-reference history and caches its
per-turn logits. Each action is then compared with that fixed reference using

`gain = KL(p_base || p_same_token_allref) - KL(p_action || p_same_token_allref)`.

Positive values mean reduced distribution divergence, **not improved task
accuracy**. The primary statistic averages over non-EOS content token positions.
All-token sums/means and reverse-KL diagnostics are retained. Empty content
returns `None`, and is excluded from table fitting. No ground-truth answer is
used. The reference history may contain base-generated textual mistakes; this
experiment isolates distribution fidelity conditioned on identical text.

Schema: `trajkv.mechanism_fidelity.v2`.

- `actions[].current_fidelity` holds current divergence and gains.
- `actions[].future[].fidelity` holds future divergence and gains.
- Original `current` and `future[].metrics` remain token-likelihood diagnostics.
- `current_gain_table`, `future_gain_table`, `action_scatter` use fidelity gain
  as the main value/axis, with explicit metric and direction labels.
- `N3_precomputed_allref_vs_four_arm` checks the precomputed complete reference
  against the independently constructed all-reference four-arm branch.
- The manifest records the frozen v1 source hash and v2 runtime source hashes.

Fit both objectives on the same development sessions and action library:

```powershell
python code/policies/fit_gain_table.py `
  --records results/your_fidelity_run/records.jsonl `
  --training-ids configs/data/clevr_dev32_runner.json `
  --output results/your_fidelity_run/frozen_gain_tables.json
```

The fitter refuses v1 likelihood-only records. It retains only matched
current/future observations and the action library shared by the included
sessions. It reports skipped/missing sessions rather than inventing estimates.
Future gains first average target-turn content-token means, then average probe
instances within sessions and sessions equally. Current gains use the same
instances. Resource scores divide session-mean gain by session-mean injected
residual read bytes, not by physical HBM measurements.

Online scheduling consumes only the frozen output:

```python
table["tables"]["future"]["phase_layer_group_ranking"]["early"]  # group IDs
table["tables"]["current"]["ranked_actions"]                   # action strings
table["tables"]["future"]["policy_lookup"]                    # expected gains
table["tables"]["future"]["policy_score_lookup"]              # gain / residual byte
```

The reference traces and future/reference outcomes remain offline calibration
inputs. Keep the source hashes and `training_session_ids` alongside any online
quality evaluation to separate fit and held-out data.
