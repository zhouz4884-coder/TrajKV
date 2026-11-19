# Round8 delivery preparation addendum

The original v1 publication plan is preserved. The pending final-input skeleton is `configs/publication/round8_delivery_plan_v2.json`. Its main fitted-book and quality paths are the matched cap8 chain; automatic discovery retains completed legacy development analyses with their original exposed/cross-density roles. It does not relabel repeated or overlapping views as confirmation.

`code/a2/analyze_round8_final_quality.py` adds the Mean `complete_mean.json` gate and delegates all earlier specialized gates without changing their files or the scorer. Mean must bind its canonical final/intermediate/engine, original sources, config, external and sample inputs, complete source-by-variant coverage and 28-layer unchanged-field/owner/all-real-token records. Actual decode chunk and single-application evidence are checked. No pending Mean raw answer or reference is opened. The same final gates now run in the new R8 publication verifier before reference access, including original-source SHA checks for redacted archive bodies.

The synthetic CPU checker `check_round8_final_quality_gate.py` tests pending/final SHA, coverage, density, layer, owner, source-FP16, ranking, decode-chunk and repeated-application failures. These metadata fixtures are separate from task-quality and model evidence.

Readiness only, without public/Git/network writes:

```powershell
python -B code/publication/prepare_round8_delivery.py --plan configs/publication/round8_delivery_plan_v2.json --plan-only
```

Use an unused capture name for each later source snapshot; the final capture must follow final worker completion and metadata:

```powershell
python -B code/a2/collect_round8_runtime_sources.py --capture final_stage_v1
```

The collector is read-only remotely and starts no GPU. The explicit versioned runtime index is used by preparation. Keep the first canonical index and previous captures unchanged. Capturing available source versions does not prove every listed module executed, and failed/empty launch controls remain non-QA tracking evidence.

After the four actual final inputs and all stage evidence exist, root runs preflight, prepares into a new immutable artifacts directory, independently verifies the package and uses the existing publisher plan. Only root's explicit publisher `--execute` writes GitHub. The actual committed and remote tree must cover force-added `configs/data/` inputs, content identities and the increment while retaining previous history/releases. Raw maintenance and private routing remain excluded.
