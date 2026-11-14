# Independent cycle analysis

`analyze_cycle.py` reads completed session records and offline references. It
does not import the generation runner or expose answers to it. It snapshots
each input's bytes in memory before analysis so the hashes describe exactly
the records analyzed, even when later snapshots arrive from an active run.

```powershell
python code/analysis/analyze_cycle.py `
  --records results/dev_baselines_gpu0_v1/records.jsonl `
  --references configs/data/clevr_references.jsonl `
  --samples configs/data/clevr_dev32_runner.json `
  --output results/analysis_dev_baselines_gpu0_v1 `
  --figures figures/dev_baselines_gpu0_v1
```

Multiple record/reference/sample files can be passed. Duplicate configuration,
session and method rows cause an error rather than silently selecting the last
run. Model, generation settings, execution protocol, data protocol, read-budget cap and split
define separate comparison groups. For bridge data pass its runner manifest
or copy `native_session_id` and `reference_turn_offset` into the result. Its
unscored description turn must not be shifted onto the first QA reference.

## Scoring and statistics

Primary scoring is normalized full-string exact match. Its definition matches
the historical `p182_dialogue_metrics.normalize_short_answer` and the exact
part of `p183_recovery_core.score_answer`: lowercase/strip, remove an initial
`Answer:`, remove terminal `. ! ?`, map zero–ten number words and grey/gray.
The extra CLEVR material/shape aliases are reported as an auxiliary metric.
Long text containing a correct word does not pass exact match. This is a
specified CLEVR short-answer protocol, not a claim to reproduce the official
GQA, DocVQA or natural-dialogue evaluators.

`paired_diffs.csv` retains both repaired and newly introduced errors. Paired
confidence intervals resample source images, retaining every question and
dialog belonging to the sampled image. The statistic is the question-weighted
mean delta. A one-image snapshot reports its observed value without pretending
to estimate a population confidence interval. Missing questions and missing
references are reported, not replaced by guessed labels.

`method_summary.csv` includes all available scored data and later QA turns
separately. The resource-quality figure uses the same-question intersection
across displayed methods. Its coverage can be smaller during an active run.
The manifest's expected record set and completion receipt are checked; partial
or receipt-less runs are explicitly labelled in the report and figures.

## Resource accounting

`resource_sessions.csv` and `read_events.csv` preserve committed/replay/probe
base-visual, text, residual and reference reads. A margin replay rereads both
base visual KV and text KV. Residual payload alone is not its cost. Token counts,
replay counts, replay rate per observed margin step, owned tensor bytes and
peak allocated bytes are exported. Missing TTFT/TPOT measurements remain empty.

Logical read volume is not measured HBM traffic. It excludes the initial native
visual prefill that occurs before the instrumented fused calls. Owned visual
memory includes packed, materialized and retained diagnostic owners; KVarN
mechanism ports with packed plus materialized FP16 tensors cannot be reported
as packed-only memory savings. Peak allocation includes the model and scratch
buffers and is kept distinct from owned KV memory.

## Figure contract

- Question: how does actual observed answer quality vary with logical read work
  and owned visual memory on the same question set? No direction is presumed.
- Evidence: resource-quality point comparison and committed/replayed read
  breakdown. Curves require actual records; no mock experiments are generated.
- Archetype/backend: quantitative grid; Python matplotlib exclusively.
- Export: 7.2-inch two-panel comparison, editable TrueType PDF and 300-dpi PNG;
  restrained method colors, image-cluster 95% percentile intervals when estimable.
- Integrity: method points are not connected into an invented tradeoff curve;
  oracle references are hollow triangles; partial snapshots are labelled.

## Mechanism scatter interface

Pass `--mechanism path/to/points.jsonl`. A plotting-ready row has:

```json
{"intervention_id":"case1:L4:p12","session_id":"case1","image_id":"image1",
 "delta_current":0.02,"delta_future":0.05,
 "current_metric":"logit_KL_reduction","future_metric":"mean_logit_KL_reduction",
 "horizon":3,"action":"refine_L4","oracle":false}
```

This schema example defines fields only; it is not an experiment. Alternatively
supply `current_loss_before/after` and `future_loss_before/after`; reductions are
computed as before minus after. Scatter groups must have the same metric and
horizon definitions. No y=x diagonal is drawn across potentially different
scales. Undefined metric/horizon rows are exported but not plotted.

Run CPU-only checks with:

```powershell
python -m unittest discover -s code/analysis -p 'test_*.py' -v
```
