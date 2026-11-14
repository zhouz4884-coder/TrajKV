# Intra-step probe analysis

`analyze_probe.py` accepts development free-running records only. It never
loads or tunes the final evaluation. Example:

```powershell
python code/analysis/analyze_probe.py `
  --records results/probe_dev8_gpu1_v1/records.jsonl `
  --references configs/data/clevr_references.jsonl `
  --output results/analysis_probe_dev8_v1 `
  --figures figures/probe_dev8_v1
```

The reference reader selects only the IDs present in those development records
before decoding answer arrays. It is used only for offline exact scoring.

## Signal and outcomes

`probe_events.csv` retains layer, phase, literal configured threshold, observed
relative L2/RMS, decision, feature synchronization time, and local charge checks.
`probe_layer_phase_summary.csv` separates actual observations from invocations
skipped because the global read budget was unavailable. A requested upgrade is
not evidence that all later layers could afford refinement; actual bytes come
from the fused-read ledger. Optional threshold sweeps describe signal crossings
on already observed events, not counterfactual policy quality.

The probe compares the same query under stored base and base+residual values.
Its correction magnitude is an online feature, not true FP16 error or a future
quality label. Timings explicitly include feature synchronization; they do not
measure only arithmetic or provide repeated-run serving latency guarantees.

## Two cost books, one global extra budget

The local controller records the full two-read probe cost:

`2 * (base_visual + text) + residual`.

The global extra cost above normal base attention pays:

`all residual reads + probe/replay/repair rereads of base_visual and text`.

Thus a probe's first normal base/text read belongs in normal attention, and its
second base/text read plus residual is extra. Adding the controller's entire
two-read charge to global extra double-counts a normal read. The tool preserves
both quantities and independently recomputes the global ledger, budget headroom
and overrun flags. These are logical fused-call payloads, not measured HBM bytes.

## Storage and comparisons

Figures compare policies only within an exact owner-layout/kernel signature,
with the same source-question intersection. Actual base and residual owners,
kernel variant/hash and model/generation settings are retained. In particular,
a compact K-only residual representation is not pooled with an older policy
that READS only K residuals but still STORES both K and V enhancements.

`analyze_cycle.py` also preserves these fields and explicitly rejects pooling
the same named variant across differing stored layouts or kernel revisions.
Its raw read breakdown includes probe base, text and residual separately.

## Figure contract

The two observed relationships are (i) online correction-magnitude distributions
by layer/phase/threshold and (ii) answer quality versus fully counted extra reads
at fixed stored representation. No quality advantage is assumed. Figures use
Python matplotlib, editable PDF and 300-dpi PNG. Confidence intervals resample
source images when at least two clusters are available; one-image smoke data
show observed points without a population interval. Real records are required;
empty inputs produce no figures.
