# Development mechanism analysis

Run `analyze_mechanisms.py` with a completed mechanism directory and carrier
surgery directory, an output directory, and optionally a figure directory.
It writes per-turn four-arm metrics, numerical nulls, action ranks, dose curves,
matched controls, K/V contrasts, all local counterexamples and an input-hashed
summary. Confidence intervals resample original image clusters.

The four-arm KL is `KL(reference || arm)`. An independently constructed
reference-history/current-reference arm is the common distribution reference.
The scalar KL finite differences describe an interventional response surface;
they are not automatically a natural-mediation decomposition or additive
fractions of task errors. History and current-read replacement can help on
average while hurting particular turns; those counterexamples are retained.

V1 action scores are likelihood changes of the fixed base-generated tokens.
They may reward or penalize an incorrect answer and are never labelled
correctness. V2 action scores instead reduce KL to the same-token all-reference
history; this is distribution fidelity, also not task accuracy. Mean target-turn
effects are averaged within a write intervention before ranking actions, so a
two-turn future horizon does not duplicate its current effect.

Carrier surgery interpolates existing text K/V toward its reference-history
value at lexical candidate spans. It is a synthetic state intervention, not
a legal codec or a deployable repair policy. Controls match position count,
layer set and logical value-write bytes. They do NOT match perturbation L2;
the analysis explicitly reports zero-amplitude controls and source amplitudes.
Output KL/RMS and argmax flips are all exported. Greater displacement from the
base trace does not by itself establish a beneficial correction.

The multi-panel chart uses only measured records. It shows paired image-level
four-arm means, current/future action scores with their explicit metric, next-
turn dose-response means, and matched K/V responses. The carrier dose bands
are image-cluster 95% intervals. Zero responses are retained on symmetric-log
axes. Per-case nonmonotonicity and controls that exceed a carrier response
remain in the source tables and summary rather than being removed.

KVarN mechanism ports are scored using `analyze_cycle.py`. Their quality can
be compared on identical questions when initial-logit parity permits. A
materialized reference path is flagged `cost_comparison_allowed=False` and is
excluded from packed-runtime efficiency plots; its actual owned tensors and
times remain in a separate resource table. The quality-only chart hatches
these reference paths.
