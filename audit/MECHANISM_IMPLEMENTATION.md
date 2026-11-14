# Same-token mechanism runner

Implementation: `code/experiments/mechanism.py`. This document describes an
implementation, not a completed experiment or a passed scientific gate.

## Four arms

1. Generate one canonical transcript using packed base2 visual reads, exposing
   every visual token. Ground-truth answers are absent from the runner input.
2. Starting from the common initial native visual prefill, independently
   teacher-force the canonical text under original FP16 visual reads and under
   base2 visual reads. Preserve token IDs, logical positions, suffix chunking,
   answer-token stepping, and chat-closure tokens.
3. At each subsequent turn, restore each legal history and cross the current
   visual read: `history_ref__current_ref`, `history_ref__current_base`,
   `history_base__current_ref`, `history_base__current_base`.
4. Teacher-force the same current target string in every arm. Report complete
   vocabulary KL, target token log probabilities, argmax token IDs, and text K/V
   differences at equal positions. Current-read intervention includes the
   current question suffix and answer-token computation; this is not a claim
   that only the final attention read was changed.

`ref` uses original FP16 visual tensors through the same P202 fused attention
backend. The diagnostic bank intentionally retains both packed and original
visual tensors; its allocation is not deployment memory. The initial prompt
and first answer-token prediction precede compression and are common to all
arms. All historical answer tokens are the base model's generated tokens,
including its errors; reference history means re-encoding, not replacing text
with correct answers.

Contrast signs are changes in canonical-token log likelihood. A negative base
effect means worse likelihood of that fixed string; it does not necessarily
mean worse ground-truth accuracy. These controlled cache interventions are not
automatically natural direct/indirect causal effects.

## Numerical blank controls

- N1: independently repeat the full all-reference teacher-forced trajectory,
  comparing logits and text state to the reference arm after every turn.
- N2: compare canonical free-selected base generation with its same-token,
  same-position base re-encoding, including text states and every scored logit.
- Report the entire observed null distribution. No fixed `1e-3` rejection rule
  is applied. Structural token-count or position mismatches raise an error.
- N1 does not establish equivalence to a different native attention backend.

## Packed-residual action probes

`--actions all` expands to four contiguous layer quartiles crossed with three
write phases. `layer0:text_prefill`, `layer1:early`, and `layer2:late` can be
selected individually. `early` reads residuals while consuming the first two
answer tokens; `late` reads them while consuming subsequent answer tokens.
`text_prefill` reads residuals only for the current question suffix, so it is a
deliberate no-op on the first turn, whose suffix is already in native prefill.

Every action starts from the same base history and teacher-forces the same text.
Only its selected layer/phase reads the stored packed residuals. Future turns
use base-only visual reads while carrying the resulting action-written text
state. Current and future canonical-token likelihood changes therefore share
the same reference transcript. `--future-horizon` controls how many subsequent
turns are replayed. Actions are fixed in advance; future tokens do not select
the action. Their observed gains are development proxies for later freezing,
not answer-quality results or an already validated allocator.

The runner writes `records.jsonl` with four arms, nulls, full action details,
`current_gain_table`, and `future_gain_table`. It also writes a flat
`action_scatter.jsonl` using:

```
intervention_id, session_id, image_id, delta_current, delta_future,
current_metric, future_metric, current_content_tokens, future_content_tokens,
horizon, policy, action, write_turn, target_turn,
current_read_cost, future_read_cost
```

Positive deltas mean higher mean log probability of the base-generated content
tokens, excluding EOS. Rows with zero content tokens must be excluded from
content-likelihood analysis. Per-call base, residual, text, and FP16 reference
payloads are retained by phase and layer. They are logical payload counts, not
hardware HBM measurements, and include the real extra replay work.

## Entry point and local validation

Required flags match `run_cycle.py`: `--model --samples --legacy-scripts
--kivi-source --output`. A small GPU validation can use `--count 1 --turns 3
--max-tokens 12 --actions layer0:text_prefill layer1:early --action-turns 2`.
The caller controls physical GPU assignment using `CUDA_VISIBLE_DEVICES`; this
runner addresses only its visible `cuda:0`. No GPU process was started by the
implementing sub-agent.

Local CPU validation completed: Python compilation; action parsing; zero KL
for identical logits and common logit translations; positive likelihood gain
for a raised target logit; position-filtered state differences; rejection of
unequal position IDs. A CUDA model run is still required to validate end-to-end
behavior and the empirical null distributions.
