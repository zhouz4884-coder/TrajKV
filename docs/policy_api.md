# Precision policy API

The CPU-only module is `code/policies`. Put the absolute `code/` directory on
`sys.path` (or launch with `PYTHONPATH` pointing there), then import `policies`.
It does not access model weights, modify KV tensors, or make GPU performance
claims. The executor supplies event boundaries and read/compute costs.

## Event and feature contract

```python
from policies import ReadEvent, Feature, TokenSpan

event = ReadEvent(
    event_id="session03:t2:decode17:L4H0",
    session_id="session03", turn=2,
    token_start=123, token_end=124,  # half-open absolute KV token positions
    step_index=17, phase="decode", layer_group="L4", head_group="H0",
    base_read_bytes=4096, residual_read_bytes=1024,
    # Measured from the trial forward at this decision point:
    features={"margin": Feature(0.18, "online", "trial logits"),
              "fact_carrier": Feature(True, "online", "observed text parser")},
    carrier_spans=(TokenSpan(122, 125, label="entity", source="observed prefix"),),
    unit_key="decode|L4|H0|entity", cost_source="counted",
)
```

All entries in `features` must be `Feature(value, availability, source)`.
Availability is `online`, `reference`, or `future`. Online policies reject any
privileged feature they access. Unused privileged fields may coexist in a log
for analysis, but cannot supply the policy's score. `offline_oracle` requires
its target feature to be labelled `future`, and marks every result as oracle.

Token spans can cover one token (`yes`, a number, an attribute) or a multi-token
entity. There is no minimum length, no length-derived importance score, and no
claim that a span is genuinely important merely because a parser labelled it.
An explicit online `fact_carrier=True` or overlap with an online carrier span
enables the carrier condition. The executor must align positions correctly:
logits at the current consumed token predict the NEXT token, whose own KV is
normally produced when it is consumed on the next step.

## Policies

Use `make_policy(name, **parameters)`:

| Name | Parameters | Decision |
|---|---|---|
| `uniform` | `rate=1.0` | Causal evenly-spaced opportunities; `.25` selects every fourth event |
| `random` | `rate=.5, seed=0` | Seeded Bernoulli selection |
| `periodic` | `period=4, width=1, offset=0` | Burst window based on supplied `step_index` |
| `current_margin` | `threshold` | Refine if current online margin is at most threshold |
| `fact_carrier_margin` | `threshold` | Same margin condition AND online carrier flag/span |
| `future_gain_table` | `table, min_gain_per_byte=0, calibration_id=""` | Positive calibrated gain per residual byte above threshold |
| `offline_oracle` | `feature="oracle_future_gain", two_pass=False` | Greedy ranking by privileged future gain per extra byte |

`future_gain_table` maps `event.unit_key` to a number or an ONLINE `Feature`.
The fallback key is `phase|layer_group|head_group`. Fit this table only on the
development/calibration partition. The entries represent predictions available
before the action; they do not read future queries at deployment. Store the
table provenance under `calibration_id` and save its actual fit/eval split in
the experiment manifest.

The oracle is a greedy diagnostic reference, **not a proven optimum or bound**.
It is never a deployable method. Its `two_pass` flag must match the action
family being compared; otherwise its resource profile describes a different
action. Oracle fitting/search cost belongs in a separate offline-cost report.

For a planned replay:

```python
from policies import Budget, make_policy

policy = make_policy("current_margin", threshold=0.2)
budget = Budget(max_extra_read_bytes=1_000_000)
plan = policy.allocate(events, budget)
print(plan.selected_ids, plan.projected_cost.to_dict(), plan.budget_ok)
```

Online policies consume the iterable **in causal order** and never sort future
event scores. Keep the same policy object across streaming calls to preserve
its schedule/RNG. `allocate` does not perform forwards or commit costs; its
output is a projection. For cumulative planning, pass `starting_cost` and
`starting_refined_events` from the current ledger. Planning the full trace
does not make future margin values available to earlier decisions.

Uniform, random, periodic, and the gain table are direct-read policies. Margin
policies are trial/replay policies. If comparing allocation signals with the
same two-pass protocol, charge the same probe/replay work in the executor for
all alternatives. Match actual cumulative bytes, not just selected event count.

## Cost convention

`Cost` has disjoint fields:

- `base_read_bytes`: one normal base forward, whether or not it triggers.
- `residual_read_bytes`: enhanced representation read in the selected forward.
- `probe_read_bytes`: EXTRA feature-extraction traffic, excluding the base pass.
- `replay_read_bytes`: base traffic reread during a refined replay.
- `base_compute_us`, `probe_compute_us`, `replay_compute_us`: corresponding
  disjoint compute costs. Leave unknown times as zero only for planning and
  mark them unmeasured in the experiment output.

`extra_read_bytes = residual + probe + replay`.
`total_read_bytes = base + extra`.
A trial with 100 base bytes followed by a refined replay with 100 base + 20
residual bytes consumes 220 total bytes and 120 extra bytes. It does not cost
only 20 bytes. A non-triggered trial still consumes its base and probe costs.
Default replay base traffic equals `base_read_bytes`; supply
`ReadEvent.replay_read_bytes` when the execution path reruns a different region.

`Budget` optionally caps extra bytes, total bytes, probe bytes, replay bytes,
total compute microseconds, and refined event count. Persistent storage is a
separate model/encoding-level constraint, because a selective read policy does
not alter the resident representation. KV materialization, temporary buffers,
and allocator peaks must be reported by the execution backend.

Cost fields are backend-supplied counts. They are not an inference of physical
HBM traffic. For a `cost_source="declared"` event they are assumptions; use
`counted` for exact tensor/launch-byte accounting and `measured` only where an
actual measurement supports it. Cache hits, fused reads, and materialized
dequantization can make physical bandwidth differ from logical read volume.

## Trial, replay, commit

```python
from policies import Cost, CostLedger, MarginTransaction

ledger = CostLedger()
tx = MarginTransaction(event, policy, budget, ledger)
# Executor has run the low-precision trial in an isolated pending branch.
decision = tx.trial(actual_trial_cost=Cost(base_read_bytes=4096))
if decision.refined:
    # Restore PRE-TRIAL state; rerun the affected computation with residuals.
    # The executor measures/counts this replay and chooses its new KV/logits.
    record = tx.commit(refined=True,
        actual_replay_cost=Cost(replay_read_bytes=4096, residual_read_bytes=1024),
        cost_source="counted")
else:
    # Commit only the already-computed trial branch.
    record = tx.commit(refined=False, cost_source="counted")
```

`trial()` produces a pending decision and leaves the ledger untouched.
`commit()` may run exactly once and records consumed costs. `abort()` closes
the trial without refinement, but still records the consumed trial work.
Actual overruns are retained and flagged with `record.budget_ok=False`; they
are never silently omitted. The executor must not start replay if the pending
decision declined it. This state machine does not reserve costs across parallel
workers: use one serialized ledger/budget per execution stream or add external
reservation synchronization.

Physical correctness remains the executor's responsibility: hold trial writes
outside the live cache, restore the pre-trial checkpoint before refined replay,
overwrite every affected layer, and commit exactly one branch. A high margin
is only a policy feature; this module does not claim it proves future safety.

## CPU validation

From the project directory:

```powershell
python -m unittest discover -s code/policies/tests -v
```

Checks cover privileged-feature rejection, double-pass read accounting,
single-commit behavior, retained real overruns, one-token and multi-token
carriers, explicit oracle tagging, and causal/reproducible schedules.
