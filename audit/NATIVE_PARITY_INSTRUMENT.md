# Native SDPA / full fused instrument calibration

Runner: `code/experiments/native_parity.py`.

This is a genuine cross-backend check. Two independent standard native SDPA
visual prefills create two distinct cache objects with disjoint tensor owners.
One cache keeps ordinary Transformers update methods and combined KV tensors.
The second is attached to `ProgressiveBank(method='full')`, retaining original
FP16 visual support and using P202 fused attention. Native forward passes use
`attention_bridge(native=True)`; fused passes use `p184_working_set.attention`.
Each call asserts that the native branch did not invoke a fused bank hook and
that the fused branch did invoke it. This does not compare the fused backend
with itself.

By default, the native branch greedily selects a canonical answer token, which
is then committed to both branches. Question suffix chunks, answer-token
stepping, chat-closure tokens and original positions are identical. Optional
`--generate-side fused` reverses the trajectory owner. Call order alternates by
default, rather than always running one backend first. No reference answers
are present in the runner input.

`records.jsonl` contains every computation's full-vocabulary logit maximum/RMS
difference, KL(native || fused), argmax disagreement, target log likelihood,
and post-RoPE text K/V differences at the original native positions. It also
reports the newly written text rows separately. Initial and terminal checks
include visual K/V. The initial-prefill row is explicitly an independent
native/native repeat control, before the first fused append. Greedy prediction
rows identify whether their next token was actually committed; logits after
EOS or forced closures are retained as numerical observations but are not
silently counted as actual answer predictions.

`sessions.jsonl` contains generated tokens, initial/final state summaries, the
empirical error distributions, all argmax disagreements and the subset at
committed predictions. No numerical pass threshold is hard-coded. Differences
must be interpreted against the observed distribution, margins and answer
changes. Same-trajectory parity alone does not establish independent free-run
quality equivalence, quantized quality, or serving efficiency. Instrumented
call order and state checks make the collected wall time unsuitable as a
latency benchmark. `full_fused_owned` describes only that branch; the separate
native cache is additional diagnostic memory.

Required flags are `--model --samples --legacy-scripts --kivi-source --output`.
Default scale is four development sessions, three turns, FP16. A small GPU
validation can use `--count 1 --turns 3 --max-tokens 12`. Physical GPU selection
belongs to the caller through `CUDA_VISIBLE_DEVICES`; the script uses the one
visible `cuda:0`. No GPU was started by the implementing sub-agent.

Local validation completed: Python compilation; exact visual/text comparison
through noncontiguous original positions; a planted V-only change isolated to
new rows; unchanged K; rejection of unequal logical lengths; native-generated
target selection with intentional cross-backend argmax disagreement. These
are CPU instrument checks, not CUDA parity results.
