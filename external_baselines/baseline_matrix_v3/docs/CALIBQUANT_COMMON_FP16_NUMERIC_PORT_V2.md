# CalibQuant common-FP16 numerical port v2

This separately named port addresses numerical boundaries in the common FP16
benchmark environment. It is not a new quantizer, a bit-allocation method, or a
claim to reproduce the author's default BF16 arithmetic bit for bit. All old
source variants and results remain identifiable legacy evidence.

`CalibQuantFP32AffineTextAdapter` reuses the existing FP32-affine state and the
official packed QK/WV kernels. K/V clipping, integer codes, packed layout, FP16
scale/zero metadata, offsets, model weights and exact-text KV dtype stay fixed.
The persistent state has exactly the same eight tensors and physical bytes as
the existing affine port; no decoded K/V or extra position array is retained.

The new `owner.attend` entry reads visual logits through the existing FP32
affine/calibration path. Exact-text QK uses FP32 operands and output before the
joint FP32 softmax, which is then cast to FP16 probabilities. The source PV
boundaries remain explicit: hard-FP16 packed WV, FP32 visual affine arithmetic
followed by a FP16 return, FP16 exact-text PV, and FP16 final summation. All FP32
operands/logits are temporary and must contribute to actual measured peaks and
time. The shared cache and attention bridges are unchanged.

The Hybrid cache supplies a complete native prefix and appends exact text. Each
read verifies sorted original text positions, the current query tail, and
`first_query_position = visual_token_count + preceding_exact_text_count`.
Together with the Hybrid construction invariant, this permits all visual keys
and a causal mask on exact-text keys without an extra persistent positions
buffer. Extra attention biases or nonstandard masks are rejected by the
existing bridge's `attend` admission.

The CPU operator gate passed four cases: 1/2 bits crossed with GQA 28/4 and 32/8,
using 13 visual tokens. It checks exact persistent-owner identity, an independent
FP64 arithmetic reference conditioned on the source hard-FP16 kernel boundary,
future-value causal intervention, invalid-position rejection, and finite
operands whose unscaled Half text dot overflows while the new joint output
remains finite. The CUDA version of this same gate still requires Root execution.

Six standard `evaluate` P0 smoke jobs are prepared, with max64 and no references
or scoring: M1/M3 use two existing unlabeled TextVQA requests each; M2 uses the
four actual failed IDs from the preserved 105-row interrupted run. Each model is
tested at both 1 and 2 bits, for 16 arm-QA executions total. These smokes and the
CUDA operator gate are not yet results in this CPU preparation. They do not
complete the replacement 500-QA cells, and old predictions cannot be relabeled
as this new class.

Paths, frozen dependencies, CUDA job and the six new configs/jobs are listed in
`audit/Calib_FP32AffineText_v2_preparation_v1.json`. Standard-runner registration
and strict prediction-cache admission are maintained independently by A. Root
alone controls transport, staging, execution and recovery of the main schedule.
