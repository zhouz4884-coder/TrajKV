# Visual K mean compensation: CPU prototype and integration contract

The current MetadataDirectionBank, Round7 KeyBank and Round8Bank compute the visual source mean before encoding, store it as FP16, and form centered WHT keys. Gain/factor folding and sequential residual encoding follow that step. The inspected paths do not subsequently compensate for the mean of the final decoded keys. The reader adds `q·direction_mean/sqrt(128)` to visual scores; its FP16 text-key branch has no corresponding visual header bias.

The new `code/a2/round8_mean_correction.py` reads only actual final packed fields. It decodes each head/chunk with `decode_key`, then every selected K sign stage with the existing sequential FP16 decoder. It averages **all real visual tokens** using K lookup axis0. The source visual FP16 tensors must already have been deleted. The correction is

\[
m=\operatorname{mean}_{i\in\mathrm{all\ real\ visual}}\hat z_i,\qquad
\mu_{\mathrm{dec}}=Q_{16}\bigl(\mu_0-H_{32}m\bigr).
\]

Mean accumulation uses FP64 sums of the final FP16 rows; WHT and subtraction use FP32 before the final header cast. The existing `direction_mean.copy_` slot is overwritten in place. Indices, gains, packed signs, residual amplitudes, V, text, books and the reader remain unchanged. Temporary encoder decoding is released; it is not another retained or serving FP16 visual cache. The old sampled native `.pt` dump is insufficient for this header update: its128 samples must not be mistaken for the full-image mean.

This operation must run **once after parent encoding and before reads**. It is not idempotent: a second application would treat the already-corrected header as a new target and subtract the same decoded mean again. Do not call it per query, or introduce a mean/ray iteration. The provided `mean_corrected_bank_class(Round8Bank)` hook performs one pass after `super()._initialize`, checks unchanged actual owner accounting, and preserves identity when `visual_mean_correction=false`.

For ideal normalized self-adjoint WHT, the final visual reconstruction mean is

\[
\operatorname{mean}\hat K=\mu_{\mathrm{dec}}+Hm.
\]

Its target here is the **stored FP16 source mean** \(\mu_0\), rather than the unavailable exact source mean. If \(c_{32}=\operatorname{fl}_{32}(\mu_0-H_{32}m)\), then a componentwise bound is

\[
|\operatorname{mean}\hat K-\mu_0|
\le 2^{-11}|c_{32}|+2^{-25}
 +|c_{32}-(\mu_0-H_{64}m)|.
\]

The first terms bound normal/subnormal FP16 rounding; the last term measures FP32 computation error against the FP64 reference. Bias relative to the exact source mean additionally includes \(|\mu_0-\mu_{\mathrm{true}}|\), including the original reduction/header quantization error. Nonfinite header proposals fail before writing the slot.

For a fixed query and fixed keys, shifting every key by the same vector adds a common logit and leaves softmax unchanged. Shifting only the visual segment changes the visual/text group log odds:

\[
\Delta\log\frac{P(\mathrm{visual}\mid q)}{P(\mathrm{text}\mid q)}
=\frac{q^\top(\mu_{\mathrm{dec}}-\mu_0)}{\sqrt{128}}.
\]

Within-visual conditional weights remain unchanged in this ideal fixed-query calculation. The reader rounds \(Hq\) to FP16 and performs FP32 reductions, so mean compensation does not certify exact deployed-reader bias elimination. Writing \(e_q=Q_{16}(H_{32}q)-Hq\) and \(r=\mu_{\mathrm{dec}}+Hm-\mu_0\), the corresponding real-tensor mean-score residual is

\[
\frac{q^\top r+e_q^\top m}{\sqrt{128}},
\qquad
|\cdot|\le\frac{\|q\|\|r\|+\|e_q\|\|m\|}{\sqrt{128}},
\]

before the reader's additional accumulation rounding. Full-model query feedback and task accuracy remain empirical questions.

The CPU check is:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
python code/a2/round8_mean_correction_check.py --output outputs/round8_v1/mean_correction_cpu_v1/check.json
```

Choose a new output path when rerunning. Existing evidence is not overwritten. Current checks cover global K4, global K8 and position K4; no/one/two K residual stages; direct versus chunked final decode; all-key translation invariance; mixed-segment visibility and log-odds formula; FP16 residual bounds; unchanged non-mean tensor fingerprints/storage pointers; and a CPU-only subclass fixture. Main L28/Hkv4 mean-header storage remains28×4×128×2=28,672B, with zero additional encoded cache fields. This does not imply zero encoding work or zero Python diagnostic logging memory.

A future **new** runner should register `mean_corrected_bank_class(Round8Bank)` for the existing Round8 format, give the control an explicit false flag and the candidate true, and preserve the normal `complete_round8` input-hash postcheck. Add a separate mean-correction receipt with source fingerprints,28-layer coverage, unchanged J/g/sign/alpha/V/text fingerprints and equal actual owner bytes. Keep fresh output/config names and cap8 alignment. Begin with the old K4/binary-V43-stage baseline; do not combine mean correction with ray/postres or iterative recoding in this first test. No old runner/reader/source file needs editing.

No SSH, GPU/model forward, native dataset diagnostic, quality measurement or serving benchmark was performed for this prototype. Existing bias/centering methods can supply components; neither the algebra nor this CPU fixture establishes novelty or task benefit.
