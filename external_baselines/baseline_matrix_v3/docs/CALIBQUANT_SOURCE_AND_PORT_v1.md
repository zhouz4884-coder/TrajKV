# CalibQuant source audit and faithful visual adapter

Audit date: 2026-11-19. This document records a port and its gates; it is not a claim of paper-score reproduction.

## Identity and pinned source

- Paper: [CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs, arXiv:2502.14882v2](https://arxiv.org/abs/2502.14882v2), 2025-03-24 revision. The paper links `insuhan/calibquant` as its code.
- Author repository: [insuhan/calibquant](https://github.com/insuhan/calibquant), MIT license, pinned at [`1dcfce862162e4a3888ef137df65b36cdc19efd0`](https://github.com/insuhan/calibquant/tree/1dcfce862162e4a3888ef137df65b36cdc19efd0), dated 2025-04-03 UTC.
- Unmodified upstream checkout: `third_party/calibquant`. `audit/upstream/calibquant_source_lock_v1.json` separately records Git blob identities, checkout byte SHA256 and normalized LF SHA256.
- New adapter: `code/methods/calibquant_adapter.py`; independent gate: `code/gates/calibquant_math_gate.py`. Old TrajKV/VisContract ports are neither imported nor used as acceptance evidence.

## Released algorithm and format

The released CalibQuant branch in [patch_internvl.py](https://github.com/insuhan/calibquant/blob/1dcfce862162e4a3888ef137df65b36cdc19efd0/patch_internvl.py#L219) quantizes **post-RoPE visual K and V**, independently for each batch/head/channel, reducing minimum and maximum over the complete visual token axis. It retains system/text KV at original precision. The released 1-bit range multipliers are K=0.5 and V=0.3; other supported bitwidths use 1.0. The scale is `(max-min)/(2**bits-1)` after applying that multiplier, zero point is `-min/scale`, and integer codes use `round(clamp(x/scale+zero,0,2**bits-1))`.

Codes are packed along head dimension into uint8, most significant code first. One byte holds eight 1-bit values, four 2-bit values, two 4-bit values or one 8-bit value. Scale and zero are channel vectors, **not 1-bit fields**; LUTs and all these vectors count toward stored bytes. No rotation, trained codebook, residual outlier store or QJL sketch belongs to the CalibQuant branch. The repository contains a separate QJL baseline; `infer.py` currently selects `quant_method="qjl"` by default, so blindly running it does not execute CalibQuant.

The source defers affine dequantization into QK and WV. With code matrices Ck,Cv, channel scales Sk,Sv and zero points Zk,Zv:

```
visual_logits = ((Q * Sk) @ Ck.T - sum(Q * Sk * Zk)) / sqrt(D)
visual_output = (Wvisual @ Cv) * Sv - Sv * Zv * sum(Wvisual)
```

For bits <=2 the source then calibrates **visual logits before combining them with text logits and softmax**: for each head/single query, min is mapped to min and max to max-3 with the released `+1e-8` denominator. This can change visual/text probability mass and cannot be reproduced by replacing K,V with dequantized tensors in ordinary SDPA alone. See [released normalization and WV](https://github.com/insuhan/calibquant/blob/1dcfce862162e4a3888ef137df65b36cdc19efd0/patch_internvl.py#L280).

The source does not clamp this affine map. Its slope is `(range-3)/(range+1e-8)`: a positive range below 3 gives a **negative slope and reverses visual-logit ordering**; range=3 collapses scores to the minimum; range>3 preserves ordering. At zero range, the FP16 `1e-8` denominator can underflow, yielding nonfinite scores. These are properties of the frozen released rule, not silently corrected by the port. They must remain visible in numerical diagnostics and any failed run.

## Calibration and data exclusion

The [paper, Section 4.2](https://arxiv.org/html/2502.14882v2#S4.SS2) describes a grid search over integer calibration offsets in {0,1,2,3}, fixed across prompts/layers/heads, and examines COCO Caption attention distributions. Its experiments include InternVL2.5 and LLaVA and use H100 GPUs. The source release supplies fixed constants, but no calibration-data download, split manifest, search runner or model-specific calibration artifact. Therefore the exact author calibration provenance cannot be independently reconstructed from this release.

This port **freezes the published source constants without fitting on any new data**. Evaluation TextVQA, InfoVQA, DocVQA and ChartQA images/questions/answers are not used for calibration or method selection. Model-native P0 inputs diagnose execution, positions and arithmetic only; no parameter is selected from their task scores. Thus no separate calibration image array is required for the published-default port. A future recalibrated variant would require a new declared training-only source manifest, image/source disjointness checks against every evaluation split, a frozen objective/grid and a separate method/version label before scores are inspected.

## Model and execution compatibility

The released monkey patch specifically expects `InternVLChatModel.generate`, `InternLM2FlashAttention2.forward`, fused `wqkv`, `wo`, its older rotary API and a fixed `SPROMPT_LEN=41`. It targets InternVL2.5-8B/26B. These are concrete reasons it cannot be applied unchanged to HF InternVL3, Qwen2.5-VL or Qwen3-VL. They do not establish that the quantizer is mathematically incompatible.

The new adapter consumes the native model's **already-positioned** Q/K and original V. It preserves KV heads, repeats only transient reader inputs for GQA, and does not replace the native mRoPE or DeepStack path. Original visual positions and full-precision text KV remain engine-owned. Compatibility requires a model hook that obtains post-RoPE Q and integrates calibrated visual logits plus WV at every decoder layer. If a model adapter can only mutate a Hugging Face Cache and leave SDPA intact, that path cannot honestly be labeled CalibQuant.

The released `infer.py` loads BF16 model weights while its packed kernels emit FP16 intermediates and its attention patch casts back to BF16. The matrix fixes the entire comparison to FP16 model weights and KV, including this adapter. This declared dtype change is part of the matrix protocol and must not be presented as reproducing the author BF16 runtime exactly; no weight quantization is involved.

The matrix protocol captures an image-only prefix, compresses it and recomputes every question suffix token. The released script compresses visual KV after the entire prompt prefill and uses compression for autoregressive decode. The port is explicitly a **faithful visual adapter under the shared suffix replay protocol**, rather than a byte-identical execution of the author prompt protocol. For a batched suffix, calibration is applied independently per query as repeated released single-query reads; reducing extrema jointly over suffix queries changes the method.

## Packed reader and owner contract

`CalibQuantAdapter(bits=1, reader_backend="upstream_triton").encode(k,v)` returns a state with:

- `read(q)` / `attention_logits(q)`: calibrated visual logits `[B,QH,T,Nvisual]`.
- `weighted_value(visual_probabilities)`: visual contribution `[B,QH,T,D]` after a **full text+visual softmax** performed by the engine.
- `owner_tensors()` and `byte_report()`: exactly two packed payloads, four channel scale/zero arrays and two tiny LUTs. The original visual FP16 storage is neither referenced nor aliased. All token positions remain in input order; engine-owned positions must be counted once in the total system owner audit.
- `manifest()`: exact source revision, constants, no calibration-data use and protocol/runtime differences.

The default reader calls the unmodified source's Triton packed QK and WV, looping over suffix queries because the released kernels assume q_len=1. The source has no SM90/FP8 intrinsic requirement; its bit operations and FP32 reductions are candidates for SM86, subject to real compilation and numeric gates. H100 measurements are not 3090 evidence. The source WV kernel omits its computed column-load mask; head_dim=128 gives 16 bytes at 1-bit and all released block sizes divide that width, avoiding the partial-column case for the declared model dimensions. Other dimensions require a separate guard/port and new evidence.

`torch_unfused` is a clearly labeled fallback runtime implementation of the same arithmetic. It temporarily expands integer codes to FP16 for matmul and charges those allocations and elapsed time. It stores no dequantized FP16 visual cache. This variant cannot support claims about the published fused-kernel throughput.

Constant/nonfinite channels make the released scale/zero calculation undefined; this adapter records a method failure instead of silently changing the quantizer. The canonical integer LUT avoids the source's fragile discovery from only the first 10,000 codes; it preserves the complete legal code mapping and its few extra entries are charged.

## Acceptance gates and current limits

1. The independent gate extracts unmodified `quant` and `encode` through AST, compares packed payloads and channel metadata, and verifies unpacked codes exactly.
2. Float64 oracles using independent source-quantized codes verify post-scale QK, the released single-query calibration and WV including competing text logits. Tests cover GQA geometries, non-power-of-two token lengths and 1/2-bit formats.
3. The owner gate checks real storage sizes, absence of aliases to original K/V, and collection of original inputs. CPU runs are explicitly `CPU_CONTROL_ONLY`.
4. Real CUDA math/owner gates and full-model native FP16 bridge, all-layer query-hook, multi-token visual suffix replay, generation and source-paired quality gates are required before benchmark `PASS`. A math-only result does not unlock quality by itself.

## Observed CUDA math/owner result

`results/p0/calibquant_math_cuda_v1.json` passed **12 actual RTX3090 SM86 cases**, using the unmodified upstream packed Triton QK/WV. The cases cover N=17/257/512, KVH/QH=4/28 and 8/32, 1/2 bits and three independent suffix-query reads. All packed payloads and quantization parameters match source-extracted official functions; original FP16 inputs are released and no owner aliases their storage. The independent float64 integer-code oracle gives maximum absolute error 0.0009765625 for uncalibrated QK, 0 for calibrated visual logits and 0 for WV. These are numerical checks, not task scores.

The observed read-only runtime is torch 2.4.1+cu121, CUDA 12.1 and Triton 3.0.0. This establishes execution for the declared cases despite the README's Triton 3.2.0 installation example. First compilation and official autotuning are included in the 424.73-second gate wall time; no throughput claim is inferred from it. Peak tensor allocation for the entire gate including autotuning is 269,958,144 bytes, which is distinct from persistent compressed KV owner bytes.

`audit/calibquant_cuda_acceptance_v1.json` checks 15 conditions, including live process argv, actual cache-environment values on the data volume, immutable snapshot hash, source dependency fingerprints, exact result/log retrieval and worker termination. Its status remains `PASS_CUDA_MATH_OWNER_ONLY` with `quality_unlocked=false`: model-native FP16 bridging, complete-layer hooks and actual visual suffix replay remain required. No new quality result, speedup or successful model integration is claimed by this source audit.
