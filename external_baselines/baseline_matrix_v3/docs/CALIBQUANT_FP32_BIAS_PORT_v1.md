# CalibQuant visual FP32-bias-reduction port v1

This is a separately identified precision-compatible implementation of CalibQuant Eq. 6 under the matrix's fixed FP16 model/KV contract. It is not a fitted method and does not replace or relabel the preserved published-source FP16 branch.

## Identity and exact change

- Adapter: `methods.calibquant_fp32_bias_adapter.CalibQuantFP32BiasAdapter`, method ID `calibquant_visual_fp32_bias_reduction_v1`.
- Adapter SHA256: `a89f926fc889bfa75ac9b20e80fd4b4877172c099b04dbc771b406c4e5dbb10d`.
- Source: author repository [insuhan/calibquant@1dcfce8](https://github.com/insuhan/calibquant/tree/1dcfce862162e4a3888ef137df65b36cdc19efd0), paper [arXiv:2502.14882v2](https://arxiv.org/abs/2502.14882v2).
- The released `infer.py` defaults to BF16, while this matrix fixes model weights/KV to FP16 for all methods. The original CalibQuant visual uniform quantizer, K/V clip factors, packed byte order, scale/zero dtype, source QK/WV kernels, logit-calibration map and native model weights remain unchanged.

The only new calculation is:

```
bias_products = scale * zero * Q          # original FP16 products
bias = (sum(bias_products, dtype=FP32) / sqrt(D)).to(FP16)
logits = original_scaled_packed_QK - bias # original FP16 logits/calibration
```

No clipping, `nan_to_num`, new calibration search or persistent FP32 copy is added. All persistent owners remain exactly the original two uint8 payloads, four FP16 channel vectors and two small LUTs. Temporary FP32 reduction/division, native reader elapsed time and memory peaks are recorded explicitly.

## Why a separate precision port is needed

Under actual M1/FP16, the published 2-bit path failed on both declared P0 images at decoder layer 27. Q, scaled Q, packed QK dot products and individual channel-bias terms were finite. Casting the bias **sum** to FP16 before dividing overflowed to Inf for 21 or 26 head/query pairs; original calibration then produced NaN. The independent declared-rounding oracle and packed reader had identical nonfinite counts. Prior decoder/MLP values were finite. These original failures remain frozen in `audit/calibquant_M1_model_acceptance_v1.json`, with derived q/state tensors retained only on the remote data volume and independently hashed.

The new port moves only the sum/division boundary before the FP16 cast. CPU controls check byte-identical packed formats/owners, finite-source numerical differences and an isolated finite-product/sum-overflow case. They are labeled CPU controls and are not substitutes for model CUDA evidence.

## C1 pilot support

`audit/calibquant_fp32_bias_C1_support_v1.json` clears **1-bit only** for the declared M1 cap8 pilot, using native checkpoint revision `259a3b64a14623c0ec91a045cb43f7c5af5fa6af` on RTX3090/SM86. Both real inputs (1792 and 1280 visual tokens; 22 and 28 suffix tokens) exercise every decoder layer, the complete question suffix and independent native greedy generation. Same-prefix native DynamicCache and FP16 Hybrid bridges have bitwise-equal suffix logits/post-RoPE Q and equal complete generated IDs. Original full-prefix FP16 cache references are released and packed visual owners remain immutable.

The visual codec owns 6,537,440 and 4,702,432 bytes respectively (1.017892 and 1.025049 bits per visual KV scalar); mapping and exact-text bytes are separately counted by the unified owner audit. The full-attention declared-rounding oracle maximum error is 0.00390625. Independent FP64 Eq. 6 forms agree exactly in the observed cases; maximum relative L2 deviation of the implementation's uncalibrated logits from ideal FP64 is about 0.000713. These numerical checks do not establish official task quality.

Finite source-logit differences are also retained, rather than hidden: changing the bias cast boundary can produce absolute differences up to 8 on large-magnitude calibrated FP16 logits. This port has its own method identity; predictions and scores must never be mixed with the old FP16 branch.

## C2 status and instrument limits

The new 2-bit port completed the second input under the original instrument but tripped its fixed full-attention tolerance on one element of the first input, despite finite arithmetic. A focused frozen-input diagnostic distinguishes implementation fidelity from ideal FP64 rounding error and preserves actual/reference values, ULP distances, relative L2 and error quantiles. Output ULP counts near cancellation are not interchangeable with raw-kernel ULP counts. The short diagnostic also verifies complete finite suffix/free generation; it does not silently widen the original threshold or convert a failed instrument run into a task score.

C2 remains a separately identified pending point until its finite-precision contract is accepted. It does not block the already cleared C1 pilot. The original published-FP16 C1 pass, original C2 overflow, new precision-port results and diagnostic/instrument failures all remain separately versioned. No public-task labels or scores were used to choose parameters.
