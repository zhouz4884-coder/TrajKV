# Native Qwen interface audit

This records the installed Transformers 5.12.0 source retrieved read-only from
the original data-volume environment. It establishes an adapter contract, not a
claim that native P0 or compressed inference passed.

## Locked checkpoints and actual geometry

| Model | Revision | Native architecture | Layers | KV heads | Head width | RoPE | DeepStack |
|---|---|---|---:|---:|---:|---|---|
| M2 Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 | Qwen2_5_VLForConditionalGeneration | 28 | 4 | 128 | sections 16,24,24 | none |
| M3 Qwen3-VL-4B-Instruct | ebb281ec70b05090aa6165b016eac8ec08e71b17 | Qwen3VLForConditionalGeneration | 36 | 8 | 128 | interleaved sections 24,20,20 | visual layers 5,11,17 |

Both upstream configurations declare BF16. The matrix explicitly loads FP16 for
all methods, records the override, and does not use weight quantization. The
shared inventory only had GPTQ/AWQ variants of Qwen2.5-VL; those are ineligible.
The FP16 checkpoints must pass full index/shard integrity before native loading.

Official configurations: [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/config.json),
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/config.json).

## Position and prefix contract

Both installed `get_rope_index` signatures require `input_ids` and
`mm_token_type_ids`; the processor supplies the latter (text=0,image=1,video=2).
Passing image grids without modality IDs raises a native error. Save the exact
three-axis native positions and `rope_deltas`, input IDs, image grids, and native
visual-token indices. Image IDs are 151655; boundary IDs are 151652/151653.

The whole native image forward must execute. Qwen3 vision returns its ordinary
merged image embedding and three DeepStack embeddings. The language model adds
these at its first three decoder layers over the visual mask. Retaining only
the last vision output, or bypassing the first three additions, is invalid.

The causal prefix may end at the last native visual token plus one. It can be
captured from a complete native forward and cropped at that boundary; future
questions cannot influence that causal prefix. Recompute every question suffix
using the original native position slices and unchanged RoPE convention. Token
eviction changes physical storage indices, not the original positional values.

## Attention hook and mask requirements

For M2, native attention applies `apply_multimodal_rotary_pos_emb` to Q and K.
For M3, native per-head Q/K RMSNorm precedes `apply_rotary_pos_emb`, whose cos/sin
already implement interleaved mRoPE. Both then call
`past_key_values.update(K,V,layer_idx)` and finally dispatch through
`ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager)`.

A custom backend therefore receives **post-RoPE Q/K and original V** without
reimplementing RoPE, QK norm, vision, or DeepStack. The backend can implement
packed visual logits and weighted values while retaining exact FP16 text KV.
CalibQuant needs visual-logit calibration before the joint text/visual softmax;
dequantizing a cache and invoking unmodified SDPA is insufficient.

Registering an attention tag alone is unsafe: `masking_utils.py` checks its mask
registry and returns no mask for unknown tags. Register both interfaces:

```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
ALL_ATTENTION_FUNCTIONS.register(tag, backend)
ALL_MASK_ATTENTION_FUNCTIONS.register(tag, eager_mask)
```

The custom backend must honor the returned causal/additive mask, including the
causal relation within a multi-token question suffix. Native vision attention
also uses the global attention registry; dispatch on the language-attention
module or set only the language config tag so vision remains native.

The implemented `attention_bridge.py` instead temporarily intercepts the local
`sdpa` entry by exact language-module identity and restores it on exit. This
keeps the original SDPA mask factory and delegates vision to the original
function. The two-registry warning above applies if a new tag is introduced.

`Cache.update(K,V,layer_idx)` returns K/V for attention. Additional mask-facing
interfaces are `get_seq_length(layer_idx=0)`,
`get_mask_sizes(query_length,layer_idx)`, `get_max_cache_shape(layer_idx=0)`, and
`is_sliding`. Preserve logical sequence length separately from retained visual
storage length. Missing logical positions may produce both wrong masks and
wrong decode mRoPE deltas.

## P0 evidence and limitations

`native_model_smoke.py` loads original weights and native processor, checks every
layer's FP16 cache geometry, captures native three-axis positions/rope deltas,
checks actual DeepStack inputs, measures resident parameter storage and CUDA
peaks, and compares the first manually selected greedy token with native
`generate`. The fixed 10 images have distinct original dimensions and come from
frozen historical R8 select sources; they serve only implementation/hardware
checks and never contribute benchmark task scores.

For incremental manual decode, continue the saved native text positions rather
than relying on a full-length attention mask to infer a one-token position. The
complete answer still uses native `generate`.

This gate alone does not verify the compressed reader, byte owners, or suffix
equivalence. Those must pass separately before P1.

## Frozen native/bridge evidence

M2 `M2_native_fp16_card1280_v7` passed all ten native FP16 cases using the
official card's 256–1280-token pixel-range example. Actual visual counts were
888–1240. Resident parameters were 16,584,333,312 bytes and the inference peak
was 16,892,775,936 bytes. All ten greedy first tokens matched the native
generation contract with repetition penalty 1.05 and FP32 logits processing.
`M2_fp16_native_bridge_v2` passed both real-image bridge cases: the 1240-token
document's full nine generated tokens and the 962-token scene image's full
three generated tokens matched native HF. Both suffix-logit maximum absolute
differences were 0.0234375. Original suffix/decode positions and complete prompt
history were checked; this is finite-case token-trace evidence.

M3 `M3_native_fp16_nativepixels_v7` passed all ten native FP16 cases under the
official image processor's 16,777,216-pixel maximum. Actual visual counts were
672–4032; all ten first greedy tokens matched native generation and every
prefill retained all three DeepStack feature maps. Resident parameter storage
was 8,875,631,616 bytes; the measured inference allocation peak was
10,255,327,232 bytes. This is a technical gate, not a public task score.

`M3_fp16_native_bridge_v2` then passed two real-image FP16 bridge cases: the
4032-token document produced the same complete nine-token generation and the
736-token scene image produced the same complete three-token generation as
native HF. Original question positions and all subsequent decode positions
matched the native three-axis contract. Raw suffix logits differed by at most
0.046875 and 0.0390625 respectively. The evidence establishes these two token
traces, not universal equivalence or bitwise-equal logits.
