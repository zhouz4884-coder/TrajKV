# KIVI source-example g32/R32 visual point

This is a separately named external baseline point. It preserves the old g128/R0 technical anchor unchanged. The pinned [KIVI `example.py`](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/example.py) sets K/V to 2 bits, group size 32 and residual length 32. This is the **pinned source example configuration**, not a claim that every original-paper experiment used these settings.

`configs/methods/kivi_example_source_v1.json` verifies the local normalized source bytes against the pinned Git blobs for `example.py`, `models/llama_kivi.py` and `quant/new_pack.py`. `code/methods/kivi_example_adapter.py::KIVIExampleAdapter` admits only b2/g32/R32, FP16, batch one and head dimension 128. Its production encoder requires CUDA and calls the original Triton min/max/packing function unchanged. It does not insert a CPU substitute.

## Source policy and scope

The source prefill policy splits K by `residual_length`: `N_K_quant=N-N%R`, with the remaining `N%R` keys in FP16. It quantizes `max(0,N-R)` values and retains the last `min(N,R)` values in FP16. At this point R=G=32; using `N%G` happens to agree, but the port explicitly retains the source's R-based rule.

K is grouped across 32 token positions per channel. V is grouped across 32 channels within each token: **four independent scale/min pairs per D128 token**. Both axes use the official FP16 min/max reconstruction and little-endian int32 packing. Exact text KV remains outside the visual owner. The new packed reader keeps original positions, exact text, GQA and causal masking.

The original KIVI Llama system quantizes the rolling whole cache. This visual port freezes the image prefix and leaves all subsequent text FP16, consistent with the external matrix's comparison protocol. The retained FP16 visual tails therefore remain part of that immutable visual owner. It is not labeled as the complete original rolling Llama implementation.

## Independent kernel and accounting

`code/methods/kivi_example_reader.py` is independent of the frozen g128 reader. Its GROUP constexpr controls K scale/min token indexing. V metadata uses `owner*N_vq*(D/G)+token*(D/G)+channel/G`; broadcasting one g128-style scale across an entire D128 token would be incorrect.

For batch one, H KV heads, dimension D, group G and b payload bits, per-layer physical bytes are:

```
payload = H*D*(N_K_quant+N_V_quant)*b/8
metadata = 4*(H*D*(N_K_quant/G) + H*N_V_quant*(D/G))
FP16 tails = 2*H*D*((N-N_K_quant)+(N-N_V_quant))
original visual positions = 8*N
```

`owner_report` counts unique actual storages, including both scale/min arrays, FP16 tails and positions. The global cache audit additionally counts common visual positions and sequence metadata once. At g32 the two FP16 metadata values alone cost 1 bit per quantized scalar; this point cannot be reported as 2.0 or 2.25 effective bits.

For H=4/D=128, per-layer state accounting gives 3.875 effective bits at N=256, 3.225 at N=1280 and 3.17857 at N=1792, before the cache's separately counted common position map. These are geometry/accounting predictions, not measured model scores. Very short prefixes can retain most/all values or keys in FP16 and cost substantially more. Every actual result must retain its measured owner bytes rather than replacing them with this asymptotic formula.

## Current acceptance and next real gate

The CPU check passed 48 synthetic source-codec/decode/owner/reference-attention cases across N=17/31/32/33/63/64/65/257/512/1008/1792/4032, GQA 28:4 and 32:8, and Q=1/9. Both original source prefill implementations are AST-checked at 13 lengths. Distinct distributions in V's four channel groups expose one-scale broadcasting errors. CPU controls use the author's PyTorch codec functions, then independent source decoding; they do not execute the production Triton encoder or GPU reader. `audit/kivi_example_cpu_acceptance_v1.json` states **CUDA pending / no model quality**. Detailed CPU evidence is on E.

Root can stage the new modules and run the actual CUDA gate, without loading an 8B model:

```text
MAIN_PYTHON code/gates/kivi_example_gate.py --device cuda:0 --output NEW_DATA_DISK_RESULT.json
```

On CUDA the gate calls the untouched official encoder, checks packed codes/scales/minima against the author-source oracle, and disables `state.materialize` while exercising the actual packed reader. It compares with independent original-source decode plus Torch SDPA, checks original-position masking/tails and charges all owner bytes. The gate is implemented but **has not been launched by this CPU-only task**.

After CUDA acceptance, add the new module/class to the separately reviewed registry and cache-source policy, then run real model/suffix/answer smoke before expanding. This task did not change `evaluate.py`, its allowlist, the old KIVI adapter/reader or the existing cache policy. The method configuration is `configs/methods/kivi_example_K2V2_g32_R32_v1.json`; its status remains CUDA/E2E pending.
