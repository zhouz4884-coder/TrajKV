# OnlyTerp Hmix12 fixed-source 8/64 plan

The point is the source Algorithm-1/MSE visual variant with QJL disabled. It is not prod and is not a reproduction of the paper's 2.5-bit LongBench table. The original two real TextVQA answers are retained unchanged; natural language output alone does not establish quality.

The first expansion is Text4 plus Info4. Normal `SourceRun` proof verifies two actual Text cache hits; the new GPU work is Text2 plus Info4. Target configs and actual missing jobs are listed in `configs/matrix/OnlyTerp_Hmix12_stage8_reuse_v2.json`. Original drafts, probe config and raw bytes are unchanged. The new configs omit the old descriptive `stage_role` field, while the approved helper accepts that field as string-only metadata for the original probe. No local monkeypatch is used.

After the first eight fixed questions all have valid same-contract records, refresh with `ops/prepare_onlyterp_pilot_reuse_v2.py --phase stage64 --cached-text PROBE2 TEXT_MISSING2 --cached-info INFO4 --output NEW_E_DIRECTORY --run-id NEW_ID`. It checks the original first four in **each** dataset before writing any 64-phase missing job. The normal projection is Text32 with four hits/28 missing and Info32 with four hits/28 missing. The actual count is recomputed from real sources; it is not assumed. A pre-completion CPU check correctly blocked the 64 phase at two missing Text and four missing Info answers.

Every derived view copies original records byte for byte, retaining actual packed-owner bytes, the FP32 materialize reference reader's transient allocations, CUDA peaks and instrumented request timing. These quality-run timings are not the controlled speed benchmark. TextVQA ten-vote soft accuracy and InfoVQA's fixed ANLS protocol are scored separately; no cross-dataset quality mean is reported. Hmix23 and Huniform1 remain declared, untested choices awaiting Root's decision.

## Dense QR feasibility, without implementation

The primary paper explicitly uses Gaussian-matrix QR to construct its random rotation in [§3.1](https://arxiv.org/html/2504.19874v1#S3.SS1). The pinned third-party `src/cache.py` exposes this as `rotation_mode='dense'`, with independent 96/32 rotations for mixed channels. Dense QR therefore matches that theoretical rotation model more directly than the practical Hadamard/sign variant. This does not establish a quality improvement.

It is not a one-parameter change to the current packed owner: the owner stores Hadamard sign bits, assumes 128 padded regular coordinates and uses the corresponding codebook. A faithful dense variant would need its own matrix representation and 96-dimensional regular codebook/storage widths. No such change is made here.

Mixed1/2 dense payload would be `(96+2*32)/128=1.25` bits, saving 0.25 versus padded Hadamard. But one shared K/V set of FP32 96×96 and 32×32 matrices costs `4*(96²+32²)=40,960` bytes per layer/head. For M1's 28 layers and four KV heads this is 4,587,520 GPU bytes, before any retained CPU duplicate. Its amortized visual scalar cost is `1280/N` bits: about 0.714 at N=1792, which exceeds the 0.25 payload saving. These are source/geometry calculations, not GPU measurements. All required constants and any temporary regeneration cost would need honest accounting.

The recommendation is to finish the current Hadamard Hmix12 8/64 evidence first. Dense QR can later be a separate higher-byte theoretical-fidelity control; it should not replace or silently rename this approximately 1.76-bit source point.
