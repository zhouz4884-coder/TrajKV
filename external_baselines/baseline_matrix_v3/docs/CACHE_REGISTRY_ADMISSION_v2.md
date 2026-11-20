# Bounded adapter-registry admission

`evaluate.py` adds exactly three literal module/class pairs to its existing five-entry allowlist:

| Module | Class |
| --- | --- |
| `methods.kivi_example_adapter` | `KIVIExampleAdapter` |
| `methods.turboquant_buffered_adapter` | `TurboQuantBufferedAdapter` |
| `methods.turboquant_onlyterp_mse_adapter` | `OnlyTerpMSEAdapter` |

Every other evaluator AST node, including the entire stat-binding `model_identity` implementation, is unchanged from source `c4d73e61efff308051dd4f5e7c450ae4d38f7dc3406f61456b26e935a1423ed4`. The resulting source is `baff0e908230ed9d96547107ff36e0ea543128ac9a8da033329209d9ff59da22`. Exact before/after files are retained under `audit/source_snapshots/`. No existing worker snapshot, original method, input or result was overwritten.

This is registry admission, **not CUDA, model, quality or main-matrix approval** for any new method. Root schedules those independent gates. No GPU job was launched by this compatibility task.

## Restricted numerical equivalence

The new immutable `configs/cache/evaluate_identity_equivalence_v2.json` retains the two previously reviewed I/O variants and adds only this exact new evaluator source. The cache verifies all three entire-file SHA values, each exact `model_identity` AST SHA, and the complete numerical AST after only these transformations:

1. Replace the already-reviewed identity I/O function with its existing fixed sentinel.
2. Require every legacy literal registry entry to remain; allow only the three fixed additions above; remove those additions for the legacy numerical-AST comparison.

An unknown source version, modified identity function, missing legacy entry, fourth extra pair or other numerical statement is rejected. The policy is not a general instruction to ignore registry/validation changes. The original v1 policy remains byte-identical, SHA256 `05a456a447c3d9fd7eac14d6cc1d2b0d98abcf5c67dfa3c48ec0268f95673703`.

Existing branches retain the original semantic fingerprint and numerical-AST SHA `c96836da3d6a22df0d346deafada31f690015e6e2e227f99641fbe51fe3330cb`. Actual whole-source SHA values remain in the immutable raw/source proofs. This prevents a membership-only registry extension from forcing repeated execution of unchanged F/K/C/E predictions.

## Independent method contracts

Method module/class, constructor arguments, per-layer seeds and relevant implementation/upstream fingerprints remain part of each effective prediction contract. The buffered 0xSero variant and OnlyTerp-MSE variant cannot reuse the unbuffered T0 predictions, even if a caller accidentally reuses a display ID.

- KIVI source-example: its new adapter/reader, official quantizer, pinned `example.py` and frozen source-default proof. The upstream Llama policy is documentary; the launcher intentionally omits model directories. Its GPU gate parses a byte-exact SHA-locked licensed copy at `audit/upstream/kivi_example_llama_kivi_source_v1.py`, not an unavailable snapshot model path. `audit/kivi_example_snapshot_fix_v2.json` preserves the old gate and records the packaging-only fix; the algorithm/encoder/reader are unchanged.
- Buffered 0xSero: its new wrapper, the unchanged original adapter, its executed upstream Python/JSON files and source guard lock. Its constructor/window and new source module create a distinct contract from T0.
- OnlyTerp-MSE: P confirmed the self-contained adapter source `ef52ac83eaabae8cbc41c46bc976bc6d6fcb696f2897bcb2bda534e2431ee86e`, the v2 source guard and `third_party/turboquant_onlyterp/src/cache.py`. The dependency map rejects unreviewed adapter revisions. Unused experimental kernels, LUT/vLLM paths and the extensionless LICENSE are not invented runtime dependencies. The license remains separate source provenance. This dependency confirmation is based on source review/CPU controls; it does not claim GPU compatibility.

The v2 policy's `pending_dependency_maps` annotation records its initial preparation state. The later `audit/cache_registry_v2_complete_maps_acceptance.json` resolves the OnlyTerp source-map confirmation without changing the policy's admitted evaluator sources. The initial pending-map helper and its 21-case acceptance are preserved; the final helper has a new source fingerprint and 23-case acceptance.

## CPU evidence

`audit/cache_registry_v2_complete_maps_acceptance.json` passes 23 CPU checks, producing zero new model answers. For each of the four original M1 TextVQA methods F/K/C1/E, the same 128 P1 predictions still match the fixed P2 cohort and leave exactly 372 missing questions. Their entire original effective-contract SHA values are unchanged. Changed cap, constructor, prompt, executed reader, numerical program or unreviewed source still invalidates reuse. Structural new-method map checks are explicitly in-memory controls, not fabricated GPU raw runs.

Root should stage the new evaluator, cache engine, v2 policy, complete dependency files and the g32 audit fixture together. Already-running queues continue from their own immutable source snapshots.

## Snapshot policy location

The frozen launcher copies `configs/methods` into each worker source snapshot, but not `configs/cache`. The helper now checks the snapshot/source root first and, if the reviewed file is absent, uses the launcher's `BASELINE_DATA_ROOT/configs/cache`. The entire canonical JSON is locked for both v1 and v2; a changed policy is rejected and cannot silently fall back to a weaker version. LF/pretty-print changes alone are allowed because their parsed policy content is identical. No launcher, running queue or inference method was changed for this layout fix.

`audit/cache_registry_v2_layout_acceptance.json` is the final 26-case CPU acceptance. It retains all 23 compatibility/dependency tests and adds snapshot-without-policy resolution, equivalent JSON formatting acceptance and altered-policy rejection. The previous 21/23-case reports and helper sources remain preserved. Root's separate real P2 harvester results are not attributed to these CPU tests.
