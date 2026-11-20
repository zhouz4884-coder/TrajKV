# Native model P0 results

All three models passed ten real-image FP16 prefill/first-decode/native-answer
cases. The ten original image dimensions are distinct. Sources were selected in frozen R8 select order, filtering only original
dimensions, before these P0 executions. Their historical exposure is preserved
in the manifest. They are technical-only inputs and contribute no official
benchmark score.

| Model | Fixed input setting | Actual visual tokens | Resident parameter bytes | Inference peak allocated bytes | Native first-token matches |
|---|---|---:|---:|---:|---:|
| M1 InternVL3-8B-hf | native crop_to_patches, max_patches=8 | 1280–1792 | 15,888,747,520 | 16,302,934,528 | 10/10 |
| M2 Qwen2.5-VL-7B-Instruct | official card example min_pixels=200704, max_pixels=1003520 | 888–1240 | 16,584,333,312 | 16,892,775,936 | 10/10 |
| M3 Qwen3-VL-4B-Instruct | original processor min_pixels=65536, max_pixels=16777216 | 672–4032 | 8,875,631,616 | 10,255,327,232 | 10/10 |

Weights were loaded as FP16 without weight quantization. M2 retained native
repetition penalty 1.05; M1's unspecified penalty has native effective value
1.0, and M3 explicitly uses 1.0. Greedy scoring converts logits to FP32 before
the native repetition-penalty processor. Every cache layer's geometry and FP16
dtype were checked. M2/M3 include original three-axis get_rope_index results,
rope_deltas, processor/modality IDs and complete input/output token IDs; M3
retains all three DeepStack features.

These peaks describe the declared ten processed inputs, not every possible
image at the processor's maximum. Load allocation peaks are separately retained
in the raw markers and summary table. No model required OOM recovery or an
architecture substitution for these declared inputs.

Frozen raw runs:

- `results/native_p0/M1_native_fp16_cap8_v2`
- `results/native_p0/M2_native_fp16_card1280_v7`
- `results/native_p0/M3_native_fp16_nativepixels_v7`

The browsable table and exact raw-record digests are
`results/native_p0/native_summary_3models_v1.csv` and `.json`. M1's run originally
recorded an unassigned revision; subsequent full weight/required-metadata
verification maps its exact bytes to the official locked tree. The table keeps
the original run field and the later verified revision separately.

M1's technical run used the unchanged reference environment; M2/M3 used the
project's isolated overlay with that read-only frozen base. The root overlay
receipt verifies common numerical package origins and source SHA identities.
Formal benchmark inference uses the declared overlay and pinned model tree.

Native P0 does not establish compressed-reader faithfulness or public-task
quality. Qwen FP16 bridge gates are recorded separately; their complete token
traces may agree while suffix logits have ordinary FP16 split differences.
Frozen failed attempts document a repetition-default handling bug, UTF-8 output
serialization, and an incorrectly versioned launch path. Each repair used a new
run; original failures remain available and are not marked as model failures.
