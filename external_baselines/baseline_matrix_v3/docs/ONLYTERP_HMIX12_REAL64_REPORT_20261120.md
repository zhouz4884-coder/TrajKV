# OnlyTerp Hmix12: real 64-question result

The fixed pilot is complete: Text32 and Info32, all 64 genuine GPU answers closed with `PASS` and independently verified raw/source SHA chains. The task generated 62 new answers and reused the original two-question probe; the eight-question intermediate view is part of the same 64, not extra independent data. Inputs, FP16 model, native preprocessing, max-new-tokens64 and the source MSE/QJL0 method stayed fixed. Original drafts and source records remain unchanged.

| Dataset / metric | Hmix12 | Same-source FP16 | Paired difference, pp | 95% cluster-bootstrap CI for difference |
| --- | ---: | ---: | ---: | --- |
| TextVQA ten-vote soft, 32 sources | 45.000% | 81.875% | −36.875 | [−54.375, −20.625] |
| InfoVQA fixed vendor ANLS, 32 sources | 16.607% | 75.833% | −59.226 | [−75.313, −41.368] |

These are separate metrics and are not averaged together. The four-question intermediate scores, 75% Text and 50% Info, were optimistic relative to the completed fixed 32-source cohorts. Every answer remains in its denominator; no low-scoring case was turned into a technical failure or dropped.

The measured complete visual-owner rate is 1.75733 mean bits for Text, range 1.75706–1.76053, and 1.75745 for Info, range 1.75649–1.75901. Every request satisfies the 1.8-bit ceiling. The mean complete visual owner is 10,580,528 bytes in each selected cohort. Owner accounting retains packed payload, norms, maps, shared CPU/GPU codebook constants and positions.

The explicit reader remains `torch_materialize_fp32_reference`. Actual peak CUDA allocated bytes are 16,289,841,664 (Text) and 16,400,157,696 (Info). The largest reported decoded K/V temporary for a single layer is 7,340,032 / 9,437,184 bytes respectively; this is not every temporary and is not summed across sequential layers. Original raw records retain the entire memory report and actual instrumented timing. Mean quality-run wall time is 4.757s Text / 14.776s Info, with mean generated lengths 3.5 / 15.969 tokens; three Info cases reached the common 64-token cap and remain scored unchanged. These concurrent, audited quality-run timings are not controlled speed measurements or optimized-paper kernel performance.

The correct interpretation is a real low-budget **OnlyTerp source Algorithm-1/MSE practical Hadamard variant, QJL disabled**. It is not TurboQuant-prod or the original paper's 2.5-bit table. The source arithmetic/CUDA support and complete execution passed; the quality loss is a measured property of this point under the common visual-only protocol. It does not establish failure of the original paper or of higher-bit variants.

My recommendation is to retain this validated low-budget point but not expand it alone as the strongest representative of the family. The same first32 verified P1 C1 point is descriptively 58.75% Text / 53.854% Info at about 1.022 bits; K2R0 is 79.375% / 75.104% at about 2.315 bits. These are pilot comparisons, not claims about full-split dominance. Root's already-declared higher-bit source Hmix23 should accompany the low-bit point in the next coverage decision. No new width, algorithm repair, dense rotation or own-method experiment was run by this task.

For a concrete next step, `configs/matrix/OnlyTerp_Hmix12_P2_500_optional_v4.json` prepares Text500 and Info500 from exact actual 32-hit contracts, leaving 468 new answers per dataset. Neither optional 500 job is enqueued or launched. Root decides whether/when to run them together with the remaining source/budget coverage.

Results and paired evidence: `results/matrix/OnlyTerp_Hmix12_real64_v1.json` / `.csv`, `audit/OnlyTerp_Hmix12_Text32_vs_F_v3.json`, `audit/OnlyTerp_Hmix12_Info32_vs_F_v3.json`. Original raw, source proofs and exact derived/scored views are mirrored on E and canonical worker output remains on the 8TB data volume.

## Scheduler handoff

Only verified CPU schedulers were paused. No GPU worker was signaled, killed or restarted. The v6 boundary contained 40 active/pending original tasks; final conservation verifies three naturally completed and every other original task preserved in the live plan or its original wait state. All original node objects and relative order remain exact, apart from the explicitly completed own-task pilot nodes retired from their prior pending slots.

The live controller is `bm_External_baselines_P2_Fbank_OT64_E8_queue_v8`, PID1048149, using `configs/queues/External_baselines_P2_Fbank_OT64_E8_queue_v8.json`. GPU0 full DocVQA worker PID1041987 continues naturally. After OnlyTerp64, GPU1 naturally started original full ChartQA worker PID1051751. The ready full Text4500 alias follows Chart on GPU1; ready full Info2301 follows Doc on GPU0. Their original unused job drafts are explicitly marked not to run, the original configs/output paths are unchanged, and unused canonical outputs were checked before aliasing. Only the four newly ready E-window8 missing jobs were added; the other twelve pre-existing budget jobs were not duplicated. No formal timing job was inserted.

The node proof is `audit/OnlyTerp64_final_node_conservation_v1.json`; alias mapping is `configs/matrix/P3_M1_F_affinity_aliases_v3.json`. Queue journals, pauses and every earlier controller/plan remain intact. Root can resume ownership of the live controller after this completed handoff without stopping either active GPU worker.
