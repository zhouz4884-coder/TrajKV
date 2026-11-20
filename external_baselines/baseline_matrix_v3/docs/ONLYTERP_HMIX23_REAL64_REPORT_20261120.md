# OnlyTerp source Hmix23: completed real 64-question pilot

The existing higher-precision source MSE point is now measured on the same fixed Text32 and Info32 cohorts as Hmix12. All 64 original GPU records are `PASS`; no failure, truncated answer or hard question was dropped. The probe2, six additional stage8 questions and 56 additional stage64 questions are one 64-question corpus, not 72 independent answers. Original source modules/algorithms, old parameter drafts, max-new-tokens64, native FP16 model and complete question-suffix protocol remain unchanged.

| Dataset / metric | Hmix23 | Same-source FP16 | Paired difference, pp | 95% paired source-cluster CI |
| --- | ---: | ---: | ---: | --- |
| TextVQA ten-vote soft, 32 sources | 78.4375% | 81.875% | −3.4375 | [−10.625, +3.750] |
| InfoVQA fixed vendor ANLS, 32 sources | 73.125% | 75.8333% | −2.7083 | [−11.458, +4.792] |

The metrics stay separate. CI crossing zero is not evidence of equivalence. The earlier four-source 100% Text / 81.667% Info observations were intermediate results; the fixed 32-source values supersede them for this pilot. The same-source low-budget Hmix12 values were 45% / 16.607%. Thus the family has an implemented higher-precision point with substantial measured quality recovery; the low-bit point must not stand in for the entire source family.

Hmix23 is the third-party **OnlyTerp source Algorithm-1/MSE practical Hadamard variant, QJL disabled**, regular2/outlier3. It is not prod or an exact reproduction of the original paper's 3.5-bit table. Its complete measured mean rate is 3.00735 bits (Text) / 3.00746 (Info), with ranges 3.00708–3.01056 / 3.00650–3.00904. It is outside the 1.8/2.25-bit ceilings. Nominal widths cannot replace physical owner accounting.

The mean complete visual owner is 18,107,024 bytes in each cohort. Maximum actual CUDA allocated bytes are 16,289,841,664 / 16,400,157,696. The single-layer decoded FP32 visual K/V temporary reaches 7,340,032 / 9,437,184 bytes; this excludes other bit-expansion/text/softmax temporaries and is not summed across sequential layers. Original records retain all codebook CPU/GPU constants, norms, maps, positions and physical ownership reports.

The actual reader is still the explicit `torch_materialize_fp32_reference`, not an optimized source-paper kernel. Audited quality-run wall averages are 4.989s Text / 5.383s Info; output lengths average 3.8125 / 4.25 tokens, and no case reaches the 64-token cap. These concurrently collected instrumented timings are retained as prototype costs, not a controlled speed claim. The separate speed workflow remains separate.

My judgment is that this is a credible higher-precision source-family M1 reference worth fixed-500 confirmation under Root's overall scheduling decision. It is not automatically the strongest measured Pareto point: descriptive same-source K2R0 values are 79.375% / 75.104% at approximately 2.315 bits, and this pilot is too small to replace the predeclared broader baseline matrix. The next useful comparison is the source family across actual budgets, without treating the weakest low-bit point as its representative.

`configs/matrix/OnlyTerp_Hmix23_P2_500_optional_v6.json` prepares each dataset's fixed 500 with exactly 32 verified same-contract hits and 468 missing answers. Neither optional500 job is enqueued or launched. There is no automatic three-model/four-dataset expansion and no new rotation, width or own-method algorithm in this task.

All 5 original raw/source-proof runs, exact composed views and offline scored rows are retained on E; canonical worker results remain on the 8TB project volume. Browsable summary: `results/matrix/OnlyTerp_Hmix23_real64_v1.json` / `.csv`. Paired evidence: `audit/OnlyTerp_Hmix23_Text32_vs_F_v5.json` and `audit/OnlyTerp_Hmix23_Info32_vs_F_v5.json`.

The scheduler waited for the original full ChartQA worker to finish naturally before any Hmix23 GPU work. Full DocVQA was never signaled or restarted. The source mixed-prod CUDA/probe and A/B late-question controls scheduled afterward are independent diagnostics, not rows in this MSE table, not reusable MSE predictions and not formal memory/speed benchmarks. Their interpretation belongs to Root's separate source-boundary review.

The additional requested controls also completed their exact scheduler contracts: mixed-prod CUDA `PASS_ONLYTERP_MIXED_PROD_SOURCE_PACKING_OWNER_GATE` with `GPU_used=true`; mixed-prod probe two cases with three arms each `DIAGNOSTIC_RAW_COMPLETE`; boundary control two cases with six arms each `BOUNDARY_DIAGNOSTIC_RAW_COMPLETE`. None was promoted to the MSE main table or scored here. No threshold was relaxed and no source/algorithm was altered to obtain these statuses.

The live scheduler is `bm_External_baselines_P2_Fbank_OT23_64_IP_boundary_queue_v10`, PID1079201. It naturally launched the preserved full-Text4500 GPU1 alias, PID1085864, after all controls. Full DocVQA PID1041987 continues on GPU0. All 43 active/pending nodes at Hmix23 takeover are conserved: one original Chart run naturally completed, the original Doc remains in the wait state, and the remaining 41 node objects remain exact in the current plan. Original aliases, all core/budget nodes and four E-window additions were neither dropped nor duplicated. `audit/OnlyTerp_Hmix23_final_node_and_aux_conservation_v1.json` independently verifies controller argv, node identity/order and all three auxiliary completion contracts. Queue ownership returns to Root after this completed handoff; neither active worker needs to be stopped.
