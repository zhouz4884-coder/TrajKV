# Speed-stage preparation, 2026-11-20

The requested speed cohorts are prepared from real, SHA-verified P1 model outputs. No ground truth was read, no new model answer was generated, and no GPU timing job was launched by this preparation. The benchmark driver, evaluator, cache engine and method sources were not changed.

The executable job/config matrix is `configs/matrix/SPEED_stage_v2.json`. Its CPU source-verification table is `results/matrix/SPEED_stage_v1.csv`; the final expected-file copy/stage map is `audit/speed_stage_preparation_final_v2.json`. The prior v1 matrix is retained; v2 corrects only the optional local staging-copy basename for the preserved M1 Text16 expected file.

| Cohort | Methods | Formal QA | Verified model-output ID records | Source/CPU plans | Hardware timing |
| --- | --- | ---: | ---: | --- | --- |
| M1 TextVQA | F / K2R0 / C1 / E | 16 | 64 | PASS, existing template and expected preserved | Not launched by preparation |
| M2 TextVQA | F / K2R0 / E | 8 | 24 | PASS | Not launched by preparation |
| M3 TextVQA | F / K2R0 / E | 8 | 24 | PASS | Not launched by preparation |
| M1 InfoVQA | F / K2R0 / C1 / E | 8 | 32 | PASS | Not launched by preparation |

K2R0 here is the frozen g128/R0 KIVI-style technical anchor, not the newly implemented source-example g32/R32 point. No output from another class or configuration was substituted. M2 retains its original native processor/pixel settings and repetition penalty; M3 retains its native pixels, three-axis positions and DeepStack. Each method within a cohort uses the same fixed first QA IDs from the frozen P2 input order, which are covered by the corresponding real P1 records. Selection used no score or reference answer.

Each cohort first has a separate 2-QA hardware smoke, then its 8/16-QA formal job. Both use three measured repeats and one full warmup per QA/method, with the existing fixed-seed interleaved schedule. A formal job requires its smoke to finish `PASS_TIMED_AND_TOKEN_VALIDATED`. These speed completion statuses differ from the raw-inference queue's markers; Root schedules the listed launch jobs rather than silently appending them to an incompatible queue.

Across the four formal jobs the plan contains 432 measured trials and 144 warmups. The four small smokes contain another 84 measured trials and 28 warmups. These are planned request counts, not completed GPU work or reported latency. No latency, speedup or throughput result is created by CPU planning.

All job templates provisionally select physical GPU 1 and require exclusive compute access. `host_quiet_asserted=false` remains the default. This preparation did not establish GPU exclusivity or a quiet host. Root can choose a controlled window and create explicitly named new versions for a different GPU or quiet assertion; the existing M1 Text16 template and all frozen v1 inputs remain unchanged.

The new expected originals are on E:

- `${EVIDENCE_ROOT}/results/speed_preparation/stage_v1/expected_M2_text8_v1.json`
- `${EVIDENCE_ROOT}/results/speed_preparation/stage_v1/expected_M3_text8_v1.json`
- `${EVIDENCE_ROOT}/results/speed_preparation/stage_v1/expected_M1_info8_v1.json`

The existing M1 Text16 original remains `${EVIDENCE_ROOT}/cache_validation/speed_expected_M1_text16_v1.json`. D contains only small job/config/summary JSON and the browsable status table. Root's copy/stage map gives the exact expected remote basename and SHA. The original E files remain authoritative even if Root makes small staging copies under `configs/speed`.

Eight other model/dataset combinations are explicitly pending speed-expected export. Their locally observed P2 view closures are recorded in the matrix, without claiming that missing local evidence proves a remote worker is incomplete. Completed P2 views were not converted into unrequested speed cohorts here, and incomplete/unverified cohorts received no invented expected IDs. New g32, buffered TurboQuant, OnlyTerp-MSE, C2 or E-window variants require their own matching quality evidence and separately chosen speed cohort; C1/T0 IDs cannot stand in for them.
