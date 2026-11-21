# M1 Calib numeric-v2: both bitwidths, all four fixed500 domains

This milestone contains exactly the predeclared M1 Calib C1/C2 cells selected from frozen core114 matrix fb816d1680734c32ce1474eb8dcc89a860f703ca772783d371311e785f801d5a. All8 cells/4000 fixed-subset questions are actual new-port PASS raw/source/score. The archive closes8 fresh500 original donors and8 derived views. Views and supporting CPU tables do not add GPU inference. Original candidate7/8 snapshots remain historical and unchanged. This is not the whole126 fixed500 /24 full-split goal.

The implementation is the common-FP16 source-constant visual Calib port CalibQuantFP32AffineTextAdapter (numeric v2). It retains the official source quantizer, clipping, packed QK/WV, FP16 metadata/weights/text KV and offsets; visual affine/calibration and exact-text QK use declared FP32 temporaries. It is not bitwise author-BF16 execution or reproduction of the unavailable author calibration-search/data provenance. No parameters or sources were selected using these scores. Legacy Calib6 and other models are outside this phase.

| Tier | Dataset / metric | QA | Quality % [source-cluster95% CI] | Complete mean bytes | Complete mean/max bits |
|---|---|---:|---:|---:|---:|
| C1 numeric-v2 | textvqa / textvqa_mmf_ten_vote_leave_one_out_v1 | 500 | 56.580000 [52.379000, 60.820000] | 6126942.976 | 1.022243175 / 1.043980190 |
| C1 numeric-v2 | infovqa / infovqa_anls_qwenvl_vendor_rawlen_ge0p5_v1 | 500 | 43.095491 [38.862593, 47.095515] | 5838203.648 | 1.025013157 / 1.127476283 |
| C1 numeric-v2 | docvqa / docvqa_anls_qwenvl_vendor_rawlen_ge0p5_v1 | 500 | 58.141229 [54.280109, 62.176066] | 6540741.376 | 1.020343241 / 1.043980190 |
| C1 numeric-v2 | chartqa / chartqa_pix2struct_relaxed_5pct_v1 | 500 | 64.400000 [60.195000, 68.600000] | 6253841.152 | 1.024532359 / 1.127476283 |
| C2 numeric-v2 | textvqa / textvqa_mmf_ten_vote_leave_one_out_v1 | 500 | 77.160000 [73.539500, 80.600000] | 12125808.128 | 2.022282182 / 2.044061570 |
| C2 numeric-v2 | infovqa / infovqa_anls_qwenvl_vendor_rawlen_ge0p5_v1 | 500 | 62.300735 [58.313972, 66.184801] | 11548972.544 | 2.025057565 / 2.127720424 |
| C2 numeric-v2 | docvqa / docvqa_anls_qwenvl_vendor_rawlen_ge0p5_v1 | 500 | 83.730400 [80.933659, 86.684886] | 12952483.328 | 2.020378545 / 2.044061570 |
| C2 numeric-v2 | chartqa / chartqa_pix2struct_relaxed_5pct_v1 | 500 | 84.800000 [81.395000, 87.800000] | 12379321.856 | 2.024575829 / 2.127720424 |

TextVQA uses the locked official MMF ten-vote score; InfoVQA and DocVQA use locked ANLS; ChartQA uses relaxed accuracy. Different dataset metrics are not averaged. Each CI is a question-weighted source-cluster percentile bootstrap with2000 draws and frozen seed20261119, recomputed from locked refs during validation. Fixed P2 subsets are reused pilot cohorts, not independent confirmation or official full splits. Failures would make the complete primary unavailable; none of these4000 were filtered or replaced with fake zero.

ChartQA reports both250-question groups:

| Tier | Group | QA | Quality % [95% CI] |
|---|---|---:|---:|
| C1_bias32 numeric-v2 | human | 250 | 56.800 [50.800, 62.800] |
| C1_bias32 numeric-v2 | aug | 250 | 72.000 [66.400, 78.000] |
| C2_bias32 numeric-v2 | human | 250 | 79.200 [74.400, 84.000] |
| C2_bias32 numeric-v2 | aug | 250 | 90.400 [86.800, 93.600] |

Complete visual bytes are reconstructed for every request using the frozen physical helper621b9a3b: actual unique storage capacities/charges, original positions, per-channel scale/zero, LUT/constants and native GQA/layer/N geometry. Mean bits and ratio-of-sums bits remain distinct. All2000 C1 requests meet complete1.3bit (and1.8bit) budgets; all2000 C2 requests exceed1.8bit and meet2.25bit. No score is recomputed on a budget-selected subset. These within-family points do not establish the complete126 frontier, equivalence, causal iso-byte effects, an algorithm winner or a TrajKV lead.

No numeric F-relative claims are emitted: native-F is not declared as an independent auxiliary source-closed phase in this package. No controlled speed comparison is claimed; wall/TTFT/encode fields in score_matrix are instrumented quality-prototype observations, and reference reader timings are not optimized-kernel throughput. Published code/refs/source snapshots are included with exact source/public/archive and Git-LF identities kept separately. Original images, weights, tensors, compiler caches, private routes/credentials and chats are excluded.

Reproduction uses unchanged public_phase_contract_v2 validation/source_closure, old621 physical_complete_bytes, score_matrix, exporter_v3, cold verifier and publisher_v5. Only this declared8-cell milestone is complete. GPU execution, queue, original ledgers and frozen582 sources were not altered by packaging; actual publication is a separate Root operation.

Packaging boundary: the private ops/export_public.py Sanitizer dependency is not delivered in this public support inventory. Its own security-rule definitions trigger the unchanged scanner, so its exact optional path was omitted under reviewed scope; original SHA and failed-stage evidence are retained. The public bundle supports standalone cold verification and score/source/ref reproduction. It does not claim standalone re-execution of the exporter, which still uses the unchanged private Sanitizer. No numerical reader, original result/source proof, scorer, refs, independent cold verifier or integrity/security guard is omitted or changed.
