# Round11 primary D96 per-domain results

384 new free-answer QA and 288 reused reference views. Heterogeneous task scores are not pooled.

| Dataset | n | FP16 | R7low | R7high | JointV | Spatial | JointSpatial | Shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| textvqa | 32 | 79.0625 | 75.6250 | 74.6875 | 77.8125 | 75.6250 | 77.8125 | 75.6250 |
| infovqa | 32 | 93.5096 | 90.4193 | 96.0096 | 91.6346 | 90.4193 | 90.4193 | 91.6346 |
| docvqa | 16 | 100.0000 | 97.9167 | 97.9167 | 97.9167 | 100.0000 | 97.9167 | 97.9167 |
| chartqa | 16 | 75.0000 | 81.2500 | 81.2500 | 81.2500 | 81.2500 | 81.2500 | 81.2500 |

Scores are percentages under each domain's own frozen primary metric.

| Dataset | Mode | Delta pp | Improved/degraded/equal | String changed | Token sequence changed | First argmax changed |
|---|---|---:|---:|---:|---:|---:|
| textvqa | JointV | 2.1875 | 1/0/31 | 3 | 3 | 2 |
| textvqa | Spatial | 0.0000 | 0/0/32 | 0 | 0 | 0 |
| textvqa | JointSpatial | 2.1875 | 1/0/31 | 4 | 4 | 3 |
| textvqa | Shuffled | 0.0000 | 0/0/32 | 0 | 0 | 0 |
| infovqa | JointV | 1.2153 | 1/0/31 | 1 | 1 | 0 |
| infovqa | Spatial | 0.0000 | 0/0/32 | 0 | 0 | 0 |
| infovqa | JointSpatial | 0.0000 | 0/0/32 | 0 | 0 | 0 |
| infovqa | Shuffled | 1.2153 | 1/0/31 | 1 | 1 | 0 |
| docvqa | JointV | 0.0000 | 0/0/16 | 1 | 1 | 1 |
| docvqa | Spatial | 2.0833 | 1/0/15 | 1 | 1 | 1 |
| docvqa | JointSpatial | 0.0000 | 0/0/16 | 1 | 1 | 1 |
| docvqa | Shuffled | 0.0000 | 0/0/16 | 0 | 0 | 0 |
| chartqa | JointV | 0.0000 | 0/0/16 | 0 | 0 | 0 |
| chartqa | Spatial | 0.0000 | 0/0/16 | 0 | 0 | 0 |
| chartqa | JointSpatial | 0.0000 | 0/0/16 | 0 | 0 | 0 |
| chartqa | Shuffled | 0.0000 | 0/0/16 | 0 | 0 | 0 |

All candidate requests have exactly the same complete visual-owner bytes as their paired R7low request.
Full-vocabulary logits were not compared. Prototype timing is descriptive and does not establish speed gains.
Bytes, prototype times, covered-token mode fractions and unadjusted paired intervals are in candidate_vs_R7low.csv.
