# Cap8-aligned repair: completed bounded execution

New collection uses cap8, same train32/select32 source split, frozen book/factors and43-stage mask. All eight endpoints and660 QA answers completed. Both GPU leases are released. This is exposed selection development; the overall Round8 stage/confirmation/quiet delivery is not marked complete.

| Arm | Doc ANLS % | Text official accuracy % | Doc complete bits |
|---|---:|---:|---:|
| full8 | 90.1485 | 87.7273 | 16.002232 |
| r7_frozen_joint43_control | 92.2289 | 90.4545 | 1.294877 |
| r8_k4old_vbinary | 92.2289 | 90.4545 | 1.294877 |
| r8_k8re_vbinary | 94.4385 | 85.9091 | 1.295504 |
| r8_k4old_v8re | 95.3868 | 81.8182 | 1.295524 |
| r8_k8re_v8re | 93.2676 | 81.8182 | 1.296152 |
| r8_k4fit_vbinary | 92.2289 | 90.4545 | 1.294877 |
| r8_k4pos_vbinary | 94.8605 | 90.4545 | 1.295504 |
| r8_k4old_vbinary_jointgain | 95.3868 | 90.4545 | 1.294877 |
| r8_k8dec_vbinary | 91.6083 | 90.4545 | 1.295524 |
| r8_k4old_v8dec | 92.2289 | 86.3636 | 1.295524 |

Doc V8re and jointgain improve the baseline point ANLS by3.1579pp and Exact from32/38 to34/38. Their paired source CI is[0,8.9474]pp, p=.5, Holm55=1.0. K8re is+2.2095pp on Doc with CI[-1.2028,8.1081]pp; Text is-4.5455pp. V8re/both lose8.6364pp on Text. The combined K8+V8 arm does not add the single-axis Doc gains.

The old cap16-calibration to cap8-quality chain had23/32 select-source Nv mismatches; the new chain has0/32. Old results are preserved with cross-density scope. Full/R7/R8-old/jointgain repeated controls are tracked by exact answer/token identity, not counted as new independent sources.

Cap8 select protected-head geometry: SSE/row29.56749→28.54561; radial term grows0.63686 while angular term drops1.65874 and mean cosine increases only7.61e-5. Of94,208 rows,10,388 improve SSE while cosine worsens. These geometric terms do not identify a causal task mechanism.

The only execution issue was a prelaunch SFTP temporary-mode error; its0B temp and trace remain preserved. No GPU worker failed, no old source/result was overwritten, no .pt/image/weights were downloaded, and no confirmation GT or GitHub action occurred.

See `aligned_results.csv`, `paired_vs_baseline.csv`, `density_scope_comparison.csv` and the bounded execution receipt `audit/a2/round8_cap8_aligned_repair_execution_v3.json`.
