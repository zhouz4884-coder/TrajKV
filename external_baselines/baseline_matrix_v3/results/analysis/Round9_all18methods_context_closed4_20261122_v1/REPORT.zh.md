# 同fixed500的18方法上下文（只读派生）

状态：PASS_ALL4_FORMAL500_WITH_ARCHIVED47_CONTEXT；Drop4正式完成4/4。
旧17方法47个完整格复用已发布R7证据；旧13待补、8未计划继续null。新增fresh只限Drop4四域各500，共2000；没有启动旧pending任务。
主预选20格的配对CI仍由原正式主分析单独提供；补充27个旧完整格只展示上下文，没有额外重扫raw、重评分或重算配对CI。
Text=十票soft accuracy；Info/Doc=vendor raw-length ANLS；Chart=relaxed accuracy。域间不平均，FP16仅同500经验参考，不含官方全量FP16。

| 方法 | Text % | Info % | Doc % | Chart % |
|---|---:|---:|---:|---:|
| FP16：同一fixed500 | 80.140 | 76.012 | 93.298 | 90.000 |
| KIVI-style：2bit / g128 / R0 | 79.640 | 未闭/null | 90.987 | 88.800 |
| KIVI-style：2bit / g128 / R128 | 未闭/null | 未闭/null | 未计划 | 未计划 |
| KIVI-style：4bit / g128 / R0 | 79.340 | 未闭/null | 未计划 | 未计划 |
| KIVI-style：4bit / g128 / R128 | 未闭/null | 未闭/null | 未计划 | 未计划 |
| KIVI source example：2bit / g32 / R32 | 79.040 | 73.632 | 91.038 | 89.800 |
| CalibQuant numeric-v2：1bit | 56.580 | 43.095 | 58.141 | 64.400 |
| CalibQuant numeric-v2：2bit | 77.160 | 62.301 | 83.730 | 84.800 |
| PyramidKV：W64 / drop0.888 | 65.380 | 未闭/null | 32.292 | 55.400 |
| PyramidKV：W64 / drop0.91875 | 未闭/null | 未闭/null | 未计划 | 未计划 |
| PyramidKV：W8 / drop0.888 | 45.580 | 28.764 | 17.672 | 45.000 |
| PyramidKV：W8 / drop0.91875 | 57.260 | 27.953 | 25.861 | 53.800 |
| OnlyTerp-MSE：Hmix1/2，QJL=0 | 59.360 | 22.978 | 60.238 | 66.800 |
| OnlyTerp-MSE：Hmix2/3，QJL=0 | 74.880 | 71.377 | 87.661 | 84.400 |
| TurboQuant 0xSero source：K3/V2 g32 R128 | 未闭/null | 未闭/null | 未闭/null | 未闭/null |
| R7 K1 gt_cond43 | 76.520 | 68.302 | 90.310 | 88.800 |
| R7 K2 G4_256 cond | 78.700 | 73.317 | 91.290 | 89.600 |
| Round9 Drop4：固定V0 / profile7ec | 76.100 | 68.107 | 89.277 | 88.800 |

完整mean/max bits、mean bytes/MiB、每题严格1.3/1.8资格与状态见long/wide CSV；不能用均值冒充全500预算或只筛预算合格题计算质量。不同实际字节的比较不是等字节因果、CI不作等价或赢家结论。
旧47原raw/source/score可在已验证R7 release归档复核：https://github.com/zhouz4884-coder/TrajKV/releases/download/r7-m1-fixed500-anchor8-20261121-v1/R7_M1_two_points_fixed500_all8_20261121_v1.zip
固定commit=63281a90f159fec99dace96e88267d9a53c49014；tree=972421cabe1997112f9173fc84cd94276338568a；ZIP SHA256=3cdb988753da915f56d109b7f2dec03757da57735a486c54eb279c3166394354。
旧source/public/Git-LF SHA分别见summary.archived_context_provenance；此新表不冒称未来新archive包含补充27的大raw或对它们完成新冷重算。
新候选只通过冻结74f主分析的实际500 source/score、同ID/token/position/boundary/context和逐题完整owner门后给分；未闭不预估、不做partial quality。
