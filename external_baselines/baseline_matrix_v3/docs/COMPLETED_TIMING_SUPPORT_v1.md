# 已完成的分域 timing 支持

本次新测728 trials = legacy656 + 独立进程Calib32 + 强代表硬件smoke40；另附旧M1 Text2 prerequisite32作为既有支持，不能算作本次新测。强代表其余520仍未运行。所有trial已通过独立token/context/owner/重复次数检查，不增加正式质量QA。

legacy正式结果仅下列四个cohort，合计576 trials（432 measured+144 warmup）；其余3个legacy smoke共80 trials另列。host_quiet=false，GPU compute独占不等于整机安静。GPU stream interval不是纯kernel时间；输出自然长度不同，不按更短答案wall排序，不给跨模型/域平均加速比。

| cohort | model / dataset | method | QA | measured / warm | TTFT host ms (median) | natural ITL ms (mean) | encode stream ms (median) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| M1_text16 | M1 / textvqa | F_native_split | 16 | 48 / 16 | 625.268 | 40.425 | 0.210 |
| M1_text16 | M1 / textvqa | KIVI_style_K2V2_g128_R0 | 16 | 48 / 16 | 685.557 | 49.782 | 51.918 |
| M1_text16 | M1 / textvqa | CalibQuant_visual_FP32bias_1bit | 16 | 48 / 16 | 934.851 | 89.777 | 58.041 |
| M1_text16 | M1 / textvqa | PyramidKV_visual_drop0p888_unknown | 16 | 48 / 16 | 689.285 | 55.610 | 48.036 |
| M1_info8 | M1 / infovqa | F_native_split | 8 | 24 / 8 | 677.135 | 41.055 | 0.221 |
| M1_info8 | M1 / infovqa | KIVI_style_K2V2_g128_R0 | 8 | 24 / 8 | 747.507 | 49.448 | 52.313 |
| M1_info8 | M1 / infovqa | CalibQuant_visual_FP32bias_1bit | 8 | 24 / 8 | 1076.878 | 88.577 | 57.971 |
| M1_info8 | M1 / infovqa | PyramidKV_visual_drop0p888_unknown | 8 | 24 / 8 | 736.157 | 57.197 | 48.251 |
| M2_text8 | M2 / textvqa | F_native_split | 8 | 24 / 8 | 594.212 | 42.673 | 0.183 |
| M2_text8 | M2 / textvqa | KIVI_style_K2V2_g128_R0 | 8 | 24 / 8 | 650.737 | 51.665 | 50.515 |
| M2_text8 | M2 / textvqa | PyramidKV_visual_drop0p888_unknown | 8 | 24 / 8 | 665.625 | 57.711 | 46.240 |
| M3_text8 | M3 / textvqa | F_native_split | 8 | 24 / 8 | 342.249 | 62.771 | 0.130 |
| M3_text8 | M3 / textvqa | KIVI_style_K2V2_g128_R0 | 8 | 24 / 8 | 419.469 | 72.750 | 63.259 |
| M3_text8 | M3 / textvqa | PyramidKV_visual_drop0p888_unknown | 8 | 24 / 8 | 421.768 | 80.259 | 58.517 |

独立Calib32是两个两题进程（F+C1、F+C2），只观察本进程方法首次记录调用与warm重复，磁盘binary cache保留，没有all-disk-cold或autotune winner身份观测。强代表retry40也是两题五profile的技术smoke，不能替代Text16/Info8正式测量。旧stage缺配置失败为0测量，重试成功没有将40算成80。

全域标量、完整bytes、CUDA峰值、自然输出长度和physical GPU编号见 `results/timing_completed_public_v1/domain_summary.csv`；每trial原件/expected模型输出/上下文与原始执行源证明均由固定support清单列出。源实现与移植prototype reader的耗时不冒称原论文优化kernel的速度。
