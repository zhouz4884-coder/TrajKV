# Cap8 calibration alignment repair

The old native collection used `arguments.max_patches=16`. Quality variants applied `cap=8`, overriding the quality argument's 16. Those frozen runs remain valid **cap16 calibration → cap8 quality cross-density development**; they do not establish matched-cap calibration performance. Their source, books, results and geometry are preserved.

The new recipe changes only collection patch cap to 8 and uses new output names. Source train32/select32, original book16, frozen factors, 23K20V mask, token sample maximum128, seed8, calibration code, eight endpoints and maximum eight fitting iterations remain fixed. `configs/a2/round8_v1/native_collect_cal32_select32_cap8_v2.json` has exact SHA256 `b2d86edfd544eaf2b8f34911e26b521133f09198a19a66a490bb54a94cd4006c`.

Root launches only after the current six-task GPU lease finishes and is transferred. No launch/SSH/GPU operation was performed during this preparation. All code entries use the isolated stage; model/data/config/book/results use their existing global data-disk paths, P72 uses its 8TB runtime, and compile/tmp paths use the established R8 shared data-disk cache. Existing 360-case reader PASS is retained without rerunning it.

```powershell
python code/launch_round8_data_disk.py configs/jobs/r8_native_collect_cal32_select32_cap8_gpu0_v2.json
```

Require the new dump `results/round8_v1/native_cal32_select32_cap8_v2` COMPLETE with64 sources and verify its `config_sha256` equals the new cap8 config. Then fit independently:

```powershell
python code/launch_round8_data_disk.py configs/jobs/r8_native_fit_cap8_k_gpu0_v2.json
python code/launch_round8_data_disk.py configs/jobs/r8_native_fit_cap8_v_gpu1_v2.json
```

These produce `results/round8_v1/native_books_cap8_k_v2` and `native_books_cap8_v_v2`. Source membership and native frame are frozen before sampling; train only fits endpoints, select only scores their frozen centers. No old output is overwritten.

The new generator reuses the existing endpoint/owner logic in memory, without editing the old generator. Its default output prefix is `postfit_cap8_aligned11_v3`:

```powershell
python code/a2/prepare_round8_cap8_postfit_evaluation.py --plan-only
python code/a2/prepare_round8_cap8_postfit_evaluation.py
```

Before actual generation, download only the new native manifest and all eight endpoint JSONs plus both completion receipts to matching local paths. Native tensors remain remote. Generation verifies the exact cap8 collection config SHA, fixed source stimulus hashes and64-source union, original book/factor/mask, and each book's new dump-manifest SHA/provenance. It refuses the old cap16 manifest. Both QA `arguments.max_patches` and every `variant.cap`, including Full and frozen R7 controls, are explicitly8.

Successful generation creates new configs `configs/a2/round8_v1/postfit_cap8_aligned11_v3_{doc,text}.json`, corresponding `job_postfit_cap8_aligned11_v3_{doc,text}.json`, and a new preparation receipt. Root can then run:

```powershell
python code/launch_round8_data_disk.py configs/a2/round8_v1/job_postfit_cap8_aligned11_v3_doc.json
python code/launch_round8_data_disk.py configs/a2/round8_v1/job_postfit_cap8_aligned11_v3_text.json
```

The same eleven arms and fixed endpoint-sharing rules are retained. The config records per-source native visual-token counts, which must be compared with actual quality `visual_tokens` after execution. Equal cap declarations alone are not a substitute for that observed-Nv check. These selection sources remain historically exposed development sources.

The separate geometry job points to the new cap8 native dump:

```powershell
python code/launch_round8_data_disk.py configs/jobs/r8_joint_geometry_native_cap8_cpu_v2.json
```

Its output is `results/round8_v1/joint_gain_geometry_native_cap8_v2`. It uses four CPU threads and hides CUDA. The old cap16 geometry retains its original density/sampled-source scope; neither result is a causal explanation of cap8 task quality. Diagnostics/profile work can follow after the priority matched-cap collection, fitting and quality runs.
