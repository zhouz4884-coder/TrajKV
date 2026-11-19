# Round8 native calibration jobs, v1

Prepared locally on 2026-11-18. These files have not launched a process or used SSH/GPU. Root owns upload, resource checks and execution.

The isolated source root is `/path/to/TrajKV/stages/round8_20261118_v1`. Every entry script is taken from its immutable job snapshot. Configurations, datasets, old books, results and new caches use absolute paths on the original project data disk. The model retains the frozen configuration's existing absolute path under the data volume's `models/` directory. Jobs invoke the P72 interpreter at the actual migrated 8TB runtime path; the unchanged launcher may still use the original compatible prefix alias for its controller.

## Inputs and fixed fields

`native_collect_cal32_select32_v1.json` collects 32 train and 32 select sources, using one grouped input row per source. It uses no reference-answer file. Train is 16 Doc plus 16 Text sources; select is a disjoint 16 plus 16 sources. Full-image K FP16 mean, V binary partition/all-image min/scale and frozen metadata fold precede sampling at most 128 token positions per source. Native source tensors are diagnostic owners and are not retained by a deployed codec.

The baseline is `r7_joint_k1_gt_cond` from the frozen `configs/a2/round7_v1/policy_select_doc.json`. Its `selected_stages` field contains the 224-entry priority, and `res_stages_limit=43` selects its first 43 entries: 23 K and 20 V, all first stages. The collector receives precisely these 43 stages. The deployed bank keeps the original full priority table and its 448-byte rank charge. The selected-mask canonical-JSON SHA256 is `31a5f9ec3d6464139fe040138d045cb1b61872b171fd1f8566f575e7926fbf4a`.

The frozen book/factor pair is `gaussian_vq_g4_16.json` / `factors_for_book16_inherited.json`; its encoder-book hash is checked before collection. The old functional K factor is already in stored decoder-only gain. V uses its frozen all-image frame and has no row gain.

## Upload and launch order

First finish disk maintenance and verify the P72 interpreter and both GPUs. Upload/hash-check the isolated source tree, including `code/a2` and `code/experiments`, and the new global configuration/dataset files. `round8_check.py` needs the frozen `code/experiments/fused_progressive.py` source in its snapshot. The launcher uploads a job JSON and snapshots Python source; it does not upload the referenced global configuration or data assets.

All outputs and job snapshot names below must be unused. An interrupted/failed frozen run receives a new version name; do not remove or overwrite its evidence. Prerequisite fields in the job manifests are instructions for root; this launcher does not schedule dependencies.

Run the bounded collector on GPU0 and the independent packed reader control on GPU1:

```powershell
python code/launch_round8_data_disk.py configs/jobs/r8_native_collect_cal32_select32_gpu0_v1.json
python code/launch_round8_data_disk.py configs/jobs/r8_reader_check_gpu1_v1.json
```

Wait for both recorded workers to finish. Require the reader's `complete.json` to report `PASS`, `gpu_executed=true`, and its checks to pass. Require the native collector's `complete.json` to report `COMPLETE` and 64 sources; verify its manifest contains exactly the frozen train32/select32 memberships, source-shard hashes and expected book/factor identities. The fitter additionally checks each sampled initializer's actual FP16 base against the recorded old deployed base before any fit.

After both GPUs are released, launch K and V fitting independently:

```powershell
python code/launch_round8_data_disk.py configs/jobs/r8_native_fit_k_gpu0_v1.json
python code/launch_round8_data_disk.py configs/jobs/r8_native_fit_v_gpu1_v1.json
```

Both use eight maximum update iterations, chunk 2048, train32 only, equal total source weights, native-coordinate FP16 objective, and separate `decoder-only` / `reencode` endpoints. An endpoint stops if no finite candidate improves its training aggregate. Select tensors are loaded only after its final centers have frozen and are scored without affecting training, stopping or acceptance. Parent may then select methods using these endpoint scores and real-model development results.

## Outputs

All paths below are relative to the original global project root on the data disk.

| Job | Output | Meaning |
| --- | --- | --- |
| GPU0 collector | `results/round8_v1/native_cal32_select32_v1/` | 64 bounded native-source shards, manifest and completion receipt |
| GPU1 reader control | `results/round8_v1/reader_control_gpu1_v1/` | Synthetic actual CUDA packed-reader/codec arithmetic checks; no model quality claim |
| GPU0 K fit | `results/round8_v1/native_books_k_v1/` | K global16x4, global256x8, position32x16x4; each decoder-only and reencode |
| GPU1 V fit | `results/round8_v1/native_books_v_v1/` | V global256x8; decoder-only and reencode |

Each fitted endpoint exports FP16-valued centers, their exact little-endian tensor SHA, JSON-file SHA in `complete_books.json`, actual table bytes, source/split/input identities, field policy, training hard-MSE history and select-only score. The global K4 table costs 128 B; K8, K4 position control and V8 each cost 4096 B. K decoder-only also retains the old 128-B encoder book and must bill it separately.

These fitting jobs do not establish model quality, encoding throughput or reader speed. The later fixed-mask model four-arm and matched controls use these exported endpoints without fitting the combined arm again. Preparation identities are recorded in `outputs/round8_v1/job_preparation_v1.json`; the implementation contracts are in `native_fit_plan_v1.json`.
