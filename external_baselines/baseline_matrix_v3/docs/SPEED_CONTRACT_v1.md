# Controlled port end-to-end speed

The driver is implemented and CPU-plan checked. **No new GPU timing has been produced by this implementation task.** `audit/speed_driver_cpu_acceptance_v1.json` records 13 CPU schema/source checks, not a performance result. A real CUDA smoke and controlled run are still required.

## Workload and timing boundary

`code/benchmark_speed.py` loads one original FP16 model and native processor once. All method configs must share checkpoint revision/path, processor/pixel settings, question suffix and generation cap. Each trial independently decodes the original image bytes, runs the native image-only prefix, creates the real compressed visual owner, and recomputes the entire question suffix before native greedy generation. No native first-answer logit or previous method prefix/cache is reused.

Original encoded images are read from the data volume and SHA-verified into CPU RAM before timing. Image decoding to RGB and native preprocessing/device transfer occur anew inside every timed request. Checkpoint verification, loading and original image disk reads are reported outside the request boundary. This models an already-loaded streaming service receiving image bytes in memory; it does not measure cold model startup, storage/network ingestion or concurrent serving throughput.

Each measured QA/method receives one complete untimed warmup by default, then three measured repeats. The fixed seed determines a method shuffle **within each same-QA repeat**. All methods process the same ordered, predetermined IDs. The default selection is the first 16 entries in the frozen input order. Explicit IDs are supported. Optional native-N quantile selection uses only IDs, image identities and visual-token counts from a native FP16 reference with the same model/cap; it does not read a reference answer or task score.

Question-prefix independence is checked once per QA before warmup. Every actual trial still performs its own native preprocessing/prefix. Qwen's original three-axis positions, native delta, DeepStack, full generation history and repetition penalty remain active. Consumed pixel tensors and GPU audit maps are released as in the quality path. A single first-suffix hook updates the tiny native delta reference and records CUDA events; there is no per-forward finite check or synchronization hook.

The timed region excludes external owner reports, SHA computations, JSON/disk writes, explicit garbage collection and quality-output comparisons. It uses CUDA events at phase boundaries without per-phase synchronization, followed by one final synchronization before request completion. Intrinsic checks already built into frozen protocol/cache/codec functions remain part of their implementation cost; this driver does not rewrite those functions to improve a timing result. Streamer token arrival uses the actual native generated CPU IDs. Text decoding is included in request wall time.

## Reported quantities

- `wall_ms` is complete streaming request time, ending after text decoding and final CUDA completion.
- `ttft_ms` is time from request start to the first generated token's CPU arrival, including preprocessing, image-prefix prefill, compression and first question-suffix inference.
- `inter_token_ms` records consecutive generated-token arrivals. `decode_tokens_per_second` uses the post-first-token interval; `generation_tokens_per_second` includes the first suffix/generation interval, and full-request token rate includes the whole request. EOS counts are preserved; no token-length cleanup is applied.
- `preprocess_ms` is host preprocessing time. `prefill_ms`, `encode_ms`, `generation_ms` and `question_prefill_ms` are CUDA **stream intervals**, not separately synchronized wall measurements. Host dispatch intervals are reported alongside them. Stream intervals may include CPU submission gaps and are not claimed as pure kernel-active time.
- Output token IDs/count, original raw prediction, true peak CUDA allocated/reserved bytes, pre-trial allocator state and complete physical visual-owner bytes are retained. Owner storage accounting happens after timing. The native visual denominator and all persistent metadata/positions/constants follow the existing quality owner audit.

Outputs may have different lengths across methods because compression changes answers. The driver never computes a scalar wall-only speedup or ranks methods solely by total wall time. Tables must show output lengths, TTFT, per-token intervals/rates and quality together. Three repeats per QA reduce timing noise; they do not create three independent images. Any uncertainty analysis must preserve QA/source pairing.

The default PyTorch caching allocator is retained across interleaved trials. Explicit GC occurs outside timing; `empty_cache` is not called by default. A separately declared empty-between-trials mode exists and must not be mixed into one speed table. Reserved memory can reflect a previous method's retained allocator pool; allocated peaks and actual owner bytes are the relevant separate measurements.

## Validation and exclusivity

Use original model output IDs from the matching quality run, not task ground truth. `ops/prepare_speed_expected.py` exports a small `baseline_matrix.expected_model_predictions.v1` file from SHA-verified original raw runs. It contains method-specific output IDs and source/model/runtime/processor/generation fingerprints. The timing driver checks that source context before measurement, then verifies every warmup/measured output against its corresponding quality token sequence and input fingerprints. Complete visual-owner bytes must also agree.

A token mismatch, failed warmup, missing trial or execution failure makes the method's primary speed aggregate ineligible; successful conditional diagnostics remain explicitly labeled. With no expected-output file, timing may be collected for diagnosis but is not accepted as a quality-validated canonical comparison. This is fixed-cohort runtime validation, not a proof of equivalence on every possible input.

Actual runs require one visible SM86 CUDA GPU and original FP16 weights. The default driver checks that no other compute PID shares the declared physical GPU before and between QAs. Root schedules exclusive GPU access and separately records whether the host is quiet. It must not launch this benchmark on a GPU occupied by another worker. A single-GPU streaming port result is not original-paper system throughput or a universal algorithmic speed bound.

## Configuration and execution

`configs/speed/M1_text16_interleaved_v1.json` is a **planned** remote job template for F/K/C/E on fixed TextVQA 16. It expects `configs/speed/expected_M1_text16_v1.json` to be staged first. No worker is launched by these preparation tools. Relevant fields:

```json
{
  "schema": "baseline_matrix.speed_job.v1",
  "eval_configs": ["full_target_F.json", "full_target_K.json", "full_target_C.json", "full_target_E.json"],
  "inputs": "textvqa_p2_inputs.jsonl",
  "selection": {"mode": "first", "count": 16},
  "repeat": 3,
  "warmup_per_qa": 1,
  "seed": 20261120,
  "expected_predictions": "expected_model_outputs.json",
  "allocator_policy": "retain_caching_allocator",
  "verify_gpu_exclusive": true,
  "host_quiet_asserted": false
}
```

Use full-target eval configs as method templates; the speed job explicitly selects its common cohort rather than using method-specific missing-only inference lists. Relative eval/input paths resolve against the project source root. The frozen input source lock is required. All result directories are created exclusively.

```powershell
# CPU only, from local metadata and already mirrored original quality raw runs:
python ops/prepare_speed_expected.py --config LOCAL_SPEED_JOB.json --quality-runs RAW_F RAW_K RAW_C RAW_E --output EXPECTED.json --plan-output ${EVIDENCE_ROOT}/NEW_EXPECTED_PLAN
python code/benchmark_speed.py --config LOCAL_SPEED_JOB.json --output ${EVIDENCE_ROOT}/NEW_CPU_PLAN --plan
```

The local acceptance used `${EVIDENCE_ROOT}/cache_validation/speed_job_M1_text16_CPU_v1.json` and exported 64 prior model outputs to `speed_expected_M1_text16_v1.json`. It planned 64 full warmup requests and 192 measured requests without importing Torch/Transformers or producing a latency value. Root can stage the expected-output file at the remote template's declared path, then schedule a small real smoke before the controlled run. All actual worker caches, models, images, temporary files and outputs remain on the project data volume via the standard remote launcher.

## Future explicitly verified ports

An optional driver-local `adapter_extensions` registry admits a future new module/class only with exact source-file SHAs and a SHA-locked support receipt. It does not mutate `evaluate.ALLOWLIST` or any cache policy. The default F/K/C/E path reuses the core loader/factory. A newly registered port still needs a correct quality-context expected-output manifest and an actual CUDA smoke; registry acceptance alone is not numerical or performance validation. Microkernel timings, if added later, belong in a separate result and cannot replace this full request boundary.
