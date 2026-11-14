# Predeclared replication read-budget sweep

Entry: `code/experiments/budget_sweep.py`.

This worker sequentially invokes the unchanged `run_cycle.py` four times,
reloading the model in each process. Budgets are fixed at 4, 8, 16 and 32 MiB;
every budget uses method `asym` and policies `uniform`, `periodic`,
`table_future`, `probe_l7_t0.02`. No CLI option changes this candidate bank.
Every child selects the same first 16 rows of the supplied replication32
manifest by default, retaining source-image IDs and hashes. `--count` permits
a separately recorded smaller or larger predeclared prefix.

Worker command (replace only the worker-specific path placeholders):

```text
python code/experiments/budget_sweep.py --model MODEL_DIR --samples configs/data/replication32_runner.json --legacy-scripts LEGACY_SCRIPTS_DIR --kivi-source KIVI_CODEC_FILE --gain-table FROZEN_GAIN_TABLE.json --output results/budget_sweep_rep16 --count 16 --turns 10 --max-tokens 24 --max-patches 4
```

`--dry-run` records the full plan and argv without spawning a child or importing
torch. To execute after a dry-run, repeat the identical command with `--resume`
instead of `--dry-run`. Resume rejects changed source, sample or gain-table
hashes; it verifies and skips completed children and appends `--resume` only to
an existing unfinished child. No model generation was launched while preparing
this entry point.

Output:

```text
OUTPUT/sweep_manifest.json        complete argv, fixed cohort, source hashes
OUTPUT/sweep_status.json          each attempt's argv, exit code and completion
OUTPUT/budget_4MiB/               original run_cycle output structure
OUTPUT/budget_8MiB/
OUTPUT/budget_16MiB/
OUTPUT/budget_32MiB/
OUTPUT/budget_NMiB.child.log      combined child stdout and stderr
OUTPUT/complete.json              only after all four validated children
```

For the default 16 images, each child must have exactly 64 unique
session/method/policy records, for 256 records overall. Before the parent
completion marker, the worker verifies all four child completion files, original
selected IDs, source-image pairing, turn counts and declared logical read caps.
An exit code of zero or an isolated child `complete.json` is insufficient.
The wrapper reads no task references or quality scores to select a policy.

Each child is separately compatible with `analyze_cycle.py --records
OUTPUT/budget_NMiB/records.jsonl ... --baseline asym:uniform`. Use a separate
analysis output directory per budget. A full-precision comparison requires a
separately measured reference on the same source images; this wrapper does not
manufacture one. The resource axis is the frozen runner's **extra logical
tensor bytes**, not measured physical HBM traffic or total cache storage.

CPU orchestration controls: `audit/budget_sweep_cpu_validation.json` records
eight checks covering dry-run, all-budget completion, fixed candidate argv,
resume, failed children, incomplete children and source/table changes. Temporary
fixture records were discarded. They are not GPU, quality or performance data.
