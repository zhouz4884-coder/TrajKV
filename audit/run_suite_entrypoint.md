# Fixed external evaluation suite worker

Entry: `code/experiments/run_suite.py`. It imports only the standard library.

```text
python code/experiments/run_suite.py --config configs/external_suite.json --output results/external_suite
```

The configuration is an object containing a nonempty `runs` list. Every item
has `name`, `entrypoint` and `arguments` (a list of strings). Allowed entrypoints
are exactly the same-directory `run_cycle.py` and `kvarn_quality_smoke.py`.
Names are single path segments containing letters, digits, underscore or hyphen;
duplicates, including case variants, are rejected. The suite owns `--output`
and `--resume`, including argparse abbreviations, so they must not occur in
the argument arrays. No shell is used.

Children run sequentially with `sys.executable` and project-root cwd. Each
receives `--output OUTPUT/NAME`. The suite records the exact config bytes,
config copy, every argv, all local experiment Python/helper hashes, available
reference-port helper hashes and explicit sample/table/reference/codec/model
identity hashes. The manifest freezes those identities before child launch.

`--dry-run` records the plan without launching children or writing a completion
marker. Execute that identical plan later using `--resume`. Changed config,
source, input or argv identities reject resume. A child is skipped only if a
previous zero exit and its unchanged `complete.json` were recorded. A missing
completion file causes a resumed child launch. Each launch records argv, exit,
log and completion hash in `suite_status.json`.

The parent `complete.json` appears only after every child exits successfully
and supplies a valid completion object. Starting a resumed execution removes
any stale parent completion marker until all children have been revalidated.
The suite neither scores outputs nor chooses parameters from their results.

Outputs are `suite_manifest.json`, `suite_status.json`, `NAME.child.log`,
unchanged child artifact directories `NAME/`, and final `complete.json`.
CPU-only controls are recorded in `audit/run_suite_cpu_validation.json`;
temporary synthetic control files are discarded and are not research results.
