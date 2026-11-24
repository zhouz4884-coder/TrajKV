# Round12 incremental publication

Use only a completed, Root-sealed production export. The default command runs
the complete local cold verifier and checks both current reviewed verifier
sources; it does not access authentication, network or Git writes.

```powershell
python -X utf8 -B ops/publish_public_round12_v1.py --stage-dir ${EVIDENCE_ROOT}/public_staging/ROUND12_20261124_v1 --expected-base 28eb91cdb0d8f9de16946b33529da660c02c8988
```

After reviewing the final export, add `--execute` and a new receipt path:

```powershell
python -X utf8 -B ops/publish_public_round12_v1.py --stage-dir ${EVIDENCE_ROOT}/public_staging/ROUND12_20261124_v1 --expected-base 28eb91cdb0d8f9de16946b33529da660c02c8988 --execute --receipt ${EVIDENCE_ROOT}/results/round12/GITHUB_ROUND12_DELIVERY_v1.json
```

The expected base is an explicit reviewed commit, not a permanently assumed
remote head. Refresh it before actual execution. Existing approved Git helper
credentials stay in process memory. This script does not implement a new
credential route or store credentials.

The publisher copies only `delivery_channels.browser_paths` into the existing
public checkout. It forces the exact selected paths into the Git index, including
ignored `configs/data` references, and verifies actual LF-normalized Git blob
bodies before committing. Archive-only evidence remains in the release ZIP and
does not silently disappear from the delivery.

The stage-local `PUBLISH_ROUND12_JOURNAL_V1.json` records the exact source epoch,
payload, base commit, old tree, releases, tag objects and prepared commit. A
bounded transport failure can be retried with the same command and journal.
Changed payloads, publisher code or expected bases require inspection rather than
rewriting the old journal. Existing release assets and tags are never overwritten.

All prior paths remain present. Earlier results/configs and every other namespace
must keep their blob OIDs. Shared code/index documents may receive the explicit
new version, while the previous bodies remain available in retained history and
earlier releases.

After the ordinary push, independent verification checks the server commit,
parent and tree, then downloads the fixed-commit source archive and verifies every
expected new browser blob body against Git OID and LF SHA256. It downloads the
entire release archive, index and delivery document, checks their SHA256 values,
and checks all ZIP member identities. Old releases/assets and raw annotated tags
plus peeled targets are rechecked. A final delivery receipt appears only after
all checks pass.

To repeat only remote verification for a committed journal, use a new receipt:

```powershell
python -X utf8 -B ops/publish_public_round12_v1.py --stage-dir ${EVIDENCE_ROOT}/public_staging/ROUND12_20261124_v1 --expected-base 28eb91cdb0d8f9de16946b33529da660c02c8988 --verify-remote --receipt ${EVIDENCE_ROOT}/results/round12/GITHUB_ROUND12_REMOTE_RECHECK_v1.json
```

`--verify-remote` does not push or create releases/assets. It still uses the
approved read account, writes local verification downloads/receipts and repeats
the local full cold verification. A completed upload must not be inferred from
the local `READY_FOR_ROOT_ROUND12_EXECUTION` status.

CPU/local-Git checks (no network or authentication):

```powershell
python -X utf8 -B ops/check_round12_publisher_v1.py
```

These synthetic checks validate publication safety behavior, not research
performance or actual remote delivery.
