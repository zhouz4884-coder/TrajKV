# Round8 incremental delivery contract

The four new entry points are `prepare_round8_delivery.py`, `round8_content_index.py`, `verify_round8_increment.py` and `publish_round8_update.py`, under `code/publication`. They retain the Round7 incremental publication protocol and tag/release history checks. They do not modify the Round7 scripts. The new tag is exactly `research-round8-20261118`; no success claim is hard coded.

The current template is `configs/publication/round8_delivery_plan_v1.json`. Its final execution receipt, final report, README entry and method freeze are deliberately pending. Preparation cannot bypass them. The receipt must say `RESEARCH_COMPLETE_READY_FOR_PUBLICATION`, `research_execution_completed=true` and `github_sync_pending=true`; the freeze contains `frozen_files` entries with concrete `path` and original-byte `sha256`.

Plan-only inventories current completions and pending final inputs. It performs no Git/network operations and does not touch the public checkout:

```powershell
python code/publication/prepare_round8_delivery.py --plan configs/publication/round8_delivery_plan_v1.json --plan-only
```

Final root execution, after freezing the completed stage:

```powershell
python code/publication/prepare_round8_delivery.py --plan configs/publication/round8_delivery_plan_v1.json --preflight-only
python code/publication/prepare_round8_delivery.py --plan configs/publication/round8_delivery_plan_v1.json --output artifacts/round8_delivery_20261118_v1 --runtime-sources artifacts/round8_original_stage_jobs_v1
python code/publication/verify_round8_increment.py --assets artifacts/round8_delivery_20261118_v1 --output artifacts/round8_delivery_20261118_v1/VERIFICATION.json
python code/publication/publish_round8_update.py --assets artifacts/round8_delivery_20261118_v1
```

The final command above is a local verified publication plan. Root alone adds `--execute` to publish after reviewing the concrete package. Preparation/verification are local only; `--execute` is the sole mutation route. Existing checkout changes/divergence, old tags/releases, wrong account/repository, missing actual tree coverage, ignored data manifests, or mismatched archive/Git identities stop publication.

Quality analyses must have COMPLETE summaries with PASS audits, complete official raw/scored answer coverage, original source/reference hashes, `round8_quality_owner.audit_owner` recounts, and each raw producer's `complete_round8.json`. Its `engine_complete_sha256`, `manifest_sha256` and `encoder_inputs_unchanged` are checked. `complete.json` alone is insufficient. Automatically discovered completed development views retain their original analysis role; they do not become blind confirmation or independent samples. Other raw runs remain tracking evidence, including failed/partial/unexecuted attempts.

Both fitted-book manifests are required. The eight endpoints (global K4, global K8, K4 position control and global V8, each decoder-only/reencode) must match original JSON SHA, little-endian FP16 center SHA, actual shapes/bytes, disjoint train/select source membership, and pinned split/dump/old-book inputs. Source and answer manifests are included; `configs/data/` paths are explicitly force staged and checked in the actual Git tree.

`--runtime-sources` accepts a local exact copy of the isolated stage's `jobs/` tree, including `*.source/sha256.json` and Python bodies, or the existing collector layout with `runtime_index.json`/`COLLECTION.json` and source objects. Original inventories are preserved as source-indexed content; safe Python versions are deduplicated by original source SHA. Every publishable `manifest.sources` hash used by quality runs must have its original content object, rather than a substituted current working-tree file. An inventory establishes available source versions, not proof that every module executed. Failed and unexecuted snapshots keep that scope.

The actual `collect_round8_runtime_sources.py` format is supported directly: `--runtime-sources results/round8_v1/runtime_sources`. A later immutable capture can instead pass its explicit `runtime_index_<capture>.json`; the matching `COLLECTION_<capture>.json` is selected without replacing the canonical index. Membership is pinned to that capture's source mappings, so later unrelated objects added to the shared `objects/` directory are not silently absorbed. Original per-job inventories, ended receipts/logs/completions and source aliases receive an explicit public metadata-to-content mapping. The raw collection is excluded from ordinary browsable-results scanning.

To test real source sanitization and content identities before final inputs exist:

```powershell
python code/publication/prepare_round8_delivery.py --plan configs/publication/round8_delivery_plan_v1.json --runtime-sources results/round8_v1/runtime_sources/runtime_index_current_v1.json --runtime-plan-only --output artifacts/round8_runtime_derivation_test_v2
```

Choose an unused output name. This mode creates only a `RUNTIME_DERIVATION_TEST` archive/index/check with `publishable=false`, no `DELIVERY.json`, and explicit pending final inputs. It cannot pass the final publisher. `runtime_delivery_adapter_check_v1.json` records the real first capture's successful 12-job/396-object and 4,627 inventory plus 2,075 manifest-mapping closure check.

Public bodies are derived copies. Source SHA, exact archive member SHA/`content_ref`, LF-normalized Git body SHA and predicted Git blob OID remain separate. The publisher verifies actual staged/committed/remote trees and body bytes, retains prior tags/releases/assets and binds the archive plus content index to the new commit. Frozen Round7 source paths are never installed over old checkout files; historical consumed versions are kept as archive source objects instead.

All raw `outputs/maintenance_20261118` content is prohibited. A separate optional `maintenance_public_summary` may contain only schema/status, booleans for accepted migration/runtime/cache placement, GPU count, free-byte totals, a private receipt SHA and a short scope. It cannot be a raw maintenance file or expose machine paths, device names, hosts, usernames, connections or credentials. Tensor/checkpoint/native `.pt` data, original images/model weights and private chats/routing helpers are excluded throughout.

The pipeline is prepared while final confirmation/quiet measurements are still running. No package, upload or completed-stage claim is created by plan-only.
