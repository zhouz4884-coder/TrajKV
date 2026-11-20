# External baseline increment: M1_FP16_FOUR_OFFICIAL_FULL_REFERENCE_BANKS

Declared scope: M1 native FP16 reference bank for four complete official splits: TextVQA validation5000, DocVQA validation5349, InfographicVQA validation2801, ChartQA test2500 (human1250+aug1250). This phase does not complete P2/P3 compressed baselines or the whole external library.

Closed/scored cells: 4/4; questions: 15650/15650.

The primary phase totals above exclude the independently declared auxiliary groups below. Auxiliary view counts do not imply additional unique GPU inference; provenance retains reused predictions. Reports are bound by explicit completed-evidence inventories; source/configuration files may also describe unrun targets. Completion is assessed only against these declared groups, not the whole external SOTA library or all P2/P3 research. A progress preview cannot be published by the reviewed publisher. Derived views retain original predictions and instrumented timing; they are not new GPU inference runs. All original donor runs and source/retrieval proofs are archived, including donors from earlier phases.

PUBLIC_CONTENT_REFS.json separates original source, sanitized archive and LF-normalized Git identities. Original SHA fields remain source identities. Per-line maps are separate archived JSONL sidecars. Code/configuration/tables/small manifests are browsable in Git; full raw evidence is in the ZIP. Images, weights, tensor dumps, private routes and compiler caches are excluded.

From the extracted namespace run `python -B code/verify_public_export.py --root . --recompute-scores`. This validates donor closure, exact prediction provenance, fixed-reference metrics, cluster CIs and declared coverage. Use exact archive bytes for source guards that distinguish original CRLF from Git LF. Runtime and method limitations remain explicit in reports.

- Auxiliary `ONLYTERP_HMIX12_REAL64`: 2 views / 64 view questions. OT_Hmix12 real fixed pilot: Text32 and Info32, 64 distinct QA for this source point; metrics reported separately. Not full official split, not production/paper-exact table, not controlled speed.
- Auxiliary `ONLYTERP_HMIX23_REAL64`: 2 views / 64 view questions. OT_Hmix23 real fixed pilot: Text32 and Info32, 64 distinct QA for this source point; metrics reported separately. Not full official split, not production/paper-exact table, not controlled speed.
- Auxiliary `ONLYTERP_SHARED_FP16_MATCH64_SUPPORT`: 2 views / 64 view questions. Shared native FP16 matched reference views: Text32 and Info32; zero new GPU QA; support only, not additive to native full-bank inference totals.

The current primary report is [M1 native full reference v2](docs/MILESTONE_M1_FP16_ALL4_FULL_REFERENCE_v2.md). It records nine compatibility jobs / eighteen model-method responses over two shared Text inputs. Confidence intervals use source-cluster resampling.
