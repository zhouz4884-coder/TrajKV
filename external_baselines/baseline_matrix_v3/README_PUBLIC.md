# External baseline increment: M1_FP16_FOUR_OFFICIAL_FULL_REFERENCE_BANKS

Declared scope: M1 native FP16 reference bank for four complete official splits: TextVQA validation5000, DocVQA validation5349, InfographicVQA validation2801, ChartQA test2500 (human1250+aug1250). This phase does not complete P2/P3 compressed baselines or the whole external library.

Closed/scored cells: 4/4; questions: 15650/15650.

The primary phase totals above exclude the independently declared auxiliary groups below. Auxiliary view counts do not imply additional unique GPU inference; provenance retains reused predictions. Reports are bound by explicit completed-evidence inventories; source/configuration files may also describe unrun targets. Completion is assessed only against these declared groups, not the whole external SOTA library or all P2/P3 research. A progress preview cannot be published by the reviewed publisher. Derived views retain original predictions and instrumented timing; they are not new GPU inference runs. All original donor runs and source/retrieval proofs are archived, including donors from earlier phases.

PUBLIC_CONTENT_REFS.json separates original source, sanitized archive and LF-normalized Git identities. Original SHA fields remain source identities. Per-line maps are separate archived JSONL sidecars. Code/configuration/tables/small manifests are browsable in Git; full raw evidence is in the ZIP. Images, weights, tensor dumps, private routes and compiler caches are excluded.

From the extracted namespace run `python -B code/verify_public_export.py --root . --recompute-scores`. This validates donor closure, exact prediction provenance, fixed-reference metrics, cluster CIs and declared coverage. Use exact archive bytes for source guards that distinguish original CRLF from Git LF. Runtime and method limitations remain explicit in reports.

- Auxiliary `TIMING560_REQUIRED_ORIGINAL_PREDICTION_SUPPORT`: 3 views / 1500 view questions. Existing exact fixed500 quality views required only to carry complete original timing-prediction source closures. Not new GPU inference, extra timing QA, P2-114 completion or full official splits.
