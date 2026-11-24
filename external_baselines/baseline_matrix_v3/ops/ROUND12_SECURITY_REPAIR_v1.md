# Round12 export scanner self-match repair

The immutable generic scanner has source SHA256
`ae711282f004b6704d76e2f09901ef840480d505706b50bf44a3fd8a02a030ef`.
Its regex definitions describe sensitive patterns; they contain no actual
credential, private machine identity or functioning connection route. Applying
its path replacement rules to its own Python source changed regex semantics.
The failed export is retained, and no research data or GPU output is changed.

`round12_public_security_v1.py` recognizes only public relative path
`ops/export_public.py` with that exact complete source SHA. It checks the exact
line SHA256 values below before masking these regex-definition lines for the
privacy scan. Every other line is scanned by the original frozen rules. The
source is exported as original bytes, without changing its executable behavior.

| Original line | Exact original line SHA256, including line ending |
|---:|---|
|128|880147fee7f883c5509856a0d33531a625a7c8a6e78b6dd1c0890c064f2a7356|
|186|50e832a9cedc0b88a190689e313fdb17bd8dfb8f67e1c04f0adc4a7af796b158|
|187|09ec401f43aff4b39d19de9441aee315b742624fdffe320be356baa8c16d6f86|
|189|ed930b42e2ffee898398d58580e5c6f465b4eb12d7b1bc1152cfdb2cd5c55e50|

No path-only, label-only or regex-wide exception exists. Appending a sensitive
value changes the full source SHA and disables this handling. The check script
verifies rejection of synthetic credential, endpoint and signed URL payloads both
in modified scanner source and ordinary support files. A renamed exact source is
also scanned without the exception.

The v2 exporter can reuse bodies from the preserved failed export only when
Root-pinned index, audit and original sealed-scope fingerprints all match. It
accepts only that one documented self-match failure. Each reused file must have
the same source SHA in the new Root scope and old sealed scope; the actual cached
public file and its JSONL line-identity sidecar are rehashed against the pinned
index and rescanned. The copied bytes are rehashed again. Changed/new files are
transformed again. The scanner itself is never reused from the damaged cached
body. New scope, phase, README, content index, archive and cold receipts are
generated in a new stage. The failed stage remains unchanged.

The standalone cold verifier repeats privacy scans, exact identities, source and
producer bindings, official metrics, physical budgets and archive checks. The
privacy scanner retains whole-body matching for non-JSONL text, including
patterns whose whitespace spans lines. Only legal JSONL records are scanned one
record at a time; JSON validity and row identities are independently verified,
and literal line breaks cannot occur inside valid JSON strings. An additional
synthetic multiline check verifies rejection by the original whole-body rules.
The publisher checks all portable verifier source hashes against the package before
running the cold verifier. No relaxation applies to predictions, references,
scores or other source files.

Local checks use synthetic payloads and one real cached raw/identity pair. They
do not represent new model inference or a GitHub upload.
