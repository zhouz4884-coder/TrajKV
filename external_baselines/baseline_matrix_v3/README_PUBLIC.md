# External baseline library: P1 increment

This increment records the declared external-baseline protocol, pinned source,
technical gates and the available P1 evidence. It does not claim completion
of later phases or a full-split score when the manifest selects a subset.

The full exact-byte payload is in the accompanying ZIP. Git contains browsable
source/configuration/tables and small manifests; bulky per-QA evidence remains
in the ZIP. Original images, model weights, compiler caches, private controller
code and connection information are excluded. Obtain the pinned public datasets
and model revisions separately.

`PUBLIC_CONTENT_REFS.json` distinguishes original source bytes, sanitized public
archive bytes, and LF-normalized Git bytes/blob OIDs. Paths are templates such as
`${DATA_ROOT}`; prediction text, question text, answers, token IDs and numeric
results retain their meaning. Original SHA fields inside derived evidence still
refer to the source; resolve them through the content index rather than treating
them as a hash of the sanitized file. Per-line source/public mappings are included.

Run `python code/verify_public_export.py --root .` from an extracted namespace
to validate every public file. Add `--recompute-scores` to independently recompute
the selected offline metrics from all archived scored rows. The scorer requires
NumPy. The original strict private-source verifier is not a sanitizer-aware API.

For strict upstream byte guards use the archive: `.gitattributes` normalizes Git
text to LF, which can differ from the original checkout's CRLF bytes. Runtime
dependencies and port boundaries are stated in the model/method reports. All
matrix inference uses original FP16 weights; technical CUDA controls are separate
from benchmark quality. Prototype instrumented timing is not a controlled speed
claim. Automatic GitHub publication is a separate step after export verification.
