# Round12 actual research cutoff

This package records only the Root-sealed actual scope. Fresh tasks, successful terminal answers, failure-status rows, pilots, reused reference views, probes, calibration and QA0 are separate.

The archive contains explicitly selected text/scalar predictions, source proofs, configs, references, metrics, physical-owner records and report. It excludes original images, weights, private tensors/PT, connection routes, credentials and private chat. No GPU or private tensor replay is claimed.

Cold verification: `python -B ops/verify_public_round12_v1.py --root . --archive <ZIP>` (add `--allow-fixture` only for synthetic self-tests). Python and numpy are sufficient; no network or GPU is used.

`PUBLIC_CONTENT_REFS.json` separates original source SHA, sanitized body SHA, Git LF/blob identity and fixed archive member references. The publisher must still verify actual remote commit/tree/blob bodies, downloaded release assets and preservation of earlier history. A local package is not an upload.
