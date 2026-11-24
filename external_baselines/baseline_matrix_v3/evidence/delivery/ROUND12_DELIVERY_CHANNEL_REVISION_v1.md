# Round12 delivery-channel revision

The full VALUE_OFFSET_DIAGNOSTIC_v1.json exceeds the existing 16 MiB browser-file limit. Its original bytes, SHA and public member path remain unchanged and it is now explicitly archive-only. The entire diagnostic, including all owner traces, remains in the release ZIP; no evidence is deleted and the limit is not raised.

VALUE_OFFSET_DIAGNOSTIC_COMPACT_v1.json is a browser-visible summary of the already recorded scalar counts, first-logit measurements and descriptive per-branch statistics. It binds the full original SHA d99baa4311131db5988709eb20269c6248bd79ede6560f668c76dac4a36ecc35. No GPU execution, new answer, rescoring or scientific-result change occurs.

Scope v2 and the failed first export remain preserved. This new scope is unsealed; Root must seal and export it into a new stage, then independently verify and publish it. Preparation is not upload.
