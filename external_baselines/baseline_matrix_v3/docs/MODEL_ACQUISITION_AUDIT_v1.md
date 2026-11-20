# Model acquisition audit

Checkpoint revisions stay fixed throughout every acquisition attempt:

- M1: OpenGVLab/InternVL3-8B-hf at 259a3b64a14623c0ec91a045cb43f7c5af5fa6af.
- M2: Qwen/Qwen2.5-VL-7B-Instruct at cc594898137f460bfe9f0759e9844b3ce807cfb5.
- M3: Qwen/Qwen3-VL-4B-Instruct at ebb281ec70b05090aa6165b016eac8ec08e71b17.

## M1 exact identity

The existing shared FP16 checkpoint's four complete weight SHA256 digests match
the official LFS objects. Sixteen files required at inference also match their
official content identities. The first identity verifier incorrectly compared
the LFS-hosted tokenizer JSON with the SHA1 of its Git pointer; its preserved
v1 report therefore says TREE_MISMATCH. The corrected, LFS-aware v2 report is
PASS_EXACT_REQUIRED_TREE. Independent comparison of the official raw tokenizer
and local raw tokenizer establishes exact byte identity and equal canonical
JSON, with no changed fields.

The new project owns a pinned 18-file official tree under `models/M1_<revision>`.
Sixteen verified existing objects use hard links; the official README and
`.gitattributes` use newly fetched locked bytes. The shared original remains
unchanged. Its identity receipt records each source inode, size, mtime and exact
content SHA so subsequent read-only loads can reuse the proven identity.

## M2/M3 network recovery

The inventory had Qwen2.5-VL GPTQ/AWQ variants but lacked the required original
M2 and M3 weights. Those quantized variants were not used.

| Attempt | Observation / treatment |
|---|---|
| r1 direct HF from remote | Repeated connection reset on huggingface.co; mirror probe returned 403. Preserve logs and partial state. |
| r2 local-memory SFTP relay | No weights landed on a local C/D filesystem; only transient in-memory chunks went to the data volume. Throughput was poor, so weight relays were stopped. Metadata relay completed and was verified. |
| r3/r4 public CDN redirect | Local requests resolved public redirects; their signed URLs crossed SSH stdin in memory and never entered argv, files, reports or logs. Remote CDN access returned valid HTTP206. Whole-file transfers were too slow. |
| r5 segmented, 16 workers | 32MiB ranges, exact Content-Range/byte-count checks, data-volume segment files; completed segments retained. |
| r6 segmented | M3 used 64 workers and M2 used 32, reusing r5's completed segments. Live argv and source/lock identities are recorded separately. |
| r7 M2 round-robin | M2 moved to 64 workers across its five shards to avoid one slow CDN object starving later shards. Reuses the same validated r5 segment pool. M3 r6 continues independently. |
| r8 exact range-prefix resume | Some HTTP bodies ended early despite correct range headers. Both workers now preserve their valid temporary prefix and request only its missing suffix, with renewed public redirects; all completed r5 segment files remain reused. Reports include actual argv, Python and source SHA. Full assembled LFS SHA is still mandatory. |

Original attempt reports may retain RUNNING after an explicitly stopped worker;
they are frozen observations, not proof of current liveness. Current progress
uses the report PID and the actual `/proc` argv together. A timeout alone never
triggers a restart.

These are transport changes only. Final files are published in the locked
revision directory only after the assembled file size and official LFS SHA256
match. The safetensors index must name exactly the downloaded shards. Native
P0 refuses to load M2/M3 until the corresponding integrity report says PASS for
the exact locked revision. Partial ranges and transfer progress are not model
completion or inference evidence.

All weights, original images, environments, segment pools, partial files,
compilation caches and temporary directories remain on the remote data volume.
