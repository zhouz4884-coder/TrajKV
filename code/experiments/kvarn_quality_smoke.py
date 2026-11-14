"""KVarN mechanism-port quality smoke through the shared run_cycle executor.

This entry does not reproduce official vLLM serving. It materializes the
decoded KVarN visual KV and compares generation against full/base2/native4 in
the exact same run_session path. Packed references and materialized tensors
are both resident and must both be counted. No speedup claim is permitted.

Run --preflight-only on CPU to inspect source readiness without touching GPU.
Reference answers remain in the separate evaluator. If --reference is supplied,
token-forced records are explicitly excluded from application-quality scoring.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
import run_cycle

ROOT = Path(__file__).resolve().parents[2]
PORT_DIR = ROOT / "third_party/reference_ports"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight() -> dict:
    bank_path = Path(run_cycle.__file__).with_name("bank.py")
    bank_source = bank_path.read_text(encoding="utf-8")
    ast.parse(bank_source)
    ast.parse(Path(run_cycle.__file__).read_text(encoding="utf-8"))
    if "r.storage_tensors().values()" not in bank_source:
        raise RuntimeError("bank.owned must count storage_tensors().values(), not dict keys")
    if "method.startswith('kvarn')" not in bank_source:
        raise RuntimeError("shared bank lacks the KVarN branch")
    if not (PORT_DIR / "kvarn_reference.py").exists():
        raise RuntimeError("KVarN reference port is missing")
    sys.path.insert(0, str(PORT_DIR))
    from kvarn_reference import EXPECTED_SHA256, UPSTREAM_COMMIT, source_fingerprints
    actual = source_fingerprints()
    if actual != EXPECTED_SHA256:
        raise RuntimeError("KVarN pinned source hashes changed")
    files = [Path(__file__), Path(run_cycle.__file__), bank_path,
             ROOT / "code/bytes.py", PORT_DIR / "kvarn_reference.py"]
    return {"status": "PASS_STATIC_CPU_PREFLIGHT", "gpu_used": False,
            "upstream_commit": UPSTREAM_COMMIT, "upstream_source_sha256": actual,
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
            "result_label": "KVarN low-level reference [mechanism-port]",
            "memory_scope": "packed_reference_plus_materialized_decoded_visual_kv",
            "required_gpu_followup": ["codec_finiteness", "same_executor_quality",
                                      "resident_tensor_accounting"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "samples", "legacy-scripts", "kivi-source", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--methods", nargs="+", choices=("full", "base2", "native4", "kvarn2", "kvarn4"),
                   default=["full", "base2", "native4", "kvarn2", "kvarn4"])
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--max-patches", type=int, default=4)
    p.add_argument("--read-budget-bytes", type=int, help="Common allowance metadata; fixed codec performs no optional residual reads")
    p.add_argument("--reference", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    checked = preflight()
    if args.preflight_only:
        print(json.dumps(checked, indent=2))
        return
    for name in ("model", "samples", "legacy_scripts", "kivi_source", "output"):
        if getattr(args, name) is None:
            p.error("--" + name.replace("_", "-") + " is required for GPU evaluation")
    if args.count < 1 or args.start < 0 or args.turns < 1 or args.max_tokens < 1:
        p.error("invalid count/start/turns/max-tokens")
    args.margin = 0.0
    args.max_replays = 0
    args.save_states = False
    args.output.mkdir(parents=True, exist_ok=args.resume)
    run_cycle.install_legacy(args.legacy_scripts)
    rows = json.loads(args.samples.read_text(encoding="utf-8"))
    rows = rows[args.start:args.start + args.count]
    if not rows:
        raise ValueError("empty sample slice")
    forced = {}
    if args.reference:
        for line in args.reference.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["method"] == "full" and r.get("protocol") == "free_running":
                forced[r["session_id"]] = r["turns"]
        missing = [r["id"] for r in rows if r["id"] not in forced]
        if missing:
            raise ValueError(f"missing full free-running reference trajectories: {missing}")
    log_path = args.output / "records.jsonl"
    seen = set()
    initial_hashes = {}
    if args.resume and log_path.exists():
        old_records = list(map(json.loads, log_path.read_text(encoding="utf-8").splitlines()))
        seen = {(r["session_id"], r["method"]) for r in old_records}
        for r in old_records:
            old_hash = initial_hashes.setdefault(r["session_id"], r["initial_logits_sha256"])
            if old_hash != r["initial_logits_sha256"]:
                raise RuntimeError("resumed records already contain mismatched native initial logits")
    manifest = {**checked, "execution_status": "prepared",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "selected_ids": [r["id"] for r in rows],
                "samples_sha256": sha256(args.samples),
                "reference_sha256": sha256(args.reference) if args.reference else None,
                "reference_answers_visible_to_runner": False,
                "quality_scoring_allowed": args.reference is None,
                "fixed_token_warning": "teacher-forced answer text is not an application-quality result"}
    manifest_path = args.output / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity_fields = ("samples_sha256", "source_sha256", "reference_sha256",
                           "quality_scoring_allowed", "selected_ids")
        argument_fields = ("model", "methods", "turns", "max_tokens", "max_patches")
        if (any(previous.get(k) != manifest[k] for k in identity_fields) or
                any(previous.get("arguments", {}).get(k) != manifest["arguments"][k]
                    for k in argument_fields)):
            raise RuntimeError("resume source/sample/protocol identity changed; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    snapshot_dir = args.output / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(run_cycle.__file__), Path(run_cycle.__file__).with_name("bank.py"),
                 PORT_DIR / "kvarn_reference.py", args.legacy_scripts / "p202_fused.py", args.kivi_source):
        shutil.copy2(path, snapshot_dir / path.name)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    manifest.update(preflight_result=checked,execution_status="running",gpu_used=True,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,inference_dtype="float16")
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    count = 0
    with torch.inference_mode(), log_path.open("a", encoding="utf-8") as log:
        for row in rows:
            for method in args.methods:
                if (row["id"], method) in seen:
                    continue
                result, _ = run_cycle.run_session(model, processor, row, args, method, "base",
                                                  forced=forced.get(row["id"]))
                expected_hash = initial_hashes.setdefault(row["id"], result["initial_logits_sha256"])
                if result["initial_logits_sha256"] != expected_hash:
                    raise RuntimeError(f"native initial-logit mismatch for {row['id']} / {method}")
                result["native_initial_logit_parity"] = True
                result["quality_scoring_allowed"] = args.reference is None
                result["baseline_label"] = ("KVarN low-level reference [mechanism-port]"
                                             if method.startswith("kvarn") else method)
                result["cost_comparison_allowed"] = not method.startswith("kvarn")
                if method.startswith("kvarn"):
                    result["upstream_commit"] = checked["upstream_commit"]
                    if result["owned_final"].get("runtime") != "materialized_kvarn_reference":
                        raise RuntimeError("KVarN memory scope no longer matches the declared path")
                log.write(json.dumps(result) + "\n")
                log.flush()
                count += 1
    manifest.update(execution_status="complete",records_added=count,native_initial_logit_parity=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (args.output / "complete.json").write_text(json.dumps({
        "status": "COMPLETE_SHARED_EXECUTOR_QUALITY_SMOKE", "records_added": count,
        "quality_scoring_allowed": args.reference is None, "paid_calls": 0,
        "official_serving_reproduction": False}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
