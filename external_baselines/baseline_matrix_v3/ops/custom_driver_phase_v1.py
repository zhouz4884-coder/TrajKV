"""Explicit R7 producer closure. Canonical SourceRun admission is unchanged.

R9 is deliberately rejected until its real producer/fit evidence is frozen.
This module has no network/GPU/Git entry point and never edits original data.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys

from public_phase_contract_v2 import (ROOT, EVIDENCE, load, sha, require, resolve, inside,
    contract_from_arguments, validate_phase as canonical_validate_phase,
    source_closure as canonical_source_closure)
import analyze_r7_fixed500_v2 as r7

R7_SCHEMA = "trajkv.r7_research.frozen_matrix500.v2"
FAMILY = "r7_m1_anchor8_custom_driver_v2"
SUPPORT = EVIDENCE / "results/publication/R7_custom_driver_support_v2"
POLICY_REL = "results/publication/R7_custom_driver_support_v2/R7_SOURCE_POLICY.json"


def role_path(cfg, role):
    if role == "kivi_source": return cfg["kivi_source"]
    if role.startswith("a2/"): return cfg["legacy_a2"] + "/" + role[3:]
    if role.startswith("experiments/"): return cfg["legacy_experiments"] + "/" + role[len("experiments/"):]
    if role.startswith("scripts/"): return cfg["legacy_scripts"] + "/" + role[len("scripts/"):]
    if role == "code/bytes.py": return cfg["legacy_a2"].rsplit("/", 1)[0] + "/bytes.py"
    raise ValueError("Unknown frozen role")


def required_baselines():
    import r7_make_competitive_table_v1 as competition
    index, _ = competition.snapshot_inputs(competition.SNAPSHOT)
    selected = {r["config"]: r["config_sha256"] for r in index["cells"]
                if not r["method_key"].startswith("R7_") and r["status"] == "COMPLETE_FIXED500"}
    require(len(selected) == 39, "Frozen39 baseline snapshot changed")
    return selected


def require_auxiliary_scope(groups):
    actual = {str(a["config"].relative_to(ROOT)).replace("\\", "/"): sha(a["config"])
              for group in groups[1:] for a in group["artifacts"]}
    require(all(actual.get(path) == digest for path, digest in required_baselines().items()), "All39 frozen completed baseline configs must be auxiliary artifacts")


def write_new(path, value):
    body = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        def unique_object(pairs):
            require(len({key for key, _ in pairs}) == len(pairs), "Duplicate keys in frozen custom metadata")
            return dict(pairs)
        existing = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        normalized = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        require(normalized(existing) == normalized(value), "Frozen custom policy differs")
        # Preserve the original immutable bytes; key ordering is not a new policy.
    else:
        with path.open("xb") as stream: stream.write(body)


def assignment(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing frozen source constant: " + name)


def is_custom(manifest):
    schema = manifest.get("runner_schema")
    require(schema in (None, R7_SCHEMA), "Unknown custom/R9 producer is not implemented; never fall back to SourceRun")
    return schema == R7_SCHEMA


def exact_ready8():
    require(sha(ROOT / "ops/analyze_r7_fixed500_v2.py") == "3d282b201d474a61473c16c397fb90a3bde287b8fec6f35c6ecb21c42e5fcda9", "Frozen R7 validator changed")
    rows = r7.formal_rows(r7.MANIFEST); readiness = r7.readiness(rows, r7.HARVEST)
    require(readiness["ready"], "PENDING_REAL8_CUSTOM_PRODUCER: " + str(readiness["closed_metadata_cells"]) + "/8")
    return rows, readiness


def validate_phase(contract):
    if contract.get("family") == FAMILY:
        rows, ready = exact_ready8()
        matrix = load(resolve(contract["matrix"]))
        require(len(matrix["rows"]) == 8 and sum(int(x["count"]) for x in matrix["rows"]) == 4000, "R7 publication is exactly8/4000")
        expected = {x["config"]: x for x in rows}
        require(set(x["config"] for x in matrix["rows"]) == set(expected), "R7 declared8 config set differs")
        cells = {x["name"]: item["cell"] for x, item in zip(rows, ready["cells"])}
        for row in matrix["rows"]:
            frozen = expected[row["config"]]; cell = cells[frozen["name"]]
            require(row["count"] == 500 and row["config_sha256"] == frozen["config_sha256"] and
                    resolve(row["view"]) == Path(cell["raw"]).resolve() and resolve(row["scored"]) == Path(cell["scored"]).resolve(), "R7 declared path/count/identity differs")
            require(is_custom(load(Path(cell["raw"]) / "manifest.json")), "R7 requires actual custom runner schema")
            r7.check_source(frozen, cell)
    else:
        # No unknown producer can enter the unchanged canonical validator.
        for row in load(resolve(contract["matrix"]))["rows"]:
            path = resolve(row["view"]) if row.get("view") else resolve(Path(contract["view_root"]) / Path(row["config"]).stem)
            if (path / "manifest.json").exists():
                require(not is_custom(load(path / "manifest.json")), "Custom R7 artifacts require explicit R7 phase family")
    phase, artifacts, matrix_path = canonical_validate_phase(contract)
    if contract.get("family") == FAMILY:
        phase.update(custom_driver_provenance_schema="trajkv.public_custom_driver.v1", SourceRun_admission=False)
        for artifact in artifacts: artifact["kind"] = "custom_R7_original_inference_run"
        for observed in phase["observed"]: observed["kind"] = "custom_R7_original_inference_run"
    return phase, artifacts, matrix_path


def source_policy(rows):
    """Copy only exact pinned role sources; no original source is rewritten."""
    bridge = ROOT / "code/r7_legacy_bridge_v1.py"
    require(sha(bridge) == r7.PINS["code/r7_legacy_bridge_v1.py"], "Frozen bridge source changed")
    roles = assignment(bridge, "_DEFAULT_SOURCES")
    roles.update({"a2/round6_depth_bank.py": "1d423bf03cdc76c0650776230d6a1093710e9c9fb8ac2cd84974d791aa65c6f7",
                  "a2/round6_depth_attention.py": "f96277c726c39f2880e8618525b4bdcbac0049e68e7c4d3e2eb699fa24dfd945"})
    files = []; role_rows = {}
    for role, digest in roles.items():
        if role == "kivi_source":
            original = ROOT.parent / "VisContract_P14_2/third_party/KIVI/quant/new_pack.py"
            relative = "kivi_source/new_pack.py"
        elif role.startswith("scripts/"):
            original = ROOT.parent / "VisContract_P14_2" / role; relative = role
        elif role.startswith("code/"):
            original = ROOT.parent / "TrajKV" / role; relative = role
        else:
            original = ROOT.parent / "TrajKV/code" / role; relative = role
        require(sha(original) == digest, "Required original role source changed: " + role)
        target = SUPPORT / "role_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        body = original.read_bytes()
        if target.exists(): require(target.read_bytes() == body, "Frozen role copy differs")
        else:
            with target.open("xb") as stream: stream.write(body)
        role_rows[role] = {"file": target.relative_to(EVIDENCE).as_posix(), "sha256": digest}
        files.append(target)
    profiles = {}
    for row in rows:
        cfg = load(ROOT / row["config"]); profile = cfg["variant"]["name"]
        asset_rows = {}
        for role, digest in cfg["expected_asset_sha256"].items():
            remote = cfg["variant"][role]
            require("/configs/r7_research/" in remote, "Unexpected frozen asset location")
            local = ROOT / ("configs/r7_research/" + remote.split("/configs/r7_research/", 1)[1])
            require(sha(local) == digest, "Original frozen asset changed")
            asset_rows[role] = {"file": local.relative_to(ROOT).as_posix(), "sha256": digest}
            files.append(local)
        value = {"assets": asset_rows, "method_id": cfg["method"]["id"], "variant": cfg["variant"]}
        if profile in profiles: require(profiles[profile] == value, "Profile changed across domains")
        profiles[profile] = value
    decoder = SUPPORT / "decoder/tokenizer_config.json"
    decoder.parent.mkdir(parents=True, exist_ok=True)
    require(sha(r7.TOKENIZER_PROOF) == r7.TOKENIZER_SHA, "Frozen cleanupFalse proof changed")
    body = r7.TOKENIZER_PROOF.read_bytes()
    if decoder.exists(): require(decoder.read_bytes() == body, "Decoder config copy changed")
    else:
        with decoder.open("xb") as stream: stream.write(body)
    files += [decoder, r7.MANIFEST]
    policy = {"schema": "trajkv.r7_public_source_policy.v1", "runner_schema": R7_SCHEMA,
              "SourceRun_admission": False, "canonical_cache_admission": False,
              "fixed8_manifest_sha256": r7.MANIFEST_SHA, "required_source_hashes": {k: r7.PINS[k] for k in r7.DECLARED},
              "fixed8_manifest_file": r7.MANIFEST.relative_to(EVIDENCE).as_posix(),
              "tokenizer_cleanup_proof": {"file": decoder.relative_to(EVIDENCE).as_posix(), "sha256": r7.TOKENIZER_SHA},
              "required_role_sources": role_rows, "profiles": profiles,
              "required_baseline39_config_sha256": required_baselines(),
              "frozen_jobs": [{"config": x["config"], "config_sha256": x["config_sha256"], "job": x["job"],
                               "job_sha256": x["job_sha256"], "dataset": x["dataset"], "profile": x["profile"], "GPU": x["GPU"]} for x in rows]}
    write_new(SUPPORT / "R7_SOURCE_POLICY.json", policy)
    return policy, list(dict.fromkeys(files + [SUPPORT / "R7_SOURCE_POLICY.json"]))


def source_closure(artifacts):
    custom, canonical = [], []
    for artifact in artifacts:
        (custom if is_custom(load(artifact["raw"] / "manifest.json")) else canonical).append(artifact)
    closed = canonical_source_closure(canonical)
    if not custom: return closed
    rows, ready = exact_ready8(); policy, extra_files = source_policy(rows)
    by_config = {str((ROOT / row["config"]).resolve()): (row, item["cell"]) for row, item in zip(rows, ready["cells"])}
    require(len(custom) == 8, "R7 closure cannot publish a subset")
    for artifact in custom:
        require(str(artifact["config"].resolve()) in by_config, "Unknown custom config")
        row, cell = by_config[str(artifact["config"].resolve())]
        m, _ = r7.check_source(row, cell)
        require(m["runner_schema"] == R7_SCHEMA, "Wrong custom producer schema")
        raw = Path(cell["raw"]); proof = load(raw / "source_retrieval_receipt_r7_v2.json")
        require(proof["executed_legacy_source_sha256"] == load(raw / "complete.json")["executed_legacy_source_sha256"], "Actual legacy graph differs")
        actual = proof["executed_legacy_source_sha256"]
        for role, item in policy["required_role_sources"].items():
            name = role_path(m["config"], role)
            if name in actual: require(actual[name] == item["sha256"], "Actual executed role differs from frozen source: " + role)
        required_executed = ("a2/round7_key_bank.py", "a2/round7_key_attention.py") if m["config"]["variant"]["name"] == "r7_joint_k1_gt_cond" else ("a2/round6_depth_bank.py", "a2/round6_depth_attention.py")
        require(all(actual.get(role_path(m["config"], role)) == policy["required_role_sources"][role]["sha256"] for role in required_executed), "Actual bank/reader not frozen source roles")
        files = [raw / name for name in ("manifest.json", "records.jsonl", "complete.json", "retrieval_receipt.json", "source_retrieval_receipt_r7_v2.json")]
        files += [raw / entry["file"] for entry in proof["files"]]
        files += extra_files + [ROOT / row["config"], ROOT / row["job"]]
        original_variant = m["config"]["variant_source"]
        require("/configs/r7_research/frozen_assets_v2/" in original_variant["path"], "Original profile source path changed")
        original_file = ROOT / ("configs/r7_research/frozen_assets_v2/" + original_variant["path"].split("/configs/r7_research/frozen_assets_v2/", 1)[1])
        require(sha(original_file) == original_variant["sha256"] == "aba87db8c87bd1ca8ca71685baa7a942c0450c8ecc1302c9ccbe183c67703ef4", "Original frozen profile source changed")
        files.append(original_file)
        closed.append({"path": raw, "kind": "custom_R7_original_inference_run", "files": list(dict.fromkeys(files))})
    return closed


def prepare_bundle(output):
    """Freeze exact8 plus all currently closed declared canonical references."""
    rows, ready = exact_ready8()
    import r7_make_competitive_table_v1 as competition
    index, _ = competition.snapshot_inputs(competition.SNAPSHOT)
    extraction = competition.module("custom_snapshot_extract", competition.SNAPSHOT / "reproduce_snapshot_v2.py", competition.EXTRACT_SHA)
    _, common = r7.load_references(r7.REFERENCES, r7.TOKENIZER_PROOF)
    baseline = competition.refresh_baselines(index, extraction, common, r7)
    out = resolve(output); require(not out.exists(), "Use new bundle directory"); out.mkdir(parents=True)
    primary = [{"model": "M1", "dataset": row["dataset"], "count": 500, "config": row["config"],
                "config_sha256": row["config_sha256"], "view": cell["cell"]["raw"], "scored": cell["cell"]["scored"],
                "producer_kind": "custom_R7_original_inference_run"} for row, cell in zip(rows, ready["cells"])]
    auxiliary = [{"model": "M1", "dataset": c["dataset"], "count": 500, "config": c["config"], "config_sha256": c["config_sha256"],
                  "view": c["view"], "scored": c["scored"]} for c in baseline.values() if c["status"] == "COMPLETE_FIXED500"]
    require(len(auxiliary) >= 39, "Previously closed39 baseline references cannot disappear")
    for label, family, matrix_rows in [("R7_ANCHOR8_FIXED500", FAMILY, primary), ("M1_COMPLETED_BASELINE_REFERENCES", "canonical_M1_fixed500_aux", auxiliary)]:
        matrix = out / (label + "_matrix.json")
        write_new(matrix, {"schema": "baseline_matrix.fixed_matrix.v1", "rows": matrix_rows, "expected_cells": len(matrix_rows), "expected_QA": 500 * len(matrix_rows)})
        write_new(out / (label + "_contract.json"), {"schema": "baseline_matrix.public_phase_contract.v2", "phase_label": label,
            "family": family, "matrix": str(matrix), "view_root": str(EVIDENCE / "results"), "score_root": str(EVIDENCE / "results/scored"),
            "completion_policy": "all_expected_QA_pass", "completion_scope": "declared_matrix_only", "whole_library_completion_assessed": False,
            "coverage_scope": "R7 primary8 fresh500; canonical references are separately counted reused prediction views; no whole Round9 or external library completion."})
    write_new(out / "PENDING_BASELINE_INVENTORY.json", {"cells": [c for c in baseline.values() if c["status"] != "COMPLETE_FIXED500"], "primary_quality_QA_added": 0})
    return {"status": "PREPARED_CUSTOM_R7_8_WITH_CANONICAL_AUXILIARIES_NOT_EXPORTED", "R7_cells": 8, "R7_QA": 4000,
            "auxiliary_cells": len(auxiliary), "auxiliary_view_QA": 500 * len(auxiliary), "new_GPU_QA_by_packaging": 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-bundle", type=Path); parser.add_argument("--check-readiness", action="store_true")
    args = parser.parse_args()
    if args.check_readiness:
        rows = r7.formal_rows(r7.MANIFEST); ready = r7.readiness(rows, r7.HARVEST)
        print(json.dumps({k: ready[k] for k in ("status", "expected_cells", "closed_metadata_cells", "ready")})); return
    require(args.prepare_bundle is not None, "Explicit new bundle output required")
    print(json.dumps(prepare_bundle(args.prepare_bundle)))


if __name__ == "__main__": main()
