"""Sequential, predeclared read-budget sweep around the frozen run_cycle CLI.

The four budgets and four policies are fixed before subprocess launch. No
quality scores are read and no configuration is selected from child outcomes.
This module imports only the standard library; --dry-run never loads a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


BUDGET_MIB = (4, 8, 16, 32)
POLICIES = ("uniform", "periodic", "table_future", "probe_l7_t0.02")
METHOD = "asym"
SCHEMA = "trajkv.budget_sweep.v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "samples", "legacy-scripts", "kivi-source", "output", "gain-table"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--count", type=int, default=16, help="First N rows of the same replication32 manifest; default 16")
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--max-patches", type=int, default=4)
    p.add_argument("--description-tokens", type=int, default=96)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--max-replays", type=int, default=2)
    p.add_argument("--hide-visual-after-description", action="store_true")
    p.add_argument("--save-states", action="store_true")
    p.add_argument("--python", default=sys.executable, help="Worker interpreter used for every child")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print and freeze argv/source identities only; no child process or complete marker")
    return p.parse_args(argv)


def build_plan(args) -> dict:
    for name in ("model", "samples", "legacy_scripts", "kivi_source", "output", "gain_table"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.count, args.turns, args.max_tokens, args.max_patches, args.description_tokens) < 1:
        raise ValueError("Counts, turn/token limits, and patch limits must be positive")
    source = Path(__file__).resolve()
    cycle = source.with_name("run_cycle.py")
    paths = [source, cycle, source.with_name("bank.py"), source.with_name("self_probe.py"),
             source.with_name("fused_progressive.py"), args.legacy_scripts / "p202_fused.py",
             args.samples, args.gain_table, args.kivi_source]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    table = read_json(args.gain_table)
    if not table.get("tables", {}).get("future", {}).get("phase_layer_group_ranking"):
        raise ValueError("A frozen future phase/layer-group gain table is required")
    rows = read_json(args.samples)
    if not isinstance(rows, list) or len(rows) < args.count:
        raise ValueError("Sample manifest must contain at least --count rows")
    selected = rows[:args.count]
    if len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Selected session IDs must be unique")
    image_ids = [str(r["source_image_id"]) for r in selected]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("This sweep requires distinct source images for the selected sessions")
    common = [str(args.python), str(cycle)]
    for name in ("model", "samples", "legacy-scripts", "kivi-source", "gain-table"):
        common += ["--" + name, str(getattr(args, name.replace("-", "_")))]
    common += ["--methods", METHOD, "--policies", *POLICIES,
               "--count", str(args.count), "--start", "0", "--turns", str(args.turns),
               "--max-tokens", str(args.max_tokens), "--max-patches", str(args.max_patches),
               "--description-tokens", str(args.description_tokens),
               "--margin", str(args.margin), "--max-replays", str(args.max_replays)]
    if args.hide_visual_after_description:
        common.append("--hide-visual-after-description")
    if args.save_states:
        common.append("--save-states")
    children = []
    for mib in BUDGET_MIB:
        output = args.output / ("budget_" + str(mib) + "MiB")
        argv = common + ["--read-budget-bytes", str(mib * 1024 * 1024), "--output", str(output)]
        children.append({"budget_mib": mib, "read_budget_bytes": mib * 1024 * 1024,
                         "output": str(output), "argv": argv,
                         "expected_records": len(selected) * len(POLICIES)})
    model_identity = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        path = args.model / name
        if path.is_file():
            model_identity[name] = digest(path)
    return {"schema": SCHEMA, "method": METHOD, "policies": list(POLICIES),
            "budgets_mib": list(BUDGET_MIB), "selection_rule": "first count rows, original input order, start=0",
            "selected": [{"session_id": r["id"], "source_image_id": str(r["source_image_id"]),
                          "image_hashes": r["hashes"], "turns": min(args.turns, len(r["questions"]))} for r in selected],
            "sources": {str(path): digest(path) for path in paths},
            "model_identity": model_identity, "children": children,
            "quality_outcomes_used_for_selection": False,
            "read_budget_scope": "extra logical tensor bytes in frozen run_cycle; not measured HBM traffic",
            "model_reload": "one independent child process per budget"}


def assert_sources_unchanged(plan: dict) -> None:
    for name, expected in plan["sources"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError("Frozen sweep source changed: " + name)


def check_child(plan: dict, child: dict) -> dict:
    """Check completeness and pairing without scoring or selecting any output."""
    folder = Path(child["output"])
    required = [folder / name for name in ("manifest.json", "records.jsonl", "complete.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Child did not produce all completion artifacts: " + str(folder))
    manifest, complete = read_json(required[0]), read_json(required[2])
    ids = [r["session_id"] for r in plan["selected"]]
    if manifest.get("selected_ids") != ids:
        raise RuntimeError("Child selected IDs differ from the paired sweep cohort")
    arguments = manifest["arguments"]
    if arguments.get("methods") != [METHOD] or arguments.get("policies") != list(POLICIES):
        raise RuntimeError("Child strategies differ from the predeclared sweep")
    if arguments.get("read_budget_bytes") != child["read_budget_bytes"]:
        raise RuntimeError("Child budget differs from the predeclared sweep")
    records = [json.loads(line) for line in required[1].read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(sid, METHOD, policy) for sid in ids for policy in POLICIES}
    keys = [(r["session_id"], r["method"], r["policy"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError("Missing or duplicate child session/method/policy records")
    source_images = {r["session_id"]: r for r in plan["selected"]}
    for row in records:
        selected = source_images[row["session_id"]]
        if str(row["image_id"]) != selected["source_image_id"] or len(row["turns"]) != selected["turns"]:
            raise RuntimeError("Child changed the source-image pairing or turn count")
        if not row.get("read_budget_ok") or row["extra_logical_read_bytes"] > child["read_budget_bytes"]:
            raise RuntimeError("Child exceeded its declared logical read budget")
    return {"budget_mib": child["budget_mib"], "records": len(records),
            "paired_images": len(ids), "complete": complete,
            "artifacts": {path.name: digest(path) for path in required},
            "quality_scores_read": False}


def execute(args, plan: dict, runner=subprocess.run) -> None:
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume only for the identical frozen plan")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    plan_path = args.output / "sweep_manifest.json"
    if plan_path.exists():
        saved = read_json(plan_path)
        if saved["plan"] != plan:
            raise RuntimeError("Resume requires the identical source hashes, selection, and argv")
    else:
        write_json(plan_path, {"created_at_utc": now(), "status": "PLANNED", "plan": plan})
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_NO_GPU", "manifest": str(plan_path),
                          "children": plan["children"]}, ensure_ascii=False), flush=True)
        return
    status_path = args.output / "sweep_status.json"
    status = read_json(status_path) if status_path.exists() else {"status": "RUNNING", "attempts": [], "verified_children": []}
    verified = []
    for child in plan["children"]:
        assert_sources_unchanged(plan)
        folder = Path(child["output"])
        if args.resume and (folder / "complete.json").exists():
            verified.append(check_child(plan, child))
            continue
        command = list(child["argv"])
        if folder.exists():
            if not args.resume:
                raise FileExistsError(folder)
            command.append("--resume")
        attempt = {"budget_mib": child["budget_mib"], "argv": command,
                   "started_at_utc": now(), "source_hashes_verified": True}
        status["attempts"].append(attempt)
        status["status"] = "RUNNING"
        write_json(status_path, status)
        logpath = args.output / ("budget_" + str(child["budget_mib"]) + "MiB.child.log")
        try:
            with logpath.open("a", encoding="utf-8") as log:
                outcome = runner(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            attempt.update(returncode=outcome.returncode, finished_at_utc=now(), log=str(logpath))
            if outcome.returncode != 0:
                raise RuntimeError("Child process failed for budget " + str(child["budget_mib"]) + " MiB")
            assert_sources_unchanged(plan)
            verified.append(check_child(plan, child))
        except BaseException as exc:
            status.update(status="FAILED", failure=str(exc), failed_budget_mib=child["budget_mib"])
            write_json(status_path, status)
            raise
        status["verified_children"] = verified
        write_json(status_path, status)
        print(json.dumps({"status": "CHILD_COMPLETE", "budget_mib": child["budget_mib"],
                          "records": verified[-1]["records"]}), flush=True)
    assert_sources_unchanged(plan)
    # Each child must remain structurally complete at the point of parent closeout.
    verified = [check_child(plan, child) for child in plan["children"]]
    status.update(status="COMPLETE", verified_children=verified, finished_at_utc=now())
    write_json(status_path, status)
    write_json(args.output / "complete.json", {
        "schema": SCHEMA, "status": "ALL_FOUR_BUDGETS_COMPLETE", "finished_at_utc": now(),
        "sweep_manifest_sha256": digest(plan_path), "method": METHOD, "policies": list(POLICIES),
        "budgets_mib": list(BUDGET_MIB), "sessions_per_budget": len(plan["selected"]),
        "records": sum(item["records"] for item in verified), "children": verified,
        "paid_calls": 0, "quality_scores_read": False})
    print(json.dumps({"status": "ALL_FOUR_BUDGETS_COMPLETE", "output": str(args.output)}), flush=True)


def main(argv=None):
    args = parse_args(argv)
    execute(args, build_plan(args))


if __name__ == "__main__":
    main()
