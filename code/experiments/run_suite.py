"""Sequential fixed-config worker; no model import, scoring or parameter search."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ENTRYPOINTS = {"run_cycle.py", "kvarn_quality_smoke.py"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def option(arguments, flag):
    for index, item in enumerate(arguments):
        if item == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def plan_suite(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("runs"), list) or not config["runs"]:
        raise ValueError("config.runs must be a nonempty list")
    seen, runs, input_files = set(), [], set()
    for row in config["runs"]:
        if not isinstance(row, dict):
            raise ValueError("Every run must be an object")
        name, entrypoint, arguments = row.get("name"), row.get("entrypoint"), row.get("arguments")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) is None:
            raise ValueError("Run name must be a safe single path segment: " + repr(name))
        if name.casefold() in seen:
            raise ValueError("Duplicate run name: " + name)
        seen.add(name.casefold())
        if entrypoint not in ENTRYPOINTS:
            raise ValueError("Only same-directory immutable entrypoints are allowed")
        if not isinstance(arguments, list) or not all(isinstance(x, str) and x and "\x00" not in x for x in arguments):
            raise ValueError("arguments must be a list of nonempty strings without NUL")
        for token in arguments:
            key = token.split("=", 1)[0]
            # argparse accepts long-option abbreviations; reserve those too.
            if key == "--" or (key.startswith("--") and any(x.startswith(key) for x in ("--output", "--resume"))):
                raise ValueError("Suite owns --output/--resume, including abbreviations")
        for flag in ("--samples", "--gain-table", "--reference", "--kivi-source"):
            value = option(arguments, flag)
            if value:
                path = Path(value)
                path = path if path.is_absolute() else PROJECT / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                input_files.add(path.resolve())
        legacy = option(arguments, "--legacy-scripts")
        if legacy:
            path = Path(legacy)
            path = path if path.is_absolute() else PROJECT / path
            kernel = path / "p202_fused.py"
            if not kernel.is_file():
                raise FileNotFoundError(kernel)
            input_files.add(kernel.resolve())
        model = option(arguments, "--model")
        if model:
            path = Path(model)
            path = path if path.is_absolute() else PROJECT / path
            for filename in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
                if (path / filename).is_file():
                    input_files.add((path / filename).resolve())
        destination = output / name
        runs.append({"name": name, "entrypoint": entrypoint, "output": str(destination),
                     "argv": [sys.executable, str(HERE / entrypoint), *arguments, "--output", str(destination)]})
    # Hash local entrypoints and every local experiment helper without importing
    # torch. Hash reference-port helpers too when they are present in the bundle.
    sources = set(HERE.glob("*.py"))
    byte_helper = PROJECT / "code/bytes.py"
    if byte_helper.is_file():
        sources.add(byte_helper)
    ports = PROJECT / "third_party/reference_ports"
    if ports.is_dir():
        sources.update(ports.rglob("*.py"))
    for name in ENTRYPOINTS:
        if not (HERE / name).is_file():
            raise FileNotFoundError(HERE / name)
    return {"schema": "trajkv.fixed_suite.v1", "config_path": str(config_path),
            "config_sha256": digest(config_path), "config": config,
            "cwd": str(PROJECT), "python": sys.executable,
            "source_hashes": {str(p.resolve()): digest(p) for p in sorted(sources)},
            "input_hashes": {str(p): digest(p) for p in sorted(input_files)},
            "runs": runs, "quality_scores_used_for_control": False}


def unchanged(plan):
    if digest(plan["config_path"]) != plan["config_sha256"]:
        raise RuntimeError("Suite config identity changed")
    for group in ("source_hashes", "input_hashes"):
        for path, expected in plan[group].items():
            if digest(path) != expected:
                raise RuntimeError("Suite source/input identity changed: " + path)


def complete_identity(run):
    path = Path(run["output"]) / "complete.json"
    if not path.is_file():
        raise RuntimeError("Child exited without complete.json: " + run["name"])
    if not isinstance(load(path), dict):
        raise ValueError("Child completion must be a JSON object")
    return {"path": str(path), "sha256": digest(path)}


def execute(plan, output, *, resume=False, dry_run=False, runner=subprocess.run):
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError("Use --resume only for an identical existing suite")
    output.mkdir(parents=True, exist_ok=resume)
    manifest = output / "suite_manifest.json"
    if manifest.exists():
        if load(manifest)["plan"] != plan:
            raise RuntimeError("Resume config/source/input/argv identity changed")
    else:
        save(manifest, {"created_at_utc": timestamp(), "plan": plan})
    unchanged(plan)
    if dry_run:
        print(json.dumps({"status": "DRY_RUN_NO_CHILDREN", "manifest": str(manifest), "runs": plan["runs"]}), flush=True)
        return
    # A resumed attempt must not leave a stale parent marker visible if any
    # previously completed child has disappeared or its rerun later fails.
    parent_complete = output / "complete.json"
    if parent_complete.exists():
        parent_complete.unlink()
    status_path = output / "suite_status.json"
    status = load(status_path) if status_path.exists() else {"status": "RUNNING", "runs": {}, "attempts": []}
    for run in plan["runs"]:
        unchanged(plan)
        previous = status["runs"].get(run["name"], {})
        if resume and previous.get("returncode") == 0 and (Path(run["output"]) / "complete.json").is_file():
            if complete_identity(run) != previous["complete"]:
                raise RuntimeError("Completed child identity changed: " + run["name"])
            continue
        command = list(run["argv"])
        if Path(run["output"]).exists():
            if not resume:
                raise FileExistsError(run["output"])
            command.append("--resume")
        attempt = {"name": run["name"], "argv": command, "started_at_utc": timestamp()}
        status["attempts"].append(attempt)
        status["status"] = "RUNNING"
        save(status_path, status)
        try:
            log_path = output / (run["name"] + ".child.log")
            with log_path.open("a", encoding="utf-8") as stream:
                result = runner(command, cwd=plan["cwd"], stdout=stream, stderr=subprocess.STDOUT, check=False)
            attempt.update(returncode=result.returncode, finished_at_utc=timestamp(), log=str(log_path))
            if result.returncode != 0:
                raise RuntimeError("Child failed: " + run["name"])
            unchanged(plan)
            completed = complete_identity(run)
            status["runs"][run["name"]] = {"returncode": 0, "complete": completed, "argv": command}
            save(status_path, status)
            print(json.dumps({"status": "CHILD_COMPLETE", "name": run["name"]}), flush=True)
        except BaseException as error:
            status.update(status="FAILED", failure=str(error), failed_run=run["name"])
            save(status_path, status)
            raise
    unchanged(plan)
    for run in plan["runs"]:
        if complete_identity(run) != status["runs"][run["name"]]["complete"]:
            raise RuntimeError("Child completion changed before parent closeout")
    status.update(status="COMPLETE", finished_at_utc=timestamp())
    save(status_path, status)
    save(output / "complete.json", {"schema": "trajkv.fixed_suite.v1", "status": "ALL_SUITE_RUNS_COMPLETE",
         "finished_at_utc": timestamp(), "suite_manifest_sha256": digest(manifest),
         "runs": status["runs"], "quality_scores_read": False})
    print(json.dumps({"status": "ALL_SUITE_RUNS_COMPLETE", "output": str(output)}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    execute(plan_suite(args.config, args.output), args.output, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
