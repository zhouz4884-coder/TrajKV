"""Independent, read-only evidence audit; writes only its requested report.

Checks receipts, archived source hashes, frozen splits, image hashes, gain-table
isolation, budget exclusions and summary/record counts. A running or partially
downloaded run is PENDING, not failed. Stale exploratory summaries are retained.
Nothing is deleted, rerun, stopped or silently repaired by this validator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re


def truth(value):
    return value is True or str(value).lower() in {"true", "1"}


def row_id(row):
    return str(row.get("id", row.get("session_id", "")))


class Audit:
    def __init__(self, root, remote_sources=False):
        self.root = Path(root).resolve()
        self.findings = []
        self.hashes = {}
        self.contents = {}
        self.lines = {}
        self.remote_sources = remote_sources
        self.client = self.sftp = None
        self.remote_hash_cache = {}

    def note(self, level, code, path, **details):
        item = {"level": level, "code": code, "path": str(path), **details}
        self.findings.append(item)
        return item

    def raw(self, path):
        path = Path(path).resolve()
        if path not in self.contents:
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            self.contents[path] = data
            self.hashes[path] = hashlib.sha256(data).hexdigest()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                self.note("PENDING", "FILE_CHANGED_DURING_READ", path)
        return self.contents[path]

    def sha(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for data in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(data)
            self.hashes[path] = h.hexdigest()
        return self.hashes[path]

    def json(self, path):
        return json.loads(self.raw(path).decode("utf-8-sig"))

    def jsonl(self, path):
        path = Path(path).resolve()
        if path in self.lines:
            return self.lines[path]
        data = self.raw(path)
        lines = data.decode("utf-8-sig").splitlines()
        rows = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not data.endswith((b"\n", b"\r")):
                    self.note("PENDING", "UNFINISHED_LAST_JSONL_ROW", path, line=index + 1)
                else:
                    self.note("ERROR", "INVALID_JSONL", path, line=index + 1)
                    break
        self.lines[path] = rows
        return rows

    def local_path(self, value):
        """Resolve only project/sibling workspace references; never scan a host."""
        text = str(value).replace("\\", "/")
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(self.root.parent)
                return candidate.resolve()
            except ValueError:
                pass
        marker = "/" + self.root.name + "/"
        if marker.lower() in text.lower():
            start = text.lower().rfind(marker.lower()) + len(marker)
            relative = text[start:]
            if relative.startswith("jobs/") and ".source/" in relative:
                mapped = self.root / "audit/job_sources" / relative[len("jobs/"):]
                if mapped.exists():
                    return mapped
            return self.root / relative
        if text.startswith(("code/", "configs/", "results/", "audit/", "third_party/")):
            return self.root / text
        return None

    def remote_snapshot_hash(self, run, basename):
        if not self.remote_sources:
            return None
        relative_run = run.relative_to(self.root / "results").as_posix()
        cache_key = (relative_run, basename)
        if cache_key in self.remote_hash_cache:
            return self.remote_hash_cache[cache_key]
        if self.client is None:
            import sys
            sys.path.insert(0, str(self.root / "code"))
            import remote
            self.client = remote.connect()
            self.sftp = self.client.open_sftp()
            self.remote_root = remote.REMOTE_ROOT
        path = self.remote_root + "/results/" + relative_run + "/source_snapshot/" + basename
        try:
            h = hashlib.sha256()
            with self.sftp.open(path, "rb") as stream:
                for block in iter(lambda: stream.read(65536), b""):
                    h.update(block)
            result = h.hexdigest()
        except OSError:
            result = None
        self.remote_hash_cache[cache_key] = result
        return result

    def other_archived_source(self, run, basename, expected):
        """Content-addressed recovery evidence, without rewriting any archive."""
        recovered = self.root / "audit/recovered-by-content-hash" / expected / basename
        if recovered.is_file():
            if self.sha(recovered) == expected:
                return "RECOVERED_BY_EXACT_CONTENT_HASH", str(recovered)
            self.note("ERROR", "RECOVERED_OBJECT_HASH_MISMATCH", recovered,
                      expected_sha256=expected, actual_sha256=self.sha(recovered))
        candidates = [p.parent for p in sorted((self.root / "results").rglob("manifest.json"))
                      if p.parent != run]
        for other in candidates:
            local = other / "source_snapshot" / basename
            if local.is_file() and self.sha(local) == expected:
                return "ARCHIVED_OTHER_RUN_IDENTICAL_CONTENT", str(local)
        if self.remote_sources:
            for other in candidates:
                if self.remote_snapshot_hash(other, basename) == expected:
                    return "ARCHIVED_OTHER_REMOTE_RUN_IDENTICAL_CONTENT", (
                        self.remote_root + "/results/" + other.relative_to(self.root / "results").as_posix()
                        + "/source_snapshot/" + basename)
        return None, None

    def immutable_job_source_hash(self, source):
        if not self.remote_sources:
            return None
        text = str(source).replace("\\", "/")
        if self.client is None:
            import sys
            sys.path.insert(0, str(self.root / "code"))
            import remote
            self.client = remote.connect()
            self.sftp = self.client.open_sftp()
            self.remote_root = remote.REMOTE_ROOT
        if not (text.startswith(self.remote_root + "/jobs/") and ".source/" in text
                and ".." not in text.split("/") and text.endswith(".py")):
            return None
        cache_key = ("immutable_job_source", text)
        if cache_key not in self.remote_hash_cache:
            try:
                h = hashlib.sha256()
                with self.sftp.open(text, "rb") as stream:
                    for block in iter(lambda: stream.read(65536), b""):
                        h.update(block)
                self.remote_hash_cache[cache_key] = h.hexdigest()
            except OSError:
                self.remote_hash_cache[cache_key] = None
        return self.remote_hash_cache[cache_key]

    def declared_input_hash(self, source, expected=None):
        local = self.local_path(source)
        if local and local.is_file():
            return self.sha(local), str(local)
        if expected:
            basename = str(source).replace("\\", "/").rsplit("/", 1)[-1]
            retained = self.root / "audit/recovered-by-content-hash" / expected / basename
            if retained.is_file():
                return self.sha(retained), str(retained)
        model_path = self.root / "audit/model_identity.json"
        if model_path.exists():
            identity = self.json(model_path)
            if str(source).replace("\\", "/").startswith(identity.get("model_root", "__none__") + "/"):
                basename = str(source).replace("\\", "/").rsplit("/", 1)[-1]
                item = next((x for x in identity.get("files", []) if x["name"] == basename), None)
                if item and item.get("exists") and item.get("stable_during_hash"):
                    return item["sha256"], str(model_path) + "#" + basename
        return None, None

    def sources(self, run, manifest):
        declared = manifest.get("sources", manifest.get("source_sha256", manifest.get("contract", {}).get("sources", {})))
        results = []
        for source, expected in declared.items():
            basename = str(source).replace("\\", "/").rsplit("/", 1)[-1]
            snapshot = run / "source_snapshot" / basename
            if not snapshot.exists():
                numbered = [p for p in (run / "source_snapshot").glob("*_" + basename)
                            if re.match(r"^\d+_", p.name)]
                exact = [p for p in numbered if self.sha(p) == expected]
                if exact:
                    snapshot = exact[0]
                elif len(numbered) == 1:
                    snapshot = numbered[0]
            current = self.local_path(source)
            origin = str(snapshot)
            if snapshot.is_file():
                actual, status = self.sha(snapshot), "ARCHIVED_LOCAL"
            else:
                actual = self.remote_snapshot_hash(run, basename)
                if actual is not None:
                    status = "ARCHIVED_REMOTE_READ_ONLY"
                    origin = self.remote_root + "/results/" + run.relative_to(self.root / "results").as_posix() + "/source_snapshot/" + basename
                elif self.immutable_job_source_hash(source) is not None:
                    actual = self.immutable_job_source_hash(source)
                    status, origin = "IMMUTABLE_JOB_SOURCE_REMOTE", str(source)
                elif current and current.is_file() and self.sha(current) == expected:
                    actual, status = expected, "MATCH_CURRENT_ONLY"
                    origin = str(current)
                else:
                    status, origin = self.other_archived_source(run, basename, expected)
                    actual = expected if status else None
                    status = status or "HISTORICAL_SOURCE_NOT_LOCAL"
            match = actual == expected if actual is not None else None
            if match is False:
                self.note("ERROR", "ARCHIVED_SOURCE_HASH_MISMATCH", snapshot,
                          source=source, expected=expected, actual=actual)
            elif match is None:
                # Current source may legitimately have evolved since this run.
                self.note("PENDING", "SOURCE_SNAPSHOT_NOT_AVAILABLE", snapshot,
                          declared_source=source, expected_sha256=expected)
            results.append({"source": source, "expected_sha256": expected,
                            "observed_sha256": actual, "status": status, "match": match,
                            "observed_source_location": origin})
        if not declared:
            self.note("WARNING", "NO_DECLARED_SOURCE_HASHES", run / "manifest.json")
        return results

    def budgets(self, path, records):
        violations, inconsistencies = [], []
        for record in records:
            if record.get("schema") == "trajkv.timing_control.v1":
                record = record.get("runner_result", record)
            if "method" not in record:
                continue  # mechanisms have separate diagnostic budgets
            events = record.get("read_events", [])
            extra = 0
            for event in events:
                for key in ("base_visual_bytes", "residual_bytes", "text_bytes", "reference_bytes"):
                    if event.get(key, 0) < 0:
                        self.note("ERROR", "NEGATIVE_EVENT_BYTES", path, session=record["session_id"], field=key)
                extra += int(event.get("residual_bytes", 0))
                if event.get("pass_kind") in {"replay", "repair", "probe"}:
                    extra += sum(int(event.get(k, 0)) for k in ("base_visual_bytes", "text_bytes", "reference_bytes"))
            declared = record.get("extra_logical_read_bytes")
            cap = record.get("read_budget_bytes")
            actual_ok = cap is None or extra <= cap
            key = {"session_id": record["session_id"], "method": record["method"],
                   "policy": record.get("policy", "base")}
            if declared is not None and declared != extra:
                item = {**key, "declared_extra_bytes": declared, "recounted_extra_bytes": extra}
                inconsistencies.append(item)
                self.note("ERROR", "READ_LEDGER_RECOUNT_MISMATCH", path, **item)
            if record.get("read_budget_ok") is not None and record["read_budget_ok"] != actual_ok:
                self.note("ERROR", "BUDGET_OK_FLAG_MISMATCH", path, **key,
                          reported=record["read_budget_ok"], recounted=actual_ok)
            if not actual_ok or record.get("read_budget_ok") is False:
                violations.append({**key, "cap_bytes": cap, "extra_bytes": extra,
                                   "exclude_from_equal_budget_claims": True,
                                   "preserve_original_record": True})
        if violations:
            self.note("EXCLUSION", "OVER_BUDGET_RECORDS_RETAINED", path, count=len(violations),
                      claim_scope="exclude from equal-budget claims; no automatic deletion or experiment halt")
        return {"violations": violations, "ledger_inconsistencies": inconsistencies}

    def container(self, run, plan_path):
        saved = self.json(plan_path)
        plan = saved["plan"]
        suite = plan_path.name == "suite_manifest.json"
        children = plan.get("runs" if suite else "children", [])
        complete_path = run / "complete.json"
        complete = self.json(complete_path) if complete_path.exists() else None
        children_report = []
        for child in children:
            child_path = self.local_path(child["output"]) or run / (
                child.get("name") or "budget_" + str(child.get("budget_mib")) + "MiB")
            receipt_path = child_path / "complete.json"
            exists = receipt_path.exists()
            expected_hash = None
            if complete and suite:
                expected_hash = complete.get("runs", {}).get(child.get("name"), {}).get("complete", {}).get("sha256")
            elif complete:
                old = next((x for x in complete.get("children", []) if x.get("budget_mib") == child.get("budget_mib")), {})
                expected_hash = old.get("artifacts", {}).get("complete.json")
            match = self.sha(receipt_path) == expected_hash if exists and expected_hash else None
            if complete and expected_hash is None:
                self.note("ERROR", "COMPOSITE_PARENT_OMITS_CHILD_RECEIPT", plan_path,
                          child=child.get("name", child.get("budget_mib")))
            if match is False:
                self.note("ERROR", "COMPOSITE_CHILD_RECEIPT_HASH_MISMATCH", receipt_path)
            if not exists:
                self.note("WARNING" if complete else "PENDING", "COMPOSITE_CHILD_NOT_COMPLETE_OR_NOT_LOCAL", child_path)
            child_manifest_path = child_path / "manifest.json"
            if not suite and child_manifest_path.exists():
                child_manifest = self.json(child_manifest_path)
                child_args = child_manifest.get("arguments", {})
                expected_ids = [r["session_id"] for r in plan.get("selected", [])]
                if (child_args.get("read_budget_bytes") != child.get("read_budget_bytes")
                        or child_manifest.get("selected_ids") != expected_ids
                        or child_args.get("methods") != [plan.get("method")]
                        or child_args.get("policies") != plan.get("policies")):
                    self.note("ERROR", "BUDGET_SWEEP_CHILD_CONTRACT_CHANGED", child_manifest_path)
            children_report.append({"output": str(child_path), "complete_present": exists, "receipt_hash_match": match})
        if complete:
            expected_plan = complete.get("suite_manifest_sha256" if suite else "sweep_manifest_sha256")
            if expected_plan and expected_plan != self.sha(plan_path):
                self.note("ERROR", "COMPOSITE_PLAN_HASH_MISMATCH", plan_path)
            if not suite and "records" in complete:
                expected_records = sum(int(c.get("expected_records", 0)) for c in children)
                if expected_records and complete["records"] != expected_records:
                    self.note("ERROR", "BUDGET_SWEEP_PARENT_RECORD_COUNT_MISMATCH", plan_path,
                              declared=complete["records"], expected=expected_records)
        else:
            self.note("PENDING", "COMPOSITE_SUITE_NOT_COMPLETE", run)
        source_manifest = {"sources": plan.get("source_hashes", plan.get("sources", {}))}
        input_checks = []
        for source, expected in plan.get("input_hashes", {}).items():
            actual, origin = self.declared_input_hash(source, expected)
            match = actual == expected if actual is not None else None
            if match is False:
                self.note("ERROR", "COMPOSITE_DECLARED_INPUT_HASH_MISMATCH", plan_path, source=source)
            elif match is None:
                self.note("PENDING", "COMPOSITE_DECLARED_INPUT_NOT_LOCAL", plan_path, source=source)
            input_checks.append({"source": source, "match": match, "observed_source_location": origin})
        return {"run": run.relative_to(self.root / "results").as_posix(),
                "artifact_kind": "composite_suite" if suite else "budget_sweep",
                "status": "COMPLETE_PARENT_RECEIPT_PRESENT" if complete else "PENDING_COMPOSITE",
                "model_inference_records": None, "parent_is_not_an_extra_model_run": True,
                "completion": complete, "children": children_report, "input_identity_checks": input_checks,
                "sources": self.sources(run, source_manifest)}

    def timing_run(self, run, manifest, complete, records, records_path):
        contract = manifest.get("contract", {})
        configurations = contract.get("configurations", [])
        ids = set(map(str, contract.get("selected_ids", [])))
        expected = len(ids) * len(configurations) * int(contract.get("repeats", 0))
        keys = [(r.get("session_id"), r.get("configuration"), r.get("repeat")) for r in records]
        if len(keys) != len(set(keys)):
            self.note("ERROR", "DUPLICATE_TIMING_MEASUREMENT", records_path)
        if complete and complete.get("measured_trials") != len(records):
            self.note("WARNING", "TIMING_RECEIPT_LOCAL_COUNT_MISMATCH", records_path)
        if len(records) != expected:
            self.note("WARNING" if complete else "PENDING", "TIMING_MEASUREMENTS_NOT_COMPLETE", records_path,
                      expected=expected, observed=len(records))
        for record in records:
            if record.get("quality") != "not_scored" or record.get("application_accuracy") is not None:
                self.note("ERROR", "TEACHER_FORCED_TIMING_MISLABELLED_AS_QUALITY", records_path)
            if record.get("token_identity", {}).get("output_token_ids_identical") is False:
                self.note("ERROR", "TEACHER_FORCED_TOKEN_SCHEDULE_CHANGED", records_path, trial_id=record.get("trial_id"))
        nonquiet = sum(r.get("quiet_verified") is False for r in records)
        if nonquiet:
            self.note("EXCLUSION", "NONQUIET_TIMING_TRIALS_RETAINED", records_path, count=nonquiet)
        if not complete:
            self.note("PENDING", "TIMING_RUN_NOT_COMPLETE", run)
        return {"run": run.relative_to(self.root / "results").as_posix(), "artifact_kind": "teacher_forced_timing",
                "status": "COMPLETE_TIMING_RECEIPT_PRESENT" if complete else "PENDING_TIMING",
                "records": len(records), "measurement_trials": len(records), "expected_measurements": expected,
                "application_quality_records": 0, "teacher_forced_tokens_are_not_quality": True,
                "completion": complete, "sources": self.sources(run, manifest),
                "budget_audit": self.budgets(records_path, records)}

    def runs(self):
        reports = []
        root = self.root / "results"
        containers = sorted(set(root.rglob("suite_manifest.json")) | set(root.rglob("sweep_manifest.json")))
        for path in containers:
            reports.append(self.container(path.parent, path))
        for manifest_path in sorted(root.rglob("manifest.json")):
            run = manifest_path.parent
            manifest = self.json(run / "manifest.json")
            complete_path = run / "complete.json"
            complete = self.json(complete_path) if complete_path.exists() else None
            if manifest.get("kind") == "optional_hardware_counter_side_check":
                reports.append({"run": run.relative_to(root).as_posix(), "artifact_kind": "hardware_counter_probe",
                                "status": (complete or {}).get("status", "PENDING_HARDWARE_PROBE"),
                                "captured_kernels": (complete or {}).get("captured_kernels"),
                                "model_inference_records": None, "application_quality_records": None,
                                "physical_dram_bytes": (complete or {}).get("physical_dram_bytes"),
                                "not_a_zero_quality_model_run": True, "completion": complete,
                                "sources": self.sources(run, manifest)})
                continue
            records_path = run / "records.jsonl"
            session_path = run / "sessions.jsonl"
            path = records_path if records_path.exists() else session_path
            records = self.jsonl(path) if path.exists() else []
            if "contract" in manifest and manifest.get("quality") == "not_scored":
                reports.append(self.timing_run(run, manifest, complete, records, path))
                continue
            sessions = self.jsonl(session_path) if session_path.exists() else records
            selected = set(map(str, manifest.get("selected_ids", [])))
            args = manifest.get("arguments", {})
            record_ids = {str(r.get("session_id")) for r in records if r.get("session_id")}
            status = "COMPLETE_RECEIPT_PRESENT" if complete else "PENDING_NO_COMPLETION_RECEIPT"
            expected_count = None
            if args.get("methods") and selected:
                expected = {(sid, method, policy) for sid in selected for method in args["methods"]
                            for policy in (args.get("policies", ["base"]) if method.startswith("asym") else ["base"])}
                observed_keys = [(str(r["session_id"]), r["method"], r.get("policy", "base"))
                                 for r in records if "method" in r]
                expected_count = len(expected)
                missing = expected - set(observed_keys)
                unexpected = set(observed_keys) - expected
                if len(observed_keys) != len(set(observed_keys)):
                    self.note("ERROR", "DUPLICATE_RUN_RECORD", path)
                if unexpected:
                    self.note("ERROR", "UNEXPECTED_RUN_KEYS", path, keys=[list(k) for k in sorted(unexpected)])
                if missing:
                    self.note("WARNING" if complete else "PENDING", "MISSING_EXPECTED_RECORDS", path,
                              count=len(missing), completion_receipt=complete is not None)
                    status = "NEEDS_LOCAL_RECORD_RECONCILIATION" if complete else "PENDING_RUNNING_OR_NOT_DOWNLOADED"
            elif selected - record_ids:
                self.note("WARNING" if complete else "PENDING", "MISSING_EXPECTED_SESSIONS", path,
                          session_ids=sorted(selected - record_ids))
            if complete:
                if "records_added" in complete and complete["records_added"] > len(records):
                    self.note("WARNING", "RECEIPT_EXCEEDS_LOCAL_RECORDS", path,
                              receipt=complete["records_added"], local=len(records))
                if "sessions_added" in complete:
                    local_sessions = {r.get("session_id") for r in sessions}
                    if complete["sessions_added"] > len(local_sessions):
                        self.note("WARNING", "RECEIPT_EXCEEDS_LOCAL_SESSIONS", session_path)
            else:
                self.note("PENDING", "RUN_NOT_FINISHED_OR_NOT_DOWNLOADED", run)
            reports.append({"run": run.relative_to(root).as_posix(), "artifact_kind": "model_or_mechanism_run",
                            "status": status, "records": len(records),
                            "sessions": len(record_ids), "expected_records": expected_count,
                            "completion": complete, "sources": self.sources(run, manifest),
                            "budget_audit": self.budgets(path, records)})
        return reports

    def splits_and_images(self, hash_images=True):
        manifests = {}
        image_results = {}
        for path in sorted((self.root / "configs/data").glob("*runner.json")):
            rows = self.json(path)
            identities = set()
            sources = set()
            ids = set()
            session_images = {}
            for row in rows:
                ids.add(row_id(row))
                sources.add(str(row.get("source_image_id", row.get("image_id", ""))))
                images, hashes = row.get("images", []), row.get("hashes", [])
                row_images = []
                if len(images) != len(hashes):
                    self.note("ERROR", "IMAGE_HASH_COUNT_MISMATCH", path, session_id=row_id(row))
                for index, filename in enumerate(images):
                    expected = hashes[index] if index < len(hashes) else None
                    identity = "sha256:" + expected if expected else "source:" + str(row.get("source_image_id"))
                    identities.add(identity)
                    row_images.append(identity)
                    if filename not in image_results:
                        p = Path(filename)
                        exists = p.is_file()
                        actual = self.sha(p) if exists and hash_images else None
                        image_results[filename] = {"exists": exists, "sha256": actual,
                                                   "expected_sha256": expected,
                                                   "match": actual == expected if actual else None}
                        if not exists:
                            self.note("PENDING", "LOCAL_IMAGE_NOT_AVAILABLE", p)
                        elif actual is not None and expected and actual != expected:
                            self.note("ERROR", "LOCAL_IMAGE_HASH_MISMATCH", p, expected=expected, actual=actual)
                session_images[row_id(row)] = sorted(set(row_images))
            manifests[path.stem] = {"path": str(path), "sessions": len(rows), "ids": ids,
                                    "images": identities, "source_image_ids": sources,
                                    "session_images": session_images}
        frozen_path = self.root / "audit/data/frozen_evaluation.json"
        frozen = self.json(frozen_path) if frozen_path.exists() else {"datasets": {}}
        definitions = dict(frozen.get("datasets", {}))
        external_path = self.root / "audit/data/external_eval16_freeze.json"
        if external_path.exists():
            external = self.json(external_path)
            for dataset in ("docvqa", "gqa"):
                name = dataset + "_external_eval16_remote_runner"
                key = "configs/data/" + name + ".json"
                if key in external.get("outputs", {}):
                    definitions[name] = {"sha256": external["outputs"][key],
                                         "ids": external.get("selections", {}).get(dataset, {}).get("ids", [])}
        confirmation_path = self.root / "audit/data/docvqa_confirm64_freeze.json"
        if confirmation_path.exists():
            confirmation = self.json(confirmation_path)
            definitions["docvqa_confirm64_remote_runner"] = {
                "sha256": confirmation["manifest_sha256"], "ids": confirmation["ids"]}
        frozen_checks = []
        eval_names = set(definitions)
        dev_names = [name for name in manifests if name not in eval_names and
                     "heldout_candidate" not in name and any(k in name for k in ("dev", "smoke"))]
        dev_images = set().union(*(manifests[n]["images"] for n in dev_names)) if dev_names else set()
        eval_union = set()
        for name, definition in definitions.items():
            path = self.root / "configs/data" / (name + ".json")
            if name not in manifests or not path.exists():
                self.note("PENDING", "FROZEN_EVALUATION_MANIFEST_NOT_AVAILABLE", path)
                continue
            current = manifests[name]
            match = self.sha(path) == definition["sha256"]
            id_match = current["ids"] == set(map(str, definition.get("ids", [])))
            overlap = current["images"] & dev_images
            cross_eval = current["images"] & eval_union
            if not match or not id_match:
                self.note("ERROR", "FROZEN_EVALUATION_CHANGED", path, hash_match=match, id_match=id_match)
            if overlap:
                self.note("ERROR", "FROZEN_EVAL_DEV_IMAGE_OVERLAP", path, images=sorted(overlap))
            if cross_eval:
                self.note("ERROR", "FROZEN_EVALUATION_BANKS_OVERLAP", path, images=sorted(cross_eval))
            eval_union.update(current["images"])
            frozen_checks.append({"manifest": name, "sha256_match": match, "ids_match": id_match,
                                  "dev_image_overlap": len(overlap), "other_frozen_eval_overlap": len(cross_eval)})
        return {"manifests": manifests, "local_images": image_results, "frozen_checks": frozen_checks,
                "development_manifests": dev_names, "frozen_evaluation_manifests": sorted(eval_names)}

    def gain_tables(self, data):
        reports = []
        evals = [data["manifests"][n] for n in data["frozen_evaluation_manifests"] if n in data["manifests"]]
        eval_ids = set().union(*(r["ids"] for r in evals)) if evals else set()
        eval_sources = set().union(*(r["source_image_ids"] for r in evals)) if evals else set()
        eval_hashes = set().union(*(r["images"] for r in evals)) if evals else set()
        for path in sorted((self.root / "configs").glob("*gain*.json")):
            table = self.json(path)
            training = set(map(str, table.get("training_session_ids", [])))
            image_map = table.get("training_image_ids", {})
            training_images = set(map(str, image_map.values() if isinstance(image_map, dict) else image_map))
            training_hashes = set()
            for manifest in data["manifests"].values():
                for sid in training:
                    training_hashes.update(manifest.get("session_images", {}).get(sid, []))
            overlap_ids, overlap_images = training & eval_ids, training_images & eval_sources
            overlap_hashes = training_hashes & eval_hashes
            if overlap_ids or overlap_images or overlap_hashes:
                self.note("ERROR", "GAIN_TABLE_TRAIN_EVAL_LEAKAGE", path,
                          session_ids=sorted(overlap_ids), image_ids=sorted(overlap_images),
                          image_hashes=sorted(overlap_hashes))
            observed_ids = {str(r["session_id"]) for r in table.get("observations", []) if r.get("session_id")}
            if observed_ids - training:
                self.note("ERROR", "GAIN_TABLE_UNDECLARED_TRAINING_OBSERVATIONS", path,
                          session_ids=sorted(observed_ids - training))
            if not training:
                self.note("WARNING", "GAIN_TABLE_TRAINING_IDS_NOT_RECORDED", path)
            source_checks = []
            for source, expected in table.get("source_sha256", {}).items():
                local = self.local_path(source)
                match = self.sha(local) == expected if local and local.exists() else None
                if match is False:
                    self.note("ERROR", "GAIN_TABLE_TRAINING_SOURCE_CHANGED", path, source=source)
                elif match is None:
                    self.note("PENDING", "GAIN_TABLE_TRAINING_SOURCE_UNAVAILABLE", path, source=source)
                source_checks.append({"source": source, "match": match})
            reports.append({"table": str(path), "sha256": self.sha(path), "training_sessions": len(training),
                            "eval_session_overlap": sorted(overlap_ids), "eval_image_overlap": sorted(overlap_images),
                            "eval_image_hash_overlap": sorted(overlap_hashes),
                            "source_checks": source_checks})
        return reports

    def summaries(self):
        reports = []
        for path in sorted((self.root / "results").rglob("summary.json")):
            summary = self.json(path)
            if summary.get("schema") == "trajkv.timing_control_summary.v1":
                records_path = path.parent / "records.jsonl"
                records = self.jsonl(records_path) if records_path.exists() else []
                configured = summary.get("configurations", [])
                declared = sum(int(c.get("measured_trials", 0)) for c in configured)
                if records_path.exists() and declared != len(records):
                    self.note("ERROR", "TIMING_SUMMARY_MEASUREMENT_COUNT_MISMATCH", path,
                              declared=declared, raw_measurements=len(records))
                if summary.get("quality") != "not_scored" or summary.get("application_accuracy") is not None:
                    self.note("ERROR", "TIMING_SUMMARY_MISLABELLED_AS_QUALITY", path)
                for configuration in configured:
                    actual = sum(r.get("configuration") == configuration["configuration"] for r in records)
                    if records_path.exists() and actual != configuration.get("measured_trials"):
                        self.note("ERROR", "TIMING_CONFIGURATION_COUNT_MISMATCH", path,
                                  configuration=configuration["configuration"], declared=configuration.get("measured_trials"), actual=actual)
                reports.append({"summary": str(path), "artifact_kind": "teacher_forced_timing_summary",
                                "declared_measurement_trials": declared, "raw_measurement_trials": len(records),
                                "application_quality_records": 0, "timing_is_not_zero_accuracy": True})
                continue
            inputs = summary.get("input_sha256", {})
            fresh, current_files_match, record_paths, checks = True, True, [], []
            for source, expected in inputs.items():
                local = self.local_path(source)
                match = self.sha(local) == expected if local and local.exists() else None
                current_files_match = current_files_match and match is True
                binding, recovered = local, None
                if match is not True:
                    receipt_path = path.parent / "recovered_inputs/recovery_receipt.json"
                    if receipt_path.exists():
                        receipt = self.json(receipt_path)
                        candidate = self.local_path(receipt.get("snapshot", ""))
                        if (receipt.get("status") == "RECOVERED_EXACT_ORIGINAL_INPUT_SNAPSHOT"
                                and receipt.get("recorded_input_sha256") == expected
                                and receipt.get("snapshot_sha256") == expected
                                and candidate and candidate.is_file() and self.sha(candidate) == expected):
                            binding, recovered = candidate, candidate
                verified = match is True or recovered is not None
                fresh = fresh and verified
                checks.append({"source": source, "match_current_file": match,
                               "declared_original_input_verified": verified,
                               "recovered_original_snapshot": str(recovered) if recovered else None,
                               "historical_partial_status_preserved": summary.get("partial", summary.get("status") == "PARTIAL") if recovered else None})
                if binding and binding.is_file() and (str(source).replace("\\", "/").endswith("records.jsonl")):
                    record_paths.append(binding)
            if inputs and not fresh:
                self.note("WARNING", "STALE_OR_UNAVAILABLE_ANALYSIS_INPUT_SNAPSHOT", path,
                          claim_scope="retain as historical snapshot; regenerate before current-result claims")
            resource_csv, scored_csv = path.parent / "resource_sessions.csv", path.parent / "scored_turns.csv"
            csv_rows = []
            if resource_csv.exists() and "n_records" in summary:
                with resource_csv.open(encoding="utf-8-sig", newline="") as handle:
                    n = sum(1 for _ in csv.DictReader(handle))
                if n != summary["n_records"]:
                    self.note("ERROR", "SUMMARY_RESOURCE_CSV_COUNT_MISMATCH", path, csv_count=n, declared=summary["n_records"])
            if scored_csv.exists():
                with scored_csv.open(encoding="utf-8-sig", newline="") as handle:
                    csv_rows = list(csv.DictReader(handle))
                n = sum(truth(r.get("scored", True)) for r in csv_rows)
                if "n_scored_turns" in summary and n != summary["n_scored_turns"]:
                    self.note("ERROR", "SUMMARY_SCORED_CSV_COUNT_MISMATCH", path, csv_count=n, declared=summary["n_scored_turns"])
            if fresh and record_paths and "n_records" in summary:
                n = sum(len(self.jsonl(p)) for p in record_paths)
                if n != summary["n_records"]:
                    self.note("ERROR", "SUMMARY_RAW_RECORD_COUNT_MISMATCH", path, raw_count=n, declared=summary["n_records"])
            if fresh and record_paths and csv_rows:
                raw_index, raw_reference_index = {}, {}
                by_source, by_source_reference = {}, {}
                for raw_path in record_paths:
                    aliases = {str(raw_path.resolve())}
                    for check in checks:
                        if check.get("recovered_original_snapshot") == str(raw_path):
                            declared = self.local_path(check["source"])
                            if declared:
                                aliases.add(str(declared.resolve()))
                    for record in self.jsonl(raw_path):
                        for turn in record.get("turns", []):
                            key = (str(record.get("session_id")), str(record.get("method")),
                                   str(record.get("policy", "base")), int(turn["turn"]))
                            raw_index[key] = turn.get("answer", "")
                            reference_key = (*key[:3], key[3] - int(record.get("reference_turn_offset", 0)))
                            raw_reference_index[reference_key] = turn.get("answer", "")
                            for alias in aliases:
                                by_source[(alias, *key)] = turn.get("answer", "")
                                by_source_reference[(alias, *reference_key)] = turn.get("answer", "")
                missing, mismatch = 0, 0
                for row in csv_rows:
                    number = row.get("turn", row.get("reference_turn"))
                    if number is None:
                        self.note("ERROR", "SCORED_CSV_HAS_NO_TURN_KEY", path)
                        continue
                    key = (row["session_id"], row["method"], row.get("policy", "base"), int(number))
                    source = self.local_path(row.get("source_records", ""))
                    index = raw_index if "turn" in row else raw_reference_index
                    if source:
                        index = by_source if "turn" in row else by_source_reference
                        key = (str(source.resolve()), *key)
                    if key not in index:
                        missing += 1
                    elif index[key] != row.get("prediction", row.get("answer")):
                        mismatch += 1
                if missing or mismatch:
                    self.note("ERROR", "SCORED_ROWS_RAW_PREDICTION_MISMATCH", path, missing=missing, changed_predictions=mismatch)
            reports.append({"summary": str(path), "inputs_match_current_files": current_files_match if inputs else None,
                            "declared_original_inputs_verified": fresh if inputs else None,
                            "historical_status": summary.get("status"),
                            "input_checks": checks, "declared_records": summary.get("n_records"),
                            "declared_scored_turns": summary.get("n_scored_turns"), "scored_csv_rows": len(csv_rows)})
        return reports

    def model_identity(self):
        path = self.root / "audit/model_identity.json"
        if not path.exists():
            self.note("PENDING", "MODEL_IDENTITY_NOT_RECORDED", path)
            return {"status": "NOT_AVAILABLE"}
        identity = self.json(path)
        if identity.get("status") != "PASS_FULL_MODEL_IDENTITY":
            self.note("PENDING", "MODEL_IDENTITY_HASHING_OR_INCOMPLETE", path, observed_status=identity.get("status"))
        hashes = {r["name"]: r.get("sha256") for r in identity.get("files", []) if r.get("exists")}
        comparisons = []
        for manifest_path in sorted((self.root / "results").rglob("manifest.json")):
            manifest = self.json(manifest_path)
            for name, expected in manifest.get("model_identity", {}).items():
                actual = hashes.get(name)
                match = expected == actual if actual else None
                if match is False:
                    self.note("ERROR", "RUN_MODEL_METADATA_HASH_MISMATCH", manifest_path, filename=name)
                comparisons.append({"manifest": str(manifest_path), "filename": name, "match": match})
        return {"status": identity.get("status"), "shards": identity.get("shards", []),
                "comparisons": comparisons, "full_weights_bound_to_historical_runs": False,
                "historical_binding_limit": "historical run manifests hash metadata/index, not all weight shards; current full-shard identity does not prove historical immutability"}

    def run(self, hash_images=True):
        try:
            runs = self.runs()
            data = self.splits_and_images(hash_images)
            gains = self.gain_tables(data)
            summaries = self.summaries()
            model = self.model_identity()
            # Sets are internal only; the final evidence is JSON portable.
            for item in data["manifests"].values():
                for key in ("ids", "images", "source_image_ids"):
                    item[key] = sorted(item[key])
            counts = dict(Counter(r["level"] for r in self.findings))
            return {"schema": "trajkv.delivery_validation.v1", "status": "VALIDATED_WITH_FINDINGS" if self.findings else "PASS_DECLARED_CHECKS",
                    "checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    "root": str(self.root), "read_only_audit": True, "gpu_used": False,
                    "research_should_continue": True, "unfinished_runs_are_not_failures": True,
                    "findings_by_level": counts, "findings": self.findings, "runs": runs,
                    "artifact_count_by_kind": dict(Counter(r["artifact_kind"] for r in runs)),
                    "data": data, "gain_tables": gains, "summaries": summaries, "model_identity": model,
                    "validator_sha256": self.sha(Path(__file__))}
        finally:
            if self.sftp:
                self.sftp.close()
            if self.client:
                self.client.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", type=Path)
    p.add_argument("--remote-sources", action="store_true", help="Read missing historical source snapshots over the authorized SSH route")
    p.add_argument("--skip-image-hashes", action="store_true")
    p.add_argument("--strict-integrity", action="store_true", help="Nonzero exit only for proven integrity ERROR findings; PENDING does not fail")
    args = p.parse_args(argv)
    report = Audit(args.root, args.remote_sources).run(not args.skip_image_hashes)
    output = args.output or args.root / "audit/delivery_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "summaries": len(report["summaries"]), "findings": report["findings_by_level"],
                      "output": str(output)}))
    return 1 if args.strict_integrity and report["findings_by_level"].get("ERROR", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
