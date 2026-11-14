"""Evidence-integrity edge cases. No SSH, model, or GPU is used."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_delivery import Audit


class DeliveryChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "TrajKV"
        for name in ("results", "configs/data", "audit/data", "code"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_unfinished_jsonl_and_run_are_pending_not_failed(self):
        run = self.root / "results/running"
        self.write_json("results/running/manifest.json", {
            "arguments": {"methods": ["full"]}, "selected_ids": ["a", "b"]})
        (run / "records.jsonl").write_text('{"session_id":"a","method":"full"}\n{"session_id":')
        audit = Audit(self.root)
        runs = audit.runs()
        self.assertTrue(runs[0]["status"].startswith("PENDING"))
        self.assertFalse(any(f["level"] == "ERROR" for f in audit.findings))
        self.assertIn("UNFINISHED_LAST_JSONL_ROW", {f["code"] for f in audit.findings})

    def test_replay_base_and_text_cost_are_counted_and_overrun_retained(self):
        record = {"session_id": "s", "method": "asym", "policy": "repair_once",
                  "extra_logical_read_bytes": 37, "read_budget_bytes": 30, "read_budget_ok": False,
                  "read_events": [{"pass_kind": "committed", "residual_bytes": 5},
                                  {"pass_kind": "repair", "residual_bytes": 3,
                                   "base_visual_bytes": 20, "text_bytes": 9}]}
        audit = Audit(self.root)
        result = audit.budgets(self.root / "records.jsonl", [record])
        self.assertEqual(result["violations"][0]["extra_bytes"], 37)
        self.assertTrue(result["violations"][0]["preserve_original_record"])
        self.assertTrue(result["violations"][0]["exclude_from_equal_budget_claims"])
        self.assertFalse(result["ledger_inconsistencies"])
        self.assertEqual(audit.findings[0]["level"], "EXCLUSION")

    def test_historical_archive_wins_over_changed_current_source(self):
        run = self.root / "results/old"
        archive = run / "source_snapshot/runner.py"
        archive.parent.mkdir(parents=True)
        archive.write_text("old version\n")
        (self.root / "code/runner.py").write_text("new version\n")
        expected = hashlib.sha256(archive.read_bytes()).hexdigest()
        audit = Audit(self.root)
        result = audit.sources(run, {"sources": {"code/runner.py": expected}})
        self.assertTrue(result[0]["match"])
        self.assertEqual(result[0]["status"], "ARCHIVED_LOCAL")
        self.assertFalse(audit.findings)

    def test_split_leakage_is_detected_by_bytes_not_different_image_names(self):
        a, b = self.root / "a.png", self.root / "b.png"
        a.write_bytes(b"identical image bytes")
        b.write_bytes(a.read_bytes())
        h = hashlib.sha256(a.read_bytes()).hexdigest()
        self.write_json("configs/data/dev_runner.json", [
            {"id": "dev", "source_image_id": "named-a", "images": [str(a)], "hashes": [h]}])
        evaluation = self.write_json("configs/data/final_eval64_runner.json", [
            {"id": "eval", "source_image_id": "named-b", "images": [str(b)], "hashes": [h]}])
        self.write_json("audit/data/frozen_evaluation.json", {"datasets": {
            "final_eval64_runner": {"sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
                                    "ids": ["eval"]}}})
        audit = Audit(self.root)
        result = audit.splits_and_images()
        self.assertEqual(result["frozen_checks"][0]["dev_image_overlap"], 1)
        self.assertIn("FROZEN_EVAL_DEV_IMAGE_OVERLAP", {f["code"] for f in audit.findings})

    def test_gain_table_alias_session_cannot_hide_same_source_image(self):
        self.write_json("configs/frozen_gain_table.json", {
            "training_session_ids": ["different-training-id"],
            "training_image_ids": {"different-training-id": "shared-source-image"}})
        data = {"frozen_evaluation_manifests": ["final"], "manifests": {
            "final": {"ids": {"evaluation-id"}, "source_image_ids": {"shared-source-image"},
                      "images": set()}}}
        audit = Audit(self.root)
        audit.gain_tables(data)
        self.assertIn("GAIN_TABLE_TRAIN_EVAL_LEAKAGE", {f["code"] for f in audit.findings})

    def test_identical_archive_in_another_run_can_resolve_missing_source(self):
        run, other = self.root / "results/run", self.root / "results/other"
        self.write_json("results/other/manifest.json", {})
        archive = other / "source_snapshot/runner.py"
        archive.parent.mkdir()
        archive.write_text("recoverable old source\n")
        expected = hashlib.sha256(archive.read_bytes()).hexdigest()
        audit = Audit(self.root)
        result = audit.sources(run, {"sources": {"code/runner.py": expected}})
        self.assertTrue(result[0]["match"])
        self.assertEqual(result[0]["status"], "ARCHIVED_OTHER_RUN_IDENTICAL_CONTENT")
        self.assertEqual(result[0]["observed_source_location"], str(archive))
        self.assertFalse((run / "source_snapshot").exists())

    def test_explicit_recovered_object_is_verified_without_rewriting_manifest(self):
        source = b"exact historical bytes\n"
        expected = hashlib.sha256(source).hexdigest()
        obj = self.root / "audit/recovered-by-content-hash" / expected / "bank.py"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(source)
        run = self.root / "results/run"
        manifest = self.write_json("results/run/manifest.json", {"sources": {"code/bank.py": expected}})
        before = manifest.read_bytes()
        audit = Audit(self.root)
        result = audit.sources(run, audit.json(manifest))
        self.assertTrue(result[0]["match"])
        self.assertEqual(result[0]["status"], "RECOVERED_BY_EXACT_CONTENT_HASH")
        self.assertEqual(before, manifest.read_bytes())
        self.assertFalse((run / "source_snapshot").exists())

    def test_stale_partial_summary_is_preserved_and_warned(self):
        raw = self.root / "results/run/records.jsonl"
        raw.parent.mkdir(parents=True)
        raw.write_text('{"session_id":"new"}\n')
        self.write_json("results/analysis/summary.json", {
            "n_records": 1, "input_sha256": {str(raw): "0" * 64}, "status": "PARTIAL"})
        audit = Audit(self.root)
        audit.summaries()
        self.assertTrue(raw.exists())
        self.assertIn("STALE_OR_UNAVAILABLE_ANALYSIS_INPUT_SNAPSHOT", {f["code"] for f in audit.findings})
        self.assertFalse(any(f["level"] == "ERROR" for f in audit.findings))

    def test_fresh_summary_cannot_claim_wrong_raw_record_count(self):
        raw = self.root / "results/run/records.jsonl"
        raw.parent.mkdir(parents=True)
        raw.write_text('{"session_id":"a"}\n{"session_id":"b"}\n')
        self.write_json("results/analysis/summary.json", {
            "n_records": 1, "input_sha256": {str(raw): hashlib.sha256(raw.read_bytes()).hexdigest()}})
        audit = Audit(self.root)
        audit.summaries()
        self.assertIn("SUMMARY_RAW_RECORD_COUNT_MISMATCH", {f["code"] for f in audit.findings})

    def test_exact_recovered_partial_input_removes_stale_warning_not_partial_history(self):
        raw = self.root / "results/run/records.jsonl"
        raw.parent.mkdir(parents=True)
        first = b'{"session_id":"a"}\n'
        raw.write_bytes(first + b'{"session_id":"b"}\n')
        expected = hashlib.sha256(first).hexdigest()
        summary = self.write_json("results/analysis/summary.json", {
            "n_records": 1, "partial": True, "status": "PARTIAL", "input_sha256": {str(raw): expected}})
        before = summary.read_bytes()
        recovered = self.root / "results/analysis/recovered_inputs/original19.jsonl"
        recovered.parent.mkdir()
        recovered.write_bytes(first)
        self.write_json("results/analysis/recovered_inputs/recovery_receipt.json", {
            "status": "RECOVERED_EXACT_ORIGINAL_INPUT_SNAPSHOT", "recorded_input_sha256": expected,
            "snapshot_sha256": expected, "snapshot": str(recovered)})
        audit = Audit(self.root)
        report = audit.summaries()[0]
        self.assertFalse(report["inputs_match_current_files"])
        self.assertTrue(report["declared_original_inputs_verified"])
        self.assertFalse(audit.findings)
        self.assertEqual(before, summary.read_bytes())

    def test_composite_parent_and_nested_child_are_different_artifacts(self):
        parent = self.root / "results/suite"
        child = parent / "docvqa"
        receipt = self.write_json("results/suite/docvqa/complete.json", {"records_added": 1})
        self.write_json("results/suite/docvqa/manifest.json", {
            "arguments": {"methods": ["full"]}, "selected_ids": ["s"]})
        (child / "records.jsonl").write_text('{"session_id":"s","method":"full","policy":"base"}\n')
        plan = self.write_json("results/suite/suite_manifest.json", {
            "plan": {"schema": "trajkv.fixed_suite.v1", "runs": [{"name": "docvqa", "output": str(child)}],
                     "source_hashes": {}, "input_hashes": {}}})
        self.write_json("results/suite/complete.json", {
            "suite_manifest_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
            "runs": {"docvqa": {"complete": {"sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}}}})
        audit = Audit(self.root)
        runs = audit.runs()
        self.assertEqual(Counter(r["artifact_kind"] for r in runs),
                         Counter({"composite_suite": 1, "model_or_mechanism_run": 1}))
        self.assertFalse(any(f["level"] == "ERROR" for f in audit.findings))
        self.assertEqual(next(r for r in runs if r["artifact_kind"] == "composite_suite")["model_inference_records"], None)

    def test_budget_sweep_preserves_child_budget_contract(self):
        child = self.root / "results/sweep/budget_4MiB"
        receipt = self.write_json("results/sweep/budget_4MiB/complete.json", {"records_added": 1})
        self.write_json("results/sweep/budget_4MiB/manifest.json", {
            "selected_ids": ["s"], "arguments": {"methods": ["asym"], "policies": ["periodic"],
                                                    "read_budget_bytes": 8 * 2**20}})
        plan = self.write_json("results/sweep/sweep_manifest.json", {"plan": {
            "schema": "trajkv.budget_sweep.v1", "method": "asym", "policies": ["periodic"],
            "selected": [{"session_id": "s"}], "sources": {},
            "children": [{"budget_mib": 4, "read_budget_bytes": 4 * 2**20,
                          "expected_records": 1, "output": str(child)}]}})
        self.write_json("results/sweep/complete.json", {
            "sweep_manifest_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(), "records": 1,
            "children": [{"budget_mib": 4, "artifacts": {"complete.json": hashlib.sha256(receipt.read_bytes()).hexdigest()}}]})
        audit = Audit(self.root)
        audit.container(plan.parent, plan)
        self.assertIn("BUDGET_SWEEP_CHILD_CONTRACT_CHANGED", {f["code"] for f in audit.findings})

    def test_teacher_forced_timing_counts_trials_without_quality_scoring(self):
        run = self.root / "results/timing"
        archived = run / "source_snapshot/00_timing_control.py"
        archived.parent.mkdir(parents=True)
        archived.write_bytes(b"pinned timing source\n")
        expected_hash = hashlib.sha256(archived.read_bytes()).hexdigest()
        self.write_json("results/timing/manifest.json", {"quality": "not_scored", "arguments": {},
            "contract": {"selected_ids": ["s"], "configurations": [{"configuration": "full:base"}],
                         "repeats": 1, "sources": {"code/timing_control.py": expected_hash}}})
        self.write_json("results/timing/complete.json", {"measured_trials": 1, "expected_measurements": 1,
                                                         "quality": "not_scored"})
        record = {"schema": "trajkv.timing_control.v1", "session_id": "s", "configuration": "full:base",
                  "repeat": 0, "quality": "not_scored", "application_accuracy": None, "quiet_verified": True,
                  "token_identity": {"output_token_ids_identical": True},
                  "runner_result": {"session_id": "s", "method": "full", "policy": "base",
                                    "read_events": [], "extra_logical_read_bytes": 0, "read_budget_ok": True}}
        (run / "records.jsonl").write_text(json.dumps(record) + "\n")
        self.write_json("results/timing/summary.json", {"schema": "trajkv.timing_control_summary.v1",
            "quality": "not_scored", "application_accuracy": None,
            "configurations": [{"configuration": "full:base", "measured_trials": 1}]})
        audit = Audit(self.root)
        reported = audit.runs()[0]
        summaries = audit.summaries()
        self.assertEqual(reported["artifact_kind"], "teacher_forced_timing")
        self.assertEqual(reported["measurement_trials"], 1)
        self.assertEqual(reported["application_quality_records"], 0)
        self.assertEqual(reported["sources"][0]["status"], "ARCHIVED_LOCAL")
        self.assertTrue(summaries[0]["timing_is_not_zero_accuracy"])
        self.assertFalse(any(f["level"] == "ERROR" for f in audit.findings))

    def test_failed_hardware_counter_capture_is_not_zero_model_inference(self):
        self.write_json("results/hardware/manifest.json", {"kind": "optional_hardware_counter_side_check", "sources": {}})
        self.write_json("results/hardware/complete.json", {
            "status": "HARDWARE_COUNTER_UNAVAILABLE_CUDA_INIT_ERROR36", "captured_kernels": 0, "records_added": 0,
            "physical_dram_bytes": None})
        audit = Audit(self.root)
        report = audit.runs()[0]
        self.assertEqual(report["artifact_kind"], "hardware_counter_probe")
        self.assertIsNone(report["model_inference_records"])
        self.assertIsNone(report["application_quality_records"])
        self.assertTrue(report["not_a_zero_quality_model_run"])
        self.assertFalse(any(f["code"].startswith("MISSING_EXPECTED") for f in audit.findings))

    def test_legacy_composite_input_accepts_only_exact_retained_bytes(self):
        data = b"immutable legacy kernel\n"
        expected = hashlib.sha256(data).hexdigest()
        obj = self.root / "audit/recovered-by-content-hash" / expected / "p202_fused.py"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(data)
        audit = Audit(self.root)
        actual, origin = audit.declared_input_hash("/remote/VisContract/scripts/p202_fused.py", expected)
        self.assertEqual(actual, expected)
        self.assertEqual(origin, str(obj))

    def test_reference_turn_answer_csv_is_joined_to_exact_raw_source(self):
        raw = self.root / "results/run/records.jsonl"
        raw.parent.mkdir(parents=True)
        row = {"session_id": "bridge", "method": "asym", "policy": "probe", "reference_turn_offset": 1,
               "turns": [{"turn": 2, "answer": "red"}]}
        raw.write_text(json.dumps(row) + "\n")
        self.write_json("results/analysis/summary.json", {"n_records": 1, "n_scored_turns": 1,
            "input_sha256": {str(raw): hashlib.sha256(raw.read_bytes()).hexdigest()}})
        csv_path = self.root / "results/analysis/scored_turns.csv"
        csv_path.write_text('session_id,method,policy,reference_turn,answer,source_records\n'
                            + f'bridge,asym,probe,1,red,{raw}\n')
        audit = Audit(self.root)
        audit.summaries()
        self.assertFalse(audit.findings)
        csv_path.write_text(csv_path.read_text().replace(',red,', ',blue,'))
        changed = Audit(self.root)
        changed.summaries()
        self.assertIn("SCORED_ROWS_RAW_PREDICTION_MISMATCH", {f["code"] for f in changed.findings})

    def test_inventory_index_finds_historical_source_and_model_json(self):
        import collect_run_inventory as inventory
        old = self.root / "results/old/source_snapshot/bank.py"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"historical exact source\n")
        (self.root / "code/bank.py").write_bytes(b"new source\n")
        expected = hashlib.sha256(old.read_bytes()).hexdigest()
        metadata = b'{"model":"fixed"}\n'
        model_hash = hashlib.sha256(metadata).hexdigest()
        obj = self.root / "audit/recovered-by-content-hash" / model_hash / "config.json"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(metadata)
        retained = inventory.retained_source_index(self.root)
        checks = inventory.source_checks({"manifest.json": {"sources": {
            "code/bank.py": expected, "/remote/models/config.json": model_hash}}},
            self.root / "results/current", self.root, retained)
        self.assertTrue(checks["all_declared_files_verified"])
        self.assertEqual(checks["status_counts"], {"VERIFIED_AVAILABLE_COPY": 2})


if __name__ == "__main__":
    unittest.main()
