"""Pinned replacement scope only: Calib24/P2 and Calib4/P3, no execution.

Unchanged90 P2 and native/other16 P3 targets retain their exact original config
identity. New Calib cells can only come from their independent v2 state.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from public_phase_contract_v2 import ROOT, EVIDENCE, load, sha, require, resolve
import prepare_remaining_phase_delivery_v1 as legacy

P2 = ROOT / "configs/matrix/P2_complete114_Calib_FP32Affine_Text32_v2.json"
P2_SHA = "fb816d1680734c32ce1474eb8dcc89a860f703ca772783d371311e785f801d5a"
P3 = ROOT / "configs/matrix/P3_M1_representative_targets_Calib_FP32Affine_Text32_v4.json"
P3_SHA = "e30793e00388aa1e882f70e3411ed25ab234eaab89b3d055d35dea5c86255cb8"
LEGACY_GATE_SHA = "621b9a3bf29811e3b11292397c1248b19046154a8924dee87cd2fa1ae10c0650"
CALIB_STATE = EVIDENCE / "results/extension_execution/calib_numeric_v2/harvest/state.json"
FULL_STATE = EVIDENCE / "results/full_execution/m1_representative_numeric_v4/harvest/state.json"
FULL_LEDGER_DIR = EVIDENCE / "results/full_execution/m1_representative_numeric_v4"
FULL_PREPARATION = EVIDENCE / "results/full_preparation/m1_representative_numeric_v4"
SPEED_RUN = "m1_canonical_source_representatives_numeric_v3"
SPEED_LEDGER_DIR = EVIDENCE / "results/timing_execution" / SPEED_RUN
CALIB_CLASS = ("methods.calibquant_fp32_text_adapter", "CalibQuantFP32AffineTextAdapter")
ADAPTER_SHA = "d2b1c26caee553622ba159ffb2df8748b6d1d02731103cd109bfc121a0d43075"


def key(row): return row["model"], row["method"], row["dataset"]
def config_path(row): return row.get("full_target_config") or row.get("config") or row.get("target_config")
def qa_count(row): return int(row.get("count", row.get("target_QA", row.get("expected_QA", 0))))
def is_calib(row): return load(resolve(config_path(row)))["method"].get("family") == "calibquant"


def frozen_matrix(which):
    require(which in ("p2", "p3"), "Unsupported numeric scope")
    path, digest, n = (P2, P2_SHA, 114) if which == "p2" else (P3, P3_SHA, 20)
    require(sha(path) == digest, "Pinned numeric-v2 target matrix changed")
    require(sha(ROOT / "ops/prepare_remaining_phase_delivery_v1.py") == LEGACY_GATE_SHA, "Original gate changed")
    require(sha(ROOT / "code/methods/calibquant_fp32_text_adapter.py") == ADAPTER_SHA, "Frozen numerical adapter changed")
    matrix = load(path); rows = matrix["rows"]
    require(len(rows) == n and len({key(r) for r in rows}) == n, "Target scope count/uniqueness differs")
    require(sum(qa_count(r) for r in rows) == (57000 if which == "p2" else 78250), "Target QA scope differs")
    changed = 0
    for row in rows:
        path = resolve(config_path(row)); cfg = load(path)
        if row.get("config_sha256"): require(sha(path) == row["config_sha256"], "Target config changed")
        require(cfg["model_id"] == row["model"], "Target model differs")
        if cfg["method"].get("family") == "calibquant":
            changed += 1; method = cfg["method"]
            require((method["adapter_module"], method["adapter_class"]) == CALIB_CLASS, "Legacy Calib class cannot enter numeric-v2 scope")
            bits = 1 if row["method"] == "C1_bias32" else 2
            require(row["method"] in ("C1_bias32", "C2_bias32") and method["id"] == f"CalibQuant_visual_FP32Affine_Text32_v2_{bits}bit", "Logical tier/numeric implementation differs")
            require(method["constructor_args"] == {"bits": bits, "reader_backend": "upstream_triton"}, "Source bit/backend parameters changed")
    require(changed == (24 if which == "p2" else 4), "Replacement-cell count differs")
    if which == "p2":
        for group in matrix["groups"]: require(sha(resolve(group["new"])) == group["new_sha256"], "Numeric partition declaration changed")
    return matrix


def quality_matrix(which):
    frozen = frozen_matrix(which); previous = legacy.quality_matrix(which)
    old = {key(r): r for r in previous["rows"]}; rows = []
    cells = legacy.state(CALIB_STATE if which == "p2" else FULL_STATE)["cells"]
    native = legacy.state(EVIDENCE / "results/harvest/native_F_v1/state.json").get("full_cells", {})
    for row in frozen["rows"]:
        original = old[key(row)]; changed = is_calib(row)
        cfg = config_path(row); stem = Path(cfg).stem
        if which == "p2" and not changed:
            require(cfg == original["config"] and sha(resolve(cfg)) == original["config_sha256"], "Unchanged90 target was altered")
            item = dict(original)
        elif which == "p2":
            item = legacy.target(row, original["partition"], cells.get(stem),
                EVIDENCE / "results/extension_views/calib_numeric_v2", EVIDENCE / "results/scored/extension_calib_numeric_v2")
        else:
            if not changed: require(cfg == original["config"] and sha(resolve(cfg)) == original["config_sha256"], "Unchanged full target was altered")
            cell = native.get(row["dataset"], {}) if row["method"] == "F" else cells.get(stem, {})
            item = legacy.target(row, "existing_native4" if row["method"] == "F" else "canonical_numeric_compressed16", cell,
                EVIDENCE / "results/representative_full_views/m1_representative_numeric_v4",
                EVIDENCE / "results/scored/representative_full_m1_representative_numeric_v4")
        item.update(numeric_scope="Calib_FP32Affine_Text32_v2", numerical_replacement=changed,
                    actual_method_id=load(resolve(cfg))["method"]["id"])
        rows.append(item)
    declarations = [{"path": str(P2 if which == "p2" else P3), "sha256": P2_SHA if which == "p2" else P3_SHA},
                    {"path": str(Path(__file__)), "sha256": sha(__file__)}]
    if which == "p2": declarations += [{"path": str(resolve(g["new"])), "sha256": g["new_sha256"]} for g in frozen["groups"]]
    return {"schema": "baseline_matrix.fixed_matrix.v1", "phase": "P2_NUMERIC_V2_COMPLETE114_FIXED500" if which == "p2" else "P3_NUMERIC_V4_M1_ALL20_FULL",
            "rows": rows, "expected_cells": len(rows), "expected_QA": sum(r["count"] for r in rows), "metrics_not_mixed": True,
            "source_declarations": declarations,
            "scope": "Complete fixed numeric-implementation scope only; legacy Calib predictions excluded; no new GPU count from views; speed remains a separate new-v2 measurement stage.",
            "new_canonical_speed_namespace": SPEED_RUN, "old_canonical560_not_reused_as_new_numeric_speed": True}


def canonical_speed_readiness():
    # This provider never infers actual timing success from ledger declarations.
    # A future dedicated timing validator must verify all four new raw/source
    # nodes. Merely having the historical560 or planned new560 is insufficient.
    return {"status": "WAIT_NEW_NUMERIC_V2_CANONICAL560_ACTUAL_EVIDENCE", "required_namespace": SPEED_RUN,
            "ledger_directory": str(SPEED_LEDGER_DIR), "expected_trials": 560, "warmup_trials": 140,
            "measured_trials": 420, "old_numeric_C2_speed_admitted": False,
            "scope_provider_is_not_a_timing_validator": True}
