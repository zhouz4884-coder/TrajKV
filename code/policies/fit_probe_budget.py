"""Fit fixed residual-probe thresholds from no-upgrade development traces.

This is a read-budget calibration, not a reward/accuracy optimizer. The only
event features used are online enhanced-vs-base delta and planned read bytes.
Answers, task references, reference logits and future outcomes are never used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "trajkv.probe_budget_calibration.v1"
MIB = 1 << 20
DEFAULT_BUDGETS = (4*MIB, 8*MIB, 16*MIB, 32*MIB)
POLICY_PATTERN = re.compile(r"^probe_l([0-9]+)_t(.+)$")
READ_FIELDS = ("base_visual_bytes", "text_bytes", "residual_bytes")


def _bytes(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer byte count")
    return value


def _delta(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("Online relative_l2 must be finite and nonnegative")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Online relative_l2 must be finite and nonnegative")
    return value


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def literal_policy(layer: int, threshold: float) -> str:
    """Round-trips through the frozen runner's float(parts[2][1:]) parser."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Policy threshold must be finite and nonnegative")
    name = f"probe_l{int(layer)}_t{format(threshold, '.17g')}"
    assert float(name.split("_")[2][1:]) == threshold
    return name


def threshold_candidates(deltas: list[float]) -> list[float]:
    """For delta >= eta, an event drops out immediately ABOVE its delta.

    Equal deltas are inseparable at a fixed threshold. Zero is the least legal
    threshold, so a zero-delta event with nonzero planned cost is included there.
    """
    candidates = {0.0}
    for value in deltas:
        boundary = math.nextafter(_delta(value), math.inf)
        if math.isfinite(boundary):
            candidates.add(boundary)
    return sorted(candidates)


def _generation_settings(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        return {}, {}
    raw = manifest.read_bytes()
    arguments = json.loads(raw.decode("utf-8-sig")).get("arguments", {})
    keys = ("max_tokens", "turns", "max_patches", "description_tokens", "hide_visual_after_description", "seed")
    return ({key: arguments[key] for key in keys if key in arguments},
            {str(manifest.resolve()): hashlib.sha256(raw).hexdigest()})


def load_calibration(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Project each record onto online signal/cost/geometry fields only.

    Generation answer strings can be present in the source JSONL, but are not
    accessed, copied, scored, or used by this loader. No reference file is opened.
    """
    sessions = []
    sources = {}
    seen = set()
    for path in paths:
        raw = path.read_bytes()
        sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        settings, manifest_hash = _generation_settings(path)
        sources.update(manifest_hash)
        for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("session_id", ""))
            image = str(row.get("image_id", ""))
            if not sid or not image:
                raise ValueError(f"{path}:{line_number}: session_id and image_id are required")
            if row.get("split") not in ("development", "calibration", "train"):
                raise ValueError(f"{sid}: only development/calibration/train traces may fit thresholds")
            if row.get("protocol") != "free_running":
                raise ValueError(f"{sid}: expected the unmodified free-running no-upgrade calibration trajectory")
            if "read_budget_bytes" not in row or row["read_budget_bytes"] is not None:
                raise ValueError(f"{sid}: calibration requires explicitly unbounded read_budget_bytes=None")
            if row.get("reference_answers_visible_to_runner") is True:
                raise ValueError(f"{sid}: answer-visible generation cannot fit this online-only calibration")
            policy = str(row.get("policy", ""))
            match = POLICY_PATTERN.fullmatch(policy)
            if not match:
                raise ValueError(f"{sid}: expected a literal probe_lN_tTHRESHOLD calibration policy")
            layer = int(match.group(1))
            calibration_threshold = _delta(match.group(2))
            if row.get("intra_step_probe_skipped"):
                raise ValueError(f"{sid}: skipped probes make the all-probe floor incomplete")
            owned = row.get("owned_initial", {})
            if owned.get("diagnostic_fp16_reference", 0):
                raise ValueError(f"{sid}: reference owners are not part of this online-only calibration")
            denominator = owned.get("visual_fp16_denominator")
            if not denominator or denominator <= 0:
                raise ValueError(f"{sid}: a positive visual FP16 byte denominator is required")
            events = []
            shapes, dtypes = set(), set()
            event_ids = set()
            for event in row.get("intra_step_probes", []):
                decision = event.get("decision")
                if not event.get("probe_observed") or not isinstance(decision, dict):
                    raise ValueError(f"{sid}: every calibration event must contain an observed probe decision")
                if decision.get("upgrade") or decision.get("use_enhanced_probe"):
                    raise ValueError(f"{sid}: calibration contains upgrades; future deltas/costs are no longer a no-upgrade trace")
                if int(event.get("probe_layer", -1)) != layer:
                    raise ValueError(f"{sid}: literal policy layer and observed probe layer differ")
                availability = str(decision.get("feature_availability", event.get("feature_availability", "")))
                if not availability.lower().startswith("online"):
                    raise ValueError(f"{sid}: feature availability must explicitly be online")
                if decision.get("finite_signal") is False:
                    raise ValueError(f"{sid}: nonfinite probe signal cannot fit a fixed scalar threshold")
                delta = _delta(decision.get("relative_l2"))
                if delta >= calibration_threshold or decision.get("threshold_met"):
                    raise ValueError(f"{sid}: calibration threshold was met despite the claimed no-upgrade run")
                cost = _bytes(decision.get("planned_upgrade_bytes"), "planned_upgrade_bytes")
                event_id = event.get("event_id")
                if event_id in event_ids:
                    raise ValueError(f"{sid}: duplicate probe event_id {event_id!r}")
                event_ids.add(event_id)
                shape = decision.get("shape")
                if shape:
                    if len(shape) != 4:
                        raise ValueError(f"{sid}: expected probe shape [batch,heads,query_tokens,head_dim]")
                    shapes.add((int(shape[0]), int(shape[1]), int(shape[3])))
                if decision.get("input_dtype"):
                    dtypes.add(str(decision["input_dtype"]))
                events.append(dict(event_id=event_id, turn=event.get("turn"), phase=event.get("phase"),
                    input_count=event.get("input_count"), relative_l2=delta, planned_upgrade_bytes=cost,
                    feature_availability="online"))
            if not events:
                raise ValueError(f"{sid}: no observed calibration probes")
            if len(shapes) > 1 or len(dtypes) > 1:
                raise ValueError(f"{sid}: mixed probe head geometry or dtype inside one session")

            probe_floor = 0
            observed_probe_calls = 0
            committed_residual = 0
            raw_extra = 0
            layers = set()
            for event in row.get("read_events", []):
                kind = event.get("pass_kind")
                if "layer" in event:
                    layers.add(int(event["layer"]))
                values = {key: _bytes(event.get(key, 0), key) for key in READ_FIELDS}
                if event.get("reference_bytes", 0):
                    raise ValueError(f"{sid}: reference read cost is forbidden in online-only calibration")
                if kind == "probe":
                    probe_floor += sum(values.values())
                    observed_probe_calls += _bytes(event.get("calls", 0), "probe calls")
                elif kind == "committed":
                    committed_residual += values["residual_bytes"]
                elif kind in ("replay", "repair") and any(values.values()):
                    raise ValueError(f"{sid}: a no-upgrade calibration trace cannot contain replay/repair work")
                raw_extra += values["residual_bytes"] + (values["base_visual_bytes"] + values["text_bytes"]
                                                           if kind in ("probe", "replay", "repair") else 0)
            if committed_residual:
                raise ValueError(f"{sid}: no-upgrade calibration contains committed residual reads")
            if observed_probe_calls != len(events):
                raise ValueError(f"{sid}: observed event count differs from actual extra probe read calls")
            if raw_extra != probe_floor:
                raise ValueError(f"{sid}: raw extra ledger includes work beyond the fixed probe floor")
            if row.get("extra_logical_read_bytes") is not None and row["extra_logical_read_bytes"] != probe_floor:
                raise ValueError(f"{sid}: reported extra ledger differs from actual probe base+text+residual reads")
            if row.get("read_ledger_exact") is False:
                raise ValueError(f"{sid}: runtime marked the read ledger inexact")
            geometry = dict(method=row.get("method"), model=row.get("model"),
                data_protocol=row.get("data_protocol"), generation_settings=settings,
                visual_tokens=row.get("visual_tokens"), layer_count=len(layers),
                probe_head_geometry=list(next(iter(shapes))) if shapes else None,
                input_dtype=next(iter(dtypes)) if dtypes else None,
                stored_visual_base_ratio=round(float(owned.get("visual_base", 0))/denominator, 10),
                stored_visual_residual_ratio=round(float(owned.get("visual_residual", 0))/denominator, 10),
                runtime=owned.get("runtime"), kernel_variant=row.get("kernel_variant"),
                kernel_source_sha256=row.get("kernel_source_sha256"))
            group_id = _digest_json(dict(geometry=geometry, probe_layer=layer))
            identity = (group_id, sid)
            if identity in seen:
                raise ValueError(f"Duplicate calibration session for geometry/layer: {identity}")
            seen.add(identity)
            sessions.append(dict(group_id=group_id, geometry=geometry, method=row.get("method"),
                probe_layer=layer, session_id=sid, image_id=image, split=row["split"],
                source_records=str(path.resolve()), calibration_policy=policy,
                fixed_probe_extra_bytes=probe_floor, events=events,
                observed_probe_calls=observed_probe_calls, actual_extra_ledger_matches=True))
    return sessions, sources


def _predict(session: dict[str, Any], threshold: float) -> dict[str, Any]:
    selected = [event for event in session["events"] if event["relative_l2"] >= threshold]
    upgrade = sum(event["planned_upgrade_bytes"] for event in selected)
    floor = _bytes(session["fixed_probe_extra_bytes"], "fixed_probe_extra_bytes")
    return dict(session_id=session["session_id"], image_id=session["image_id"],
        fixed_probe_extra_bytes=floor, predicted_upgrade_residual_bytes=upgrade,
        predicted_extra_bytes=floor+upgrade, triggered_events=len(selected), observed_events=len(session["events"]))


def fit_group(sessions: list[dict[str, Any]], *, probe_layer: int,
              budgets_bytes: tuple[int, ...] = DEFAULT_BUDGETS,
              safety_fraction: float = .9) -> dict[str, Any]:
    if not sessions:
        raise ValueError("Cannot fit a threshold without real calibration sessions")
    if not math.isfinite(safety_fraction) or not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be finite in (0,1]")
    for budget in budgets_bytes:
        if _bytes(budget, "budget_bytes") == 0:
            raise ValueError("budget_bytes must be positive")
    candidates = threshold_candidates([event["relative_l2"] for session in sessions for event in session["events"]])
    n = len(sessions)
    floor_sum = sum(_bytes(session["fixed_probe_extra_bytes"], "fixed_probe_extra_bytes") for session in sessions)
    cap = {budget: Decimal(str(safety_fraction))*budget for budget in budgets_bytes}
    scan = []
    predictions_by_candidate = []
    for threshold in candidates:
        predicted = [_predict(session, threshold) for session in sessions]
        total = sum(row["predicted_extra_bytes"] for row in predicted)
        row = dict(threshold=threshold, literal_policy_name=literal_policy(probe_layer, threshold),
            total_predicted_extra_bytes=total, mean_predicted_extra_bytes=total/n,
            mean_fixed_probe_extra_bytes=floor_sum/n,
            mean_predicted_upgrade_residual_bytes=(total-floor_sum)/n,
            maximum_predicted_session_extra_bytes=max(r["predicted_extra_bytes"] for r in predicted),
            triggered_events=sum(r["triggered_events"] for r in predicted),
            meets_mean_cap_by_budget_bytes={str(budget): Decimal(total) <= cap[budget]*n for budget in budgets_bytes})
        scan.append(row)
        predictions_by_candidate.append(predicted)
    budgets = []
    for budget in budgets_bytes:
        base = dict(budget_bytes=budget, budget_mib=budget/MIB,
            mean_prediction_cap_bytes=float(cap[budget]), safety_fraction=safety_fraction,
            mean_fixed_probe_extra_bytes=floor_sum/n, deployment_hard_cap_required=True)
        if Decimal(floor_sum) > cap[budget]*n:
            budgets.append(dict(base, status="INFEASIBLE_PROBE_FLOOR", threshold=None, literal_policy_name=None,
                reason="All observed probes alone exceed the mean safety-adjusted budget; thresholding upgrades cannot remove this floor"))
            continue
        chosen = next((i for i, row in enumerate(scan) if row["meets_mean_cap_by_budget_bytes"][str(budget)]), None)
        if chosen is None:
            budgets.append(dict(base, status="INFEASIBLE_FINITE_THRESHOLD", threshold=None, literal_policy_name=None,
                reason="No finite nonnegative threshold in the observed float domain meets the mean cap"))
            continue
        row, predicted = scan[chosen], predictions_by_candidate[chosen]
        by_image = defaultdict(list)
        for item in predicted:
            by_image[item["image_id"]].append(item["predicted_extra_bytes"])
        budgets.append(dict(base, status="FEASIBLE_MEAN_PREDICTION", threshold=row["threshold"],
            literal_policy_name=row["literal_policy_name"], chosen_scan_index=chosen,
            predicted_mean_extra_bytes=row["mean_predicted_extra_bytes"],
            predicted_mean_upgrade_residual_bytes=row["mean_predicted_upgrade_residual_bytes"],
            maximum_predicted_session_extra_bytes=row["maximum_predicted_session_extra_bytes"],
            predicted_sessions_over_hard_cap=sum(item["predicted_extra_bytes"] > budget for item in predicted),
            per_session_predictions=predicted,
            per_image_predictions=[dict(image_id=image, sessions=len(values), mean_predicted_extra_bytes=sum(values)/len(values))
                                   for image, values in sorted(by_image.items())]))
    return dict(probe_layer=probe_layer, training_session_ids=sorted(s["session_id"] for s in sessions),
        training_image_ids=sorted({s["image_id"] for s in sessions}), n_sessions=n,
        mean_fixed_probe_extra_bytes=floor_sum/n, threshold_scan=scan, budgets=budgets,
        calibration_sessions=sessions, mean_weighting="equal development sessions; images retained for audit",
        prediction="all observed probe extra base+text+residual reads plus selected planned downstream residual reads")


def fit_probe_budgets(paths: list[Path], *, budgets_bytes: tuple[int, ...] = DEFAULT_BUDGETS,
                      safety_fraction: float = .9) -> dict[str, Any]:
    sessions, sources = load_calibration(paths)
    grouped = defaultdict(list)
    for session in sessions:
        grouped[session["group_id"]].append(session)
    groups = []
    for group_id, chosen in sorted(grouped.items()):
        fitted = fit_group(chosen, probe_layer=chosen[0]["probe_layer"],
                           budgets_bytes=budgets_bytes, safety_fraction=safety_fraction)
        groups.append(dict(group_id=group_id, method=chosen[0]["method"], geometry=chosen[0]["geometry"], **fitted))
    return dict(schema=SCHEMA, status="OK" if groups else "EMPTY_CALIBRATION", groups=groups,
        training_session_ids=sorted({session["session_id"] for session in sessions}),
        source_sha256=sources, budgets_bytes=list(budgets_bytes), safety_fraction=safety_fraction,
        online_features_used=["relative_l2 of current enhanced/base attention outputs", "logical read costs"],
        forbidden_features_used=[], reference_answer_access=False, future_or_reference_logit_access=False,
        objective="fixed read-budget threshold calibration; no claim of optimality or task-quality prediction",
        extra_read_scope="probe pass base_visual_bytes+text_bytes+residual_bytes; local controller two-read charges are not added",
        threshold_rule="relative_l2 >= eta", threshold_candidates="0 and nextafter(each distinct observed delta,+infinity)",
        boundary_policy="lowest finite candidate satisfying the session-mean safety cap; Decimal comparison at exact byte boundary",
        deployment_scope="freeze literal policy and apply the existing hard cumulative cap; upgrades can alter later trajectories so forecast is approximate")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets-mib", nargs="+", type=float, default=[4, 8, 16, 32])
    parser.add_argument("--safety-fraction", type=float, default=.9)
    args = parser.parse_args(argv)
    budgets = []
    for value in args.budgets_mib:
        if not math.isfinite(value) or value <= 0 or value*MIB != int(value*MIB):
            parser.error("Each budget must convert to a positive whole byte count")
        budgets.append(int(value*MIB))
    if len(budgets) != len(set(budgets)):
        parser.error("Budget entries must be distinct")
    result = fit_probe_budgets(args.records, budgets_bytes=tuple(budgets), safety_fraction=args.safety_fraction)
    result["fit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], groups=len(result["groups"]),
        training_sessions=len(result["training_session_ids"]), output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
