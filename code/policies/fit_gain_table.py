"""Fit matched current/future action tables from mechanism_fidelity v2 only.

No reference answer is used. The objective is reference-distribution fidelity,
not task accuracy. The fitted table is frozen before online evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "trajkv.gain_tables.fidelity.v2"
ACTION_SCHEMA = "trajkv.mechanism_fidelity.v2"


def _read_jsonl(path: Path):
    content = path.read_bytes()
    result = []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != ACTION_SCHEMA:
            raise ValueError(f"{path}:{line_number}: expected {ACTION_SCHEMA}; v1 likelihood gains cannot fit a fidelity table")
        result.append(row)
    return result, hashlib.sha256(content).hexdigest()


def _finite(value, name):
    if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _mean(values):
    return sum(values)/len(values) if values else None


def training_id_file(path: Path) -> tuple[set[str], str]:
    """Accept a runner JSON list, a JSON list of IDs, or one ID per text line."""
    data = path.read_bytes()
    text = data.decode("utf-8-sig").strip()
    if text.startswith(("[", "{")):
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("training_session_ids", payload.get("selected_ids", []))
        ids = {str(item.get("id", item.get("session_id"))) if isinstance(item, dict) else str(item)
               for item in payload}
    else:
        ids = {line.strip() for line in text.splitlines() if line.strip()}
    if not ids or "None" in ids:
        raise ValueError("Training ID file contains no valid IDs")
    return ids, hashlib.sha256(data).hexdigest()


def fit_gain_tables(paths: list[Path], *, training_ids: set[str] | None = None,
                    allow_nondevelopment: bool = False) -> dict[str, Any]:
    observations = []
    skipped = []
    sources = {}
    candidate_sessions = set()
    image_ids = {}
    seen = set()
    settings = []
    for path in paths:
        records, source_hash = _read_jsonl(path)
        sources[str(path.resolve())] = source_hash
        manifest = path.parent / "manifest.json"
        if manifest.exists():
            raw = manifest.read_bytes()
            arguments = json.loads(raw.decode("utf-8-sig")).get("arguments", {})
            chosen = {k: arguments[k] for k in ("model", "turns", "max_tokens", "max_patches", "future_horizon") if k in arguments}
            settings.append(chosen)
            sources[str(manifest.resolve())] = hashlib.sha256(raw).hexdigest()
        for record in records:
            sid = str(record["session_id"])
            if training_ids is not None and sid not in training_ids:
                skipped.append(dict(session_id=sid, reason="outside_training_ids"))
                continue
            split = record.get("split")
            if not allow_nondevelopment and split not in ("development", "calibration", "train"):
                raise ValueError(f"{sid}: split={split!r}; fitting requires development/calibration/train sessions")
            candidate_sessions.add(sid)
            image_ids[sid] = record.get("image_id")
            for action in record.get("actions", []):
                name = str(action["action"])
                write_turn = int(action["write_turn"])
                key = (sid, write_turn, name)
                if key in seen:
                    raise ValueError(f"Duplicate fit action: {key}")
                seen.add(key)
                fidelity = action.get("current_fidelity", {})
                current = fidelity.get("content_fidelity_gain_mean")
                future_rows = [future for future in action.get("future", [])
                               if future.get("fidelity", {}).get("content_fidelity_gain_mean") is not None]
                cost = action.get("current_read_cost", {}).get("totals", {})
                residual = cost.get("residual_bytes")
                if current is None or not future_rows or not residual or residual <= 0:
                    skipped.append(dict(session_id=sid, write_turn=write_turn, action=name,
                        reason="requires matched current/future content fidelity and positive residual read cost"))
                    continue
                if fidelity.get("kl_direction") != "D_KL(p_candidate || p_same_token_allref)":
                    raise ValueError(f"{key}: unexpected KL direction")
                future_gains = [_finite(row["fidelity"]["content_fidelity_gain_mean"], "future_fidelity_gain")
                                for row in future_rows]
                current = _finite(current, "current_fidelity_gain")
                horizons = [int(row["target_turn"])-write_turn for row in future_rows]
                if min(horizons) <= 0:
                    raise ValueError(f"{key}: future target must follow write turn")
                observations.append(dict(session_id=sid, image_id=record.get("image_id"), split=split,
                    action=name, write_turn=write_turn, layer_group=int(action["layer_group"]), phase=str(action["phase"]),
                    current_gain=current, future_gain=_mean(future_gains),
                    future_gain_sum_over_turns=sum(future_gains), target_turn_count=len(future_rows), horizons=horizons,
                    residual_read_bytes=int(residual),
                    action_total_logical_read_bytes=sum(int(cost.get(k, 0)) for k in
                        ("base_visual_bytes", "residual_bytes", "text_bytes", "reference_bytes")),
                    current_content_tokens=int(fidelity.get("content_token_count", 0)),
                    future_content_tokens=sum(int(row["fidelity"].get("content_token_count", 0)) for row in future_rows)))
    known_settings = {json.dumps(row, sort_keys=True) for row in settings if row}
    if len(known_settings) > 1:
        raise ValueError("Mixed model/generation/horizon settings; fit separate tables to preserve a same-target comparison")
    requested = sorted(training_ids) if training_ids is not None else sorted(candidate_sessions)
    if not observations:
        return dict(schema=SCHEMA, status="EMPTY_FIT", requested_training_session_ids=requested,
                    training_session_ids=[], source_sha256=sources, skipped=skipped, tables={},
                    primary_objective="reference logit KL fidelity, not answer accuracy")
    per_session_actions = defaultdict(set)
    for observation in observations:
        per_session_actions[observation["session_id"]].add(observation["action"])
    shared_actions = set.intersection(*(value for value in per_session_actions.values()))
    # Both objectives use exactly the same observations and common action library.
    included = [row for row in observations if row["action"] in shared_actions]
    if not included:
        raise ValueError("No common current/future action library across the selected training sessions")
    training = sorted({row["session_id"] for row in included})
    tables = {}
    for objective in ("current", "future"):
        entries = []
        for name in sorted(shared_actions):
            action_rows = [row for row in included if row["action"] == name]
            by_session = defaultdict(list)
            for row in action_rows:
                by_session[row["session_id"]].append(row)
            # A long dialog or many write probes does not count as extra sessions.
            session_gains = [_mean([row[f"{objective}_gain"] for row in group]) for group in by_session.values()]
            session_costs = [_mean([row["residual_read_bytes"] for row in group]) for group in by_session.values()]
            gain, cost = _mean(session_gains), _mean(session_costs)
            template = action_rows[0]
            if any((row["phase"], row["layer_group"]) != (template["phase"], template["layer_group"]) for row in action_rows):
                raise ValueError(f"Action name {name!r} maps to inconsistent execution units")
            entries.append(dict(action=name, phase=template["phase"], layer_group=template["layer_group"],
                key=f"{template['phase']}|layer{template['layer_group']}|all",
                expected_gain=gain, gain_per_residual_byte=gain/cost,
                unit_residual_read_bytes=cost, min_residual_read_bytes=min(row["residual_read_bytes"] for row in action_rows),
                max_residual_read_bytes=max(row["residual_read_bytes"] for row in action_rows),
                training_sessions=len(by_session), action_instances=len(action_rows),
                positive_instance_fraction=sum(row[f"{objective}_gain"] > 0 for row in action_rows)/len(action_rows),
                per_session=[dict(session_id=sid, mean_gain=_mean([row[f"{objective}_gain"] for row in group]),
                                  mean_residual_read_bytes=_mean([row["residual_read_bytes"] for row in group]))
                             for sid, group in sorted(by_session.items())]))
        ranked = sorted(entries, key=lambda row: (-row["gain_per_residual_byte"], row["action"]))
        by_phase = {phase: [row["layer_group"] for row in ranked if row["phase"] == phase]
                    for phase in sorted({row["phase"] for row in entries})}
        tables[objective] = dict(entries=entries, ranked_actions=[row["action"] for row in ranked],
            phase_layer_group_ranking=by_phase,
            positive_ranked_actions=[row["action"] for row in ranked if row["expected_gain"] > 0],
            policy_lookup={row["key"]: row["expected_gain"] for row in entries},
            policy_score_lookup={row["key"]: row["gain_per_residual_byte"] for row in entries},
            score_definition="ratio of session-mean fidelity gain to session-mean residual read bytes",
            online_feature_availability="online_after_table_frozen",
            source_outcome_availability="offline_reference_only")
    return dict(schema=SCHEMA, status="OK", requested_training_session_ids=requested,
        training_session_ids=training, training_image_ids={sid: image_ids[sid] for sid in training},
        missing_requested_sessions=sorted(set(requested)-set(training)),
        shared_action_library=sorted(shared_actions),
        omitted_nonshared_action_instances=len(observations)-len(included),
        same_training_observations_for_current_and_future=True,
        source_sha256=sources, model_generation_settings=settings[0] if settings else {},
        primary_objective="content-token mean D_KL(p_base||p_allref)-D_KL(p_action||p_allref), not answer accuracy",
        reference_condition="complete original-FP16-visual history teacher-forced on identical canonical base tokens",
        future_aggregation="mean of per-target-turn content-token mean fidelity gains, then equal-session averaging",
        resource_scope="logical injected residual bytes per action; not measured HBM traffic; fixed stored representation",
        fit_reference_access="offline only; online schedule reads frozen ranking/table, never logits from reference or future",
        observations=included, skipped=skipped, tables=tables)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-ids", type=Path)
    parser.add_argument("--allow-nondevelopment", action="store_true")
    args = parser.parse_args(argv)
    ids, id_hash = training_id_file(args.training_ids) if args.training_ids else (None, None)
    result = fit_gain_tables(args.records, training_ids=ids, allow_nondevelopment=args.allow_nondevelopment)
    result["fit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.training_ids:
        result["training_id_file"] = str(args.training_ids.resolve())
        result["training_id_file_sha256"] = id_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], training_sessions=len(result["training_session_ids"]),
                          shared_actions=len(result.get("shared_action_library", [])), output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
