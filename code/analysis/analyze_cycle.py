"""Independent offline scoring and analysis of real TrajKV cycle records.

Reference answers are read only here, after generation. Empty input produces an
explicit EMPTY_INPUT report and no plots. No synthetic observations are added.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


SCORE_PROTOCOL = "legacy_clevr_normalized_exact_v1"
ALIASES = {"ball": "sphere", "round thing": "sphere", "block": "cube", "big": "large",
           "tiny": "small", "metallic": "metal", "shiny": "metal", "matte": "rubber"}
NUMBER_WORDS = dict(zip(("zero", "one", "two", "three", "four", "five", "six", "seven",
                         "eight", "nine", "ten"), map(str, range(11))))
NUMBER_WORDS["grey"] = "gray"
READ_COLUMNS = ("base_visual_bytes", "residual_bytes", "text_bytes", "reference_bytes")
_INPUT_BYTES: dict[str, bytes] = {}
_READ_WARNINGS: list[str] = []


def read_input(path: Path) -> bytes:
    key = str(path.resolve())
    if key not in _INPUT_BYTES:
        _INPUT_BYTES[key] = path.read_bytes()
    return _INPUT_BYTES[key]


def normalize_short_answer(text: Any) -> str:
    """Exact copy of the normalization used by historical p182/p183 scoring.

    No substring or first-word extraction: a long answer containing the target
    is not automatically a correct short answer.
    """
    text = str(text).strip().lower()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = re.sub(r"[.!?]+$", "", text).strip()
    return NUMBER_WORDS.get(text, text)


def score_answer(prediction: Any, targets: Any) -> dict[str, Any]:
    if not isinstance(targets, (tuple, list)):
        targets = [targets]
    if not targets:
        raise ValueError("Cannot score against an empty reference list")
    pred = normalize_short_answer(prediction)
    normalized = [normalize_short_answer(t) for t in targets]
    alias = ALIASES.get(pred, pred)
    return {"exact": float(pred in normalized),
            "clevr_alias": float(any(alias == ALIASES.get(t, t) for t in normalized)),
            "normalized_prediction": pred, "normalized_targets": normalized}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    content = read_input(path).decode("utf-8-sig")
    lines = content.splitlines()
    for line_number, text in enumerate(lines, 1):
        if not text.strip():
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            if line_number == len(lines) and not content.endswith(("\n", "\r")):
                _READ_WARNINGS.append(f"Ignored unfinished final JSONL row: {path}:{line_number}")
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(read_input(path)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v
                             for k, v in row.items()})


def cluster_bootstrap(values: Iterable[tuple[str, float]], *, repeats: int = 2000,
                      seed: int = 1212) -> dict[str, Any]:
    """Question-weighted mean; resample original images, not individual turns.

    Repeated dialogs of the same source image remain in one cluster. One-image
    input has an observed mean but no estimated population confidence interval.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for cluster, value in values:
        if cluster is None or str(cluster) == "":
            raise ValueError("Image cluster ID is required for cluster bootstrap")
        if not math.isfinite(value):
            raise ValueError("Nonfinite bootstrap value")
        groups[str(cluster)].append(float(value))
    if not groups:
        return dict(mean=None, ci_lower=None, ci_upper=None, n_questions=0, n_images=0,
                    status="empty", repeats=0)
    sums = np.array([sum(v) for v in groups.values()], dtype=np.float64)
    counts = np.array([len(v) for v in groups.values()], dtype=np.float64)
    result = dict(mean=float(sums.sum() / counts.sum()), ci_lower=None, ci_upper=None,
                  n_questions=int(counts.sum()), n_images=len(groups),
                  status="insufficient_clusters", repeats=0)
    if len(groups) < 2:
        return result
    if repeats < 100:
        raise ValueError("Use at least 100 image-cluster bootstrap replicates")
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repeats, 256):
        indexes = rng.integers(0, len(groups), size=(min(256, repeats-start), len(groups)))
        samples.extend((sums[indexes].sum(1) / counts[indexes].sum(1)).tolist())
    low, high = np.quantile(samples, [0.025, 0.975])
    result.update(ci_lower=float(low), ci_upper=float(high), status="ok", repeats=repeats)
    return result


def load_references(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for path in paths:
        for row in load_jsonl(path):
            sid = str(row.get("id", row.get("session_id", "")))
            if not sid:
                raise ValueError(f"Reference row in {path} has no id")
            for reference in row["references"]:
                turn = int(reference["turn"])
                targets = reference.get("answers", reference.get("answer"))
                if targets is None:
                    raise ValueError(f"Missing reference answer: {sid}, turn {turn}")
                entry = dict(targets=targets, template=reference.get("template"),
                             reference_split=row.get("split"), reference_file=str(path))
                key = (sid, turn)
                if key in result and result[key]["targets"] != targets:
                    raise ValueError(f"Conflicting references for {key}")
                result[key] = entry
    return result


def load_sample_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in rows:
            result[str(row.get("id", row.get("session_id")))] = row
    return result


def infer_settings(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    arguments = json.loads(read_input(manifest_path).decode("utf-8-sig")).get("arguments", {})
    # Method lists, output directories, batch start/count and diagnostic output
    # options do not define the same-target generation configuration.
    keys = ("max_tokens", "max_patches", "turns", "dtype", "attn_implementation",
            "generation_config", "seed")
    return {key: arguments[key] for key in keys if key in arguments}


def input_completeness(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    observed = {(r["session_id"], r["method"], r.get("policy", "base")) for r in records}
    manifest_path = path.parent / "manifest.json"
    receipt = (path.parent / "complete.json").exists()
    result = dict(source_records=str(path), observed_records=len(records),
                  completion_receipt_present=receipt, expected_records=None, missing_records=None,
                  status="UNKNOWN_COMPLETENESS")
    if manifest_path.exists():
        manifest = json.loads(read_input(manifest_path).decode("utf-8-sig"))
        args = manifest.get("arguments", {})
        methods = args.get("methods", [])
        policies = args.get("policies", ["base"])
        ids = manifest.get("selected_ids", [])
        if methods and ids:
            expected = {(sid, method, policy) for sid in ids for method in methods
                        for policy in (policies if method.startswith("asym") else ["base"])}
            missing = expected-observed
            result.update(expected_records=len(expected), missing_records=len(missing),
                          missing_record_keys=[list(k) for k in sorted(missing)],
                          status="PARTIAL" if missing else ("COMPLETE" if receipt else "COMPLETE_RECORD_SET_NO_RECEIPT"))
    return result


def variant_label(method: str, policy: str) -> str:
    if method == "full":
        return "Full precision"
    if method == "native4":
        return "Native K4/V4"
    if method == "base2":
        return "K2/V2"
    if method.startswith("kvarn"):
        return f"{method.upper()} mechanism port"
    labels = {"always": "always refined", "asym": "always refined", "base": "base only",
              "margin": "margin replay", "carrier_margin": "carrier-margin replay"}
    return f"{method}: {labels.get(policy, policy)}"


def storage_profile(record: dict[str, Any]) -> dict[str, Any]:
    """Normalized owner layout plus kernel revision; no text-capacity grouping.

    A method name is not an encoding identity: compact K-only residuals can
    retain fewer tensors than an older K-only READ policy with both sides stored.
    """
    owned=record.get("owned_initial", {})
    denominator=owned.get("visual_fp16_denominator")
    ratio=lambda key: round(float(owned.get(key,0))/denominator,10) if denominator else None
    return dict(method=record.get("method"), visual_base_ratio=ratio("visual_base"),
                visual_residual_ratio=ratio("visual_residual"),
                extra_packed_reference_ratio=ratio("kvarn_packed_reference"),
                diagnostic_reference_ratio=ratio("diagnostic_fp16_reference"),
                runtime=owned.get("runtime"), kernel_variant=record.get("kernel_variant"),
                kernel_source_sha256=record.get("kernel_source_sha256"),
                explicit_layout=record.get("representation_layout", record.get("encoding_layout")))


def summarize_resources(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    turns = record.get("turns", [])
    result["generated_tokens"] = sum(len(t.get("token_ids", [])) for t in turns)
    result["generated_turns"] = len(turns)
    result["forced_chat_closures"] = sum(bool(t.get("forced_chat_closure")) for t in turns)
    result["replay_count"] = sum(int(t.get("replay_count", 0)) for t in turns)
    result["margin_steps"] = sum(len(t.get("margin", [])) for t in turns)
    result["replays_per_margin_step"] = (result["replay_count"] / result["margin_steps"]
                                          if result["margin_steps"] else None)
    probes = record.get("margin_probes", [])
    result["argmax_changes"] = sum(bool(p.get("changed")) for p in probes)
    result["argmax_change_rate_per_replay"] = (result["argmax_changes"] / len(probes) if probes else None)
    result["session_ms"] = record.get("session_ms")
    result["turn_engine_ms"] = sum(float(t.get("engine_ms", 0)) for t in turns)
    result["peak_allocated_bytes"] = record.get("peak_allocated")
    # Do not relabel session average as TPOT or invent a TTFT measurement.
    result["ttft_ms"] = record.get("ttft_ms")
    result["tpot_ms"] = record.get("tpot_ms")
    result["logical_read_scope"] = record.get("read_byte_scope", "unspecified")
    result["logical_read_excludes_initial_native_prefill"] = True
    result["logical_read_is_measured_hbm_traffic"] = False
    result["quality_scoring_allowed"] = record.get("quality_scoring_allowed", True)
    result["cost_comparison_allowed"] = record.get("cost_comparison_allowed", True)
    result["native_initial_logit_parity"] = record.get("native_initial_logit_parity")
    result["baseline_label"] = record.get("baseline_label", record.get("method"))
    result["resource_runtime_class"] = record.get("owned_initial", {}).get("runtime", "unspecified")
    profile=storage_profile(record)
    result["storage_layout_signature"]=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:12]
    result["storage_layout_profile"]=profile
    result["kernel_variant"]=record.get("kernel_variant")
    result["kernel_source_sha256"]=record.get("kernel_source_sha256")
    result["reported_extra_logical_read_bytes"]=record.get("extra_logical_read_bytes")
    result["read_budget_bytes"]=record.get("read_budget_bytes")
    result["read_budget_ok_recorded"]=record.get("read_budget_ok")
    result["read_ledger_exact_recorded"]=record.get("read_ledger_exact")
    result["intra_step_probe_events"]=len(record.get("intra_step_probes",[]))
    result["intra_step_probe_skipped"]=len(record.get("intra_step_probe_skipped",[]))
    result["intra_step_probe_upgrades"]=sum(bool((e.get("decision") or {}).get("upgrade")) for e in record.get("intra_step_probes",[]))
    result["probe_feature_wall_ms_with_sync"]=sum(float((e.get("decision") or {}).get("feature_wall_ms_including_synchronization",0))
                                                  for e in record.get("intra_step_probes",[]))
    result["controller_two_read_probe_charged_bytes"]=sum(int(e.get("charged_bytes",0)) for e in record.get("intra_step_probes",[]))
    result["read_accounting_ambiguous_reference"] = False
    for kind in ("committed", "replay", "probe", "other"):
        for column in READ_COLUMNS:
            result[f"{kind}_{column}"] = 0
    result["fused_attention_calls"] = 0
    result["fused_query_tokens"] = 0
    for event in record.get("read_events", []):
        kind = event.get("pass_kind", "committed")
        kind = kind if kind in ("committed", "replay", "probe") else "other"
        result["fused_attention_calls"] += int(event.get("calls", 0))
        result["fused_query_tokens"] += int(event.get("query_tokens", 0))
        for column in READ_COLUMNS:
            result[f"{kind}_{column}"] += int(event.get(column, 0))
        if event.get("reference_bytes", 0) and "effective_visual_bytes" not in event:
            result["read_accounting_ambiguous_reference"] = True
    for column in READ_COLUMNS:
        result[f"total_{column}"] = sum(result[f"{kind}_{column}"]
                                          for kind in ("committed", "replay", "probe", "other"))
    logical = sum(result[f"total_{column}"] for column in READ_COLUMNS)
    result["logical_total_read_bytes"] = None if result["read_accounting_ambiguous_reference"] else logical
    result["replay_logical_bytes"] = sum(result[f"replay_{column}"] for column in READ_COLUMNS)
    result["probe_extra_logical_bytes"]=sum(result[f"probe_{column}"] for column in READ_COLUMNS)
    # Match the frozen runtime's explicit extra-read ledger, retaining probe
    # base/text rereads. A controller's two-read probe charge includes the first
    # normal read, so adding that whole amount here would double-count it.
    extra=sum(int(e.get("residual_bytes",0))+
              (int(e.get("base_visual_bytes",0))+int(e.get("text_bytes",0))
               if e.get("pass_kind") in ("replay","repair","probe") else 0)
              for e in record.get("read_events",[]))
    result["computed_extra_logical_read_bytes"]=extra
    reported=record.get("extra_logical_read_bytes")
    result["extra_read_ledger_delta_bytes"]=extra-reported if reported is not None else None
    result["extra_read_ledger_equal"]=extra==reported if reported is not None else None
    budget=record.get("read_budget_bytes")
    result["read_budget_ok_computed"]=extra<=budget if budget is not None else None
    result["read_budget_headroom_bytes"]=budget-extra if budget is not None else None
    for suffix in ("initial", "final"):
        owned = record.get(f"owned_{suffix}", {})
        for key in ("visual_base", "visual_residual", "text_capacity", "position",
                    "kvarn_packed_reference", "diagnostic_fp16_reference", "total_unique",
                    "visual_fp16_denominator", "packed_visual_ratio", "encoded_reference_ratio",
                    "materialized_visual_ratio", "full_visual_retained", "runtime"):
            result[f"owned_{suffix}_{key}"] = owned.get(key)
        # Includes every reported visual owner, not just the packed payload.
        visual_keys = ("visual_base", "visual_residual", "kvarn_packed_reference", "diagnostic_fp16_reference")
        visual = sum(int(owned.get(key, 0)) for key in visual_keys)
        result[f"owned_{suffix}_all_visual_bytes"] = visual if owned else None
        denominator = owned.get("visual_fp16_denominator")
        result[f"owned_{suffix}_all_visual_ratio"] = visual / denominator if denominator else None
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def analyze_records(paths: list[Path], references: dict, samples: dict, *, repeats: int,
                    seed: int, baseline: str) -> dict[str, Any]:
    scored = []
    sessions = []
    cost_events = []
    probes = []
    warnings = []
    seen = set()
    layouts = {}
    group_descriptions = {}
    for path in paths:
        settings = infer_settings(path)
        for record in load_jsonl(path):
            sid = str(record["session_id"])
            sample = samples.get(sid, {})
            native = str(record.get("native_session_id", sample.get("native_session_id", sid)))
            offset = int(record.get("reference_turn_offset", sample.get("reference_turn_offset", 0)))
            image = record.get("image_id", sample.get("source_image_id", sample.get("image_id")))
            if not image:
                raise ValueError(f"{sid}: source image ID required; pass --samples or add image_id to records")
            dataset = record.get("dataset", sample.get("dataset", "clevr_dialog" if native.startswith("clevr") else "unspecified"))
            data_protocol = record.get("data_protocol", sample.get("protocol", "native"))
            protocol = record.get("protocol", "unspecified")
            description = dict(model=record.get("model"), protocol=protocol, dataset=dataset,
                               data_protocol=data_protocol, split=record.get("split", sample.get("split")),
                               backend=record.get("backend"), read_budget_bytes=record.get("read_budget_bytes"),
                               settings=settings)
            group = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:12]
            group_descriptions[group] = description
            method, policy = str(record["method"]), str(record.get("policy", "base"))
            variant = f"{method}:{policy}"
            layout=storage_profile(record)
            layout_key=(group,variant)
            if layout_key in layouts and layouts[layout_key]!=layout:
                raise ValueError(f"Storage/kernel revision differs for {variant} in group {group}; "
                                 "analyze compact-axis and older retained-layout runs separately")
            layouts[layout_key]=layout
            key = (group, sid, variant)
            if key in seen:
                raise ValueError(f"Duplicate run for {key}; distinguish configurations or exclude duplicate files")
            seen.add(key)
            common = dict(group_id=group, session_id=sid, native_session_id=native, image_id=str(image),
                          dataset=dataset, split=description["split"], protocol=protocol,
                          data_protocol=data_protocol, method=method, policy=policy, variant=variant,
                          label=variant_label(method, policy), source_records=str(path))
            sessions.append(dict(common, **summarize_resources(record)))
            for event in record.get("read_events", []):
                cost_events.append(dict(common, **event))
            for probe in record.get("margin_probes", []):
                probes.append(dict(common, **probe))
            turn_seen = set()
            for turn in record.get("turns", []):
                number = int(turn["turn"])
                if number in turn_seen:
                    raise ValueError(f"Duplicate turn {number}: {key}")
                turn_seen.add(number)
                reference_turn = number-offset
                reference = references.get((native, reference_turn))
                scoring_allowed = record.get("quality_scoring_allowed", True)
                output = dict(common, turn=number, reference_turn=reference_turn,
                              prediction=turn.get("answer", ""), generated_tokens=len(turn.get("token_ids", [])),
                              engine_ms=turn.get("engine_ms"), replay_count=turn.get("replay_count", 0),
                              scored=reference is not None and scoring_allowed, score_protocol=SCORE_PROTOCOL,
                              oracle="oracle" in policy.lower())
                if not scoring_allowed:
                    output.update(exact=None, clevr_alias=None, scoring_note="quality_scoring_disallowed_by_runtime_parity")
                    warnings.append(f"Quality scoring explicitly disallowed: {sid}, {variant}")
                elif reference is None:
                    output.update(exact=None, clevr_alias=None,
                                  scoring_note="unscored_bridge_prefix" if reference_turn <= 0 else "missing_reference")
                    if reference_turn > 0:
                        warnings.append(f"Missing reference: {native}, turn {reference_turn}")
                else:
                    output.update(score_answer(turn.get("answer", ""), reference["targets"]))
                    output.update(reference_answer=reference["targets"], template=reference["template"], scoring_note="scored")
                scored.append(output)

    valid = [r for r in scored if r["scored"]]
    index = {(r["group_id"], r["session_id"], r["reference_turn"], r["variant"]): r for r in valid}
    variants = defaultdict(set)
    for r in valid:
        variants[r["group_id"]].add(r["variant"])
    pairs = []
    pair_summary = []
    for group, method_ids in variants.items():
        if baseline not in method_ids:
            warnings.append(f"Group {group}: requested baseline {baseline} absent; no paired contrast")
            continue
        baseline_keys = {(s, t) for g, s, t, v in index if g == group and v == baseline}
        for variant in sorted(method_ids-{baseline}):
            variant_keys = {(s, t) for g, s, t, v in index if g == group and v == variant}
            subset = []
            for sid, turn in sorted(baseline_keys | variant_keys):
                a = index.get((group, sid, turn, baseline))
                b = index.get((group, sid, turn, variant))
                matched = a is not None and b is not None
                entry = dict(group_id=group, session_id=sid, image_id=(a or b)["image_id"],
                             reference_turn=turn, baseline=baseline, variant=variant, matched=matched,
                             baseline_prediction=a["prediction"] if a else None,
                             method_prediction=b["prediction"] if b else None,
                             baseline_exact=a["exact"] if a else None, method_exact=b["exact"] if b else None,
                             delta_exact=b["exact"]-a["exact"] if matched else None,
                             delta_alias=b["clevr_alias"]-a["clevr_alias"] if matched else None,
                             delta_tokens=b["generated_tokens"]-a["generated_tokens"] if matched else None)
                pairs.append(entry)
                if matched:
                    subset.append(entry)
            for scope in ("all_scored", "later_turns"):
                rows = [r for r in subset if scope == "all_scored" or r["reference_turn"] >= 2]
                stats = cluster_bootstrap(((r["image_id"], r["delta_exact"]) for r in rows), repeats=repeats, seed=seed)
                pair_summary.append(dict(group_id=group, variant=variant, baseline=baseline, scope=scope,
                    **stats, delta_pp=100*stats["mean"] if stats["mean"] is not None else None,
                    repaired=sum(r["baseline_exact"] == 0 and r["method_exact"] == 1 for r in rows),
                    new_errors=sum(r["baseline_exact"] == 1 and r["method_exact"] == 0 for r in rows),
                    unmatched_baseline_questions=len(baseline_keys-variant_keys),
                    unmatched_method_questions=len(variant_keys-baseline_keys)))

    summaries = []
    turn_summary = []
    curves = []
    for group, method_ids in variants.items():
        common_keys = None
        for variant in method_ids:
            keys = {(s, t) for g, s, t, v in index if g == group and v == variant}
            common_keys = keys if common_keys is None else common_keys & keys
        for variant in sorted(method_ids):
            rows = [r for r in valid if r["group_id"] == group and r["variant"] == variant]
            resource_rows = [r for r in sessions if r["group_id"] == group and r["variant"] == variant]
            for scope in ("all_scored", "later_turns"):
                scope_rows = [r for r in rows if scope == "all_scored" or r["reference_turn"] >= 2]
                stats = cluster_bootstrap(((r["image_id"], r["exact"]) for r in scope_rows), repeats=repeats, seed=seed)
                summaries.append(dict(group_id=group, variant=variant, label=rows[0]["label"], scope=scope,
                                      **stats, alias_accuracy=_mean(scope_rows, "clevr_alias"),
                                      n_sessions=len({r["session_id"] for r in scope_rows})))
            for turn in sorted({r["reference_turn"] for r in rows}):
                selected = [r for r in rows if r["reference_turn"] == turn]
                stats = cluster_bootstrap(((r["image_id"], r["exact"]) for r in selected), repeats=repeats, seed=seed)
                turn_summary.append(dict(group_id=group, variant=variant, reference_turn=turn, **stats))
            matched = [r for r in rows if (r["session_id"], r["reference_turn"]) in (common_keys or set())]
            matched_sessions = {r["session_id"] for r in matched}
            same_resources = [r for r in resource_rows if r["session_id"] in matched_sessions]
            if matched:
                stats = cluster_bootstrap(((r["image_id"], r["exact"]) for r in matched), repeats=repeats, seed=seed)
                curve = dict(group_id=group, variant=variant, label=matched[0]["label"], **stats,
                             n_sessions=len(matched_sessions), oracle=matched[0]["oracle"],
                             cost_comparison_allowed=all(r["cost_comparison_allowed"] for r in same_resources),
                             runtime_classes=sorted({r["resource_runtime_class"] for r in same_resources}),
                             pairing="same question intersection across displayed variants")
                for key in ("logical_total_read_bytes", "replay_logical_bytes", "generated_tokens", "session_ms",
                            "replay_count", "margin_steps", "owned_initial_all_visual_bytes",
                            "owned_initial_all_visual_ratio", "owned_initial_total_unique", "peak_allocated_bytes",
                            "computed_extra_logical_read_bytes", "probe_extra_logical_bytes", "read_budget_bytes"):
                    curve[key] = _mean(same_resources, key)
                for kind in ("committed", "replay", "probe", "other"):
                    for column in READ_COLUMNS:
                        curve[f"{kind}_{column}"] = _mean(same_resources, f"{kind}_{column}")
                steps = sum(r["margin_steps"] for r in same_resources)
                curve["replays_per_margin_step"] = sum(r["replay_count"] for r in same_resources)/steps if steps else None
                curves.append(curve)
    return dict(scored_turns=scored, resource_sessions=sessions, read_events=cost_events,
                margin_probes=probes, paired_diffs=pairs, paired_summary=pair_summary,
                method_summary=summaries, turn_summary=turn_summary, curve_summary=curves,
                groups=group_descriptions, warnings=sorted(set(warnings)))


def load_mechanism_points(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    points, warnings = [], []
    for path in paths:
        for row in load_jsonl(path):
            current = row.get("delta_current")
            future = row.get("delta_future")
            if current is None and all(k in row for k in ("current_loss_before", "current_loss_after")):
                current = row["current_loss_before"]-row["current_loss_after"]
            if future is None and all(k in row for k in ("future_loss_before", "future_loss_after")):
                future = row["future_loss_before"]-row["future_loss_after"]
            if current is None or future is None:
                warnings.append(f"Mechanism row lacks current/future paired effects: {row.get('intervention_id')}")
                continue
            if not all(math.isfinite(float(v)) for v in (current, future)):
                raise ValueError("Nonfinite mechanism effect")
            point = dict(row, delta_current=float(current), delta_future=float(future), source_records=str(path))
            # Missing metric/horizon definitions remain exportable, but do not
            # silently become a publishable scatter with guessed dimensions.
            point["plot_eligible"] = all(k in row for k in ("current_metric", "future_metric", "horizon"))
            points.append(point)
    return points, warnings


def plot_results(result: dict[str, Any], points: list[dict[str, Any]], directory: Path) -> list[str]:
    if not result["curve_summary"] and not any(p["plot_eligible"] for p in points):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .8, "legend.frameon": False})
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    palette = ["#526D82", "#6FA3B5", "#C5A56B", "#7FA580", "#9C8CB4", "#CE8C82", "#68737D", "#B6A7CA",
               "#A87656", "#679D96", "#B29A60", "#AC7F9D"]
    markers = ["o", "s", "D", "P", "v", "X", "h", "*", "d", "p", "<", ">"]

    def save(fig, stem):
        for suffix in ("png", "pdf"):
            path = directory / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            outputs.append(str(path))
        plt.close(fig)

    grouped = defaultdict(list)
    for row in result["curve_summary"]:
        grouped[row["group_id"]].append(row)
    for group, rows in grouped.items():
        all_rows = rows
        rows = [row for row in all_rows if row.get("cost_comparison_allowed", True)]
        # Mechanism ports retain ordinary answer scoring, but a materialized
        # reference runtime is not a packed-serving efficiency comparator.
        if len(rows) != len(all_rows):
            fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
            for i, row in enumerate(all_rows):
                y = 100*row["mean"]
                error = None if row["ci_lower"] is None else [[y-100*row["ci_lower"]], [100*row["ci_upper"]-y]]
                ax.bar(i, y, color=palette[i % len(palette)], hatch="//" if not row["cost_comparison_allowed"] else None)
                ax.errorbar(i, y, yerr=error, color="#424B54", fmt="none", capsize=2)
            ax.set_xticks(range(len(all_rows)), [row["label"] for row in all_rows], rotation=20, ha="right", fontsize=7)
            ax.set_ylabel("Normalized exact accuracy (%)")
            ax.set_ylim(0, 102)
            ax.set_title("Quality only · hatched: materialized mechanism ports", fontsize=9)
            save(fig, f"quality_only_{group}")
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
        for i, row in enumerate(rows):
            color = palette[i % len(palette)]
            y = 100*row["mean"]
            error = None if row["ci_lower"] is None else [[y-100*row["ci_lower"]], [100*row["ci_upper"]-y]]
            for ax, key, scale in ((axes[0], "logical_total_read_bytes", 2**30),
                                   (axes[1], "owned_initial_all_visual_bytes", 2**20)):
                if row.get(key) is not None:
                    ax.errorbar(row[key]/scale, y, yerr=error, fmt="^" if row["oracle"] else markers[i % len(markers)],
                                color=color, markerfacecolor="white" if row["oracle"] else color,
                                capsize=2, markersize=5, label=row["label"])
        axes[0].set_xlabel("Logical tensor reads / session (GiB)")
        axes[1].set_xlabel("Owned visual tensors (MiB)")
        for ax, letter in zip(axes, "ab"):
            ax.set_ylabel("Normalized exact accuracy (%)")
            ax.set_ylim(-2, 102)
            ax.text(-.15, 1.03, letter, transform=ax.transAxes, weight="bold", fontsize=10)
            ax.grid(axis="y", color="#ECEFF1", linewidth=.5)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(handles)), fontsize=7)
        n = rows[0]["n_images"]
        prefix = "Partial development snapshot" if result.get("partial") else "Development comparison"
        fig.suptitle(f"{prefix} · {n} image clusters", fontsize=9)
        save(fig, f"resource_quality_{group}")

        fig, ax = plt.subplots(figsize=(7.2, 3.1), layout="constrained")
        terms = [("committed_base_visual_bytes", "Committed visual base", "#526D82"),
                 ("committed_text_bytes", "Committed text", "#A7B9C5"),
                 ("committed_residual_bytes", "Committed residual", "#7FA580"),
                 ("replay_base_visual_bytes", "Replay visual base", "#B99661"),
                 ("replay_text_bytes", "Replay text", "#E0C9A3"),
                 ("replay_residual_bytes", "Replay residual", "#C58D83"),
                 ("probe_base_visual_bytes", "Probe base reread", "#9C8CB4"),
                 ("probe_text_bytes", "Probe text reread", "#C7B7DC"),
                 ("probe_residual_bytes", "Probe residual", "#AE7F9B")]
        bottom = np.zeros(len(rows))
        for key, label, color in terms:
            values = np.array([r.get(key) or 0 for r in rows])/2**30
            if not np.any(values):
                continue
            ax.bar(range(len(rows)), values, bottom=bottom, color=color, label=label, width=.65)
            bottom += values
        auxiliary = np.array([sum((r.get(f"{kind}_{column}") or 0)
                                  for kind in ("other",) for column in READ_COLUMNS)
                              + (r.get("committed_reference_bytes") or 0)
                              + (r.get("replay_reference_bytes") or 0)
                              + (r.get("probe_reference_bytes") or 0) for r in rows])/2**30
        if np.any(auxiliary):
            ax.bar(range(len(rows)), auxiliary, bottom=bottom, color="#B6A7CA", label="Probe / other / reference", width=.65)
        ax.set_xticks(range(len(rows)), [r["label"] for r in rows], rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Logical tensor reads / session (GiB)")
        ax.legend(loc="upper left", ncol=3, fontsize=6)
        ax.set_title(("Partial snapshot · " if result.get("partial") else "")+
                     "probe/replay includes base visual and text KV", fontsize=9)
        save(fig, f"read_breakdown_{group}")

    mechanism_groups = defaultdict(list)
    for p in points:
        if p["plot_eligible"]:
            key = (p["current_metric"], p["future_metric"], str(p["horizon"]))
            mechanism_groups[key].append(p)
    for (current_metric, future_metric, horizon), rows in mechanism_groups.items():
        fig, ax = plt.subplots(figsize=(4.2, 3.5), layout="constrained")
        actions = sorted({str(r.get("action", r.get("policy", "intervention"))) for r in rows})
        for i, action in enumerate(actions):
            subset = [r for r in rows if str(r.get("action", r.get("policy", "intervention"))) == action]
            for oracle in (False, True):
                chosen = [r for r in subset if bool(r.get("oracle", False)) == oracle]
                if chosen:
                    ax.scatter([r["delta_current"] for r in chosen], [r["delta_future"] for r in chosen],
                        color=palette[i % len(palette)], facecolors="none" if oracle else palette[i % len(palette)],
                        marker="^" if oracle else "o", s=20, alpha=.75,
                        label=action+(" (oracle)" if oracle else ""))
        ax.axhline(0, color="#ADB5BD", linewidth=.7)
        ax.axvline(0, color="#ADB5BD", linewidth=.7)
        ax.set_xlabel(f"Current effect ({current_metric})")
        ax.set_ylabel(f"Future effect ({future_metric})")
        ax.set_title(f"Intervention effects · horizon {horizon}", fontsize=9)
        ax.legend(fontsize=7)
        ident = hashlib.sha256(repr((current_metric, future_metric, horizon)).encode()).hexdigest()[:10]
        save(fig, f"current_future_{ident}")
    return outputs


def main(argv=None) -> int:
    _INPUT_BYTES.clear()
    _READ_WARNINGS.clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--references", nargs="+", type=Path, required=True)
    parser.add_argument("--samples", nargs="*", type=Path, default=[])
    parser.add_argument("--mechanism", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--baseline", default="full:base")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1212)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    references = load_references(args.references)
    samples = load_sample_metadata(args.samples)
    result = analyze_records(args.records, references, samples, repeats=args.bootstrap,
                             seed=args.seed, baseline=args.baseline)
    completeness = [input_completeness(path) for path in args.records]
    result["partial"] = any(row["status"] != "COMPLETE" for row in completeness)
    points, warnings = load_mechanism_points(args.mechanism)
    result["warnings"].extend(warnings+_READ_WARNINGS)
    for key in ("scored_turns", "resource_sessions", "read_events", "margin_probes", "paired_diffs",
                "paired_summary", "method_summary", "turn_summary", "curve_summary"):
        write_csv(args.output / f"{key}.csv", result[key])
    write_csv(args.output / "mechanism_points.csv", points)
    inputs = args.records+args.references+args.samples+args.mechanism
    status = ("PARTIAL" if result["partial"] else "OK") if result["resource_sessions"] else "EMPTY_INPUT"
    summary = dict(status=status, input_completeness=completeness,
        score_protocol=SCORE_PROTOCOL, primary_metric="normalized_exact",
        alias_metric="auxiliary CLEVR alias accuracy", baseline=args.baseline,
        n_records=len(result["resource_sessions"]), n_scored_turns=sum(r["scored"] for r in result["scored_turns"]),
        n_unscored_turns=sum(not r["scored"] for r in result["scored_turns"]),
        bootstrap=dict(unit="source image", repeats=args.bootstrap, seed=args.seed, interval="percentile 95%"),
        groups=result["groups"], warnings=result["warnings"],
        paired_summary=result["paired_summary"], method_summary=result["method_summary"],
        resource_scope="Tensor-owner bytes and logical fused-call reads; not physical HBM bandwidth",
        input_sha256={str(p.resolve()): sha256(p) for p in inputs},
        script_sha256=sha256(Path(__file__)), reference_answers_read_only_by_offline_analysis=True)
    figure_paths = plot_results(result, points, args.figures) if args.figures else []
    summary["figures"] = figure_paths
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(status=summary["status"], records=summary["n_records"],
                         scored_turns=summary["n_scored_turns"], figures=len(figure_paths)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
