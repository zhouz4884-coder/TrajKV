"""Analyze measured four-arm, action and carrier-surgery development records.

V1 target-token likelihood is a diagnostic, never answer accuracy or fidelity.
Carrier surgery is a synthetic numerical intervention, never a serving policy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_cycle import cluster_bootstrap, load_jsonl, sha256, write_csv, write_json


ARMS = ("history_ref__current_ref", "history_ref__current_base",
        "history_base__current_ref", "history_base__current_base")


def distribution(values):
    values = np.asarray([float(x) for x in values if x is not None], dtype=float)
    if not len(values):
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None, p05=None, p95=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)))


def metric_row(metrics):
    rows = metrics.get("per_token", [])
    content = [r for r in rows if not r.get("is_eos", False)]
    rms = math.sqrt(sum(float(r.get("rms_logit_difference", 0))**2 for r in rows)/len(rows)) if rows else None
    return dict(kl=metrics.get("mean_kl_reference_to_actual"), rms_logits=rms,
                max_abs_logits=metrics.get("max_abs_logit_difference"),
                argmax_flips=metrics.get("argmax_changes", 0), token_count=metrics.get("token_count", len(rows)),
                content_argmax_flips=sum(bool(r.get("argmax_changed")) for r in content),
                content_token_count=metrics.get("content_token_count", len(content)),
                base_target_logprob_diagnostic=metrics.get("content_logprob_gain_mean"))


def ranks(values, descending=True):
    values = np.asarray(values, dtype=float)
    ordered = np.argsort(-values if descending else values, kind="stable")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i+1
        while j < len(values) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        result[ordered[i:j]] = (i+j-1)/2+1
        i = j
    return result


def spearman(x, y):
    if len(x) < 2:
        return None
    a, b = ranks(x), ranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_group(rows, fields, metrics, repeats, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    results = []
    for key, chosen in sorted(groups.items(), key=lambda item: repr(item[0])):
        for metric in metrics:
            selected = [r for r in chosen if r.get(metric) is not None]
            stats = cluster_bootstrap(((r["image_id"], float(r[metric])) for r in selected), repeats=repeats, seed=seed)
            results.append(dict(zip(fields, key), metric=metric, **stats,
                                n_sessions=len({r["session_id"] for r in selected})))
    return results


def four_arm_analysis(records, repeats=2000, seed=1212):
    rows, contrasts, nulls, action_rows = [], [], [], []
    for record in records:
        common = dict(session_id=record["session_id"], image_id=record["image_id"],
                      split=record.get("split"), schema=record.get("schema"))
        for result in record["results"]:
            turn = result["turn"]
            for arm, metrics in result["arms"].items():
                rows.append(dict(common, turn=turn, arm=arm, **metric_row(metrics)))
            kl = {arm: result["arms"][arm]["mean_kl_reference_to_actual"] for arm in ARMS}
            rr, rb, br, bb = (kl[name] for name in ARMS)
            state = result.get("history_state_before", {}).get("aggregate", {})
            contrasts.append(dict(common, turn=turn, direct_kl=rb-rr, history_kl=br-rr,
                joint_kl=bb-rr, kl_interaction=bb-br-rb+rr,
                reference_history_rescue_kl=bb-rb, reference_read_rescue_kl=bb-br,
                history_k_rms=state.get("k", {}).get("rms"), history_v_rms=state.get("v", {}).get("rms"),
                target_logprob_contrasts_diagnostic=result.get("contrasts")))
            for name, null in result["numerical_nulls"].items():
                state = null.get("history_state", {}).get("aggregate", {})
                nulls.append(dict(common, turn=turn, control=name, **metric_row(null),
                    state_k_max=state.get("k", {}).get("max_abs"), state_v_max=state.get("v", {}).get("max_abs")))
        for action in record.get("actions", []):
            fidelity = action.get("current_fidelity")
            metric = "KL_fidelity_gain_not_accuracy" if fidelity else "v1_base_target_logprob_diagnostic_not_accuracy"
            current = (fidelity["content_fidelity_gain_mean"] if fidelity
                       else action["current"]["content_logprob_gain_mean"])
            for future in action["future"]:
                future_fidelity = future.get("fidelity")
                gain = (future_fidelity["content_fidelity_gain_mean"] if future_fidelity
                        else future["metrics"]["content_logprob_gain_mean"])
                cost = action["current_read_cost"]["totals"]
                action_rows.append(dict(common, action=action["action"], layer_group=action["layer_group"],
                    phase=action["phase"], write_turn=action["write_turn"], target_turn=future["target_turn"],
                    horizon=future["target_turn"]-action["write_turn"], metric=metric,
                    current_gain=current, future_gain=gain,
                    residual_read_bytes=cost["residual_bytes"],
                    action_logical_read_bytes=sum(cost.get(k, 0) for k in ("base_visual_bytes", "residual_bytes", "text_bytes", "reference_bytes"))))
    summaries = summarize_group(rows, ["arm"], ["kl", "rms_logits", "argmax_flips"], repeats, seed)
    contrast_summary = summarize_group(contrasts, [], ["direct_kl", "history_kl", "joint_kl", "kl_interaction",
        "reference_history_rescue_kl", "reference_read_rescue_kl", "history_k_rms", "history_v_rms"], repeats, seed)
    null_summary = []
    for control in sorted({r["control"] for r in nulls}):
        selected = [r for r in nulls if r["control"] == control]
        null_summary.append(dict(control=control, n_turns=len(selected),
            logit_max_abs=distribution(r["max_abs_logits"] for r in selected),
            kl=distribution(r["kl"] for r in selected),
            total_argmax_flips=sum(r["argmax_flips"] for r in selected),
            state_k_max=max((r["state_k_max"] or 0 for r in selected), default=None),
            state_v_max=max((r["state_v_max"] or 0 for r in selected), default=None)))

    # Average horizons within a write intervention before aggregating it. This
    # prevents counting the identical current effect twice for two future turns.
    interventions = defaultdict(list)
    for row in action_rows:
        interventions[(row["session_id"], row["action"], row["write_turn"], row["metric"])].append(row)
    reduced = []
    for selected in interventions.values():
        template = selected[0]
        valid_future = [r["future_gain"] for r in selected if r["future_gain"] is not None]
        reduced.append(dict(template, future_gain=float(np.mean(valid_future)) if valid_future else None,
                            target_turns=len(selected)))
    action_summary = []
    for metric in sorted({r["metric"] for r in reduced}):
        subset = [r for r in reduced if r["metric"] == metric]
        for action in sorted({r["action"] for r in subset}):
            selected = [r for r in subset if r["action"] == action]
            entry = dict(action=action, metric=metric, layer_group=selected[0]["layer_group"], phase=selected[0]["phase"],
                         n_images=len({r["image_id"] for r in selected}), n_write_interventions=len(selected))
            for name in ("current_gain", "future_gain"):
                stats = cluster_bootstrap(((r["image_id"], r[name]) for r in selected if r[name] is not None), repeats=repeats, seed=seed)
                entry.update({f"{name}_{k}": v for k, v in stats.items()})
            entry["mean_residual_read_bytes"] = float(np.mean([r["residual_read_bytes"] for r in selected]))
            action_summary.append(entry)
        ranked = [r for r in action_summary if r["metric"] == metric]
        current_ranks = ranks([r["current_gain_mean"] for r in ranked])
        future_ranks = ranks([r["future_gain_mean"] for r in ranked])
        for row, current, future in zip(ranked, current_ranks, future_ranks):
            row.update(current_rank=float(current), future_rank=float(future), rank_change=float(future-current))
    return dict(four_arm_turns=rows, four_arm_summary=summaries, four_arm_contrasts=contrasts,
                four_arm_contrast_summary=contrast_summary, numerical_nulls=nulls,
                numerical_null_summary=null_summary, action_points=action_rows,
                action_interventions=reduced, action_ranking=action_summary)


def carrier_analysis(records, controls, sessions, repeats=2000, seed=1212):
    rows, audits = [], []
    for record in records:
        surgery = record["surgery"]
        common = dict(intervention_id=record["intervention_id"], session_id=record["session_id"], image_id=record["image_id"],
            carrier_id=record["carrier_id"], write_turn=record["write_turn"], position_type=record["position_type"],
            axis=record["axis"], alpha=float(record["alpha"]), selected_positions=record["selected"]["positions"],
            n_positions=len(record["selected"]["positions"]), selected_layers=surgery["selected_layers"],
            n_layers=len(surgery["selected_layers"]), logical_value_write_bytes=surgery["logical_value_write_bytes"],
            intervention_l2=surgery["intervention_l2"], intervention_max_abs=surgery["intervention_max_abs"],
            lexical_candidate=record["carrier"].get("candidate_word"),
            positions_unchanged=surgery["positions_unchanged"], token_ids_modified=surgery["token_ids_modified"],
            token_schedule_unchanged=record["token_schedule_sha256_before"] == record["token_schedule_sha256_after"])
        audits.append(dict(common, position_hash_unchanged=surgery["position_sha256_before"] == surgery["position_sha256_after"]))
        for future in record["future"]:
            rows.append(dict(common, target_turn=future["turn"], horizon=future["horizon"], **metric_row(future["metrics"])))
    summaries = summarize_group(rows, ["position_type", "axis", "alpha", "horizon"],
                               ["kl", "rms_logits", "argmax_flips"], repeats, seed)
    index = {(r["session_id"], r["carrier_id"], r["write_turn"], r["axis"], r["alpha"], r["horizon"], r["position_type"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate surgery condition")
    matched = []
    kv = []
    for row in rows:
        if row["position_type"] == "carrier":
            for control in ("prompt_control", "random_control"):
                key = (row["session_id"], row["carrier_id"], row["write_turn"], row["axis"], row["alpha"], row["horizon"], control)
                other = index.get(key)
                if other is None:
                    raise ValueError(f"Missing matched control: {key}")
                matched.append(dict(session_id=row["session_id"], image_id=row["image_id"], carrier_id=row["carrier_id"],
                    write_turn=row["write_turn"], axis=row["axis"], alpha=row["alpha"], horizon=row["horizon"], control=control,
                    carrier_kl=row["kl"], control_kl=other["kl"], delta_kl=row["kl"]-other["kl"],
                    carrier_rms=row["rms_logits"], control_rms=other["rms_logits"],
                    carrier_argmax_flips=row["argmax_flips"], control_argmax_flips=other["argmax_flips"],
                    carrier_intervention_l2=row["intervention_l2"], control_intervention_l2=other["intervention_l2"],
                    equal_intervention_l2=abs(row["intervention_l2"]-other["intervention_l2"]) <= 1e-9,
                    equal_position_count=row["n_positions"] == other["n_positions"],
                    equal_layer_set=row["selected_layers"] == other["selected_layers"],
                    equal_value_write_bytes=row["logical_value_write_bytes"] == other["logical_value_write_bytes"],
                    carrier_positions=row["selected_positions"], control_positions=other["selected_positions"],
                    position_overlap_count=len(set(row["selected_positions"]) & set(other["selected_positions"])),
                    logical_value_write_bytes=row["logical_value_write_bytes"]))
        if row["axis"] == "k":
            key = (row["session_id"], row["carrier_id"], row["write_turn"], "v", row["alpha"], row["horizon"], row["position_type"])
            other = index.get(key)
            if other is None:
                raise ValueError(f"Missing matched V surgery: {key}")
            kv.append(dict(session_id=row["session_id"], image_id=row["image_id"], carrier_id=row["carrier_id"],
                write_turn=row["write_turn"], position_type=row["position_type"], alpha=row["alpha"], horizon=row["horizon"],
                k_kl=row["kl"], v_kl=other["kl"], v_minus_k_kl=other["kl"]-row["kl"],
                k_rms=row["rms_logits"], v_rms=other["rms_logits"],
                k_argmax_flips=row["argmax_flips"], v_argmax_flips=other["argmax_flips"],
                k_intervention_l2=row["intervention_l2"], v_intervention_l2=other["intervention_l2"],
                equal_position_count=row["n_positions"] == other["n_positions"],
                equal_layer_set=row["selected_layers"] == other["selected_layers"],
                equal_value_write_bytes=row["logical_value_write_bytes"] == other["logical_value_write_bytes"],
                logical_value_write_bytes=row["logical_value_write_bytes"]))
    dose_groups = defaultdict(list)
    for row in rows:
        dose_groups[(row["session_id"], row["carrier_id"], row["write_turn"], row["position_type"], row["axis"], row["horizon"])].append(row)
    monotonicity = []
    for key, chosen in dose_groups.items():
        chosen.sort(key=lambda r: r["alpha"])
        values = [r["kl"] for r in chosen]
        tolerance = max(1e-12, max(abs(x) for x in values)*1e-7)
        monotonicity.append(dict(zip(("session_id", "carrier_id", "write_turn", "position_type", "axis", "horizon"), key),
            alphas=[r["alpha"] for r in chosen], kl_values=values,
            nondecreasing=all(b >= a-tolerance for a, b in zip(values, values[1:])),
            argmax_flips=[r["argmax_flips"] for r in chosen]))
    control_rows = []
    for control in controls:
        common = dict(session_id=control["session_id"], image_id=control["image_id"],
                      write_turn=control["write_turn"], control=control["control"])
        if "metrics" in control:
            control_rows.append(dict(common, target_turn=control["write_turn"], horizon=0,
                                     **metric_row(control["metrics"])))
        else:
            for future in control.get("future", []):
                control_rows.append(dict(common, target_turn=future["turn"], horizon=future["horizon"],
                                         **metric_row(future["metrics"])))
    matched_summary = summarize_group(matched, ["axis", "alpha", "horizon", "control"], ["delta_kl"], repeats, seed)
    return dict(surgery_points=rows, surgery_audits=audits, dose_summary=summaries,
                matched_controls=matched, matched_control_summary=matched_summary, kv_contrasts=kv,
                dose_monotonicity=monotonicity, carrier_control_metrics=control_rows,
                carrier_session_metadata=[{k:v for k,v in row.items() if k not in ("read_events", "canonical_turns")}
                                          for row in sessions])


def findings(four, carrier):
    contrasts = four["four_arm_contrasts"]
    actions = four["action_interventions"]
    alpha1 = [r for r in carrier["kv_contrasts"] if r["position_type"] == "carrier" and r["alpha"] == 1]
    paired = [r for r in carrier["matched_controls"] if r["alpha"] == 1]
    audits = carrier["surgery_audits"]
    zero = [r for r in carrier["surgery_points"] if r["alpha"] == 0]
    numerical_controls = [r for r in carrier["carrier_control_metrics"] if r["control"] in
                          ("source_base_reencoding_vs_original", "base_repeat_null")]
    amplitude_summary = []
    for position in sorted({r["position_type"] for r in audits}):
        for axis in ("k", "v"):
            selected = [r for r in audits if r["position_type"]==position and r["axis"]==axis and r["alpha"]==1]
            amplitude_summary.append(dict(position_type=position, axis=axis, alpha=1,
                intervention_l2=distribution(r["intervention_l2"] for r in selected),
                zero_amplitude_cases=[dict(session_id=r["session_id"],carrier_id=r["carrier_id"],
                                          write_turn=r["write_turn"]) for r in selected if r["intervention_l2"]==0]))
    rank_stats = []
    for metric in sorted({r["metric"] for r in four["action_ranking"]}):
        rows = [r for r in four["action_ranking"] if r["metric"] == metric]
        rank_stats.append(dict(metric=metric, spearman_current_future=spearman(
            [r["current_gain_mean"] for r in rows], [r["future_gain_mean"] for r in rows]),
            current_order=[r["action"] for r in sorted(rows, key=lambda r:r["current_rank"])],
            future_order=[r["action"] for r in sorted(rows, key=lambda r:r["future_rank"])]))
    return dict(
        four_arm=dict(n_sessions=len({r["session_id"] for r in contrasts}), n_turns=len(contrasts),
            effect_summary=four["four_arm_contrast_summary"], numerical_nulls=four["numerical_null_summary"],
            history_reference_rescue_negative=[r for r in contrasts if r["reference_history_rescue_kl"] < -1e-12],
            current_reference_rescue_negative=[r for r in contrasts if r["reference_read_rescue_kl"] < -1e-12]),
        actions=dict(rankings=rank_stats, n_write_interventions=len(actions),
            current_positive_future_negative=[r for r in actions if r["current_gain"] is not None and r["future_gain"] is not None
                                              and r["current_gain"] > 0 and r["future_gain"] < 0],
            current_negative_future_positive=[r for r in actions if r["current_gain"] is not None and r["future_gain"] is not None
                                              and r["current_gain"] < 0 and r["future_gain"] > 0],
            interpretation="v1 likelihood is diagnostic only; v2 fidelity remains distribution fidelity, not accuracy"),
        carrier=dict(n_sessions=len({r["session_id"] for r in audits}), n_interventions=len(audits),
            n_lexical_cases=len({(r["session_id"], r["carrier_id"], r["write_turn"]) for r in audits}),
            all_positions_unchanged=all(r["positions_unchanged"] and r["position_hash_unchanged"] for r in audits),
            all_token_schedules_unchanged=all(r["token_schedule_unchanged"] and not r["token_ids_modified"] for r in audits),
            all_matched_position_counts=all(r["equal_position_count"] for r in carrier["matched_controls"]),
            all_matched_layer_sets=all(r["equal_layer_set"] for r in carrier["matched_controls"]),
            all_matched_value_write_bytes=all(r["equal_value_write_bytes"] for r in carrier["matched_controls"]),
            matched_control_position_overlap_count=sum(r["position_overlap_count"] for r in carrier["matched_controls"]),
            position_counts=sorted({r["n_positions"] for r in audits}),
            logical_write_byte_values=sorted({r["logical_value_write_bytes"] for r in audits}),
            alpha1_amplitude_summary=amplitude_summary,
            amplitude_matching_scope="position/layer/write-byte matched, not perturbation-L2 matched",
            alpha0_max_abs_logits=max((r["max_abs_logits"] for r in zero), default=None),
            alpha0_argmax_flips=sum(r["argmax_flips"] for r in zero),
            numerical_control_max_abs_logits=max((r["max_abs_logits"] for r in numerical_controls), default=None),
            total_future_argmax_flips=sum(r["argmax_flips"] for r in carrier["surgery_points"]),
            total_future_content_argmax_flips=sum(r["content_argmax_flips"] for r in carrier["surgery_points"]),
            n_future_responses=len(carrier["surgery_points"]),
            n_dose_curves=len(carrier["dose_monotonicity"]),
            dose_nondecreasing_tolerance="max(1e-12, maximum absolute curve KL * 1e-7); descriptive only",
            dose_nonmonotone=[r for r in carrier["dose_monotonicity"] if not r["nondecreasing"]],
            carrier_alpha1_v_dominant=[r for r in alpha1 if r["v_minus_k_kl"] > 1e-12],
            carrier_alpha1_k_dominant=[r for r in alpha1 if r["v_minus_k_kl"] < -1e-12],
            carrier_alpha1_equal_count=sum(abs(r["v_minus_k_kl"]) <= 1e-12 for r in alpha1),
            carrier_alpha1_control_dominant=[r for r in paired if r["delta_kl"] < -1e-12],
            interpretation="lexical candidate locations; continuous reference-directed synthetic KV interpolation; output displacement from base is not correctness"))


def plot_mechanisms(four, carrier, output):
    if not four["four_arm_turns"] or not carrier["surgery_points"]:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
        "font.size":8, "pdf.fonttype":42, "axes.spines.top":False, "axes.spines.right":False,
        "legend.frameon":False, "axes.linewidth":.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), layout="constrained")
    blue, green, orange = "#526D82", "#7FA580", "#C5A56B"
    ax = axes[0, 0]
    session_ids = sorted({r["session_id"] for r in four["four_arm_turns"]})
    for sid in session_ids:
        values = [np.mean([r["kl"] for r in four["four_arm_turns"] if r["session_id"]==sid and r["arm"]==arm]) for arm in ARMS]
        ax.plot(range(4), values, color="#B9C3CA", lw=.7, alpha=.7, marker="o", ms=2)
    summary = [next(r for r in four["four_arm_summary"] if r["arm"]==arm and r["metric"]=="kl") for arm in ARMS]
    means = np.array([r["mean"] for r in summary])
    errors = np.array([[m-r["ci_lower"] for m,r in zip(means,summary)], [r["ci_upper"]-m for m,r in zip(means,summary)]])
    ax.errorbar(range(4), means, yerr=errors, color=blue, fmt="o", ms=4, capsize=2, lw=1)
    ax.set_xticks(range(4), ["Ref / Ref", "Ref / Base", "Base / Ref", "Base / Base"], rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("KL(reference || arm), nats/token")
    ax.set_title("History / current visual read", fontsize=9)

    ax = axes[0, 1]
    for phase, marker, color in (("text_prefill", "s", blue), ("early", "o", green), ("late", "^", orange)):
        selected = [r for r in four["action_ranking"] if r["phase"]==phase]
        ax.scatter([r["current_gain_mean"] for r in selected], [r["future_gain_mean"] for r in selected],
                   color=color, marker=marker, s=23, label=phase.replace("text_", ""))
        for row in selected:
            if abs(row["rank_change"]) >= 4:
                ax.annotate(f"g{row['layer_group']}", (row["current_gain_mean"],row["future_gain_mean"]),
                            xytext=(3,3), textcoords="offset points", fontsize=6)
    ax.axhline(0, color="#C7CDD1", lw=.7); ax.axvline(0, color="#C7CDD1", lw=.7)
    is_fidelity = any(r["metric"].startswith("KL_") for r in four["action_ranking"])
    metric = "KL fidelity gain" if is_fidelity else "fixed-base token logp change"
    ax.set_xlabel(f"Current {metric}"); ax.set_ylabel(f"Future {metric}")
    ax.set_title("Action ranking · not accuracy", fontsize=9); ax.legend(fontsize=6)

    ax = axes[1, 0]
    for position, color in (("carrier", blue), ("prompt_control", green), ("random_control", orange)):
        for axis, style, marker in (("k", "-", "o"), ("v", "--", "s")):
            chosen = [r for r in carrier["dose_summary"] if r["position_type"]==position and r["axis"]==axis
                      and r["horizon"]==1 and r["metric"]=="kl"]
            chosen.sort(key=lambda r:r["alpha"])
            ax.plot([r["alpha"] for r in chosen], [r["mean"] for r in chosen], color=color,
                    linestyle=style, marker=marker, ms=3, lw=1, label=f"{position.replace('_control','')} {axis.upper()}")
            if position == "carrier":
                ax.fill_between([r["alpha"] for r in chosen], [r["ci_lower"] for r in chosen],
                                [r["ci_upper"] for r in chosen], color=color, alpha=.08, linewidth=0)
    ax.set_xlabel("Reference-directed interpolation dose α")
    ax.set_ylabel("KL(base || intervened), nats/token")
    ax.set_title("Next-turn displacement · synthetic surgery", fontsize=9)
    ax.legend(fontsize=5.8, ncol=2, title="Carrier bands: image-cluster 95% CI", title_fontsize=5.8)

    ax = axes[1, 1]
    rows = [r for r in carrier["kv_contrasts"] if r["position_type"]=="carrier" and r["alpha"]==1]
    for horizon, color in ((1, blue),(2, green)):
        chosen = [r for r in rows if r["horizon"]==horizon]
        ax.scatter([r["k_kl"] for r in chosen], [r["v_kl"] for r in chosen], color=color, s=20,
                   marker="o" if horizon==1 else "s", alpha=.8, label=f"horizon {horizon}")
    maximum = max(max(r["k_kl"],r["v_kl"]) for r in rows)
    ax.plot([0, maximum], [0, maximum], color="#ADB5BD", lw=.8, linestyle=":")
    ax.set_xscale("symlog", linthresh=1e-7); ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_xlabel("K-surgery output KL"); ax.set_ylabel("V-surgery output KL")
    ax.set_title("Matched locations and write bytes · α = 1", fontsize=9)
    ax.legend(fontsize=6)
    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-.17, 1.05, letter, transform=ax.transAxes, weight="bold", fontsize=10)
    n_four=len({r["image_id"] for r in four["four_arm_turns"]})
    n_surgery=len({r["image_id"] for r in carrier["surgery_points"]})
    fig.suptitle(f"Development observations · four arms: {n_four} images · surgery: {n_surgery} images", fontsize=10)
    output.mkdir(parents=True, exist_ok=True)
    paths=[]
    version="fidelity_v2" if is_fidelity else "v1"
    for extension in ("png", "pdf"):
        path=output/f"mechanism_development_{version}.{extension}"
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");paths.append(str(path))
    plt.close(fig)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-dir",type=Path,required=True)
    parser.add_argument("--carrier-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    files=[args.mechanism_dir/"records.jsonl",args.carrier_dir/"surgery_records.jsonl",
           args.carrier_dir/"controls.jsonl",args.carrier_dir/"sessions.jsonl"]
    records, surgeries, controls, sessions = [load_jsonl(path) for path in files]
    four=four_arm_analysis(records,args.bootstrap,args.seed)
    carrier=carrier_analysis(surgeries,controls,sessions,args.bootstrap,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    for source in (four,carrier):
        for key,rows in source.items():
            if isinstance(rows,list):
                write_csv(args.output/f"{key}.csv",rows)
    report=findings(four,carrier)
    report.update(input_sha256={str(path.resolve()):sha256(path) for path in files},
        analysis_script_sha256=sha256(Path(__file__)),
        bootstrap=dict(unit="source image",repeats=args.bootstrap,seed=args.seed),
        scope="development numerical mechanisms; no test-set generalization or task accuracy claim")
    report["figures"]=plot_mechanisms(four,carrier,args.figures) if args.figures else []
    write_json(args.output/"summary.json",report)
    print(json.dumps(dict(four_arm_sessions=report["four_arm"]["n_sessions"],
        surgery_sessions=report["carrier"]["n_sessions"],
        surgery_interventions=report["carrier"]["n_interventions"],figures=len(report["figures"]))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
