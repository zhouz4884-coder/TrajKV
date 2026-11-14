"""Reproducible frozen external-suite report; no GPU, tuning or rep32 reads."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import analyze_cycle as shared
import analyze_external as external

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis_external_suite_v1"
BASELINE = "asym:periodic"
ORDER = ["full:base", "base2:base", "native4:base", "asym:uniform", "asym:periodic",
         "asym:table_future", "asym:probe_l7_t0.02", "asym:margin_write"]
LABELS = dict(zip(ORDER, ["Full", "Base2", "Native4", "Uniform", "Periodic", "Future table", "Probe", "Margin write"]))
CASES = {"docvqa": ("docvqa_eval16", "docvqa_external_eval16_remote_runner.json"),
         "gqa": ("gqa_eval16", "gqa_external_eval16_remote_runner.json"),
         "convbench": ("convbench_dev4", "convbench_dev4_remote_runner.json")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ci(rows, field):
    rows = sorted(rows, key=lambda r: (r["session_id"], int(r["reference_turn"])))
    return shared.cluster_bootstrap(((r["image_id"], float(r[field])) for r in rows), repeats=2000, seed=1212)


def paired(rows, method, baseline, field="score"):
    index = {(r["session_id"], int(r["reference_turn"]), r["variant"]): r for r in rows}
    keys = sorted((s, t) for s, t, v in index if v == baseline)
    points, details = [], []
    for sid, turn in keys:
        a, b = index[(sid, turn, method)], index[(sid, turn, baseline)]
        delta = float(a[field]) - float(b[field])
        points.append((a["image_id"], delta))
        details.append({"session_id": sid, "turn": turn, "image_id": a["image_id"],
                        "method": method, "baseline": baseline, "field": field, "delta": delta,
                        "method_prediction": a["prediction"], "baseline_prediction": b["prediction"],
                        "reference_answers": a["reference_answers"]})
    result = shared.cluster_bootstrap(points, repeats=2000, seed=1212)
    return {"method": method, "baseline": baseline, "field": field, **result,
            "improved_questions": sum(r["delta"] > 0 for r in details),
            "worsened_questions": sum(r["delta"] < 0 for r in details)}, details


def bare_question(question):
    prefix = "Answer with a short answer only. Do not explain.\nQuestion: "
    return question[len(prefix):] if question.startswith(prefix) else question


def provenance(samples, rows, dataset):
    bench = ROOT.parent / "benchmark_samples"
    hashes = {}
    def image_hash(path):
        path = Path(path)
        if path not in hashes:
            hashes[path] = sha(path)
        return hashes[path]
    problems = []
    for sample in samples:
        for path, expected in zip(sample["images"], sample["hashes"]):
            if not Path(path).is_file() or image_hash(path) != expected:
                problems.append({"session_id": sample["id"], "issue": "image_missing_or_hash_mismatch", "path": path})
    raw = {}
    if dataset == "docvqa":
        source = bench / "docvqa_samples.jsonl"
        for row in jsonl(source):
            path = bench / "docvqa_images" / Path(row["image_path"].replace("\\", "/")).name
            if path.exists():
                raw.setdefault((image_hash(path), row["question"]), row["answer"])
    elif dataset == "gqa":
        source = bench / "gqa_flat_samples.jsonl"
        raw = {(str(x["image_id"]), x["question"]): x["answer"] for x in jsonl(source)}
    else:
        source = ROOT.parent / "VisContract_P14_2/results/p104_convbench_preflight/convbench_samples.jsonl"
        for row in jsonl(source):
            for t in row["turn_questions"]:
                raw[(row["sample_id"], t["question"])] = t["gold_answers"]
    references = external.load_references([ROOT / f"configs/data/{dataset}_references.jsonl"])
    matched = 0
    exact_question_matches = 0
    question_repairs = []
    malformed = []
    observed = {(r["session_id"], int(r["reference_turn"])) for r in rows}
    for sample in samples:
        for number, question in enumerate(sample["questions"], 1):
            if (sample["id"], number) not in observed:
                continue
            identity = sample["hashes"][0] if dataset == "docvqa" else (
                str(sample["source_image_id"]) if dataset == "gqa" else sample["id"])
            prepared_question = bare_question(question)
            original = raw.get((identity, prepared_question))
            if original is None:
                repair_candidates = [(q, a) for (iid, q), a in raw.items()
                                     if iid == identity and q.replace("\ufffd\ufffd", "’") == prepared_question]
                if len(repair_candidates) == 1:
                    old_question, original = repair_candidates[0]
                    question_repairs.append({"session_id": sample["id"], "turn": number,
                                             "source_question": old_question, "prepared_question": prepared_question,
                                             "repair": "double replacement character to curly apostrophe; observed frozen curation, no scoring change"})
            else:
                exact_question_matches += 1
            expected = references.get((sample.get("native_session_id", sample["id"]), number))
            original_list = original if isinstance(original, list) else [original]
            if original is None or expected != original_list:
                problems.append({"session_id": sample["id"], "turn": number,
                                 "issue": "local_source_reference_or_question_mismatch", "expected": expected,
                                 "source": original})
            else:
                matched += 1
            suspicious = [x for x in (expected or []) if not isinstance(x, str) or
                          not x.strip() or x.strip().lower() in {"nan", "null"} or "\ufffd" in x]
            if suspicious or "\ufffd" in question:
                malformed.append({"session_id": sample["id"], "turn": number,
                                  "question_encoding_replacement": "\ufffd" in question,
                                  "suspicious_references": suspicious})
    return {"source": str(source), "source_sha256": sha(source),
            "observed_questions_matched_to_local_source": matched,
            "questions_matching_source_verbatim": exact_question_matches,
            "observed_question_encoding_repairs": question_repairs,
            "image_files_verified": len(samples), "problems": problems,
            "encoding_or_reference_flags": malformed,
            "limit": "local curated source identity verified; not a fresh audit of hidden official annotations or all acceptable alternatives"}


def analyze_suite():
    summaries, tables, diagnostics, audits, paired_tables, flip_rows = {}, [], [], {}, [], []
    scored_by_dataset = {}
    for dataset, (folder, sample_name) in CASES.items():
        run = ROOT / "results/external_suite_gpu1_v1" / folder
        sample_path = ROOT / "configs/data" / sample_name
        external.main(["--dataset", dataset, "--records", str(run / "records.jsonl"),
                       "--references", str(ROOT / f"configs/data/{dataset}_references.jsonl"),
                       "--samples", str(sample_path), "--output", str(OUT / dataset), "--baseline", BASELINE])
        summary = json.loads((OUT / dataset / "summary.json").read_text(encoding="utf-8-sig"))
        summaries[dataset] = summary
        scored = csv_rows(OUT / dataset / "scored_turns.csv")
        scored_by_dataset[dataset] = scored
        records = jsonl(run / "records.jsonl")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8-sig"))
        samples = json.loads(sample_path.read_text(encoding="utf-8-sig"))
        selected = set(manifest["selected_ids"])
        samples = [s for s in samples if s["id"] in selected]
        smap = {s["id"]: s for s in samples}
        cap = int(manifest["arguments"]["max_tokens"])
        limit = int(manifest["arguments"]["turns"])
        record_keys = [(r["session_id"], r["method"], r["policy"]) for r in records]
        audit = {"records": len(records), "completion_records_added": json.loads((run / "complete.json").read_text(encoding="utf-8-sig"))["records_added"],
                 "sessions": len(selected), "duplicate_record_keys": len(record_keys) - len(set(record_keys)),
                 "expected_question_turns_per_method": sum(min(limit, len(s["questions"])) for s in samples),
                 "unasked_source_questions_due_turn_cap": sum(max(0, len(s["questions"]) - limit) for s in samples),
                 "budget_violations": sum(r.get("read_budget_ok") is False for r in records),
                 "ledger_not_verified": sum(r.get("read_ledger_exact") is not True for r in records),
                 "sample_manifest_sha256": sha(sample_path), "records_sha256": sha(run / "records.jsonl"),
                 "invalid_turn_counts": []}
        for r in records:
            expected = min(limit, len(smap[r["session_id"]]["questions"]))
            if len(r["turns"]) != expected:
                audit["invalid_turn_counts"].append({"session_id": r["session_id"], "method": r["method"],
                                                      "policy": r["policy"], "expected": expected, "observed": len(r["turns"])})
        audit["provenance"] = provenance(samples, scored, dataset)
        if dataset in ("docvqa", "gqa"):
            development = json.loads((ROOT / f"configs/data/{dataset}_dev32_runner.json").read_text(encoding="utf-8-sig"))
            audit["external_dev32_image_hash_overlap"] = sorted(
                {h for s in samples for h in s["hashes"]} & {h for s in development for h in s["hashes"]})
        audit["initial_native_logit_disagreements"] = sum(
            len({r["initial_logits_sha256"] for r in records if r["session_id"] == sid}) != 1 for sid in selected)
        audits[dataset] = audit
        for variant in [v for v in ORDER if any(r["variant"] == v for r in scored)]:
            selected_rows = [r for r in scored if r["variant"] == variant]
            primary = next(x for x in summary["method_summary"] if x["variant"] == variant and x["scope"] == "all_scored")
            contrast = next((x for x in summary["paired_summary"] if x["variant"] == variant and x["scope"] == "all_scored"), None)
            auxiliary_field = ("case_whitespace_exact_aux" if dataset == "gqa" else
                               "anls_strict_threshold_aux" if dataset == "docvqa" else None)
            aux = ci(selected_rows, auxiliary_field) if auxiliary_field else {}
            tables.append({"dataset": dataset, "variant": variant, "metric": summary["primary_metric"],
                           "n_images": primary["n_images"], "n_questions": primary["n_questions"],
                           "primary_score": primary["mean"], "primary_ci_lower": primary["ci_lower"],
                           "primary_ci_upper": primary["ci_upper"], "exact_accuracy": primary["exact_accuracy"],
                           "delta_vs_periodic": contrast["mean"] if contrast else 0,
                           "delta_ci_lower": contrast["ci_lower"] if contrast else 0,
                           "delta_ci_upper": contrast["ci_upper"] if contrast else 0,
                           "auxiliary_metric": auxiliary_field, "auxiliary_score": aux.get("mean"),
                           "auxiliary_ci_lower": aux.get("ci_lower"), "auxiliary_ci_upper": aux.get("ci_upper"),
                           "threshold_boundary_questions": primary["threshold_boundary_questions"]})
            turns = [t for r in records if r["method"] + ":" + r["policy"] == variant for t in r["turns"]]
            case_only = sum(float(x["score"]) == 0 and any(x["prediction"].lower() == g.lower()
                             for g in json.loads(x["reference_answers"])) for x in selected_rows) if dataset == "gqa" else None
            diagnostics.append({"dataset": dataset, "variant": variant, "questions": len(turns), "token_cap": cap,
                                "cap_reached": sum(len(t["token_ids"]) >= cap for t in turns),
                                "cap_reached_and_forced_closure": sum(len(t["token_ids"]) >= cap and t["forced_chat_closure"] for t in turns),
                                "forced_chat_closures": sum(t["forced_chat_closure"] for t in turns),
                                "empty_predictions": sum(not t["answer"].strip() for t in turns),
                                "gqa_case_only_recovered": case_only,
                                "gqa_case_whitespace_aux_correct": sum(float(x.get("case_whitespace_exact_aux", 0)) for x in selected_rows) if dataset == "gqa" else None})
        if dataset in ("docvqa", "gqa"):
            candidate = "asym:table_future" if dataset == "docvqa" else "asym:probe_l7_t0.02"
            for comparator in [v for v in ORDER if v != candidate and any(r["variant"] == v for r in scored)]:
                fields = ["score"] + (["case_whitespace_exact_aux"] if dataset == "gqa" else [])
                for field in fields:
                    estimate, details = paired(scored, candidate, comparator, field)
                    paired_tables.append({"dataset": dataset, **estimate})
                    flip_rows.extend({"dataset": dataset, **x} for x in details if x["delta"] != 0)
    for name, rows in (("external_summary", tables), ("format_diagnostics", diagnostics),
                       ("additional_paired_contrasts", paired_tables), ("changed_question_examples", flip_rows)):
        shared.write_csv(OUT / (name + ".csv"), rows)
    positives = [r for r in flip_rows if r["dataset"] == "docvqa" and r["baseline"] == "base2:base"
                 and r["method"] == "asym:table_future" and r["field"] == "score" and r["delta"] > 0]
    shared.write_csv(OUT / "docvqa_future_vs_base2_positive_examples.csv", positives)
    result = {"status": "COMPLETE_FROZEN_EXTERNAL_ANALYSIS", "summaries": summaries,
              "table": tables, "format_diagnostics": diagnostics, "audits": audits,
              "additional_paired_contrasts": paired_tables,
              "root_replication32_outcomes_read": False, "parameters_or_samples_changed": False,
              "new_inference_or_judging_calls": 0,
              "bootstrap": {"cluster": "source image", "repeats": 2000, "seed": 1212,
                            "scope": "exploratory intervals, no multiplicity-adjusted significance claim"},
              "reference_validity_findings": [{
                  "dataset": "convbench", "session_id": "convbench_37", "turns": [2, 3],
                  "issue": "T2 reference moves knight f3 to e5; T3 previous-move reference switches to bishop a4 to b5, inconsistent with T1 pawn list",
                  "reference_file_sha256": sha(ROOT / "configs/data/convbench_references.jsonl"),
                  "action": "retain unchanged lexical diagnostic; no official semantic quality claim or posthoc sample filtering"}],
              "script_sha256": sha(Path(__file__))}
    shared.write_json(OUT / "external_review.json", result)
    return result


def make_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1, 1.1]}, layout="constrained")
    comparisons = ["full:base", "native4:base", "base2:base", "asym:uniform", BASELINE]
    pairs = [next(x for x in result["additional_paired_contrasts"] if x["dataset"] == "docvqa" and x["baseline"] == v) for v in comparisons]
    ax = axes[0]
    for i, row in enumerate(pairs):
        mean, low, high = (100 * row[k] for k in ("mean", "ci_lower", "ci_upper"))
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", color="#0F4D92", capsize=3, ms=5)
    ax.axvline(0, color="#767676", lw=1, ls="--")
    ax.set_yticks(range(len(pairs)), ["Future − " + LABELS[v] for v in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel("DocVQA ANLS difference (percentage points)")
    ax.set_title("A  Gains depend on the comparator", loc="left", fontsize=12, weight="bold")
    ax.set_xlim(-7, 14)
    ax = axes[1]
    gqa = {r["variant"]: r for r in result["table"] if r["dataset"] == "gqa"}
    for i, v in enumerate(ORDER):
        raw, auxiliary = 100 * gqa[v]["primary_score"], 100 * gqa[v]["auxiliary_score"]
        ax.plot([raw, auxiliary], [i, i], color="#CFCECE", lw=2)
        ax.plot(raw, i, "o", color="#767676", ms=5, label="Raw exact (primary)" if i == 0 else None)
        ax.plot(auxiliary, i, "s", color="#3775BA", ms=5, label="Lower + strip (diagnostic)" if i == 0 else None)
    ax.set_yticks(range(len(ORDER)), [LABELS[v] for v in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("GQA answer matching (%)")
    ax.set_title("B  Answer case dominates raw exact", loc="left", fontsize=12, weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(.98, -.035), ncol=2, fontsize=9)
    fig.text(.01, -.025, "Frozen 16-image subsets · paired image-cluster 95% intervals · exploratory comparisons", fontsize=9, color="#4D4D4D")
    target = ROOT / "figures/external_validation_v1"
    files = []
    for ext in ("png", "pdf"):
        path = target.with_suffix("." + ext)
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = analyze_suite()
    files = make_figure(result)
    result["figures"] = files
    shared.write_json(OUT / "external_review.json", result)
    print(json.dumps({"status": result["status"], "figures": files,
                      "audits": {k: {"records": v["records"], "source_questions_verified": v["provenance"]["observed_questions_matched_to_local_source"],
                                     "problems": len(v["provenance"]["problems"]), "flags": len(v["provenance"]["encoding_or_reference_flags"])}
                                 for k, v in result["audits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
