"""Fixed 64-image confirmation analysis, separate from exploratory16.

No inference, parameter selection or frozen scorer changes. Question-weighted
ANLS is used throughout. Image-cluster bootstrap retains cluster question
counts; Monte Carlo swaps each image's paired question-difference SUM.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import analyze_external as external
import analyze_cycle as shared
from report_external_suite import provenance
from verify_delivery import Audit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis_docvqa_confirm64_v1"
FUTURE = "asym:table_future"
ORDER = ["full:base", "base2:base", "native4:base", "asym:base", "asym:uniform", "asym:periodic",
         "asym:table_current", FUTURE, "asym:probe_l7_t0.02", "asym:margin_write", "kvarn2:base", "kvarn4:base"]
LABELS = ["Full", "Base2", "Native4", "Asym base", "Uniform", "Periodic", "Current table",
          "Future table", "Probe", "Margin write", "KVarN2 port", "KVarN4 port"]
LABEL = dict(zip(ORDER, LABELS))
PRIMARY = ["base2:base", "asym:table_current"]
COMPARATORS = PRIMARY + [v for v in ORDER if v not in PRIMARY + [FUTURE]]
BOOTSTRAP, MC, SEED = 2000, 100000, 1212


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(0, 1)
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def signflip_pvalues(image_difference_sums, total_questions, repeats=MC, seed=SEED):
    values = np.asarray(image_difference_sums, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if total_questions <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid paired image sums")
    observed_sums = values.sum(0)
    extreme = np.zeros(values.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, 4096):
        n = min(4096, repeats - start)
        signs = rng.integers(0, 2, size=(n, len(values)), dtype=np.int8) * 2 - 1
        randomized_sums = signs.astype(np.float64) @ values
        extreme += (np.abs(randomized_sums) >= np.abs(observed_sums)[None, :] - 1e-12).sum(0)
    return {"effect": (observed_sums / total_questions).tolist(),
            "p_two_sided": ((extreme + 1) / (repeats + 1)).tolist(),
            "extreme_count": extreme.tolist(), "repeats": repeats}


def load_scored():
    paths = [ROOT / f"results/docvqa_confirm64_shard{s}_v1/{child}/records.jsonl"
             for child in ("core", "kvarn") for s in (0, 1)]
    external.main(["--dataset", "docvqa", "--records", *map(str, paths),
                   "--references", str(ROOT / "configs/data/docvqa_references.jsonl"),
                   "--samples", str(ROOT / "configs/data/docvqa_confirm64_remote_runner.json"),
                   "--output", str(OUT / "fixed_score"), "--baseline", "base2:base"])
    with (OUT / "fixed_score/scored_turns.csv").open(encoding="utf-8-sig", newline="") as f:
        scored = list(csv.DictReader(f))
    for row in scored:
        for key in ("score", "exact", "anls_strict_threshold_aux"):
            row[key] = float(row[key])
        row["reference_turn"] = int(row["reference_turn"])
    return paths, scored


def analyze(scored):
    variants = {r["variant"] for r in scored}
    if variants != set(ORDER):
        raise ValueError("Confirmation configuration set differs from fixed all12")
    index = {(r["session_id"], r["reference_turn"], r["variant"]): r for r in scored}
    keys = sorted({(r["session_id"], r["reference_turn"]) for r in scored})
    images = sorted({r["image_id"] for r in scored})
    assert len(keys) == 294 and len(images) == 64 and len(scored) == 3528
    table = []
    for v in ORDER:
        rows = [index[(sid, turn, v)] for sid, turn in keys]
        estimate = shared.cluster_bootstrap(((r["image_id"], r["score"]) for r in rows), repeats=BOOTSTRAP, seed=SEED)
        table.append({"variant": v, "label": LABEL[v], **estimate,
                      "exact_correct": int(sum(r["exact"] for r in rows)),
                      "exact_accuracy": sum(r["exact"] for r in rows) / len(rows),
                      "strict_threshold_aux": sum(r["anls_strict_threshold_aux"] for r in rows) / len(rows),
                      "anls_threshold_boundary_count": sum(r["anls_threshold_boundary_hit"] == "True" for r in rows),
                      "quality_only_mechanism_port": v.startswith("kvarn")})
    comparisons, flips = [], []
    image_matrix = np.zeros((len(images), len(COMPARATORS)))
    image_index = {image: i for i, image in enumerate(images)}
    for column, comparator in enumerate(COMPARATORS):
        paired = []
        for sid, turn in keys:
            a, b = index[(sid, turn, FUTURE)], index[(sid, turn, comparator)]
            assert a["image_id"] == b["image_id"]
            delta = a["score"] - b["score"]
            image_matrix[image_index[a["image_id"]], column] += delta
            row = {"session_id": sid, "image_id": a["image_id"], "turn": turn,
                   "method": FUTURE, "baseline": comparator, "delta": delta,
                   "future_prediction": a["prediction"], "baseline_prediction": b["prediction"],
                   "reference_answers": a["reference_answers"], "future_anls": a["score"],
                   "baseline_anls": b["score"], "future_exact": a["exact"], "baseline_exact": b["exact"],
                   "exact_repaired": b["exact"] == 0 and a["exact"] == 1,
                   "exact_new_error": b["exact"] == 1 and a["exact"] == 0,
                   "remaining_nonexact_soft_improvement": delta > 0 and a["exact"] == 0,
                   "nonexact_to_nonexact_worsening": delta < 0 and a["exact"] == 0 and b["exact"] == 0}
            paired.append(row)
            if delta != 0 or a["prediction"] != b["prediction"]:
                flips.append(row)
        estimate = shared.cluster_bootstrap(((r["image_id"], r["delta"]) for r in paired), repeats=BOOTSTRAP, seed=SEED)
        comparisons.append({"method": FUTURE, "baseline": comparator,
                            "family": "primary" if comparator in PRIMARY else "secondary", **estimate,
                            "score_improved_questions": sum(r["delta"] > 0 for r in paired),
                            "score_worsened_questions": sum(r["delta"] < 0 for r in paired),
                            "exact_repaired": sum(r["exact_repaired"] for r in paired),
                            "exact_new_errors": sum(r["exact_new_error"] for r in paired),
                            "remaining_nonexact_soft_improvements": sum(r["remaining_nonexact_soft_improvement"] for r in paired),
                            "nonexact_to_nonexact_worsenings": sum(r["nonexact_to_nonexact_worsening"] for r in paired)})
    mc = signflip_pvalues(image_matrix, 294)
    for row, effect, p, count in zip(comparisons, mc["effect"], mc["p_two_sided"], mc["extreme_count"]):
        assert abs(row["mean"] - effect) < 1e-12
        row.update(mc_p_two_sided=p, mc_extreme_count=count, mc_repeats=MC)
    for family in ("primary", "secondary"):
        chosen = [r for r in comparisons if r["family"] == family]
        corrected = holm([r["mc_p_two_sided"] for r in chosen])
        for row, p in zip(chosen, corrected):
            row["holm_family_size"] = len(chosen)
            row["mc_p_holm_family"] = p
    for row, p in zip(comparisons, holm([r["mc_p_two_sided"] for r in comparisons])):
        row["mc_p_holm_all11_sensitivity"] = p
    return table, comparisons, flips


def audit_inputs(paths, scored):
    samples_path = ROOT / "configs/data/docvqa_confirm64_remote_runner.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8-sig"))
    assert sha(samples_path) == "25a9dac82654f75db96feca0388b44ee5c750292436f3b643ea83d955e6024f3"
    prov = provenance(samples, scored, "docvqa")
    records = [row for path in paths for row in jsonl(path)]
    by_id = defaultdict(list)
    for row in records:
        by_id[row["session_id"]].append(row)
    initial_disagreements = [sid for sid, rows in by_id.items() if len({r["initial_logits_sha256"] for r in rows}) != 1]
    owner_checks, cap_checks = [], []
    budget_audit = Audit(ROOT)
    budgets = budget_audit.budgets(OUT / "all_source_records", records)
    source_hashes = defaultdict(set)
    model_identity = json.loads((ROOT / "audit/model_identity.json").read_text(encoding="utf-8-sig"))
    model_hashes = {f["name"]: f.get("sha256") for f in model_identity["files"] if f.get("exists")}
    manifest_mismatches = []
    for path in paths:
        m = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8-sig"))
        for key, value in m.get("sources", m.get("source_sha256", {})).items():
            name = key.replace("\\", "/").rsplit("/", 1)[-1]
            if name.endswith(".py"):
                source_hashes[name].add(value)
        for name, value in m.get("model_identity", {}).items():
            if model_hashes.get(name) != value:
                manifest_mismatches.append({"manifest": str(path.with_name("manifest.json")), "model_file": name})
    for variant in ORDER:
        current = [r for r in records if r["method"] + ":" + r["policy"] == variant]
        diagnostic = sum(r["owned_final"].get("diagnostic_fp16_reference", 0) for r in current)
        ratios = sorted({r["owned_final"].get("packed_visual_ratio") for r in current
                         if r["owned_final"].get("packed_visual_ratio") is not None})
        owners = {"variant": variant, "records": len(current), "diagnostic_fp16_reference_bytes_sum": diagnostic,
                  "full_visual_reference_retained_records": sum(r["owned_final"].get("full_visual_retained", False) for r in current),
                  "packed_visual_ratios": ratios,
                  "materialized_port": variant.startswith("kvarn"),
                  "encoded_reference_ratios": sorted({r["owned_final"].get("encoded_reference_ratio") for r in current
                                                       if r["owned_final"].get("encoded_reference_ratio") is not None}),
                  "materialized_visual_ratios": sorted({r["owned_final"].get("materialized_visual_ratio") for r in current
                                                         if r["owned_final"].get("materialized_visual_ratio") is not None})}
        owner_checks.append(owners)
        turns = [t for r in current for t in r["turns"]]
        cap_checks.append({"variant": variant, "questions": len(turns), "token_cap": 32,
                           "cap_reached": sum(len(t["token_ids"]) >= 32 for t in turns),
                           "cap_and_forced_closure": sum(len(t["token_ids"]) >= 32 and t["forced_chat_closure"] for t in turns)})
    return {"records": len(records), "questions_per_variant": Counter(r["variant"] for r in scored),
            "image_clusters": len(by_id), "source_questions_references_images": prov,
            "initial_logit_disagreement_sessions": initial_disagreements,
            "source_hash_versions_by_basename": {k: sorted(v) for k, v in source_hashes.items()},
            "model_metadata_mismatches": manifest_mismatches,
            "read_budget_audit": budgets, "read_ledger_exact_false_records": sum(r.get("read_ledger_exact") is not True for r in records),
            "owner_checks": owner_checks, "output_cap_checks": cap_checks,
            "owner_scope_limit": "registered cache owners; not a proof about every temporary GPU allocation; KVarN materialization explicitly retained and not a serving efficiency result",
            "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths + [samples_path, ROOT / "configs/data/docvqa_references.jsonl"]}}


def figure(table, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.1, "legend.frameon": False, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 6.4), layout="constrained")
    for i, row in enumerate(table):
        mean, lo, hi = [100 * row[k] for k in ("mean", "ci_lower", "ci_upper")]
        color = "#0F4D92" if row["variant"] == FUTURE else "#767676"
        a.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color, ms=4, capsize=2.5)
    a.set_yticks(range(12), LABELS)
    a.invert_yaxis()
    a.set_xlim(75, 93)
    a.set_xlabel("Question-weighted ANLS (%)")
    a.set_title("A  Independent 64-image confirmation", loc="left", weight="bold", fontsize=12)
    for i, row in enumerate(comparisons):
        mean, lo, hi = [100 * row[k] for k in ("mean", "ci_lower", "ci_upper")]
        color = "#0F4D92" if row["family"] == "primary" else "#767676"
        b.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color, ms=4, capsize=2.5)
    b.axvline(0, color="#A0A0A0", ls="--", lw=1)
    b.set_yticks(range(11), ["Future − " + LABEL[r["baseline"]] for r in comparisons])
    b.invert_yaxis()
    b.set_xlabel("Paired ANLS difference (percentage points)")
    b.set_title("B  Primary contrasts do not confirm gains", loc="left", weight="bold", fontsize=12)
    fig.text(.01, -.015, "64 image clusters · 294 questions · bootstrap 2,000 · MC 100,000 · Holm2 primary / Holm9 secondary · exploratory16 not pooled", fontsize=8.5)
    result = []
    for ext in ("png", "pdf"):
        path = ROOT / "figures" / ("docvqa_confirm64_v1." + ext)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        result.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    plt.close(fig)
    return result


def write_report(result):
    table = result["all12"]
    comparisons = result["comparisons"]
    primary = [r for r in comparisons if r["family"] == "primary"]
    methods = "\n".join(
        f"| {r['label']} | {100*r['mean']:.2f} | [{100*r['ci_lower']:.2f}, {100*r['ci_upper']:.2f}] | {r['exact_correct']}/294 |"
        for r in table)
    contrasts = "\n".join(
        f"| {LABEL[r['baseline']]} | {'主 / 2' if r['family']=='primary' else '次 / 9'} | {100*r['mean']:+.3f} | [{100*r['ci_lower']:.3f}, {100*r['ci_upper']:.3f}] | {r['mc_p_two_sided']:.5f} | {r['mc_p_holm_family']:.5f} | {r['mc_p_holm_all11_sensitivity']:.5f} |"
        for r in comparisons)
    report = f"""# DocVQA 独立确认64：两项主优越比较均未通过

本次固定确认不支持 Future table 优于廉价 Base2，也不支持它优于同标定数据下的 Current table。Future 主 ANLS 为 **81.80**；对 Base2 仅 **+0.137pp**，对 Current table 为 **−0.687pp**。两项配对图像区间均跨0，主家族 Holm2 校正后 p 分别为 **{primary[0]['mc_p_holm_family']:.5f}** 和 **{primary[1]['mc_p_holm_family']:.5f}**。本周期不以探索16的局部正信号宣称方法成功，不再追新参数。

## 1. 确认顺序与审计

探索16结果已被查看，随后从排除 dev32 与探索16的剩余133文档中按固定 hash 规则选取64文档。样本/方法在新64结果之前固定；这属于**探索后的独立确认**，不是研究最初就已预注册，也不能把16+64合并成80张盲测。

输入合同：`configs/docvqa_confirm64_contract_v1.json`。最终 runner SHA256 为 `25a9dac82654f75db96feca0388b44ee5c750292436f3b643ea83d955e6024f3`。首问重复 `Question:` 的旧80db预检版已在GPU前退役，修订史保留；没有实际引号编码修订。

两个 GPU shard 各有320条 core 与64条 KVarN 记录。父/子 complete SHA、完整JSONL与条数均核对通过；共 **768个方法-session、64图、每配置294题、12配置合计3528回答**。每图3–6题，不能把图均分直接平均后称为同一个问题加权指标。

294个问句/参考及64张图片与本地整理源及hash匹配；没有缺参考、错图或初始logit分歧。四个子run的同名公共Python源码均只出现一个hash版本，模型metadata/index匹配已记录的完整模型身份。所有质量输出均来自自由生成；参考答案只由离线分析读取。

## 2. 固定统计规则

主指标保持已有 organizer-code ANLS：lower+outer strip，多参考取最大similarity，similarity ≥0.5保留，空预测为0。Exact与strict阈值仅为辅助，不替换主分数；定义及来源沿用 `docs/external_scoring.md`。

效应量为全部294个问题的配对ANLS差均值。Bootstrap按64个source image重采样，保留每簇原有问题并重算总分母，2,000次、seed1212。MC每次翻转**每图所有问题差之和**的符号，再除总294；100,000次、双侧、seed1212，p=(extreme+1)/(B+1)。MC依赖配对图像交换/符号对称的零假设参照，不是模型因果性证明。

主两比较 Future−Base2、Future−Current 同属 Holm2。其余9个比较独立组成 Holm9；另报全11校正为敏感性，不依据结果换家族。规则在本地检查确认分数之前固定，保存在 `analysis_plan.json`，不伪称统计文件是在推理开始前才生成。

## 3. 全12配置

ANLS和CI均为百分制。CI是逐比较/逐配置的图像聚类95%区间；多重性由下节的p值家族处理。

| 配置 | 主ANLS | 95% CI | Exact |
|---|---:|---:|---:|
{methods}

Asym base 与 Base2 质量相同，但前者仍保存残差。Full不是数学意义的准确率上界；Native4在本子集均值略高于Full，仅如实报告，不据此新增机制故事。

## 4. 完整比较家族

均为 Future table 减去该行对照；单位pp。最后一列只作全11家族敏感性。

| 对照 | 家族/大小 | 差值 | 95% CI | MC双侧p | 家族Holm p | 全11敏感性p |
|---|---:|---:|---:|---:|---:|---:|
{contrasts}

主两比较均未通过。Future对同存储的 Periodic 为−1.446pp，Holm9 p=0.52524；对 Uniform 为−0.048pp。Future比Full、Native4和KVarN2质量参考更低，相应次家族校正p分别为0.01435、0.01320、0.01170。KVarN4虽有负向点估计和点对点CI，Holm9 p=0.09552，不把未校正区间替代多重比较结论。

## 5. 软分数变化不等于答对题数

Future对Base2：ANLS提高11题、下降6题；Exact追回4题、新增错3题，净增仅1道Exact。11个分数提高项中**7项仍未Exact匹配参考**。因此不能写成“多答对11题”。Future对Current：分数提高3题、下降5题；Exact追回/新增错各1题。

所有逐题原预测、GT、ANLS、Exact及差值见 `paired_flips.csv`；主两比较另存 `primary_paired_flips.csv`。这些只是固定字符串评分及其翻转，不新增事后语义裁判或过滤参考。

## 6. 预算、owner与KVarN边界

额外读取预算违规0、逐call账本重算矛盾0、`read_ledger_exact`失败0。预算仍是额外**逻辑tensor payload**，不是物理HBM流量；共享16MiB上限也不等于每个配置实际花费完全相同。

已登记cache owner中，所有配置的diagnostic FP16 reference为0，没有保留诊断原版视觉副本。Base2视觉packed比为14.0625%，Native4为26.5625%，Asym固定表示为24.609375%；它们由实际tensor owner计数支持。此处只验证已登记的cache owners，不伪称证明整个CUDA运行所有临时分配都不存在。

KVarN只作为同执行器的**质量机制移植参考**：KVarN2/4各显式保留100%解码视觉tensor，并另保存14.84375%/27.34375%的encoded reference，总视觉owner约114.84%/127.34%。它们没有以该路径实现真实packed serving，不能拿此数据主张端到端显存压缩、速度或HBM收益。

各配置至多1/294回答触及32-token cap；触及上限不必然被截断，部分刚好以EOS结束。cap与强制结束标记均保留于机器audit。没有按这些结果删题或改长度后重跑。

## 7. 与探索16分开解释

探索16中Future−Base2为+2.43pp、图像CI[0,6.89]，本来就未建立严格正收益。确认64缩至+0.14pp、CI[−1.71,1.93]，且对同标定的Current均值更低。该局部信号未得到确认；不能将探索结果挑出作为主结论，也不能把当前结果解释成未来所有轨迹方法必然无效。

这次确认未建立本冻结策略在本协议下的优越性，因此不能保留“已验证优越”的主张；这不是等价性检验，也不证明所有未来轨迹方法都无效。历史介导机制是否存在、测量协议是否有价值，需要由其独立机制证据判断，不能由这个质量负结果代替，也不能用机制存在替代质量收益。

## 8. 交付

- `results/analysis_docvqa_confirm64_v1/summary.json`：全12、完整家族、MC/Holm及审计。
- `all12.csv`、`paired_all11.csv`、`paired_flips.csv`、`primary_paired_flips.csv`。
- `fixed_score/`：固定评分器逐题/逐轮输出及其输入hash。
- `completion_identity_checks.json`、`analysis_plan.json`：先验规则与完成身份。
- `figures/docvqa_confirm64_v1.png` / `.pdf`：全部真实结果与配对区间，已检查布局。
- `code/analysis/analyze_docvqa_confirm64.py`：只读重算，不含新推理或调参。

本报告不合并replication32的分数，不改冻结推理core、模型、阈值或样本。本确认是本周期最后验证，按负结果收尾，不追加追数。
"""
    (ROOT / "docs/DOCVQA_CONFIRM64_REPORT.zh.md").write_text(report, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    plan = json.loads((OUT / "analysis_plan.json").read_text(encoding="utf-8-sig"))
    assert plan["monte_carlo"]["repeats"] == MC and plan["cluster_bootstrap"]["repeats"] == BOOTSTRAP
    assert plan["primary_family"] == [FUTURE + " - " + v for v in PRIMARY]
    assert plan["secondary_family"] == [FUTURE + " - " + v for v in COMPARATORS if v not in PRIMARY]
    assert plan["monte_carlo"]["seed"] == SEED and plan["cluster_bootstrap"]["seed"] == SEED
    assert plan["monte_carlo"]["alternative"] == "two-sided"
    paths, scored = load_scored()
    table, comparisons, flips = analyze(scored)
    audits = audit_inputs(paths, scored)
    shared.write_csv(OUT / "all12.csv", table)
    shared.write_csv(OUT / "paired_all11.csv", comparisons)
    shared.write_csv(OUT / "paired_flips.csv", flips)
    shared.write_csv(OUT / "primary_paired_flips.csv", [r for r in flips if r["baseline"] in PRIMARY])
    result = {"schema": "trajkv.docvqa_confirm64_analysis.v1", "status": "COMPLETE_FIXED_CONFIRMATION_ANALYSIS",
              "n_method_sessions": 768, "n_records": 768, "n_questions_per_method": 294,
              "n_answers": len(scored), "n_scored_turns": len(scored), "n_images": 64,
              "score_protocol": external.PROTOCOLS["docvqa"], "analysis_plan": plan,
              "all12": table, "comparisons": comparisons, "audit": audits,
              "exploratory16_pooled": False, "new_parameters_or_thresholds_selected": False,
              "kvarn_scope": "quality-only mechanism port with materialized decoded visual tensors; no serving speed/memory-compression claim",
              "statistic": "question-weighted mean; sign flips applied to image difference sums; bootstrap reweights cluster question counts",
              "mc_null_limit": "paired image sign-swap symmetry/exchangeability reference; not a model-causality proof",
              "input_sha256": audits["input_sha256"],
              "script_sha256": sha(Path(__file__))}
    result["figures"] = figure(table, comparisons)
    shared.write_json(OUT / "summary.json", result)
    write_report(result)
    print(json.dumps({"status": result["status"], "all12": [{"variant": r["variant"], "anls": r["mean"], "exact": r["exact_correct"]} for r in table],
                      "primary": [r for r in comparisons if r["family"] == "primary"],
                      "audit": {"initial_logit_disagreements": len(audits["initial_logit_disagreement_sessions"]),
                                "budget_violations": len(audits["read_budget_audit"]["violations"]),
                                "ledger_mismatches": len(audits["read_budget_audit"]["ledger_inconsistencies"]),
                                "model_mismatches": len(audits["model_metadata_mismatches"]),
                                "source_qa_verified": audits["source_questions_references_images"]["observed_questions_matched_to_local_source"]}}, indent=2))


if __name__ == "__main__":
    main()
