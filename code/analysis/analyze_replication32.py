"""Frozen budget-calibrated replication and nested 16-image budget sweep.

This module reports completed outcomes only. No threshold fitting or policy
selection is performed. The 16 sweep images are reused from the 32-image cohort,
so their experiments must not be counted as 48 independent image clusters.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import analyze_cycle as cycle
import analyze_probe
from analyze_frozen_eval64 import LABELS,csv_rows,holm_adjust,number
from append_kvarn_quality import contrast


BASELINE="asym:periodic"
OLD_PROBE="asym:probe_l7_t0.02"


def label(variant,primary):
    if variant==primary:return "Probe L7 budget-calibrated"
    if variant==OLD_PROBE:return "Probe L7 original .02"
    if "probe_l0_t0.159" in variant:return "Probe L0 budget-calibrated"
    if "probe_l14_t0.299" in variant:return "Probe L14 budget-calibrated"
    return LABELS.get(variant,variant)


def replication(scored,resources,contract,samples,*,draws,bootstrap,seed):
    variants=["full:base","base2:base","native4:base"]+[f"asym:{p}" for p in contract["asym_policies"]]
    primary=f"asym:{contract['primary_new_policy']}"
    ids={r["id"] for r in samples};images={r["source_image_id"] for r in samples}
    index={(r["session_id"],int(r["reference_turn"]),r["variant"]):r for r in scored}
    expected={(sid,t,v) for sid in ids for t in range(1,11) for v in variants}
    if len(ids)!=32 or len(images)!=32 or len(variants)!=12 or len(resources)!=384 or len(scored)!=3840 or set(index)!=expected:
        raise ValueError("Replication must exactly match frozen 32-image × 12-configuration × 10-turn coverage")
    if any(r["scored"]!="True" for r in scored):raise ValueError("Unscored replication answer")
    methods,pairs=[],[]
    for variant in variants:
        measured=[r for r in resources if r["variant"]==variant]
        for scope in ("all_scored","later_turns"):
            chosen=[r for r in scored if r["variant"]==variant and (scope=="all_scored" or int(r["reference_turn"])>=2)]
            stats=cycle.cluster_bootstrap(((r["image_id"],float(r["exact"])) for r in chosen),repeats=bootstrap,seed=seed)
            row=dict(variant=variant,label=label(variant,primary),scope=scope,**stats,
                     correct=int(sum(float(r["exact"]) for r in chosen)),primary=variant==primary)
            for field in ("computed_extra_logical_read_bytes","logical_total_read_bytes","session_ms","generated_tokens",
                          "owned_initial_all_visual_ratio","probe_extra_logical_bytes","probe_base_visual_bytes",
                          "probe_text_bytes","probe_residual_bytes","committed_residual_bytes"):
                values=[number(r,field) for r in measured]
                row[f"mean_{field}"]=float(np.mean(values))
                row[f"median_{field}"]=float(np.median(values))
                row[f"min_{field}"]=min(values);row[f"max_{field}"]=max(values)
            row["all_ledger_exact"]=all(r["extra_read_ledger_equal"]=="True" and r["read_ledger_exact_recorded"]=="True" for r in measured)
            row["all_budgets_ok"]=all(r["read_budget_ok_computed"]=="True" and number(r,"read_budget_bytes")==contract["read_budget_bytes"] for r in measured)
            methods.append(row)
            if variant!=BASELINE:
                pairs.append(dict(contrast(index,variant,BASELINE,scope,draws=draws,bootstrap=bootstrap,seed=seed),
                                  primary=variant==primary,label=label(variant,primary)))
    for scope in ("all_scored","later_turns"):
        family=[r for r in pairs if r["scope"]==scope]
        if len(family)!=11:raise ValueError("Replication Periodic family must contain 11 contrasts")
        for row,p in zip(family,holm_adjust([r["p_mc_two_sided"] for r in family])):
            row.update(p_holm_11=p,holm_family_size=11,
                       family="primary_periodic_11" if scope=="all_scored" else "secondary_later_periodic_11")
    old_pairs=[dict(contrast(index,primary,OLD_PROBE,scope,draws=draws,bootstrap=bootstrap,seed=seed),
                    family="planned_new_vs_old_iteration_contrast_outside_periodic_11",
                    inference_note="raw paired p and CI; does not replace the primary 11-comparison family")
               for scope in ("all_scored","later_turns")]
    return dict(primary_variant=primary,methods=methods,paired_inference=pairs,new_vs_old_probe=old_pairs,
                n_images=32,n_answers=3840,n_configurations=12,
                all_ledger_exact=all(r["all_ledger_exact"] for r in methods),
                all_budgets_ok=all(r["all_budgets_ok"] for r in methods))


def frozen_probe_report(records_path,scored,contract,samples,*,bootstrap,seed):
    """Use the pure reporting core after validating frozen evaluation membership.

    The development-only loader/threshold-grid interface remains unavailable;
    this path examines actual configured decisions, never proposes a new eta.
    """
    raw=cycle.load_jsonl(records_path)
    ids={r["id"] for r in samples}
    variants={"full:base","base2:base","native4:base"}|{f"asym:{p}" for p in contract["asym_policies"]}
    expected={(sid,variant) for sid in ids for variant in variants}
    actual={(r["session_id"],f"{r['method']}:{r.get('policy','base')}") for r in raw}
    if actual!=expected or len(raw)!=384:raise ValueError("Frozen probe report received non-contract sessions/configurations")
    settings=cycle.infer_settings(records_path)
    enriched=[]
    for row in raw:
        if row.get("read_budget_bytes")!=contract["read_budget_bytes"]:
            raise ValueError("Frozen probe report received an unexpected cap")
        context=dict(model=row.get("model"),protocol=row.get("protocol"),data_protocol=row.get("data_protocol"),settings=settings)
        profile=cycle.storage_profile(row)
        enriched.append(dict(row,source_records=str(records_path),context_id=analyze_probe.digest_json(context),context=context,
                             storage_id=analyze_probe.digest_json(profile),storage_profile=profile))
    references={}
    for row in scored:
        text=row["reference_answer"]
        value=json.loads(text) if text.startswith("[") else text
        key=(row["native_session_id"],int(row["reference_turn"]))
        if key in references and references[key]!=value:raise ValueError("Inconsistent offline scoring target")
        references[key]=value
    # No sweep grid is passed: only the literal thresholds already in the
    # frozen contract are summarized, with no tuning output.
    result=analyze_probe.analyze(enriched,references,bootstrap,seed,threshold_grid=())
    return {key:result[key] for key in ("probe_events","probe_skipped","probe_layer_phase_summary","phase_read_costs","probe_sessions")}


def sweep(scored,resources,cycle_summary,replication_samples):
    expected_ids={r["id"] for r in replication_samples[:16]}
    expected_images={r["source_image_id"] for r in replication_samples[:16]}
    caps=(4*2**20,8*2**20,16*2**20,32*2**20)
    variants={"asym:uniform","asym:periodic","asym:table_future",OLD_PROBE}
    if len(resources)!=256 or len(scored)!=2560:raise ValueError("Sweep must contain exactly 256 sessions and 2560 answers")
    groups=cycle_summary["groups"]
    indexed={(r["session_id"],r["variant"],int(number(r,"read_budget_bytes"))):r for r in resources}
    if set(indexed)!={(sid,v,c) for sid in expected_ids for v in variants for c in caps}:
        raise ValueError("Sweep methods/caps/images differ from the frozen nested design")
    if {r["image_id"] for r in scored}!=expected_images:
        raise ValueError("Sweep images must exactly be the replication32 first-16 subset")
    if any(r["scored"]!="True" for r in scored):raise ValueError("Unscored sweep answer")
    pairs=cycle_summary["paired_summary"]
    qualities=[]
    for group,description in groups.items():
        cap=int(description["read_budget_bytes"])
        for variant in sorted(variants):
            rows=[r for r in scored if r["group_id"]==group and r["variant"]==variant]
            measurements=[r for r in resources if r["group_id"]==group and r["variant"]==variant]
            if len(rows)!=160 or len(measurements)!=16:raise ValueError("Incomplete per-cap sweep cell")
            stats=cycle.cluster_bootstrap(((r["image_id"],float(r["exact"])) for r in rows),repeats=10000,seed=1212)
            row=dict(group_id=group,cap_bytes=cap,cap_mib=cap/2**20,variant=variant,**stats,
                correct=int(sum(float(r["exact"]) for r in rows)),
                mean_actual_extra_bytes=float(np.mean([number(r,"computed_extra_logical_read_bytes") for r in measurements])),
                min_actual_extra_bytes=min(number(r,"computed_extra_logical_read_bytes") for r in measurements),
                max_actual_extra_bytes=max(number(r,"computed_extra_logical_read_bytes") for r in measurements),
                mean_session_ms=float(np.mean([number(r,"session_ms") for r in measurements])),
                mean_generated_tokens=float(np.mean([number(r,"generated_tokens") for r in measurements])),
                all_ledger_exact=all(r["extra_read_ledger_equal"]=="True" and r["read_ledger_exact_recorded"]=="True" for r in measurements),
                all_budgets_ok=all(r["read_budget_ok_computed"]=="True" for r in measurements))
            if variant!=BASELINE:
                pair=next(p for p in pairs if p["group_id"]==group and p["variant"]==variant and p["scope"]=="all_scored")
                row.update(delta_vs_periodic_pp=pair["delta_pp"],delta_ci_lower_pp=100*pair["ci_lower"],delta_ci_upper_pp=100*pair["ci_upper"],
                           repaired=pair["repaired"],new_errors=pair["new_errors"])
            qualities.append(row)
    return dict(n_unique_images=16,n_cap_conditions=4,n_configurations_per_cap=4,n_answers=2560,
        reused_within_replication32=True,overlap_with_replication32_images=16,
        unique_clusters_across_replication_and_sweep=32,not_48_independent_images=True,
        source_image_ids=sorted(expected_images),quality=qualities,
        all_ledger_exact=all(r["all_ledger_exact"] for r in qualities),all_budgets_ok=all(r["all_budgets_ok"] for r in qualities))


def plots(rep,probe_rows,sweep_result,rep_dir,sweep_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    outputs={"replication":[],"sweep":[]}
    rep_dir.mkdir(parents=True,exist_ok=True);sweep_dir.mkdir(parents=True,exist_ok=True)
    primary=rep["primary_variant"]
    pairs=[r for r in rep["paired_inference"] if r["scope"]=="all_scored"]
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.1),layout="constrained",gridspec_kw={"width_ratios":[1.1,1]})
    for i,row in enumerate(pairs):
        color="#C39465" if row["primary"] else "#7D95A6"
        axes[0].errorbar(row["delta_pp"],i,xerr=[[row["delta_pp"]-row["ci_lower_pp"]],[row["ci_upper_pp"]-row["delta_pp"]]],
                         color=color,fmt="o",ms=3,capsize=2)
    axes[0].set_yticks(range(len(pairs)),[row["label"] for row in pairs],fontsize=6)
    axes[0].invert_yaxis();axes[0].axvline(0,color="#ADB5BD",lw=.8)
    axes[0].set_xlabel("Δ exact vs Periodic (pp)");axes[0].set_title("All 11 frozen replication contrasts",fontsize=9)
    candidates=(OLD_PROBE,primary)
    positions=np.arange(4)
    for j,variant in enumerate(candidates):
        for k,phase in enumerate(("text_prefill","decode")):
            rows=[r for r in probe_rows if r["variant"]==variant and r["phase"]==phase]
            n=sum(r["n_observed"] for r in rows);up=sum(r["upgrades"] for r in rows)
            axes[1].bar(j*2+k,up/n if n else 0,color="#526D82" if j==0 else "#C39465",width=.65)
            if len(rows)==1 and rows[0]["upgrade_rate_ci_lower"] is not None:
                value=up/n
                axes[1].errorbar(j*2+k,value,yerr=[[value-rows[0]["upgrade_rate_ci_lower"]],
                    [rows[0]["upgrade_rate_ci_upper"]-value]],fmt="none",ecolor="#424B54",capsize=2,lw=.7)
            axes[1].text(j*2+k,(up/n if n else 0)+.025,f"{up}/{n}",ha="center",fontsize=6)
    axes[1].set_xticks(positions,["Old / prefill","Old / decode","Calibrated / prefill","Calibrated / decode"],rotation=30,ha="right",fontsize=6)
    axes[1].set_ylim(0,1.12);axes[1].set_ylabel("Upgrade requests / observed probes")
    axes[1].set_title("Configured η7 selectivity · observed events",fontsize=9)
    for ax,letter in zip(axes,"ab"):ax.text(-.15,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    fig.suptitle("Reserved replication · 32 images · 320 QA per configuration",fontsize=9)
    for ext in ("png","pdf"):
        path=rep_dir/f"replication32_quality_selectivity.{ext}";fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");outputs["replication"].append(str(path))
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),layout="constrained")
    variants=("asym:uniform",BASELINE,"asym:table_future",OLD_PROBE)
    for variant,color,marker in zip(variants,("#526D82","#7FA580","#C5A56B","#9C8CB4"),("o","s","D","^")):
        rows=sorted([r for r in sweep_result["quality"] if r["variant"]==variant],key=lambda r:r["cap_bytes"])
        xs=[r["cap_mib"] for r in rows]
        ys=[100*r["mean"] for r in rows]
        axes[0].plot(xs,ys,color=color,marker=marker,ms=4,lw=1,label=LABELS[variant])
        axes[0].fill_between(xs,[100*r["ci_lower"] for r in rows],[100*r["ci_upper"] for r in rows],color=color,alpha=.06,linewidth=0)
        axes[1].plot(xs,[r["mean_actual_extra_bytes"]/2**20 for r in rows],color=color,marker=marker,ms=4,lw=1,label=LABELS[variant])
    axes[0].set_ylim(0,100);axes[0].set_ylabel("Normalized exact accuracy (%)")
    axes[0].set_title("Quality · image-cluster 95% CI",fontsize=9)
    axes[1].plot([4,32],[4,32],color="#ADB5BD",lw=.7,linestyle=":",label="actual = cap")
    axes[1].text(.02,.97,"Dotted line: actual = cap",transform=axes[1].transAxes,va="top",fontsize=6,color="#68737D")
    axes[1].set_ylabel("Mean realized extra reads (MiB/session)");axes[1].set_title("Cap and actual reads differ",fontsize=9)
    for ax,letter in zip(axes,"ab"):
        ax.set_xscale("log",base=2);ax.set_xticks([4,8,16,32],["4","8","16","32"]);ax.set_xlabel("Read allowance (MiB/session)")
        ax.text(-.15,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=7)
    fig.suptitle("Same 16 images at all four caps · subset of replication32, not 48 independent images",fontsize=8)
    for ext in ("png","pdf"):
        path=sweep_dir/f"budget_sweep_same16.{ext}";fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");outputs["sweep"].append(str(path))
    plt.close(fig)
    return outputs


def write_reports(rep,sweep_result,rep_dir,sweep_dir):
    primary=next(r for r in rep["paired_inference"] if r["primary"] and r["scope"]=="all_scored")
    direct=next(r for r in rep["new_vs_old_probe"] if r["scope"]=="all_scored")
    lines=["# 预算标定后的独立 replication32", "",
        "冻结合同保留 32 原图、12 配置、每配置 320 题，共 3,840 个回答。阈值只来自原 dev8 的无升级 online δ/cost；没有用本批结果重拟合。", "",
        f"主要新策略 `{rep['primary_variant']}` 对 Periodic：Δ {primary['delta_pp']:+.3f}pp，95% image-cluster CI [{primary['ci_lower_pp']:+.3f},{primary['ci_upper_pp']:+.3f}]；修复/新增错 {primary['repaired']}/{primary['new_errors']}，sign-flip p={primary['p_mc_two_sided']:.6g}，Holm11 p={primary['p_holm_11']:.6g}。", "",
        f"预先安排的新旧 Probe7 配对：Δ {direct['delta_pp']:+.3f}pp，CI [{direct['ci_lower_pp']:+.3f},{direct['ci_upper_pp']:+.3f}]，修复/新增错 {direct['repaired']}/{direct['new_errors']}，raw paired p={direct['p_mc_two_sided']:.6g}。这是单列迭代比较，不替代 Periodic 主 11 比较家族。", "",
        "| 方法 | 正确/320 | Exact | 对 Periodic Δpp [95% CI] | Holm11 p | actual extra MiB/会话 |",
        "|---|---:|---:|---:|---:|---:|"]
    pairs={r["variant"]:r for r in rep["paired_inference"] if r["scope"]=="all_scored"}
    for row in rep["methods"]:
        if row["scope"]!="all_scored":continue
        pair=pairs.get(row["variant"])
        delta=(f"{pair['delta_pp']:+.3f} [{pair['ci_lower_pp']:+.3f},{pair['ci_upper_pp']:+.3f}]" if pair else "reference")
        p=f"{pair['p_holm_11']:.6g}" if pair else "—"
        lines.append(f"| {row['label']} | {row['correct']}/320 | {100*row['mean']:.3f}% | {delta} | {p} | {row['mean_computed_extra_logical_read_bytes']/2**20:.4f} |")
    lines += ["", "## 冻结阈值的真实选择性", "",
              "| 策略 | phase | 观察/调用机会 | 升级请求/观察 | 预算跳过 |", "|---|---|---:|---:|---:|"]
    for row in rep["probe_selectivity"]:
        lines.append(f"| {label(row['variant'],rep['primary_variant'])} | {row['phase']} | {row['n_observed']}/{row['n_events']} | {row['upgrades']}/{row['n_observed']} | {row['n_skipped']} |")
    lines += ["", "预算校准恢复了选择性；这不是质量收益本身。实际 mean extra 由观察到的 probe、升级和硬 cap 共同形成，不能将缺失 probe 当作零信号。原始账全部独立复算，资源与自然生成时间分开保存。", "",
        "## 固定 probe floor 的作用域", "",
        "dev8 无升级轨迹中，2-bit Asym mean fixed probe extra 为 12.5488 MiB，1-bit 为 60.4065 MiB。1-bit 的长错误回答增加了探针调用次数；4–32 MiB 无解只针对这些观察轨迹、采集所有探针且仅阈值化后半层升级的预测模型。它不证明所有 1-bit 部署数学不可行。", "",
        "独立审计见 identity_audit.json；原64评测和其结论保留，Doc确认数据没有用于本分析。"]
    (rep_dir/"replication_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=["# 同 16 图的四预算 sweep", "",
        "每个额度使用相同 16 张图像、4 固定策略和每策略 160 题；四额度合计 2,560 个回答，但不是 64 个独立图像。它们是 replication32 的前16图，所以两实验的独立图像并集为32，不是48。", "",
        "Probe 使用固定旧阈值 `probe_l7_t0.02`；没有在各额度替换为新标定 η 或按结果选额度。", "",
        "| Cap MiB | 策略 | 正确/160 | Exact [95% image CI] | actual extra MiB/会话 | 对 Periodic Δpp [95% CI] |",
        "|---:|---|---:|---:|---:|---:|"]
    for row in sorted(sweep_result["quality"],key=lambda r:(r["cap_bytes"],r["variant"])):
        delta=(f"{row['delta_vs_periodic_pp']:+.3f} [{row['delta_ci_lower_pp']:+.3f},{row['delta_ci_upper_pp']:+.3f}]"
               if "delta_vs_periodic_pp" in row else "reference")
        lines.append(f"| {row['cap_mib']:g} | {LABELS[row['variant']]} | {row['correct']}/160 | {100*row['mean']:.3f}% [{100*row['ci_lower']:.3f},{100*row['ci_upper']:.3f}] | {row['mean_actual_extra_bytes']/2**20:.4f} | {delta} |")
    lines += ["", "各额度分别按这16张原图聚类 bootstrap。全部账与硬 cap 检查通过。actual extra 包含 probe 的 base/text 重读，不是只计 residual；逻辑 payload 不标成 HBM 流量。自然会话耗时和生成 token 数另表保留。", "",
        "16 MiB 的共享重复条件与 rep32 对应条件的生成 token/actual extra 独立审计逐条一致；这是执行重复性证据，不增加独立统计样本数。"]
    (sweep_dir/"sweep_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-analysis-dir",type=Path,required=True)
    parser.add_argument("--rep-records",type=Path,required=True)
    parser.add_argument("--sweep-analysis-dir",type=Path,required=True)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--samples",type=Path,required=True)
    parser.add_argument("--budget-table",type=Path,required=True)
    parser.add_argument("--rep-figures",type=Path,required=True)
    parser.add_argument("--sweep-figures",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=10000)
    parser.add_argument("--mc-draws",type=int,default=200000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    contract=json.loads(args.contract.read_text(encoding="utf-8-sig"));samples=json.loads(args.samples.read_text(encoding="utf-8-sig"))
    if cycle.sha256(args.budget_table)!=contract["threshold_source_sha256"]:
        raise ValueError("Budget table differs from frozen replication contract")
    scored=csv_rows(args.rep_analysis_dir/"scored_turns.csv");resources=csv_rows(args.rep_analysis_dir/"resource_sessions.csv")
    rep=replication(scored,resources,contract,samples,draws=args.mc_draws,bootstrap=args.bootstrap,seed=args.seed)
    probes=frozen_probe_report(args.rep_records,scored,contract,samples,bootstrap=args.bootstrap,seed=args.seed)
    for key,rows in probes.items():cycle.write_csv(args.rep_analysis_dir/f"frozen_{key}.csv",rows)
    for key in ("methods","paired_inference","new_vs_old_probe"):
        cycle.write_csv(args.rep_analysis_dir/f"replication_{key}.csv",rep[key])
    rep["probe_selectivity"]=probes["probe_layer_phase_summary"]
    sweep_result=sweep(csv_rows(args.sweep_analysis_dir/"scored_turns.csv"),csv_rows(args.sweep_analysis_dir/"resource_sessions.csv"),
                       json.loads((args.sweep_analysis_dir/"summary.json").read_text(encoding="utf-8-sig")),samples)
    cycle.write_csv(args.sweep_analysis_dir/"sweep_quality.csv",sweep_result["quality"])
    cycle.write_csv(args.sweep_analysis_dir/"sweep_resource.csv",[{k:v for k,v in row.items() if k in
        ("cap_bytes","cap_mib","variant","mean_actual_extra_bytes","min_actual_extra_bytes","max_actual_extra_bytes",
         "mean_session_ms","mean_generated_tokens","all_ledger_exact","all_budgets_ok")} for row in sweep_result["quality"]])
    table=json.loads(args.budget_table.read_text(encoding="utf-8-sig"))
    floors=[dict(method=g["method"],probe_layer=g["probe_layer"],n_training_sessions=g["n_sessions"],
        observed_mean_fixed_probe_floor_bytes=g["mean_fixed_probe_extra_bytes"],
        statuses=[dict(budget_bytes=b["budget_bytes"],status=b["status"],literal_policy_name=b.get("literal_policy_name")) for b in g["budgets"]])
        for g in table["groups"]]
    rep["observed_calibration_floors"]=floors
    rep["floor_scope"]="Only fixed probes on the observed no-upgrade development trajectories; base1 long erroneous answers create more probe calls. Not a universal impossibility result for 1-bit deployment."
    outputs=plots(rep,rep["probe_selectivity"],sweep_result,args.rep_figures,args.sweep_figures)
    rep["figures"]=outputs["replication"];sweep_result["figures"]=outputs["sweep"]
    rep.update(contract_sha256=cycle.sha256(args.contract),threshold_table_sha256=cycle.sha256(args.budget_table),
        no_threshold_refitting=True,no_final64_or_doc_outcome_tuning=True,
        primary_family="11 frozen comparisons to Periodic; later-turn family secondary",
        direct_iteration_scope="planned new-vs-old paired comparison reported separately; no substitution for primary Holm11",
        bootstrap=dict(unit="source image",repeats=args.bootstrap,seed=args.seed),
        signflip=dict(unit="whole image",draws=args.mc_draws,seed=args.seed,two_sided=True,plus_one=True),
        script_sha256=cycle.sha256(Path(__file__)))
    sweep_result.update(no_budget_selection_from_outcomes=True,bootstrap=dict(unit="same original 16 images per cap",repeats=10000,seed=1212),
                        no_pooling_with_replication32=True,script_sha256=cycle.sha256(Path(__file__)))
    for report,directory in ((rep,args.rep_analysis_dir),(sweep_result,args.sweep_analysis_dir)):
        audit=directory/"identity_audit.json"
        if audit.exists():report["identity_audit"]={"path":str(audit.resolve()),"sha256":cycle.sha256(audit)}
    rep["input_sha256"]={str(p.resolve()):cycle.sha256(p) for p in
        (args.rep_records,args.contract,args.samples,args.budget_table,args.rep_analysis_dir/"summary.json")}
    sweep_result["input_analysis_summary_sha256"]=cycle.sha256(args.sweep_analysis_dir/"summary.json")
    cycle.write_json(args.sweep_analysis_dir/"coverage_overlap_audit.json",{k:sweep_result[k] for k in
        ("n_unique_images","n_cap_conditions","n_configurations_per_cap","n_answers","reused_within_replication32",
         "overlap_with_replication32_images","unique_clusters_across_replication_and_sweep","not_48_independent_images","source_image_ids")})
    cycle.write_json(args.rep_analysis_dir/"replication_summary.json",rep)
    cycle.write_json(args.sweep_analysis_dir/"sweep_summary.json",sweep_result)
    write_reports(rep,sweep_result,args.rep_analysis_dir,args.sweep_analysis_dir)
    primary=next(r for r in rep["paired_inference"] if r["primary"] and r["scope"]=="all_scored")
    print(json.dumps(dict(rep_images=32,rep_answers=3840,primary_delta_pp=primary["delta_pp"],primary_holm11=primary["p_holm_11"],
                         sweep_unique_images=16,sweep_answers=2560,figures=sum(map(len,outputs.values())))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
