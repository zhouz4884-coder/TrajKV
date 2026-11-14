"""Postprocess the complete frozen 64-image, 17-configuration evaluation.

All policies are retained. Paired sign flips swap method labels at the original
image-cluster level; two-sided Monte Carlo p values receive Holm correction
over the 16 comparisons to the prespecified periodic comparator. No policy or
threshold is selected or changed by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_cycle as cycle


LABELS={"full:base":"Full FP16","base2:base":"Base K2/V2","native4:base":"Native K4/V4",
    "asym:base":"Asym base read","asym:uniform":"Uniform","asym:random":"Random","asym:periodic":"Periodic",
    "asym:prefill":"Text prefill","asym:early":"Early","asym:margin":"Margin replay",
    "asym:margin_write":"Margin write","asym:carrier_write":"Carrier write",
    "asym:table_current":"Current table","asym:table_future":"Future table",
    "asym:probe_l0_t0.05":"Probe L0 / .05","asym:probe_l7_t0.02":"Probe L7 / .02",
    "asym:probe_l14_t0.02":"Probe L14 / .02"}
REPRESENTATIVES=("full:base","base2:base","native4:base","asym:uniform","asym:periodic",
                 "asym:table_future","asym:probe_l0_t0.05","asym:probe_l7_t0.02","asym:probe_l14_t0.02")
SIMPLE_CONTROLS=("asym:base","asym:uniform","asym:random","asym:periodic","asym:prefill","asym:early")


def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def number(row,key):
    value=row.get(key)
    return float(value) if value not in (None,"") else None


def holm_adjust(p_values):
    """Holm step-down adjusted p values in original comparison order."""
    values=np.asarray(p_values,dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values<0) or np.any(values>1):
        raise ValueError("Holm correction requires finite p values in [0,1]")
    order=np.argsort(values,kind="stable")
    adjusted=np.empty(len(values))
    running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(values)-rank)*values[index])
        adjusted[index]=min(1.,running)
    return adjusted.tolist()


def paired_signflip_counts(image_difference_counts,*,draws=200000,seed=1212):
    """Two-sided cluster sign-flip Monte Carlo test of the net paired change.

    Input is the integer sum of question-level (method - comparator) outcomes
    within each image. Entire image labels flip together. Zero differences stay
    in the image count but do not create artificial independent observations.
    """
    differences=np.asarray(image_difference_counts,dtype=np.int64)
    if differences.ndim!=1 or not len(differences):
        raise ValueError("Expected a nonempty vector of image-level paired counts")
    if draws<1:
        raise ValueError("Monte Carlo draws must be positive")
    observed=int(abs(differences.sum()))
    if not np.any(differences):
        return dict(p_mc_two_sided=1.,draws=0,requested_draws=draws,seed=seed,
                    extreme_draws=None,observed_absolute_net_count=0,informative_images=0,
                    status="degenerate_all_image_differences_zero")
    rng=np.random.default_rng(seed)
    extreme=0
    for start in range(0,draws,8192):
        n=min(8192,draws-start)
        signs=rng.integers(0,2,size=(n,len(differences)),dtype=np.int8)*2-1
        randomized=signs@differences
        extreme+=int(np.count_nonzero(np.abs(randomized)>=observed))
    p=(extreme+1)/(draws+1)
    return dict(p_mc_two_sided=p,draws=draws,requested_draws=draws,seed=seed,
                extreme_draws=extreme,observed_absolute_net_count=observed,
                informative_images=int(np.count_nonzero(differences)),
                monte_carlo_standard_error=float(np.sqrt(p*(1-p)/(draws+1))),
                status="two_sided_image_cluster_label_swap_plus_one")


def coverage(scored,resources,contract,samples):
    expected_variants=["full:base","base2:base","native4:base"]+[f"asym:{p}" for p in contract["asym_policies"]]
    sample_ids={r["id"] for r in samples}
    image_ids={str(r.get("source_image_id",r.get("image_id"))) for r in samples}
    if len(sample_ids)!=64 or len(image_ids)!=64 or len(expected_variants)!=17:
        raise ValueError("Frozen main study must contain 64 original image clusters and 17 configurations")
    if len(resources)!=64*17 or len(scored)!=64*17*10:
        raise ValueError("Incomplete formal record/answer count")
    resource_index={}
    for row in resources:
        key=(row["session_id"],row["variant"])
        if key in resource_index:
            raise ValueError(f"Duplicate formal session/configuration: {key}")
        resource_index[key]=row
    expected={(sid,variant) for sid in sample_ids for variant in expected_variants}
    if set(resource_index)!=expected:
        raise ValueError("Formal methods/sessions differ from frozen manifest")
    turn_keys=set()
    for row in scored:
        if row.get("scored")!="True" or row.get("scoring_note")!="scored":
            raise ValueError("A formal answer lacks an independent score")
        key=(row["session_id"],row["variant"],int(row["reference_turn"]))
        if key in turn_keys:
            raise ValueError(f"Duplicate formal question: {key}")
        turn_keys.add(key)
        if row["image_id"] not in image_ids:
            raise ValueError("Unknown source image in formal answers")
    if turn_keys!={(sid,variant,t) for sid in sample_ids for variant in expected_variants for t in range(1,11)}:
        raise ValueError("Formal question coverage is not the complete frozen Cartesian product")
    bad_ledgers=[(r["session_id"],r["variant"]) for r in resources
        if r.get("extra_read_ledger_equal")!="True" or r.get("read_ledger_exact_recorded")!="True"]
    bad_budgets=[(r["session_id"],r["variant"]) for r in resources
        if r.get("read_budget_ok_computed")!="True" or number(r,"read_budget_bytes")!=contract["extra_read_allowance_bytes"]]
    return dict(n_sessions=64,n_images=64,n_configurations=17,n_session_configurations=len(resources),
        n_answers=len(scored),complete_cartesian_coverage=True,expected_variants=expected_variants,
        ledger_failures=bad_ledgers,budget_failures=bad_budgets,
        all_ledger_exact=not bad_ledgers,all_budgets_respected=not bad_budgets,
        source_split_labels=sorted({r["split"] for r in scored}),
        split_label_scope="Original candidate labels are retained; formal membership is defined by the independently frozen 64-image manifest")


def build_tables(scored,resources,contract,*,baseline,draws,bootstrap,seed):
    variants=["full:base","base2:base","native4:base"]+[f"asym:{p}" for p in contract["asym_policies"]]
    primary=f"asym:{contract['primary_new_policy']}"
    images=sorted({r["image_id"] for r in scored})
    answers={(r["session_id"],int(r["reference_turn"]),r["variant"]):r for r in scored}
    session_ids=sorted({r["session_id"] for r in scored})
    res={(r["session_id"],r["variant"]):r for r in resources}
    method_rows=[]
    paired=[]
    image_diffs=[]
    resource_diffs=[]
    for variant in variants:
        rows=[r for r in scored if r["variant"]==variant]
        measurements=[r for r in resources if r["variant"]==variant]
        base_measurements=[res[(r["session_id"],baseline)] for r in measurements]
        for scope in ("all_scored","later_turns"):
            chosen=[r for r in rows if scope=="all_scored" or int(r["reference_turn"])>=2]
            stats=cycle.cluster_bootstrap(((r["image_id"],float(r["exact"])) for r in chosen),repeats=bootstrap,seed=seed)
            method=dict(variant=variant,label=LABELS.get(variant,variant),scope=scope,**stats,
                        correct=int(sum(float(r["exact"]) for r in chosen)),primary=variant==primary)
            for field in ("computed_extra_logical_read_bytes","logical_total_read_bytes","session_ms","generated_tokens",
                          "owned_initial_all_visual_bytes","owned_initial_all_visual_ratio","peak_allocated_bytes",
                          "probe_extra_logical_bytes","committed_residual_bytes","probe_base_visual_bytes","probe_text_bytes",
                          "probe_residual_bytes","replay_logical_bytes"):
                values=[number(r,field) for r in measurements]
                method[f"mean_{field}"]=float(np.mean(values)) if all(v is not None for v in values) else None
                method[f"min_{field}"]=min(values) if all(v is not None for v in values) else None
                method[f"max_{field}"]=max(values) if all(v is not None for v in values) else None
                method[f"median_{field}"]=float(np.median(values)) if all(v is not None for v in values) else None
                if field=="session_ms":
                    method["p95_session_ms"]=float(np.quantile(values,.95)) if all(v is not None for v in values) else None
            method["same_stored_layout_as_periodic"]=all(r["storage_layout_signature"]==b["storage_layout_signature"]
                                                        for r,b in zip(measurements,base_measurements))
            method["common_extra_allowance_bytes"]=contract["extra_read_allowance_bytes"]
            method["all_actual_extra_equal_to_periodic"]=all(number(r,"computed_extra_logical_read_bytes")==number(b,"computed_extra_logical_read_bytes")
                                                               for r,b in zip(measurements,base_measurements))
            method_rows.append(method)
            if variant==baseline:
                continue
            questions=[]
            by_image=defaultdict(lambda:dict(net_count=0,repaired=0,new_errors=0,n_questions=0))
            for sid in session_ids:
                for turn in range(1 if scope=="all_scored" else 2,11):
                    a,b=answers[(sid,turn,baseline)],answers[(sid,turn,variant)]
                    old,new=int(float(a["exact"])),int(float(b["exact"]))
                    difference=new-old
                    questions.append((a["image_id"],difference))
                    item=by_image[a["image_id"]]
                    item["net_count"]+=difference;item["n_questions"]+=1
                    item["repaired"]+=int(old==0 and new==1);item["new_errors"]+=int(old==1 and new==0)
            stats=cycle.cluster_bootstrap(questions,repeats=bootstrap,seed=seed)
            test=paired_signflip_counts([by_image[image]["net_count"] for image in images],draws=draws,seed=seed)
            pair=dict(variant=variant,label=LABELS.get(variant,variant),baseline=baseline,scope=scope,
                primary=variant==primary,**stats,delta_pp=stats["mean"]*100,
                ci_lower_pp=stats["ci_lower"]*100,ci_upper_pp=stats["ci_upper"]*100,
                repaired=sum(r["repaired"] for r in by_image.values()),new_errors=sum(r["new_errors"] for r in by_image.values()),
                p_mc_two_sided=test["p_mc_two_sided"],mc_draws=test["draws"],mc_seed=seed,
                informative_images=test["informative_images"],mc_status=test["status"])
            paired.append(pair)
            for image,item in sorted(by_image.items()):
                image_diffs.append(dict(image_id=image,variant=variant,baseline=baseline,scope=scope,
                                         delta_accuracy=item["net_count"]/item["n_questions"],**item))
        if variant!=baseline:
            for field in ("computed_extra_logical_read_bytes","logical_total_read_bytes","session_ms","generated_tokens",
                          "owned_initial_all_visual_bytes","probe_extra_logical_bytes","committed_residual_bytes"):
                differences=[(a["image_id"],number(a,field)-number(b,field)) for a,b in zip(measurements,base_measurements)]
                stats=cycle.cluster_bootstrap(differences,repeats=bootstrap,seed=seed)
                resource_diffs.append(dict(variant=variant,baseline=baseline,metric=field,**stats,
                    same_allowance=all(number(a,"read_budget_bytes")==number(b,"read_budget_bytes") for a,b in zip(measurements,base_measurements)),
                    actual_consumption_equal=all(value==0 for _,value in differences)))
    for scope in ("all_scored","later_turns"):
        family=[r for r in paired if r["scope"]==scope]
        if len(family)!=16:
            raise ValueError("The prespecified multiplicity family must contain 16 comparisons")
        for row,adjusted in zip(family,holm_adjust([r["p_mc_two_sided"] for r in family])):
            row.update(p_holm_16=adjusted,holm_family_size=16,
                       inference_scope="primary overall family" if scope=="all_scored" else "secondary later-turn family")
    return method_rows,paired,image_diffs,resource_diffs


def plot(methods,paired,baseline,primary,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    index={r["variant"]:r for r in methods if r["scope"]=="all_scored"}
    differences=[r for r in paired if r["scope"]=="all_scored"]
    fig=plt.figure(figsize=(7.2,6.3),layout="constrained")
    grid=fig.add_gridspec(2,2,width_ratios=(1,1.1))
    absolute=fig.add_subplot(grid[0,0]);resource=fig.add_subplot(grid[1,0]);forest=fig.add_subplot(grid[:,1])
    blue,orange,gray,green="#526D82","#C39465","#A7B3BE","#7FA580"
    reps=[index[v] for v in REPRESENTATIVES]
    labels=["Full","K2/V2","K4/V4","Uniform","Periodic","Future","Probe 0","Probe 7","Probe 14"]
    for i,row in enumerate(reps):
        y=row["mean"]*100
        color=orange if row["variant"]==primary else (green if row["variant"]==baseline else blue)
        absolute.bar(i,y,color=color,width=.7)
        absolute.errorbar(i,y,yerr=[[y-row["ci_lower"]*100],[row["ci_upper"]*100-y]],color="#424B54",fmt="none",capsize=1.5,lw=.7)
    absolute.set_xticks(range(len(reps)),labels,rotation=45,ha="right",fontsize=6)
    absolute.set_ylim(0,100);absolute.set_ylabel("Normalized exact accuracy (%)")
    absolute.set_title("Representative frozen configurations",fontsize=8)
    resource_reps=("asym:base","asym:uniform","asym:periodic","asym:table_future",
                   "asym:probe_l0_t0.05","asym:probe_l7_t0.02","asym:probe_l14_t0.02")
    for i,variant in enumerate(resource_reps):
        row=index[variant]
        color=orange if variant==primary else (green if variant==baseline else blue)
        resource.barh(i,row["mean_computed_extra_logical_read_bytes"]/2**20,color=color,height=.65)
    resource.axvline(16,color="#ADB5BD",linestyle=":",lw=.8)
    resource.set_yticks(range(len(resource_reps)),[LABELS[v] for v in resource_reps],fontsize=6)
    resource.invert_yaxis();resource.set_xlim(0,16.6)
    resource.set_xlabel("Mean realized extra logical reads (MiB/session)")
    resource.set_title("Same Asym storage · common cap ≠ equal reads",fontsize=8)
    for i,row in enumerate(differences):
        color=orange if row["primary"] else (blue if row["variant"]=="asym:table_future" else gray)
        forest.errorbar(row["delta_pp"],i,xerr=[[row["delta_pp"]-row["ci_lower_pp"]],[row["ci_upper_pp"]-row["delta_pp"]]],
                        fmt="o",color=color,capsize=2,ms=3,lw=.9)
    forest.axvline(0,color="#ADB5BD",lw=.8)
    forest.set_yticks(range(len(differences)),[LABELS[r["variant"]]+(" (primary)" if r["primary"] else "") for r in differences],fontsize=6)
    forest.invert_yaxis();forest.set_xlabel("Δ exact vs periodic (percentage points)")
    forest.set_title("All 16 frozen comparisons · image-cluster 95% CI",fontsize=8)
    for ax,letter in ((absolute,"a"),(forest,"b"),(resource,"c")):
        ax.text(-.12,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    fig.suptitle("Frozen local image-heldout evaluation · 64 images · 640 QA per configuration",fontsize=9)
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for extension in ("png","pdf"):
        path=directory/f"frozen_eval64_main.{extension}"
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");paths.append(str(path))
    plt.close(fig)
    return paths


def format_ci(row):
    return f"{row['delta_pp']:+.3f} [{row['ci_lower_pp']:+.3f}, {row['ci_upper_pp']:+.3f}]"


def write_report(path,report):
    main={r["variant"]:r for r in report["paired_inference"] if r["scope"]=="all_scored"}
    later={r["variant"]:r for r in report["paired_inference"] if r["scope"]=="later_turns"}
    primary=main[report["primary_variant"]]
    all_method_index={r["variant"]:r for r in report["methods"] if r["scope"]=="all_scored"}
    best_text="、".join(f"`{v}` ({all_method_index[v]['correct']}/640)" for v in report["highest_observed_simple_controls"])
    all_one=all(r["p_holm_16"]==1 for r in main.values())
    family_text="全部 16 个主比较的 Holm p 都为 1。" if all_one else "全部 16 个主比较的 Holm 校正结果完整列于下表。"
    primary_supported=primary["delta_pp"]>0 and primary["ci_lower_pp"]>0 and primary["p_holm_16"]<.05
    conclusion="本轮支持预设主要策略相对 Periodic 的质量优势。" if primary_supported else "本轮没有建立预设主要策略的质量优势，也没有据此证明各方法等价。"
    rows=["# 冻结 64 图主评测", "", "完整保留 17 配置、64 原始图像、每配置 640 题，共 10,880 个回答。", "",
        f"预设比较基准：`{report['baseline']}`；预设主要新策略：`{report['primary_variant']}`。",
        "统计按 64 张图像聚类；质量为严格规范化 CLEVR exact。主比较使用图像级双侧 sign-flip Monte Carlo p，并对 16 个比较作 Holm 校正。", "",
        f"主要策略 Δ(pp) = {format_ci(primary)}；MC p={primary['p_mc_two_sided']:.6g}，Holm p={primary['p_holm_16']:.6g}；修复 {primary['repaired']} 题，新增错误 {primary['new_errors']} 题。", "",
        f"预设朴素 Asym 控制集合中的最高观察分数为 {best_text}；这只是完整表中的描述性最高值，推断基准始终保持 Periodic。",
        family_text+conclusion, "",
        "## 全部冻结方法", "", "| 方法 | 正确/640 | Exact | 对 Periodic Δpp [95% CI] | Holm p | 实际 extra MiB/会话 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in report["methods"]:
        if row["scope"]!="all_scored":continue
        comparison=main.get(row["variant"])
        delta=format_ci(comparison) if comparison else "reference"
        p=f"{comparison['p_holm_16']:.6g}" if comparison else "—"
        rows.append(f"| {row['label']} | {row['correct']}/640 | {row['mean']*100:.3f}% | {delta} | {p} | {row['mean_computed_extra_logical_read_bytes']/2**20:.4f} |")
    rows += ["", "## 后续 T2–T10", "", "每配置 576 题，仍为 64 个图像 cluster。此处为独立的次要 16 比较家族，未与主家族混合校正。", "",
             "| 方法 | Δpp [95% CI] | Holm p | 修复/新增错误 |", "|---|---:|---:|---:|"]
    for variant in (report["primary_variant"],"asym:table_future","asym:probe_l0_t0.05","asym:probe_l14_t0.02"):
        row=later[variant]
        rows.append(f"| {LABELS[variant]} | {format_ci(row)} | {row['p_holm_16']:.6g} | {row['repaired']}/{row['new_errors']} |")
    rows += ["", "## 资源与范围", "",
        "所有配置使用同一 16 MiB extra 逻辑读额度；实际消耗并不相等。资源差及其图像聚类区间完整见 resource_comparisons.csv。",
        "Full、Base2、Native4 是表示不同的质量/资源参照；同驻留比较限 Asym 策略。逻辑 payload 不是实测 HBM 流量。自然生成时间同时受输出长度影响，不解释成独立内核加速。",
        "最终数据只用于本次冻结报告，未调整算法、阈值、训练表或方法清单。历史 split 标签由构建器保留，正式测试成员由事前冻结的 64 图 manifest 定义。",
        "", "完整数值、逐图配对差、所有方法与输入哈希见机器可读 formal_summary.json。"]
    path.write_text("\n".join(rows)+"\n",encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--samples",type=Path,required=True)
    parser.add_argument("--baseline",default="asym:periodic")
    parser.add_argument("--figures",type=Path,required=True)
    parser.add_argument("--mc-draws",type=int,default=200000)
    parser.add_argument("--bootstrap",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    contract=json.loads(args.contract.read_text(encoding="utf-8-sig"))
    samples=json.loads(args.samples.read_text(encoding="utf-8-sig"))
    scored_path=args.analysis_dir/"scored_turns.csv"; resource_path=args.analysis_dir/"resource_sessions.csv"
    scored,resources=csv_rows(scored_path),csv_rows(resource_path)
    audit=coverage(scored,resources,contract,samples)
    methods,paired,image_diffs,resource_diffs=build_tables(scored,resources,contract,baseline=args.baseline,
        draws=args.mc_draws,bootstrap=args.bootstrap,seed=args.seed)
    for name,rows in (("all_methods_quality",methods),("paired_inference",paired),("image_paired_differences",image_diffs),
                      ("resource_comparisons",resource_diffs)):
        cycle.write_csv(args.analysis_dir/f"{name}.csv",rows)
    primary=f"asym:{contract['primary_new_policy']}"
    figures=plot(methods,paired,args.baseline,primary,args.figures)
    means={r["variant"]:r["mean"] for r in methods if r["scope"]=="all_scored"}
    best=max(means[v] for v in SIMPLE_CONTROLS)
    report=dict(status="OK" if audit["all_ledger_exact"] and audit["all_budgets_respected"] else "RESOURCE_AUDIT_FAILURE",
        baseline=args.baseline,primary_variant=primary,coverage_audit=audit,
        methods=methods,paired_inference=paired,resource_comparisons=resource_diffs,
        highest_observed_simple_controls=[v for v in SIMPLE_CONTROLS if means[v]==best],
        strongest_control_scope="descriptive maximum in the prespecified simple-control set; inference comparator remains periodic",
        bootstrap=dict(unit="source image",repeats=args.bootstrap,seed=args.seed),
        signflip=dict(unit="whole-image paired label swaps",draws=args.mc_draws,seed=args.seed,
            two_sided=True,ties="absolute randomized net count >= absolute observed net count",plus_one=True),
        multiplicity="Holm over all 16 periodic contrasts in the primary overall family; later-turn family is secondary",
        input_sha256={str(p.resolve()):cycle.sha256(p) for p in (scored_path,resource_path,args.contract,args.samples)},
        script_sha256=cycle.sha256(Path(__file__)),figures=figures,
        no_final_data_tuning=True,scope="local image-heldout evaluation; no official server or cross-model claim")
    identity_audit=args.analysis_dir/"identity_audit.json"
    if identity_audit.exists():
        report["identity_audit"]={"path":str(identity_audit.resolve()),"sha256":cycle.sha256(identity_audit)}
    cycle.write_json(args.analysis_dir/"formal_summary.json",report)
    cycle.write_json(args.analysis_dir/"coverage_audit.json",audit)
    write_report(args.analysis_dir/"formal_report.md",report)
    focus=next(r for r in paired if r["primary"] and r["scope"]=="all_scored")
    print(json.dumps(dict(status=report["status"],images=64,answers=len(scored),primary_delta_pp=focus["delta_pp"],
                         primary_holm_p=focus["p_holm_16"],figures=len(figures))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
