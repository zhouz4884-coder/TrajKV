"""Append a separate KVarN64 quality-reference family to the frozen main study.

The original 17-method primary report and its Holm-16 family are read only.
Materialized KVarN mechanism ports are not packed-runtime efficiency baselines.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

import analyze_cycle as cycle
from analyze_frozen_eval64 import LABELS,csv_rows,holm_adjust,number,paired_signflip_counts


PORTS=("kvarn2:base","kvarn4:base")
TRIO=("asym:periodic",)+PORTS


def contrast(index,variant,baseline,scope,*,draws,bootstrap,seed):
    selected=[r for key,r in index.items() if key[2]==variant and
              (scope=="all_scored" or key[1]>=2)]
    differences=[]
    by_image=defaultdict(int)
    repaired=new_errors=0
    for row in selected:
        key=(row["session_id"],int(row["reference_turn"]),baseline)
        ref=index[key]
        old,new=int(float(ref["exact"])),int(float(row["exact"]))
        differences.append((row["image_id"],new-old));by_image[row["image_id"]]+=new-old
        repaired+=int(old==0 and new==1);new_errors+=int(old==1 and new==0)
    stats=cycle.cluster_bootstrap(differences,repeats=bootstrap,seed=seed)
    test=paired_signflip_counts([by_image[image] for image in sorted(by_image)],draws=draws,seed=seed)
    return dict(variant=variant,baseline=baseline,scope=scope,**stats,
        delta_pp=100*stats["mean"],ci_lower_pp=100*stats["ci_lower"],ci_upper_pp=100*stats["ci_upper"],
        repaired=repaired,new_errors=new_errors,p_mc_two_sided=test["p_mc_two_sided"],mc_draws=test["draws"],
        mc_seed=seed,informative_images=test["informative_images"])


def analyze(scored,resources,original,*,draws,bootstrap,seed):
    expected=set(original["coverage_audit"]["expected_variants"])|set(PORTS)
    if len(resources)!=64*19 or len(scored)!=64*19*10:
        raise ValueError("Expected complete 64-image × 19-configuration × 10-turn quality data")
    if {r["variant"] for r in scored}!=expected:
        raise ValueError("Combined configuration set differs from the original 17 plus KVarN2/4")
    index={(r["session_id"],int(r["reference_turn"]),r["variant"]):r for r in scored}
    if len(index)!=len(scored) or any(r["scored"]!="True" for r in scored):
        raise ValueError("Combined answers have duplicate or unscored questions")
    ids=sorted({r["session_id"] for r in scored});images={r["image_id"] for r in scored}
    if len(ids)!=64 or len(images)!=64:
        raise ValueError("Combined family must retain exactly the same 64 source images")
    if set(index)!={(sid,t,v) for sid in ids for t in range(1,11) for v in expected}:
        raise ValueError("Incomplete Cartesian question coverage")
    original_methods=original["methods"]
    methods=[]
    for old in original_methods:
        chosen=[r for r in scored if r["variant"]==old["variant"] and
                (old["scope"]=="all_scored" or int(r["reference_turn"])>=2)]
        if int(sum(float(r["exact"]) for r in chosen))!=old["correct"]:
            raise ValueError("Original 17-method scored outcomes changed; original primary inference cannot be copied")
        methods.append(dict(old,inference_family="original_primary_17_methods_unchanged",mechanism_port=False))
    for variant in PORTS:
        measured=[r for r in resources if r["variant"]==variant]
        if len(measured)!=64 or any(r["cost_comparison_allowed"]!="False" for r in measured):
            raise ValueError("KVarN quality reference must retain its explicit non-serving cost boundary")
        if any(r["resource_runtime_class"]!="materialized_kvarn_reference" for r in measured):
            raise ValueError("Expected explicitly materialized KVarN mechanism-port runtime")
        for scope in ("all_scored","later_turns"):
            chosen=[r for r in scored if r["variant"]==variant and
                    (scope=="all_scored" or int(r["reference_turn"])>=2)]
            stats=cycle.cluster_bootstrap(((r["image_id"],float(r["exact"])) for r in chosen),repeats=bootstrap,seed=seed)
            methods.append(dict(variant=variant,label=("KVarN2" if variant.startswith("kvarn2") else "KVarN4")+" † mechanism-port",
                scope=scope,**stats,correct=int(sum(float(r["exact"]) for r in chosen)),mechanism_port=True,
                inference_family="secondary_quality_reference",competitive_latency_allowed=False,
                mean_owned_visual_ratio=float(np.mean([number(r,"owned_initial_all_visual_ratio") for r in measured])),
                mean_materialized_visual_ratio=float(np.mean([number(r,"owned_initial_materialized_visual_ratio") for r in measured])),
                mean_encoded_reference_ratio=float(np.mean([number(r,"owned_initial_encoded_reference_ratio") for r in measured])),
                cost_scope="packed encoding plus materialized FP16 visual owner; no packed serving efficiency comparison"))
    secondary=[]
    for scope in ("all_scored","later_turns"):
        family=[contrast(index,v,TRIO[0],scope,draws=draws,bootstrap=bootstrap,seed=seed) for v in PORTS]
        for row,p in zip(family,holm_adjust([r["p_mc_two_sided"] for r in family])):
            secondary.append(dict(row,p_holm_secondary2=p,holm_family_size=2,
                                  family="secondary_KVarN_vs_periodic" if scope=="all_scored" else "secondary_later_KVarN_vs_periodic"))
    trio=[]
    patterns=defaultdict(int)
    for sid in ids:
        for turn in range(1,11):
            entries=[index[(sid,turn,variant)] for variant in TRIO]
            pattern="".join(str(int(float(r["exact"]))) for r in entries)
            patterns[pattern]+=1
            trio.append(dict(session_id=sid,image_id=entries[0]["image_id"],reference_turn=turn,correctness_pattern=pattern,
                periodic_exact=int(float(entries[0]["exact"])),kvarn2_exact=int(float(entries[1]["exact"])),
                kvarn4_exact=int(float(entries[2]["exact"])),periodic_answer=entries[0]["prediction"],
                kvarn2_answer=entries[1]["prediction"],kvarn4_answer=entries[2]["prediction"]))
    direct=[]
    for scope in ("all_scored","later_turns"):
        direct.append(dict(contrast(index,PORTS[1],PORTS[0],scope,draws=draws,bootstrap=bootstrap,seed=seed),
                           family="descriptive_cross_KVarN_pair_not_in_secondary2",p_holm_secondary2=None))
    references=[]
    for variant,baseline in ((PORTS[0],"full:base"),(PORTS[0],"base2:base"),(PORTS[1],"native4:base")):
        for scope in ("all_scored","later_turns"):
            references.append(dict(contrast(index,variant,baseline,scope,draws=draws,bootstrap=bootstrap,seed=seed),
                                   family="descriptive_representation_reference_pair_outside_primary_family"))
    return dict(methods=methods,secondary_pairs=secondary,three_way_outcomes=trio,
                three_way_pairs=secondary+direct,reference_pairs=references,
                three_way_patterns=dict(sorted(patterns.items())),three_way_pattern_order=list(TRIO),
                original_primary_inference=original["paired_inference"])


def plot(result,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout="constrained")
    index={r["variant"]:r for r in result["methods"] if r["scope"]=="all_scored"}
    variants=("full:base","base2:base","native4:base","asym:periodic")+PORTS
    labels=("Full","K2/V2","K4/V4","Periodic","KVarN2 †","KVarN4 †")
    for i,variant in enumerate(variants):
        row=index[variant];y=100*row["mean"]
        port=variant in PORTS
        axes[0].bar(i,y,color="#C5A56B" if port else "#526D82",hatch="//" if port else None,width=.65)
        axes[0].errorbar(i,y,yerr=[[y-100*row["ci_lower"]],[100*row["ci_upper"]-y]],color="#424B54",fmt="none",capsize=2,lw=.8)
    axes[0].set_xticks(range(len(variants)),labels,rotation=30,ha="right",fontsize=7)
    axes[0].set_ylim(0,100);axes[0].set_ylabel("Normalized exact accuracy (%)")
    axes[0].set_title("Same 64 images · quality reference",fontsize=9)
    rows=result["secondary_pairs"]
    for i,row in enumerate(rows):
        axes[1].errorbar(row["delta_pp"],i,xerr=[[row["delta_pp"]-row["ci_lower_pp"]],[row["ci_upper_pp"]-row["delta_pp"]]],
                         fmt="o",color="#526D82" if row["scope"]=="all_scored" else "#C5A56B",capsize=2,ms=4)
    axes[1].set_yticks(range(len(rows)),[("KVarN2" if r["variant"]==PORTS[0] else "KVarN4")+
                                      (" · all" if r["scope"]=="all_scored" else " · T2–10") for r in rows],fontsize=7)
    axes[1].invert_yaxis();axes[1].axvline(0,color="#ADB5BD",lw=.8)
    axes[1].set_xlabel("Δ exact vs Periodic (percentage points)")
    axes[1].set_title("Secondary pairs · image-cluster 95% CI",fontsize=9)
    for ax,letter in zip(axes,"ab"):
        ax.text(-.15,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    fig.suptitle("† Materialized mechanism ports · excluded from packed-runtime efficiency comparison",fontsize=8)
    directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for extension in ("png","pdf"):
        path=directory/f"eval64_kvarn_secondary_quality.{extension}"
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");outputs.append(str(path))
    plt.close(fig)
    return outputs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--original-main",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=10000)
    parser.add_argument("--mc-draws",type=int,default=200000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    original=json.loads(args.original_main.read_text(encoding="utf-8-sig"))
    scored_path=args.analysis_dir/"scored_turns.csv";resources_path=args.analysis_dir/"resource_sessions.csv"
    scored=csv_rows(scored_path)
    original_scored_path=args.original_main.parent/"scored_turns.csv"
    old_scored=csv_rows(original_scored_path)
    combined_index={(r["session_id"],r["reference_turn"],r["variant"]):r for r in scored}
    for old in old_scored:
        new=combined_index[(old["session_id"],old["reference_turn"],old["variant"])]
        if any(new[field]!=old[field] for field in ("exact","image_id","prediction")):
            raise ValueError("A primary-family per-question outcome changed; keep the original primary analysis frozen")
    result=analyze(scored,csv_rows(resources_path),original,
                   draws=args.mc_draws,bootstrap=args.bootstrap,seed=args.seed)
    for key in ("methods","secondary_pairs","three_way_outcomes","three_way_pairs","reference_pairs","original_primary_inference"):
        cycle.write_csv(args.analysis_dir/f"kvarn_{key}.csv",result[key])
    cycle.write_csv(args.analysis_dir/"all19_quality.csv",result["methods"])
    result["figures"]=plot(result,args.figures)
    result.update(status="OK",n_images=64,n_configurations=19,n_answers=12160,
        original_main_report=str(args.original_main.resolve()),original_main_sha256=cycle.sha256(args.original_main),
        original_primary_per_question_outcomes_unchanged=True,
        primary_family_scope="original 17 methods / Holm16 inference copied without modification",
        secondary_family_scope="KVarN2 and KVarN4 vs Periodic, Holm2; later-turn Holm2 is secondary",
        third_pair_scope="KVarN4 vs KVarN2 is descriptive, outside the two baseline comparisons",
        mechanism_port_scope="quality-only low-level quantization reference; full materialized visual KV plus encoded owners; no packed efficiency claim",
        no_final_data_tuning=True,input_sha256={str(p.resolve()):cycle.sha256(p) for p in (scored_path,resources_path,args.original_main,original_scored_path)},
        script_sha256=cycle.sha256(Path(__file__)))
    audit=args.analysis_dir/"kvarn_identity_audit.json"
    if audit.exists():result["identity_audit"]={"path":str(audit.resolve()),"sha256":cycle.sha256(audit)}
    cycle.write_json(args.analysis_dir/"quality19_summary.json",result)
    lines=["# 冻结主评测与 KVarN64 质量参考", "", "19 配置 × 64 图像 × 10 轮，共 12,160 个回答。原 17 配置的报告、图与 Holm16 主结论保持原样。", "",
        "KVarN2/4 为 †materialized mechanism-port，只作为质量参考；两方法对 Periodic 的结果属于独立 Holm2 secondary 家族。", "",
        "| 方法 | 正确/640 | Exact [95% image CI] |", "|---|---:|---:|"]
    for row in result["methods"]:
        if row["scope"]=="all_scored":
            lines.append(f"| {row['label']} | {row['correct']}/640 | {100*row['mean']:.3f}% [{100*row['ci_lower']:.3f}, {100*row['ci_upper']:.3f}] |")
    lines += ["", "## KVarN 对 Periodic 的 secondary 比较", "",
        "| 方法/范围 | Δpp [95% CI] | 修复/新增错误 | MC p | Holm2 p |", "|---|---:|---:|---:|---:|"]
    for row in result["secondary_pairs"]:
        lines.append(f"| {row['variant']} / {row['scope']} | {row['delta_pp']:+.3f} [{row['ci_lower_pp']:+.3f}, {row['ci_upper_pp']:+.3f}] | {row['repaired']}/{row['new_errors']} | {row['p_mc_two_sided']:.6g} | {row['p_holm_secondary2']:.6g} |")
    lines += ["", "KVarN2/4 的真实视觉 owner 分别为 FP16 visual 的 1.1484375/1.2734375：其中保留物化 FP16 值和编码参考。不能将 encoded-reference ratio 当作当前运行时显存节省，也不与 packed 方法做部署延迟/吞吐优劣推断。", "",
        "Periodic/KVarN2/KVarN4 的 640 行同题三方答案、正确性组合及三组配对翻转全部保留；pattern 位顺序为 Periodic/KVarN2/KVarN4。Full/Base2/Native4 的辅助配对另表保存。最终数据仅用于报告，没有参与阈值拟合。",
        "", "独立身份核对见 kvarn_identity_audit.json。子 manifest 的 preflight/prepared 与原 candidate split 标签保留为已知旧字段，实际完成与同 64 图身份由 complete、运行记录、冻结样本哈希及 initial-logit 对齐另外核实。"]
    (args.analysis_dir/"quality19_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="OK",images=64,configurations=19,answers=12160,figures=len(result["figures"]))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
