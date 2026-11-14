"""Development-only analysis of real intra-step residual probes and read budgets.

The probe signal is enhanced-versus-base output displacement, not FP16 error.
Logical reads include the extra base/text read needed to compute that signal.
No bank, runner, policy, or frozen final-evaluation configuration is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

import analyze_cycle as cycle
from diagnose_margin import selected_references


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:12]


def distribution(values):
    values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return dict(n=0,mean=None,median=None,minimum=None,maximum=None,p05=None,p95=None)
    return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
                minimum=min(values),maximum=max(values),p05=float(np.quantile(values,.05)),
                p95=float(np.quantile(values,.95)))


def load_records(paths):
    records=[]
    completeness=[]
    for path in paths:
        settings=cycle.infer_settings(path)
        completeness.append(cycle.input_completeness(path))
        for raw in cycle.load_jsonl(path):
            if raw.get("split") not in ("development","calibration","train"):
                raise ValueError("Probe exploration analysis accepts development records only; no final-test tuning")
            if raw.get("protocol")!="free_running":
                raise ValueError("Use free-running development records for probe quality/resource comparisons")
            context=dict(model=raw.get("model"),dataset=raw.get("dataset"),data_protocol=raw.get("data_protocol"),
                         protocol=raw.get("protocol"),settings=settings)
            profile=cycle.storage_profile(raw)
            records.append(dict(raw,source_records=str(path),context_id=digest_json(context),context=context,
                                storage_id=digest_json(profile),storage_profile=profile))
    return records,completeness


def analyze(records,references,repeats=2000,seed=1212,threshold_grid=()):
    events,skipped,sessions,phase_costs,turns=[],[],[],[],[]
    seen=set()
    for record in records:
        variant=f"{record['method']}:{record.get('policy','base')}"
        key=(record["context_id"],record["storage_id"],record["session_id"],variant)
        if key in seen:
            raise ValueError(f"Duplicate probe session: {key}")
        seen.add(key)
        common=dict(context_id=record["context_id"],storage_id=record["storage_id"],
            session_id=record["session_id"],image_id=record["image_id"],method=record["method"],
            policy=record.get("policy","base"),variant=variant,source_records=record["source_records"])
        resources=cycle.summarize_resources(record)
        sessions.append(dict(common,**resources))
        summary=record.get("intra_step_probe_summary") or {}
        for event in record.get("intra_step_probes",[]):
            decision=event.get("decision") or {}
            charges=sum(int(charge.get("bytes",0)) for charge in event.get("charges",[]))
            charged=int(event.get("charged_bytes",0))
            rows=dict(common,event_id=event.get("event_id"),turn=event.get("turn"),phase=event.get("phase"),
                probe_layer=event.get("probe_layer",summary.get("probe_layer")),threshold=event.get("threshold",summary.get("threshold")),
                input_start=event.get("input_start"),input_count=event.get("input_count"),
                probe_observed=event.get("probe_observed",False),
                threshold_met=bool(decision.get("threshold_met",False)),upgrade=bool(decision.get("upgrade",False)),
                use_enhanced_probe=bool(decision.get("use_enhanced_probe",False)),reason=decision.get("reason"),
                relative_l2=decision.get("relative_l2"),rms=decision.get("rms"),delta_l2=decision.get("delta_l2"),
                base_l2=decision.get("base_l2"),base_rms=decision.get("base_rms"),
                base_norm_below_epsilon=decision.get("base_norm_below_epsilon"),finite_signal=decision.get("finite_signal"),
                controller_two_read_charged_bytes=charged,
                controller_charge_list_sum_bytes=charges,controller_charge_list_exact=charges==charged,
                controller_cumulative_exact=(int(event.get("spent_bytes_at_start",0))+charged==int(event["spent_bytes_now"]))
                                            if "spent_bytes_now" in event else None,
                controller_budget_bytes=event.get("budget_bytes"),controller_budget_overrun_bytes=event.get("budget_overrun_bytes"),
                planned_upgrade_bytes=decision.get("planned_upgrade_bytes"),
                feature_wall_ms_with_sync=decision.get("feature_wall_ms_including_synchronization"),
                feature_availability=decision.get("feature_availability",event.get("feature_availability")),
                signal_meaning=decision.get("signal_meaning"),
                upgraded_scope="later-layer request; actual executed residual bytes are accounted at session/phase level")
            events.append(rows)
        for skip in record.get("intra_step_probe_skipped",[]):
            skipped.append(dict(common,**skip,threshold=summary.get("threshold"),probe_layer=skip.get("layer",summary.get("probe_layer"))))
        for phase in sorted({e.get("phase") for e in record.get("read_events",[])}):
            source=[e for e in record["read_events"] if e.get("phase")==phase]
            costs={}
            for kind in ("committed","probe","replay","repair"):
                subset=[e for e in source if e.get("pass_kind")==kind]
                for column in cycle.READ_COLUMNS:
                    costs[f"{kind}_{column}"]=sum(int(e.get(column,0)) for e in subset)
            costs["computed_extra_logical_read_bytes"]=sum(int(e.get("residual_bytes",0))+
                (int(e.get("base_visual_bytes",0))+int(e.get("text_bytes",0)) if e.get("pass_kind") in ("probe","replay","repair") else 0)
                for e in source)
            costs["raw_logical_read_bytes"]=sum(int(e.get(k,0)) for e in source for k in cycle.READ_COLUMNS)
            phase_costs.append(dict(common,phase=phase,**costs))
        native=record.get("native_session_id",record["session_id"])
        offset=int(record.get("reference_turn_offset",0))
        for turn in record["turns"]:
            number=int(turn["turn"])-offset
            if number<=0:
                continue
            target=references.get((native,number))
            if target is None:
                raise ValueError(f"Missing selected development reference: {native}, turn {number}")
            turns.append(dict(common,reference_turn=number,answer=turn["answer"],
                              **cycle.score_answer(turn["answer"],target)))

    groups=defaultdict(list)
    for event in events:
        key=tuple(event[k] for k in ("context_id","storage_id","variant","probe_layer","phase","threshold"))
        groups[key].append(event)
    summaries,sweep=[],[]
    for key,rows in sorted(groups.items(),key=lambda item:repr(item[0])):
        common=dict(zip(("context_id","storage_id","variant","probe_layer","phase","threshold"),key))
        observed=[r for r in rows if r["probe_observed"]]
        rates=cycle.cluster_bootstrap(((r["image_id"],float(r["upgrade"])) for r in observed),repeats=repeats,seed=seed)
        context_skips=[r for r in skipped if r["context_id"]==key[0] and r["storage_id"]==key[1]
                       and r["variant"]==key[2] and r["probe_layer"]==key[3] and r["phase"]==key[4]]
        summaries.append(dict(common,n_events=len(rows),n_observed=len(observed),n_skipped=len(context_skips),
            upgrades=sum(r["upgrade"] for r in observed),upgrade_rate=rates["mean"],
            upgrade_rate_ci_lower=rates["ci_lower"],upgrade_rate_ci_upper=rates["ci_upper"],n_images=rates["n_images"],
            threshold_met=sum(r["threshold_met"] for r in observed),
            relative_l2_distribution=distribution(r["relative_l2"] for r in observed),
            rms_distribution=distribution(r["rms"] for r in observed),
            invalid_signals=sum(r["finite_signal"] is False for r in observed),
            near_zero_denominators=sum(bool(r["base_norm_below_epsilon"]) for r in observed),
            controller_charges_exact=all(r["controller_charge_list_exact"] is not False and r["controller_cumulative_exact"] is not False for r in rows)))
        candidates=sorted(set(threshold_grid) | ({float(key[5])} if key[5] is not None else set()))
        for threshold in candidates:
            eligible=[r for r in observed if r["relative_l2"] is not None and math.isfinite(float(r["relative_l2"]))]
            rate=cycle.cluster_bootstrap(((r["image_id"],float(r["relative_l2"]>=threshold)) for r in eligible),repeats=repeats,seed=seed)
            sweep.append(dict(common,diagnostic_threshold=threshold,threshold_rate=rate["mean"],
                ci_lower=rate["ci_lower"],ci_upper=rate["ci_upper"],n_events=len(eligible),n_images=rate["n_images"],
                scope="descriptive threshold crossing among observed probes; not a replayed policy or quality estimate"))

    shared=defaultdict(list)
    for row in turns:
        shared[(row["context_id"],row["storage_id"])].append(row)
    curves=[]
    for (context,storage),rows in shared.items():
        variants=sorted({r["variant"] for r in rows})
        same_keys=None
        for variant in variants:
            keys={(r["session_id"],r["reference_turn"]) for r in rows if r["variant"]==variant}
            same_keys=keys if same_keys is None else same_keys&keys
        for variant in variants:
            chosen=[r for r in rows if r["variant"]==variant and (r["session_id"],r["reference_turn"]) in (same_keys or set())]
            ids={r["session_id"] for r in chosen}
            resources=[r for r in sessions if r["context_id"]==context and r["storage_id"]==storage
                       and r["variant"]==variant and r["session_id"] in ids]
            stats=cycle.cluster_bootstrap(((r["image_id"],r["exact"]) for r in chosen),repeats=repeats,seed=seed)
            if not chosen:
                continue
            cost=lambda name:float(np.mean([r[name] for r in resources if r.get(name) is not None])) if any(r.get(name) is not None for r in resources) else None
            curves.append(dict(context_id=context,storage_id=storage,variant=variant,**stats,n_sessions=len(ids),
                mean_extra_read_bytes=cost("computed_extra_logical_read_bytes"),
                mean_raw_read_bytes=cost("logical_total_read_bytes"),
                mean_owned_visual_bytes=cost("owned_initial_all_visual_bytes"),
                mean_owned_visual_ratio=cost("owned_initial_all_visual_ratio"),
                mean_committed_residual_bytes=cost("committed_residual_bytes"),
                mean_probe_base_reread_bytes=cost("probe_base_visual_bytes"),
                mean_probe_text_reread_bytes=cost("probe_text_bytes"),
                mean_probe_residual_bytes=cost("probe_residual_bytes"),
                mean_replay_all_bytes=cost("replay_logical_bytes"),
                mean_generated_tokens=cost("generated_tokens"),
                mean_session_ms=cost("session_ms"),
                mean_probe_feature_wall_ms_with_sync=cost("probe_feature_wall_ms_with_sync"),
                all_ledger_equal=all(r["extra_read_ledger_equal"] is not False for r in resources),
                all_recorded_ledger_exact=all(r["read_ledger_exact_recorded"] is not False for r in resources),
                all_budgets_ok=all(r["read_budget_ok_computed"] is not False for r in resources),
                pairing="same question intersection within an exact stored-owner/kernel layout"))
    return dict(probe_events=events,probe_skipped=skipped,probe_sessions=sessions,
                phase_read_costs=phase_costs,probe_layer_phase_summary=summaries,
                descriptive_threshold_sweep=sweep,scored_turns=turns,same_storage_quality=curves)


def plot(result,directory):
    if not result["probe_sessions"]:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    colors=["#526D82","#7FA580","#C5A56B","#9C8CB4","#CE8C82","#6FA3B5","#A87656"]
    grouped=defaultdict(list)
    for row in result["same_storage_quality"]:
        grouped[(row["context_id"],row["storage_id"])].append(row)
    for (context,storage),rows in grouped.items():
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout="constrained")
        for i,row in enumerate(rows):
            y=row["mean"]*100
            error=None if row["ci_lower"] is None else [[y-row["ci_lower"]*100],[row["ci_upper"]*100-y]]
            axes[0].errorbar(row["mean_extra_read_bytes"]/2**20,y,yerr=error,fmt="o",color=colors[i%len(colors)],
                            capsize=2,label=row["variant"],ms=5)
        axes[0].set_xlabel("Extra logical reads / session (MiB)")
        axes[0].set_ylabel("Normalized exact accuracy (%)")
        axes[0].set_ylim(-2,102);axes[0].legend(fontsize=6)
        terms=[("mean_committed_residual_bytes","Upgrade residual","#7FA580"),
               ("mean_probe_base_reread_bytes","Probe base reread","#526D82"),
               ("mean_probe_text_reread_bytes","Probe text reread","#A7B9C5"),
               ("mean_probe_residual_bytes","Probe residual","#9C8CB4"),
               ("mean_replay_all_bytes","Replay reads","#C5A56B")]
        bottom=np.zeros(len(rows))
        for name,label,color in terms:
            values=np.array([r.get(name) or 0 for r in rows])/2**20
            if np.any(values):
                axes[1].bar(range(len(rows)),values,bottom=bottom,color=color,label=label,width=.65)
                bottom+=values
        axes[1].set_xticks(range(len(rows)),[r["variant"].split(":",1)[1] for r in rows],rotation=25,ha="right",fontsize=6)
        axes[1].set_ylabel("Extra logical reads / session (MiB)");axes[1].legend(fontsize=5.7)
        axes[0].set_title("Identical stored representation",fontsize=9)
        axes[1].set_title("Probe base/text rereads are paid",fontsize=9)
        for ax,letter in zip(axes,"ab"):
            ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
        fig.suptitle(f"Development · {rows[0]['n_images']} image clusters · visual owner ratio {rows[0]['mean_owned_visual_ratio']:.4f}",fontsize=9)
        for extension in ("png","pdf"):
            path=directory/f"same_storage_probe_{context}_{storage}.{extension}"
            fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");outputs.append(str(path))
        plt.close(fig)

    groups=defaultdict(list)
    for event in result["probe_events"]:
        if event["probe_observed"] and event["relative_l2"] is not None:
            groups[(event["context_id"],event["storage_id"])].append(event)
    for (context,storage),events in groups.items():
        categories=sorted({(e["probe_layer"],e["phase"],e["threshold"]) for e in events})
        data=[[e["relative_l2"] for e in events if (e["probe_layer"],e["phase"],e["threshold"])==key] for key in categories]
        fig,ax=plt.subplots(figsize=(max(4.5,min(9,.65*len(categories))),3),layout="constrained")
        ax.boxplot(data,showfliers=True,flierprops=dict(markersize=2,marker="."),medianprops=dict(color="#526D82"))
        ax.scatter(range(1,len(categories)+1),[key[2] for key in categories],marker="_",color="#C58D83",s=45,label="Configured threshold")
        ax.set_xticks(range(1,len(categories)+1),[f"L{layer} {phase}\nt={threshold:g}" for layer,phase,threshold in categories],
                      rotation=20,ha="right",fontsize=6)
        ax.set_ylabel("Relative L2: enhanced vs base output")
        ax.set_title("Observed development probe signal · not FP16 error",fontsize=9);ax.legend(fontsize=6)
        for extension in ("png","pdf"):
            path=directory/f"probe_delta_{context}_{storage}.{extension}"
            fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");outputs.append(str(path))
        plt.close(fig)
    return outputs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",nargs="+",type=Path,required=True)
    parser.add_argument("--references",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--threshold-grid",nargs="*",type=float,default=[])
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    records,completeness=load_records(args.records)
    ids={r.get("native_session_id",r["session_id"]) for r in records}
    references=selected_references(args.references,ids)
    result=analyze(records,references,args.bootstrap,args.seed,args.threshold_grid)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in result.items():
        cycle.write_csv(args.output/f"{name}.csv",rows)
    sessions=result["probe_sessions"]
    audit=dict(n_sessions=len(sessions),all_computed_ledger_equal=all(r["extra_read_ledger_equal"] is not False for r in sessions),
        n_sessions_with_reported_ledger=sum(r["extra_read_ledger_equal"] is not None for r in sessions),
        all_recorded_ledger_exact=all(r["read_ledger_exact_recorded"] is not False for r in sessions),
        all_computed_budgets_ok=all(r["read_budget_ok_computed"] is not False for r in sessions),
        n_sessions_with_read_budget=sum(r["read_budget_bytes"] is not None for r in sessions),
        ledger_mismatches=[r for r in sessions if r["extra_read_ledger_equal"] is False],
        budget_overruns=[r for r in sessions if r["read_budget_ok_computed"] is False])
    report=dict(status="EMPTY_INPUT" if not records else ("PARTIAL" if any(c["status"]!="COMPLETE" for c in completeness) else "OK"),
        completeness=completeness,budget_and_ledger_audit=audit,
        probe_layer_phase_summary=result["probe_layer_phase_summary"],same_storage_quality=result["same_storage_quality"],
        n_probe_events=len(result["probe_events"]),n_probe_skipped=len(result["probe_skipped"]),
        contexts={r["context_id"]:r["context"] for r in records},
        storage_profiles={r["storage_id"]:r["storage_profile"] for r in records},
        feature_scope="current same-query base/residual correction magnitude; no FP16 reference or future outcomes used online",
        cost_scope="global extra = all residual reads + probe/replay/repair base and text rereads; not measured HBM bandwidth",
        controller_cost_scope="two-read local probe charge includes a normal first read; do not add it wholesale to global extra",
        quality_scope="same-question, same-owner/kernel development comparison under each policy's own generated history",
        final_test_access=False,input_sha256={str(p.resolve()):cycle.sha256(p) for p in args.records+[args.references]},
        script_sha256=cycle.sha256(Path(__file__)),bootstrap=dict(unit="source image",repeats=args.bootstrap,seed=args.seed))
    report["figures"]=plot(result,args.figures) if args.figures else []
    cycle.write_json(args.output/"summary.json",report)
    print(json.dumps(dict(status=report["status"],sessions=len(sessions),probes=report["n_probe_events"],figures=len(report["figures"]))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
