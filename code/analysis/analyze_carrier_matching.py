"""Separate analysis of norm-matched carrier controls at one declared base bitwidth.

Does not pool v1 base2 observations with base1 norm-matched trajectories. Labels
such as 'matched' are audited against actual post-write L2 ratios and positions.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

import analyze_cycle as cycle
from analyze_mechanisms import carrier_analysis,distribution


def analyze(directory,repeats=2000,seed=1212):
    files=[directory/name for name in ("surgery_records.jsonl","controls.jsonl","sessions.jsonl")]
    records,controls,sessions=[cycle.load_jsonl(path) for path in files]
    if any(r.get("split") not in ("development","calibration","train") for r in records):
        raise ValueError("Norm-matching analysis is development-only")
    bits=sorted({r.get("base_bits") for r in records})
    if len(bits)!=1 or bits[0] is None:
        raise ValueError("A declared single base_bits value is required; analyze bitwidths separately")
    eligible=[r for r in records if r.get("location_pair_complete") is True and r.get("matching_case_status")=="matched"]
    omitted=[dict(intervention_id=r["intervention_id"],session_id=r["session_id"],base_bits=r.get("base_bits"),
                  matching_case_status=r.get("matching_case_status"),location_pair_complete=r.get("location_pair_complete"))
             for r in records if r not in eligible]
    result=carrier_analysis(eligible,controls,sessions,repeats,seed)
    metadata={r["intervention_id"]:dict(base_bits=r["base_bits"],matching_case_status=r["matching_case_status"],
        location_pair_complete=r["location_pair_complete"],dose_kind=r.get("dose_kind"),
        delta_scale=r["surgery"].get("delta_scale"),dose_rounded_to_zero=r["surgery"].get("dose_rounded_to_zero"),
        exact_reference_endpoint=r["surgery"].get("exact_reference_endpoint"),
        injection_l2_ratio_to_alpha_carrier=r["surgery"].get("injection_l2_ratio_to_alpha_carrier")) for r in eligible}
    for collection in ("surgery_points","surgery_audits"):
        for row in result[collection]:
            row.update(metadata[row["intervention_id"]])
    for row in result["matched_controls"]:
        denominator=row["carrier_intervention_l2"]
        row["actual_control_to_carrier_l2_ratio"]=row["control_intervention_l2"]/denominator if denominator else None
        row["base_bits"]=bits[0]
    cases=defaultdict(list)
    for row in records:
        cases[(row["session_id"],row["carrier_id"],row["write_turn"])].append(row)
    case_rows=[]
    for key,rows in cases.items():
        ratios=[r["surgery"].get("injection_l2_ratio_to_alpha_carrier") for r in rows if r["alpha"]>0]
        case_rows.append(dict(session_id=key[0],carrier_id=key[1],write_turn=key[2],base_bits=bits[0],
            n_interventions=len(rows),matching_status_counts=dict(Counter(r.get("matching_case_status") for r in rows)),
            all_location_pairs_complete=all(r.get("location_pair_complete") is True for r in rows),
            actual_injection_ratio=distribution(ratios)))
    audit_rows=result["surgery_audits"]
    amp_summary=[]
    for position in sorted({r["position_type"] for r in audit_rows}):
        for axis in ("k","v"):
            rows=[r for r in audit_rows if r["position_type"]==position and r["axis"]==axis and r["alpha"]>0]
            amp_summary.append(dict(position_type=position,axis=axis,
                ratio_to_alpha_carrier=distribution(r["injection_l2_ratio_to_alpha_carrier"] for r in rows),
                zero_positive_dose_cases=sum(r["intervention_l2"]==0 for r in rows),
                rounded_to_zero_cases=sum(bool(r["dose_rounded_to_zero"]) for r in rows)))
    contrasts=[]
    for axis in ("k","v"):
        for control in ("prompt_control","random_control"):
            rows=[r for r in result["matched_controls"] if r["axis"]==axis and r["control"]==control and r["alpha"]==1]
            stats=cycle.cluster_bootstrap(((r["image_id"],r["delta_kl"]) for r in rows),repeats=repeats,seed=seed)
            contrasts.append(dict(axis=axis,control=control,alpha=1,**stats,
                carrier_mean_kl=float(np.mean([r["carrier_kl"] for r in rows])) if rows else None,
                control_mean_kl=float(np.mean([r["control_kl"] for r in rows])) if rows else None,
                carrier_larger_cases=sum(r["delta_kl"]>1e-12 for r in rows),
                control_larger_cases=sum(r["delta_kl"]< -1e-12 for r in rows),
                actual_l2_ratio=distribution(r["actual_control_to_carrier_l2_ratio"] for r in rows)))
    points=result["surgery_points"]
    alpha1=[r for r in result["kv_contrasts"] if r["position_type"]=="carrier" and r["alpha"]==1]
    summary=dict(base_bits=bits[0],n_sessions=len({r["session_id"] for r in records}),n_cases=len(cases),
        n_interventions=len(records),n_matched_interventions=len(eligible),n_future_responses=len(points),
        matching_case_status_counts=dict(Counter(r.get("matching_case_status") for r in records)),
        all_location_pairs_complete=all(r.get("location_pair_complete") is True for r in records),
        all_token_schedules_unchanged=all(r["token_schedule_unchanged"] and not r["token_ids_modified"] for r in audit_rows),
        all_position_hashes_unchanged=all(r["position_hash_unchanged"] and r["positions_unchanged"] for r in audit_rows),
        all_matched_position_counts=all(r["equal_position_count"] for r in result["matched_controls"]),
        all_matched_layer_sets=all(r["equal_layer_set"] for r in result["matched_controls"]),
        all_matched_write_bytes=all(r["equal_value_write_bytes"] for r in result["matched_controls"]),
        actual_amplitude_ratios=amp_summary,alpha1_carrier_minus_control=contrasts,
        alpha0_max_logit_difference=max((r["max_abs_logits"] for r in points if r["alpha"]==0),default=None),
        total_future_argmax_comparison_flips=sum(r["argmax_flips"] for r in points),
        total_future_compared_token_positions=sum(r["token_count"] for r in points),
        alpha1_carrier_v_greater_count=sum(r["v_minus_k_kl"]>1e-12 for r in alpha1),
        alpha1_carrier_k_greater_count=sum(r["v_minus_k_kl"]< -1e-12 for r in alpha1),
        n_dose_curves=len(result["dose_monotonicity"]),
        n_nonmonotone_dose_curves=sum(not r["nondecreasing"] for r in result["dose_monotonicity"]),
        interpretation="synthetic state-response displacement, not accuracy; base1 canonical text and chosen spans differ from v1 base2; no pooled bitwidth mean",
        norm_matching_scope="within each axis, locations target carrier injection norm; post-FP16-write ratios reported explicitly; K and V norms are not matched to one another",
        omitted=omitted,input_sha256={str(path.resolve()):cycle.sha256(path) for path in files})
    result["matching_cases"]=case_rows
    return result,summary


def plot(result,summary,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    if not result["surgery_points"]:
        return []
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout="constrained")
    colors={"carrier":"#526D82","prompt_control":"#7FA580","random_control":"#C5A56B"}
    audits=result["surgery_audits"]
    for position,color in colors.items():
        for axis,style in (("k","-"),("v","--")):
            xs=sorted({r["alpha"] for r in audits if r["alpha"]>0})
            means,low,high=[],[],[]
            for alpha in xs:
                values=[r["injection_l2_ratio_to_alpha_carrier"] for r in audits
                        if r["position_type"]==position and r["axis"]==axis and r["alpha"]==alpha]
                means.append(float(np.median(values)));low.append(float(np.min(values)));high.append(float(np.max(values)))
            axes[0].plot(xs,means,color=color,linestyle=style,marker="o",ms=2.5,
                         label=f"{position.replace('_control','')} {axis.upper()}")
            axes[0].fill_between(xs,low,high,color=color,alpha=.06,linewidth=0)
    axes[0].axhline(1,color="#ADB5BD",lw=.7)
    axes[0].set_xlabel("Dose α");axes[0].set_ylabel("Actual injected L2 / target carrier L2")
    axes[0].set_title("Post-write norm matching",fontsize=9);axes[0].legend(fontsize=5.5,ncol=2)
    rows=[r for r in result["matched_controls"] if r["alpha"]==1]
    for control,color in (("prompt_control",colors["prompt_control"]),("random_control",colors["random_control"])):
        chosen=[r for r in rows if r["control"]==control]
        axes[1].scatter([r["carrier_kl"] for r in chosen],[r["control_kl"] for r in chosen],
                        color=color,s=15,alpha=.65,label=control.replace("_control"," control"))
    maximum=max(max(r["carrier_kl"],r["control_kl"]) for r in rows)
    axes[1].plot([0,maximum],[0,maximum],color="#ADB5BD",lw=.8,linestyle=":")
    axes[1].set_xlabel("Carrier-location output KL");axes[1].set_ylabel("Matched-control output KL")
    axes[1].set_title("α = 1 · same axis and write bytes",fontsize=9);axes[1].legend(fontsize=6)
    for ax,letter in zip(axes,"ab"):
        ax.text(-.16,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    fig.suptitle(f"Separate base{summary['base_bits']} development · {summary['n_sessions']} images · {summary['n_cases']} lexical cases",fontsize=9)
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for extension in ("png","pdf"):
        path=directory/f"carrier_normmatched_base{summary['base_bits']}.{extension}"
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");paths.append(str(path))
    plt.close(fig)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    result,summary=analyze(args.input_dir,args.bootstrap,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in result.items():
        if isinstance(rows,list):
            cycle.write_csv(args.output/f"{name}.csv",rows)
    summary["figures"]=plot(result,summary,args.figures) if args.figures else []
    summary["script_sha256"]=cycle.sha256(Path(__file__))
    cycle.write_json(args.output/"summary.json",summary)
    print(json.dumps(dict(base_bits=summary["base_bits"],sessions=summary["n_sessions"],
                         cases=summary["n_cases"],matched_interventions=summary["n_matched_interventions"],
                         figures=len(summary["figures"]))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
