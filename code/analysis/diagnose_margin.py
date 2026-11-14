"""Development-only margin/error association; no GPU or future-token alignment.

Only the first generated non-termination token of each answer is examined.
For replay policies the recorded margin is a trial observation while the answer
may have been changed by replay. All associations are application-level, using
each method's own history; they are not same-history causal estimates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

from analyze_cycle import load_jsonl, score_answer, sha256, write_csv, write_json


THRESHOLDS = (0.1, 0.5, 1., 2., 4., 8.)


def selected_references(path, allowed_ids):
    """Inspect only selected development IDs before decoding answer arrays."""
    references = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            match = re.search(r'"id"\s*:\s*"([^"\\]+)"', line)
            if not match or match.group(1) not in allowed_ids:
                continue
            row = json.loads(line)
            if row.get("split") not in (None, "development", "calibration", "train"):
                raise ValueError(f"Selected reference {row['id']} is outside development")
            for target in row["references"]:
                references[(row["id"], int(target["turn"]))] = target.get("answers", target.get("answer"))
    return references


def first_margin(turn, termination_ids):
    tokens, margins = turn.get("token_ids", []), turn.get("margin", [])
    if len(tokens) != len(margins):
        raise ValueError("Generated token IDs and trial margins must align within an answer")
    for position, (token, margin) in enumerate(zip(tokens, margins)):
        if int(token) in termination_ids:
            continue
        if not math.isfinite(float(margin)) or float(margin) < 0:
            raise ValueError("First-token margin must be finite and nonnegative")
        return int(position), int(token), float(margin)
    return None, None, None


def _estimate_ci(observed, samples):
    values = np.asarray(samples, dtype=float)
    valid = values[np.isfinite(values)]
    return dict(value=float(observed) if observed is not None and np.isfinite(observed) else None,
                ci_lower=float(np.quantile(valid, .025)) if len(valid) else None,
                ci_upper=float(np.quantile(valid, .975)) if len(valid) else None,
                valid_bootstrap_repeats=int(len(valid)))


def _divide(numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    return np.divide(numerator, denominator, out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan),
                     where=denominator != 0)


def detection_statistics(rows, label, thresholds=THRESHOLDS, repeats=2000, seed=1212):
    eligible_all = [r for r in rows if r.get(label) is not None]
    available = [r for r in eligible_all if r.get("margin") is not None]
    if not available:
        return dict(summary=dict(n_questions=0,n_images=0,n_positives=0,auroc=None), thresholds=[])
    images = sorted({r["image_id"] for r in eligible_all})
    image_index = {value:i for i,value in enumerate(images)}
    ids = np.array([image_index[r["image_id"]] for r in available])
    y = np.array([int(r[label]) for r in available])
    margin = np.array([r["margin"] for r in available], dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0,len(images),size=(repeats,len(images)))
    # One row per image-cluster draw. Every turn from the sampled image follows
    # it; duplicate images are multiplicity weights rather than extra subjects.
    weights = np.zeros((repeats,len(images)),dtype=int)
    np.add.at(weights,(np.repeat(np.arange(repeats),len(images)),indexes.reshape(-1)),1)
    n_by_image = np.bincount(ids,minlength=len(images))
    positive_by_image = np.bincount(ids,weights=y,minlength=len(images))
    all_ids=np.array([image_index[r["image_id"]] for r in eligible_all])
    all_y=np.array([int(r[label]) for r in eligible_all])
    all_n_by_image=np.bincount(all_ids,minlength=len(images))
    all_positive_by_image=np.bincount(all_ids,weights=all_y,minlength=len(images))
    n_boot = weights@n_by_image
    positive_boot = weights@positive_by_image
    all_n_boot=weights@all_n_by_image
    all_positive_boot=weights@all_positive_by_image
    threshold_rows=[]
    for threshold in thresholds:
        triggered = margin < threshold
        trigger_by_image=np.bincount(ids,weights=triggered.astype(int),minlength=len(images))
        tp_by_image=np.bincount(ids,weights=(triggered*y),minlength=len(images))
        triggered_boot=weights@trigger_by_image
        tp_boot=weights@tp_by_image
        tp, triggers, positives = int((triggered*y).sum()),int(triggered.sum()),int(y.sum())
        metrics = {
            "precision":(_divide(tp,triggers),_divide(tp_boot,triggered_boot)),
            "recall":(_divide(tp,positives),_divide(tp_boot,positive_boot)),
            "trigger_rate":(_divide(triggers,len(y)),_divide(triggered_boot,n_boot)),
            "high_margin_error_fraction":(_divide(positives-tp,positives),_divide(positive_boot-tp_boot,positive_boot)),
            "operational_recall":(_divide(tp,all_y.sum()),_divide(tp_boot,all_positive_boot)),
            "operational_trigger_rate":(_divide(triggers,len(all_y)),_divide(triggered_boot,all_n_boot)),
        }
        row=dict(threshold=float(threshold),rule="margin < threshold",n_questions=len(y),n_images=len(images),
                 positives=positives,triggered=triggers,true_positives=tp,false_positives=triggers-tp,
                 false_negatives=positives-tp,total_questions_including_missing=len(all_y),
                 positives_including_missing=int(all_y.sum()),
                 missing_margin_positives=int(all_y.sum())-positives,
                 operational_missing_margin_rule="no observed nontermination token => no content-margin trigger")
        for name,(observed,samples) in metrics.items():
            stats=_estimate_ci(float(observed),samples if len(images)>1 else [])
            row[name]=stats["value"]
            row.update({f"{name}_{key}":value for key,value in stats.items() if key!="value"})
        threshold_rows.append(row)

    # AUROC for risk score -margin; ties receive half credit. Score groups are
    # fixed, while image resampling changes their positive/negative weights.
    scores=-margin
    unique=np.unique(scores)
    pos_groups=np.stack([np.bincount(ids,weights=((scores==score)*y),minlength=len(images)) for score in unique])
    neg_groups=np.stack([np.bincount(ids,weights=((scores==score)*(1-y)),minlength=len(images)) for score in unique])
    def auc_from_counts(positive,negative):
        before=np.cumsum(negative,axis=-1)-negative
        return _divide(np.sum(positive*(before+.5*negative),axis=-1),
                       np.sum(positive,axis=-1)*np.sum(negative,axis=-1))
    observed=float(auc_from_counts(pos_groups.sum(1),neg_groups.sum(1)))
    auc_boot=auc_from_counts(weights@pos_groups.T,weights@neg_groups.T)
    auc_stats=_estimate_ci(observed,auc_boot if len(images)>1 else [])
    return dict(summary=dict(n_questions=len(y),n_images=len(images),n_positives=int(y.sum()),
        n_negatives=int((1-y).sum()),auroc=auc_stats["value"],auroc_ci_lower=auc_stats["ci_lower"],
        positives_including_missing=int(all_y.sum()),missing_margin_positives=int(all_y.sum())-int(y.sum()),
        auroc_ci_upper=auc_stats["ci_upper"],auroc_valid_bootstrap_repeats=auc_stats["valid_bootstrap_repeats"],
        auroc_risk_score="negative first-token margin",auroc_status="N/A_one_class" if len(np.unique(y))<2 else "ok"),
        thresholds=threshold_rows)


def load_diagnostics(paths,reference_path,termination_ids):
    records=[]
    settings={}
    for path in paths:
        manifest=path.parent/"manifest.json"
        arguments=json.loads(manifest.read_text(encoding="utf-8-sig")).get("arguments",{}) if manifest.exists() else {}
        settings[str(path)]={key:arguments.get(key) for key in ("max_tokens","max_patches","turns")}
        for record in load_jsonl(path):
            if record.get("split") not in ("development","calibration","train"):
                raise ValueError("This diagnostic accepts development records only; no held-out/final test records")
            if record.get("protocol")!="free_running":
                raise ValueError("Margin/error application association requires free-running records")
            record=dict(record,source_records=str(path),settings=settings[str(path)])
            records.append(record)
    allowed={r.get("native_session_id",r["session_id"]) for r in records}
    references=selected_references(reference_path,allowed)
    baseline={}
    for record in records:
        if record["method"]=="full" and record.get("policy","base")=="base":
            signature=(record["session_id"],record.get("model"),json.dumps(record["settings"],sort_keys=True),record.get("data_protocol"))
            if signature in baseline:
                raise ValueError("Duplicate compatible full baseline; specify a single reference run")
            baseline[signature]=record
    outputs=[]
    seen=set()
    for record in records:
        native=record.get("native_session_id",record["session_id"])
        offset=int(record.get("reference_turn_offset",0))
        signature=(record["session_id"],record.get("model"),json.dumps(record["settings"],sort_keys=True),record.get("data_protocol"))
        full=baseline.get(signature)
        full_turns={int(t["turn"]):t for t in full["turns"]} if full else {}
        full_source=full["source_records"] if full else None
        for turn in record["turns"]:
            number=int(turn["turn"]);reference_turn=number-offset
            if reference_turn<=0:
                continue
            target=references.get((native,reference_turn))
            if target is None:
                raise ValueError(f"Missing selected development reference: {native}, turn {reference_turn}")
            variant=f"{record['method']}:{record.get('policy','base')}"
            key=(record["session_id"],number,variant)
            if key in seen:
                raise ValueError(f"Duplicate diagnostic row: {key}")
            seen.add(key)
            index,token,margin=first_margin(turn,termination_ids)
            exact=score_answer(turn["answer"],target)["exact"]
            matched_full=full_turns.get(number)
            full_exact=score_answer(matched_full["answer"],target)["exact"] if matched_full else None
            outputs.append(dict(session_id=record["session_id"],image_id=record["image_id"],turn=number,
                reference_turn=reference_turn,method=record["method"],policy=record.get("policy","base"),variant=variant,
                focus_method=record["method"] in ("base1","base2"),source_records=record["source_records"],
                answer=turn["answer"],exact=exact,answer_error=1-int(exact),
                first_content_candidate_index=index,first_content_candidate_token_id=token,margin=margin,
                full_answer=matched_full["answer"] if matched_full else None,full_exact=full_exact,
                negative_flip=int(full_exact==1 and exact==0) if full_exact is not None else None,
                positive_flip=int(full_exact==0 and exact==1) if full_exact is not None else None,
                full_source_records=full_source,
                initial_logit_hash_equal=(record.get("initial_logits_sha256")==full.get("initial_logits_sha256")) if full else None,
                cross_run_full=full_source!=record["source_records"] if full else None,
                replay_count=turn.get("replay_count",0),
                margin_source="low-precision trial logits before any replay for this step",
                association_scope="each method's own free-running history; no later-token positional matching"))
    return outputs,settings


def analyze(rows,repeats,seed):
    summaries,thresholds=[],[]
    variants=sorted({r["variant"] for r in rows})
    for variant in variants:
        for scope in ("all_scored","later_turns"):
            selected=[r for r in rows if r["variant"]==variant and (scope=="all_scored" or r["reference_turn"]>=2)]
            for target,label in (("answer_error","answer_error"),("full_correct_method_wrong","negative_flip"),
                                 ("error_given_full_correct","answer_error")):
                eligible=[r for r in selected if target!="error_given_full_correct" or r["full_exact"]==1]
                stats=detection_statistics(eligible,label,repeats=repeats,seed=seed)
                common=dict(variant=variant,scope=scope,target=target)
                summaries.append(dict(common,**stats["summary"],
                    total_selected_questions=len(eligible),missing_margin=sum(r["margin"] is None for r in eligible),
                    total_answer_errors=sum(r["answer_error"] for r in eligible),
                    high_margin_error_definition="positive target with margin >= fixed threshold"))
                thresholds.extend(dict(common,**row) for row in stats["thresholds"])
    return summaries,thresholds


def plot_diagnostics(summaries,thresholds,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
        "legend.frameon":False})
    selected=[r for r in thresholds if r["scope"]=="later_turns" and r["variant"] in ("base1:base","base2:base")]
    if not selected:
        return []
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),layout="constrained")
    colors={"base1:base":"#C5A56B","base2:base":"#526D82"}
    for variant,color in colors.items():
        for target,style in (("answer_error","-"),("full_correct_method_wrong","--")):
            rows=sorted([r for r in selected if r["variant"]==variant and r["target"]==target],key=lambda r:r["threshold"])
            if not rows:
                continue
            if target=="full_correct_method_wrong" and not any(r["positives"] for r in rows):
                continue
            x=[r["threshold"] for r in rows]
            label=variant.split(":")[0]+(" errors" if target=="answer_error" else " negative flips")
            for ax,metric in zip(axes,("precision","recall","high_margin_error_fraction")):
                y=[r[metric] if r[metric] is not None else np.nan for r in rows]
                ax.plot(x,y,color=color,linestyle=style,marker="o",ms=3,lw=1,label=label)
                if target=="answer_error":
                    ax.fill_between(x,[r[f"{metric}_ci_lower"] if r[f"{metric}_ci_lower"] is not None else np.nan for r in rows],
                                      [r[f"{metric}_ci_upper"] if r[f"{metric}_ci_upper"] is not None else np.nan for r in rows],
                                    color=color,alpha=.10,linewidth=0)
    for ax,title,letter in zip(axes,("Trigger precision","Detection recall","Errors above threshold"),"abc"):
        ax.set_xscale("log",base=2);ax.set_xticks([.1,.5,1,2,4,8],[".1",".5","1","2","4","8"])
        ax.set_ylim(-.03,1.03);ax.set_title(title,fontsize=9);ax.set_xlabel("First-token margin threshold")
        ax.text(-.16,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=10)
    axes[0].set_ylabel("Fraction")
    handles,labels=axes[0].get_legend_handles_labels()
    notes=[]
    for variant in colors:
        sample=next((r for r in summaries if r["variant"]==variant and r["scope"]=="later_turns" and r["target"]=="answer_error"),None)
        if sample and sample["n_positives"]==sample["n_questions"]:
            notes.append(f"{variant.split(':')[0]}: all available later answers wrong (AUROC N/A)")
        sample=next((r for r in summaries if r["variant"]==variant and r["scope"]=="later_turns" and r["target"]=="full_correct_method_wrong"),None)
        if sample and sample["n_positives"]==0:
            notes.append(f"{variant.split(':')[0]}: no negative flips (AUROC N/A)")
    fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=7,
               title="\n".join(notes) or None,title_fontsize=6)
    fig.suptitle("Development association · later QA turns · available first-token margins",fontsize=9)
    directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for extension in ("png","pdf"):
        path=directory/f"first_margin_diagnostic.{extension}"
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white");outputs.append(str(path))
    plt.close(fig)
    return outputs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",nargs="+",type=Path,required=True)
    parser.add_argument("--references",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--termination-ids",nargs="+",type=int,required=True)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=1212)
    args=parser.parse_args(argv)
    rows,settings=load_diagnostics(args.records,args.references,set(args.termination_ids))
    summaries,thresholds=analyze(rows,args.bootstrap,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"turn_diagnostics.csv",rows)
    write_csv(args.output/"margin_summary.csv",summaries)
    write_csv(args.output/"threshold_metrics.csv",thresholds)
    figures=plot_diagnostics(summaries,thresholds,args.figures) if args.figures else []
    report=dict(status="OK",n_turns=len(rows),n_images=len({r["image_id"] for r in rows}),
        thresholds=list(THRESHOLDS),bootstrap=dict(unit="source image",repeats=args.bootstrap,seed=args.seed),
        termination_ids=args.termination_ids,
        first_token_definition="first generated token not in explicit termination IDs; tokenizer-free content candidate, not semantic-token certification",
        association_scope="each method's own history; margin policies may act after trial; later tokens are not position matched",
        final_test_access=False,selected_development_session_ids=sorted({r["session_id"] for r in rows}),
        settings=settings,summary=summaries,threshold_metrics=thresholds,figures=figures,
        input_sha256={str(path.resolve()):sha256(path) for path in args.records+[args.references]},
        script_sha256=sha256(Path(__file__)))
    write_json(args.output/"summary.json",report)
    print(json.dumps(dict(status="OK",questions=len(rows),images=report["n_images"],figures=len(figures))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
