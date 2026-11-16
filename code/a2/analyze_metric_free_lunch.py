"""Zero-bit metric-alignment accounting for the 1.8-bit KV format.

Two things are computed here, both from artefacts already on disk:

1. The exact excess risk of the L2-optimal sign code under a rank-one metric
   M = I + gamma*u u^T:

       J(s0) - D0 = gamma * (u . r_perp)^2 / (1 + gamma*beta),   beta = (u.s0)^2/d
       s0 = sign(r),  r_perp = r - (s0.r/d) s0,  D0 = ||r_perp||^2

   and its random-isotropy expectation Gap/D0 -> gamma/(d+gamma).

2. The measured spectrum of the query metric in
   results/round5_v1/spectrum_analysis/summary.json (per layer, per KV head),
   which fixes gamma for this model instead of assuming gamma = 1.

Also prints the free-improvement / metric-leakage map

       Gap/D0 = gamma * rho_M / (d + gamma*beta)

so that a single measured number rho_M (metric leakage of the deployed code)
translates into a predicted free gain.

Run:  python code/a2/analyze_metric_free_lunch.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECTRUM = ROOT / "results/round5_v1/spectrum_analysis/summary.json"
OUT = ROOT / "results/round6_v1/metric_free_lunch"


# ---------------------------------------------------------------- bit ledger
def bit_ledger():
    """Per visual token, per (layer, KV head): 400 bit base+gain, 144 bit residual stage."""
    k_code = 32 * 8          # 32 blocks of 4 dims, 8 bit per block, uint8 indices
    k_gain = 16              # one FP16 row gain per token
    v_code = 128             # 1 bit per channel, bit-packed
    residual_stage = 128 + 16  # 128 sign bit + one FP16 magnitude
    total = k_code + k_gain + v_code
    return dict(k_code=k_code, k_gain=k_gain, v_code=v_code,
                base=total, residual_stage=residual_stage,
                scalars_per_unit=256,
                base_bits_per_scalar=total / 256.0)


# ------------------------------------------------------------- closed forms
def gap_closed_form(r, u, gamma):
    """Exact J(sign(r)) - ||r_perp||^2 for M = I + gamma u u^T (numpy-free)."""
    d = len(r)
    s0 = [1.0 if x >= 0 else -1.0 for x in r]
    a = sum(s * x for s, x in zip(s0, r))
    b = sum(s * x for s, x in zip(s0, u))
    c = sum(x * y for x, y in zip(u, r))
    beta = b * b / d
    return gamma * (d * c - a * b) ** 2 / (d * d * (1.0 + gamma * beta)), a * a / d


def participation_ratio(ev):
    ev = [e for e in ev if e > 0]
    s = sum(ev)
    return s * s / sum(e * e for e in ev)


def spectrum_table():
    data = json.loads(SPECTRUM.read_text())
    rows = []
    for layer in data["per_layer"]:
        q = layer["query_regimes"]["prefill"]["eigenvalues"]
        k = layer["k_pca_eigenvalues"]
        for h, ev in enumerate(q):
            c = sum(ev) / len(ev)
            lam1 = ev[0]
            rows.append(dict(layer=layer["layer"], head=h,
                             mean_eig=c, lam1=lam1,
                             gamma=(lam1 - c) / c,
                             lam1_share=lam1 / (c * len(ev)),
                             q_pr=participation_ratio(ev),
                             k_pr=participation_ratio(k[h]),
                             k_top4_share=sum(k[h][:4]) / sum(k[h])))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = bit_ledger()
    rows = spectrum_table()
    g = [r["gamma"] for r in rows]
    qpr = [r["q_pr"] for r in rows]
    kpr = [r["k_pr"] for r in rows]
    share = [r["lam1_share"] for r in rows]

    # d = 128 attention channels per head
    d = 128
    gamma_med = st.median(g)
    # isotropic-random reference beta = 1/d
    beta_ref = 1.0 / d
    free_gain_iso = gamma_med / (d + gamma_med)     # Gap/D0 at rho_M = 1, beta = 1/d
    max_free_gain = 1.0 - 1.0 / (1.0 + gamma_med)   # rho_M -> infinity limit = 1 - lambda1/tr
    cut_of_metric = 1.0 - st.median(share)          # upper bound 1 - lambda1/tr

    rho_map = []
    for rho in (1, 2, 3, 5, 8, 12):
        gap = gamma_med * rho / (d * (1.0 + gamma_med * beta_ref))
        rho_map.append(dict(rho_m=rho, gap_over_d0=gap,
                            residual_cut_of_total=0.1228 * min(gap, 1.0)))

    summary = dict(
        bit_ledger=ledger,
        residual_share_of_budget=0.2210 / 1.7995,
        gamma=dict(min=min(g), median=gamma_med, max=max(g)),
        query_metric=dict(pr_min=min(qpr), pr_median=st.median(qpr), pr_max=max(qpr),
                          lambda1_share_median=st.median(share)),
        k_covariance=dict(pr_min=min(kpr), pr_median=st.median(kpr), pr_max=max(kpr)),
        free_gain=dict(isotropic_reference=free_gain_iso,
                       ceiling_1_minus_lambda1_share=cut_of_metric,
                       rho_map=rho_map),
        per_layer_head=rows,
        note=("gamma is computed from the UNCENTERED query second moment; it is an "
              "upper bound on the loss-relevant anisotropy. rho_M must be measured "
              "on the deployed code error, not inferred from q alone."),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    print("bit ledger (per token, per layer/KV-head, 256 scalars):")
    for k, v in ledger.items():
        print(f"  {k:18s} {v}")
    print()
    print(f"residual share of budget          {100 * 0.2210 / 1.7995:.2f}%")
    print(f"gamma (lambda1-c)/c               min {min(g):.2f}  median {gamma_med:.2f}  max {max(g):.2f}")
    print(f"query metric PR (out of 128)      min {min(qpr):.1f}  median {st.median(qpr):.1f}  max {max(qpr):.1f}")
    print(f"lambda1 / trace                   median {100 * st.median(share):.1f}%")
    print(f"K covariance PR                   min {min(kpr):.1f}  median {st.median(kpr):.1f}  max {max(kpr):.1f}")
    print()
    print("free gain map  Gap/D0 = gamma*rho_M/(d*(1+gamma*beta)),  beta = 1/d  => gamma*rho_M/(d+gamma)")
    print(f"  isotropic reference (rho_M=1)   {100 * free_gain_iso:.1f}% of L2 distortion")
    print(f"  ceiling 1 - lambda1/trace       {100 * cut_of_metric:.1f}%")
    print(f"  {'rho_M':>6s} {'Gap/D0':>9s} {'= share of total distortion':>30s}")
    for row in rho_map:
        print(f"  {row['rho_m']:6.1f} {100 * row['gap_over_d0']:8.1f}% {100 * row['residual_cut_of_total']:29.2f}%")
    print()
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
