"""TriAxialKV-style tag allocator (paper mechanism port, NOT official code).

Given calibrated per-tag attention distortion D_tag(2/4) and explicit deployed
byte costs, select 2 or 4 bits per tag. Exhaustive enumeration is exact for a
small active tag set. Calibration, template tagging, quantization and SGLang
serving are NOT reproduced here. Costs must come from code/bytes.py and include
metadata; this module never derives byte costs from nominal bit widths.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math


@dataclass(frozen=True)
class TagOption:
    tag: str
    distortion_2: float
    distortion_4: float
    bytes_2: int
    bytes_4: int


def allocate_tags(options: list[TagOption], budget_bytes: int,
                  exact_max_tags: int = 20) -> dict:
    if budget_bytes < 0:
        raise ValueError("negative budget")
    if len({o.tag for o in options}) != len(options):
        raise ValueError("duplicate tags")
    for o in options:
        if (o.bytes_2 < 0 or o.bytes_4 < o.bytes_2 or
                not math.isfinite(o.distortion_2) or
                not math.isfinite(o.distortion_4) or
                min(o.distortion_2, o.distortion_4) < 0):
            raise ValueError(f"invalid option: {o.tag}")
    base = sum(o.bytes_2 for o in options)
    if base > budget_bytes:
        raise ValueError("budget cannot hold the all-2bit configuration")
    if len(options) <= exact_max_tags:
        best = None
        for upgrade in product((False, True), repeat=len(options)):
            cost = sum(o.bytes_4 if u else o.bytes_2 for o, u in zip(options, upgrade))
            if cost > budget_bytes:
                continue
            loss = sum(o.distortion_4 if u else o.distortion_2 for o, u in zip(options, upgrade))
            candidate = (loss, cost, upgrade)
            if best is None or candidate < best:
                best = candidate
        loss, cost, upgrade = best
        solver = "exact_enumeration"
    else:
        upgrade = [False] * len(options)
        cost = base
        # The paper's greedy benefit/cost principle. Negative measured gains
        # are retained as evidence and are never silently clipped in the data.
        order = sorted(range(len(options)), key=lambda i: (
            -(options[i].distortion_2 - options[i].distortion_4) /
            max(1, options[i].bytes_4 - options[i].bytes_2), options[i].tag))
        for i in order:
            o = options[i]
            if o.distortion_4 < o.distortion_2 and cost + o.bytes_4 - o.bytes_2 <= budget_bytes:
                upgrade[i] = True
                cost += o.bytes_4 - o.bytes_2
        loss = sum(o.distortion_4 if u else o.distortion_2 for o, u in zip(options, upgrade))
        solver = "greedy_approximation"
    return {"label": "TriAxialKV-style tag allocator [mechanism-port]",
            "solver": solver, "allocated_bytes": cost, "budget_bytes": budget_bytes,
            "calibrated_distortion_sum": loss,
            "bits": {o.tag: 4 if u else 2 for o, u in zip(options, upgrade)}}


if __name__ == "__main__":
    import json
    # Unequal byte costs make ranking by distortion alone incorrect.
    toy = [TagOption("older_image", 9, 1, 3, 11),
           TagOption("current_user", 7, 2, 2, 5),
           TagOption("assistant", 6, 1, 2, 5)]
    result = allocate_tags(toy, 13)
    assert result["bits"] == {"older_image": 2, "current_user": 4, "assistant": 4}
    assert result["allocated_bytes"] == 13
    assert result["calibrated_distortion_sum"] == 12
    print(json.dumps({"status": "PASS_SYNTHETIC_BUDGETED_ALLOCATOR", **result}))
