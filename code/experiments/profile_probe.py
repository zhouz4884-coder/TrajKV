"""One real-query base/refined attention pair for externally controlled NCU.

No frozen executor is changed. Profile only the first decode read at layer 7
of asym:probe_l7_t0.02, then exit after two calls. This is a hardware-counter
side check, not a benchmark or a latency result. NCU must use cache-control
none and clock-control none; any replay count must be reported by the caller.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import run_cycle


class PairCaptured(BaseException):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "samples", "legacy-scripts", "kivi-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-patches", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.turns = 1
    args.max_tokens = 1
    args.margin = 0.
    args.max_replays = 0
    args.save_states = False
    args.read_budget_bytes = None
    args.gain_table_data = None
    args.hide_visual_after_description = False
    args.description_tokens = 1
    run_cycle.install_legacy(args.legacy_scripts)
    import p202_fused
    from transformers import AutoProcessor, AutoModelForImageTextToText
    selected = json.loads(args.samples.read_text())[0]
    manifest = {"method": "asym", "policy": "probe_l7_t0.02", "session_id": selected["id"],
                "sample_manifest_sha256": run_cycle.digest(args.samples),
                "code_sha256": {str(q): run_cycle.digest(q) for q in (
                    Path(__file__), Path(run_cycle.__file__), Path(run_cycle.__file__).with_name("bank.py"),
                    args.legacy_scripts / "p202_fused.py", args.kivi_source)},
                "scope": "two_attention_kernel_calls_same_real_query_not_end_to_end_speed",
                "cache_flush_requested": False, "clock_change_requested": False,
                "profiler_replay_behavior": "controlled_by_ncu_cli; inspect raw profiler output",
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    original_bank = run_cycle.ProgressiveBank
    original_attention = p202_fused.attention
    state = {"bank": None, "calls": 0, "started": False, "first_q": None,
             "first_q_ptr": None, "base_output": None, "capture": []}

    def capture_bank(*a, **kw):
        bank = original_bank(*a, **kw)
        state["bank"] = bank
        return bank

    def profiled_attention(q, bank_state, *a, **kw):
        bank = state["bank"]
        selected_call = (bank is not None and bank.phase == "decode" and
                         bank_state is bank.states[7] and state["calls"] < 2)
        if not selected_call:
            return original_attention(q, bank_state, *a, **kw)
        refinement = kw.get("refinement")
        if state["calls"] == 0:
            if refinement is not None:
                raise RuntimeError("First selected call is not the low-precision base read")
            state["first_q"] = q
            state["first_q_ptr"] = q.data_ptr()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()  # cudaProfilerStart, no explicit cache flush
            state["started"] = True
        elif (q is not state["first_q"] or q.data_ptr() != state["first_q_ptr"]
              or refinement is None):
            raise RuntimeError("Second selected call is not a refined read of the same unchanged q")
        output = original_attention(q, bank_state, *a, **kw)
        state["calls"] += 1
        state["capture"].append({"call": state["calls"], "refined": refinement is not None,
                                 "q_shape": list(q.shape), "q_stride": list(q.stride()),
                                 "state_length": bank_state.length, "text_length": bank_state.text_length,
                                 "visual_tokens": int(bank_state.visual_pos.numel())})
        if state["calls"] == 1:
            state["base_output"] = output
            return output
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()  # cudaProfilerStop after the second attention kernel
        state["started"] = False
        delta = output.float() - state["base_output"].float()
        record = {"status": "PROFILE_TWO_KERNEL_CAPTURE_COMPLETED", "calls": state["capture"],
                  "same_query_object_and_storage": True,
                  "input_mutation_check": "reader source does not write q; identical object and pointer; no extra clone between the reads",
                  "q_sha256": hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest(),
                  "base_refined_output_relative_l2": float(delta.norm() / output.float().norm().clamp_min(1e-12)),
                  "no_quality_or_speed_conclusion": True}
        (args.output / "capture.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"status": record["status"], "captured_calls": 2}), flush=True)
        raise PairCaptured()

    run_cycle.ProgressiveBank = capture_bank
    p202_fused.attention = profiled_attention
    try:
        torch.manual_seed(1212)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.float16,
            device_map="cuda:0", attn_implementation="sdpa").eval()
        with torch.inference_mode():
            try:
                run_cycle.run_session(model, processor, selected, args, "asym", "probe_l7_t0.02")
            except PairCaptured:
                pass
        if state["calls"] != 2:
            raise RuntimeError(f"Expected exactly two selected calls, observed {state['calls']}")
    finally:
        if state["started"]:
            torch.cuda.profiler.stop()
        p202_fused.attention = original_attention
        run_cycle.ProgressiveBank = original_bank


if __name__ == "__main__":
    main()
