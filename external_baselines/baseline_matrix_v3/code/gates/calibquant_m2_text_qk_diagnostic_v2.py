"""Single-input text-QK arithmetic diagnostic. Root schedules any GPU run.

V1 is preserved. Tensor hooks are narrowly scoped to the exact text matmul in
the existing bridge; original-arm tensors are returned without replacement.
The separately named control changes only that text matmul to FP32, on every
generation forward. Neither variant is a production method or formal score.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from gates.calibquant_m2_nonfinite_diagnostic_v1 import Trace, instrument, sha

TARGET = "textvqa:validation:36759"
ARMS = ["F_native", "C1_bias32_original", "C1_textQK_FP32_diagnostic"]


def validate_config(config):
    assert config["schema"] == "baseline_matrix.M2_text_QK_diagnostic.v2"
    assert config["model_id"] == "M2" and config["diagnostic_id"] == TARGET
    assert config["arms"] == ARMS and config["max_new_tokens"] == 64
    assert config["preprocessing"] == {"min_pixels": 200704, "max_pixels": 1003520}
    assert config["method"]["constructor_args"] == {"bits": 1, "reader_backend": "upstream_triton"}
    assert config["method"]["adapter_class"] == "CalibQuantFP32BiasAdapter"
    assert config["expected_original"]["actual_visual_token_count"] == 1230
    assert config["expected_original"]["input_tokens"] == 1274
    assert config["expected_original"]["question_suffix_tokens"] == 29
    assert config["trace_generation_forward_calls"] == 1
    assert not {"refs", "references", "answers", "labels", "scores"}.intersection(config)
    assert config["diagnostic_output"].endswith("/results/p0/calibquant_M2_text_QK_36759_diagnostic_v2")


class TextTrace(Trace):
    def __init__(self, torch, output, arm):
        super().__init__(torch, output, arm)
        self.pending_raw = self.pending_div = None
        self.current_scale = None
        self.text_operator_calls = 0
        self.text_reference = []
        self.text_private_capture = None

    def reference(self, q, kt, raw_half, fp32_dot):
        """Same actual operands, FP32 tensor and one FP64 scalar reference."""
        t = self.torch
        self.observe("text_QK_q_actual", q)
        self.observe("text_QK_textK_actual_transposed", kt)
        self.observe("text_QK_raw_FP16_matmul", raw_half,
                     kind="actual" if self.arm == "C1_bias32_original" else "diagnostic_unused_Half_operator")
        self.observe("text_QK_raw_FP32_matmul_same_operands", fp32_dot, kind="diagnostic_reference")
        scaled32 = fp32_dot / (1.0 / self.current_scale)
        self.observe("text_QK_FP32_dot_then_scale_reference", scaled32, kind="diagnostic_reference")
        bad = ~t.isfinite(raw_half)
        maximum = fp32_dot.abs().argmax()
        if bool(bad.any()):
            coordinate = bad.nonzero()[0].cpu().tolist()
        else:
            flat = int(maximum)
            shape = list(fp32_dot.shape)
            coordinate = []
            for extent in reversed(shape): coordinate.append(flat % extent); flat //= extent
            coordinate.reverse()
        b, h, query, key = coordinate
        qrow, krow = q[b, h, query], kt[b, h, :, key]
        gold = (qrow.double() * krow.double()).sum()
        one = {"layer": self.layer, "generation_forward": self.generation_call,
               "coordinate": coordinate, "coordinate_selection": "first_bad_raw_FP16" if bool(bad.any()) else "maxabs_FP32_reference",
               "raw_FP16": str(raw_half[tuple(coordinate)].item()),
               "raw_FP32": float(fp32_dot[tuple(coordinate)]), "raw_FP64_scalar_gold": float(gold),
               "scaled_FP32": float(scaled32[tuple(coordinate)]),
               "scaled_FP64_scalar_gold": float(gold / (1.0 / self.current_scale)),
               "scale": float(self.current_scale), "FP32_dot_maxabs": float(fp32_dot.abs().max()),
               "FP32_dot_count_abs_above_FP16_max65504": int((fp32_dot.abs() > 65504).sum()),
               "actual_raw_FP16_nonfinite_count": int(bad.sum()),
               "original_text_position": int(self.current_text_positions[key]),
               "original_query_position": int(self.current_query_positions[query]),
               "causal_position_allows_selected_key": bool(self.current_text_positions[key] <= self.current_query_positions[query]),
               "q_all_finite": bool(t.isfinite(q).all()), "textK_all_finite": bool(t.isfinite(kt).all()),
               "interpretation": "Compare finite operands and unscaled reference with Half range; do not infer internal accumulator precision solely from final stored output"}
        self.text_reference.append(one)
        if bool(bad.any()) and self.text_private_capture is None:
            path = self.output / (self.arm + "_text_QK_first_bad_private.pt")
            t.save({"scope": "PRIVATE_ONE_HEAD_ACTUAL_TEXT_OPERANDS_NOT_PUBLIC", "event": one,
                    "q_actual_one_head": q[b:b+1, h:h+1].detach().cpu(),
                    "textK_actual_one_head": kt[b:b+1, h:h+1].transpose(-1, -2).detach().cpu(),
                    "raw_FP16_one_head": raw_half[b:b+1, h:h+1].detach().cpu(),
                    "FP32_reference_one_head": fp32_dot[b:b+1, h:h+1].detach().cpu()}, path)
            self.text_private_capture = {"path": str(path), "bytes": path.stat().st_size,
                                         "sha256": sha(path), "publish": False}


@contextmanager
def text_qk_instrument(cache, trace):
    """Exact tensor-identity/geometry gate; all other matmul/div/casts delegate."""
    t = trace.torch
    old_mm, old_div, old_float = t.Tensor.__matmul__, t.Tensor.__truediv__, t.Tensor.float
    with instrument(cache, trace):
        base_read = cache.read

        def cache_read(unused, index, q, module, mask, scaling, original_sdpa, **kwargs):
            trace.current_scale = q.shape[-1] ** -.5 if scaling is None else scaling
            return base_read(index, q, module, mask, scaling, original_sdpa, **kwargs)

        def mm(left, right):
            active = (trace.layer is not None and left is trace.q and isinstance(right, t.Tensor)
                      and left.ndim == 4 and right.ndim == 4 and right.shape[:2] == left.shape[:2]
                      and right.shape[-2] == left.shape[-1]
                      and right.shape[-1] == cache.layers[trace.layer].keys.shape[-2])
            if not active: return old_mm(left, right)
            native = cache.layers[trace.layer].keys
            expected = native.repeat_interleave(left.shape[1] // native.shape[1], dim=1).transpose(-1, -2)
            same = (right == expected) | (t.isnan(right) & t.isnan(expected))
            assert right.dtype == expected.dtype and right.device == expected.device and right.stride() == expected.stride() and bool(same.all()), "INPUT_MISMATCH: text-QK RHS is not this layer's native exact-text K"
            trace.text_operator_calls += 1
            diagnostic = trace.arm == "C1_textQK_FP32_diagnostic"
            # Original arm always delegates Half. The control also computes it
            # during the first traced forward, as explicitly UNUSED evidence.
            raw_half = old_mm(left, right) if not diagnostic or trace.enabled else None
            fp32_dot = None
            if trace.enabled or diagnostic:
                fp32_dot = old_mm(old_float(left), old_float(right))
            if trace.enabled:
                trace.current_text_positions = cache.layers[trace.layer].text_positions
                trace.current_query_positions = cache.layers[trace.layer].query_positions
                trace.reference(left, right, raw_half, fp32_dot)
            result = fp32_dot if diagnostic else raw_half
            trace.pending_raw, trace.pending_div = result, None
            if trace.enabled:
                trace.observe("text_QK_actual_return_to_bridge_before_division", result,
                              kind="actual", diagnostic_FP32_replacement=diagnostic)
            return result

        def div(value, denominator):
            result = old_div(value, denominator)
            if value is trace.pending_raw:
                if trace.enabled:
                    trace.observe("text_QK_actual_division_result", result,
                                  actual_denominator=float(denominator))
                trace.pending_div = result
            return result

        def as_float(value, *args, **kwargs):
            result = old_float(value, *args, **kwargs)
            if value is trace.pending_div:
                if trace.enabled: trace.observe("text_QK_actual_final_float_cast", result)
                trace.pending_raw = trace.pending_div = None
            return result

        cache.read = types.MethodType(cache_read, cache)
        t.Tensor.__matmul__, t.Tensor.__truediv__, t.Tensor.float = mm, div, as_float
        try: yield
        finally:
            t.Tensor.__matmul__, t.Tensor.__truediv__, t.Tensor.float = old_mm, old_div, old_float
            cache.read = base_read


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--cpu-validate", action="store_true")
    args = parser.parse_args(); config = json.loads(args.job.read_text(encoding="utf-8")); validate_config(config)
    if args.cpu_validate:
        print(json.dumps({"status": "PASS_STATIC_CONFIG_ONLY", "actual_GPU_runs": 0, "arms": ARMS})); return
    assert args.output == Path(config["diagnostic_output"])
    for relative, expected in config["numeric_source_sha256"].items():
        assert sha(ROOT / relative) == expected, "Frozen numeric dependency changed: " + relative
    original_path = Path(config["original_eval_config"])
    assert sha(original_path) == config["original_eval_config_sha256"]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    for key in ("model_id", "model_revision", "model_path", "model_source_lock", "inputs", "max_new_tokens", "question_suffix", "preprocessing", "method"):
        assert original[key] == config[key], "Original behavior differs: " + key
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "baseline_matrix.M2_text_QK_diagnostic_result.v2", "status": "RUNNING", "arms": [],
              "scope": "ONE_EXISTING_INPUT_TEXT_QK_ARITHMETIC_DIAGNOSTIC_NOT_FORMAL_500_OR_SPEED", "references_read": False,
              "quality_scored": False, "driver_sha256": sha(__file__), "config": config, "started_unix": time.time(),
              "max_new_tokens": 64, "original_C1_output_replaced": False}
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from dataset_registry import load_inputs
        from model_registry import get_model_spec
        from evaluate import model_identity, tensor_sha
        from qwen_protocol import prepare_qwen_inputs, prefix_forward, generate_cached, generate_native
        from attention_bridge import attention_context
        from cache_bridge import HybridVisualCache
        from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasAdapter
        torch.set_grad_enabled(False); torch.manual_seed(0); torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6)
        spec = get_model_spec("M2"); report["model_identity"] = model_identity(config, spec)
        processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(config["model_path"], dtype=torch.float16,
                  device_map={"": 0}, local_files_only=True, attn_implementation="sdpa").eval()
        model.requires_grad_(False)
        assert all(p.dtype == torch.float16 for p in model.parameters()) and not getattr(model, "is_quantized", False)
        report["generation_native"] = model.generation_config.to_dict()
        rows = [r for r in load_inputs(config["inputs"]) if r["id"] == TARGET]; assert len(rows) == 1
        row = rows[0]; assert sha(row["image_path"]) == row["original_image_sha256"]
        report["input_source"] = {"id": TARGET, "image_sha256": row["original_image_sha256"], "registry_sha256": sha(config["inputs"])}
        with Image.open(row["image_path"]) as source: image = source.convert("RGB")
        for name in ARMS:
            arm = {"name": name, "status": "RUNNING"}; cache = batch = None; handles = []
            trace = TextTrace(torch, args.output, name); start = time.perf_counter()
            try:
                batch = prepare_qwen_inputs(processor, image, row["question"] + config["question_suffix"],
                          model=model, image_token_id=spec["image_token_id"], **config["preprocessing"])
                actual = {"input_ids_sha256": tensor_sha(batch.ids), "native_position_ids_sha256": tensor_sha(batch.positions),
                          "actual_visual_token_count": int(batch.visual_positions.numel()), "input_tokens": batch.full_length,
                          "question_suffix_tokens": batch.suffix_length}
                assert actual == config["expected_original"], "INPUT_MISMATCH: original failed input differs"
                arm["input_contract_exact"] = actual
                if name != "F_native":
                    prefix = prefix_forward(model, batch); cache = prefix.past_key_values; del prefix
                    batch.rope_deltas = model.model.rope_deltas; vp = batch.visual_positions
                    adapter = CalibQuantFP32BiasAdapter(1, "upstream_triton")
                    owners = [adapter.encode(layer.keys.index_select(-2, vp), layer.values.index_select(-2, vp),
                              positions=vp, layer_idx=i) for i, layer in enumerate(cache.layers)]
                    cache = HybridVisualCache(cache, vp, owners, reader_backend=None, consume_native=True); owners = None
                    ids, mask = batch.ids, batch.mask
                    batch.inputs = {"input_ids": ids, "attention_mask": mask}; batch.prefix_inputs = {}
                    batch.positions = batch.positions.detach().cpu(); batch.visual_positions = batch.visual_positions.detach().cpu()
                    gc.collect()
                def before(module, unused): trace.generation_call += 1
                def after(module, unused, result):
                    event = trace.observe("outer_forward_logits_original", getattr(result, "logits", None)); trace.outer_calls.append(event)
                    if event.get("nonfinite_is_failure", True): raise RuntimeError("NONFINITE_LOGITS: raw outer-forward logits are missing/nonfinite")
                handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
                with attention_context(model, cache=cache) if cache is not None else nullcontext():
                    with text_qk_instrument(cache, trace) if cache is not None else nullcontext():
                        generated = (generate_cached(model, batch, cache, max_new_tokens=64, return_dict_in_generate=False)
                                     if cache is not None else generate_native(model, batch, max_new_tokens=64, return_dict_in_generate=False))
                ids = generated[0, batch.full_length:].detach().cpu().tolist(); assert ids
                arm.update(status="PASS", output_token_ids=ids, output_tokens=len(ids), raw_prediction=processor.decode(ids, skip_special_tokens=True))
                del generated
            except Exception as exc: arm.update(status="FAILED_RUN", error=str(exc), traceback=traceback.format_exc())
            finally:
                for handle in handles: handle.remove()
                torch.cuda.synchronize()
                arm.update(first_bad_actual_operator_boundary=trace.first_bad_actual, first_bad_shadow_candidate=trace.first_bad_shadow,
                           stage_trace=trace.events, outer_forward_calls=trace.outer_calls, text_QK_reference=trace.text_reference,
                           text_operator_calls=trace.text_operator_calls, private_text_capture=trace.text_private_capture,
                           private_first_bad_capture=trace.first_bad_capture, instrumented_wall_ms=(time.perf_counter() - start) * 1000,
                           text_FP32_control_applies_all_generation_forwards=name == "C1_textQK_FP32_diagnostic",
                           diagnostic_reference_Half_operator_never_changes_original_arm=name != "C1_textQK_FP32_diagnostic")
                report["arms"].append(arm)
                with (args.output / "arms.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(arm) + "\n")
                print(json.dumps({"arm": name, "status": arm["status"], "first_bad_actual": trace.first_bad_actual}), flush=True)
                cache = batch = trace = None; gc.collect(); torch.cuda.empty_cache()
        report["status"] = "DIAGNOSTIC_RAW_COMPLETE" if len(report["arms"]) == 3 else "DIAGNOSTIC_FAILED"
    except Exception as exc: report.update(status="DIAGNOSTIC_FAILED", error=str(exc), traceback=traceback.format_exc()); raise
    finally:
        report["completed_unix"] = time.time()
        with (args.output / "complete.json").open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2)


if __name__ == "__main__": main()
