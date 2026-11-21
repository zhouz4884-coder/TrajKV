"""Root-scheduled one-input M2 numerical diagnostic, never a quality benchmark.

Production functions are delegated unchanged. Source-order intermediate traces
are reconstructed from the ACTUAL packed-kernel returns after each original
owner call. Instrumentation adds synchronizations/temporary storage and cannot
be used to report latency, peak memory, or a repaired production score.
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
TARGET = "textvqa:validation:36759"
ARMS = ["F_native", "C1_bias32_original", "C1_affine32_diagnostic"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    forbidden = {"refs", "references", "reference_answers", "answers", "labels", "scores"}
    assert not forbidden.intersection(config), "Reference/score fields prohibited"
    assert config["schema"] == "baseline_matrix.M2_nonfinite_diagnostic.v1"
    assert config["model_id"] == "M2" and config["diagnostic_id"] == TARGET
    assert config["arms"] == ARMS and config["max_new_tokens"] == 64
    assert config["preprocessing"] == {"min_pixels": 200704, "max_pixels": 1003520}
    assert config["method"]["constructor_args"] == {"bits": 1, "reader_backend": "upstream_triton"}
    assert config["method"]["adapter_class"] == "CalibQuantFP32BiasAdapter"
    assert config["expected_original"]["actual_visual_token_count"] == 1230
    assert config["expected_original"]["input_tokens"] == 1274
    assert config["expected_original"]["question_suffix_tokens"] == 29
    assert config["trace_generation_forward_calls"] == 1
    assert Path(config["diagnostic_output"]).as_posix().endswith("/results/p0/calibquant_M2_C1_nonfinite_36759_diagnostic_v1")


class Trace:
    def __init__(self, torch, output, arm):
        self.torch, self.output, self.arm = torch, output, arm
        self.events, self.outer_calls = [], []
        self.first_bad = None
        self.first_bad_actual = None
        self.first_bad_shadow = None
        self.layer = self.owner = self.q = None
        self.generation_call = 0
        self.qk_raw, self.wv_raw = [], []
        self.mode = None
        self.first_bad_capture = None

    @property
    def enabled(self):
        return self.generation_call == 1

    def observe(self, stage, value, *, kind="actual", allow_negative_inf=False, **extra):
        t = self.torch
        if not isinstance(value, t.Tensor):
            event = {"stage": stage, "present_tensor": False, "layer": self.layer,
                     "generation_forward": self.generation_call, "kind": kind, **extra}
            bad = True
        else:
            finite = t.isfinite(value)
            nan, pos, neg = t.isnan(value), t.isposinf(value), t.isneginf(value)
            bad_mask = nan | pos | (neg if not allow_negative_inf else t.zeros_like(neg))
            bad = bool(bad_mask.any())
            event = {"stage": stage, "present_tensor": True, "layer": self.layer,
                     "generation_forward": self.generation_call, "kind": kind,
                     "shape": list(value.shape), "dtype": str(value.dtype),
                     "nan": int(nan.sum()), "posinf": int(pos.sum()), "neginf": int(neg.sum()),
                     "zero_count": int((value == 0).sum()), "nonfinite_is_failure": bad,
                     "negative_inf_allowed_as_mask": allow_negative_inf, **extra}
            if bool(finite.any()):
                vals = value[finite].float()
                event.update(finite_min=float(vals.min()), finite_max=float(vals.max()),
                             finite_maxabs=float(vals.abs().max()))
            if bad:
                coordinate = bad_mask.nonzero()[0].cpu().tolist()
                event["first_bad_coordinate"] = coordinate
                event["first_bad_value"] = str(value[tuple(coordinate)].detach().cpu().item())
        self.events.append(event)
        if bad and kind == "actual" and self.first_bad_actual is None:
            self.first_bad_actual = dict(event)
        if bad and kind != "actual" and self.first_bad_shadow is None:
            self.first_bad_shadow = dict(event)
        if bad and self.first_bad is None:
            self.first_bad = dict(event)
            self._capture(value, event)
        return event

    def _capture(self, value, event):
        """One bounded head of existing operands, private remote data disk only."""
        t = self.torch
        payload = {"scope": "PRIVATE_DIAGNOSTIC_OPERANDS_NOT_PUBLIC_OR_QUALITY",
                   "arm": self.arm, "event": event}
        coordinate = event.get("first_bad_coordinate", [0, 0])
        head = coordinate[1] if len(coordinate) > 1 else 0
        if self.q is not None:
            head = min(head, self.q.shape[1] - 1)
            query = coordinate[2] if len(coordinate) > 2 else 0
            query = min(query, self.q.shape[-2] - 1)
            payload["q_one_head_one_query"] = self.q[:, head:head+1, query:query+1].detach().cpu()
            payload.update(query_head=head, query_index=query)
        if self.owner is not None:
            group = self.q.shape[1] // self.owner.k_packed.shape[1] if self.q is not None else 1
            kv_head = min(head // group, self.owner.k_packed.shape[1] - 1)
            payload["owner_one_KV_head"] = {
                name: (x[:, kv_head:kv_head+1].detach().cpu() if x.ndim == 4 else x.detach().cpu())
                for name, x in self.owner.owner_tensors().items()}
            payload.update(kv_head=kv_head, bits=self.owner.bits)
        if isinstance(value, t.Tensor):
            payload["bad_value_first8192_flat"] = value.detach().reshape(-1)[:8192].cpu()
        path = self.output / (self.arm + "_first_bad_private.pt")
        t.save(payload, path)
        self.first_bad_capture = {"path": str(path), "sha256": sha(path),
                                  "bytes": path.stat().st_size, "publish": False}

    def shadow_read(self, owner, q, actual, kwargs):
        from methods.calibquant_adapter import repeat_kv
        t = self.torch
        group = owner._groups(q.shape[1])
        ks, kz = repeat_kv(owner.k_scale, group), repeat_kv(owner.k_zero, group)
        factor = kwargs.get("softmax_scale")
        factor = owner.head_dim ** -.5 if factor is None else factor
        obs = lambda name, x, **kw: self.observe(name, x, kind="source_order_shadow_from_actual_operands", **kw)
        obs("q_times_scale_FP16", q * ks)
        raw = t.cat(self.qk_raw, dim=2)
        self.observe("packed_QK_raw_hard_FP16_output", raw)
        affine = self.arm == "C1_affine32_diagnostic"
        if affine:
            products = ks.float() * kz.float() * q.float()
            obs("bias_products_FP32", products)
            bias_sum = products.sum(-1, keepdim=True)
            obs("bias_reduction_FP32", bias_sum)
            bias = bias_sum * factor
            obs("bias_rescale_FP32", bias)
            logits = raw.float() * factor - bias
        else:
            raw_scaled = raw * factor
            obs("raw_QK_times_factor_FP16", raw_scaled)
            products = ks * kz * q
            obs("bias_products_original_FP16_left_association", products)
            bias_sum = products.sum(-1, keepdim=True, dtype=t.float32)
            obs("bias_reduction_FP32", bias_sum)
            bias_rescaled = bias_sum / (1.0 / factor)
            obs("bias_division_FP32", bias_rescaled)
            bias = bias_rescaled.to(q.dtype)
            obs("bias_cast_FP16", bias)
            logits = raw_scaled - bias
        obs("uncalibrated_visual_logits", logits)
        if kwargs.get("calibrate", True) and owner.bits <= 2:
            high, low = logits.amax(-1, keepdim=True), logits.amin(-1, keepdim=True)
            obs("calibration_high", high); obs("calibration_low", low)
            span = high - low
            obs("calibration_span_high_minus_low", span)
            eps = t.tensor(1e-8, dtype=span.dtype, device=span.device)
            obs("calibration_epsilon_in_arithmetic_dtype", eps, diagnostic_constant=True)
            denominator = span + 1e-8
            obs("calibration_denominator_span_plus_epsilon", denominator)
            numerator = logits - low
            obs("calibration_numerator_logits_minus_low", numerator)
            normalized = numerator / denominator
            obs("calibration_normalized", normalized)
            gain = high + owner.norm_offset_max - low - owner.norm_offset_min
            obs("calibration_gain_original_association", gain)
            logits = normalized * gain + low + owner.norm_offset_min
            obs("calibration_shadow_final", logits)
        self.observe("owner_read_return_original", actual)
        equal = bool(t.equal(logits, actual))
        matching_nan = bool(t.all((logits == actual) | (t.isnan(logits) & t.isnan(actual))))
        self.events.append({"stage": "shadow_vs_actual_read", "layer": self.layer,
                            "bitwise_equal_torch_equal": equal, "same_values_and_nan_positions": matching_nan,
                            "replaces_original_output": False})
        if self.first_bad_shadow is not None and self.first_bad_shadow.get("layer") == self.layer:
            self.first_bad_shadow["read_shadow_matches_actual_values_and_nan_positions"] = matching_nan
        if self.first_bad is not None and self.first_bad.get("kind") != "actual" and self.first_bad.get("layer") == self.layer:
            self.first_bad["read_shadow_matches_actual_values_and_nan_positions"] = matching_nan

    def shadow_wv(self, owner, weights, actual):
        from methods.calibquant_adapter import repeat_kv
        t = self.torch
        raw = t.cat(self.wv_raw, dim=2)
        self.observe("packed_WV_raw_hard_FP16_output", raw)
        group = owner._groups(weights.shape[1])
        vs, vz = repeat_kv(owner.v_scale, group), repeat_kv(owner.v_zero, group)
        affine = self.arm == "C1_affine32_diagnostic"
        if affine:
            vs, vz = vs.float(), vz.float()
            left = raw.float() * vs
            right = (vs * vz) * weights.float().sum(-1, keepdim=True)
        else:
            left = raw * vs
            right = vs * vz * weights.sum(-1, keepdim=True)
        self.observe("WV_affine_left", left, kind="source_formula_shadow")
        self.observe("WV_affine_right", right, kind="source_formula_shadow")
        self.observe("WV_affine_subtract_before_return_cast", left - right, kind="source_formula_shadow")
        self.observe("owner_weighted_value_return_original", actual)
        shadow_return = (left - right).to(weights.dtype) if affine else left - right
        matching = bool(t.all((shadow_return == actual) | (t.isnan(shadow_return) & t.isnan(actual))))
        self.events.append({"stage": "shadow_vs_actual_weighted_value", "layer": self.layer,
                            "same_values_and_nan_positions": matching, "replaces_original_output": False})
        for candidate in (self.first_bad, self.first_bad_shadow):
            if candidate is not None and candidate.get("kind") != "actual" and candidate.get("layer") == self.layer and candidate["stage"].startswith("WV_"):
                candidate["weighted_shadow_matches_actual_values_and_nan_positions"] = matching


@contextmanager
def instrument(cache, trace):
    """Intercept original returns. No replacement algebra for either arm."""
    import torch
    from methods.calibquant_adapter import _load_upstream
    upstream = _load_upstream()
    old_qk, old_wv, old_softmax = upstream.fused_kernel, upstream.fused_kernel_wv, torch.softmax
    old_cache_read = cache.read
    owner_originals = []

    def raw_qk(*args, **kwargs):
        result = old_qk(*args, **kwargs)
        if trace.enabled and trace.mode == "read": trace.qk_raw.append(result.detach())
        return result

    def raw_wv(*args, **kwargs):
        result = old_wv(*args, **kwargs)
        if trace.enabled and trace.mode == "wv": trace.wv_raw.append(result.detach())
        return result

    def softmax(value, *args, **kwargs):
        active = trace.enabled and trace.layer is not None
        if active:
            trace.observe("joint_visual_plus_text_logits_actual", value, allow_negative_inf=True)
            count = trace.owner.visual_tokens
            trace.observe("joint_text_logits_actual_masked", value[..., count:], allow_negative_inf=True)
        result = old_softmax(value, *args, **kwargs)
        if active: trace.observe("joint_softmax_actual_before_FP16_cast", result)
        return result

    def read_cache(unused, layer_idx, q, *args, **kwargs):
        trace.layer, trace.owner, trace.q = layer_idx, cache.layers[layer_idx].visual_owner, q
        try:
            if trace.enabled: trace.observe("native_post_RoPE_q", q)
            result = old_cache_read(layer_idx, q, *args, **kwargs)
            if trace.enabled: trace.observe("joint_attention_output_original", result[0])
            return result
        finally:
            trace.layer = trace.owner = trace.q = None

    for layer in cache.layers:
        owner = layer.visual_owner
        old_read, old_weighted = owner.read, owner.weighted_value
        owner_originals.append((owner, old_read, old_weighted))

        def read_owner(q, _owner=owner, _original=old_read, **kwargs):
            if not trace.enabled: return _original(q, **kwargs)
            trace.mode, trace.qk_raw = "read", []
            try:
                actual = _original(q, **kwargs)
                trace.shadow_read(_owner, q, actual, kwargs)
                return actual
            finally: trace.mode, trace.qk_raw = None, []

        def weighted(weights, _owner=owner, _original=old_weighted):
            if not trace.enabled: return _original(weights)
            trace.observe("joint_visual_probabilities_actual_FP16", weights)
            trace.mode, trace.wv_raw = "wv", []
            try:
                actual = _original(weights)
                trace.shadow_wv(_owner, weights, actual)
                return actual
            finally: trace.mode, trace.wv_raw = None, []

        owner.read, owner.weighted_value = read_owner, weighted
    upstream.fused_kernel, upstream.fused_kernel_wv, torch.softmax = raw_qk, raw_wv, softmax
    cache.read = types.MethodType(read_cache, cache)
    try: yield
    finally:
        cache.read = old_cache_read
        upstream.fused_kernel, upstream.fused_kernel_wv, torch.softmax = old_qk, old_wv, old_softmax
        for owner, old_read, old_weighted in owner_originals:
            owner.read, owner.weighted_value = old_read, old_weighted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-validate", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.job.read_text(encoding="utf-8")); validate_config(config)
    if args.cpu_validate:
        print(json.dumps({"status": "PASS_STATIC_CONFIG_ONLY", "target": TARGET, "arms": ARMS,
                          "actual_GPU_runs": 0, "original_input_runtime_check_pending": True})); return
    assert args.output == Path(config["diagnostic_output"]), "Diagnostic output must be the fixed new data-disk run"
    for relative, expected in config["numeric_source_sha256"].items():
        assert sha(ROOT / relative) == expected, "Frozen numeric source changed: " + relative
    original_config = Path(config["original_eval_config"])
    assert sha(original_config) == config["original_eval_config_sha256"]
    original = json.loads(original_config.read_text(encoding="utf-8"))
    for key in ("model_id", "model_revision", "model_path", "model_source_lock", "inputs", "max_new_tokens", "question_suffix", "preprocessing", "method"):
        assert original[key] == config[key], "Original behavior differs: " + key
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "baseline_matrix.M2_nonfinite_diagnostic_result.v1", "status": "RUNNING",
              "scope": "REAL_SINGLE_INPUT_DIAGNOSTIC_NOT_FORMAL_500_QA_OR_SPEED", "references_read": False,
              "quality_scored": False, "repairs_original_C1": False, "arms": [], "started_unix": time.time(),
              "driver_sha256": sha(__file__), "job_config_sha256": sha(args.job),
              "config": config, "actual_QA_inputs": 1, "expected_arm_executions": 3}
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
        from methods.calibquant_fp32_affine_adapter import CalibQuantFP32AffineAdapter
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
        rows = [r for r in load_inputs(config["inputs"]) if r["id"] == TARGET]
        assert len(rows) == 1
        row = rows[0]
        assert sha(row["image_path"]) == row["original_image_sha256"]
        report["input_source"] = {"id": TARGET, "original_image_sha256": row["original_image_sha256"],
                                  "input_registry_sha256": sha(config["inputs"])}
        with Image.open(row["image_path"]) as raw: image = raw.convert("RGB")
        for name in ARMS:
            arm = {"name": name, "status": "RUNNING"}; cache = batch = None
            trace = Trace(torch, args.output, name); started = time.perf_counter(); handles = []
            try:
                batch = prepare_qwen_inputs(processor, image, row["question"] + config["question_suffix"], model=model,
                          image_token_id=spec["image_token_id"], **config["preprocessing"])
                actual = {"input_ids_sha256": tensor_sha(batch.ids), "native_position_ids_sha256": tensor_sha(batch.positions),
                          "actual_visual_token_count": int(batch.visual_positions.numel()), "input_tokens": batch.full_length,
                          "question_suffix_tokens": batch.suffix_length}
                assert actual == config["expected_original"], "INPUT_MISMATCH: original failed input differs"
                arm["input_contract_exact"] = actual
                def before(module, unused): trace.generation_call += 1
                def after(module, unused, value):
                    logits = getattr(value, "logits", None)
                    event = trace.observe("outer_forward_logits_original", logits)
                    trace.outer_calls.append(event)
                    if event.get("nonfinite_is_failure", True):
                        raise RuntimeError("NONFINITE_LOGITS: raw outer-forward logits are missing/nonfinite")
                if name != "F_native":
                    prefix = prefix_forward(model, batch); cache = prefix.past_key_values; del prefix
                    batch.rope_deltas = model.model.rope_deltas
                    adapter = (CalibQuantFP32BiasAdapter(1, "upstream_triton") if name == "C1_bias32_original"
                               else CalibQuantFP32AffineAdapter(1, "upstream_triton"))
                    owners = []
                    for i, layer in enumerate(cache.layers):
                        vp = batch.visual_positions
                        owner = adapter.encode(layer.keys.index_select(-2, vp), layer.values.index_select(-2, vp), positions=vp, layer_idx=i)
                        trace.layer, trace.owner = i, owner
                        for field in ("k_scale", "k_zero", "v_scale", "v_zero"):
                            trace.observe("encoded_" + field, getattr(owner, field))
                        owners.append(owner)
                    trace.layer = trace.owner = None
                    cache = HybridVisualCache(cache, vp, owners, reader_backend=None, consume_native=True)
                    arm["layer0_owner_shapes"] = {k: list(v.shape) for k, v in owners[0].owner_tensors().items()}
                    owners = None
                    ids, mask = batch.ids, batch.mask
                    batch.inputs = {"input_ids": ids, "attention_mask": mask}; batch.prefix_inputs = {}
                    batch.positions = batch.positions.detach().cpu(); batch.visual_positions = batch.visual_positions.detach().cpu()
                    gc.collect()
                handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
                context = attention_context(model, cache=cache) if cache is not None else nullcontext()
                with context:
                    with instrument(cache, trace) if cache is not None else nullcontext():
                        generated = (generate_cached(model, batch, cache, max_new_tokens=64, return_dict_in_generate=False)
                                     if cache is not None else generate_native(model, batch, max_new_tokens=64, return_dict_in_generate=False))
                ids = generated[0, batch.full_length:].detach().cpu().tolist(); assert ids
                arm.update(status="PASS", output_token_ids=ids, output_tokens=len(ids),
                           raw_prediction=processor.decode(ids, skip_special_tokens=True))
                del generated
            except Exception as exc:
                arm.update(status="FAILED_RUN", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            finally:
                for handle in handles: handle.remove()
                torch.cuda.synchronize()
                arm.update(first_bad_in_traced_source_order=trace.first_bad, private_first_bad_capture=trace.first_bad_capture,
                           first_bad_actual_operator_boundary=trace.first_bad_actual,
                           first_bad_shadow_candidate=trace.first_bad_shadow,
                           attribution_rule="Shadow candidates require matching original reader return; otherwise only actual boundary is established",
                           first_generation_forward_stage_trace=trace.events, outer_forward_calls=trace.outer_calls,
                           instrumented_wall_ms=(time.perf_counter() - started) * 1000,
                           trace_limit="all layers of first generation forward only; outer logits on subsequent max64 freegen calls",
                           source_formula_shadows_replace_outputs=False)
                report["arms"].append(arm)
                with (args.output / "arms.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(arm) + "\n")
                cache = batch = trace = None; gc.collect(); torch.cuda.empty_cache()
                print(json.dumps({"id": TARGET, "arm": name, "status": arm["status"],
                                  "first_bad": arm.get("first_bad_in_traced_source_order")}), flush=True)
        report["status"] = "DIAGNOSTIC_RAW_COMPLETE" if len(report["arms"]) == 3 else "DIAGNOSTIC_FAILED"
        report["arm_failures_preserved"] = sum(a["status"] != "PASS" for a in report["arms"])
    except Exception as exc:
        report.update(status="DIAGNOSTIC_FAILED", error=str(exc), traceback=traceback.format_exc()); raise
    finally:
        report["completed_unix"] = time.time()
        with (args.output / "complete.json").open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2)


if __name__ == "__main__": main()
