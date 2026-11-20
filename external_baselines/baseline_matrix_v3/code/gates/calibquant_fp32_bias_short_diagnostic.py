"""One C2/source0 run: source-precision fidelity vs ideal FP64 instrument.

No task scoring or parameter changes. Uses prior passed same-prefix FP16
controls; isolates layer26, then checks finite complete suffix/free generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "C2_SOURCE0_SAME_OPERATOR_PRECISION_DIAGNOSTIC_NO_QUALITY_SCORE",
              "job": job, "started_unix": time.time(), "quality_unlocked_by_this_diagnostic_alone": False}
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from native_model_smoke import native_rows
        from model_registry import get_model_spec
        from m1_protocol import prepare_m1_inputs, m1_prefix_forward, m1_forward_suffix, m1_generate_cached
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context
        from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasAdapter
        from calibquant_fp32_bias_oracle import source_dense_attention
        from calibquant_operator_reference import source_operator_attention, difference_summary
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_capability() == (8, 6)
        spec = get_model_spec("M1")
        processor = AutoProcessor.from_pretrained(job["model_path"], local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(job["model_path"], dtype=torch.float16,
                    device_map={"": 0}, local_files_only=True, attn_implementation="sdpa").eval()
        source = native_rows({**job, "limit": 1000})[0]
        image = Image.open(source["image_path"]).convert("RGB")
        batch = prepare_m1_inputs(processor, image, source["question"], image_token_id=spec["image_token_id"], max_patches=8)
        report.update(model_revision=spec["revision"], source_id=source["source_id"],
            image_sha256=hashlib.sha256(Path(source["image_path"]).read_bytes()).hexdigest(),
            visual_tokens=int(batch.visual_positions.numel()), suffix_tokens=batch.suffix_length,
            torch=torch.__version__, transformers=transformers.__version__, hardware=torch.cuda.get_device_name())
        prefix = m1_prefix_forward(model, batch)
        adapter = CalibQuantFP32BiasAdapter(2, "upstream_triton")
        owners = [adapter.encode(layer.keys.index_select(-2, batch.visual_positions), layer.values.index_select(-2, batch.visual_positions))
                  for layer in prefix.past_key_values.layers]
        cache = HybridVisualCache(prefix.past_key_values, batch.visual_positions, owners, reader_backend="upstream_triton")
        del prefix
        original_read = cache.read
        calls = {i: [] for i in range(28)}
        captured = False
        def diagnostic_read(index, q, *read_args, **read_kwargs):
            nonlocal captured
            actual = original_read(index, q, *read_args, **read_kwargs)
            assert bool(torch.isfinite(actual[0]).all()), f"Nonfinite actual attention layer {index}"
            calls[index].append({"query_tokens": q.shape[-2], "finite": True})
            if index == 26 and q.shape[-2] > 1 and not captured:
                captured = True
                layer = cache.layers[index]
                input_evidence = args.output / "source0_C2_layer26_inputs.pt"
                torch.save({"q": q.detach(), "owner": layer.visual_owner.owner_tensors(), "text_keys": layer.keys.detach(),
                    "text_values": layer.values.detach(), "visual_positions": layer.visual_positions, "text_positions": layer.text_positions,
                    "query_positions": layer.query_positions, "attention_mask": read_args[1], "scaling": read_args[2],
                    "actual_output": actual[0].detach(), "bits": 2}, input_evidence)
                report["remote_only_input_tensor_evidence"] = {"file": input_evidence.name, "bytes": input_evidence.stat().st_size,
                    "sha256": hashlib.sha256(input_evidence.read_bytes()).hexdigest(), "download_policy": "remote data volume only; saved before any instrument/report function"}
                fp64, fp64_diagnostics = source_dense_attention(cache, index, q, read_args[1], read_args[2])
                authored, operator_diagnostics = source_operator_attention(cache, index, q, read_args[1])
                report["actual_vs_ideal_fp64_instrument"] = difference_summary(actual[0], fp64)
                report["actual_vs_author_AST_same_precision_mapped_operator_reference"] = difference_summary(actual[0], authored)
                report["Eq6_fp64_diagnostics"] = fp64_diagnostics
                report["raw_qk_source_vs_fp64"] = operator_diagnostics["raw_qk_source_vs_fp64"]
                report["raw_wv_source_vs_fp64_same_probabilities"] = operator_diagnostics["raw_wv_source_vs_fp64_same_probabilities"]
                location = report["actual_vs_ideal_fp64_instrument"]["max_abs_location"]
                b, t, h, d = location
                weighted_index = (b, h, t, d)
                source_value = operator_diagnostics["weighted_source"][weighted_index]
                ideal_value = operator_diagnostics["weighted_fp64"][weighted_index]
                scale_value = operator_diagnostics["value_scale"][b, h, 0, d]
                report["max_error_raw_value_context"] = {"output_location_B_T_H_D": location,
                    "source_raw_weighted_code": float(source_value), "fp64_rounded_raw_weighted_code_same_source_probabilities": float(ideal_value),
                    "value_channel_scale": float(scale_value),
                    "raw_weighted_difference_times_scale": float((source_value.float() - ideal_value.float()) * scale_value.float())}
                evidence = args.output / "source0_C2_layer26_finite_oracle_capture.pt"
                layer = cache.layers[index]
                torch.save({"q": q.detach(), "owner": layer.visual_owner.owner_tensors(), "text_keys": layer.keys.detach(),
                    "text_values": layer.values.detach(), "visual_positions": layer.visual_positions, "text_positions": layer.text_positions,
                    "query_positions": layer.query_positions, "attention_mask": read_args[1], "scaling": read_args[2],
                    "actual_output": actual[0].detach(), "fp64_reference_output": fp64.detach(),
                    "same_precision_source_AST_output": authored.detach(), "bits": 2}, evidence)
                report["remote_only_tensor_evidence"] = {"file": evidence.name, "bytes": evidence.stat().st_size,
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(), "download_policy": "remote data volume only"}
            return actual
        cache.read = diagnostic_read
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with attention_context(model, cache=cache):
            output = m1_forward_suffix(model, batch, cache, logits_to_keep=batch.suffix_length)
        assert bool(torch.isfinite(output.logits).all()) and captured
        report["complete_suffix_all_28_layers_finite"] = all(len(v) == 1 and v[0]["query_tokens"] == batch.suffix_length for v in calls.values())
        del output
        cache.crop(batch.boundary)
        with attention_context(model, cache=cache):
            generated = m1_generate_cached(model, batch, cache, max_new_tokens=16)
        torch.cuda.synchronize()
        report["suffix_and_freegen_with_instrument_ms"] = (time.perf_counter() - start) * 1000
        report["peak_allocated_bytes_including_instrument"] = torch.cuda.max_memory_allocated()
        tokens = generated[0, batch.full_length:].cpu().tolist()
        report["own_compressed_suffix_generated_token_ids"] = tokens
        report["raw_prediction_unscored"] = processor.decode(tokens, skip_special_tokens=True)
        report["all_attention_calls_finite"] = all(row["finite"] for rows in calls.values() for row in rows)
        report["calls_by_layer"] = calls
        report["method_manifest"] = adapter.manifest()
        report["source_hashes"] = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "code/gates/calibquant_operator_reference.py", ROOT / "code/methods/calibquant_fp32_bias_adapter.py",
                         ROOT / "code/gates/calibquant_fp32_bias_oracle.py", ROOT / "code/cache_bridge.py", ROOT / "code/attention_bridge.py"]}
        report["status"] = "DIAGNOSTIC_COMPLETED_IMPLEMENTATION_FIDELITY_SEPARATE_FROM_FP64_ERROR"
    except Exception as exc:
        report.update(status="FAILED_RUN", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["completed_unix"] = time.time()
        with (args.output / "complete.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(json.dumps({"status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
