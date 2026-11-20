"""Same-input controls locating native full/split FP16 shape differences."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "SAME_INPUT_FP16_FULL_SPLIT_AND_HYBRID_DIAGNOSTIC_NO_COMPRESSION_NO_TASK_SCORE",
              "job": job, "started_unix": time.time()}
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from transformers.cache_utils import DynamicCache
        from native_model_smoke import native_rows
        from model_registry import get_model_spec
        from m1_protocol import (prepare_m1_inputs, m1_native_full_forward, m1_prefix_forward,
                                 m1_forward_suffix, m1_generate_native, m1_generate_cached)
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context
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
        phase = "initial"
        position_snapshots = []
        def capture_positions(module, positional, keyword):
            positions = keyword.get("position_ids")
            cache = keyword.get("past_key_values")
            position_snapshots.append({"phase": phase,
                "position_ids": positions.detach().cpu().tolist() if positions is not None else None,
                "logical_cache_length_before_call": int(cache.get_seq_length()) if cache is not None else 0})
        position_handle = model.model.language_model.register_forward_pre_hook(capture_positions, with_kwargs=True)
        source = native_rows({**job, "limit": 1000})[job.get("source_indices", [0])[0]]
        image = Image.open(source["image_path"]).convert("RGB")
        batch = prepare_m1_inputs(processor, image, source["question"], image_token_id=spec["image_token_id"],
                                  max_patches=int(job.get("max_patches", 8)))
        report.update(torch=torch.__version__, transformers=transformers.__version__, hardware=torch.cuda.get_device_name(),
                      model_revision=spec["revision"], image_sha256=sha(source["image_path"]),
                      source_id=source["source_id"], prefix_tokens=batch.boundary, suffix_tokens=batch.suffix_length,
                      visual_tokens=int(batch.visual_positions.numel()))
        torch.cuda.reset_peak_memory_stats()

        def clone_prefix(cache):
            clone = DynamicCache()
            for index, layer in enumerate(cache.layers):
                clone.update(layer.keys[:, :, :batch.boundary].contiguous().clone(),
                             layer.values[:, :, :batch.boundary].contiguous().clone(), index)
            return clone

        def difference(a, b):
            diff = (a.float() - b.float()).abs()
            return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "bitwise_equal": bool(torch.equal(a, b)),
                    "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)),
                    "last_token_argmax_equal": bool(torch.equal(a[..., -1, :].argmax(-1), b[..., -1, :].argmax(-1))),
                    "all_suffix_argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}

        phase = "native_full"
        full = m1_native_full_forward(model, batch, logits_to_keep=batch.suffix_length)
        full_logits = full.logits.detach().float().cpu()
        from_full_prefix = clone_prefix(full.past_key_values)
        del full
        gc.collect()
        phase = "same_input_native_full_repeat"
        repeat_full = m1_native_full_forward(model, batch, logits_to_keep=batch.suffix_length)
        report["same_input_native_full_repeat"] = difference(repeat_full.logits.cpu(), full_logits)
        del repeat_full
        phase = "native_prefix_only"
        prefix = m1_prefix_forward(model, batch)
        prefix_cache = prefix.past_key_values
        report["native_full_prefix_vs_prefix_only_cache"] = []
        for index, (left, right) in enumerate(zip(from_full_prefix.layers, prefix_cache.layers)):
            report["native_full_prefix_vs_prefix_only_cache"].append({"layer": index,
                "k": difference(left.keys, right.keys), "v": difference(left.values, right.values)})
        from_prefix = clone_prefix(prefix_cache)
        hybrid_prefix = clone_prefix(prefix_cache)
        del prefix, prefix_cache
        gc.collect()
        phase = "native_split_full_generated_prefix"
        from_full = m1_forward_suffix(model, batch, from_full_prefix, logits_to_keep=batch.suffix_length)
        report["native_full_vs_native_split_using_full_generated_prefix"] = difference(from_full.logits.cpu(), full_logits)
        del from_full
        from_full_prefix.crop(batch.boundary)
        phase = "native_split_prefix_only_prefill"
        with attention_context(model, capture_queries=True, window_size=batch.suffix_length) as native_capture:
            split = m1_forward_suffix(model, batch, from_prefix, logits_to_keep=batch.suffix_length)
        split_logits = split.logits.detach().float().cpu()
        report["native_full_vs_native_split_using_prefix_only_prefill"] = difference(split_logits, full_logits)
        del split
        from_prefix.crop(batch.boundary)
        hybrid = HybridVisualCache(hybrid_prefix, batch.visual_positions)
        original_read = hybrid.read
        query_differences = []
        def audited_read(index, q, *read_args, **read_kwargs):
            query_differences.append({"layer": index, "difference": difference(q, native_capture["queries_by_layer"][index]),
                                      "query_positions": hybrid.layers[index].query_positions.cpu().tolist()})
            return original_read(index, q, *read_args, **read_kwargs)
        hybrid.read = audited_read
        phase = "hybrid_split_prefix_only_prefill"
        with attention_context(model, cache=hybrid):
            bridge = m1_forward_suffix(model, batch, hybrid, logits_to_keep=batch.suffix_length)
        report["native_split_vs_hybrid_split_same_prefix_same_shape"] = difference(bridge.logits.cpu(), split_logits)
        report["native_split_vs_hybrid_post_rope_q"] = query_differences
        del bridge
        hybrid.read = original_read
        hybrid.crop(batch.boundary)
        # Compare complete native greedy output; each condition computes its own
        # first token from the entire question suffix under its own prefix.
        generation = {}
        phase = "native_full_generate"
        native_generated = m1_generate_native(model, batch, max_new_tokens=int(job.get("max_new_tokens", 16)))
        generation["native_full"] = native_generated[0, batch.full_length:].cpu().tolist()
        del native_generated
        for name, cache in [("native_split_full_generated_prefix", from_full_prefix),
                            ("native_split_prefix_only_prefill", from_prefix)]:
            phase = name + "_generate"
            generated = m1_generate_cached(model, batch, cache, max_new_tokens=int(job.get("max_new_tokens", 16)))
            generation[name] = generated[0, batch.full_length:].cpu().tolist()
            del generated
        phase = "hybrid_split_generate"
        with attention_context(model, cache=hybrid):
            generated = m1_generate_cached(model, batch, hybrid, max_new_tokens=int(job.get("max_new_tokens", 16)))
        generation["hybrid_split_prefix_only_prefill"] = generated[0, batch.full_length:].cpu().tolist()
        report["generation_token_ids"] = generation
        report["all_generation_sequences_equal"] = all(ids == generation["native_full"] for ids in generation.values())
        report["raw_predictions"] = {name: processor.decode(ids, skip_special_tokens=True) for name, ids in generation.items()}
        report["native_language_position_snapshots"] = position_snapshots
        position_handle.remove()
        report["source_hashes"] = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in
                                  [Path(__file__), ROOT / "code/m1_protocol.py", ROOT / "code/cache_bridge.py", ROOT / "code/attention_bridge.py"]}
        torch.cuda.synchronize()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["status"] = "DIAGNOSTIC_COMPLETED_NOT_QUALITY_UNLOCK"
    except Exception as exc:
        report.update(status="FAILED_RUN", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["completed_unix"] = time.time()
        with (out / "complete.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({k: report[k] for k in ["status"]}), flush=True)


if __name__ == "__main__":
    main()
