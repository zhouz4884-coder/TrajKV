"""Actual M1 image-prefix/all-layer-suffix CalibQuant P0 gate.

No task labels/scoring/calibration are used. Same-prefix/same-shape native FP16
split execution must match the uncompressed HybridVisualCache bridge exactly.
This gate never changes model weights, the released codec or its parameters.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def physical_report(cache):
    seen = set()
    records = {}
    for name, tensor in cache.named_tensors().items():
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr())
        counted = 0 if identity in seen else storage.nbytes()
        seen.add(identity)
        records[name] = {"storage_bytes": counted, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    codec = sum(r["storage_bytes"] for name, r in records.items() if ".visual." in name)
    positions = sum(r["storage_bytes"] for name, r in records.items() if "positions" in name)
    text = sum(r["storage_bytes"] for name, r in records.items() if name.endswith(("text_keys", "text_values")))
    total = sum(r["storage_bytes"] for r in records.values())
    return {"all_cache_tensor_storage_bytes": total, "visual_codec_tensor_storage_bytes": codec,
            "all_positions_storage_bytes": positions, "exact_text_kv_storage_bytes": text,
            "owner_storages": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    outdir = args.output or Path(job["output_dir"])
    outdir.mkdir(parents=True, exist_ok=False)
    marker = outdir / "marker.json"
    result = {"status": "RUNNING", "scope": "M1_NATIVE_FP16_BRIDGE_AND_CALIBQUANT_PACKED_SUFFIX_P0_NO_TASK_SCORE",
              "job": job, "job_sha256": sha(args.job), "cases": [], "started_unix": time.time(),
              "python_version": platform.python_version(), "quality_measured": False}
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    active = None
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from model_registry import get_model_spec
        from native_model_smoke import native_rows
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context, language_attention_modules
        from methods.calibquant_adapter import CalibQuantAdapter
        from m1_protocol import (prepare_m1_inputs, m1_native_full_forward, m1_prefix_forward,
                                 m1_forward_suffix, m1_generate_native, m1_generate_cached, clone_m1_prefix_cache)
        from calibquant_dense_oracle import source_dense_attention, tensor_stats

        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_capability() == (8, 6), "SM86 hardware gate required"
        assert job["model_id"] == "M1", "This gate declares M1 serial-position scope only"
        spec = get_model_spec("M1")
        assert spec["revision"], "Pinned official M1 identity required"
        identity_path = Path(job["model_identity_report"])
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        assert identity["status"] == "PASS_EXACT_REQUIRED_TREE" and identity["official_revision"] == spec["revision"]
        model_path = Path(job["model_path"])
        identity_files = {row["file"]: row for row in identity["files"]}
        for name in ["config.json", "generation_config.json", "model.safetensors.index.json"]:
            assert sha(model_path / name) == identity_files[name]["sha256"]
        for name, row in identity_files.items():
            if name.endswith(".safetensors"):
                assert (model_path / name).stat().st_size == row["bytes"]
        result.update(model_id="M1", model_revision=spec["revision"], dtype="float16",
                      identity_report_sha256=sha(identity_path), torch_version=torch.__version__,
                      transformers_version=transformers.__version__, cuda_version=torch.version.cuda,
                      hardware=torch.cuda.get_device_name(), capability=[8, 6])
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.float16,
                    device_map={"": 0}, local_files_only=True, attn_implementation="sdpa").eval()
        torch.cuda.synchronize()
        result["load_ms"] = (time.perf_counter() - start) * 1000
        result["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        assert all(p.dtype == torch.float16 for p in model.parameters()), "Weight quantization/dtype mismatch"
        language, attention_modules = language_attention_modules(model)
        assert len(attention_modules) == spec["layers"] == 28
        result["native_generation_config"] = model.generation_config.to_dict()
        result["generation_overrides"] = {"do_sample": False, "max_new_tokens": int(job.get("max_new_tokens", 16))}
        result["native_module_hashes"] = {}
        for component in [model, language, attention_modules[0]]:
            path = Path(sys.modules[type(component).__module__].__file__)
            result["native_module_hashes"][path.name] = sha(path)
        result["source_hashes"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in
            [Path(__file__), ROOT / "code/cache_bridge.py", ROOT / "code/attention_bridge.py",
             ROOT / "code/methods/calibquant_adapter.py", ROOT / "code/model_registry.py",
             ROOT / "code/m1_protocol.py", Path(__file__).with_name("calibquant_dense_oracle.py")]}

        def timed(function):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = function()
            torch.cuda.synchronize()
            return value, (time.perf_counter() - start) * 1000

        rows = native_rows({**job, "limit": 1000})
        indices = job.get("source_indices", [0, 6])
        assert len(indices) >= 2 and len(set(indices)) == len(indices)
        bitwidths = job.get("bitwidths", [1, 2])
        assert bitwidths == [1, 2], "Declared P0 covers both C1 and C2"
        adapter = None
        for index in indices:
            source = rows[index]
            path = Path(source["image_path"])
            active = {"status": "RUNNING", "source_index": index, "source_id": source["source_id"],
                      "original_image_sha256": sha(path), "calibquant_runs": []}
            if source.get("source_image_sha256", source.get("image_sha256")):
                assert active["original_image_sha256"] == source.get("source_image_sha256", source.get("image_sha256"))
            image = Image.open(path).convert("RGB")
            active["original_dimensions"] = list(image.size)
            question = source.get("question", "Describe the image briefly.")
            batch = prepare_m1_inputs(processor, image, question, image_token_id=spec["image_token_id"],
                                      max_patches=int(job.get("max_patches", 8)))
            inputs, prefix_inputs = batch.inputs, batch.prefix_inputs
            ids, mask = batch.ids, batch.mask
            visual, boundary, length = batch.visual_positions, batch.boundary, batch.full_length
            suffix_length, positions, prompt = batch.suffix_length, batch.positions, batch.prompt
            assert suffix_length >= 2, "Multi-token question suffix required"
            active.update(full_prompt=prompt, full_prompt_hash=text_sha(prompt), chat_template_hash=text_sha(processor.chat_template),
                          question=question, input_tokens=length, actual_visual_token_count=int(visual.numel()),
                          visual_positions=visual.cpu().tolist(), prefix_tokens=boundary, suffix_tokens=suffix_length,
                          prefix_token_sha256=tensor_sha(ids[:, :boundary]), question_suffix_sha256=tensor_sha(ids[:, boundary:]),
                          cap_or_max_pixels=int(job.get("max_patches", 8)))
            # Verify that a distinct future question cannot change the encoded
            # native image prefix. Only the first question suffix is evaluated.
            alternate = [{"role": "user", "content": [{"type": "image", "image": image},
                          {"type": "text", "text": "What objects and text are visible in this image?"}]}]
            alt_prompt = processor.apply_chat_template(alternate, tokenize=False, add_generation_prompt=True)
            alt = processor(text=[alt_prompt], images=[image], return_tensors="pt", crop_to_patches=True,
                            max_patches=int(job.get("max_patches", 8)))
            alt_visual = (alt["input_ids"][0] == spec["image_token_id"]).nonzero(as_tuple=False).flatten()
            assert int(alt_visual[-1]) + 1 == boundary
            assert torch.equal(alt["input_ids"][:, :boundary], ids[:, :boundary].cpu())
            assert tensor_sha(alt["pixel_values"]) == tensor_sha(inputs["pixel_values"])
            active["question_unknown_prefix_verified"] = True
            del alt, alternate

            torch.cuda.reset_peak_memory_stats()
            native_output, active["native_full_prefill_ms"] = timed(lambda: m1_native_full_forward(model, batch, logits_to_keep=suffix_length))
            native_logits = native_output.logits.detach().float().cpu()
            del native_output
            gc.collect()
            native_generated, active["native_generate_ms"] = timed(lambda: m1_generate_native(model, batch,
                       max_new_tokens=int(job.get("max_new_tokens", 16))))
            native_tokens = native_generated[0, length:].cpu().tolist()
            active["native_output_token_ids"] = native_tokens
            active["native_raw_prediction"] = processor.decode(native_tokens, skip_special_tokens=True)
            del native_generated
            active["native_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            gc.collect()

            def make_cache(compressed, supplied_cache=None, supplied_prefix_ms=None):
                prefix_output = None
                if supplied_cache is None:
                    prefix_output, prefix_ms = timed(lambda: m1_prefix_forward(model, batch))
                    native_cache = prefix_output.past_key_values
                else:
                    native_cache, prefix_ms = supplied_cache, supplied_prefix_ms
                refs = [weakref.ref(t) for layer in native_cache.layers for t in (layer.keys, layer.values)]
                pointers = {t.untyped_storage().data_ptr() for layer in native_cache.layers for t in (layer.keys, layer.values)}
                owners = None
                torch.cuda.synchronize()
                encode_start = time.perf_counter()
                if compressed:
                    owners = [adapter.encode(layer.keys.index_select(-2, visual), layer.values.index_select(-2, visual))
                              for layer in native_cache.layers]
                cache = HybridVisualCache(native_cache, visual, owners,
                                          reader_backend="upstream_triton" if compressed else None)
                torch.cuda.synchronize()
                encode_ms = (time.perf_counter() - encode_start) * 1000
                assert not native_cache.layers
                del prefix_output, native_cache
                gc.collect()
                assert all(ref() is None for ref in refs), "Original full-prefix FP16 tensor retained"
                assert not pointers.intersection(t.untyped_storage().data_ptr() for t in cache.named_tensors().values())
                return cache, prefix_ms, encode_ms

            def run_bridge(cache, source_oracle=False, native_suffix_queries=None, native_generation_queries=None):
                calls = {i: [] for i in range(spec["layers"])}
                active["current_bridge_calls"] = calls
                query_phase = "suffix"
                query_offsets = {i: 0 for i in range(spec["layers"])}
                original_read = cache.read
                def audited_read(layer_index, q, *read_args, **read_kwargs):
                    entry = {"query_tokens": q.shape[-2], "query_heads": q.shape[1],
                             "query_dtype": str(q.dtype), "post_rope_native_dispatch": True}
                    calls[layer_index].append(entry)
                    expected_queries = native_suffix_queries if query_phase == "suffix" else native_generation_queries
                    if expected_queries is not None:
                        offset = query_offsets[layer_index]
                        expected_q = expected_queries[layer_index][..., offset:offset + q.shape[-2], :]
                        query_offsets[layer_index] += q.shape[-2]
                        entry["same_prefix_native_post_rope_queries_bitwise_equal"] = bool(torch.equal(q.cpu(), expected_q))
                        assert entry["same_prefix_native_post_rope_queries_bitwise_equal"], "Native split vs Hybrid post-RoPE Q mismatch"
                        assert cache.layers[layer_index].query_positions.tolist() == list(range(
                            cache.layers[layer_index].last_update_start, cache.layers[layer_index].last_update_start + q.shape[-2]))
                    actual = original_read(layer_index, q, *read_args, **read_kwargs)
                    if source_oracle:
                        # read_args: module, original native mask, native scale,
                        # original SDPA. No owner.read/weighted_value is used by
                        # the independent expanded-code reference.
                        expected, diagnostics = source_dense_attention(cache, layer_index, q, read_args[1], read_args[2])
                        entry.update(diagnostics)
                        difference = (actual[0].float() - expected.float()).abs()
                        entry["actual_output_stats"] = tensor_stats(actual[0])
                        entry["dense_source_oracle_max_abs_error"] = float(difference.max()) if bool(torch.isfinite(difference).all()) else None
                        if not bool(torch.isfinite(actual[0]).all() & torch.isfinite(expected).all()):
                            dump = outdir / f"numeric_failure_source{index}_C{adapter.bits}_layer{layer_index}.pt"
                            torch.save({"q": q.detach(), "owner": {name: value.detach() for name, value in cache.layers[layer_index].visual_owner.owner_tensors().items()},
                                        "text_keys": cache.layers[layer_index].keys.detach(), "text_values": cache.layers[layer_index].values.detach(),
                                        "visual_positions": cache.layers[layer_index].visual_positions.detach(),
                                        "text_positions": cache.layers[layer_index].text_positions.detach(),
                                        "query_positions": cache.layers[layer_index].query_positions.detach(),
                                        "attention_mask": read_args[1], "scaling": read_args[2], "bits": adapter.bits}, dump)
                            entry["numeric_failure_tensor_evidence"] = {"file": dump.name, "sha256": sha(dump), "bytes": dump.stat().st_size,
                                "policy": "derived activations only, remote data volume; not copied to local or public automatically"}
                        torch.testing.assert_close(actual[0], expected, atol=.0078125, rtol=.005)
                        assert bool(torch.isfinite(actual[0]).all())
                    return actual
                cache.read = audited_read
                with attention_context(model, cache=cache):
                    output, suffix_ms = timed(lambda: m1_forward_suffix(model, batch, cache, logits_to_keep=suffix_length))
                logits = output.logits.detach().float().cpu()
                del output
                assert all(len(v) == 1 and v[0]["query_tokens"] == suffix_length for v in calls.values())
                assert cache.get_seq_length() == length
                first_calls = {i: list(v) for i, v in calls.items()}
                cache.crop(boundary)
                calls = {i: [] for i in range(spec["layers"])}
                active["current_bridge_calls"] = calls
                query_phase = "generation"
                query_offsets = {i: 0 for i in range(spec["layers"])}
                # Leave position_ids to the native generation helper, which
                # extends positions correctly after every generated token.
                with attention_context(model, cache=cache):
                    generated, generate_ms = timed(lambda: m1_generate_cached(model, batch, cache,
                            max_new_tokens=int(job.get("max_new_tokens", 16))))
                tokens = generated[0, length:].cpu().tolist()
                del generated
                assert tokens, "No free-generated answer"
                assert all(v and v[0]["query_tokens"] == suffix_length and
                           all(c["query_tokens"] == 1 for c in v[1:]) for v in calls.values())
                assert len({len(v) for v in calls.values()}) == 1
                assert cache.get_seq_length() == length + len(tokens) - 1
                cache.read = original_read
                return logits, tokens, {"suffix_ms": suffix_ms, "generate_ms": generate_ms,
                                       "suffix_calls_by_layer": first_calls, "generation_calls_by_layer": calls}

            torch.cuda.reset_peak_memory_stats()
            shared_prefix, shared_prefix_ms = timed(lambda: m1_prefix_forward(model, batch))
            native_split_cache = clone_m1_prefix_cache(shared_prefix.past_key_values)
            hybrid_input_cache = clone_m1_prefix_cache(shared_prefix.past_key_values)
            del shared_prefix
            gc.collect()
            with attention_context(model, capture_queries=True, window_size=suffix_length) as split_queries:
                split_output, active["native_split_suffix_ms"] = timed(lambda: m1_forward_suffix(
                    model, batch, native_split_cache, logits_to_keep=suffix_length))
            native_split_logits = split_output.logits.detach().float().cpu()
            native_suffix_queries = {i: q.cpu() for i, q in split_queries["queries_by_layer"].items()}
            del split_output, split_queries
            native_split_cache.crop(boundary)
            with attention_context(model, capture_queries=True, window_size=suffix_length + int(job.get("max_new_tokens", 16))) as generation_queries:
                split_generated, active["native_split_generate_ms"] = timed(lambda: m1_generate_cached(
                    model, batch, native_split_cache, max_new_tokens=int(job.get("max_new_tokens", 16))))
            native_split_tokens = split_generated[0, length:].cpu().tolist()
            native_generation_queries = {i: q.cpu() for i, q in generation_queries["queries_by_layer"].items()}
            del split_generated, generation_queries, native_split_cache
            gc.collect()
            active["native_split_output_token_ids"] = native_split_tokens
            active["native_full_and_split_generation_equal"] = native_split_tokens == native_tokens
            shape_difference = (native_split_logits - native_logits).abs()
            active["native_full_vs_native_split_diagnostic"] = {"max_abs": float(shape_difference.max()),
                "mean_abs": float(shape_difference.mean()),
                "relative_l2": float(torch.linalg.vector_norm(shape_difference) / torch.linalg.vector_norm(native_logits).clamp_min(1e-12)),
                "all_suffix_argmax_agreement": float((native_split_logits.argmax(-1) == native_logits.argmax(-1)).float().mean())}
            fp_cache, active["fp16_prefix_ms"], active["fp16_owner_encode_ms"] = make_cache(False, hybrid_input_cache, shared_prefix_ms)
            del hybrid_input_cache
            fp_logits, fp_tokens, fp_trace = run_bridge(fp_cache, native_suffix_queries=native_suffix_queries,
                                                       native_generation_queries=native_generation_queries)
            max_error = float((fp_logits - native_split_logits).abs().max())
            active["fp16_bridge_vs_native_split_logit_max_abs_error"] = max_error
            active["fp16_bridge_vs_native_split_logit_mean_abs_error"] = float((fp_logits - native_split_logits).abs().mean())
            active["fp16_equivalence_contract"] = "same native prefix, same full question suffix shape, bitwise logits/post-RoPE Q and exact greedy IDs"
            assert torch.equal(fp_logits, native_split_logits), "Same-prefix native split vs Hybrid FP16 split logits mismatch"
            assert fp_tokens == native_split_tokens, "Same-prefix native split vs Hybrid FP16 greedy sequence mismatch"
            active["fp16_bridge_output_token_ids"] = fp_tokens
            active["fp16_bridge_trace"] = fp_trace
            active["fp16_bridge_owner"] = physical_report(fp_cache)
            active["fp16_bridge_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            del fp_cache, fp_logits, native_suffix_queries, native_generation_queries
            gc.collect()
            torch.cuda.empty_cache()

            for bits in bitwidths:
                active["current_bitwidth"] = bits
                adapter = CalibQuantAdapter(bits=bits, reader_backend="upstream_triton")
                c_record = {"bits": bits, "method_manifest": adapter.manifest()}
                torch.cuda.reset_peak_memory_stats()
                c_cache, c_record["prefix_ms"], c_record["encode_ms"] = make_cache(True)
                before = physical_report(c_cache)
                owner_hashes = [{name: tensor_sha(t) for name, t in layer.visual_owner.owner_tensors().items()} for layer in c_cache.layers]
                numeric_hidden_trace = []
                diagnostic_handles = []
                if bits == 2:
                    for layer_index, decoder in enumerate(language.layers):
                        def before_layer(module, positional, keyword, layer_index=layer_index):
                            hidden = keyword.get("hidden_states", positional[0] if positional else None)
                            numeric_hidden_trace.append({"layer": layer_index, "stage": "decoder_input", "stats": tensor_stats(hidden)})
                        diagnostic_handles.append(decoder.register_forward_pre_hook(before_layer, with_kwargs=True))
                        for stage, component in [("input_layernorm", decoder.input_layernorm), ("attention_output_projection", decoder.self_attn.o_proj),
                                                 ("post_attention_layernorm", decoder.post_attention_layernorm), ("mlp_output", decoder.mlp)]:
                            def after_component(module, positional, output, layer_index=layer_index, stage=stage):
                                numeric_hidden_trace.append({"layer": layer_index, "stage": stage, "stats": tensor_stats(output)})
                            diagnostic_handles.append(component.register_forward_hook(after_component))
                try:
                    c_logits, c_tokens, c_trace = run_bridge(c_cache, source_oracle=True)
                except Exception as exc:
                    c_record.update(status="FAILED_RUN", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                        failure_class="NUMERIC_OR_SOURCE_ORACLE_GATE_FAILED", current_bridge_calls=active["current_bridge_calls"],
                        numeric_hidden_trace=numeric_hidden_trace, owner_at_failure=physical_report(c_cache),
                        source_parameters_unchanged=True, no_score_produced=True)
                    active["calibquant_runs"].append(c_record)
                    with (outdir / f"failure_source{index}_C{bits}.json").open("x", encoding="utf-8") as handle:
                        json.dump(c_record, handle, indent=2)
                    print(json.dumps({"event": "method_config_failed_preserved", "source_index": index, "bits": bits,
                                      "error_type": type(exc).__name__}), flush=True)
                    if "illegal memory" in str(exc).lower() or "out of memory" in str(exc).lower():
                        raise
                    del c_cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                finally:
                    for handle in diagnostic_handles:
                        handle.remove()
                assert bool(torch.isfinite(c_logits).all()), "Nonfinite CalibQuant full-model logits"
                after_hashes = [{name: tensor_sha(t) for name, t in layer.visual_owner.owner_tensors().items()} for layer in c_cache.layers]
                assert owner_hashes == after_hashes, "Packed visual owners changed during suffix/generation"
                assert all(len(layer.visual_owner.owner_tensors()) == 8 for layer in c_cache.layers)
                for layer in c_cache.layers:
                    owner = layer.visual_owner
                    assert owner.k_packed.dtype == owner.v_packed.dtype == torch.uint8
                    assert all(t.shape[-2] == 1 for name, t in owner.owner_tensors().items() if name.endswith(("scale", "zero")))
                del owner
                denominator = spec["layers"] * 2 * spec["kv_heads"] * int(visual.numel()) * spec["head_dim"] * 2
                c_record.update(output_token_ids=c_tokens, raw_prediction=processor.decode(c_tokens, skip_special_tokens=True),
                    logit_sha256=tensor_sha(c_logits), vs_native_split_logit_max_abs_difference=float((c_logits - native_split_logits).abs().max()),
                    trace=c_trace, owner_before_suffix=before, owner_after_generation=physical_report(c_cache),
                    original_fp16_cache_released=True, packed_visual_owners_immutable=True,
                    full_visual_fp16_byte_denominator=denominator,
                    actual_codec_visual_bits=8 * before["visual_codec_tensor_storage_bytes"] / (denominator / 2),
                    mapping_bytes_reported_separately=True,
                    peak_allocated_bytes_including_dense_oracle=torch.cuda.max_memory_allocated(),
                    numeric_hidden_trace=numeric_hidden_trace,
                    runtime_scope="P0 includes temporary dense-code reference; not benchmark performance", status="PASS")
                active["calibquant_runs"].append(c_record)
                del c_cache, c_logits
                gc.collect()
                torch.cuda.empty_cache()
            active["status"] = "PASS" if all(r["status"] == "PASS" for r in active["calibquant_runs"]) else "PARTIAL_METHOD_PASS"
            result["cases"].append(active)
            with (outdir / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(active, ensure_ascii=False) + "\n")
            print(json.dumps({"status": active["status"], "source_index": index, "visual_tokens": int(visual.numel()),
                              "suffix_tokens": suffix_length, "fp16_bridge_max_error": max_error,
                              "calibquant_bits_passed": [r["bits"] for r in active["calibquant_runs"] if r["status"] == "PASS"]}), flush=True)
            del native_logits, native_split_logits, inputs, prefix_inputs, ids, mask, positions, image, batch
            gc.collect()
            torch.cuda.empty_cache()
        passed_bits = [bits for bits in bitwidths if all(any(r["bits"] == bits and r["status"] == "PASS" for r in case["calibquant_runs"]) for case in result["cases"])]
        result["status"] = "PASS_M1_CALIBQUANT_NATIVE_SUFFIX_P0" if passed_bits == bitwidths else "PARTIAL_PASS_M1_CALIBQUANT_NATIVE_SUFFIX_P0"
        result["quality_unlocked_bitwidths_for_declared_M1_protocol"] = passed_bits
        result["quality_unlocked_for_declared_M1_protocol"] = 1 in passed_bits
        result["quality_scores_still_not_measured"] = True
    except Exception as exc:
        result.update(status="FAILED_RUN", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if active is not None:
            (outdir / "failed_case.json").write_text(json.dumps(active, indent=2), encoding="utf-8")
        (outdir / "exception.txt").write_text(result["traceback"], encoding="utf-8")
        raise
    finally:
        result["completed_unix"] = time.time()
        marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (outdir / "complete.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
