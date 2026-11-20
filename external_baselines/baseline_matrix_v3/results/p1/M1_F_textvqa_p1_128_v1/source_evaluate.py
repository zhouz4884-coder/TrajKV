"""One-model, one-method raw inference runner. Never opens reference answers.

Each QA performs a fresh native image-prefix prefill, visual encoding, and HF
greedy generation that recomputes the entire question suffix. Failure rows
remain in immutable coverage; scoring is a separate program.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BASELINE_DATA_ROOT", str(ROOT)))
ALLOWLIST = {
    ("methods.kivi_style", "KIVIStyleAdapter"),
    ("methods.calibquant_adapter", "CalibQuantAdapter"),
    ("methods.calibquant_fp32_bias_adapter", "CalibQuantFP32BiasAdapter"),
    ("methods.turboquant_adapter", "TurboQuantAdapter"),
    ("methods.eviction_adapter", "PyramidKVVisualAdapter"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""): h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream: json.dump(value, stream, ensure_ascii=False, indent=2)


def load_plan(config_path):
    from dataset_registry import load_inputs
    from model_registry import get_model_spec
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("question_suffix", "\nAnswer the question using a single word or phrase.")
    if set(config) & {"refs", "references", "reference_answers", "reference_path", "answers", "labels", "scores"}:
        raise ValueError("INPUT_CONTRACT_VIOLATION: Runner config must not contain reference/score paths")
    spec = get_model_spec(config["model_id"])
    if config.get("model_revision", spec["revision"]) != spec["revision"]:
        raise ValueError("INPUT_MISMATCH: official model revision differs from registry")
    rows = load_inputs(config["inputs"])
    if config.get("ids") is not None:
        lookup = {row["id"]: row for row in rows}
        rows = [lookup[identity] for identity in config["ids"]]
    else:
        start = int(config.get("start", 0))
        rows = rows[start:start + int(config.get("count", len(rows) - start))]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("INPUT_MISMATCH: empty or duplicate QA selection")
    if len({r["dataset"] for r in rows}) != 1:
        raise ValueError("One method/dataset per job is required")
    method = config["method"]
    family = method["family"].lower()
    if family not in ("fp16", "native", "fp16_hybrid", "full_native"):
        pair = (method["adapter_module"], method["adapter_class"])
        if pair not in ALLOWLIST: raise ValueError("Unapproved method adapter: " + repr(pair))
        if pair == ("methods.turboquant_adapter", "TurboQuantAdapter"):
            method.setdefault("per_layer_seed", {"base": 42, "step": 6})
        if pair == ("methods.eviction_adapter", "PyramidKVVisualAdapter"):
            method.setdefault("encode_scope", "prefix")
            method.setdefault("constructor_args", {}).setdefault("num_layers", spec["layers"])
            if method["constructor_args"]["num_layers"] != spec["layers"]:
                raise ValueError("INPUT_MISMATCH: eviction layer-budget geometry differs from model")
    if method.get("information_condition", "query_unknown") != "query_unknown":
        raise ValueError("METHOD_PORT_UNVERIFIED: query-known requires a separate selection-workload driver")
    return config, spec, rows


def model_identity(config, spec):
    """Verify original metadata and every weight SHA against fixed HF tree."""
    path = Path(config["model_path"])
    lock_path = Path(config.get("model_source_lock", DATA_ROOT / "configs/models" /
                               ("M1_hf_reference_lock_v1.json" if spec["model_id"] == "M1" else "hf_revision_lock.json")))
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    lock = raw if "siblings" in raw else raw[spec["hf_id"]]
    if lock.get("revision", lock.get("sha")) != spec["revision"]:
        raise RuntimeError("INPUT_MISMATCH: official checkpoint tree revision")
    checked = []
    for entry in lock["siblings"]:
        name = entry["rfilename"]
        if "/" in name or not name.endswith((".json", ".txt", ".jinja", ".model", ".safetensors")): continue
        target = path / name
        if not target.exists(): raise RuntimeError("Checkpoint file missing: " + name)
        if target.stat().st_size != entry["size"]: raise RuntimeError("Checkpoint size mismatch: " + name)
        digest = sha(target)
        if entry.get("lfs"):
            if digest != entry["lfs"]["sha256"]: raise RuntimeError("Checkpoint full SHA mismatch: " + name)
        else:
            body = target.read_bytes()
            blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if blob != entry["blobId"]: raise RuntimeError("Checkpoint metadata blob mismatch: " + name)
        checked.append({"file": name, "bytes": target.stat().st_size, "sha256": digest})
    return {"status": "PASS_EXACT_HF_REQUIRED_TREE", "revision": spec["revision"],
            "source_lock_sha256": sha(lock_path), "files": checked}


class GenerationObserver:
    def __init__(self, model, request_start, batch):
        self.model, self.request_start, self.batch = model, request_start, batch
        self.phase, self.calls, self.token_times, self.stream_ids = "prefix", [], [], []
        self.prompt_seen = False
        self.first_shape = None
    def before(self, module, args, kwargs):
        import torch
        if self.phase == "generation" and hasattr(self.batch, "rope_deltas"):
            # Keep a single native tiny delta allocation after protocol activation.
            self.batch.rope_deltas = self.model.model.rope_deltas
        torch.cuda.synchronize(); self.start = time.perf_counter()
    def after(self, module, args, kwargs, output):
        import torch
        torch.cuda.synchronize()
        logits = getattr(output, "logits", None)
        if logits is None or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("NONFINITE_LOGITS: raw outer-forward logits are missing/nonfinite")
        self.calls.append({"phase": self.phase, "wall_ms": (time.perf_counter() - self.start) * 1000,
                           "logits_shape": list(logits.shape), "raw_logits_finite": True})
    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True; return
        self.token_times.append(time.perf_counter())
        self.stream_ids.extend(value.detach().cpu().reshape(-1).tolist())
    def end(self): pass


@contextmanager
def observe_generation(model, observer):
    before = model.register_forward_pre_hook(observer.before, with_kwargs=True)
    after = model.register_forward_hook(observer.after, with_kwargs=True)
    original = model.generate
    def generate(*args, **kwargs):
        return original(*args, streamer=observer, return_dict_in_generate=False, **kwargs)
    model.generate = generate
    try: yield observer
    finally:
        model.generate = original
        before.remove(); after.remove()


def adapter_for(method, layer_index):
    pair = (method["adapter_module"], method["adapter_class"])
    if pair not in ALLOWLIST: raise ValueError("Unapproved dynamic method")
    cls = getattr(importlib.import_module(pair[0]), pair[1])
    args = dict(method.get("constructor_args", {}))
    seed = method.get("per_layer_seed")
    if seed is not None: args["seed"] = int(seed["base"]) + int(seed["step"]) * layer_index
    return cls(**args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config, spec, rows = load_plan(args.config)
    if args.plan:
        print(json.dumps({"status": "PLAN_VALID", "model_id": spec["model_id"], "model_revision": spec["revision"],
                          "method": config["method"], "dataset": rows[0]["dataset"], "expected_QA": len(rows),
                          "question_suffix": config["question_suffix"],
                          "expected_ids_sha256": canonical_sha([r["id"] for r in rows]), "reads_references": False})); return
    if args.output is None: parser.error("--output NEW_DIR is required for actual inference")
    args.output.mkdir(parents=True, exist_ok=False)
    expected = [r["id"] for r in rows]
    manifest = {"schema": "baseline_matrix.raw_inference.v1", "config": config, "config_sha256": sha(args.config),
                "inputs_sha256": sha(config["inputs"]), "expected_ids": expected, "expected_QA": len(rows),
                "model_spec": spec, "references_read": False, "prediction_cache_used": False,
                "runtime_source_root": str(ROOT), "runtime_executable": sys.executable,
                "started_unix": time.time(), "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
                    (Path(__file__), ROOT / "code/ownership.py", ROOT / "code/cache_bridge.py", ROOT / "code/attention_bridge.py",
                     ROOT / "code/dataset_registry.py", ROOT / "code/model_registry.py")}}
    records, load_error = [], None
    model = processor = None
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from ownership import model_storage, cache_storage, assert_no_full_prefix_alias
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context
        torch.set_grad_enabled(False); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available(): raise RuntimeError("HARDWARE_UNSUPPORTED: actual CUDA is required")
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (8, 6): raise RuntimeError("HARDWARE_UNSUPPORTED: declared RTX3090 SM86 gate")
        identity = model_identity(config, spec)
        manifest.update(model_identity=identity, torch_version=torch.__version__, transformers_version=transformers.__version__,
                        cuda_version=torch.version.cuda, hardware=torch.cuda.get_device_name(), python_version=platform.python_version())
        preprocessing = config.get("preprocessing", {})
        processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
        load_start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(config["model_path"], dtype=torch.float16,
                    device_map={"": 0}, local_files_only=True, attn_implementation="sdpa").eval()
        model.requires_grad_(False); torch.cuda.synchronize()
        if not all(p.dtype == torch.float16 for p in model.parameters()) or getattr(model, "is_quantized", False):
            raise RuntimeError("INPUT_MISMATCH: model weights must all be original FP16")
        manifest.update(model_load_ms=(time.perf_counter() - load_start) * 1000, model_residency=model_storage(model),
                        processor_id=spec["hf_id"], processor_version=transformers.__version__,
                        processor_file_hashes={r["file"]: r["sha256"] for r in identity["files"] if not r["file"].endswith(".safetensors")},
                        generation_native=model.generation_config.to_dict(),
                        generation_overrides={"do_sample": False, "max_new_tokens": int(config.get("max_new_tokens", 64)),
                                              "use_cache": True, "return_dict_in_generate": False},
                        timing_scope="instrumented prototype quality run; includes finite checks, synchronization and streamer, not canonical method speed")
        if int(model.generation_config.num_beams or 1) != 1: raise RuntimeError("Only native greedy beam-one generation supported")
        if spec["model_id"] == "M1":
            protocol = importlib.import_module("m1_protocol")
            prepare = lambda image, question: protocol.prepare_m1_inputs(processor, image, question,
                image_token_id=spec["image_token_id"], max_patches=int(preprocessing.get("max_patches", 8)))
            prefix_forward, generate_cached, generate_native = protocol.m1_prefix_forward, protocol.m1_generate_cached, protocol.m1_generate_native
        else:
            protocol = importlib.import_module("qwen_protocol")
            prepare = lambda image, question: protocol.prepare_qwen_inputs(processor, image, question, model=model,
                image_token_id=spec["image_token_id"], min_pixels=preprocessing.get("min_pixels"), max_pixels=preprocessing.get("max_pixels"))
            prefix_forward, generate_cached, generate_native = protocol.prefix_forward, protocol.generate_cached, protocol.generate_native
        manifest["protocol_source_sha256"] = sha(Path(protocol.__file__))
        method, family = config["method"], config["method"]["family"].lower()
        if family not in ("fp16", "native", "fp16_hybrid", "full_native"):
            module = importlib.import_module(method["adapter_module"])
            manifest["method_source_sha256"] = sha(Path(module.__file__))
        manifest["method_and_reader_source_hashes"] = {p.name: sha(p) for p in (ROOT / "code/methods").glob("*.py")}
        write_new(args.output / "manifest.json", manifest)
        for row in rows:
            started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            record = {**row, "status": "RUNNING", "model_id": spec["model_id"], "model_commit": spec["revision"],
                      "dtype": "float16", "model_arch": spec["architecture"], "method": method["id"], "method_contract": method,
                      "preprocessing": preprocessing, "output_token_ids": None, "raw_prediction": None, "official_score": None,
                      "score_status": "NOT_READ_OR_COMPUTED_BY_RUNNER", "timing_scope": manifest["timing_scope"]}
            cache = batch = observer = prefix_output = capture = owners = generated = image = None
            handles = None
            try:
                path = Path(row["image_path"])
                if sha(path) != row["original_image_sha256"]: raise RuntimeError("INPUT_MISMATCH: original image SHA")
                with Image.open(path) as raw:
                    record["original_dimensions"] = list(raw.size)
                    image = raw.convert("RGB")
                rendered_question = row["question"] + config["question_suffix"]
                record.update(raw_question=row["question"], rendered_question=rendered_question,
                              raw_question_sha256=hashlib.sha256(row["question"].encode()).hexdigest(),
                              rendered_question_sha256=hashlib.sha256(rendered_question.encode()).hexdigest(),
                              question_suffix=config["question_suffix"])
                batch = prepare(image, rendered_question)
                if config.get("verify_question_unknown_prefix", True) and family != "full_native":
                    alternate_question = "What is visible in the image?" if row["question"] != "What is visible in the image?" else "Describe the objects."
                    alternate = prepare(image, alternate_question + config["question_suffix"])
                    if alternate.boundary != batch.boundary or not torch.equal(alternate.ids[:, :batch.boundary], batch.ids[:, :batch.boundary]):
                        raise RuntimeError("INPUT_MISMATCH: image prefix changes with future question")
                    for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask"):
                        if key in batch.inputs and not torch.equal(batch.inputs[key], alternate.inputs[key]):
                            raise RuntimeError("INPUT_MISMATCH: native visual processing changes with question")
                    record["question_unknown_prefix_verified"] = True
                    del alternate
                else:
                    record["question_unknown_prefix_verified"] = False
                record.update(preprocess_ms=(time.perf_counter() - started) * 1000, actual_visual_token_count=int(batch.visual_positions.numel()),
                              full_prompt=batch.prompt, full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest(),
                              chat_template_hash=hashlib.sha256(processor.chat_template.encode()).hexdigest(),
                              input_tokens=batch.full_length, input_token_ids=batch.ids.detach().cpu().reshape(-1).tolist(),
                              input_ids_sha256=tensor_sha(batch.ids), question_suffix_sha256=tensor_sha(batch.ids[:, batch.boundary:]),
                              compression_boundary=batch.boundary, question_suffix_tokens=batch.suffix_length,
                              visual_positions=batch.visual_positions.detach().cpu().tolist(),
                              native_position_ids_shape=list(batch.positions.shape), native_position_ids_sha256=tensor_sha(batch.positions))
                observer = GenerationObserver(model, started, batch)
                with observe_generation(model, observer):
                    if family == "full_native":
                        observer.phase = "generation"
                        generated = generate_native(model, batch, max_new_tokens=int(config.get("max_new_tokens", 64)))
                        record["reference_scope"] = "native_full_prompt_independent_reference"
                    else:
                        capture_enabled = method.get("encode_scope", "visual") == "prefix"
                        window = int(method.get("constructor_args", {}).get("window_size", 64))
                        context = attention_context(model, capture_queries=True, window_size=window) if capture_enabled else nullcontext(None)
                        t0 = time.perf_counter()
                        with context as capture:
                            prefix_output = prefix_forward(model, batch)
                        torch.cuda.synchronize(); record["prefill_ms"] = (time.perf_counter() - t0) * 1000
                        cache = prefix_output.past_key_values
                        del prefix_output; prefix_output = None
                        vp = batch.visual_positions
                        if hasattr(batch, "rope_deltas"):
                            batch.rope_deltas = model.model.rope_deltas
                        common = {"rope_deltas": model.model.rope_deltas} if hasattr(batch, "rope_deltas") else {}
                        record["native_prefix_owner"] = cache_storage(cache, vp, common_tensors=common)
                        if family not in ("fp16", "native"):
                            refs = [weakref.ref(t) for layer in cache.layers for t in (layer.keys, layer.values)]
                            original_storage = {(str(t.device), t.untyped_storage().data_ptr()) for layer in cache.layers for t in (layer.keys, layer.values)}
                            t0 = time.perf_counter(); owners = None
                            if family != "fp16_hybrid":
                                owners = []
                                for i, layer in enumerate(cache.layers):
                                    adapter = adapter_for(method, i)
                                    if capture_enabled:
                                        owner = adapter.encode(layer.keys, layer.values, visual_positions=vp,
                                            window_queries=capture["queries_by_layer"][i], layer_idx=i,
                                            prefix_is_question_free=True, question_in_observation=False)
                                    else:
                                        owner = adapter.encode(layer.keys.index_select(-2, vp), layer.values.index_select(-2, vp),
                                                               positions=vp, layer_idx=i)
                                    owners.append(owner)
                                del layer, owner, adapter
                            cache = HybridVisualCache(cache, vp, owners, reader_backend=method.get("reader_backend"), consume_native=True)
                            record["method_owner_manifests"] = []
                            for hybrid_layer in cache.layers:
                                state = hybrid_layer.visual_owner
                                report = state.owner_report() if hasattr(state, "owner_report") else state.byte_report()
                                record["method_owner_manifests"].append({k: v for k, v in report.items() if k not in ("storage", "owners")})
                            del state, hybrid_layer
                            owners = capture = None; gc.collect(); torch.cuda.synchronize()
                            record["encode_ms"] = (time.perf_counter() - t0) * 1000
                            if any(ref() is not None for ref in refs): raise RuntimeError("OWNER_AUDIT_FAILED: full native prefix Tensor retained")
                            assert_no_full_prefix_alias(cache, original_storage)
                            record["original_full_prefix_released"] = True
                        else:
                            record["encode_ms"] = 0.0; record["original_full_prefix_released"] = "native_FP16_is_the_declared_owner"
                        # Generation needs IDs/mask and the tiny native delta, not
                        # consumed image features or a persistent three-axis map.
                        keep_ids, keep_mask = batch.ids, batch.mask
                        batch.inputs = {"input_ids": keep_ids, "attention_mask": keep_mask}
                        batch.prefix_inputs = {}; batch.positions = batch.positions.detach().cpu()
                        batch.visual_positions = batch.visual_positions.detach().cpu(); vp = batch.visual_positions
                        capture = None; gc.collect()
                        record["owner_before_suffix"] = cache_storage(cache, vp, common_tensors=common)
                        record["stored_visual_KV_bytes"] = record["owner_before_suffix"]["stored_visual_KV_bytes"]
                        record["complete_visual_owner_bytes"] = record["owner_before_suffix"]["complete_visual_owner_bytes"]
                        for field in ("stored_visual_payload_bytes", "coder_metadata_bytes", "positions_bytes", "encoder_constants_bytes", "text_positions_bytes"):
                            record[field] = record["owner_before_suffix"][field]
                        record["text_kv_bytes"] = record["owner_before_suffix"]["text_kv_bytes"]
                        denominator = spec["layers"] * 2 * spec["kv_heads"] * len(record["visual_positions"]) * spec["head_dim"] * 2
                        record["full_kv_byte_denominator"] = denominator
                        record["actual_effective_visual_bits"] = 16 * record["stored_visual_KV_bytes"] / denominator
                        record["request_input_tensor_bytes"] = keep_ids.untyped_storage().nbytes() + keep_mask.untyped_storage().nbytes()
                        common = None
                        observer.phase = "generation"
                        context = attention_context(model, cache=cache) if hasattr(cache, "named_tensors") else nullcontext()
                        with context:
                            generated = generate_cached(model, batch, cache, max_new_tokens=int(config.get("max_new_tokens", 64)))
                        record["reference_scope"] = "native_image_prefix_then_full_question_suffix_recompute"
                        common = {"rope_deltas": model.model.rope_deltas} if hasattr(batch, "rope_deltas") else {}
                        record["owner_after_generation"] = cache_storage(cache, vp, common_tensors=common)
                    tokens = generated[0, batch.full_length:].detach().cpu().tolist()
                    if not tokens: raise RuntimeError("FAILED_RUN: no generated answer tokens")
                    if observer.stream_ids != tokens: raise RuntimeError("INPUT_MISMATCH: streamer and returned native IDs differ")
                    record.update(output_token_ids=tokens, output_tokens=len(tokens), raw_prediction=processor.decode(tokens, skip_special_tokens=True),
                                  raw_decode_with_special_tokens=processor.decode(tokens, skip_special_tokens=False), status="PASS")
                record["outer_forward_calls"] = observer.calls
                record["ttft_ms"] = (observer.token_times[0] - started) * 1000 if observer.token_times else None
                record["inter_token_ms"] = [(b - a) * 1000 for a, b in zip(observer.token_times, observer.token_times[1:])]
                record["question_prefill_ms"] = next((c["wall_ms"] for c in observer.calls if c["phase"] == "generation"), None)
            except Exception as exc:
                error = str(exc)
                status = ("OOM_AT_DECLARED_INPUT" if "out of memory" in error.lower() else
                          "INPUT_MISMATCH" if "INPUT_MISMATCH" in error or "INPUT_CONTRACT" in error else
                          "OWNER_AUDIT_FAILED" if "OWNER_AUDIT_FAILED" in error else
                          "METHOD_PORT_UNVERIFIED" if "METHOD_PORT_UNVERIFIED" in error else "FAILED_RUN")
                record.update(status=status, error_type=type(exc).__name__, error=error, traceback=traceback.format_exc(),
                              output_token_ids=None, raw_prediction=None)
                if observer is not None: record["outer_forward_calls"] = observer.calls
            finally:
                record.update(wall_ms=(time.perf_counter() - started) * 1000,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              end_allocated_bytes=torch.cuda.memory_allocated())
                records.append(record)
                name = f"{len(records) - 1:06d}_" + hashlib.sha256(row["id"].encode()).hexdigest()[:16] + ".json"
                write_new(args.output / "records" / name, record)
                with (args.output / "records.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n"); out.flush(); os.fsync(out.fileno())
                cache = batch = prefix_output = capture = owners = generated = observer = image = None
                gc.collect(); torch.cuda.empty_cache()
                print(json.dumps({"id": row["id"], "status": record["status"], "completed": len(records), "expected": len(rows)}), flush=True)
    except Exception as exc:
        load_error = {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        # A model/runtime failure must not erase expected QA from coverage.
        done = {r["id"] for r in records}
        for row in rows:
            if row["id"] in done: continue
            record = {**row, "model_id": spec["model_id"], "method": config["method"]["id"], "status": "FAILED_RUN",
                      "output_token_ids": None, "raw_prediction": None, "official_score": None,
                      "error": load_error, "score_status": "NOT_READ_OR_COMPUTED_BY_RUNNER"}
            records.append(record)
            name = f"{len(records) - 1:06d}_" + hashlib.sha256(row["id"].encode()).hexdigest()[:16] + ".json"
            write_new(args.output / "records" / name, record)
            with (args.output / "records.jsonl").open("a", encoding="utf-8") as out: out.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not (args.output / "manifest.json").exists(): write_new(args.output / "manifest.json", manifest)
        counts = {}
        for record in records: counts[record["status"]] = counts.get(record["status"], 0) + 1
        assert len(records) == len(rows) and {r["id"] for r in records} == set(expected)
        final = {"status": "PASS_RAW_COMPLETE" if counts.get("PASS", 0) == len(rows) else "COMPLETE_WITH_FAILURES",
                 "expected_QA": len(rows), "recorded_QA": len(records), "counts": counts,
                 "coverage_complete": True, "failed_QA_must_remain_in_primary_denominator": True,
                 "quality_scored": False, "load_error": load_error, "completed_unix": time.time(),
                 "manifest_sha256": sha(args.output / "manifest.json"), "records_jsonl_sha256": sha(args.output / "records.jsonl")}
        write_new(args.output / "complete.json", final)
        print(json.dumps(final), flush=True)


if __name__ == "__main__": main()
