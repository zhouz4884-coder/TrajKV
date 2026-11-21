"""Temporary language-only SDPA dispatch and native post-RoPE Q capture.

No model forward, positional encoding, mask registry or HF source is replaced.
Vision modules always delegate to the original registered SDPA function.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch

_ACTIVE_CONTEXT = None


def language_attention_modules(model):
    try:
        language = model.model.language_model
        modules = [layer.self_attn for layer in language.layers]
    except AttributeError as exc:
        raise ValueError("UNSUPPORTED_MODEL: expected model.model.language_model.layers[*].self_attn") from exc
    for index, module in enumerate(modules):
        if getattr(module, "layer_idx", index) != index:
            raise ValueError("UNSUPPORTED_MODEL: native layer_idx differs from language layer order")
        if getattr(module, "sliding_window", None) not in (None, 0):
            raise ValueError("UNSUPPORTED_MODEL: sliding language attention")
    return language, modules


@contextmanager
def attention_context(model, cache=None, capture_queries=False, window_size=64):
    global _ACTIVE_CONTEXT
    if _ACTIVE_CONTEXT is not None:
        raise RuntimeError("Nested attention contexts are unsupported")
    if window_size <= 0 or (cache is not None and capture_queries):
        raise ValueError("Capture must use a native question-free prefix separately from compressed inference")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    language, modules = language_attention_modules(model)
    if cache is not None and len(cache.layers) != len(modules):
        raise ValueError("INPUT_MISMATCH: cache and native language layer counts differ")
    identities = {id(module): index for index, module in enumerate(modules)}
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    had_local = "sdpa" in ALL_ATTENTION_FUNCTIONS._local_mapping
    previous_local = ALL_ATTENTION_FUNCTIONS._local_mapping.get("sdpa")
    state = {"queries_by_layer": {}, "captured_tokens_by_layer": {},
             "capture_scope": "CALLER_VERIFIES_QUESTION_FREE_NATIVE_PREFIX",
             "reader_scope": "language_self_attention_only"}
    config_records = []
    seen = set()
    for component in [language, *modules]:
        config = component.config
        if id(config) not in seen:
            seen.add(id(config))
            config_records.append((config, config._attn_implementation))

    def dispatch(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        index = identities.get(id(module))
        original_kwargs = dict(dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        if index is None:
            return original(module, query, key, value, attention_mask, **original_kwargs)
        if capture_queries:
            previous = state["queries_by_layer"].get(index)
            latest = query.detach()
            if previous is not None:
                latest = torch.cat((previous, latest), dim=-2)
            state["queries_by_layer"][index] = latest[..., -window_size:, :].contiguous().clone()
            state["captured_tokens_by_layer"][index] = state["captured_tokens_by_layer"].get(index, 0) + query.shape[-2]
        if cache is None:
            return original(module, query, key, value, attention_mask, **original_kwargs)
        layer = cache.layers[index]
        if key.data_ptr() != layer.keys.data_ptr() or value.data_ptr() != layer.values.data_ptr():
            raise ValueError("INPUT_MISMATCH: native attention did not consume this HybridVisualCache.update result")
        return cache.read(index, query, module, attention_mask, scaling, original,
                          dropout=dropout, is_causal=is_causal, **kwargs)

    _ACTIVE_CONTEXT = state
    try:
        for config, _ in config_records:
            config._attn_implementation = "sdpa"
        ALL_ATTENTION_FUNCTIONS["sdpa"] = dispatch
        yield state
    finally:
        if had_local:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = previous_local
        elif "sdpa" in ALL_ATTENTION_FUNCTIONS._local_mapping:
            del ALL_ATTENTION_FUNCTIONS["sdpa"]
        for config, value in reversed(config_records):
            config._attn_implementation = value
        _ACTIVE_CONTEXT = None


def synthetic_gate(device):
    """Synthetic facade acceptance, independent of any model/data quality."""
    import hashlib
    import ast
    import math
    from pathlib import Path
    from types import SimpleNamespace
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import transformers
    from cache_bridge import HybridVisualCache, FP16VisualOwner
    from methods.turboquant_adapter import TurboQuantAdapter
    from methods.eviction_adapter import PyramidKVVisualAdapter
    from methods.calibquant_adapter import CalibQuantAdapter
    torch.manual_seed(81973)
    if device.type == "cuda":
        assert torch.cuda.get_device_capability(device) == (8, 6)
        torch.cuda.reset_peak_memory_stats(device)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    calib_source = Path(__file__).resolve().parents[1] / "third_party/calibquant/patch_internvl.py"
    calib_ast = ast.parse(calib_source.read_text(encoding="utf-8"))
    source_text_expr = next(node.value for node in ast.walk(calib_ast) if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "attn_weights_quant_t" for target in node.targets))
    source_text_qk = compile(ast.Expression(source_text_expr), str(calib_source), "eval")
    config = SimpleNamespace(_attn_implementation="eager")
    class Attention(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_idx, self.config = index, config
            self.num_key_value_groups, self.is_causal, self.scaling = 7, True, 0.19
    language = torch.nn.Module()
    language.config = config
    language.layers = torch.nn.ModuleList()
    for i in range(2):
        layer = torch.nn.Module()
        layer.self_attn = Attention(i)
        language.layers.append(layer)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = language
    model.eval()
    result = {"status": "RUNNING", "scope": "SYNTHETIC_CACHE_AND_SDPA_FACADE_NO_MODEL_QUALITY",
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": str(device), "tests": [], "runtime_module_file": str(Path(__file__).resolve())}
    def record(name, **values):
        result["tests"].append(dict(name=name, status="PASS", **values))
    def compare(a, b, tolerance=1e-3):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        return float((a.float() - b.float()).abs().max())
    prefix_k = [torch.randn(1, 4, 192, 128, device=device, dtype=torch.float16) for _ in range(2)]
    prefix_v = [torch.randn_like(k) for k in prefix_k]
    prefix_q = [torch.randn(1, 28, 192, 128, device=device, dtype=torch.float16) for _ in range(2)]
    vp = torch.tensor([i for i in range(12, 180) if i not in (27, 65, 129)], device=device)
    with attention_context(model, capture_queries=True, window_size=8) as capture:
        for i, layer in enumerate(language.layers):
            actual = ALL_ATTENTION_FUNCTIONS["sdpa"](layer.self_attn, prefix_q[i], prefix_k[i], prefix_v[i], None, scaling=.19)[0]
            expected = original(layer.self_attn, prefix_q[i], prefix_k[i], prefix_v[i], None, scaling=.19)[0]
            compare(actual, expected, 0)
        vision = torch.nn.Module()
        vision.is_causal = False
        vq = torch.randn(1, 2, 5, 128, device=device, dtype=torch.float16)
        actual = ALL_ATTENTION_FUNCTIONS["sdpa"](vision, vq, vq, vq, None, is_causal=False)[0]
        compare(actual, original(vision, vq, vq, vq, None, is_causal=False)[0], 0)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original and config._attn_implementation == "eager"
    for i, q in capture["queries_by_layer"].items():
        compare(q, prefix_q[i][..., -8:, :], 0)
        assert q.untyped_storage().nbytes() == q.numel() * q.element_size()
    record("native_capture_exact_vision_delegation_and_restore", captured_layers=len(capture["queries_by_layer"]))
    try:
        with attention_context(model):
            raise RuntimeError("controlled facade restoration test")
    except RuntimeError as exc:
        assert str(exc) == "controlled facade restoration test"
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original and config._attn_implementation == "eager"
    record("exception_restores_registry_and_configs")
    for name in ("FP16", "TurboQuant", "PyramidKV", "CalibQuant"):
        native = DynamicCache()
        for i in range(2):
            native.update(prefix_k[i], prefix_v[i], i)
        owners = None
        backend = None
        if name == "TurboQuant":
            owners = [TurboQuantAdapter().encode(prefix_k[i].index_select(-2, vp), prefix_v[i].index_select(-2, vp),
                                                 positions=vp, layer_idx=i) for i in range(2)]
            backend = "upstream_triton_score" if device.type == "cuda" else "torch_dequant"
        elif name == "PyramidKV":
            adapter = PyramidKVVisualAdapter(compression_ratio=.5, window_size=8, num_layers=2)
            owners = [adapter.encode(prefix_k[i], prefix_v[i], visual_positions=vp,
                                      window_queries=capture["queries_by_layer"][i], layer_idx=i,
                                      prefix_is_question_free=True) for i in range(2)]
        elif name == "CalibQuant":
            backend = "upstream_triton" if device.type == "cuda" else "torch_unfused"
            adapter = CalibQuantAdapter(bits=1, reader_backend=backend)
            owners = [adapter.encode(prefix_k[i].index_select(-2, vp), prefix_v[i].index_select(-2, vp)) for i in range(2)]
        cache = HybridVisualCache(native, vp, owners, reader_backend=backend)
        assert len(native.layers) == 0 and cache.get_seq_length() == 192 and cache.is_sliding == [False, False]
        assert cache.get_mask_sizes(3, 0) == (195, 0) and cache.get_max_cache_shape() == -1
        native_storage = {t.untyped_storage().data_ptr() for t in prefix_k + prefix_v}
        assert all(t.untyped_storage().data_ptr() not in native_storage for t in cache.named_tensors().values())
        errors = []
        scale = 128 ** -0.5 if name == "CalibQuant" else .19
        for query_length in (3, 1):
            before = cache.get_seq_length()
            with attention_context(model, cache=cache):
                for i, layer_module in enumerate(language.layers):
                    q = torch.randn(1, 28, query_length, 128, device=device, dtype=torch.float16)
                    k = torch.randn(1, 4, query_length, 128, device=device, dtype=torch.float16)
                    v = torch.randn_like(k)
                    old_text_k, old_text_v = cache.layers[i].keys, cache.layers[i].values
                    text_k, text_v = cache.update(k, v, i)
                    assert cache.layers[i].query_positions.tolist() == list(range(before, before + query_length))
                    length = before + query_length
                    mask = (torch.arange(length, device=device)[None, :] <= torch.arange(before, length, device=device)[:, None])[None, None]
                    actual = ALL_ATTENTION_FUNCTIONS["sdpa"](layer_module.self_attn, q, text_k, text_v, mask, scaling=scale)[0]
                    owner = cache.layers[i].visual_owner
                    if name == "FP16":
                        shape = (1, 4, length, 128)
                        full_k, full_v = torch.empty(shape, device=device, dtype=torch.float16), torch.empty(shape, device=device, dtype=torch.float16)
                        full_k[:, :, vp], full_v[:, :, vp] = owner.keys, owner.values
                        full_k[:, :, cache.layers[i].text_positions], full_v[:, :, cache.layers[i].text_positions] = text_k, text_v
                        expected = original(layer_module.self_attn, q, full_k, full_v, mask, scaling=scale)[0]
                        errors.append(compare(actual, expected, 0))
                    elif hasattr(owner, "attend"):
                        expected = owner.attend(q, exact_keys=text_k, exact_values=text_v,
                                                exact_positions=cache.layers[i].text_positions,
                                                query_positions=cache.layers[i].query_positions, scale=scale)
                        errors.append(compare(actual, expected.transpose(1, 2), 2e-3))
                    else:
                        # Independent source read, then full original key-order softmax;
                        # this catches scale applied twice and mixed-axis masking bugs.
                        visual_logits = owner.read(q, softmax_scale=scale).float()
                        exact_k = text_k.repeat_interleave(7, dim=1)
                        # Execute the exact assignment expression from the
                        # author's released patch, rather than mirror our bridge.
                        text_logits = eval(source_text_qk, {"torch": torch, "math": math,
                                           "self": SimpleNamespace(head_dim=128),
                                           "query_states": q, "key_states_quant_t": exact_k}).float()
                        logits = torch.empty(1, 28, query_length, length, device=device)
                        logits[..., vp] = visual_logits
                        logits[..., cache.layers[i].text_positions] = text_logits
                        probs = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1).to(q.dtype)
                        expected = owner.weighted_value(probs.index_select(-1, vp))
                        expected += probs.index_select(-1, cache.layers[i].text_positions) @ text_v.repeat_interleave(7, dim=1)
                        errors.append(compare(actual, expected.transpose(1, 2), 2e-3))
            assert cache.get_seq_length() == before + query_length
        cache.crop(192)
        assert all(layer.logical_length == 192 and layer.keys.shape[-2] == 192 - len(vp) for layer in cache.layers)
        record(name + "_logical_cache_irregular_visual_positions_mixed_reader", max_error=max(errors),
               reader_backend=backend, text_dtype=str(cache.layers[0].keys.dtype),
               consumes_native_full_prefix=True, no_native_full_storage_alias=True)
        if name == "CalibQuant":
            result["tests"][-1]["source_oracle"] = "AST exact author patch attn_weights_quant_t FP16 QK/division before FP32 softmax"
            result["tests"][-1]["source_oracle_sha256"] = hashlib.sha256(calib_source.read_bytes()).hexdigest()
    # Fail closed on protocols that could otherwise silently change semantics.
    try:
        cache.batch_repeat_interleave(2)
    except ValueError as exc:
        assert "UNSUPPORTED_CACHE_PROTOCOL" in str(exc)
    else:
        raise AssertionError("batch expansion was not rejected")
    record("unsupported_batch_expansion_rejected")
    result["status"] = "PASS"
    if device.type == "cuda":
        torch.cuda.synchronize()
        result["hardware"] = torch.cuda.get_device_name(device)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    paths = [Path(__file__), Path(__file__).with_name("cache_bridge.py")]
    result["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-gate", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = synthetic_gate(torch.device(args.device))
    except Exception:
        result = {"status": "FAILED_RUN", "scope": "SYNTHETIC_CACHE_AND_SDPA_FACADE_NO_MODEL_QUALITY",
                  "traceback": traceback.format_exc()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"status": result["status"], "tests": len(result["tests"]), "output": str(args.output)}))
