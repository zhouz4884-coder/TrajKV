"""Isolated M1 bridge to the unchanged, whole-model Round7 KeyBank.

This file does not register an evaluation method or launch an experiment.
Legacy PhaseState sees disposable model/cache proxies only. The old codec,
allocation and packed attention remain unchanged. CUDA equivalence is a
separate acceptance step; importing this module performs no legacy imports.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


_DEFAULT_ASSETS = {
    "direction_book_path": "af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589",
    "metadata_factor_path": "8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea",
    "round7_policy_path": "505dbbbc67d50023ec0f60fb31ab9a31b0a27212fe7c3e8a2e72de0307b3f38f",
}
# The a2 numerical pins match the actual R7 confirmation manifest. The
# non-a2 historical dependencies are independently recorded frozen originals.
_DEFAULT_SOURCES = {
    "a2/round7_key_bank.py": "38b6e0ea2486a900e404fa6347c91972198795f8972dc7ab2bf90527fbc1cdc5",
    "a2/round7_key_attention.py": "b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0",
    "a2/round7_key_codec.py": "80c2c16220097242b25d3f843a3b57569590131c02618e766e35f0f0791ab651",
    "a2/round7_key_layout.py": "7b9773f65103062dd51c8ccc64d9b5047f88112d951b6eb10e7938f8fc3f5a04",
    "a2/round7_key_artifacts.py": "f429aa502dae27b7616e8dfca63ac6d498d16f18d8bf2db4ad674b5039760485",
    "a2/round6_depth_codec.py": "333ef93165fd786bc40015f6f8c775ce3de20bb98da26ffa6555c38cba5f12d4",
    "a2/fast_wht.py": "9dc73998f66d5b7bafc96a5aa3db57974d1ca8e604e432e7539625f104d90ee3",
    "a2/quantizers.py": "314bc458c9c2d84579b2a02cb8a45908e0099b613368be40a07669ca5d273e38",
    "a2/affine_rate_bank.py": "8090240cd1ee5edde07d1252c41f4dfcaa8aebf167a2b7d890d37ae57edcd2c0",
    "a2/masked_bank.py": "7c313fddfdadc81939235b0a4fd843540961f8841c69d7400101b33a66737d9d",
    "a2/metadata_direction_bank.py": "bea958a75c8df84702b20ba32836ae6f4f7ffaae8f6a1024ea6de3c18659ff61",
    "experiments/bank.py": "f1ab22bf04f592263246bac82c415fd9a2b8c6adb4d813a5b46d056f08987ee5",
    "experiments/fused_progressive.py": "d0130e31d055f2c2ced6de1cc26a5cbfa99eea0642eb60ce5f4b6c359cd6520c",
    "experiments/check_fused_progressive.py": "b7578afafcb4bc522bc0742b99e4ee5ab49307fb120b3475dc553900076dac55",
    "code/bytes.py": "00625b902a29027075c13d64ca9019753e4aa153215aea6f349d25902a848115",
    "scripts/p185_model_state.py": "437dbb5e9bd34e699ddc1f589834d207248074e214dfa821143781cd4fe3c4af",
    "scripts/p185_single_state.py": "ca4bce75c1e80c0a99eb18451323f2d0c2a07a8c707a3238f1a4e8d5d9114b3c",
    "scripts/p185_segmented_kernel.py": "bf09b996f098aea5027f25de54b50aef78f305225de105d9b8c9343751eb3f64",
    "scripts/p176_kivi_codec.py": "1c278ebd1f648170696208ffed89a01e96960c18670610948d742262ad6b7cbb",
    "scripts/p201_refinement.py": "a1b8b84b9dbc655c21ead0bffa032be1ae4af97b5da4768112fe2a6106e58afb",
    "scripts/p202_refinement.py": "292aca3d0f793c164b1d4204af21573cd4092597eb3f25055ded6d879afa04aa",
    "kivi_source": "3f1140b0faff2cb524a3eb684167e403dffd488477b301826497462fd36fe540",
}
_STATE_TENSORS = (
    "direction_code", "direction_gain", "direction_mean", "v_code", "v_scale", "v_min",
    "depth_residual_code", "depth_residual_alpha", "depth_residual_lookup", "visual_pos",
)


def _require(condition, message):
    if not condition:
        raise ValueError("R7_LEGACY_BRIDGE: " + message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _storage(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())


class _CacheProxy:
    """No bound method or layer descriptor from the real DynamicCache survives."""
    def __init__(self, layers):
        self.layers = [SimpleNamespace(keys=layer.keys, values=layer.values) for layer in layers]
        self._logical_length = int(layers[0].keys.shape[-2])

    def update(self, *args, **kwargs):
        raise RuntimeError("Disposable encoding proxy must never execute a model forward")

    def get_seq_length(self, layer_idx=0):
        return self._logical_length

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self._logical_length + query_length, 0


def _model_proxy(layers):
    # PhaseState writes _p185_read on these stubs, never on the real HF model.
    modules = [SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(layers)]
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=modules)))


@contextmanager
def _legacy_paths(a2, experiments, scripts):
    previous = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        # a2 and experiments are sibling directories of the original layout.
        sys.path[:0] = [str(a2), str(experiments), str(scripts), str(a2.parent)]
        yield
    finally:
        sys.path[:] = previous
        sys.dont_write_bytecode = previous_bytecode


def _legacy_manifest(roots):
    result = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        path = Path(path).resolve()
        if path.suffix == ".py" and path.is_file() and any(_inside(path, root) for root in roots):
            result[str(path)] = {"sha256": _sha(path), "module_names": []}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and str(Path(path).resolve()) in result:
            result[str(Path(path).resolve())]["module_names"].append(name)
    return result


def _source_pin_paths(a2, experiments, scripts, kivi_source, extra=None):
    pins = dict(_DEFAULT_SOURCES)
    for name, digest in (extra or {}).items():
        _require(name not in pins or pins[name] == digest, "cannot replace a fixed R7 source pin: " + name)
        pins[name] = digest
    roots = {"a2": a2, "experiments": experiments, "scripts": scripts,
             "code": a2.parent, "kivi": Path(kivi_source).resolve().parent}
    result = {}
    for name, digest in pins.items():
        if name == "kivi_source":
            path = Path(kivi_source).resolve()
        else:
            group, relative = name.split("/", 1)
            _require(group in roots and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                     "invalid legacy source pin role: " + name)
            path = (roots[group] / relative).resolve()
            _require(_inside(path, roots[group]), "legacy source pin escaped its explicit root")
        _require(path.is_file() and len(digest) == 64 and _sha(path) == digest, "frozen legacy source changed: " + name)
        result[str(path)] = {"role": name, "sha256": digest}
    return result


def _check_loaded_pinned_modules(pins):
    for path in pins:
        module = sys.modules.get(Path(path).stem)
        if module is not None and getattr(module, "__file__", None):
            _require(str(Path(module.__file__).resolve()) == path, "conflicting imported numerical module: " + Path(path).stem)


def _validate_variant(variant, expected_assets):
    _require(variant.get("method") == "round7_key", "only the frozen round7_key profile is supported")
    _require(variant.get("key_assignment", "l2") == "l2" and variant.get("key_amplitude", "norm") == "norm",
             "first port preserves L2 assignment and norm amplitude")
    _require(variant.get("bits_v", 1) == 1 and variant.get("v_group_size", "all") == "all",
             "first port preserves whole-image V1")
    _require(variant.get("res_stages_limit") == 43 and float(variant.get("budget_bits", 1.8)) == 1.8,
             "preserve the historical 43-stage / declared1.8-cap profile")
    _require(variant.get("residual_layout_reference", "actual") == "actual", "no alternate allocation layout")
    ranking = variant.get("selected_stages")
    _require(isinstance(ranking, list) and len(ranking) == 224, "complete historical224-stage ranking required")
    _require(variant.get("selector", "none") == "none" and variant.get("store_selector_meta", False) is False,
             "no new selector metadata or scheduling")
    _require(not any(variant.get(k) for k in ("functional_v_factor_path", "key_metric_path", "local_calibration",
                                               "interaction_policy_path", "diagnostic", "compact_history")),
             "no new metric, fitted parameters, history policy or retained reference")
    assets = {}
    for field in _DEFAULT_ASSETS:
        path = Path(variant[field]).resolve()
        _require(path.is_file() and field in expected_assets and _sha(path) == expected_assets[field],
                 "missing or changed fixed asset: " + field)
        assets[field] = {"path": str(path), "sha256": expected_assets[field]}
    policy = json.loads(Path(variant["round7_policy_path"]).read_text(encoding="utf-8-sig"))
    policy_ranking = [list(unit) + [0] if len(unit) == 3 else list(unit)
                      for unit in policy["selected_units_priority"]]
    _require(policy_ranking == ranking, "variant ranking differs from its frozen policy artifact")
    return assets


class R7LegacyOwner:
    """One packed layer; native text is borrowed only during attend()."""
    def __init__(self, layer_idx, state, shared, reader, boundary):
        self.layer_idx, self.state, self.shared = layer_idx, state, shared
        self._reader, self.boundary, self._active = reader, int(boundary), False
        self.positions = state.visual_pos
        self._empty_k = state.direction_gain.new_empty((1, 4, 0, 128))
        self._empty_v = self._empty_k.new_empty((1, 4, 0, 128))
        self._empty_pos = self.positions.new_empty((0,))
        self._clear_text()

    def _clear_text(self):
        self.state.text_k, self.state.text_v = self._empty_k, self._empty_v
        self.state.text_pos, self.state.text_length = self._empty_pos, 0
        self.state.reserve = 0

    def named_tensors(self, include_constants=True):
        names = {"direction_code": "k_code", "direction_gain": "k_gain", "direction_mean": "k_mean",
                 "visual_pos": "positions", "depth_residual_lookup": "encoder_constants.lookup_view"}
        result = {names.get(name, name): getattr(self.state, name) for name in _STATE_TENSORS
                  if isinstance(getattr(self.state, name, None), torch.Tensor)}
        if include_constants:
            result.update({"encoder_constants." + name: value for name, value in self.shared.items()})
        return result

    owner_tensors = named_tensors

    def attend(self, q, *, exact_keys, exact_values, exact_positions, query_positions, scale=None):
        _require(not self._active, "concurrent use of the same layer owner is unsupported")
        _require(q.ndim == 4 and q.shape[0] == 1 and q.shape[1] == 28 and q.shape[-1] == 128,
                 "only M1 batch1 / Hq28 / D128")
        _require(exact_keys.shape == exact_values.shape and exact_keys.ndim == 4 and
                 tuple(exact_keys.shape[:2]) == (1, 4) and exact_keys.shape[-1] == 128,
                 "native exact-text GQA geometry differs")
        _require(q.stride(-1) == 1 and exact_keys.stride(-1) == exact_values.stride(-1) == 1 and
                 exact_keys.stride(-2) == exact_values.stride(-2) == 128 and
                 exact_keys.stride(1) == exact_values.stride(1), "legacy text reader requires matching row-contiguous KV strides")
        n, nt = q.shape[-2], exact_keys.shape[-2]
        _require(n > 0 and query_positions.ndim == exact_positions.ndim == 1 and
                 query_positions.numel() == n and exact_positions.numel() == nt, "position shapes differ")
        _require(query_positions.dtype == exact_positions.dtype == torch.long, "native serial int64 positions required")
        logical = int(query_positions[-1].item()) + 1
        expected = torch.arange(logical - n, logical, device=q.device, dtype=torch.long)
        _require(torch.equal(query_positions, expected) and int(query_positions[0]) >= self.boundary,
                 "queries must be a contiguous post-boundary tail")
        _require(nt >= n and torch.equal(exact_positions[-n:], query_positions), "exact text must include these queries")
        union = torch.cat((self.positions, exact_positions)).sort().values
        _require(torch.equal(union, torch.arange(logical, device=q.device, dtype=torch.long)),
                 "visual/text positions must be a disjoint complete causal history")
        _require(scale is None or math.isclose(float(scale), 128 ** -0.5, rel_tol=0.0, abs_tol=1e-12),
                 "legacy reader has fixed1/sqrt128 scale")
        _require(q.is_cuda and q.dtype == exact_keys.dtype == exact_values.dtype == torch.float16 and
                 q.device == exact_keys.device == exact_values.device == self.positions.device,
                 "unchanged reader requires same-device CUDA FP16")
        self._active = True
        try:
            self.state.text_k, self.state.text_v = exact_keys, exact_values
            self.state.text_pos, self.state.text_length, self.state.length = exact_positions, nt, logical
            return self._reader(q, self.state, self.shared["book"], self.positions.numel())
        finally:
            self._clear_text()
            self._active = False


@torch.no_grad()
def build_r7_owners(native_prefix_cache, visual_positions, variant, *, legacy_a2,
                    legacy_experiments, legacy_scripts, kivi_source, expected_asset_sha256=None,
                    expected_legacy_source_sha256=None):
    """Return (28 owners, JSON metadata), without touching the real cache/model.

    Caller retains/replaces native prefix text using its own HybridVisualCache,
    then releases original prefix K/V before the persistent-owner measurement.
    Default hashes identify published gt_cond43; an explicit hash map can select
    another already frozen policy. Default numerical source pins are immutable.
    An explicit extra source map also requires every actually imported legacy
    Python source to be covered. No codec, order or factor refitting occurs.
    """
    a2, experiments, scripts = (Path(p).resolve() for p in (legacy_a2, legacy_experiments, legacy_scripts))
    _require(a2.name == "a2" and experiments.name == "experiments" and a2.parent == experiments.parent,
             "preserve the original sibling code/a2 and code/experiments layout")
    _require(all(p.is_dir() for p in (a2, experiments, scripts)) and Path(kivi_source).is_file(), "explicit legacy paths required")
    cfg = deepcopy(variant); variant_sha = _canonical(cfg)
    assets = _validate_variant(cfg, dict(expected_asset_sha256 or _DEFAULT_ASSETS))
    source_pins = _source_pin_paths(a2, experiments, scripts, kivi_source, expected_legacy_source_sha256)
    _check_loaded_pinned_modules(source_pins)
    layers = native_prefix_cache.layers
    _require(len(layers) == 28, "only M1 L28")
    keys = layers[0].keys
    _require(keys.is_cuda and keys.dtype == torch.float16 and tuple(keys.shape[:2]) == (1, 4) and keys.shape[-1] == 128,
             "actual native M1 CUDA FP16 prefix required")
    shape, device = keys.shape, keys.device
    _require(all(layer.keys.shape == layer.values.shape == shape and layer.keys.device == layer.values.device == device and
                 layer.keys.dtype == layer.values.dtype == torch.float16 for layer in layers), "native layer geometry differs")
    vp = visual_positions
    _require(vp.ndim == 1 and vp.dtype == torch.long and vp.device == device and vp.numel() > 0 and
             torch.equal(vp, vp.unique(sorted=True)) and int(vp[0]) >= 0 and int(vp[-1]) + 1 == shape[-2],
             "prefix ends at the last sorted original visual placeholder")
    _require(vp.numel() % 32 == 0, "legacy format requires Nv divisible32; no padding is introduced")
    forbidden = {_storage(t) for layer in layers for t in (layer.keys, layer.values)}
    proxy, model = _CacheProxy(layers), _model_proxy(28)
    roots = (a2.parent, scripts, Path(kivi_source).resolve().parent)
    bank = None
    with _legacy_paths(a2, experiments, scripts):
        for name in ("bank", "bytes", "masked_bank", "round7_key_bank", "round7_key_attention", "round7_key_codec"):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__file__", None):
                _require(any(_inside(module.__file__, root) for root in roots), "conflicting imported legacy name: " + name)
        module = importlib.import_module("round7_key_bank")
        reader = importlib.import_module("round7_key_attention").attention
        sources_before = _legacy_manifest(roots)
        cls = type("_PerBuildFrozenR7KeyBank", (module.KeyBank,), {"settings": cfg})
        try:
            bank = cls(model, proxy, vp, "round7_key", scripts, Path(kivi_source), keep_reference=False)
            legacy_bytes = int(bank.owned()["complete_visual_owner_bytes"])
            _require(all(ref is None for ref in bank.references) and
                     all(not hasattr(s, "visual_k") and not hasattr(s, "visual_v") for s in bank.states),
                     "old encoder must release FP16 visual source")
            ranking = torch.tensor(list(bank.encoder_ranking_table), dtype=torch.uint8, device=device)
            descriptor = torch.tensor(list(bank.encoder_depth_policy), dtype=torch.uint8, device=device)
            shared = {"book": bank.book, "global_residual_lookup": bank.residual_lookup,
                      "encoder_ranking_bytes": ranking, "encoder_policy_bytes": descriptor}
            shared.update({"encoder_factor." + name: value for name, value in bank.encoder_factors.items()})
            owners = [R7LegacyOwner(i, state, shared, reader, shape[-2]) for i, state in enumerate(bank.states)]
            selected = deepcopy(bank.selected_residual_stages)
            # Restore only the disposable stubs. No real HF hook was installed.
            bank.close()
            bank._parent = bank.cache = bank.model = None
        finally:
            if bank is not None and bank._parent is not None:
                bank.close()
            for layer in proxy.layers:
                layer.keys = layer.values = None
        sources = _legacy_manifest(roots)
        _require(all(sources[path]["sha256"] == info["sha256"] for path, info in sources_before.items()),
                 "legacy source changed during encoding")
        _check_loaded_pinned_modules(source_pins)
        _require(all(_sha(path) == item["sha256"] for path, item in source_pins.items()), "pinned legacy source changed during encoding")
        if expected_legacy_source_sha256 is not None:
            _require(set(sources) <= set(source_pins), "explicit source map does not cover every imported legacy source")
    _require(_canonical(cfg) == variant_sha and _canonical(variant) == variant_sha, "variant mutated")
    _require(all(_sha(v["path"]) == v["sha256"] for v in assets.values()), "asset changed during encoding")
    _require(all(not hasattr(layer.self_attn, "_p185_read") for layer in model.model.language_model.layers),
             "stub hook cleanup failed")
    unique = {}
    for owner in owners:
        _require(all(not isinstance(value, torch.Tensor) or value.numel() == 0 or name in _STATE_TENSORS
                     for name, value in vars(owner.state).items()), "unreported persistent tensor in legacy state")
        for name, tensor in owner.named_tensors().items():
            _require(_storage(tensor) not in forbidden, "owner aliases original full native KV: " + name)
            _require(not tensor.requires_grad and tensor.grad_fn is None, "owner retains an autograd graph")
            if tensor.numel():
                unique.setdefault(_storage(tensor), tensor.untyped_storage().nbytes())
    metadata = {
        "schema": "trajkv.r7_legacy_whole_model_bridge.v1", "model": "M1",
        "layers": 28, "kv_heads": 4, "query_heads": 28, "head_dim": 128,
        "visual_tokens": int(vp.numel()), "prefix_boundary": int(shape[-2]),
        "variant": deepcopy(cfg), "variant_sha256": variant_sha, "fixed_assets": assets,
        "legacy_sources": sources, "kivi_source": {"path": str(Path(kivi_source).resolve()), "sha256": _sha(kivi_source)},
        "verified_legacy_source_pins": source_pins,
        "all_imported_legacy_sources_pinned": set(sources) <= set(source_pins),
        "encoding_autograd_disabled": True,
        "selected_stages": selected, "selected_stage_count": len(selected),
        "legacy_declared_complete_visual_bytes": legacy_bytes,
        "returned_owners_unique_tensor_capacity_bytes": sum(unique.values()),
        "ranking_descriptor_materialized_as_actual_cuda_uint8_storage": True,
        "original_FP16_visual_storage_retained_by_owners": False,
        "real_model_or_cache_patched": False, "stub_hooks_closed": True,
        "legacy_text_reserve_released": True, "reader_unchanged": True,
        "physical_accounting": "Global unique storages across all owners; constants and lookup slices share original backing. Hybrid originalVP/text and other persistent buffers are additional actual charges.",
        "legacy_byte_equality_or_strict1p3_claimed": False,
        "numerical_or_GPU_equivalence_acceptance": "NOT_ESTABLISHED_BY_BUILD_METADATA; requires independent CUDA controls",
    }
    return owners, metadata
