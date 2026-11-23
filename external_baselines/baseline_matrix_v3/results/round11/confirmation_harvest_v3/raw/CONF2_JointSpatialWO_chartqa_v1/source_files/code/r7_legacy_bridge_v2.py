"""Two exact frozen R7 profiles; unchanged whole-model encoders/readers.

Low: published gt_cond43 / G4_16 KeyBank and its Round7 packed reader.
High: original r7_budget_k2_cond / G4_256 DepthResidualBank and its distinct
Round6-depth reader. No codec, residual order, budget rule or factor fitting is
implemented here. V1 and every historical source remain read-only.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import torch
import r7_legacy_bridge_v1 as _v1


V1_SHA256 = "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b"
LOW = "r7_joint_k1_gt_cond"
HIGH = "r7_budget_k2_cond"
HIGH_ASSETS = {
    "direction_book_path": "9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae",
    "metadata_factor_path": "d60fe4ffcf31e36306ed2c5a4069634627c83b3af2f0b86fdb5c37e456113671",
    "round7_policy_path": "e8a58e602c047adbd4ca6ae253f6ba2eb3961158d7c75444d3e27a41ec47ed6e",
}
HIGH_SOURCES = {
    "a2/round6_depth_bank.py": "1d423bf03cdc76c0650776230d6a1093710e9c9fb8ac2cd84974d791aa65c6f7",
    "a2/round6_depth_attention.py": "f96277c726c39f2880e8618525b4bdcbac0049e68e7c4d3e2eb699fa24dfd945",
}


def _require(value, message):
    _v1._require(value, "V2: " + message)


def _frozen_v1():
    path = Path(_v1.__file__).resolve()
    _require(path == Path(__file__).resolve().with_name("r7_legacy_bridge_v1.py") and
             _v1._sha(path) == V1_SHA256, "immutable V1 bridge changed or imported from another project")


def profile_definition(name):
    """JSON-only identity; logical workpoint labels are not actual bit caps."""
    _require(name in (LOW, HIGH), "only the two named original profiles are supported")
    return {
        "name": name,
        "legacy_method": "round7_key" if name == LOW else "round6_depth",
        "legacy_bank_class": "round7_key_bank.KeyBank" if name == LOW else "round6_depth_bank.DepthResidualBank",
        "legacy_reader": "round7_key_attention.attention" if name == LOW else "round6_depth_attention.attention",
        "book_shape": [16, 4] if name == LOW else [256, 4],
        "key_code_bytes_per_token_head": 16 if name == LOW else 32,
        "fixed_asset_sha256": dict(_v1._DEFAULT_ASSETS if name == LOW else HIGH_ASSETS),
        "declared_legacy_budget_bits": 1.8,
        "legacy_stage_limit": 43 if name == LOW else None,
        "allocation": "original priority prefix capped43 and affordable" if name == LOW else
                      "original affordable priority prefix; no imposed43 limit",
        "strict1p3_or1p8_complete_residency_asserted": False,
    }


def _validate_high_variant(variant, expected_assets=None):
    _require(variant.get("name") == HIGH and variant.get("method") == "round6_depth", "wrong high-profile identity")
    _require(variant.get("cap") == 8 and variant.get("bits_v", 1) == 1 and
             variant.get("v_layout", "channel") == "channel" and variant.get("v_encoder", "lloyd") == "lloyd" and
             variant.get("v_group_size", "all") == "all", "preserve original M1/V1 input and encoding settings")
    _require(float(variant.get("budget_bits", 1.8)) == 1.8, "high point keeps original1.8 allocation budget")
    _require(all(variant.get(k) is None for k in ("res_stages_limit", "res_units_limit", "residual_units_limit")),
             "high profile has no fixed43 or other residual-stage limit")
    _require(variant.get("key_assignment", "l2") == "l2" and variant.get("key_amplitude", "norm") == "norm",
             "high source encoder keeps its implicit normalized nearest-word / norm rule")
    _require(variant.get("selector", "none") == "none" and variant.get("store_selector_meta", False) is False,
             "no new selector")
    _require(not any(variant.get(k) for k in ("functional_v_factor_path", "key_metric_path", "interaction_policy_path",
                                               "local_calibration", "diagnostic", "compact_history")), "no new fitting or policy")
    _require(variant.get("residual_layout_reference", "actual") == "actual", "no alternate allocation layout")
    ranking = variant.get("selected_stages")
    _require(isinstance(ranking, list) and len(ranking) == 224, "complete original224-stage priority required")
    _require(expected_assets is None or dict(expected_assets) == HIGH_ASSETS, "high-profile assets cannot be replaced")
    assets = {}
    for field, digest in HIGH_ASSETS.items():
        path = Path(variant[field]).resolve()
        _require(path.is_file() and _v1._sha(path) == digest, "frozen high asset changed: " + field)
        assets[field] = {"path": str(path), "sha256": digest}
    policy = json.loads(Path(variant["round7_policy_path"]).read_text(encoding="utf-8-sig"))
    priority = [list(x) + [0] if len(x) == 3 else list(x) for x in policy["selected_units_priority"]]
    _require(priority == ranking, "high profile priority differs from frozen conditional-batches artifact")
    return assets


def _extra_source_pins(values):
    result = dict(HIGH_SOURCES)
    for key, value in (values or {}).items():
        _require(key not in result or result[key] == value, "cannot replace the original high bank/reader source")
        result[key] = value
    return result


@torch.no_grad()
def build_r7_owners(native_prefix_cache, visual_positions, variant, *, legacy_a2,
                    legacy_experiments, legacy_scripts, kivi_source, expected_asset_sha256=None,
                    expected_legacy_source_sha256=None):
    """Same interface as V1, selecting the precise bank/reader by variant.name.

    Full native input K/V is caller-owned until it replaces/consumes its prefix.
    Returned owners never retain those buffers. Complete accounting must also
    include the caller's Hybrid originalVP/text and other persistent storage.
    """
    _frozen_v1()
    _require(not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32,
             "frozen reference encoding requires autocast off and TF32 disabled")
    name = variant.get("name")
    definition = profile_definition(name)
    if name == LOW:
        _require(expected_asset_sha256 is None or dict(expected_asset_sha256) == _v1._DEFAULT_ASSETS,
                 "low-profile assets cannot be replaced")
        owners, metadata = _v1.build_r7_owners(
            native_prefix_cache, visual_positions, variant, legacy_a2=legacy_a2,
            legacy_experiments=legacy_experiments, legacy_scripts=legacy_scripts,
            kivi_source=kivi_source, expected_asset_sha256=expected_asset_sha256,
            expected_legacy_source_sha256=expected_legacy_source_sha256)
        metadata = dict(metadata, schema="trajkv.r7_legacy_dual_bridge.v2", frozen_profile=definition,
                        immutable_v1_bridge_sha256=V1_SHA256,
                        dual_bridge_source_sha256=_v1._sha(__file__))
        return owners, metadata

    a2, experiments, scripts = (Path(p).resolve() for p in (legacy_a2, legacy_experiments, legacy_scripts))
    _require(a2.name == "a2" and experiments.name == "experiments" and a2.parent == experiments.parent,
             "preserve original sibling code/a2 and code/experiments layout")
    _require(all(p.is_dir() for p in (a2, experiments, scripts)) and Path(kivi_source).is_file(), "explicit legacy paths required")
    cfg = deepcopy(variant); variant_sha = _v1._canonical(cfg)
    assets = _validate_high_variant(cfg, expected_asset_sha256)
    source_pins = _v1._source_pin_paths(a2, experiments, scripts, kivi_source,
                                       _extra_source_pins(expected_legacy_source_sha256))
    _v1._check_loaded_pinned_modules(source_pins)
    layers = native_prefix_cache.layers
    _require(len(layers) == 28, "only M1L28")
    key = layers[0].keys
    _require(key.is_cuda and key.dtype == torch.float16 and tuple(key.shape[:2]) == (1, 4) and key.shape[-1] == 128,
             "native M1 CUDA FP16 prefix required")
    shape, device = key.shape, key.device
    _require(all(x.keys.shape == x.values.shape == shape and x.keys.device == x.values.device == device and
                 x.keys.dtype == x.values.dtype == torch.float16 for x in layers), "native layer geometry differs")
    vp = visual_positions
    _require(vp.ndim == 1 and vp.dtype == torch.long and vp.device == device and vp.numel() > 0 and
             torch.equal(vp, vp.unique(sorted=True)) and int(vp[0]) >= 0 and int(vp[-1]) + 1 == shape[-2] and
             vp.numel() % 32 == 0, "sorted original visual positions / exact image boundary / divisible32 required")
    forbidden = {_v1._storage(t) for x in layers for t in (x.keys, x.values)}
    proxy, model = _v1._CacheProxy(layers), _v1._model_proxy(28)
    roots = (a2.parent, scripts, Path(kivi_source).resolve().parent)
    bank = None
    with _v1._legacy_paths(a2, experiments, scripts):
        module = importlib.import_module("round6_depth_bank")
        reader = importlib.import_module("round6_depth_attention").attention
        _v1._check_loaded_pinned_modules(source_pins)
        sources_before = _v1._legacy_manifest(roots)
        cls = type("_PerBuildFrozenR7DepthBank", (module.DepthResidualBank,), {"settings": cfg})
        try:
            bank = cls(model, proxy, vp, "round6_depth", scripts, Path(kivi_source), keep_reference=False)
            _require(tuple(bank.book.shape) == (256, 4) and all(x is None for x in bank.references) and
                     all(not hasattr(s, "visual_k") and not hasattr(s, "visual_v") for s in bank.states),
                     "high encoder must preserve G4_256 and release FP16 visual source")
            old = bank.owned()
            legacy_bytes = int(old["visual_base"] + old["visual_residual"] + old["selector_metadata"] +
                               old["visual_position"] + bank.encoder_parameter_bytes)
            # The original high encoder folds these values but keeps only its
            # declared byte charge. Keep a real frozen copy for physical owner
            # accounting; do not apply it a second time or change the encoder.
            factors_json = json.loads(Path(cfg["metadata_factor_path"]).read_text(encoding="utf-8-sig"))
            factors = {key: torch.tensor(factors_json[key], dtype=torch.float32, device=device)
                       for key in ("key_head_factors", "value_head_factors", "value_center_offsets")}
            _require(factors_json["parameter_count"] == 336 and all(tuple(t.shape) == (28, 4) for t in factors.values()),
                     "exact original336 encoder factors required")
            shared = {"book": bank.book, "global_residual_lookup": bank.residual_lookup,
                      "encoder_ranking_bytes": torch.tensor(list(bank.encoder_ranking_table), dtype=torch.uint8, device=device),
                      "encoder_policy_bytes": torch.tensor(list(bank.encoder_depth_policy), dtype=torch.uint8, device=device)}
            shared.update({"encoder_factor." + key: value for key, value in factors.items()})
            owners = [_v1.R7LegacyOwner(i, state, shared, reader, shape[-2]) for i, state in enumerate(bank.states)]
            selected = deepcopy(bank.selected_residual_stages)
            allocation = {key: deepcopy(bank.depth_plan.get(key)) for key in
                          ("fixed_bytes", "allowed_bytes", "affordable_stages", "predicted_bytes", "actual_budget_bits", "stage_bytes")}
            bank.close(); bank._parent = bank.cache = bank.model = None
        finally:
            if bank is not None and bank._parent is not None:
                bank.close()
            for x in proxy.layers:
                x.keys = x.values = None
        sources = _v1._legacy_manifest(roots)
        _require(all(sources[path]["sha256"] == item["sha256"] for path, item in sources_before.items()) and
                 all(_v1._sha(path) == item["sha256"] for path, item in source_pins.items()), "source changed during high encode")
        _v1._check_loaded_pinned_modules(source_pins)
        if expected_legacy_source_sha256 is not None:
            _require(set(sources) <= set(source_pins), "explicit source map must cover every imported legacy source")
    _require(_v1._canonical(cfg) == variant_sha == _v1._canonical(variant) and
             all(_v1._sha(x["path"]) == x["sha256"] for x in assets.values()), "variant or asset changed during build")
    _require(all(not hasattr(x.self_attn, "_p185_read") for x in model.model.language_model.layers), "stub hook cleanup failed")
    unique = {}
    for owner in owners:
        _require(all(not isinstance(t, torch.Tensor) or t.numel() == 0 or key in _v1._STATE_TENSORS
                     for key, t in vars(owner.state).items()), "unreported persistent legacy tensor")
        for key, t in owner.named_tensors().items():
            _require(_v1._storage(t) not in forbidden and not t.requires_grad and t.grad_fn is None,
                     "owner retains native source or autograd graph: " + key)
            if t.numel(): unique.setdefault(_v1._storage(t), t.untyped_storage().nbytes())
    metadata = {
        "schema": "trajkv.r7_legacy_dual_bridge.v2", "model": "M1", "frozen_profile": definition,
        "layers": 28, "kv_heads": 4, "query_heads": 28, "head_dim": 128,
        "visual_tokens": int(vp.numel()), "prefix_boundary": int(shape[-2]), "variant": deepcopy(cfg),
        "variant_sha256": variant_sha, "fixed_assets": assets, "selected_stages": selected,
        "selected_stage_count": len(selected), "legacy_allocation": allocation,
        "legacy_declared_complete_visual_bytes": legacy_bytes,
        "returned_owners_unique_tensor_capacity_bytes": sum(unique.values()),
        "legacy_sources": sources, "verified_legacy_source_pins": source_pins,
        "all_imported_legacy_sources_pinned": set(sources) <= set(source_pins),
        "kivi_source": {"path": str(Path(kivi_source).resolve()), "sha256": _v1._sha(kivi_source)},
        "immutable_v1_bridge_sha256": V1_SHA256, "dual_bridge_source_sha256": _v1._sha(__file__),
        "original_FP16_visual_storage_retained_by_owners": False, "real_model_or_cache_patched": False,
        "stub_hooks_closed": True, "legacy_text_reserve_released": True, "encoding_autograd_disabled": True,
        "reader_unchanged": True, "reader_is_round6_depth_not_round7_key": True,
        "encoder_factors_materialized_for_ownership_only_not_reapplied": True,
        "ranking_descriptor_materialized_as_actual_cuda_uint8_storage": True,
        "physical_accounting": "Global unique actual tensor capacities; HybridVP/text additional; no legacy-byte equality or strict-cap substitution.",
        "legacy_byte_equality_or_strict1p3_claimed": False,
        "numerical_or_GPU_equivalence_acceptance": "NOT_ESTABLISHED_BY_METADATA; independent CUDA controls required",
    }
    return owners, metadata
