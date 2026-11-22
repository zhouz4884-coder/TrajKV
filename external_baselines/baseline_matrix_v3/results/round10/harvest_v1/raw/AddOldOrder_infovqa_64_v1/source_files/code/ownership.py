"""Physical storage accounting for immutable visual owners and exact text KV."""
from __future__ import annotations
import torch


def storage_report(tensors):
    seen, rows, total = {}, {}, 0
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        charged = 0 if identity in seen else storage.nbytes()
        first = seen.setdefault(identity, name)
        rows[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                      "logical_bytes": tensor.numel() * tensor.element_size(), "storage_bytes": storage.nbytes(),
                      "charged_bytes": charged, "shared_with": None if first == name else first}
        total += charged
    return {"total_bytes": total, "tensors": rows}


def model_storage(model):
    parameters = storage_report(dict(model.named_parameters()))
    all_tensors = {"parameters." + n: t for n, t in model.named_parameters()}
    all_tensors.update({"buffers." + n: t for n, t in model.named_buffers()})
    combined = storage_report(all_tensors)
    return {"weight_resident_bytes": parameters["total_bytes"], "all_model_tensor_bytes": combined["total_bytes"],
            "buffer_bytes_incremental": combined["total_bytes"] - parameters["total_bytes"]}


def cache_storage(cache, visual_positions, *, common_tensors=None):
    """Hybrid actual tensors, or native disjoint visual/text logical regions.

    Native allocation capacity must equal its logical K/V size, with no aliases;
    otherwise assigning the allocation to two regions would hide extra capacity.
    """
    common_tensors = common_tensors or {}
    if hasattr(cache, "named_tensors"):
        tensors = dict(cache.named_tensors())
        report = storage_report({**tensors, **{"common." + k: v for k, v in common_tensors.items()}})
        payload = metadata = positions = constants = lookup = tails = text = text_positions = other = 0
        for name, row in report["tensors"].items():
            charge = row["charged_bytes"]
            leaf = name.rsplit(".", 1)[-1]
            if name == "original_visual_positions" or name.startswith("common."):
                positions += charge
            elif ".visual." in name:
                if leaf == "positions":
                    positions += charge
                elif "encoder_constants" in name or leaf in ("k_lut", "v_lut"):
                    constants += charge
                    if leaf in ("k_lut", "v_lut"): lookup += charge
                elif leaf in ("k_code", "v_code", "k_packed", "v_packed", "key_mse_indices", "key_qjl_signs",
                              "value_codes", "keys", "values", "k_tail", "v_tail"):
                    payload += charge
                    if leaf in ("k_tail", "v_tail"): tails += charge
                else:
                    metadata += charge
            elif name.endswith(("text_keys", "text_values")):
                text += charge
            elif leaf == "text_positions":
                text_positions += charge
            else: other += charge
        complete = payload + metadata + positions + constants
        assert report["total_bytes"] == complete + text + text_positions + other
        report.update(stored_visual_KV_bytes=complete, complete_visual_owner_bytes=complete,
                      stored_visual_payload_bytes=payload, coder_metadata_bytes=metadata,
                      positions_bytes=positions, encoder_constants_bytes=constants,
                      lookup_bytes_subset_of_constants=lookup, full_precision_visual_tail_bytes_subset_of_payload=tails,
                      text_kv_bytes=text, text_positions_bytes=text_positions, other_persistent_metadata_bytes=other,
                      accounting_scope="global_unique_storages_across_all_layers_and_shared_constants")
        return report
    tensors = {}
    visual_bytes, logical = 0, 0
    nv = int(visual_positions.numel())
    for i, layer in enumerate(cache.layers):
        for kind, tensor in (("K", layer.keys), ("V", layer.values)):
            size = tensor.numel() * tensor.element_size()
            if tensor.untyped_storage().nbytes() != size:
                raise RuntimeError("OWNER_AUDIT_FAILED: native cache backing capacity exceeds logical K/V")
            tensors[f"layers.{i}.{kind}"] = tensor
            logical += size
            visual_bytes += tensor.shape[0] * tensor.shape[1] * nv * tensor.shape[-1] * tensor.element_size()
    report = storage_report(tensors)
    if report["total_bytes"] != logical:
        raise RuntimeError("OWNER_AUDIT_FAILED: native cache has overlapping K/V storage regions")
    kv_total = report["total_bytes"]
    all_report = storage_report({**tensors, **{"common." + k: v for k, v in common_tensors.items()}})
    common_bytes = all_report["total_bytes"] - kv_total
    complete = visual_bytes + common_bytes
    all_report.update(stored_visual_KV_bytes=complete, complete_visual_owner_bytes=complete,
                      stored_visual_payload_bytes=visual_bytes, coder_metadata_bytes=0,
                      positions_bytes=common_bytes, text_positions_bytes=0, text_kv_bytes=kv_total - visual_bytes,
                      encoder_constants_bytes=0,
                      accounting_scope="native_full_allocation_equals_disjoint_visual_plus_text_regions")
    return all_report


def assert_no_full_prefix_alias(cache, source_storage_ids):
    for tensor in cache.named_tensors().values():
        if (str(tensor.device), tensor.untyped_storage().data_ptr()) in source_storage_ids and tensor.numel():
            raise RuntimeError("OWNER_AUDIT_FAILED: compressed cache retains native full-prefix backing")
