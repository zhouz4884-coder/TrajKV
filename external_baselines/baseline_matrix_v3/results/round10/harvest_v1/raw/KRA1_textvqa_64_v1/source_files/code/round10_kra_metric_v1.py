"""Fit KRA1 rank-one encoding metrics from frozen fit48 query captures only.

Each source has equal weight. Within one source/layer/KV head, its seven
actual query heads and its one-to-three captured states have equal weight.
No native K/V, W_O, labels, development predictions or selection scores enter
the moment fit. The JSON stores exact runtime Half/F32 encoding parameters.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import torch

SCHEMA = "trajkv.round10.kra_metric.v1"
BASE_PROFILE = "r7_joint_k1_gt_cond"
BASE_FACTOR_SHA = "8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea"
BASE_BOOK_SHA = "af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589"
CALIBRATOR_SHA = "abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a"
METRIC_BYTES = 5980
GAMMA_CAP = 32.0
SIGMA_EPS = 1e-12
DOMAINS = ("textvqa", "infovqa", "docvqa", "chartqa")


def require(value, message):
    if not value:
        raise ValueError("ROUND10_KRA_METRIC: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def normalize_unit_order(value):
    require(isinstance(value, (list, tuple)) and len(value) == 23, "exact original K23 unit order")
    units = []
    for entry in value:
        require(len(entry) == 4, "K stage quadruples required")
        axis, layer, head, depth = entry
        require(axis == "k" and type(layer) is int and 0 <= layer < 28 and
                type(head) is int and 0 <= head < 4 and type(depth) is int and depth == 0, "legal original stage0 K unit")
        units.append([axis, layer, head, 0])
    require(len({tuple(x) for x in units}) == 23, "duplicate K metric unit")
    return units


def metric_tensors(value, *, device="cpu"):
    """Materialize exactly 5888B Half u plus92B F32 gamma; no silent norm."""
    require(isinstance(value, dict) and "u" in value and "gamma" in value, "u/gamma tensors required")
    u = torch.as_tensor(value["u"], dtype=torch.float16, device=device).detach().clone().contiguous()
    gamma = torch.as_tensor(value["gamma"], dtype=torch.float32, device=device).detach().clone().contiguous()
    require(tuple(u.shape) == (23, 128) and tuple(gamma.shape) == (23,), "u[23,128]/gamma[23]")
    require(bool(torch.isfinite(u).all() & torch.isfinite(gamma).all()) and
            bool(((gamma >= 0) & (gamma <= GAMMA_CAP)).all()), "finite u and gamma in[0,32]")
    norms = u.float().norm(dim=-1)
    require(bool(((norms >= .99) & (norms <= 1.01)).all()), "stored Half PCA directions must be near unit norm")
    require(u.untyped_storage().nbytes() + gamma.untyped_storage().nbytes() == METRIC_BYTES, "real metric capacity differs")
    return {"u": u, "gamma": gamma}


def load_metric(path, expected_sha256, *, device="cpu", expected_unit_order=None):
    require(path is not None and isinstance(expected_sha256, str) and len(expected_sha256) == 64, "exact metric file/SHA pair required")
    path = Path(path)
    require(path.suffix.lower() == ".json" and sha(path) == expected_sha256, "metric JSON identity differs")
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    require(doc.get("schema") == SCHEMA and doc.get("base_profile") == BASE_PROFILE and
            doc.get("base_metadata_factor_sha256") == BASE_FACTOR_SHA and
            doc.get("base_direction_book_sha256") == BASE_BOOK_SHA, "original metric lineage differs")
    units = normalize_unit_order(doc["unit_order23"])
    if expected_unit_order is not None:
        require(units == normalize_unit_order(expected_unit_order), "metric unit order differs from original43 filtered K23")
    require(doc.get("metric_storage_bytes") == METRIC_BYTES and
            doc.get("u_dtype") == "float16" and doc.get("gamma_dtype") == "float32" and
            doc.get("basis") == "centered_wht" and len(doc.get("parent_calibration_complete_sha256", "")) == 64,
            "metric storage/basis/parent calibration declaration differs")
    tensors = metric_tensors(doc, device=device)
    info = {key: value for key, value in doc.items() if key not in ("u", "gamma")}
    info.update(path=str(path.resolve()), sha256=expected_sha256,
                actual_tensor_sha256={key: tensor_sha(t) for key, t in tensors.items()},
                actual_persistent_tensor_bytes=METRIC_BYTES,
                metric="I + gamma*u*u^T; stored Half u used without renormalization")
    return tensors, info


def per_source_moments(events, units):
    """Pure CPU aggregation: no source can gain weight from extra EOS states."""
    result = []
    for _, layer, head, _ in units:
        chosen = [event for event in events if event["layer_idx"] == layer]
        steps = [event["step"] for event in chosen]
        require(1 <= len(chosen) <= 3 and sorted(steps) == list(range(len(chosen))), "one-to-three consecutive captured states per layer")
        require(all(event["phase"] == ("suffix_last" if event["step"] == 0 else "teacher_forced_decode") for event in chosen), "actual captured phase differs")
        queries = []
        for event in sorted(chosen, key=lambda item: item["step"]):
            q = event["q_rotated"]
            require(isinstance(q, torch.Tensor) and tuple(q.shape) == (1, 28, 1, 128) and
                    q.dtype == torch.float16 and not q.is_cuda and bool(torch.isfinite(q).all()), "actual frozen CPU Half q_rotated[1,28,1,128]")
            queries.append(q[0, head * 7:(head + 1) * 7, 0].double())
        values = torch.cat(queries, dim=0)
        result.append(values.T @ values / values.shape[0])
    return torch.stack(result)


def pca_metric(moment):
    require(tuple(moment.shape) == (23, 128, 128) and moment.dtype == torch.float64 and
            bool(torch.isfinite(moment).all()), "finite source-balanced F64 moments")
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    principal = eigenvalues[:, -1].clamp_min(0)
    trace = moment.diagonal(dim1=-2, dim2=-1).sum(-1)
    sigma2 = ((trace - principal) / 127).clamp_min(0)
    u = eigenvectors[:, :, -1]
    # Sign changes do not change uuT; fix one sign solely for reproducible JSON.
    pivots = u.abs().argmax(-1)
    signs = torch.where(u[torch.arange(23), pivots] < 0, -1., 1.)
    u = (u * signs[:, None]).half().contiguous()
    gamma = (principal / sigma2.clamp_min(SIGMA_EPS) - 1).clamp(0, GAMMA_CAP).float().contiguous()
    return {"u": u, "gamma": gamma}, {"lambda_max": principal.tolist(), "trace": trace.tolist(),
        "sigma2": sigma2.tolist(), "gamma_cap": GAMMA_CAP, "sigma2_floor": SIGMA_EPS,
        "covariance_dtype": "float64", "uncentered_second_moment": True,
        "actual_Half_u_norms": u.float().norm(dim=-1).tolist(), "moment_sha256": tensor_sha(moment)}


def fit_metric(collection_dir, output_path=None, *, expected_parent_complete_sha256=None, expected_collection_sha256=None):
    """Return a JSON-ready doc; optional output is created, never overwritten."""
    directory = Path(collection_dir)
    parent_path, collection_path, manifest_path = (directory / name for name in
                                                  ("complete.json", "fit_collection_complete.json", "manifest.json"))
    parent_sha, collection_sha, manifest_sha = sha(parent_path), sha(collection_path), sha(manifest_path)
    require(expected_parent_complete_sha256 is None or parent_sha == expected_parent_complete_sha256, "closed parent calibration SHA differs")
    require(expected_collection_sha256 is None or collection_sha == expected_collection_sha256, "fit collection SHA differs")
    parent = json.loads(parent_path.read_text(encoding="utf-8-sig"))
    collection = json.loads(collection_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(parent.get("status") == "PASS_LS48_AND_ACTUAL_MODEL_SELECT16" and parent.get("actual_GPU") is True and
            parent.get("fit_sources") == 48 and parent.get("references_read") is False and
            parent["source_hashes"]["code/round10_calibrate_v1.py"] == CALIBRATOR_SHA and
            manifest["source_hashes"] == parent["source_hashes"], "actual frozen closed calibration required")
    require(collection.get("schema") == "trajkv.round10.fit_collection.v1" and
            collection.get("status") == "PASS_REAL_MODEL_FIT48_ALL28" and collection.get("actual_GPU") is True and
            collection.get("references_read") is False, "actual fit48 capture closure required")
    config = manifest["config"]
    require(config["variant"]["name"] == BASE_PROFILE and
            config["expected_asset_sha256"]["metadata_factor_path"] == BASE_FACTOR_SHA and
            config["expected_asset_sha256"]["direction_book_path"] == BASE_BOOK_SHA, "original low-profile fit captures required")
    selected = config["variant"]["selected_stages"][:43]
    require(len(selected) == 43 and sum(unit[0] == "v" for unit in selected) == 20, "original K23/V20 fit scope")
    units = normalize_unit_order([unit for unit in selected if unit[0] == "k"])
    sources = collection["sources"]
    require(len(sources) == 48 and Counter(item["dataset"] for item in sources) == Counter({domain: 12 for domain in DOMAINS}) and
            len({(item["dataset"], item["source_id"]) for item in sources}) == 48 and
            len({item["id"] for item in sources}) == 48, "four-domain, unique, source-balanced fit48")
    moments = torch.zeros((23, 128, 128), dtype=torch.float64)
    records = []
    for item in sources:
        path = Path(item["file"])
        require(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], "immutable private fit shard identity differs")
        case = torch.load(path, map_location="cpu", weights_only=True)
        require(case.get("schema") == "trajkv.round10.ls_fit_case.v1" and case.get("all_28_layers_captured") is True and
                [list(unit) for unit in case["selected_stages"]] == selected, "original all28 fit case required")
        source = case["source"]
        require(all(source[key] == item[key] for key in ("id", "dataset", "source_id")) and
                source["original_image_sha256"] == item["context"]["image_sha256"], "fit case/source identity differs")
        events = case["events"]
        require(len(events) == item["events"] and {event["layer_idx"] for event in events} == set(range(28)), "actual all28 event coverage")
        require(all(event["source_id"] == item["source_id"] and event["id"] == item["id"] and
                    event["dataset"] == item["dataset"] and event["image_sha256"] == source["original_image_sha256"]
                    for event in events), "event provenance differs")
        contribution = per_source_moments(events, units)
        moments.add_(contribution / 48)
        records.append({key: item[key] for key in ("id", "dataset", "source_id", "sha256", "bytes", "events")})
        del case, contribution
    tensors, detail = pca_metric(moments)
    metric_tensors(tensors)
    require(sha(parent_path) == parent_sha and sha(collection_path) == collection_sha and sha(manifest_path) == manifest_sha and
            all(sha(item["file"]) == item["sha256"] for item in sources), "source changed during metric fit")
    doc = {"schema": SCHEMA, "base_profile": BASE_PROFILE, "base_metadata_factor_sha256": BASE_FACTOR_SHA,
        "base_direction_book_sha256": BASE_BOOK_SHA, "basis": "centered_wht", "key_gain_mode": "centered",
        "parent_calibration_complete_sha256": parent_sha, "fit_collection_complete_sha256": collection_sha,
        "parent_calibration_manifest_sha256": manifest_sha, "unit_order23": units,
        "u": tensors["u"].tolist(), "gamma": tensors["gamma"].tolist(), "u_dtype": "float16", "gamma_dtype": "float32",
        "metric_storage_bytes": METRIC_BYTES, "strength": 1.0, "gamma_cap": GAMMA_CAP,
        "fit_provenance": {"fit_source_count": 48, "fit_domain_counts": {domain: 12 for domain in DOMAINS},
            "source_weighting": "one48th per source; within source/layer, equal actual1-3states and7GQAqueryheads",
            "fit_input": "stored actual q_rotated only; uncentered second moment", "PCA": detail,
            "source_shards": records, "cal_fit_inputs_sha256": config["cal_fit_sha256"],
            "fit_source_hashes": parent["source_hashes"], "metric_fit_source_sha256": sha(__file__),
            "python": platform.python_version(), "torch": torch.__version__,
            "W_O_or_native_KV_used": False, "calSelect_or_development_scores_used": False, "labels_or_references_read": False},
        "parameter_selection": "gamma strength not selected here; strength1 metric fit only",
        "task_quality_or_speed_gain_claimed": False}
    if output_path is not None:
        write_new(output_path, doc)
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-complete-sha256")
    parser.add_argument("--fit-collection-sha256")
    args = parser.parse_args()
    doc = fit_metric(args.collection, args.output, expected_parent_complete_sha256=args.parent_complete_sha256,
                     expected_collection_sha256=args.fit_collection_sha256)
    print(json.dumps({"status": "FIT48_QUERY_MOMENT_METRIC_WRITTEN", "output": str(args.output),
                      "sha256": sha(args.output), "runtime_tensor_bytes": METRIC_BYTES, "fit_sources": 48,
                      "gamma": doc["gamma"], "QA_events": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
