"""KVarN low-level PyTorch mechanism port for controlled KV experiments.

This is NOT an official end-to-end KVarN/vLLM reproduction. It invokes the
unmodified upstream Sinkhorn, RTN, packing and dequantization functions via a
small loader, avoiding the vLLM package initializer and CUDA extensions.

Input layout: [..., tokens, head_dim]. Full token groups are compressed; an
incomplete trailing group stays in the input dtype. The wrapper returns tensors
for the project's canonical byte-accounting module; it makes no serving-memory
or bandwidth claim. Only the upstream quantization functions run here.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from dataclasses import dataclass

import torch

UPSTREAM_COMMIT = "7586257f1c632e63187bfacbbe21ccb51540f7b3"
UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "KVarN"
EXPECTED_SHA256 = {
    "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py":
        "3af527df8352c5886935ba64e3d99bdd52ac62f7274e8a653c0fd485c867fed9",
    "vllm/v1/attention/ops/kvarn_store.py":
        "db0a8e00f360c8aaf3cfe551ac5fdb7548875999ee04e0635ef4ab0d8848ccdf",
    "vllm/v1/attention/ops/kvarn_decode.py":
        "18b1a79d3c0507a31a71930e83e391ae75c9907c2d98c7071bec58f0ad1951e1",
}


def _load_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def _upstream():
    root = UPSTREAM_ROOT
    if source_fingerprints() != EXPECTED_SHA256:
        raise RuntimeError("Pinned KVarN source changed; re-audit before using the port")
    sink_path = root / "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py"
    store_path = root / "vllm/v1/attention/ops/kvarn_store.py"
    decode_path = root / "vllm/v1/attention/ops/kvarn_decode.py"
    sink = _load_file("_trajkv_kvarn_sinkhorn_ref", sink_path)
    decode = _load_file("_trajkv_kvarn_decode_ref", decode_path)

    # The sole vLLM import in this pure-Torch reference is variance_normalize.
    # Bind the same upstream function directly, leaving every algorithm intact.
    tree = ast.parse(store_path.read_text(encoding="utf-8"), str(store_path))
    removed = 0
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == (
            "vllm.model_executor.layers.quantization.kvarn.sinkhorn"
        ):
            if [n.name for n in node.names] != ["variance_normalize"]:
                raise RuntimeError("Upstream reference import changed; audit loader")
            removed += 1
        else:
            nodes.append(node)
    if removed != 1:
        raise RuntimeError("Upstream reference changed; expected one Sinkhorn import")
    tree.body = nodes
    store = ModuleType("_trajkv_kvarn_store_ref")
    store.__file__ = str(store_path)
    store.variance_normalize = sink.variance_normalize
    exec(compile(tree, str(store_path), "exec"), store.__dict__)
    return sink, store, decode


def source_fingerprints() -> dict[str, str]:
    paths = (
        "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py",
        "vllm/v1/attention/ops/kvarn_store.py",
        "vllm/v1/attention/ops/kvarn_decode.py",
    )
    return {p: hashlib.sha256((UPSTREAM_ROOT / p).read_bytes()).hexdigest()
            for p in paths}


@functools.lru_cache(maxsize=8)
def _hadamard(dim: int, device: str) -> torch.Tensor:
    if dim < 2 or dim & (dim - 1):
        raise ValueError("head_dim must be a power of two >= 2")
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / dim ** 0.5


@dataclass
class KVarNRecord:
    shape: tuple[int, ...]
    dtype: torch.dtype
    axis: str
    bits: int
    group: int
    full_tokens: int
    packed: dict[str, torch.Tensor]
    tail: torch.Tensor
    rotation_dtype: str

    def storage_tensors(self) -> dict[str, torch.Tensor]:
        """Use these actual tensors with code/bytes.py; no allocator claim."""
        return {**self.packed, "full_precision_tail": self.tail}


@torch.no_grad()
def encode_kvarn(x: torch.Tensor, axis: str, bits: int = 2, group: int = 128,
                 iterations: int = 16,
                 rotation_dtype: str = "float32") -> KVarNRecord:
    """Encode full tiles using upstream functions; keep an incomplete tail.

    rotation_dtype='float32' controls codec error without extra fp16 GEMM
    rounding. 'float16' also rounds rotations like the upstream serving path.
    Neither option reproduces the complete official serving implementation.
    """
    if axis not in {"k", "v"} or bits not in {2, 4}:
        raise ValueError("axis must be k/v; bits must be 2/4")
    if x.ndim < 2 or x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("expected floating tensor [..., tokens, head_dim]")
    if group < 4 or group % 4:
        raise ValueError("group must be >=4 and divisible by four")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if rotation_dtype not in {"float32", "float16"}:
        raise ValueError("rotation_dtype must be float32/float16")
    sink, store, _ = _upstream()
    n, d = x.shape[-2:]
    h = _hadamard(d, str(x.device))
    heads = 1
    for size in x.shape[:-2]:
        heads *= size
    flat = x.detach().reshape(heads, n, d)
    full_n = (n // group) * group
    tail = flat[:, full_n:, :].clone()
    packed = {}
    if full_n:
        tiles = flat[:, :full_n, :].reshape(-1, group, d)
        if rotation_dtype == "float16":
            rotated = (tiles.half() @ h.half()).float()
        else:
            rotated = tiles.float() @ h
        oriented = rotated.transpose(-1, -2).contiguous() if axis == "k" else rotated
        balanced, sc, sr = sink.variance_normalize_batched(oriented, iterations)
        func = (store.kvarn_store_tile_k_batch_from_sinkhorn if axis == "k"
                else store.kvarn_store_tile_v_batch_from_sinkhorn)
        packed = func(balanced, sc.squeeze(-2), sr.squeeze(-1), bits)
    return KVarNRecord(tuple(x.shape), x.dtype, axis, bits, group, full_n,
                       packed, tail, rotation_dtype)


@torch.no_grad()
def decode_kvarn(record: KVarNRecord) -> torch.Tensor:
    """Decode upstream packed records to the original frame, shape and dtype."""
    _, _, decode = _upstream()
    r = record
    n, d = r.shape[-2:]
    heads = 1
    for size in r.shape[:-2]:
        heads *= size
    if not r.full_tokens:
        return r.tail.reshape(r.shape).clone()
    p = r.packed
    q = decode._unpack_lowbit(p["q_packed_uint8"],
                             r.group if r.axis == "k" else d, r.bits).float()
    if r.axis == "k":
        rotated = ((q * p["s_col_K"].float().unsqueeze(-1)
                    + p["zp_K"].float().unsqueeze(-1))
                   * p["s_row_K"].float().unsqueeze(-2)).transpose(-1, -2)
    else:
        rotated = ((q * p["s_row_V"].float().unsqueeze(-1)
                    + p["zp_V"].float().unsqueeze(-1))
                   * p["s_col_V"].float().unsqueeze(-2))
    h = _hadamard(d, str(q.device))
    if r.rotation_dtype == "float16":
        unrotated = (rotated.half() @ h.half()).to(r.dtype)
    else:
        unrotated = (rotated @ h).to(r.dtype)
    prefix = unrotated.reshape(heads, r.full_tokens, d)
    return torch.cat((prefix, r.tail), dim=1).reshape(r.shape)


@torch.no_grad()
def quant_dequant_kvarn(x: torch.Tensor, axis: str, bits: int = 2,
                        group: int = 128, iterations: int = 16,
                        rotation_dtype: str = "float32") -> torch.Tensor:
    return decode_kvarn(encode_kvarn(x, axis, bits, group, iterations,
                                    rotation_dtype))
