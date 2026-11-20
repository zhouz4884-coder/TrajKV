"""Auditable visual-only adapter to 0xSero/turboquant, never Google official.

The selected third-party algorithm is K=TurboQuantProd and V=group min/max
quantization. K3V2 / K3V4 have 2.5 / 3.5 *payload* bits per K/V scalar;
norms, scales, zero points, positions, and shared constants are additional.
This is not a reproduction of the paper's mixed-channel experimental setup.
Upstream is imported unchanged; see docs/TURBOQUANT_EVICTION_SOURCE_AUDIT.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import sys
from typing import Any

import torch

UPSTREAM_REPO = "https://github.com/0xSero/turboquant"
UPSTREAM_COMMIT = "31660314b229b8d1c29bfddf8b9f5026b5121095"
PAPER = "https://arxiv.org/abs/2504.19874"
_SOURCE_VERIFIED = False


def _upstream():
    global _SOURCE_VERIFIED
    root = Path(__file__).resolve().parents[2] / "third_party" / "turboquant_0xsero"
    if not _SOURCE_VERIFIED:
        lock_path = root.parents[1] / "configs" / "methods" / "turboquant_eviction_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))["sources"]["turboquant_0xsero"]
        if lock["commit"] != UPSTREAM_COMMIT:
            raise RuntimeError("INPUT_MISMATCH: TurboQuant source lock commit changed")
        for relative, expected in lock["files"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"INPUT_MISMATCH: TurboQuant frozen upstream bytes changed: {relative}")
        _SOURCE_VERIFIED = True
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    from turboquant.quantizer import TurboQuantProd, _qjl_powers
    from turboquant.kv_cache import quantize_values, dequantize_values
    return TurboQuantProd, quantize_values, dequantize_values, _qjl_powers


def storage_bytes(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Count real storage allocations once, including any view's entire storage."""
    seen = set()
    items = {}
    total = 0
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        unique = key not in seen
        seen.add(key)
        nbytes = storage.nbytes() if unique else 0
        total += nbytes
        items[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "device": str(tensor.device), "logical_bytes": tensor.numel() * tensor.element_size(),
                       "storage_bytes": storage.nbytes(), "charged_bytes": nbytes}
    return {"total_bytes": total, "tensors": items}


@dataclass
class TurboQuantOwner:
    key_quantized: Any
    value_quantized: Any
    codec: Any
    positions: torch.Tensor
    shape: tuple[int, ...]
    source_dtype: torch.dtype
    value_bits: int
    group_size: int
    layer_idx: int

    def named_tensors(self, include_constants: bool = True) -> dict[str, torch.Tensor]:
        q, v = self.key_quantized, self.value_quantized
        out = {"key_mse_indices": q.mse_indices, "key_qjl_signs": q.qjl_signs,
               "key_norms": q.norms, "key_residual_norms": q.residual_norms,
               "value_codes": v.data, "value_scales": v.scales, "value_zeros": v.zeros,
               "positions": self.positions}
        if include_constants:
            out.update({"encoder_constants." + k: t for k, t in self.codec.named_buffers()})
            out["encoder_constants.qjl_bit_powers"] = _upstream()[3](q.qjl_signs.device)
        return out

    def owner_report(self) -> dict[str, Any]:
        all_tensors = self.named_tensors()
        payload_names = ("key_mse_indices", "key_qjl_signs", "value_codes")
        metadata_names = ("key_norms", "key_residual_norms", "value_scales", "value_zeros")
        payload = storage_bytes({k: all_tensors[k] for k in payload_names})["total_bytes"]
        metadata = storage_bytes({k: all_tensors[k] for k in metadata_names})["total_bytes"]
        positions = storage_bytes({"positions": self.positions})["total_bytes"]
        constants = storage_bytes({k: v for k, v in all_tensors.items() if k.startswith("encoder_constants.")})["total_bytes"]
        scalar_count = 2 * math.prod(self.shape)
        return {"method": f"TurboQuant_0xSero_K3V{self.value_bits}_visual_port",
                "upstream_repo": UPSTREAM_REPO, "upstream_commit": UPSTREAM_COMMIT,
                "method_paper": PAPER, "faithfulness_status": "THIRD_PARTY_ALGORITHM_VISUAL_SCOPE_PORT",
                "reader": "upstream_dequantize_then_torch_grouped_attention",
                "qjl_enabled": True, "recent_visual_buffer_tokens": 0,
                "numerical_variant": "UPSTREAM_FP32_ENCODER_AND_FP32_METADATA",
                "bits_nominal": (3 + self.value_bits) / 2,
                "payload_bytes": payload, "coder_metadata_bytes": metadata,
                "positions_bytes": positions, "encoder_constants_bytes": constants,
                "lookup_bytes": 0, "stored_visual_KV_bytes": payload + metadata + positions,
                "all_owner_bytes_including_constants": storage_bytes(all_tensors)["total_bytes"],
                "actual_effective_visual_bits": 8 * (payload + metadata + positions) / scalar_count,
                "effective_bits_including_constants": 8 * (payload + metadata + positions + constants) / scalar_count,
                "full_kv_byte_denominator": scalar_count * torch.empty((), dtype=self.source_dtype).element_size(),
                "temporary_full_dequant_KV_bytes": 2 * math.prod(self.shape) * 4,
                "temporary_full_dequant_is_persistent": False,
                "storage": storage_bytes(all_tensors)}

    def materialize(self, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Real read from packed owner. No decoded KV is retained by this owner."""
        k = self.codec.dequantize(self.key_quantized)
        v = _upstream()[2](self.value_quantized, self.group_size)
        return k.to(dtype or k.dtype), v.to(dtype or v.dtype)

    def read_visual_logits(self, query: torch.Tensor, scale: float | None = None,
                           backend: str = "torch_dequant") -> torch.Tensor:
        """Actual source packed-key Triton reader, or explicit dense reference.

        Original source only supports one Q and equal query/KV heads; dispatch
        original kernels once per query and GQA group, without duplicating K.
        This is a compatibility dispatch, not a new optimized kernel.
        """
        b, hq, nq, d = query.shape
        hkv = self.shape[1]
        if hq % hkv:
            raise ValueError("Invalid GQA geometry")
        if backend == "torch_dequant":
            k = self.codec.dequantize(self.key_quantized)
            grouped = query.float().reshape(b, hkv, hq // hkv, nq, d)
            return (grouped @ k.float().unsqueeze(2).transpose(-1, -2)).reshape(b, hq, nq, -1) * (scale or d ** -0.5)
        if backend != "upstream_triton_score":
            raise ValueError(f"Unrecognized explicit reader backend: {backend}")
        if query.device.type != "cuda":
            raise ValueError("HARDWARE_UNSUPPORTED: upstream Triton reader requires CUDA")
        from turboquant.triton_kernels import turboquant_attention_score
        out = torch.empty((b, hq, nq, self.shape[-2]), device=query.device, dtype=torch.float32)
        groups = hq // hkv
        for qi in range(nq):
            for group in range(groups):
                q = query[:, group::groups, qi:qi + 1, :].contiguous()
                scores = turboquant_attention_score(q, self.key_quantized, self.codec.mse_quantizer.Pi,
                                                    self.codec.S, self.codec.mse_quantizer.centroids,
                                                    self.key_quantized.mse_bits, self.codec.qjl_scale)
                out[:, group::groups, qi, :] = scores.reshape(b, hkv, self.shape[-2])
        return out * (scale or d ** -0.5)

    def weighted_value(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Source V decoder plus weighted reduction; full FP32 V is temporary."""
        b, hq, nq, n = probabilities.shape
        hkv, d = self.shape[1], self.shape[-1]
        if n != self.shape[-2] or hq % hkv:
            raise ValueError("Probability shape is incompatible with visual owner")
        v = _upstream()[2](self.value_quantized, self.group_size)
        p = probabilities.float().reshape(b, hkv, hq // hkv, nq, n)
        return (p @ v.float().unsqueeze(2)).reshape(b, hq, nq, d)

    def attend(self, query: torch.Tensor, *, exact_keys: torch.Tensor | None = None,
               exact_values: torch.Tensor | None = None, exact_positions: torch.Tensor | None = None,
               query_positions: torch.Tensor | None = None, scale: float | None = None,
               backend: str = "torch_dequant") -> torch.Tensor:
        if backend == "upstream_triton_score":
            b, hq, nq, d = query.shape
            hkv = self.shape[1]
            scores = self.read_visual_logits(query, scale, backend)
            positions = self.positions.view(1, 1, -1).expand(b, hkv, -1)
            if exact_keys is not None:
                if exact_values is None or exact_positions is None:
                    raise ValueError("Exact keys require exact values and original positions")
                q = query.float().reshape(b, hkv, hq // hkv, nq, d)
                exact_scores = (q @ exact_keys.float().unsqueeze(2).transpose(-1, -2)).reshape(b, hq, nq, -1)
                scores = torch.cat((scores, exact_scores * (scale or d ** -0.5)), dim=-1)
                ep = exact_positions
                if ep.ndim == 1:
                    ep = ep.view(1, 1, -1).expand(b, hkv, -1)
                positions = torch.cat((positions, ep), dim=-1)
            if query_positions is not None:
                qp = query_positions
                if qp.ndim == 1:
                    qp = qp[None, :].expand(b, -1)
                per_q_head = positions.repeat_interleave(hq // hkv, dim=1)
                visible = per_q_head[:, :, None, :] <= qp[:, None, :, None]
                scores = scores.masked_fill(~visible, -torch.inf)
            probabilities = torch.softmax(scores, dim=-1)
            out = self.weighted_value(probabilities[..., :self.shape[-2]])
            if exact_keys is not None:
                p = probabilities[..., self.shape[-2]:].reshape(b, hkv, hq // hkv, nq, -1)
                out += (p @ exact_values.float().unsqueeze(2)).reshape(b, hq, nq, d)
            return out.to(query.dtype)
        if backend != "torch_dequant":
            raise ValueError(f"Unrecognized explicit reader backend: {backend}")
        k, v = self.materialize()
        return grouped_attention(query, k, v, self.positions, exact_keys=exact_keys,
                                 exact_values=exact_values, exact_positions=exact_positions,
                                 query_positions=query_positions, scale=scale)


def grouped_attention(query, keys, values, positions, *, exact_keys=None, exact_values=None,
                      exact_positions=None, query_positions=None, scale=None):
    """[B,Hq,Q,D] query and real [B,Hkv,N,D] owners; preserve original RoPE."""
    b, hq, nq, d = query.shape
    hkv = keys.shape[1]
    if hq % hkv:
        raise ValueError("Query heads must be an integer multiple of KV heads")
    q = query.float().reshape(b, hkv, hq // hkv, nq, d)
    k = keys.float()
    v = values.float()
    pos = positions
    if pos.ndim == 1:
        pos = pos.view(1, 1, -1).expand(b, hkv, -1)
    if exact_keys is not None:
        if exact_values is None or exact_positions is None:
            raise ValueError("Exact keys require values and original positions")
        k = torch.cat((k, exact_keys.float()), dim=-2)
        v = torch.cat((v, exact_values.float()), dim=-2)
        ep = exact_positions
        if ep.ndim == 1:
            ep = ep.view(1, 1, -1).expand(b, hkv, -1)
        pos = torch.cat((pos, ep), dim=-1)
    scores = torch.matmul(q, k.unsqueeze(2).transpose(-1, -2)) * (scale or d ** -0.5)
    if query_positions is not None:
        qp = query_positions
        if qp.ndim == 1:
            qp = qp[None, :].expand(b, -1)
        visible = pos[:, :, None, None, :] <= qp[:, None, None, :, None]
        scores = scores.masked_fill(~visible, -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.unsqueeze(2)).reshape(b, hq, nq, d).to(query.dtype)


class TurboQuantAdapter:
    def __init__(self, *, value_bits=2, key_bits=3, group_size=32, seed=42):
        if key_bits != 3 or value_bits not in (2, 4):
            raise ValueError("Audited legal configurations are K3V2 and K3V4")
        self.value_bits, self.group_size, self.seed = value_bits, group_size, seed

    def encode(self, keys: torch.Tensor, values: torch.Tensor, *, positions=None, layer_idx=0, **kwargs):
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Expected matched [B,Hkv,Nvisual,D] K/V")
        if not keys.numel() or keys.shape[-1] % self.group_size:
            raise ValueError("Nonempty visual cache and group-divisible head dimension required")
        prod, quantize_values, _, _ = _upstream()
        codec = prod(dim=keys.shape[-1], bits=3, device=keys.device, dtype=torch.float32,
                     seed=self.seed + layer_idx)
        # Upstream supports FP32; explicitly use it to avoid FP16 zero-scale underflow.
        kq = codec.quantize(keys.float())
        vq = quantize_values(values.float(), bits=self.value_bits, group_size=self.group_size)
        if positions is None:
            positions = torch.arange(keys.shape[-2], device=keys.device)
        return TurboQuantOwner(kq, vq, codec, positions.detach().to(device=keys.device, dtype=torch.int64).clone(),
                               tuple(keys.shape), keys.dtype, self.value_bits, self.group_size, layer_idx)


def create_adapter(**config):
    return TurboQuantAdapter(**config)
