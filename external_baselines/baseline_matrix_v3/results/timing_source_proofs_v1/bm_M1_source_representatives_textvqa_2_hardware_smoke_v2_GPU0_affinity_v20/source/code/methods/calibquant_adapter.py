"""CalibQuant visual-cache adapter, pinned to insuhan/calibquant@1dcfce8.

The calibration of *visual attention logits* is part of this method. Merely
dequantizing this state and passing it to SDPA does not implement CalibQuant.
Inputs are post-RoPE K, V and Q in [batch, heads, tokens, head_dim] layout.
Persistent owners contain packed bytes, channel scale/zero and a tiny code LUT.
No model weights are modified and no calibration/training data are consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any

import torch

UPSTREAM_REPO = "https://github.com/insuhan/calibquant"
UPSTREAM_COMMIT = "1dcfce862162e4a3888ef137df65b36cdc19efd0"
PAPER = "https://arxiv.org/abs/2502.14882v2"
_UPSTREAM = None


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, h, n, d = x.shape
    return x[:, :, None, :, :].expand(b, h, groups, n, d).reshape(b, h * groups, n, d)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Exact official big-endian within-byte packing, along head_dim."""
    if bits not in (1, 2, 4, 8):
        raise ValueError("bits must divide 8")
    factor = 8 // bits
    if codes.shape[-1] % factor:
        raise ValueError("CalibQuant source requires head_dim divisible by 8/bits")
    grouped = codes.to(torch.int64).reshape(*codes.shape[:-1], codes.shape[-1] // factor, factor)
    shifts = bits * torch.arange(factor - 1, -1, -1, device=codes.device)
    return (grouped << shifts).sum(-1).to(torch.uint8).contiguous()


def unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    factor = 8 // bits
    shifts = bits * torch.arange(factor - 1, -1, -1, device=packed.device)
    return ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(
        *packed.shape[:-1], packed.shape[-1] * factor).to(torch.uint8)


def _quantize(x: torch.Tensor, bits: int, clipping: float):
    # Official quant() sees [B,N,H,D] and reduces dim 1; here N is dim 2.
    lower = x.amin(dim=2, keepdim=True) * clipping
    upper = x.amax(dim=2, keepdim=True) * clipping
    scale = (upper - lower) / ((1 << bits) - 1)
    if not bool(torch.all(torch.isfinite(scale) & (scale > 0))):
        # Upstream divides by zero in this case. Do not silently change its math.
        raise ValueError("METHOD_PORT_UNVERIFIED: nonfinite/constant channel in official quantization")
    zero = -lower / scale
    codes = torch.round(torch.clamp(x / scale + zero, 0, (1 << bits) - 1))
    return pack_codes(codes, bits), scale.contiguous().clone(), zero.contiguous().clone()


def _load_upstream():
    global _UPSTREAM
    if _UPSTREAM is None:
        path = Path(__file__).resolve().parents[2] / "third_party" / "calibquant" / "calibquant.py"
        # Git normalizes source to LF. This verifies source identity independent
        # of the caller's Windows checkout newlines or transfer normalization.
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != "583e06ab50f6742267b651efa9b476b651faba0e679434b6246c819068e6a75c":
            raise RuntimeError(f"CalibQuant upstream source hash mismatch: {digest}")
        spec = importlib.util.spec_from_file_location("matrix_v3_calibquant_upstream", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _UPSTREAM = module
    return _UPSTREAM


@dataclass
class CalibQuantState:
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    k_lut: torch.Tensor
    v_lut: torch.Tensor
    bits: int
    reader_backend: str
    # These are the official released constants, frozen before evaluation.
    norm_offset_max: float = -3.0
    norm_offset_min: float = 0.0

    @property
    def visual_tokens(self) -> int:
        return self.k_packed.shape[2]

    @property
    def head_dim(self) -> int:
        return self.k_packed.shape[-1] * (8 // self.bits)

    def owner_tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in (
            "k_packed", "v_packed", "k_scale", "k_zero", "v_scale", "v_zero", "k_lut", "v_lut")}

    def byte_report(self) -> dict[str, Any]:
        owners = self.owner_tensors()
        # Count actual unique underlying storages, not logical tensor extents.
        seen = set()
        sizes = {}
        for name, tensor in owners.items():
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            sizes[name] = 0 if identity in seen else storage.nbytes()
            seen.add(identity)
        total = sum(sizes.values())
        entries = 2 * math.prod(self.k_packed.shape[:-1]) * self.head_dim
        return {"stored_visual_KV_bytes": total,
                "packed_payload_bytes": sizes["k_packed"] + sizes["v_packed"],
                "coder_metadata_bytes": sum(sizes[x] for x in ("k_scale", "k_zero", "v_scale", "v_zero")),
                "lookup_bytes": sizes["k_lut"] + sizes["v_lut"],
                "positions_bytes": 0, "encoder_constants_bytes": 0,
                "full_kv_byte_denominator": entries * self.k_scale.element_size(),
                "actual_effective_visual_bits": 8 * total / entries,
                "owners": sizes,
                "positions_contract": "state preserves all visual tokens in input order; engine owns original positions",
                "reader_backend": self.reader_backend}

    def _groups(self, query_heads: int) -> int:
        kv_heads = self.k_packed.shape[1]
        if query_heads % kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        return query_heads // kv_heads

    def attention_logits(self, q: torch.Tensor, *, calibrate: bool = True,
                         softmax_scale: float | None = None) -> torch.Tensor:
        """Return calibrated *visual-only* logits [B,QH,T,N].

        Every suffix query is calibrated independently, exactly as successive
        calls to the released single-token reader. The engine concatenates these
        with ordinary text logits in original sequence order, masks, softmaxes.
        """
        if q.ndim != 4 or q.shape[-1] != self.head_dim:
            raise ValueError("q must be [B,QH,T,D] and match packed head_dim")
        groups = self._groups(q.shape[1])
        scale = repeat_kv(self.k_scale, groups)
        zero = repeat_kv(self.k_zero, groups)
        factor = self.head_dim ** -0.5 if softmax_scale is None else softmax_scale
        if self.reader_backend == "upstream_triton":
            if not q.is_cuda or q.dtype != torch.float16:
                raise ValueError("official packed kernel requires CUDA FP16 reader inputs")
            upstream = _load_upstream()
            encoded = repeat_kv(self.k_packed, groups).contiguous()
            # Upstream kernels assume q_len=1, contiguous storage and FP16 output.
            logits = torch.cat([
                upstream.fused_kernel(self.k_lut, (q[:, :, t:t + 1] * scale).contiguous(),
                                      encoded, BITS=self.bits, DATUM_PER_ELM=8 // self.bits)
                for t in range(q.shape[2])], dim=2) * factor
        elif self.reader_backend == "torch_unfused":
            # Diagnostic/runtime variant: unpacked FP16 *integer codes* are
            # temporary and charged in peak/time; no dequantized cache retained.
            codes = repeat_kv(unpack_codes(self.k_packed, self.bits), groups).to(q.dtype)
            logits = torch.matmul(q * scale, codes.transpose(-2, -1)) * factor
        else:
            raise ValueError(f"unknown CalibQuant reader backend: {self.reader_backend}")
        bias = (scale * zero * q).sum(dim=-1, keepdim=True) * factor
        logits = logits - bias
        if calibrate and self.bits <= 2:
            # Source reduces dims (2,3) only because its T is always 1. Reducing
            # all suffix queries jointly would change the released algorithm.
            high = logits.amax(dim=-1, keepdim=True)
            low = logits.amin(dim=-1, keepdim=True)
            logits = ((logits - low) / (high - low + 1e-8)) * (
                high + self.norm_offset_max - low - self.norm_offset_min) + low + self.norm_offset_min
        return logits

    read = attention_logits

    def weighted_value(self, weights: torch.Tensor) -> torch.Tensor:
        """Multiply normalized visual probabilities [B,QH,T,N] by packed V."""
        if weights.shape[-1] != self.visual_tokens:
            raise ValueError("weights must contain precisely the visual token axis")
        groups = self._groups(weights.shape[1])
        scale = repeat_kv(self.v_scale, groups)
        zero = repeat_kv(self.v_zero, groups)
        if self.reader_backend == "upstream_triton":
            if not weights.is_cuda or weights.dtype != torch.float16:
                raise ValueError("official packed kernel requires CUDA FP16 reader inputs")
            upstream = _load_upstream()
            encoded = repeat_kv(self.v_packed, groups).contiguous()
            output = torch.cat([
                upstream.fused_kernel_wv(self.v_lut, weights[:, :, t:t + 1].contiguous(),
                                         encoded, BITS=self.bits, DATUM_PER_ELM=8 // self.bits)
                for t in range(weights.shape[2])], dim=2)
        elif self.reader_backend == "torch_unfused":
            codes = repeat_kv(unpack_codes(self.v_packed, self.bits), groups).to(weights.dtype)
            output = torch.matmul(weights, codes)
        else:
            raise ValueError(f"unknown CalibQuant reader backend: {self.reader_backend}")
        return output * scale - scale * zero * weights.sum(dim=-1, keepdim=True)

    def temporary_dequantized_debug(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagnostic oracle only. Returned matrices are never stored on self."""
        return (self.k_scale * (unpack_codes(self.k_packed, self.bits).to(self.k_scale.dtype) - self.k_zero),
                self.v_scale * (unpack_codes(self.v_packed, self.bits).to(self.v_scale.dtype) - self.v_zero))


class CalibQuantAdapter:
    method_id = "calibquant_visual_v3"
    requires_custom_attention = True

    def __init__(self, bits: int = 1, reader_backend: str = "upstream_triton"):
        if bits not in (1, 2, 4, 8):
            raise ValueError("released byte packing supports bits 1,2,4,8")
        if reader_backend not in ("upstream_triton", "torch_unfused"):
            raise ValueError(reader_backend)
        self.bits = bits
        self.reader_backend = reader_backend

    @torch.no_grad()
    def encode(self, k: torch.Tensor, v: torch.Tensor, **unused) -> CalibQuantState:
        if k.shape != v.shape or k.ndim != 4 or k.shape[2] < 2:
            raise ValueError("matching K,V [B,KVH,N,D], N>=2 required")
        if k.dtype != v.dtype or k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("matching floating dtype required")
        if not bool(torch.all(torch.isfinite(k))) or not bool(torch.all(torch.isfinite(v))):
            raise ValueError("nonfinite visual K/V")
        k_packed, k_scale, k_zero = _quantize(k, self.bits, 0.5 if self.bits == 1 else 1.0)
        v_packed, v_scale, v_zero = _quantize(v, self.bits, 0.3 if self.bits == 1 else 1.0)
        lut = torch.arange(1 << self.bits, dtype=k.dtype, device=k.device)
        return CalibQuantState(k_packed, v_packed, k_scale, k_zero, v_scale, v_zero,
                               lut.clone(), lut.clone(), self.bits, self.reader_backend)

    def manifest(self) -> dict[str, Any]:
        return {"method": self.method_id, "method_paper": PAPER,
                "upstream_repo": UPSTREAM_REPO, "upstream_commit": UPSTREAM_COMMIT,
                "port_name": "post_rope_visual_gqa_suffix_replay",
                "faithfulness_status": "P0_PENDING",
                "bits_nominal": self.bits, "reader_backend": self.reader_backend,
                "k_range_multiplier": 0.5 if self.bits == 1 else 1.0,
                "v_range_multiplier": 0.3 if self.bits == 1 else 1.0,
                "norm_offset_max": -3.0 if self.bits <= 2 else None,
                "norm_offset_min": 0.0 if self.bits <= 2 else None,
                "calibration_data": "none: published upstream constants frozen without tuning",
                "calibration_provenance_gap": "paper describes grid search but released code has no calibration runner or split manifest",
                "prefill_protocol": "image prefix captured natively, compressed, all question suffix tokens replayed",
                "upstream_protocol_difference": "released script quantizes after full prompt and only reads compressed visual KV at decode",
                "runtime_variant": ("released packed Triton QK/WV, single-query loop" if self.reader_backend == "upstream_triton"
                                    else "unfused torch packed-storage reader; temporary unpacked codes charged in memory/time"),
                "rotation": False, "sparse_outlier_owner": False,
                "text_kv": "unmodified full precision; engine owner", "weight_quantization": False}
