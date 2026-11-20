"""CalibQuant visual port v1: FP32 bias SUM/DIV only, then FP16 logits.

The author inference script defaults to BF16. Under the matrix's fixed FP16
weights/KV, finite per-channel bias products can overflow in the sum's FP16
output before division by sqrt(D). This separately named port preserves the
quantizer, packed format, FP16 products, official kernels and calibration.
"""
from __future__ import annotations
import torch
from .calibquant_adapter import (CalibQuantAdapter, CalibQuantState, repeat_kv,
                                 unpack_codes, _load_upstream)


class CalibQuantFP32BiasState(CalibQuantState):
    def attention_logits(self, q: torch.Tensor, *, calibrate: bool = True,
                         softmax_scale: float | None = None) -> torch.Tensor:
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
            logits = torch.cat([
                upstream.fused_kernel(self.k_lut, (q[:, :, t:t + 1] * scale).contiguous(),
                                      encoded, BITS=self.bits, DATUM_PER_ELM=8 // self.bits)
                for t in range(q.shape[2])], dim=2) * factor
        elif self.reader_backend == "torch_unfused":
            codes = repeat_kv(unpack_codes(self.k_packed, self.bits), groups).to(q.dtype)
            logits = torch.matmul(q * scale, codes.transpose(-2, -1)) * factor
        else:
            raise ValueError(f"unknown CalibQuant reader backend: {self.reader_backend}")
        # Only precision change: multiplication remains the original FP16
        # expression. Keep sum AND division FP32, then restore original logits
        # dtype before subtraction/calibration. No clamp or nan replacement.
        bias_products = scale * zero * q
        bias = (bias_products.sum(dim=-1, keepdim=True, dtype=torch.float32) / (1.0 / factor)).to(q.dtype)
        logits = logits - bias
        if calibrate and self.bits <= 2:
            high = logits.amax(dim=-1, keepdim=True)
            low = logits.amin(dim=-1, keepdim=True)
            logits = ((logits - low) / (high - low + 1e-8)) * (
                high + self.norm_offset_max - low - self.norm_offset_min) + low + self.norm_offset_min
        return logits

    read = attention_logits


class CalibQuantFP32BiasAdapter(CalibQuantAdapter):
    method_id = "calibquant_visual_fp32_bias_reduction_v1"

    @torch.no_grad()
    def encode(self, k, v, **kwargs):
        original = super().encode(k, v, **kwargs)
        return CalibQuantFP32BiasState(**vars(original))

    def manifest(self):
        result = super().manifest()
        result.update(method=self.method_id, port_name="post_rope_visual_gqa_suffix_replay_fp32_bias_reduction_v1",
            precision_change="FP16 channel-bias products unchanged; sum/div FP32; cast to original FP16 before subtract/calibrate",
            unchanged=["clip multipliers", "integer codes", "packed uint8 format", "scale/zero dtype", "persistent owner", "official QK/WV kernels", "calibration map", "model weights FP16"],
            published_infer_default_dtype="bfloat16", matrix_weights_and_kv_dtype="float16",
            motivation_evidence="audit/calibquant_M1_model_acceptance_v1.json",
            source_published_fp16_branch="retained separately; C1 passed, C2 finite-product bias-sum overflow preserved",
            fitted_parameters=False, adds_persistent_tensors=False,
            runtime_variant="official packed QK/WV plus explicitly declared FP32 bias reduction/division")
        return result
