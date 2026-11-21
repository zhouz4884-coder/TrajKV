"""Common-FP16 CalibQuant numeric port v2; no new quantization algorithm.

Reuse the audited FP32 temporary affine/calibration port and source kernels.
Only exact-text QK additionally uses FP32 operands/output before joint softmax.
Visual codes, FP16 metadata/weights/text KV and source PV boundaries persist.
"""
from __future__ import annotations
import torch
from .calibquant_adapter import repeat_kv
from .calibquant_fp32_affine_adapter import CalibQuantFP32AffineAdapter, CalibQuantFP32AffineState


class CalibQuantFP32AffineTextState(CalibQuantFP32AffineState):
    def attend(self, q, *, exact_keys, exact_values, exact_positions, query_positions, scale):
        if q.ndim != 4 or q.shape[0] != 1 or q.dtype != torch.float16 or q.shape[-1] != self.head_dim:
            raise ValueError("UNSUPPORTED_CACHE_PROTOCOL: common FP16 batch-one Q required")
        if exact_keys.shape != exact_values.shape or exact_keys.ndim != 4 or exact_keys.shape[:2] != self.k_packed.shape[:2] or exact_keys.shape[-1] != self.head_dim:
            raise ValueError("INPUT_MISMATCH: native exact-text KV geometry")
        if exact_keys.dtype != q.dtype or exact_values.dtype != q.dtype or exact_keys.device != q.device or exact_values.device != q.device:
            raise ValueError("INPUT_MISMATCH: native exact-text KV dtype/device")
        if exact_positions.ndim != 1 or query_positions.ndim != 1 or exact_positions.numel() != exact_keys.shape[-2] or query_positions.numel() != q.shape[-2] or not query_positions.numel():
            raise ValueError("INPUT_MISMATCH: original text/query position axes")
        if exact_positions.dtype != torch.int64 or query_positions.dtype != torch.int64 or exact_positions.device != q.device or query_positions.device != q.device:
            raise ValueError("INPUT_MISMATCH: native position dtype/device")
        # HybridVisualCache owns a complete immutable native prefix and appends
        # exact text only. Validate that its complete-prefix coverage identity
        # holds rather than retaining a second positions or metadata tensor.
        first = int(query_positions[0])
        if int((exact_positions < first).sum()) + self.visual_tokens != first:
            raise ValueError("UNSUPPORTED_CACHE_PROTOCOL: visual prefix is not wholly before this query")
        if bool((exact_positions < 0).any()) or not bool((exact_positions[1:] > exact_positions[:-1]).all()):
            raise ValueError("INPUT_MISMATCH: exact-text positions must be nonnegative, sorted and unique")
        if not torch.equal(query_positions, torch.arange(first, first + q.shape[-2], device=q.device)) or not torch.equal(exact_positions[-q.shape[-2]:], query_positions):
            raise ValueError("UNSUPPORTED_CACHE_PROTOCOL: current queries must be the appended native text tail")
        groups = self._groups(q.shape[1])
        # Existing official packed QK + FP32 affine/calibration, unchanged.
        visual_logits = self.read(q, softmax_scale=scale)
        text_k = repeat_kv(exact_keys, groups)
        text_logits = (q.float() @ text_k.float().transpose(-1, -2)) / (1.0 / scale)
        text_allowed = exact_positions[None, :] <= query_positions[:, None]
        text_logits = text_logits.masked_fill(~text_allowed[None, None], -torch.inf)
        probabilities = torch.softmax(torch.cat((visual_logits, text_logits), dim=-1), dim=-1).to(q.dtype)
        count = self.visual_tokens
        # Retain the audited affine port's source PV boundary: packed WV output
        # hard-FP16; affine temporaries FP32; visual return Half. Exact-text PV
        # and the final sum remain Half, matching the existing common bridge.
        visual_output = self.weighted_value(probabilities[..., :count])
        text_output = probabilities[..., count:] @ repeat_kv(exact_values, groups)
        return visual_output + text_output


class CalibQuantFP32AffineTextAdapter(CalibQuantFP32AffineAdapter):
    method_id = "calibquant_visual_fp32_affine_text_v2"

    @torch.no_grad()
    def encode(self, k, v, **kwargs):
        old = super().encode(k, v, **kwargs)
        return CalibQuantFP32AffineTextState(**vars(old))

    def manifest(self):
        result = super().manifest()
        result.update(method=self.method_id, port_name="common_FP16_source_packed_FP32_affine_and_exact_text_QK_v2",
                      numerical_variant="Existing FP32 temporary visual affine/calibration plus FP32 exact-text QK before joint FP32 softmax",
                      position_contract="Hybrid complete native prefix, all visual keys precede current queries; coverage identity and exact-text causal positions checked per read",
                      persistent_owner_unchanged=True, adds_persistent_tensors=False, fitted_parameters=False,
                      source_PV_boundary="Hard-FP16 packed WV; FP32 visual affine then Half return; native exact-text PV Half and final joint output Half",
                      temporary_precision_cost="FP32 exact-text Q/K/logits and joint-softmax intermediates are transient; actual CUDA peak/time must include them",
                      not_author_BF16_bitwise_reproduction=True)
        return result


def create_adapter(**kwargs):
    return CalibQuantFP32AffineTextAdapter(**kwargs)
