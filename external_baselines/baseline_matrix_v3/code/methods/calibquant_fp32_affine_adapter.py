"""Explicit FP32 affine/calibration numeric port for the common FP16 model.

The author BF16 model plus hard-FP16 kernel outputs naturally promotes several
visual intermediates to FP32. This separately named port uses consistent FP32
temporary affine arithmetic; it does NOT reproduce BF16 input rounding bitwise.
The source encoder, packed bytes, FP16 metadata, clipping and -3 map are intact.
"""
from __future__ import annotations
import torch

from .calibquant_adapter import CalibQuantAdapter,CalibQuantState,repeat_kv,unpack_codes,_load_upstream


class CalibQuantFP32AffineState(CalibQuantState):
    def _raw_qk(self,q):
        if q.ndim!=4 or q.shape[-1]!=self.head_dim or q.dtype!=torch.float16:
            raise ValueError('FP32-affine common-model port requires Q [B,H,T,D] in FP16')
        groups=self._groups(q.shape[1]);scale=repeat_kv(self.k_scale,groups)
        # Keep native FP16 q*scale and original hard-FP16 kernel boundary.
        if self.reader_backend=='upstream_triton':
            if not q.is_cuda:raise ValueError('Original packed reader requires CUDA')
            upstream=_load_upstream();encoded=repeat_kv(self.k_packed,groups).contiguous()
            return torch.cat([upstream.fused_kernel(self.k_lut,(q[:,:,t:t+1]*scale).contiguous(),encoded,
                          BITS=self.bits,DATUM_PER_ELM=8//self.bits) for t in range(q.shape[-2])],2)
        if self.reader_backend=='torch_unfused':
            codes=repeat_kv(unpack_codes(self.k_packed,self.bits),groups).to(q.dtype)
            return (q*scale)@codes.transpose(-1,-2)
        raise ValueError('Unknown explicit reader backend: '+self.reader_backend)

    def attention_logits(self,q,*,calibrate=True,softmax_scale=None):
        groups=self._groups(q.shape[1]);scale=repeat_kv(self.k_scale,groups).float();zero=repeat_kv(self.k_zero,groups).float()
        factor=self.head_dim**-.5 if softmax_scale is None else float(softmax_scale)
        raw=self._raw_qk(q)
        bias=(scale*zero*q.float()).sum(-1,keepdim=True)*factor
        logits=raw.float()*factor-bias
        if calibrate and self.bits<=2:
            low,high=logits.amin(-1,keepdim=True),logits.amax(-1,keepdim=True)
            logits=((logits-low)/(high-low+1e-8))*(high+self.norm_offset_max-low-self.norm_offset_min)+low+self.norm_offset_min
        return logits

    read=attention_logits

    def _raw_weighted_codes(self,weights):
        if weights.ndim!=4 or weights.dtype!=torch.float16 or weights.shape[-1]!=self.visual_tokens:
            raise ValueError('Common-model packed WV requires FP16 probabilities with the visual axis')
        groups=self._groups(weights.shape[1])
        if self.reader_backend=='upstream_triton':
            if not weights.is_cuda:raise ValueError('Original packed reader requires CUDA')
            upstream=_load_upstream();encoded=repeat_kv(self.v_packed,groups).contiguous()
            return torch.cat([upstream.fused_kernel_wv(self.v_lut,weights[:,:,t:t+1].contiguous(),encoded,
                           BITS=self.bits,DATUM_PER_ELM=8//self.bits) for t in range(weights.shape[-2])],2)
        if self.reader_backend=='torch_unfused':
            codes=repeat_kv(unpack_codes(self.v_packed,self.bits),groups).to(weights.dtype)
            return weights@codes
        raise ValueError('Unknown explicit reader backend: '+self.reader_backend)

    def weighted_value(self,weights):
        groups=self._groups(weights.shape[1]);scale=repeat_kv(self.v_scale,groups).float();zero=repeat_kv(self.v_zero,groups).float()
        raw=self._raw_weighted_codes(weights)
        output=raw.float()*scale-(scale*zero)*weights.float().sum(-1,keepdim=True)
        # Existing bridge adds exact-text output at this common FP16 boundary.
        # The author's BF16 source casts after the joint visual+text output;
        # this prototype explicitly retains the existing bridge's boundary.
        return output.to(weights.dtype)


class CalibQuantFP32AffineAdapter(CalibQuantAdapter):
    method_id='calibquant_visual_fp32_affine_v1'

    @torch.no_grad()
    def encode(self,k,v,**kwargs):
        if k.dtype!=torch.float16 or v.dtype!=torch.float16:
            raise ValueError('This common-model numerical port freezes FP16 encoder/metadata inputs')
        original=super().encode(k,v,**kwargs)
        return CalibQuantFP32AffineState(**vars(original))

    def manifest(self):
        result=super().manifest()
        result.update(method=self.method_id,port_name='visual_packed_source_kernel_FP32_affine_calibration_common_FP16_v1',
            numerical_variant='Explicit FP32 temporary bias products/reduction, raw-QK rescale/subtraction/calibration, raw-WV affine',
            original_source_default='BF16 model plus hard-FP16 kernels promotes visual score/affine to FP32; source V bias subproducts remain BF16',
            not_author_BF16_bitwise_reproduction=True,
            unchanged=['source quantizer','integer codes','packed format','metadata dtype FP16','source QK/WV kernels','K/V clip multipliers','norm offsets','FP16 weights and text KV'],
            extra_temporary_precision_only=True,adds_persistent_tensors=False,fitted_parameters=False,
            remaining_bridge_boundary='Visual weighted result cast FP16 before existing exact-text FP16 addition; author defaults cast joint output BF16')
        return result


def create_adapter(**kwargs):return CalibQuantFP32AffineAdapter(**kwargs)
