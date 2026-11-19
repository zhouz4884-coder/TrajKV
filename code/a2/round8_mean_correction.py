"""Post-quantization visual K mean compensation in the existing FP16 header.

Only Torch and the CPU/GPU common packed decoders are imported. The helper
consumes final J/g/sign/alpha, including every sequential FP16 residual stage;
it neither needs nor retains source visual FP16 tensors. No book, payload,
gain, residual, V, text field or reader is changed. Temporary decoding is an
encoder operation, not a retained/serving FP16 visual cache.
"""
from __future__ import annotations

import math

import torch

from round8_codec import decode_key, key_book_layout
from round6_depth_codec import decode_sign_stage


def wht128(value):
    """Normalized self-adjoint WHT; FP64 is retained for CPU references."""
    if value.shape[-1] != 128 or not value.is_floating_point():
        raise ValueError('Floating D128 vectors required')
    shape = value.shape
    result = value if value.dtype == torch.float64 else value.float()
    for width in (1,2,4,8,16,32,64):
        pair = result.reshape(*shape[:-1],128//(2*width),2,width)
        left,right = pair[...,0,:],pair[...,1,:]
        result = torch.cat((left+right,left-right),-1).reshape(shape)
    return result*(128**-.5)


@torch.no_grad()
def final_packed_key_mean(code,gain,book,lookup_k,residual_code=None,residual_alpha=None,*,chunk=256):
    """Return FP64[1,H,1,128] all-real-token mean of actual final FP16 Z.

    One head/chunk is decoded at a time. FP64 accumulation makes the summation
    reproducible for native FP16 rows; the correction itself uses FP32 WHT and
    subtraction before the final FP16 header cast. All intermediates are local.
    """
    layout = key_book_layout(book)
    if (code.dtype != torch.uint8 or code.ndim != 4 or code.shape[0] != 1
            or code.shape[-1] != layout['index_bytes'] or gain.dtype != torch.float16
            or tuple(gain.shape) != (*code.shape[:-1],1) or book.dtype != torch.float16 or chunk < 1):
        raise ValueError('Actual batch1 packed code, FP16 gain/book and positive chunk required')
    heads,tokens = code.shape[1:3]
    if tokens <= 0 or tuple(lookup_k.shape) != (heads,2) or lookup_k.dtype != torch.int32:
        raise ValueError('All real visual tokens and [H,2] int32 K lookup required')
    if any(t.device != code.device for t in (gain,book,lookup_k)):
        raise ValueError('Packed fields must share a device')
    mean = torch.zeros((1,heads,1,128),device=code.device,dtype=torch.float64)
    for head in range(heads):
        slots = [int(lookup_k[head,stage]) for stage in range(2)]
        if slots[1] >= 0 and slots[0] < 0:
            raise ValueError('Second K stage requires its first stage')
        for slot in slots:
            if slot < -1:
                raise ValueError('Invalid negative residual slot')
            if slot >= 0:
                if (residual_code is None or residual_alpha is None or residual_code.dtype != torch.int32
                    or residual_alpha.dtype != torch.float16 or residual_code.ndim != 3
                    or tuple(residual_code.shape[1:]) != (tokens,4)
                    or tuple(residual_alpha.shape) != tuple(residual_code.shape[:2])
                    or slot >= residual_code.shape[0]
                    or residual_code.device != code.device or residual_alpha.device != code.device):
                    raise ValueError('Missing/invalid actual compact sign-stage pool')
        for start in range(0,tokens,chunk):
            stop = min(start+chunk,tokens)
            final = decode_key(code[0,head,start:stop],gain[0,head,start:stop],book)
            for slot in slots:
                if slot >= 0:
                    final = decode_sign_stage(final,residual_code[slot,start:stop],residual_alpha[slot,start:stop])
            if not bool(torch.isfinite(final).all()):
                raise ValueError('Nonfinite final packed K reconstruction')
            mean[0,head,0].add_(final.double().sum(0))
            del final
    return mean/tokens


@torch.no_grad()
def compensated_mean_header(old_mean,mean_final_z):
    """Q16(mu0-H32(mean(Zhat))) plus a finite-precision residual certificate.

    The reference target is the existing stored mu0, not an unavailable exact
    source mean. The certificate separates final FP16 header rounding from
    FP32 mean conversion/WHT/subtraction error against the FP64 H reference.
    """
    if (old_mean.dtype != torch.float16 or old_mean.ndim != 4 or old_mean.shape[0] != 1
        or old_mean.shape[-2:] != (1,128) or mean_final_z.shape != old_mean.shape
        or mean_final_z.device != old_mean.device
        or not bool(torch.isfinite(old_mean).all()) or not bool(torch.isfinite(mean_final_z).all())):
        raise ValueError('Finite stored FP16[1,H,1,128] mean and matching decoded mean required')
    mean64 = mean_final_z.double()
    inverse32 = wht128(mean64.float())
    pre32 = old_mean.float()-inverse32
    candidate = pre32.half()
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError('Mean compensation overflows FP16; existing header has not been written')
    ideal = old_mean.double()-wht128(mean64)
    rounding = (candidate.double()-pre32.double()).abs()
    computation = (pre32.double()-ideal).abs()
    # Unit roundoff for normal FP16 plus half a subnormal step. No relative
    # bound is assumed near zero or cancellation.
    half_bound = pre32.double().abs()*2**-11 + 2**-25
    residual = candidate.double()-ideal
    if not bool((rounding <= half_bound+1e-12).all()) or not bool((residual.abs() <= half_bound+computation+1e-12).all()):
        raise RuntimeError('FP16 mean-header residual bound failed')
    shift = candidate.double()-old_mean.double()
    old_residual = wht128(mean64)
    heads = old_mean.shape[1]
    rows = []
    for head in range(heads):
        sl = (0,head,0)
        rows.append(dict(head=head,mean_Z_l2=float(mean64[sl].norm()),
            stored_target_residual_l2_before=float(old_residual[sl].norm()),
            stored_target_residual_l2_after=float(residual[sl].norm()),
            header_shift_l2=float(shift[sl].norm()),header_shift_max_abs=float(shift[sl].abs().max()),
            FP16_rounding_max_abs=float(rounding[sl].max()),FP32_computation_max_abs=float(computation[sl].max()),
            residual_max_abs=float(residual[sl].abs().max()),residual_component_bound_max=float((half_bound[sl]+computation[sl]).max())))
    stats = dict(heads=heads,header_bytes=old_mean.numel()*old_mean.element_size(),added_persistent_tensor_bytes=0,
        target='existing stored FP16 source-mean header mu0; exact source mean is not available/retained',
        rule='FP16(mu0 - H_FP32(mean_all_real_visual(final sequential FP16 decoded Z)))',
        accumulation='FP64 sum of final FP16 decoded rows, divided by actual real Nv',
        residual_bound='abs(mean_reconstructed_K-mu0) <= 2^-11*abs(pre32)+2^-25+abs(pre32-(mu0-H_FP64(mean_Z)))',
        source_mean_bound='add abs(mu0-exact_source_mean) to bound bias relative to the true source mean',per_head=rows)
    return candidate.contiguous(),stats


@torch.no_grad()
def correct_state_mean_inplace(state,book,*,chunk=256):
    """After parent encoding, overwrite only the already-owned mean tensor."""
    if hasattr(state,'visual_k') or hasattr(state,'visual_v'):
        raise ValueError('Apply after original visual source tensors have been released')
    code = state.direction_code
    if int(state.visual_pos.numel()) != code.shape[-2]:
        raise ValueError('Compute the mean over every real visual token, without padded/missing rows')
    lookup = state.depth_residual_lookup
    if tuple(lookup.shape) != (2,code.shape[1],2):
        raise ValueError('Actual [2,H,2] compact stage lookup required')
    original_pointer = state.direction_mean.data_ptr()
    mean = final_packed_key_mean(code,state.direction_gain,book,lookup[0],
        getattr(state,'depth_residual_code',None),getattr(state,'depth_residual_alpha',None),chunk=chunk)
    candidate,stats = compensated_mean_header(state.direction_mean,mean)
    state.direction_mean.copy_(candidate)
    if state.direction_mean.data_ptr() != original_pointer:
        raise RuntimeError('Existing header storage was replaced')
    stats.update(visual_tokens=code.shape[-2],full_visual_mean=True,source_FP16_used=False,
        temporary_decode_scope='encoder-only, one head/chunk; no final decoded rows retained',mean_storage_pointer_preserved=True)
    return stats


def mean_corrected_bank_class(base_class):
    """Minimal integration hook; caller supplies its existing Round8Bank class.

    This module remains CPU-importable. The existing GPU bank/reader is only
    imported by a future runner, and no frozen source file is modified here.
    """
    class MeanCorrectedBank(base_class):
        def _initialize(self,*args,**kwargs):
            enabled = type(self).settings.get('visual_mean_correction',False)
            if not isinstance(enabled,bool):
                raise ValueError('visual_mean_correction must be an explicit boolean')
            result = super()._initialize(*args,**kwargs)
            before = self.owned()
            self.mean_correction_statistics = []
            if enabled:
                for layer,state in enumerate(self.states):
                    self.mean_correction_statistics.append(dict(layer=layer,**correct_state_mean_inplace(state,self.book)))
            after = self.owned()
            if before != after:
                raise RuntimeError('Mean-only intervention changed actual owned bytes/fields')
            self.mean_correction_enabled = enabled
            return result

        def result_metadata(self):
            result = super().result_metadata()
            result.update(visual_mean_correction=self.mean_correction_enabled,
                mean_correction_added_persistent_tensor_bytes=0,mean_correction_source_FP16_used=False,
                mean_correction_statistics=self.mean_correction_statistics,
                mean_correction_scope='one encoder-only correction of existing mean slots; J/g/sign/alpha/V/text/book/reader unchanged',
                mean_correction_serving_FP16_materialization=False)
            if self.mean_correction_enabled:
                result['depth_residual_K_reference']='Original centered-WHT residual fields unchanged; stored mean corrected after final hard decode'
            return result
    MeanCorrectedBank.__name__ = 'MeanCorrectedRound8Bank'
    return MeanCorrectedBank
