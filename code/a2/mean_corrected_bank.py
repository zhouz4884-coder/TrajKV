"""Remove the final PQ reconstruction's token-mean bias in the existing slot.

Codes, final row gains and V are identical to MetadataDirectionBank. Only the
stored key mean is replaced, after preserving the frozen gain-mode semantics.
FP32 means are actual additional cache bytes; no reference cache is retained.
"""
import hashlib
import json
from pathlib import Path

import torch

from affine_rate_bank import fold
from fast_wht import wht
from masked_bank import A2Bank
from metadata_direction_bank import MetadataDirectionBank
from quantizers import binary_centroids


class MeanCorrectedBank(MetadataDirectionBank):
    def _initialize(self, model, cache, visual, method, legacy_scripts,
                    kivi_source, keep_reference=False):
        # This deliberately repeats the frozen encoder's arithmetic so the true
        # source mean remains available before the original visual K is freed.
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts,
                           kivi_source, False)
        cfg = type(self).settings
        if cfg.get('functional_v_factor_path'):
            raise ValueError('First mean-correction experiment uses only frozen base metadata')
        self.mean_storage_dtype = cfg.get('mean_dtype', 'float16')
        dtype = {'float16': torch.float16, 'float32': torch.float32}.get(self.mean_storage_dtype)
        if dtype is None:
            raise ValueError('mean_dtype must be float16 or float32')
        device = self.states[0].visual_k.device
        nv = visual.numel()
        book_path = Path(cfg['direction_book_path'])
        self.book = torch.tensor(json.loads(book_path.read_text())['centers'],
                                 device=device, dtype=torch.float16).contiguous()
        assert self.book.shape == (256, 4)
        self.decoder_books = None
        self.decoder_hash = None
        self.headwise_budget_ok = False
        self.encoder_parameter_bytes = 0
        self.vbits = 1
        self.v_channel = True
        self.vgroup = nv if cfg.get('v_group_size') == 'all' else 128
        if cfg.get('bits_v', 1) != 1 or cfg.get('v_layout', 'channel') != 'channel':
            raise ValueError('Mean correction requires the frozen channel-V1 format')
        self.metadata_hash = None
        params = {}
        self.key_gain_mode = 'none'
        if cfg.get('metadata_factor_path'):
            path = Path(cfg['metadata_factor_path'])
            data = json.loads(path.read_text())
            self.metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if data['encoder_book_sha256'] != hashlib.sha256(book_path.read_bytes()).hexdigest():
                raise ValueError('Wrong encoder book for factors')
            if data['value_group_size'] != cfg.get('v_group_size', 128):
                raise ValueError('Calibration value-group mismatch')
            self.key_gain_mode = data['key_gain_mode']
            assert self.key_gain_mode in ('none', 'centered', 'whole')
            for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets'):
                if data.get(key) is None:
                    continue
                parameter = torch.tensor(data[key], device=device, dtype=torch.float32)
                assert parameter.shape == (len(self.states), self.states[0].visual_k.shape[1])
                assert bool(torch.isfinite(parameter).all())
                if 'factors' in key:
                    assert bool((parameter > 0).all())
                params[key] = parameter
            self.encoder_parameter_bytes = sum(t.numel()*t.element_size() for t in params.values())
            assert self.encoder_parameter_bytes == 4*data['parameter_count']
        self.mean_correction_statistics = []
        self.mean_extra_bytes = 0
        for i, state in enumerate(self.states):
            source = state.visual_k.float()
            true_mean = source.mean(-2, keepdim=True)
            initial_mean = true_mean.half().contiguous()
            state.direction_mean = initial_mean
            rotated = wht(source-state.direction_mean.float())
            norm = rotated.norm(dim=-1, keepdim=True)
            initial = (norm/(128**.5)).clamp_min(1e-20)
            parts = (rotated/initial).reshape(-1, 4)
            book = self.book.float()
            indices = []
            for start in range(0, len(parts), 65536):
                part = parts[start:start+65536]
                distance = part.square().sum(-1, keepdim=True) + book.square().sum(-1)[None] - 2*(part@book.T)
                indices.append(distance.argmin(-1).byte())
            state.direction_code = torch.cat(indices).reshape(*rotated.shape[:-1], 32).contiguous()
            template = book[state.direction_code.long()].reshape(rotated.shape)
            state.direction_gain = (norm/template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half().contiguous()
            factor = None
            if 'key_head_factors' in params:
                factor = params['key_head_factors'][i][None, :, None, None]
                state.direction_gain = (state.direction_gain.float()*factor).half().contiguous()
                if self.key_gain_mode == 'whole':
                    state.direction_mean = (state.direction_mean.float()*factor).half().contiguous()

            # Match the exact F16 codebook-times-final-gain rounding in _read.
            # H is linear, so the token mean can be transformed just once.
            packed_rotated = (template*state.direction_gain.float()).half()
            residual_mean = wht(packed_rotated.float().mean(-2, keepdim=True))
            target_mean = true_mean
            if factor is not None:
                if self.key_gain_mode == 'centered':
                    target_mean = initial_mean.float() + factor*(true_mean-initial_mean.float())
                elif self.key_gain_mode == 'whole':
                    target_mean = factor*true_mean
            before_error = state.direction_mean.float()+residual_mean-target_mean
            corrected = (target_mean-residual_mean).to(dtype=dtype).contiguous()
            after_error = corrected.float()+residual_mean-target_mean
            self.mean_extra_bytes += corrected.numel()*(corrected.element_size()-2)
            self.mean_correction_statistics.append(dict(layer=i,
                original_mean_slot_mse=float((state.direction_mean.float()-target_mean).square().mean()),
                reconstructed_mean_error_mse_before=float(before_error.square().mean()),
                reconstructed_mean_error_mse_after=float(after_error.square().mean()),
                mean_slot_change_mse=float((corrected.float()-state.direction_mean.float()).square().mean()),
                reconstructed_residual_mean_mse=float(residual_mean.square().mean())))
            state.direction_mean = corrected
            del state.visual_k
            record = binary_centroids(state.visual_v.transpose(2, 3).contiguous(), group_size=self.vgroup)
            if 'value_head_factors' in params:
                record = fold(record, 1, params['value_head_factors'][i][None, :, None, None],
                              params['value_center_offsets'][i][None, :, None, None])
            for key, tensor in zip(('code', 'scale', 'min'), record):
                setattr(state, 'v_'+key, tensor)
            del state.visual_v
        self.method = method

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend='packed global PQ with gain-preserving reconstruction mean correction',
            key_mean_corrected=True, key_mean_storage_dtype=self.mean_storage_dtype,
            key_mean_extra_cache_bytes=self.mean_extra_bytes,
            key_mean_extra_effective_bits=16*self.mean_extra_bytes/self.visual_full_bytes,
            mean_correction_statistics=self.mean_correction_statistics,
            mean_correction_preserves='code assignments, final row gains, V format and frozen V factors',
            mean_correction_target='none:m; centered:mu0+g*(m-mu0); whole:g*m; mu0=FP16(m)',
            mean_correction_scope='per-image encoder operation using reconstructed token mean in normalized WHT; no retained FP16 source/reference',
            calibration_parameter_scope='same frozen encoder factors as base; corrected key mean charged at its actual FP16/FP32 dtype')
        return result
