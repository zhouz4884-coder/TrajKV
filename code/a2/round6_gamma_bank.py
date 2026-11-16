"""Packed mixed residual reader with encoder-only F16 gamma folded into alpha.

Identity, zero, and learned controls all own the same [2,L,Hkv] F16 encoder
table and pay its complete byte cost before selecting the incumbent prefix.
No source visual FP16 K/V or extra serving-side decode tensor is retained.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from mixed_residual_bank import MixedResidualBank
from round6_student_codec import mask_policy_signature


class GammaResidualBank(MixedResidualBank):
    def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
        original_settings = type(self).settings
        cfg = dict(original_settings)
        nl = model.config.get_text_config().num_hidden_layers
        nh = model.config.get_text_config().num_key_value_heads
        gamma_bytes = 2 * nl * nh * 2
        self.gamma_control = cfg.get('gamma_control', 'learned')
        self.gamma_factor_hash = None
        if self.gamma_control not in ('identity', 'zero', 'learned'):
            raise ValueError('gamma_control must be identity, zero or learned')
        if self.gamma_control == 'learned':
            path = Path(cfg['gamma_factor_path'])
            data = json.loads(path.read_text())
            if data.get('schema') != 'trajkv_round6_gamma_v1':
                raise ValueError('Unrecognized gamma factor format')
            self.gamma_factor_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            gamma = torch.tensor(data['gamma_f16'], dtype=torch.float16)
            if data.get('encoder_book_sha256') != hashlib.sha256(Path(cfg['direction_book_path']).read_bytes()).hexdigest():
                raise ValueError('Gamma/book mismatch')
            if data.get('base_metadata_sha256') != hashlib.sha256(Path(cfg['metadata_factor_path']).read_bytes()).hexdigest():
                raise ValueError('Gamma/base mismatch')
            if data.get('incumbent_mask_policy_sha256') != mask_policy_signature(cfg):
                raise ValueError('Gamma/incumbent allocation policy mismatch')
        else:
            gamma = torch.full((2, nl, nh), 1. if self.gamma_control == 'identity' else 0., dtype=torch.float16)
        if tuple(gamma.shape) != (2, nl, nh) or not bool(torch.isfinite(gamma).all()) or bool((gamma < 0).any()):
            raise ValueError('Finite nonnegative F16 gamma[2,L,Hkv] required')
        # Reserve integer bytes first.  A nextafter offset prevents floating point
        # conversion back into bits from losing an additional byte by floor().
        full_bytes = nl * nh * visual.numel() * 128 * 4
        original_budget = float(cfg.get('budget_bits', 1.8))
        allowed = math.floor(original_budget * full_bytes / 16)
        reduced_bytes = allowed - gamma_bytes
        if reduced_bytes <= 0:
            raise ValueError('Gamma constants exceed complete owner budget')
        cfg['budget_bits'] = math.nextafter(reduced_bytes * 16 / full_bytes, math.inf)
        # Mixed initialization reads settings through type(self). Restore it even
        # on failure; all encoded tensors remain exclusively owned by this bank.
        type(self).settings = cfg
        try:
            super()._initialize(model, cache, visual, method, legacy_scripts, kivi_source, False)
        finally:
            type(self).settings = original_settings
        self.gamma_encoder_table = gamma.contiguous()  # encoder-only CPU owner
        self.gamma_encoder_constant_bytes = gamma_bytes
        self.encoder_parameter_bytes += gamma_bytes
        self.predicted_budget_bytes += gamma_bytes
        self.residual_budget_bits = original_budget
        for layer, state in enumerate(self.states):
            if not hasattr(state, 'mixed_residual_alpha'):
                continue
            selected = [(axis, head) for axis, index, head in self.selected_residual_units if index == layer]
            multipliers = torch.tensor([float(gamma[0 if axis == 'k' else 1, layer, head])
                for axis, head in selected], device=state.mixed_residual_alpha.device, dtype=torch.float16)
            state.mixed_residual_alpha.copy_((state.mixed_residual_alpha.float() * multipliers[:, None].float()).half())
        owned = self.owned()
        actual = (owned['visual_base'] + owned['visual_residual'] + owned['selector_metadata']
                  + owned['visual_position'] + self.encoder_parameter_bytes)
        if actual != self.predicted_budget_bytes or actual > allowed:
            raise RuntimeError('Gamma complete packed owner mismatch/over-budget')
        self.actual_budget_bits = 16 * actual / self.visual_full_bytes

    def result_metadata(self):
        result = super().result_metadata()
        result.update(gamma_control=self.gamma_control, gamma_factor_sha256=self.gamma_factor_hash,
            gamma_encoder_dtype='float16', gamma_encoder_constant_bytes=self.gamma_encoder_constant_bytes,
            gamma_folded_into_existing_F16_alpha=True, gamma_no_extra_serving_tensor=True,
            gamma_identity_zero_learned_same_constant_shape=True,
            gamma_mask_control_scope='same 448B encoder charge for all controls; gamma-matched mask may be shorter than original incumbent',
            gamma_encoder_table_sha256=hashlib.sha256(self.gamma_encoder_table.numpy().tobytes()).hexdigest())
        return result
