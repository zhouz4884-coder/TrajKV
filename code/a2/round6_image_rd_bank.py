"""Per-image native-D(depth) x frozen common-CE-w allocation prototype.

Only current visual source KV and frozen sensitivities enter allocation; no
question/query/answer information is inspected. A preliminary real encoding
pass measures all depths, then the unchanged packed bank encodes the chosen
legal prefix again. This extra encoding work is intentionally measured.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from fast_wht import wht, torch_wht
from round6_depth_bank import DepthResidualBank
from round6_depth_codec import encode_sign_depth, decode_sign_stage, validate_stages
from round6_fit_depth_rd import greedy_stage_order
from round6_image_rd_runner import reserve_encoder_bytes
from round6_student_codec import encode_layer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ImageRDResidualBank(DepthResidualBank):
    def _image_order(self, model, cache, visual, settings):
        nl = len(cache.layers)
        first = cache.layers[0].keys
        nh, dimension, nv = first.shape[1], first.shape[-1], int(visual.numel())
        if dimension != 128 or nv % 32 or first.dtype != torch.float16:
            raise ValueError('Image RD requires FP16 source KV, D128 and complete visual sign words')
        window = settings.get('v_group_size', 'all')
        value_group = nv if window == 'all' else int(window)
        if value_group not in (128, nv) or nv % value_group:
            raise ValueError('Complete g128 or whole-image V groups required')
        factor_path, book_path = Path(settings['metadata_factor_path']), Path(settings['direction_book_path'])
        factors = json.loads(factor_path.read_text())
        if factors['encoder_book_sha256'] != sha(book_path) or factors['value_group_size'] != window:
            raise ValueError('Frozen encoder/base/window mismatch')
        weight_path = Path(settings['image_rd_sensitivity_path'])
        artifact = json.loads(weight_path.read_text())
        if artifact.get('encoder_book_sha256') != sha(book_path) or artifact.get('fixed_metadata_sha256') != sha(factor_path):
            raise ValueError('Frozen CE sensitivities have a different encoder/base source')
        if artifact.get('K_V_gradient_sensitivities_separately_normalized') is not False:
            raise ValueError('Keep the original common K/V CE sensitivity scale')
        self.image_rd_sensitivity = torch.tensor([artifact['per_axis'][axis]['sensitivity_w'] for axis in ('k', 'v')],
            dtype=torch.float32, device='cpu').contiguous()
        if tuple(self.image_rd_sensitivity.shape) != (2, nl, nh) or not bool(torch.isfinite(self.image_rd_sensitivity).all()) or bool((self.image_rd_sensitivity < 0).any()):
            raise ValueError('Invalid CE sensitivity shape/value')
        weights = {axis: self.image_rd_sensitivity[index].tolist() for index, axis in enumerate(('k', 'v'))}
        # The actual FP32 table, rather than hidden JSON/Python FP64 precision,
        # is the sole sensitivity constant used by the new allocation.
        self.image_rd_sensitivity_hash = sha(weight_path)
        self.image_rd_sensitivity_table_hash = hashlib.sha256(self.image_rd_sensitivity.numpy().tobytes()).hexdigest()
        book = torch.tensor(json.loads(book_path.read_text())['centers'], device=first.device, dtype=torch.float16)
        if tuple(book.shape) != (256, 4):
            raise ValueError('Frozen global product book must be [256,4]')
        curves = {axis: torch.empty((nl, nh, 3), dtype=torch.float64) for axis in ('k', 'v')}
        full_bytes = 0
        for layer, cache_layer in enumerate(cache.layers):
            sk, sv = cache_layer.keys.index_select(2, visual), cache_layer.values.index_select(2, visual)
            if sk.dtype != torch.float16 or sv.dtype != torch.float16 or sk.shape != sv.shape or tuple(sk.shape[1:]) != (nh, nv, 128):
                raise ValueError('Uniform Batch1 FP16 visual source shape required')
            full_bytes += sk.numel() * sk.element_size() + sv.numel() * sv.element_size()
            entry = encode_layer(sk, sv, book, factors, layer, value_group)
            original_mean = sk.float().mean(-2, keepdim=True).half()
            rotated_source = wht((sk.float() - original_mean.float()).contiguous())
            for axis, source, initial, target in (
                    ('k', sk, entry['krot'], rotated_source),
                    ('v', sv, entry['v_base'], sv)):
                codes, amplitudes, _, _ = encode_sign_depth(target, initial, 2)
                actual = initial
                for depth in (0, 1, 2):
                    if depth:
                        actual = decode_sign_stage(actual, codes[depth - 1], amplitudes[depth - 1])
                    native = (torch_wht(actual.float()) + entry['mean'].float()).half() if axis == 'k' else actual
                    curves[axis][layer, :, depth] = (native.float() - source.float()).square().mean((0, 2, 3)).double().cpu()
        curves = {axis: value.tolist() for axis, value in curves.items()}
        order, _ = greedy_stage_order(curves, weights)
        validate_stages(order, nl, nh)
        if len(order) != 4 * nl * nh:
            raise RuntimeError('Per-image order must cover exactly two stages on each axis/L/H unit')
        diagnostic = {}
        for axis in ('k', 'v'):
            ds = [sum(curves[axis][layer][head][depth] for layer in range(nl) for head in range(nh)) for depth in (0, 1, 2)]
            diagnostic[axis] = dict(mean_native_D0=ds[0] / (nl * nh), mean_native_D1=ds[1] / (nl * nh),
                mean_native_D2=ds[2] / (nl * nh), D1_over_D0=ds[1] / max(ds[0], 1e-30),
                D2_over_D1=ds[2] / max(ds[1], 1e-30))
        # No source KV, temporary code, decoded tensor, or full D table is put
        # on self. All local prepass owners are released when this returns.
        return order, full_bytes, diagnostic

    def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
        original_settings = type(self).settings
        settings = dict(original_settings)
        if settings.get('selected_stages') or settings.get('interaction_policy_path'):
            raise ValueError('Image RD computes its own stage order; no static stage/policy override')
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            order, full_bytes, diagnostic = self._image_order(model, cache, visual, settings)
        torch.cuda.synchronize()
        self.image_rd_prepass_wall_ms = 1000 * (time.perf_counter() - started)
        self.image_rd_sensitivity_bytes = self.image_rd_sensitivity.numel() * self.image_rd_sensitivity.element_size()
        reserve = reserve_encoder_bytes(full_bytes, float(settings.get('budget_bits', 1.8)), self.image_rd_sensitivity_bytes)
        working = dict(settings)
        working.pop('selected_units', None)
        working.pop('depth_policy', None)
        working['selected_stages'] = order
        working['budget_bits'] = reserve['super_budget_bits']
        torch.cuda.synchronize()
        encoding_started = time.perf_counter()
        try:
            type(self).settings = working
            super()._initialize(model, cache, visual, method, legacy_scripts, kivi_source, False)
        finally:
            type(self).settings = original_settings
        torch.cuda.synchronize()
        self.image_rd_serving_encode_wall_ms = 1000 * (time.perf_counter() - encoding_started)
        if self.visual_full_bytes != full_bytes or self.depth_plan['allowed_bytes'] != reserve['reduced_allowed_bytes']:
            raise RuntimeError('Prepass denominator or exact reserved super cap mismatch')
        self.encoder_parameter_bytes += self.image_rd_sensitivity_bytes
        self.predicted_budget_bytes += self.image_rd_sensitivity_bytes
        self.residual_budget_bits = reserve['original_budget_bits']
        self.actual_budget_bits = 16 * self.predicted_budget_bytes / full_bytes
        self.depth_plan['fixed_bytes'] += self.image_rd_sensitivity_bytes
        self.depth_plan['predicted_bytes'] = self.predicted_budget_bytes
        self.depth_plan['allowed_bytes'] = reserve['original_allowed_bytes']
        self.depth_plan['actual_budget_bits'] = self.actual_budget_bits
        self.image_rd_budget_reservation = reserve
        self.image_rd_prepass_distortion = diagnostic
        owned = self.owned()
        actual = sum(owned[name] for name in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position')) + self.encoder_parameter_bytes
        if actual != self.predicted_budget_bytes or actual > reserve['original_allowed_bytes'] or self.actual_budget_bits > self.residual_budget_bits + 1e-12:
            raise RuntimeError('Image RD complete owner including retained FP32 w exceeds original budget')
        if any(hasattr(state, 'visual_k') or hasattr(state, 'visual_v') for state in self.states):
            raise RuntimeError('Serving state retained source FP16 visual KV')

    def owned(self):
        result = super().owned()
        sensitivity = self.image_rd_sensitivity.numel() * self.image_rd_sensitivity.element_size()
        result['image_rd_encoder_sensitivity'] = sensitivity
        result['total_unique'] += sensitivity
        # Encoder-only constants are reported as their own category. They are
        # included once by encoder_calibration_parameters_bytes in visual
        # eligibility, never also added to visual_base/selector_metadata.
        return result

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend='actual packed zero/one/two-stage reader with query-free per-image native-RD allocation',
            image_rd_enabled=True, image_rd_sensitivity_bytes=self.image_rd_sensitivity_bytes,
            image_rd_sensitivity_shape=list(self.image_rd_sensitivity.shape),
            image_rd_sensitivity_source_sha256=self.image_rd_sensitivity_hash,
            image_rd_sensitivity_fp32_table_sha256=self.image_rd_sensitivity_table_hash,
            image_rd_sensitivity_K_V_separately_normalized=False,
            image_rd_budget_reservation=self.image_rd_budget_reservation,
            image_rd_prepass_wall_ms=self.image_rd_prepass_wall_ms,
            image_rd_serving_encode_wall_ms=self.image_rd_serving_encode_wall_ms,
            image_rd_encoder_scope='current image visual source KV only, frozen common-scale FP32 CE sensitivities; no current/future query, question, answer, gradient or task score',
            image_rd_distortion_scope='per-image per-unit actual native-coordinate FP16 D(0/1/2); source proxy, not task utility',
            image_rd_prepass_distortion=self.image_rd_prepass_distortion,
            image_rd_extra_encoding_pass=True,
            image_rd_temporary_source_or_codes_retained=False,
            image_rd_full_distortion_table_retained=False,
            image_rd_official_reproduction=False,
            image_rd_research_role='single development prototype; not a retuned fresh-confirmation method')
        return result
