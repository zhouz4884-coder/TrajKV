"""Round8 packed K8/position dictionaries, native joint-V8 and joint amplitudes.

Every visual token retains its base packed K1/V1 or K2/V1 representation. Both axes may be
refined twice at the same KV-head, with real FP16 reconstruction between stages. Only the actual affordable prefix is stored; no
source FP16 visual K/V is retained. This is a new format, not an old-run edit.
"""
import hashlib
import json
from pathlib import Path
import torch
from affine_rate_bank import fold
from fast_wht import wht
from fused_progressive import decode_last_generic
from masked_bank import A2Bank, unique_bytes
from metadata_direction_bank import MetadataDirectionBank
from quantizers import binary_centroids
from round8_attention import attention
from round8_codec import (encode_key, decode_key, encode_sign_depth, plan_depth_stages, key_book_layout, key_template,
    cube_book, cube_indices_from_codes, encode_value_joint8, decode_value_joint8)
from round8_joint_gain import joint_refit_key
from round7_key_layout import plan_key_stages


class Round8Bank(MetadataDirectionBank):
    def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts, kivi_source, False)
        cfg = type(self).settings
        if cfg.get('functional_v_factor_path'):
            raise ValueError('Depth comparisons use the frozen base factors only')
        nl, nh, nv = len(self.states), self.states[0].visual_k.shape[1], visual.numel()
        dimension = self.states[0].visual_k.shape[-1]
        device = self.states[0].visual_k.device
        if dimension != 128:
            raise ValueError('D128 required')
        book_path = Path(cfg['direction_book_path'])
        book_hash = hashlib.sha256(book_path.read_bytes()).hexdigest()
        self.book = torch.tensor(json.loads(book_path.read_text())['centers'], device=device, dtype=torch.float16).contiguous()
        layout = key_book_layout(self.book)
        if not bool(torch.isfinite(self.book).all()):
            raise ValueError('Finite Round8 decoder book required')
        self.book_hash = book_hash
        self.key_layout = layout
        self.key_words = layout['words']
        self.key_dimension = layout['dimension']
        self.key_bytes_per_token = layout['index_bytes']
        self.key_assignment = cfg.get('key_assignment', 'l2')
        self.key_amplitude = cfg.get('key_amplitude', 'norm')
        self.key_encoding = cfg.get('key_encoding', 'reencode')
        self.value_encoding = cfg.get('value_encoding', 'binary')
        if self.key_encoding not in ('reencode', 'decoder-only'):
            raise ValueError('Round8 K encoding must explicitly be reencode/decoder-only')
        if self.value_encoding not in ('binary', 'cube-identity', 'decoder-only', 'reencode'):
            raise ValueError('Round8 V encoding mode is not declared')
        self.encoder_book = None
        self.encoder_book_hash = None
        self.encoder_book_bytes = 0
        if self.key_encoding == 'decoder-only':
            old_path = Path(cfg['encoder_book_path'])
            self.encoder_book_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
            self.encoder_book = torch.tensor(json.loads(old_path.read_text())['centers'],
                device=device, dtype=torch.float16).contiguous()
            if tuple(self.encoder_book.shape) != (16, 4):
                raise ValueError('Decoder-only control must retain the frozen G4_16 encoder')
            self.encoder_book_bytes = self.encoder_book.numel() * self.encoder_book.element_size()
        self.vbook = None
        self.vbook_hash = None
        self.value_book_bytes = 0
        if self.value_encoding != 'binary':
            vp = Path(cfg['value_joint_book_path'])
            self.vbook_hash = hashlib.sha256(vp.read_bytes()).hexdigest()
            self.vbook = torch.tensor(json.loads(vp.read_text())['centers'],
                device=device, dtype=torch.float16).contiguous()
            if tuple(self.vbook.shape) != (256, 8) or not bool(torch.isfinite(self.vbook).all()):
                raise ValueError('First Round8 V format requires one finite 256x8 FP16 book')
            self.value_book_bytes = self.vbook.numel() * self.vbook.element_size()
            if self.value_encoding == 'cube-identity' and not torch.equal(self.vbook, cube_book(device)):
                raise ValueError('Identity V control must use the actual binary cube')
        self.joint_gain_enabled = bool(cfg.get('joint_gain_refit', False))
        self.joint_gain_statistics = []
        if self.joint_gain_enabled and self.key_encoding != 'reencode':
            raise ValueError('Joint-gain first control is separate from decoder-only calibration')
        self.decoder_books, self.decoder_hash, self.headwise_budget_ok = None, None, False
        self.vbits, self.v_channel = 1, True
        window = cfg.get('v_group_size', 'all')
        self.vgroup = nv if window == 'all' else int(window)
        if self.vgroup not in (128, nv):
            raise ValueError('Only g128 or whole-image V metadata supported')
        if nv % 32 or nv % self.vgroup:
            raise ValueError('V bit words require N divisible by 32 and complete metadata groups; no padding')
        if self.vbook is not None and self.vgroup != nv:
            raise ValueError('Joint V controls preserve the all-image frame')
        path = Path(cfg['metadata_factor_path'])
        base = json.loads(path.read_text())
        self.metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if cfg.get('round8_factor_policy') != 'inherit_from_reference_book':
            raise ValueError('Explicit Round8 factor-inheritance policy required')
        reference = Path(cfg['metadata_reference_book_path'])
        self.metadata_reference_book_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        if base['encoder_book_sha256'] != self.metadata_reference_book_hash or base['value_group_size'] != window:
            raise ValueError('Frozen factor source/reference/window mismatch')
        if self.encoder_book is not None and self.encoder_book_hash != self.metadata_reference_book_hash:
            raise ValueError('Decoder-only encoder differs from the frozen factor reference')
        self.factor_inheritance = base.get('round7_inheritance')
        if self.factor_inheritance is not None:
            from round7_key_artifacts import verify_factor_inheritance
            verify_factor_inheritance(base)
        self.metric_hash, self.metric_u, self.metric_gamma = None, None, None
        self.metric_bytes = 0
        self.metric_metadata = None
        if cfg.get('key_metric_path'):
            raise ValueError('Round8 first controls do not reuse a fitted query metric')
        self.key_gain_mode = base['key_gain_mode']
        if self.key_gain_mode not in ('none', 'centered'):
            raise ValueError('Unsupported frozen K gain mode')
        params = {}
        for name in ('key_head_factors', 'value_head_factors', 'value_center_offsets'):
            if base.get(name) is None:
                continue
            tensor = torch.tensor(base[name], device=device, dtype=torch.float32)
            if tuple(tensor.shape) != (nl, nh) or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid frozen metadata factors')
            if 'factors' in name and not bool((tensor > 0).all()):
                raise ValueError('Frozen gains must be positive')
            params[name] = tensor
        self.encoder_factors = params
        self.factor_parameter_bytes = sum(p.numel() * p.element_size() for p in params.values())
        self.encoder_parameter_bytes = self.factor_parameter_bytes
        if self.encoder_parameter_bytes != 4 * base['parameter_count']:
            raise ValueError('Frozen factor byte mismatch')
        self.encoder_parameter_bytes += self.metric_bytes + self.encoder_book_bytes
        if cfg.get('interaction_policy_path'):
            raise ValueError('Depth uses selected_stages or selected_units, not an old single-stage interaction policy')
        base_bytes = (nl * nh * nv * (self.key_bytes_per_token + 2 + 16) + nl * nh * dimension * 2
                      + 4 * nl * nh * dimension * (nv // self.vgroup)
                      + self.book.numel() * 2 + self.value_book_bytes + 8 * nv)
        plan = plan_key_stages(layout_reference=cfg.get('residual_layout_reference'),
            key_index_bytes=self.key_bytes_per_token, book_bytes=self.book.numel() * 2,
            reference_factor_bytes=self.factor_parameter_bytes,
            layers=nl, heads=nh, visual_tokens=nv,
            visual_full_bytes=self.visual_full_bytes, base_bytes=base_bytes,
            factor_bytes=self.encoder_parameter_bytes, budget_bits=float(cfg.get('budget_bits', 1.8)),
            selected_units=cfg.get('selected_units', []), selected_stages=cfg.get('selected_stages'),
            depth_policy=cfg.get('depth_policy', 'incumbent'),
            v_mixed_pair_fraction=float(cfg.get('v_mixed_pair_fraction', .5)),
            stage_limit=cfg.get('res_stages_limit', cfg.get('res_units_limit', cfg.get('residual_units_limit'))))
        self.depth_plan = plan
        self.encoder_ranking_table = plan['ranking_constant']
        self.encoder_depth_policy = plan['policy_constant']
        self.ranking_constant_bytes, self.policy_constant_bytes = len(self.encoder_ranking_table), len(self.encoder_depth_policy)
        self.encoder_parameter_bytes += self.ranking_constant_bytes + self.policy_constant_bytes
        self.residual_lookup = torch.full((nl, 2, nh, 2), -1, device=device, dtype=torch.int32)
        self.residual_unit_bytes = plan['stage_bytes']
        self.residual_budget_bits = float(cfg.get('budget_bits', 1.8))
        self.requested_residual_stages = plan['requested_stages']
        self.selected_residual_stages = plan['selected_stages']
        self.predicted_budget_bytes = plan['predicted_bytes']
        self.depth_distortions = []
        for layer, state in enumerate(self.states):
            selected = [(axis, head, stage) for axis, index, head, stage in self.selected_residual_stages if index == layer]
            if selected:
                state.depth_residual_code = torch.empty((len(selected), nv, dimension // 32), device=device, dtype=torch.int32)
                state.depth_residual_alpha = torch.empty((len(selected), nv), device=device, dtype=torch.float16)
                self.refinements[layer] = [state.depth_residual_code, state.depth_residual_alpha]
                for slot, (axis, head, stage) in enumerate(selected):
                    self.residual_lookup[layer, 0 if axis == 'k' else 1, head, stage] = slot
            source = state.visual_k.float()
            state.direction_mean = source.mean(-2, keepdim=True).half().contiguous()
            rotated = wht(source - state.direction_mean.float())
            factor = params['key_head_factors'][layer][None, :, None, None] if 'key_head_factors' in params else None
            metric_u = None if self.metric_u is None else self.metric_u[layer][None, :, None, :].float()
            metric_gamma = None if self.metric_gamma is None else self.metric_gamma[layer][None, :, None, None]
            active_encoder = self.encoder_book if self.encoder_book is not None else self.book
            state.direction_code, state.direction_gain = encode_key(rotated, active_encoder,
                amplitude=self.key_amplitude, assignment=self.key_assignment,
                gain_factor=factor, chunk=int(cfg.get('key_chunk', 65536)))
            if factor is not None and self.key_gain_mode == 'whole':
                state.direction_mean = (state.direction_mean.float() * factor).half().contiguous()
            if any(axis == 'k' for axis, _, _ in selected):
                residual_book = active_encoder if self.key_encoding == 'decoder-only' else self.book
                actual_base = decode_key(state.direction_code, state.direction_gain, residual_book)
                self._encode_axis(layer, state, 'k', selected, rotated[0], actual_base[0])
            del state.visual_k
            record = binary_centroids(state.visual_v.transpose(2, 3).contiguous(), group_size=self.vgroup)
            if 'value_head_factors' in params:
                record = fold(record, 1, params['value_head_factors'][layer][None, :, None, None],
                              params['value_center_offsets'][layer][None, :, None, None])
            state.v_code, state.v_scale, state.v_min = record
            old_value_base = None
            if self.vbook is not None:
                old_value_base = decode_last_generic(*record, self.vgroup, 1).transpose(2, 3)
                if self.value_encoding in ('cube-identity', 'decoder-only'):
                    state.v_code = cube_indices_from_codes(record[0], nv)
                else:
                    state.v_code = encode_value_joint8(state.visual_v,
                        state.v_scale, state.v_min, self.vbook, chunk=int(cfg.get('value_chunk', 2048)))
            if any(axis == 'v' for axis, _, _ in selected):
                if self.vbook is None:
                    decoded = decode_last_generic(*record, self.vgroup, 1).transpose(2, 3)
                elif self.value_encoding in ('cube-identity', 'decoder-only'):
                    decoded = old_value_base
                else:
                    decoded = decode_value_joint8(state.v_code, state.v_scale, state.v_min, self.vbook)
                self._encode_axis(layer, state, 'v', selected, state.visual_v[0], decoded[0])
            state.depth_residual_lookup = self.residual_lookup[layer]
            del state.visual_v
        self.method = method
        owned = self.owned()
        actual = (owned['visual_base'] + owned['visual_residual'] + owned['selector_metadata']
                  + owned['visual_position'] + self.encoder_parameter_bytes)
        if actual != self.predicted_budget_bytes:
            raise RuntimeError(f'Depth residual owner mismatch: actual={actual}, predicted={self.predicted_budget_bytes}')
        self.actual_budget_bits = 16 * actual / self.visual_full_bytes
        if self.actual_budget_bits > self.residual_budget_bits + 1e-12:
            raise RuntimeError('Depth actual complete owner exceeds budget')

    def _encode_axis(self, layer, state, axis, selected, source, base):
        heads = sorted({head for item_axis, head, _ in selected if item_axis == axis})
        for head in heads:
            depth = sum(item_axis == axis and item_head == head for item_axis, item_head, _ in selected)
            codes, alphas, _, distortions = encode_sign_depth(source[head], base[head], depth)
            for stage in range(depth):
                slot = selected.index((axis, head, stage))
                state.depth_residual_code[slot].copy_(codes[stage])
                state.depth_residual_alpha[slot].copy_(alphas[stage])
            if axis == 'k' and self.joint_gain_enabled:
                if depth != 1:
                    raise ValueError('Joint-gain first control requires exactly one K stage')
                slot = selected.index((axis, head, 0))
                template = key_template(state.direction_code[0, head], self.book)
                gain, alpha, stats = joint_refit_key(source[head].float(), template,
                    state.direction_gain[0, head], state.depth_residual_code[slot],
                    state.depth_residual_alpha[slot])
                state.direction_gain[0, head].copy_(gain)
                state.depth_residual_alpha[slot].copy_(alpha)
                self.joint_gain_statistics.append(dict(layer=layer, head=head, **stats))
                from round6_depth_codec import decode_sign_stage
                new_base = decode_key(state.direction_code[0, head], gain, self.book)
                final = decode_sign_stage(new_base, state.depth_residual_code[slot], alpha)
                distortions = [float((source[head]-new_base.float()).square().sum()),
                               float((source[head]-final.float()).square().sum())]
            self.depth_distortions.append(dict(axis=axis, layer=layer, head=head, depth=depth,
                squared_error_by_depth=distortions,
                alpha2=(distortions[2] / max(distortions[1], 1e-30) if depth == 2 else None),
                scope='real packed sign stage and actual sequential FP16 decode; coordinate error, not task quality'))

    def _visual_tensors(self, state):
        tensors = super()._visual_tensors(state)
        if getattr(self, 'vbook', None) is not None:
            tensors.append(self.vbook)
        return tensors

    def owned(self):
        result = super().owned()
        lookup_bytes = unique_bytes([self.residual_lookup])
        result['selector_metadata'] += lookup_bytes
        result['total_unique'] += lookup_bytes
        result['encoder_constant_bytes'] = self.encoder_parameter_bytes
        result['total_unique'] += self.encoder_parameter_bytes
        result['complete_visual_owner_bytes'] = (result['visual_base'] + result['visual_residual'] + result['selector_metadata'] + result['visual_position'] + self.encoder_parameter_bytes)
        result['visual_with_metadata_ratio'] = (result['visual_base'] + result['visual_residual'] + result['selector_metadata']) / self.visual_full_bytes
        return result

    def read(self, q, layer):
        state = self.states[layer]
        event = self.events[(self.turn, self.phase, self.pass_kind, layer)]
        event['calls'] += 1
        event['query_tokens'] += q.shape[-2]
        event['base_visual_bytes'] += unique_bytes(self._visual_tensors(state))
        residual_bytes = unique_bytes(self.refinements[layer] or [])
        event['residual_bytes'] += residual_bytes
        event['position_bytes'] += 8 * (state.visual_pos.numel() + state.text_length)
        event['text_bytes'] += 2 * state.text_length * state.text_k.shape[1] * 128 * state.text_k.element_size()
        cost = q.numel() * 12 + q.shape[1] * 128 * 6 + q.shape[1] * q.shape[-2] * 4 + state.depth_residual_lookup.numel() * 4
        event['selection_bytes'] += cost
        self.extra_read_bytes += cost + residual_bytes
        return attention(q, state, self.book, self.vgroup, vbook=self.vbook)

    def result_metadata(self):
        result = super().result_metadata()
        selected = self.selected_residual_stages
        depths = {}
        for axis, layer, head, stage in selected:
            depths[(axis, layer, head)] = max(depths.get((axis, layer, head), 0), stage + 1)
        plan = self.depth_plan
        result.update(backend='packed all-support G4_256/G4_16 V1 with compact zero/one/two-stage K/V sign residuals and one online softmax',
            key_payload_bits=(self.key_words.bit_length()-1)/self.key_dimension,
            key_book_words=self.key_words, key_index_bytes_per_token=self.key_bytes_per_token,
            key_book_sha256=self.book_hash, key_assignment=self.key_assignment, key_amplitude=self.key_amplitude,
            encoding_assignment='legacy normalized nearest word; optional actual-FP16 block coordinate updates',
            key_gain_formula=('norm(r)^2/dot(r,t), dot<=1e-12 -> norm(r)/norm(t)' if self.key_amplitude == 'projection' else ('norm(r)/norm(t)' if self.key_amplitude == 'norm' else ('dot(r,t)/dot(t,t)' if self.key_amplitude == 'ls' else 'dot(r,Mt)/dot(t,Mt)'))),
            key_gain_rule='projection_preserving' if self.key_amplitude == 'projection' else self.key_amplitude,
            key_gain_rounding='FP16 amplitude, FP32 multiplication by inherited/fitted head factor, FP16 gain; base decode FP16; each residual decode FP16',
            key_factor_parameter_bytes=self.factor_parameter_bytes, key_factor_inheritance=self.factor_inheritance,
            key_metric_sha256=self.metric_hash, key_metric_bytes=self.metric_bytes, key_metric_metadata=self.metric_metadata,
            key_bcd_iterations=int(type(self).settings.get('key_bcd_iterations',1)) if self.key_assignment != 'l2' else 0,
            key_source_fp16_retained=False, key_nibble_packed=self.key_words == 16,
            residual_layout_reference=plan.get('layout_reference', 'actual'),
            residual_layout_reference_base_bytes=plan.get('layout_reference_base_bytes'),
            residual_layout_reference_complete_bytes=plan.get('layout_reference_predicted_bytes'),
            residual_layout_reference_affordable_stages=plan.get('layout_reference_affordable_stages'),
            residual_layout_actual_affordable_stages=plan.get('layout_actual_affordable_stages', plan['affordable_stages']),
            residual_layout_reference_incumbent_stage_cap=plan.get('layout_reference_incumbent_stage_cap'),
            residual_layout_reference_extra_rank_tables=0,
            depth_residual_selected_stages=selected,
            depth_residual_requested_stages=self.requested_residual_stages,
            depth_residual_requested_units=plan['requested_units'],
            depth_residual_stages=len(selected),
            depth_residual_K_stages=sum(axis == 'k' for axis, _, _, _ in selected),
            depth_residual_V_stages=sum(axis == 'v' for axis, _, _, _ in selected),
            depth_residual_depth1_units=sum(value == 1 for value in depths.values()),
            depth_residual_depth2_units=sum(value == 2 for value in depths.values()),
            depth_residual_K_units=sum(axis == 'k' for axis, _, _ in depths),
            depth_residual_V_units=sum(axis == 'v' for axis, _, _ in depths),
            depth_residual_stage_lookup_bytes=self.residual_lookup.numel() * 4,
            depth_residual_encoder_ranking_bytes=self.ranking_constant_bytes,
            depth_residual_encoder_ranking_sha256=hashlib.sha256(self.encoder_ranking_table).hexdigest(),
            depth_residual_encoder_policy_bytes=self.policy_constant_bytes,
            depth_residual_encoder_policy_sha256=hashlib.sha256(self.encoder_depth_policy).hexdigest(),
            depth_residual_policy=plan['depth_policy'],
            depth_residual_policy_affordable_stages=plan['affordable_stages'],
            depth_residual_incumbent_stage_cap=plan['incumbent_stage_cap'],
            depth_residual_incumbent_selected_units=plan['incumbent_selected_units'],
            depth_residual_stage_bytes=self.residual_unit_bytes,
            depth_residual_fixed_complete_bytes=plan['fixed_bytes'],
            depth_residual_complete_bytes=self.predicted_budget_bytes,
            depth_residual_allowed_bytes=plan['allowed_bytes'],
            depth_residual_actual_budget_bits=self.actual_budget_bits,
            sparse_residual_actual_budget_bits=self.actual_budget_bits,
            depth_residual_budget_verified=True,
            depth_residual_budget_bits=self.residual_budget_bits,
            depth_residual_full_visual_support=True,
            depth_residual_source_fp16_retained=False,
            depth_residual_format='RC[M,N,4] int32 + alpha[M,N] FP16; lookup[L,2,Hkv,2] int32; only actual selected stages stored',
            depth_residual_K_reference='centered WHT source K minus actual sequential FP16 reconstruction; frozen mean unchanged',
            depth_residual_V_reference='source V minus actual sequential FP16 reconstruction from calibrated V1 base',
            depth_residual_same_head_both_axes_supported=True,
            depth_residual_same_head_both_stages_supported=True,
            depth_residual_distortions=self.depth_distortions)
        result.update(stage='round8 actual packed G8/position K and native joint V',
            direction_book_dimension=self.key_dimension,
            key_book_shape=list(self.book.shape), key_position_specific=self.key_layout['positions'],
            key_encoding=self.key_encoding, value_encoding=self.value_encoding,
            key_encoder_book_sha256=self.encoder_book_hash,
            key_encoder_extra_book_bytes=self.encoder_book_bytes,
            metadata_factor_policy='explicit numerical inheritance from frozen reference; no new-book refit',
            metadata_reference_book_sha256=self.metadata_reference_book_hash,
            value_joint_book_sha256=self.vbook_hash, value_joint_book_bytes=self.value_book_bytes,
            value_index_bytes_per_token=16, value_frame='all' if self.vgroup==self.states[0].visual_pos.numel() else self.vgroup,
            value_row_gain_added=False, value_affine_rounding='FP16 multiplication then FP16 addition; no FMA',
            joint_gain_refit=self.joint_gain_enabled, joint_gain_statistics=self.joint_gain_statistics,
            joint_gain_scope='stored effective gain and existing first K alpha; codes/sign/mean unchanged',
            decoder_book_bytes=self.book.numel()*2,
            encoding_assignment=('frozen G4 encoder fields; learned decoder only' if self.key_encoding=='decoder-only'
                                 else 'declared source-normalized nearest assignment; original norm/factor FP16 fold'),
            key_gain_rule='joint-effective-gain-and-alpha' if self.joint_gain_enabled else self.key_amplitude)
        if self.joint_gain_enabled:
            result.update(key_gain_formula='selected first-stage K units: joint effective stored g and existing alpha, F16(F16(g*template)+alpha*sign); unrefined K keeps declared norm/factor rule',
                key_gain_rounding='selected first-stage K: initial norm/factor FP16 fields followed by direct FP16 effective-gain/alpha refit, no factor reapplication; other K keeps original fields; base and each sign stage rounded FP16')
        return result
