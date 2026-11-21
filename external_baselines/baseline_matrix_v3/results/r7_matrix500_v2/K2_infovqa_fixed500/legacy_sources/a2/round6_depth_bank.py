"""Static zero/one/two-stage K/V residuals with an actually compact per-layer pool.

Every visual token retains its base K2/V1 representation.  Both axes may be
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
from round6_depth_attention import attention
from round6_depth_codec import encode_sign_depth, plan_depth_stages


class DepthResidualBank(MetadataDirectionBank):
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
        if tuple(self.book.shape) != (256, 4):
            raise ValueError('Frozen global 256x4 book required')
        self.decoder_books, self.decoder_hash, self.headwise_budget_ok = None, None, False
        self.vbits, self.v_channel = 1, True
        window = cfg.get('v_group_size', 'all')
        self.vgroup = nv if window == 'all' else int(window)
        if self.vgroup not in (128, nv):
            raise ValueError('Only g128 or whole-image V metadata supported')
        if nv % 32 or nv % self.vgroup:
            raise ValueError('V bit words require N divisible by 32 and complete metadata groups; no padding')
        path = Path(cfg['metadata_factor_path'])
        base = json.loads(path.read_text())
        self.metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if base['encoder_book_sha256'] != book_hash or base['value_group_size'] != window:
            raise ValueError('Frozen metadata/book/window mismatch')
        self.key_gain_mode = base['key_gain_mode']
        if self.key_gain_mode not in ('none', 'centered', 'whole'):
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
        self.encoder_parameter_bytes = sum(p.numel() * p.element_size() for p in params.values())
        if self.encoder_parameter_bytes != 4 * base['parameter_count']:
            raise ValueError('Frozen factor byte mismatch')
        if cfg.get('interaction_policy_path'):
            raise ValueError('Depth uses selected_stages or selected_units, not an old single-stage interaction policy')
        base_bytes = (nl * nh * nv * (32 + 2 + 16) + nl * nh * dimension * 2
                      + 4 * nl * nh * dimension * (nv // self.vgroup) + self.book.numel() * 2 + 8 * nv)
        plan = plan_depth_stages(layers=nl, heads=nh, visual_tokens=nv,
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
            norm = rotated.norm(dim=-1, keepdim=True)
            initial = (norm / (dimension ** .5)).clamp_min(1e-20)
            parts, book, indices = (rotated / initial).reshape(-1, 4), self.book.float(), []
            for start in range(0, len(parts), 65536):
                part = parts[start:start + 65536]
                distance = part.square().sum(-1, keepdim=True) + book.square().sum(-1)[None] - 2 * (part @ book.T)
                indices.append(distance.argmin(-1).byte())
            state.direction_code = torch.cat(indices).reshape(*rotated.shape[:-1], 32).contiguous()
            template = book[state.direction_code.long()].reshape(rotated.shape)
            state.direction_gain = (norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half().contiguous()
            if 'key_head_factors' in params:
                gain = params['key_head_factors'][layer][None, :, None, None]
                state.direction_gain = (state.direction_gain.float() * gain).half().contiguous()
                if self.key_gain_mode == 'whole':
                    state.direction_mean = (state.direction_mean.float() * gain).half().contiguous()
            if any(axis == 'k' for axis, _, _ in selected):
                actual_base = (template * state.direction_gain.float()).half()
                self._encode_axis(layer, state, 'k', selected, rotated[0], actual_base[0])
            del state.visual_k
            record = binary_centroids(state.visual_v.transpose(2, 3).contiguous(), group_size=self.vgroup)
            if 'value_head_factors' in params:
                record = fold(record, 1, params['value_head_factors'][layer][None, :, None, None],
                              params['value_center_offsets'][layer][None, :, None, None])
            state.v_code, state.v_scale, state.v_min = record
            if any(axis == 'v' for axis, _, _ in selected):
                decoded = decode_last_generic(*record, self.vgroup, 1).transpose(2, 3)
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
            self.depth_distortions.append(dict(axis=axis, layer=layer, head=head, depth=depth,
                squared_error_by_depth=distortions,
                alpha2=(distortions[2] / max(distortions[1], 1e-30) if depth == 2 else None),
                scope='real packed sign stage and actual sequential FP16 decode; coordinate error, not task quality'))

    def owned(self):
        result = super().owned()
        lookup_bytes = unique_bytes([self.residual_lookup])
        result['selector_metadata'] += lookup_bytes
        result['total_unique'] += lookup_bytes
        result['visual_with_metadata_ratio'] = (result['visual_base'] + result['visual_residual'] + result['selector_metadata']) / self.visual_full_bytes
        return result

    def read(self, q, layer):
        state = self.states[layer]
        event = self.events[(self.turn, self.phase, self.pass_kind, layer)]
        event['calls'] += 1
        event['query_tokens'] += q.shape[-2]
        event['base_visual_bytes'] += unique_bytes([state.direction_code, state.direction_gain, state.direction_mean,
            state.v_code, state.v_scale, state.v_min, self.book])
        residual_bytes = unique_bytes(self.refinements[layer] or [])
        event['residual_bytes'] += residual_bytes
        event['position_bytes'] += 8 * (state.visual_pos.numel() + state.text_length)
        event['text_bytes'] += 2 * state.text_length * state.text_k.shape[1] * 128 * state.text_k.element_size()
        cost = q.numel() * 12 + q.shape[1] * 128 * 6 + q.shape[1] * q.shape[-2] * 4 + state.depth_residual_lookup.numel() * 4
        event['selection_bytes'] += cost
        self.extra_read_bytes += cost + residual_bytes
        return attention(q, state, self.book, self.vgroup)

    def result_metadata(self):
        result = super().result_metadata()
        selected = self.selected_residual_stages
        depths = {}
        for axis, layer, head, stage in selected:
            depths[(axis, layer, head)] = max(depths.get((axis, layer, head), 0), stage + 1)
        plan = self.depth_plan
        result.update(backend='packed all-support K2/V1 with compact zero/one/two-stage K/V sign residuals and one online softmax',
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
        return result
