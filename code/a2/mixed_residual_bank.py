"""One static K/V priority order and one compact residual pool per layer.

Every visual token retains its base K2/V1 representation.  Both axes may be
refined at the same KV-head. Only the actual affordable prefix is stored; no
source FP16 visual K/V is retained. This is a new format, not an old-run edit.
"""
import hashlib
import json
import math
import struct
from pathlib import Path
import torch
from affine_rate_bank import fold
from fast_wht import wht
from fused_progressive import decode_last_generic
from masked_bank import A2Bank, unique_bytes
from metadata_direction_bank import MetadataDirectionBank
from quantizers import binary_centroids, pack1
from mixed_residual_attention import attention


class MixedResidualBank(MetadataDirectionBank):
    def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts, kivi_source, False)
        cfg = type(self).settings
        if cfg.get('functional_v_factor_path'):
            raise ValueError('The first mixed-residual experiment uses the frozen base factors only')
        nl, nh, nv = len(self.states), self.states[0].visual_k.shape[1], visual.numel()
        dimension = self.states[0].visual_k.shape[-1]
        device = self.states[0].visual_k.device
        if dimension != 128 or nv % 128:
            raise ValueError('D128 and complete visual quantization groups required')
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
        requested = []
        self.interaction_policy_hash = None
        self.interaction_policy_strategy = None
        self.encoder_policy_table = None
        self.encoder_ranking_table = None
        self.policy_constant_bytes = 0
        self.ranking_constant_bytes = 0
        if cfg.get('interaction_policy_path'):
            if cfg.get('selected_units'):
                raise ValueError('Use a policy table or a priority list, not both')
            path = Path(cfg['interaction_policy_path'])
            artifact = torch.load(path, map_location='cpu', weights_only=False)
            if artifact.get('format') not in ('trajkv_interaction_policy_v1', 'trajkv_round5_interaction_policies_v1'):
                raise ValueError('Unrecognized interaction policy format')
            strategy = cfg['interaction_policy_strategy']
            alias = {'dp': 'dp_exact', 'dp_exact': 'dp'}.get(strategy)
            if strategy not in artifact['policies'] and alias in artifact['policies']:
                strategy = alias
            policy = artifact['policies'][strategy]
            if not isinstance(policy, torch.Tensor) or policy.dtype != torch.uint8 or policy.ndim != 2 or policy.shape[1] != nl or nh > 4:
                raise ValueError('Policy must be uint8 [max_budget+1,layers] with at most four KV-heads')
            self.encoder_policy_table = policy.detach().cpu().clone().contiguous()
            self.policy_constant_bytes = self.encoder_policy_table.numel() * self.encoder_policy_table.element_size()
            self.interaction_policy_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.interaction_policy_strategy = strategy
            for cost, row in enumerate(self.encoder_policy_table):
                if len(self._decode_policy_row(row, nh)) != cost:
                    raise ValueError('Every policy row C must select exactly C axis/head units')
            self.encoder_parameter_bytes += self.policy_constant_bytes
        else:
            for unit in cfg.get('selected_units', []):
                if len(unit) != 3:
                    raise ValueError('selected_units require [axis,layer,head] triples')
                axis, layer, head = str(unit[0]).lower(), int(unit[1]), int(unit[2])
                if axis not in ('k', 'v') or not 0 <= layer < nl or not 0 <= head < nh:
                    raise ValueError('Invalid mixed residual unit')
                requested.append((axis, layer, head))
            if len(set(requested)) != len(requested):
                raise ValueError('Duplicate mixed residual units')
            # Always charge all 224 uint16 slots in the 28x4 target format.
            # A short priority list reserves unused cold entries but does not
            # silently make the padded units eligible for selection.
            capacity = 2 * nl * nh
            ids = [(0 if axis == 'k' else 1) * nl * nh + layer * nh + head for axis, layer, head in requested]
            seen = set(ids)
            full_ids = ids + [value for value in range(capacity) if value not in seen]
            self.encoder_ranking_table = struct.pack('<' + 'H' * capacity, *full_ids)
            self.ranking_constant_bytes = len(self.encoder_ranking_table)
            self.encoder_parameter_bytes += self.ranking_constant_bytes
        self.residual_lookup = torch.full((nl, 2, nh), -1, device=device, dtype=torch.int32)
        base_bytes = (nl * nh * nv * (32 + 2 + 16) + nl * nh * dimension * 2
                      + 4 * nl * nh * dimension * (nv // self.vgroup) + self.book.numel() * 2
                      + self.residual_lookup.numel() * 4 + 8 * nv)
        self.residual_unit_bytes = nv * (dimension // 8 + 2)
        self.residual_budget_bits = float(cfg.get('budget_bits', 1.8))
        allowed = math.floor(self.residual_budget_bits * self.visual_full_bytes / 16)
        fixed_bytes = base_bytes + self.encoder_parameter_bytes
        if fixed_bytes > allowed:
            raise RuntimeError('Mixed base plus lookup/ranking/constants already exceeds budget')
        affordable = (allowed - fixed_bytes) // self.residual_unit_bytes
        maximum = len(requested) if self.encoder_policy_table is None else self.encoder_policy_table.shape[0] - 1
        count = min(maximum, affordable)
        limit = cfg.get('res_units_limit', cfg.get('residual_units_limit'))
        if limit is not None:
            count = min(count, max(0, int(limit)))
        self.policy_budget_row = count if self.encoder_policy_table is not None else None
        self.policy_capped_by_table = self.encoder_policy_table is not None and affordable > maximum
        self.requested_residual_units = requested if self.encoder_policy_table is None else None
        self.selected_residual_units = (requested[:count] if self.encoder_policy_table is None else
                                        self._decode_policy_row(self.encoder_policy_table[count], nh))
        if len(self.selected_residual_units) != count:
            raise RuntimeError('Policy row and charged residual unit count disagree')
        self.predicted_budget_bytes = fixed_bytes + count * self.residual_unit_bytes
        for layer, state in enumerate(self.states):
            selected = [(axis, head) for axis, index, head in self.selected_residual_units if index == layer]
            if selected:
                state.mixed_residual_code = torch.empty((len(selected), nv, dimension // 32), device=device, dtype=torch.int32)
                state.mixed_residual_alpha = torch.empty((len(selected), nv), device=device, dtype=torch.float16)
                self.refinements[layer] = [state.mixed_residual_code, state.mixed_residual_alpha]
                for compact_index, (axis, head) in enumerate(selected):
                    self.residual_lookup[layer, 0 if axis == 'k' else 1, head] = compact_index
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
            k_slots = [(slot, head) for slot, (axis, head) in enumerate(selected) if axis == 'k']
            if k_slots:
                slots = torch.tensor([slot for slot, _ in k_slots], device=device, dtype=torch.long)
                heads = torch.tensor([head for _, head in k_slots], device=device, dtype=torch.long)
                actual_base = (template * state.direction_gain.float()).half()
                error = (rotated.index_select(1, heads).float() - actual_base.index_select(1, heads).float())[0]
                state.mixed_residual_code.index_copy_(0, slots, pack1(error >= 0))
                state.mixed_residual_alpha.index_copy_(0, slots, error.abs().mean(-1).half())
            del state.visual_k
            record = binary_centroids(state.visual_v.transpose(2, 3).contiguous(), group_size=self.vgroup)
            if 'value_head_factors' in params:
                record = fold(record, 1, params['value_head_factors'][layer][None, :, None, None],
                              params['value_center_offsets'][layer][None, :, None, None])
            state.v_code, state.v_scale, state.v_min = record
            v_slots = [(slot, head) for slot, (axis, head) in enumerate(selected) if axis == 'v']
            if v_slots:
                slots = torch.tensor([slot for slot, _ in v_slots], device=device, dtype=torch.long)
                heads = torch.tensor([head for _, head in v_slots], device=device, dtype=torch.long)
                decoded = decode_last_generic(*record, self.vgroup, 1).transpose(2, 3)
                error = (state.visual_v.index_select(1, heads).float() - decoded.index_select(1, heads).float())[0]
                state.mixed_residual_code.index_copy_(0, slots, pack1(error >= 0))
                state.mixed_residual_alpha.index_copy_(0, slots, error.abs().mean(-1).half())
            state.mixed_residual_lookup = self.residual_lookup[layer]
            del state.visual_v
        self.method = method
        owned = self.owned()
        actual = (owned['visual_base'] + owned['visual_residual'] + owned['selector_metadata']
                  + owned['visual_position'] + self.encoder_parameter_bytes)
        if actual != self.predicted_budget_bytes:
            raise RuntimeError(f'Mixed residual owner mismatch: actual={actual}, predicted={self.predicted_budget_bytes}')
        self.actual_budget_bits = 16 * actual / self.visual_full_bytes
        if self.actual_budget_bits > self.residual_budget_bits + 1e-12:
            raise RuntimeError('Mixed residual actual complete owner exceeds budget')

    @staticmethod
    def _decode_policy_row(row, heads):
        units = []
        for layer, value in enumerate(row):
            packed = int(value)
            if packed >> (2 * heads):
                raise ValueError('Policy uses bits outside the model KV-head range')
            for head in range(heads):
                mode = (packed >> (2 * head)) & 3
                if mode & 1:
                    units.append(('k', layer, head))
                if mode & 2:
                    units.append(('v', layer, head))
        return units

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
        cost = q.numel() * 12 + q.shape[1] * 128 * 6 + q.shape[1] * q.shape[-2] * 4 + state.mixed_residual_lookup.numel() * 4
        event['selection_bytes'] += cost
        self.extra_read_bytes += cost + residual_bytes
        return attention(q, state, self.book, self.vgroup)

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend='packed all-support K2/V1 with one compact mixed K/V sign-residual pool per layer',
            mixed_residual_selected_units=self.selected_residual_units,
            mixed_residual_requested_units=self.requested_residual_units,
            mixed_residual_units=len(self.selected_residual_units),
            mixed_residual_K_units=sum(axis == 'k' for axis, _, _ in self.selected_residual_units),
            mixed_residual_V_units=sum(axis == 'v' for axis, _, _ in self.selected_residual_units),
            mixed_residual_dual_lookup_bytes=self.residual_lookup.numel() * 4,
            mixed_residual_encoder_ranking_bytes=self.ranking_constant_bytes,
            mixed_residual_encoder_ranking_sha256=(hashlib.sha256(self.encoder_ranking_table).hexdigest()
                if self.encoder_ranking_table is not None else None),
            mixed_residual_encoder_policy_bytes=self.policy_constant_bytes,
            mixed_residual_interaction_policy_sha256=self.interaction_policy_hash,
            mixed_residual_interaction_policy_strategy=self.interaction_policy_strategy,
            mixed_residual_policy_budget_row=self.policy_budget_row,
            mixed_residual_policy_capped_by_table=self.policy_capped_by_table,
            mixed_residual_unit_bytes=self.residual_unit_bytes,
            mixed_residual_actual_budget_bits=self.actual_budget_bits,
            sparse_residual_actual_budget_bits=self.actual_budget_bits,
            mixed_residual_budget_verified=True,
            mixed_residual_budget_bits=self.residual_budget_bits,
            mixed_residual_full_visual_support=True,
            mixed_residual_source_fp16_retained=False,
            mixed_residual_selector_scope=('calibration-frozen non-nested interaction policy row[C], C computed after complete owner costs; no prefix truncation'
                if self.encoder_policy_table is not None else 'one calibration-frozen K/V priority prefix, shortened only by actual image length and complete owner budget'),
            mixed_residual_format='one compact RC[M,N,4] int32 and alpha[M,N] F16 pool per layer; independent K/V [2,Hkv] lookups',
            mixed_residual_K_reference='original centered WHT(source K) minus actual FP16(book * final calibrated gain), frozen mean unchanged',
            mixed_residual_V_reference='original source V minus actual calibrated V1 decode',
            mixed_residual_same_head_both_axes_supported=True)
        return result
