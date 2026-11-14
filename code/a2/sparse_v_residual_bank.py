"""Static compact K/V sign residuals under an actual per-image owner budget.

The encoding sees source V once, keeps only the selected residual payload, and
discards every source visual FP16 tensor. Selection is a calibration-frozen
layer/KV-head order; image length determines only its affordable prefix.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from affine_rate_bank import fold
from fast_wht import wht
from fused_progressive import decode_last_generic
from masked_bank import A2Bank, unique_bytes
from metadata_direction_bank import MetadataDirectionBank
from quantizers import binary_centroids, pack1
from sparse_v_residual_attention import attention


class SparseVResidualBank(MetadataDirectionBank):
    def _initialize(self, model, cache, visual, method, legacy_scripts,
                    kivi_source, keep_reference=False):
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts,
                           kivi_source, False)
        cfg = type(self).settings
        self.residual_axis = cfg.get('residual_axis', 'v')
        if self.residual_axis not in ('k', 'v'):
            raise ValueError('residual_axis must be k or v')
        nl = len(self.states)
        nh = self.states[0].visual_k.shape[1]
        nv = visual.numel()
        dim = self.states[0].visual_k.shape[-1]
        device = self.states[0].visual_k.device
        if dim != 128 or nv % 128:
            raise ValueError('Sparse V residual format requires D128 and aligned visual length')
        book_path = Path(cfg['direction_book_path'])
        book_hash = hashlib.sha256(book_path.read_bytes()).hexdigest()
        self.book = torch.tensor(json.loads(book_path.read_text())['centers'], device=device,
                                 dtype=torch.float16).contiguous()
        if self.book.shape != (256, 4):
            raise ValueError('Expected the frozen 256x4 K book')
        self.decoder_books = None
        self.decoder_hash = None
        self.headwise_budget_ok = False
        self.vbits = 1
        self.v_channel = True
        window_setting = cfg.get('v_group_size', 'all')
        self.vgroup = nv if window_setting == 'all' else int(window_setting)
        if self.vgroup not in (128, nv):
            raise ValueError('Only g128 or whole-image V metadata is supported')
        path = Path(cfg['metadata_factor_path'])
        base = json.loads(path.read_text())
        self.metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if base['encoder_book_sha256'] != book_hash or base['value_group_size'] != window_setting:
            raise ValueError('Frozen metadata/book/window mismatch')
        self.key_gain_mode = base['key_gain_mode']
        if self.key_gain_mode not in ('none', 'centered', 'whole'):
            raise ValueError('Unsupported frozen K gain mode')
        params = {}
        for name in ('key_head_factors', 'value_head_factors', 'value_center_offsets'):
            if base.get(name) is None:
                continue
            tensor = torch.tensor(base[name], device=device, dtype=torch.float32)
            if tensor.shape != (nl, nh) or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid frozen metadata factor')
            if 'factors' in name and not bool((tensor > 0).all()):
                raise ValueError('Frozen factors must be positive')
            params[name] = tensor
        self.encoder_parameter_bytes = sum(p.numel()*p.element_size() for p in params.values())
        if self.encoder_parameter_bytes != 4*base['parameter_count']:
            raise ValueError('Frozen metadata parameter accounting mismatch')
        self.functional_v_hash = None
        self.functional_groups = 1
        self.functional_folded_bytes = 0
        self.visual_bias = None
        fg = fo = None
        if cfg.get('functional_v_factor_path'):
            path = Path(cfg['functional_v_factor_path'])
            fit = json.loads(path.read_text())
            if (fit['schema'] != 'functional_v_v1' or fit['base_metadata_sha256'] != self.metadata_hash
                    or fit['encoder_book_sha256'] != book_hash or fit['value_group_size'] != window_setting):
                raise ValueError('Functional V fit differs from the frozen base')
            self.functional_v_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.functional_groups = int(fit['channel_groups'])
            if dim % self.functional_groups:
                raise ValueError('Invalid V channel groups')
            shape = (nl, nh, self.functional_groups)
            fg = torch.tensor(fit['value_channel_factors'], device=device, dtype=torch.float32)
            fo = torch.tensor(fit['value_channel_offsets'], device=device, dtype=torch.float32)
            if fg.shape != shape or fo.shape != shape or not bool(torch.isfinite(fg).all() & torch.isfinite(fo).all()):
                raise ValueError('Invalid functional V parameters')
            if not bool((fg > 0).all()):
                raise ValueError('V gains must be positive')
            self.functional_folded_bytes = 4*(fg.numel()+fo.numel())
            self.encoder_parameter_bytes += self.functional_folded_bytes
            if fit['visual_logit_bias'] is not None:
                self.visual_bias = torch.tensor(fit['visual_logit_bias'], device=device,
                                               dtype=torch.float32).contiguous()
                if self.visual_bias.shape != (nl, nh) or not bool(torch.isfinite(self.visual_bias).all()):
                    raise ValueError('Invalid visual logit bias')

        requested = [tuple(map(int, unit)) for unit in cfg.get('selected_units', [])]
        if len(set(requested)) != len(requested) or any(len(u) != 2 or not 0 <= u[0] < nl or not 0 <= u[1] < nh for u in requested):
            raise ValueError('selected_units must be distinct valid layer/KV-head pairs')
        # The frozen encoder order uses one uint16 flattened unit id. It is
        # charged as an encoder constant, independently of the GPU lookup.
        self.ranking_constant_bytes = 2*len(requested)
        self.encoder_parameter_bytes += self.ranking_constant_bytes
        self.residual_lookup = torch.full((nl, nh), -1, device=device, dtype=torch.int32)
        base_bytes = (nl*nh*nv*(32+2+16) + nl*nh*dim*2
                      + 4*nl*nh*dim*(nv//self.vgroup) + self.book.numel()*2
                      + (0 if self.visual_bias is None else self.visual_bias.numel()*4)
                      + self.residual_lookup.numel()*4 + 8*nv)
        self.residual_unit_bytes = nv*(dim//8+2)
        self.residual_budget_bits = float(cfg.get('budget_bits', 1.8))
        allowed = math.floor(self.residual_budget_bits*self.visual_full_bytes/16)
        fixed_bytes = base_bytes+self.encoder_parameter_bytes
        if fixed_bytes > allowed:
            raise RuntimeError('Frozen base plus required lookup/constants already exceeds budget')
        count = min(len(requested), (allowed-fixed_bytes)//self.residual_unit_bytes)
        if cfg.get('residual_units_limit') is not None:
            count = min(count, max(0, int(cfg['residual_units_limit'])))
        self.selected_residual_units = requested[:count]
        self.requested_residual_units = requested
        self.predicted_budget_bytes = fixed_bytes+count*self.residual_unit_bytes

        for i, state in enumerate(self.states):
            selected_heads = [head for layer, head in self.selected_residual_units if layer == i]
            heads = torch.tensor(selected_heads, device=device, dtype=torch.long) if selected_heads else None
            source = state.visual_k.float()
            state.direction_mean = source.mean(-2, keepdim=True).half().contiguous()
            rotated = wht(source-state.direction_mean.float())
            norm = rotated.norm(dim=-1, keepdim=True)
            initial = (norm/(dim**.5)).clamp_min(1e-20)
            parts = (rotated/initial).reshape(-1, 4)
            book = self.book.float()
            indices = []
            for start in range(0, len(parts), 65536):
                part = parts[start:start+65536]
                distance = part.square().sum(-1, keepdim=True)+book.square().sum(-1)[None]-2*(part@book.T)
                indices.append(distance.argmin(-1).byte())
            state.direction_code = torch.cat(indices).reshape(*rotated.shape[:-1], 32).contiguous()
            template = book[state.direction_code.long()].reshape(rotated.shape)
            state.direction_gain = (norm/template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half().contiguous()
            if 'key_head_factors' in params:
                gain = params['key_head_factors'][i][None, :, None, None]
                state.direction_gain = (state.direction_gain.float()*gain).half().contiguous()
                if self.key_gain_mode == 'whole':
                    state.direction_mean = (state.direction_mean.float()*gain).half().contiguous()
            if selected_heads and self.residual_axis == 'k':
                # Target the original centered rotated source, while the base
                # uses the actual final calibrated FP16 row gain.
                actual_base = (template*state.direction_gain.float()).half()
                error = (rotated.index_select(1, heads).float()
                         - actual_base.index_select(1, heads).float())[0]
                state.v_residual_alpha = error.abs().mean(-1).half().contiguous()
                state.v_residual_code = pack1(error >= 0)
                self.refinements[i] = [state.v_residual_code, state.v_residual_alpha]
                for compact_index, head in enumerate(selected_heads):
                    self.residual_lookup[i, head] = compact_index
            del state.visual_k

            record = binary_centroids(state.visual_v.transpose(2, 3).contiguous(), group_size=self.vgroup)
            if 'value_head_factors' in params:
                record = fold(record, 1, params['value_head_factors'][i][None, :, None, None],
                              params['value_center_offsets'][i][None, :, None, None])
            if fg is not None:
                gain = fg[i].repeat_interleave(dim//self.functional_groups, -1)[None, :, :, None]
                offset = fo[i].repeat_interleave(dim//self.functional_groups, -1)[None, :, :, None]
                record = fold(record, 1, gain, offset)
            state.v_code, state.v_scale, state.v_min = record
            if selected_heads and self.residual_axis == 'v':
                decoded = decode_last_generic(*record, self.vgroup, 1).transpose(2, 3)
                error = (state.visual_v.index_select(1, heads).float()
                         - decoded.index_select(1, heads).float())[0]
                state.v_residual_alpha = error.abs().mean(-1).half().contiguous()
                state.v_residual_code = pack1(error >= 0)
                self.refinements[i] = [state.v_residual_code, state.v_residual_alpha]
                for compact_index, head in enumerate(selected_heads):
                    self.residual_lookup[i, head] = compact_index
            state.v_residual_lookup = self.residual_lookup[i]
            state.sparse_residual_axis = self.residual_axis
            del state.visual_v
        self.method = method
        owned = self.owned()
        actual = (owned['visual_base']+owned['visual_residual']+owned['selector_metadata']
                  + owned['visual_position']+self.encoder_parameter_bytes)
        if actual != self.predicted_budget_bytes:
            raise RuntimeError(f'Sparse V owner estimate differs: actual={actual}, predicted={self.predicted_budget_bytes}')
        self.actual_budget_bits = 16*actual/self.visual_full_bytes
        if self.actual_budget_bits > self.residual_budget_bits+1e-12:
            raise RuntimeError('Sparse V actual complete owner exceeds budget')

    def _visual_tensors(self, state):
        tensors = super()._visual_tensors(state)
        if getattr(self, 'visual_bias', None) is not None:
            tensors.append(self.visual_bias)
        return tensors

    def owned(self):
        result = super().owned()
        extra = unique_bytes([self.residual_lookup])
        result['selector_metadata'] += extra
        result['total_unique'] += extra
        result['visual_with_metadata_ratio'] = (result['visual_base']+result['visual_residual']+result['selector_metadata'])/self.visual_full_bytes
        return result

    def read(self, q, i):
        state = self.states[i]
        event = self.events[(self.turn, self.phase, self.pass_kind, i)]
        event['calls'] += 1
        event['query_tokens'] += q.shape[-2]
        event['base_visual_bytes'] += unique_bytes([state.direction_code, state.direction_gain,
            state.direction_mean, state.v_code, state.v_scale, state.v_min, self.book])
        if self.visual_bias is not None:
            event['base_visual_bytes'] += self.visual_bias[i].numel()*4
        residual = self.refinements[i]
        residual_bytes = unique_bytes(residual or [])
        event['residual_bytes'] += residual_bytes
        event['position_bytes'] += 8*(state.visual_pos.numel()+state.text_length)
        event['text_bytes'] += 2*state.text_length*state.text_k.shape[1]*128*state.text_k.element_size()
        cost = q.numel()*12+q.shape[1]*128*6+q.shape[1]*q.shape[-2]*4+state.v_residual_lookup.numel()*4
        event['selection_bytes'] += cost
        self.extra_read_bytes += cost+residual_bytes
        return attention(q, state, self.book, self.vgroup,
                         None if self.visual_bias is None else self.visual_bias[i])

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend=f'packed static selected-head {self.residual_axis.upper()} sign residual over global PQ K2/V1',
                      functional_v_sha256=self.functional_v_hash,
                      sparse_residual_axis=self.residual_axis,
                      sparse_residual_reference=('original centered WHT(source K) minus actual FP16(book * final calibrated gain); mean slot remains frozen' if self.residual_axis == 'k' else 'original source V minus actual calibrated V1 decode'),
                      sparse_residual_coordinate_system='centered rotated K' if self.residual_axis == 'k' else 'original V channels',
                      sparse_residual_selected_units=self.selected_residual_units,
                      sparse_residual_units=len(self.selected_residual_units),
                      sparse_residual_actual_budget_bits=self.actual_budget_bits,
                      sparse_v_selected_units=self.selected_residual_units,
                      sparse_v_requested_units=self.requested_residual_units,
                      sparse_v_units=len(self.selected_residual_units),
                      sparse_v_lookup_bytes=self.residual_lookup.numel()*4,
                      sparse_v_encoder_ranking_bytes=self.ranking_constant_bytes,
                      sparse_v_unit_bytes=self.residual_unit_bytes,
                      sparse_v_budget_bits=self.residual_budget_bits,
                      sparse_v_actual_budget_bits=self.actual_budget_bits,
                      sparse_v_budget_verified=True,
                      sparse_v_full_visual_support=True,
                      sparse_v_source_fp16_retained=False,
                      sparse_v_selector_scope='frozen calibration-ranked prefix shortened using actual image length/owner budget only; no query, reference answer or future',
                      sparse_v_residual_format=f'selected {self.residual_axis.upper()} KV heads only: sign1 per scalar + FP16 mean-absolute amplitude per token/128 channels; compact per-layer owner lookup')
        return result
