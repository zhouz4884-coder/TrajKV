"""Channel-group V metadata calibration on a frozen calibrated K2/V1 base."""
import hashlib
import json
from pathlib import Path
import torch
from affine_rate_bank import fold
from masked_bank import unique_bytes
from metadata_direction_bank import MetadataDirectionBank
from functional_v_attention import attention


class FunctionalVBank(MetadataDirectionBank):
    def _initialize(self, *args, **kwargs):
        super()._initialize(*args, **kwargs)
        cfg = type(self).settings
        path = Path(cfg['functional_v_factor_path'])
        fit = json.loads(path.read_text())
        if fit['schema'] != 'functional_v_v1':
            raise ValueError('Unknown functional V fit schema')
        if fit['base_metadata_sha256'] != self.metadata_hash:
            raise ValueError('Functional V fit used a different frozen base')
        if fit['encoder_book_sha256'] != hashlib.sha256(Path(cfg['direction_book_path']).read_bytes()).hexdigest():
            raise ValueError('Functional V encoder book mismatch')
        if fit['value_group_size'] != cfg.get('v_group_size', 128):
            raise ValueError('Functional V quantization window mismatch')
        device = self.states[0].direction_gain.device
        self.functional_v_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.functional_groups = int(fit['channel_groups'])
        if 128 % self.functional_groups:
            raise ValueError('V channel groups must divide 128')
        expected = (len(self.states), self.states[0].text_k.shape[1], self.functional_groups)
        gain = torch.tensor(fit['value_channel_factors'], device=device, dtype=torch.float32)
        offset = torch.tensor(fit['value_channel_offsets'], device=device, dtype=torch.float32)
        if gain.shape != expected or offset.shape != expected or not bool(torch.isfinite(gain).all() & torch.isfinite(offset).all()):
            raise ValueError('Invalid V group factors')
        if not bool((gain > 0).all()):
            raise ValueError('V gains must be positive')
        self.functional_folded_bytes = 4*(gain.numel()+offset.numel())
        self.encoder_parameter_bytes += self.functional_folded_bytes
        self.visual_bias = None
        if fit['visual_logit_bias'] is not None:
            self.visual_bias = torch.tensor(fit['visual_logit_bias'], device=device, dtype=torch.float32).contiguous()
            if self.visual_bias.shape != expected[:2] or not bool(torch.isfinite(self.visual_bias).all()):
                raise ValueError('Invalid visual logit bias')
        for i, state in enumerate(self.states):
            g = gain[i].repeat_interleave(128//self.functional_groups, -1)[None, :, :, None]
            o = offset[i].repeat_interleave(128//self.functional_groups, -1)[None, :, :, None]
            record = fold((state.v_code, state.v_scale, state.v_min), 1, g, o)
            state.v_code, state.v_scale, state.v_min = record

    def _visual_tensors(self, state):
        tensors = super()._visual_tensors(state)
        if getattr(self, 'visual_bias', None) is not None:
            tensors.append(self.visual_bias)
        return tensors

    def read(self, q, i):
        state = self.states[i]
        event = self.events[(self.turn, self.phase, self.pass_kind, i)]
        event['calls'] += 1
        event['query_tokens'] += q.shape[-2]
        payload = [state.direction_code, state.direction_gain, state.direction_mean,
                   state.v_code, state.v_scale, state.v_min]
        event['base_visual_bytes'] += unique_bytes(payload) + self.book.numel()*self.book.element_size()
        if self.visual_bias is not None:
            event['base_visual_bytes'] += self.visual_bias[i].numel()*self.visual_bias.element_size()
        event['position_bytes'] += 8*(state.visual_pos.numel()+state.text_length)
        event['text_bytes'] += 2*state.text_length*state.text_k.shape[1]*128*state.text_k.element_size()
        cost = q.numel()*12 + q.shape[1]*128*6 + q.shape[1]*q.shape[-2]*4
        event['selection_bytes'] += cost
        self.extra_read_bytes += cost
        return attention(q, state, self.book, self.vgroup,
                         None if self.visual_bias is None else self.visual_bias[i])

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend='packed global PQ K2, folded grouped V1 metadata, visual-only logit bias',
                      functional_v_sha256=self.functional_v_hash,
                      functional_v_channel_groups=self.functional_groups,
                      functional_v_folded_encoder_bytes=self.functional_folded_bytes,
                      functional_v_k_fixed=True,
                      visual_bias_bytes=0 if self.visual_bias is None else self.visual_bias.numel()*4,
                      visual_bias_scope='one FP32 parameter per layer/KV head, shared by its GQA heads; counted in actual visual owner',
                      functional_v_parameter_scope='V parameters charged as encoder constants; bias retained and charged in runtime owner, not charged twice')
        return result
