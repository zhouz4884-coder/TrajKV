"""Existing packed K2/V1 format; query-weighted *encoder-only* PQ assignment.

No online query, future query, reference answer, decoder change, or additional
serving read is used.  FP16 metric constants are separately charged once per
cache alongside the existing encoder-only metadata factors.
"""
import hashlib
import json
from pathlib import Path
import torch
from masked_bank import A2Bank
from metadata_direction_bank import MetadataDirectionBank
from quantizers import binary_centroids
from fast_wht import wht
from affine_rate_bank import fold
from metric_pq_math import metric_codes


class MetricPQBank(MetadataDirectionBank):
    def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts, kivi_source, False)
        cfg = type(self).settings
        device = self.states[0].visual_k.device
        nv, layers, heads = visual.numel(), len(self.states), self.states[0].visual_k.shape[1]
        book_path = Path(cfg['direction_book_path'])
        self.book = torch.tensor(json.loads(book_path.read_text())['centers'], device=device, dtype=torch.float16).contiguous()
        if self.book.shape != (256, 4):
            raise ValueError('Metric assignment uses the existing 256x4 book')
        metric_path = Path(cfg['metric_pq_artifact_path'])
        artifact = torch.load(metric_path, map_location='cpu', weights_only=False)
        if artifact.get('format') != 'trajkv_metric_pq_v1':
            raise ValueError('Unrecognized metric-PQ artifact')
        metric_cpu = artifact['metric_blocks']
        if tuple(metric_cpu.shape) != (layers, heads, 32, 4, 4) or metric_cpu.dtype != torch.float16:
            raise ValueError('Metric architecture/dtype mismatch')
        if not bool(torch.isfinite(metric_cpu).all()):
            raise ValueError('Nonfinite encoder metric')
        self.metric_pq_hash = hashlib.sha256(metric_path.read_bytes()).hexdigest()
        self.metric_pq_metadata = artifact['metadata']
        self.metric_encoder_bytes = metric_cpu.numel() * metric_cpu.element_size()
        self.decoder_books = None
        self.decoder_hash = None
        self.headwise_budget_ok = False
        self.encoder_parameter_bytes = self.metric_encoder_bytes
        self.vbits, self.v_channel = 1, True
        self.vgroup = nv if cfg.get('v_group_size') == 'all' else 128
        if cfg.get('bits_v', 1) != 1 or cfg.get('v_layout', 'channel') != 'channel':
            raise ValueError('Current metric-PQ format keeps channel V1')
        self.metadata_hash, self.key_gain_mode = None, 'none'
        params = {}
        if cfg.get('metadata_factor_path'):
            path = Path(cfg['metadata_factor_path'])
            data = json.loads(path.read_text())
            self.metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if data['encoder_book_sha256'] != hashlib.sha256(book_path.read_bytes()).hexdigest():
                raise ValueError('Wrong encoder book for metadata factors')
            if data['value_group_size'] != cfg.get('v_group_size', 128):
                raise ValueError('Calibration V group mismatch')
            self.key_gain_mode = data['key_gain_mode']
            if self.key_gain_mode not in ('none', 'centered', 'whole'):
                raise ValueError('Invalid K gain mode')
            for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets'):
                if data.get(key) is None:
                    continue
                parameter = torch.tensor(data[key], device=device, dtype=torch.float32)
                if parameter.shape != (layers, heads) or not bool(torch.isfinite(parameter).all()):
                    raise ValueError('Invalid metadata parameter')
                if 'factors' in key and not bool((parameter > 0).all()):
                    raise ValueError('Metadata gains must be positive')
                params[key] = parameter
            parameter_bytes = sum(t.numel() * t.element_size() for t in params.values())
            if parameter_bytes != 4 * data['parameter_count']:
                raise ValueError('Metadata parameter accounting mismatch')
            self.encoder_parameter_bytes += parameter_bytes
        self.metric_zero_fallback_blocks = 0
        self.metric_assignment_statistics = []
        book = self.book.float()
        for layer, state in enumerate(self.states):
            source = state.visual_k.float()
            state.direction_mean = source.mean(-2, keepdim=True).half().contiguous()
            rotated = wht(source - state.direction_mean.float())
            norm = rotated.norm(dim=-1, keepdim=True)
            initial = (norm / (128 ** .5)).clamp_min(1e-20)
            metric = metric_cpu[layer].to(device=device)
            codes, zero_count = metric_codes(rotated / initial, book, metric,
                max_distance_elements=int(cfg.get('metric_max_distance_elements', 16777216)))
            state.direction_code = codes
            self.metric_zero_fallback_blocks += zero_count
            template = book[codes.long()].reshape(rotated.shape)
            # Exactly the original stored norm-match rule after code assignment.
            state.direction_gain = (norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half().contiguous()
            if 'key_head_factors' in params:
                gain = params['key_head_factors'][layer][None, :, None, None]
                state.direction_gain = (state.direction_gain.float() * gain).half().contiguous()
                if self.key_gain_mode == 'whole':
                    state.direction_mean = (state.direction_mean.float() * gain).half().contiguous()
            if cfg.get('metric_assignment_diagnostics', False):
                error = template * state.direction_gain.float() - rotated
                self.metric_assignment_statistics.append({'layer': layer, 'rotated_key_mse': float(error.square().mean()),
                    'scope': 'encoder residual after stored norm gain; no answer prediction'})
            del state.visual_k
            record = binary_centroids(state.visual_v.transpose(2, 3).contiguous(), group_size=self.vgroup)
            if 'value_head_factors' in params:
                record = fold(record, 1, params['value_head_factors'][layer][None, :, None, None],
                              params['value_center_offsets'][layer][None, :, None, None])
            for key, tensor in zip(('code', 'scale', 'min'), record):
                setattr(state, 'v_' + key, tensor)
            del state.visual_v
        self.method = method

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend='unchanged packed K2 product-code reader; calibration-query-metric encoder assignment',
            metric_pq_artifact_sha256=self.metric_pq_hash, metric_pq_artifact_metadata=self.metric_pq_metadata,
            metric_pq_encoder_bytes=self.metric_encoder_bytes,
            metric_pq_zero_fallback_blocks=self.metric_zero_fallback_blocks,
            metric_pq_assignment_statistics=self.metric_assignment_statistics,
            encoding_assignment='calibration-only block-diagonal query-weighted nearest word at source_norm/sqrt128 initial amplitude; original final norm-matching gain',
            metric_pq_optimization_scope='assignment is a block metric proxy; final row renormalization can change the full metric objective',
            metric_pq_constant_scope='FP16 encoder-only metric blocks separately counted once per cache; not retained or read by serving attention',
            serving_reader_unchanged=True, online_query_metric_access=False)
        return result
