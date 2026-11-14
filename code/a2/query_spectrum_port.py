"""Query-weighted spectral K quality port, with real full-model free generation.

This bank materializes reconstructed FP16 K/V and retains its experimental
latent/basis tensors.  Every owner is charged.  It is not a packed deployment
candidate and does not claim a <=1.8-bit actual runtime cache.
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import torch

from masked_bank import A2Bank, unique_bytes
from quantizers import binary_centroids, affine_rtn
from fused_progressive import decode_last_generic
from affine_rate_bank import fold
from fast_wht import wht
from query_spectrum_math import hadamard_matrix


class QuerySpectrumPort(A2Bank):
    def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts, kivi_source, False)
        cfg = type(self).settings
        self.transform = cfg.get('transform', 'fwht')
        self.key_encoding = cfg.get('key_encoding', 'pq')
        self.rank = int(cfg.get('rank', 128))
        if not 1 <= self.rank <= 128:
            raise ValueError('Rank must be between 1 and 128')
        if self.key_encoding == 'pq' and self.rank != 128:
            raise ValueError('The fixed 256x4 PQ comparison is full rank; use projection for rank-dose experiments')
        if self.key_encoding not in ('pq', 'projection', 'roundtrip'):
            raise ValueError('Unknown key_encoding')
        if self.key_encoding == 'roundtrip' and self.rank != 128:
            raise ValueError('A numerical roundtrip must be full rank')
        self.gain_rule = cfg.get('gain_rule', 'original_space_norm')
        if self.gain_rule not in ('original_space_norm', 'transformed_space_norm'):
            raise ValueError('Declare original-space or transformed-space norm gains explicitly')
        self.projection_restore_norm = (self.key_encoding == 'projection' and
                                        bool(cfg.get('projection_restore_norm', False)))
        device = self.states[0].visual_k.device
        layers, heads = len(self.states), self.states[0].visual_k.shape[1]
        self.basis_encode = self.basis_decode = None
        self.basis_hash = None
        self.artifact_metadata = None
        self.book = None
        self.book_hash = None
        if cfg.get('direction_book_path'):
            path = Path(cfg['direction_book_path'])
            self.book_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.key_encoding == 'pq':
                self.book = torch.tensor(json.loads(path.read_text())['centers'], device=device, dtype=torch.float16).contiguous()
                if self.book.shape != (256, 4):
                    raise ValueError('The controlled PQ comparison requires the existing 256x4 book')
        if self.key_encoding == 'pq' and self.book is None:
            raise ValueError('PQ requires direction_book_path')
        dtype_name = cfg.get('basis_dtype', 'float32')
        dtype = {'float16': torch.float16, 'float32': torch.float32}.get(dtype_name)
        if dtype is None:
            raise ValueError('Basis dtype must be float16 or float32; no assumed FP8 decoder')
        if self.transform == 'random':
            generator = torch.Generator(device='cpu').manual_seed(int(cfg.get('basis_seed', 1212)))
            random = torch.randn(layers, heads, 128, 128, generator=generator, dtype=torch.float64)
            q, r = torch.linalg.qr(random)
            signs = r.diagonal(dim1=-2, dim2=-1).sign()
            q = q * torch.where(signs == 0, torch.ones_like(signs), signs).unsqueeze(-2)
            self.basis_encode = q[..., :self.rank].to(device=device, dtype=dtype).contiguous()
            self.basis_decode = q.transpose(-1, -2)[..., :self.rank, :].to(device=device, dtype=dtype).contiguous()
        elif self.transform == 'fwht':
            if self.rank != 128:
                # Truncated FWHT is an explicitly stored random-like subspace control.
                matrix = hadamard_matrix(128)
                self.basis_encode = matrix[:, :self.rank].to(device=device, dtype=dtype).contiguous()
                self.basis_decode = matrix.T[:self.rank].to(device=device, dtype=dtype).contiguous()
        else:
            path = Path(cfg['spectrum_artifact_path'])
            self.basis_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            artifact = torch.load(path, map_location='cpu', weights_only=False)
            if artifact.get('format') != 'trajkv_query_spectrum_v1':
                raise ValueError('Unrecognized spectrum artifact')
            transform = artifact['transforms'][self.transform]
            if tuple(transform['encode'].shape) != (layers, heads, 128, 128):
                raise ValueError('Basis architecture differs from the model')
            self.basis_encode = transform['encode'][..., :self.rank].to(device=device, dtype=dtype).contiguous()
            self.basis_decode = transform['decode'][..., :self.rank, :].to(device=device, dtype=dtype).contiguous()
            self.artifact_metadata = artifact['metadata']
        for matrix in (self.basis_encode, self.basis_decode):
            if matrix is not None and not bool(torch.isfinite(matrix).all()):
                raise ValueError('Nonfinite basis')
        self.direction_records, self.value_records, self.center_statistics = [], [], []
        self.reference_keys = []
        self.score_statistics = []
        self.metric_calls = defaultdict(int)
        self.collect_score_metrics = bool(cfg.get('collect_score_metrics', False))
        self.encoder_parameter_bytes = 0
        self.metadata_hash = None
        params = {}
        key_gain_mode = 'none'
        if cfg.get('metadata_factor_path'):
            path = Path(cfg['metadata_factor_path'])
            data = json.loads(path.read_text())
            self.metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if data['encoder_book_sha256'] != self.book_hash:
                raise ValueError('Metadata factors reference a different encoder book')
            if data['value_group_size'] != cfg.get('v_group_size', 128):
                raise ValueError('Value calibration group mismatch')
            key_gain_mode = data['key_gain_mode']
            for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets'):
                if data.get(key) is None:
                    continue
                parameter = torch.tensor(data[key], device=device, dtype=torch.float32)
                if parameter.shape != (layers, heads) or not bool(torch.isfinite(parameter).all()):
                    raise ValueError('Invalid metadata parameter')
                params[key] = parameter
            self.encoder_parameter_bytes = sum(t.numel() * t.element_size() for t in params.values())
        self.key_gain_mode = key_gain_mode
        vbits = int(cfg.get('bits_v', 1))
        if vbits not in (1, 2, 16):
            raise ValueError('V supports 1/2-bit comparison or FP16 oracle')
        vgroup = visual.numel() if cfg.get('v_group_size') == 'all' else int(cfg.get('v_group_size', 128))
        for layer, state in enumerate(self.states):
            original = state.visual_k
            source = original.float()
            mean = source.mean(-2, keepdim=True).half().contiguous()
            residual = source - mean.float()
            encode, decode = self._layer_basis(layer)
            transformed = self._encode(residual, encode)
            factor = (params['key_head_factors'][layer][None, :, None, None]
                      if 'key_head_factors' in params and cfg.get('apply_key_metadata', True) else None)
            if self.key_encoding == 'pq':
                initial = (transformed.norm(dim=-1, keepdim=True) / math.sqrt(128)).clamp_min(1e-20)
                parts = (transformed / initial).reshape(-1, 4)
                book = self.book.float()
                indices = []
                for start in range(0, len(parts), 65536):
                    chunk = parts[start:start + 65536]
                    distances = chunk.square().sum(-1, keepdim=True) + book.square().sum(-1)[None] - 2 * (chunk @ book.T)
                    indices.append(distances.argmin(-1).byte())
                codes = torch.cat(indices).reshape(*transformed.shape[:-1], 32).contiguous()
                template = book[codes.long()].reshape(transformed.shape)
                if self.gain_rule == 'original_space_norm':
                    unscaled = self._decode(template, decode)
                    gain = residual.norm(dim=-1, keepdim=True) / unscaled.norm(dim=-1, keepdim=True).clamp_min(1e-20)
                else:
                    gain = transformed.norm(dim=-1, keepdim=True) / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
                gain = gain.half().contiguous()
                if factor is not None:
                    # The baseline folds K calibration into the stored FP16 gain.
                    gain = (gain.float() * factor).half().contiguous()
                # Match the existing packed K reader's separate FP16 gain multiply.
                latent = (template * gain.float()).half()
                reconstructed = self._decode(latent.float(), decode)
                records = [codes, gain, mean]
            else:
                # Actual high-precision latent owner, not a hypothetical low-bit payload.
                latent = (transformed if self.key_encoding == 'roundtrip' else transformed.half()).contiguous()
                reconstructed = self._decode(latent.float(), decode)
                records = [latent, mean]
                if self.projection_restore_norm:
                    # Restore the original centered residual norm before the
                    # independently frozen head factor. Retain the actual F16
                    # gain owner; mean remains last for the whole-gain branch.
                    projection_gain = (residual.norm(dim=-1, keepdim=True) /
                        reconstructed.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half().contiguous()
                    if not bool(torch.isfinite(projection_gain).all()):
                        raise ValueError('Projection norm gain exceeds finite FP16 range')
                    reconstructed = reconstructed * projection_gain.float()
                    records = [latent, projection_gain, mean]
            if factor is not None:
                if self.key_encoding != 'pq':
                    reconstructed = reconstructed * factor
                if key_gain_mode == 'whole':
                    mean = (mean.float() * factor).half().contiguous()
                    records[-1] = mean
            reconstructed = (reconstructed + mean.float()).half().contiguous()
            state.visual_k = reconstructed
            self.direction_records.append(records)
            self.reference_keys.append(original if self.collect_score_metrics else None)
            statistics = {'layer': layer, 'key_mse': float((source - reconstructed.float()).square().mean()),
                'key_relative_mse': float((source - reconstructed.float()).square().sum() / source.square().sum().clamp_min(1e-20)),
                'mean_key_cosine': float(torch.nn.functional.cosine_similarity(source, reconstructed.float(), dim=-1).mean()),
                'mean_norm_ratio': float(reconstructed.float().norm(dim=-1).mean() / source.norm(dim=-1).mean().clamp_min(1e-20)),
                'max_abs_key_error': float((source - reconstructed.float()).abs().max())}
            self.center_statistics.append(statistics)
            if vbits == 16:
                self.value_records.append([])
                continue
            values = state.visual_v.transpose(2, 3).contiguous()
            record = binary_centroids(values, group_size=vgroup) if vbits == 1 else affine_rtn(values, vbits, vgroup)
            if 'value_head_factors' in params:
                record = fold(record, vbits, params['value_head_factors'][layer][None, :, None, None],
                              params['value_center_offsets'][layer][None, :, None, None])
            state.visual_v = decode_last_generic(*record, vgroup, vbits).transpose(2, 3).contiguous()
            self.value_records.append(list(record))
        self.method = method

    def _layer_basis(self, layer):
        if self.basis_encode is None:
            return None, None
        if self.basis_encode.ndim == 2:
            return self.basis_encode, self.basis_decode
        return self.basis_encode[layer][None], self.basis_decode[layer][None]

    @staticmethod
    def _encode(value, basis):
        return wht(value) if basis is None else value @ basis.float()

    @staticmethod
    def _decode(value, basis):
        return wht(value) if basis is None else value @ basis.float()

    def read(self, q, layer):
        if self.collect_score_metrics:
            key = (self.turn, self.phase, layer)
            limit = int(type(self).settings.get('score_metric_decode_steps', 8)) if self.phase.endswith('decode') else 1
            if self.metric_calls[key] < limit:
                state = self.states[layer]
                original = self.reference_keys[layer].float()
                current = state.visual_k.float()
                query = q[:, :, -1:].float()
                b, h, _, d = query.shape
                hkv = original.shape[1]
                grouped = query.reshape(b, hkv, h // hkv, d)
                full_scores = (grouped @ original.transpose(-1, -2)) / math.sqrt(d)
                errors = (grouped @ (current - original).transpose(-1, -2)) / math.sqrt(d)
                centered = full_scores - full_scores.mean(-1, keepdim=True)
                self.score_statistics.append({'layer': layer, 'turn': self.turn, 'phase': self.phase,
                    'sampled_queries': h, 'visual_tokens': original.shape[-2],
                    'score_error_sse': float(errors.square().sum()), 'score_reference_sse': float(full_scores.square().sum()),
                    'score_reference_centered_sse': float(centered.square().sum()),
                    'score_elements': errors.numel(),
                    'top1_same_fraction': float(((full_scores + errors).argmax(-1) == full_scores.argmax(-1)).float().mean()),
                    'scope': 'same student q, original visual K reference, last query position; no answer/correctness guarantee'})
                self.metric_calls[key] += 1
        return super().read(q, layer)

    def owned(self):
        result = super().owned()
        references = [v for group in self.direction_records + self.value_records for v in group]
        references += [value for value in (self.basis_encode, self.basis_decode, self.book) if value is not None]
        references += [value for value in self.reference_keys if value is not None]
        added = unique_bytes(references)
        result['spectrum_experiment_reference'] = added
        result['total_unique'] += added
        result['runtime'] = 'materialized_query_spectrum_quality_prototype'
        result['packed_visual_ratio'] = None
        result['materialized_visual_ratio'] = result['visual_base'] / self.visual_full_bytes
        result['encoded_reference_ratio'] = added / self.visual_full_bytes
        return result

    def result_metadata(self):
        result = super().result_metadata()
        result.update(backend='materialized query-weighted spectral quality port',
            materialized_quality_port=True, lowbit_deployment_candidate=False,
            spectrum_transform=self.transform, spectrum_rank=self.rank, spectrum_encoding=self.key_encoding,
            spectrum_artifact_sha256=self.basis_hash, direction_book_sha256=self.book_hash,
            metadata_calibration_sha256=self.metadata_hash, spectrum_artifact_metadata=self.artifact_metadata,
            spectrum_gain_rule=(self.gain_rule if self.key_encoding == 'pq' else
                'decoded_projection_original_residual_norm_match_FP16' if self.projection_restore_norm else 'none'),
            projection_restore_norm=self.projection_restore_norm,
            projection_norm_gain_bytes=(sum(record[-2].numel() * record[-2].element_size()
                for record in self.direction_records) if self.projection_restore_norm else 0),
            projection_norm_gain_scope=('per-token/KV-head centered residual norm restored before frozen K head factor; actual FP16 gain retained and counted'
                if self.projection_restore_norm else 'disabled'),
            key_gain_mode=self.key_gain_mode,
            encoder_calibration_parameters_bytes=self.encoder_parameter_bytes,
            spectrum_key_statistics=self.center_statistics, spectrum_score_statistics=self.score_statistics,
            transform_scope='source-centered row-vector y=(k-mu)A and reconstructed y_hat B+mu; text KV unchanged',
            prototype_bits_scope='actual owner includes materialized FP16 KV, retained latent/reference tensors and encode/decode bases; no packed efficiency claim')
        return result
