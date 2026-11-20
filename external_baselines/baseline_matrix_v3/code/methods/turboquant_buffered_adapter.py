"""0xSero TurboQuant visual port with the source prefill's recent window.

The window is measured on original logical prefix positions, not on a count of
visual candidates. Text KV remains exact outside this owner. No original T0
source or numerical backend is changed; the FP16 visual tail joins the same
softmax as compressed visual keys and exact text keys.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .turboquant_adapter import (TurboQuantAdapter, TurboQuantOwner, grouped_attention,
                                 storage_bytes, UPSTREAM_REPO, UPSTREAM_COMMIT, PAPER)


@dataclass
class BufferedTurboQuantOwner:
    quantized_owner: TurboQuantOwner | None
    k_tail: torch.Tensor
    v_tail: torch.Tensor
    tail_positions: torch.Tensor
    shape: tuple
    source_dtype: torch.dtype
    logical_prefix_length: int
    recent_window: int
    value_bits: int
    layer_idx: int

    @property
    def positions(self):
        if self.quantized_owner is None: return self.tail_positions
        if not self.tail_positions.numel(): return self.quantized_owner.positions
        # This is an ephemeral validation view, not a retained duplicate index.
        return torch.cat((self.quantized_owner.positions, self.tail_positions))

    def named_tensors(self, include_constants=True):
        result = {'k_tail': self.k_tail, 'v_tail': self.v_tail, 'tail.positions': self.tail_positions}
        if self.quantized_owner is not None:
            result.update({'quantized.' + name: tensor for name, tensor in
                           self.quantized_owner.named_tensors(include_constants=include_constants).items()})
        return result

    def owner_report(self):
        tensors = self.named_tensors()
        constants = storage_bytes({name: tensor for name, tensor in tensors.items() if 'encoder_constants.' in name})['total_bytes']
        positions = storage_bytes({name: tensor for name, tensor in tensors.items() if name.endswith('positions')})['total_bytes']
        all_bytes = storage_bytes(tensors)['total_bytes']
        scalar_count = 2 * math.prod(self.shape)
        return {'method': f'TurboQuant_0xSero_K3V{self.value_bits}_logical_recent{self.recent_window}_visual_port',
                'upstream_repo': UPSTREAM_REPO, 'upstream_commit': UPSTREAM_COMMIT, 'method_paper': PAPER,
                'faithfulness_status': 'THIRD_PARTY_CODEC_AND_SOURCE_PREFILL_BUFFER_VISUAL_SCOPE_PORT',
                'reader': 'original_packed_key_reader_plus_exact_visual_tail_and_text_single_softmax',
                'bits_nominal_quantized_payload': (3 + self.value_bits) / 2,
                'recent_window_logical_positions': self.recent_window,
                'logical_prefix_length': self.logical_prefix_length,
                'tail_boundary': self.logical_prefix_length - self.recent_window,
                'visual_FP16_tail_tokens': self.k_tail.shape[-2],
                'quantized_visual_tokens': 0 if self.quantized_owner is None else self.quantized_owner.shape[-2],
                'source_recent_buffer_measured_in': 'logical_prefix_tokens_including_nonvisual_positions',
                'rolling_decode_scope': 'visual prefix is frozen; all text KV stays FP16 under common matrix protocol',
                'source_dtype': str(self.source_dtype), 'codec_numerical_variant': 'UPSTREAM_FP32_ENCODER_AND_METADATA',
                'stored_visual_KV_bytes': all_bytes - constants,
                'all_owner_bytes_including_constants': all_bytes, 'encoder_constants_bytes': constants,
                'positions_bytes': positions,
                'FP16_visual_tail_bytes': self.k_tail.untyped_storage().nbytes() + self.v_tail.untyped_storage().nbytes(),
                'full_kv_byte_denominator': scalar_count * torch.empty((), dtype=self.source_dtype).element_size(),
                'effective_bits_including_constants': 8 * all_bytes / scalar_count,
                'no_full_native_prefix_alias': True}

    def materialize(self, dtype=None):
        """Explicit diagnostic decoder only; packed attend never calls it."""
        dtype = dtype or self.source_dtype
        if self.quantized_owner is None:
            return self.k_tail.to(dtype), self.v_tail.to(dtype)
        k, v = self.quantized_owner.materialize(dtype)
        return torch.cat((k, self.k_tail.to(dtype)), -2), torch.cat((v, self.v_tail.to(dtype)), -2)

    def attend(self, query, *, exact_keys=None, exact_values=None, exact_positions=None,
               query_positions=None, scale=None, backend='upstream_triton_score'):
        if backend not in ('upstream_triton_score', 'torch_dequant'):
            raise ValueError('Unrecognized explicit buffered TurboQuant reader backend: ' + backend)
        if exact_keys is not None and (exact_values is None or exact_positions is None):
            raise ValueError('Exact keys require exact values and original positions')
        if self.quantized_owner is None:
            # A prefix wholly inside the original recent window contains no
            # quantized owner. This is an explicit all-FP16 legal source case.
            return grouped_attention(query, self.k_tail, self.v_tail, self.tail_positions,
                exact_keys=exact_keys, exact_values=exact_values, exact_positions=exact_positions,
                query_positions=query_positions, scale=scale)
        keys, values, positions = exact_keys, exact_values, exact_positions
        if self.k_tail.shape[-2]:
            keys = self.k_tail if keys is None else torch.cat((self.k_tail, keys), -2)
            values = self.v_tail if values is None else torch.cat((self.v_tail, values), -2)
            tp = self.tail_positions
            if positions is not None and positions.ndim != 1:
                tp = tp.view(1, 1, -1).expand(query.shape[0], self.shape[1], -1)
            positions = tp if positions is None else torch.cat((tp, positions), -1)
        return self.quantized_owner.attend(query, exact_keys=keys, exact_values=values,
            exact_positions=positions, query_positions=query_positions, scale=scale, backend=backend)


class TurboQuantBufferedAdapter(TurboQuantAdapter):
    def __init__(self, *, recent_window=128, value_bits=2, key_bits=3, group_size=32, seed=42):
        super().__init__(value_bits=value_bits, key_bits=key_bits, group_size=group_size, seed=seed)
        if not isinstance(recent_window, int) or recent_window < 0:
            raise ValueError('recent_window must be a nonnegative logical-prefix length')
        self.recent_window = recent_window

    @torch.no_grad()
    def encode(self, keys, values, *, positions=None, layer_idx=0, logical_prefix_length=None, **kwargs):
        if keys.ndim != 4 or values.shape != keys.shape or not keys.shape[-2]:
            raise ValueError('Expected nonempty matching [B,Hkv,Nvisual,D] K/V')
        if positions is None: positions = torch.arange(keys.shape[-2], device=keys.device)
        positions = torch.as_tensor(positions, dtype=torch.int64, device=keys.device)
        if positions.ndim != 1 or positions.numel() != keys.shape[-2] or not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError('Visual positions must be sorted unique original logical positions')
        boundary = int(positions[-1]) + 1 if logical_prefix_length is None else int(logical_prefix_length)
        if boundary <= int(positions[-1]): raise ValueError('Logical prefix must cover visual positions')
        tail = positions >= boundary - self.recent_window
        quant = ~tail
        owner = super().encode(keys[:, :, quant], values[:, :, quant], positions=positions[quant], layer_idx=layer_idx) if bool(quant.any()) else None
        return BufferedTurboQuantOwner(owner, keys[:, :, tail].detach().contiguous().clone(),
            values[:, :, tail].detach().contiguous().clone(), positions[tail].detach().clone(), tuple(keys.shape),
            keys.dtype, boundary, self.recent_window, self.value_bits, layer_idx)


def create_adapter(**kwargs): return TurboQuantBufferedAdapter(**kwargs)
