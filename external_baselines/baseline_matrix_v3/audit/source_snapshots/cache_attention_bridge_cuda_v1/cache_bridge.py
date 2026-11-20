"""Minimal logical-position Cache for visual owners plus exact text KV.

Only batch=1, append-only, non-sliding decoder inference is supported. Native
models still produce Q/K normalization, RoPE and every multimodal feature.
This object stores no full-prefix/debug KV view after construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import inspect
import math
from typing import Any
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def _fail(message):
    raise ValueError("UNSUPPORTED_CACHE_PROTOCOL: " + message)


def _repeat_heads(tensor, query_heads):
    if query_heads % tensor.shape[1]:
        _fail("nonintegral GQA ratio")
    return tensor.repeat_interleave(query_heads // tensor.shape[1], dim=1)


@dataclass
class FP16VisualOwner:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @classmethod
    def encode(cls, keys, values, positions):
        return cls(keys.detach().contiguous().clone(), values.detach().contiguous().clone(),
                   positions.detach().to(device=keys.device, dtype=torch.int64).clone())

    def named_tensors(self, include_constants=True):
        return {"keys": self.keys, "values": self.values, "positions": self.positions}

    owner_tensors = named_tensors

    def materialize(self, dtype=None):
        return self.keys.to(dtype or self.keys.dtype), self.values.to(dtype or self.values.dtype)

    def owner_report(self):
        return {"method": "FP16_visual_native_sdpa_bridge", "reader": "merge_original_order_then_original_HF_SDPA",
                "stored_visual_KV_bytes": sum(t.untyped_storage().nbytes() for t in self.named_tensors().values()),
                "positions_bytes": self.positions.untyped_storage().nbytes(),
                "temporary_reconstructed_full_KV": True}


class HybridVisualLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = False

    def __init__(self, text_k, text_v, text_positions, visual_owner, original_visual_positions, logical_length):
        super().__init__()
        self.keys, self.values = text_k, text_v
        self.text_positions = text_positions
        self.visual_owner = visual_owner
        self.original_visual_positions = original_visual_positions
        self.logical_length = int(logical_length)
        self.frozen_prefix_length = int(logical_length)
        self.last_update_start, self.last_update_length = int(logical_length), 0
        self.dtype, self.device = text_k.dtype, text_k.device
        self.is_initialized = True

    @property
    def visual_positions(self):
        return getattr(self.visual_owner, "positions", self.original_visual_positions)

    @property
    def query_positions(self):
        return torch.arange(self.last_update_start, self.last_update_start + self.last_update_length,
                            device=self.device, dtype=torch.int64)

    def lazy_initialization(self, key_states, value_states):
        _fail("layers must be constructed from a complete native prefix")

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.shape != value_states.shape or key_states.ndim != 4 or key_states.shape[0] != 1:
            _fail("update requires matching batch-one [B,Hkv,Q,D] K/V")
        if key_states.shape[1] != self.keys.shape[1] or key_states.shape[-1] != self.keys.shape[-1]:
            _fail("KV geometry changed")
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype or key_states.device != self.device:
            _fail("text KV dtype/device changed")
        cache_args = dict(kwargs)
        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                _fail("unknown positional cache update arguments")
            cache_args.update(args[0])
        if "cache_kwargs" in cache_args:
            nested = cache_args.pop("cache_kwargs")
            if nested is not None:
                cache_args.update(nested)
        unknown = set(cache_args) - {"cache_position", "cos", "sin"}
        if unknown:
            _fail("unknown cache update arguments: " + repr(sorted(unknown)))
        n = key_states.shape[-2]
        positions = torch.arange(self.logical_length, self.logical_length + n, device=self.device)
        if cache_args.get("cache_position") is not None:
            supplied = cache_args["cache_position"].to(self.device).reshape(-1)
            if not torch.equal(supplied, positions):
                _fail("cache_position is not the next logical append range")
        self.last_update_start, self.last_update_length = self.logical_length, n
        self.keys = torch.cat((self.keys, key_states.detach()), dim=-2)
        self.values = torch.cat((self.values, value_states.detach()), dim=-2)
        self.text_positions = torch.cat((self.text_positions, positions))
        self.logical_length += n
        return self.keys, self.values

    def get_seq_length(self):
        return self.logical_length

    def get_mask_sizes(self, query_length):
        return self.logical_length + query_length, 0

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        target = self.logical_length + max_length if max_length < 0 else max_length
        if target < self.frozen_prefix_length:
            _fail("cannot crop immutable visual prefix")
        if target >= self.logical_length:
            return
        kept = self.text_positions < target
        self.keys = self.keys[:, :, kept].contiguous()
        self.values = self.values[:, :, kept].contiguous()
        self.text_positions = self.text_positions[kept].contiguous()
        self.logical_length = int(target)
        self.last_update_start, self.last_update_length = int(target), 0

    def reset(self):
        _fail("reset would discard an immutable visual owner; construct a new cache")

    def reorder_cache(self, beam_idx):
        if beam_idx.numel() != 1 or int(beam_idx[0]) != 0:
            _fail("beam/batch reordering is unsupported")

    def offload(self):
        _fail("offloading is unsupported")

    def prefetch(self):
        _fail("offloading is unsupported")


class HybridVisualCache(Cache):
    def __init__(self, native_prefix_cache, visual_positions, visual_owners=None, *,
                 reader_backend=None, consume_native=True):
        if hasattr(native_prefix_cache, "is_sliding") and any(native_prefix_cache.is_sliding):
            _fail("sliding/hybrid attention is unsupported")
        if hasattr(native_prefix_cache, "layers"):
            pairs = [(layer.keys, layer.values) for layer in native_prefix_cache.layers]
        elif hasattr(native_prefix_cache, "key_cache"):
            pairs = list(zip(native_prefix_cache.key_cache, native_prefix_cache.value_cache))
        else:
            pairs = list(native_prefix_cache)
        if not pairs:
            _fail("empty native prefix")
        prefix_length = pairs[0][0].shape[-2]
        device = pairs[0][0].device
        vp = torch.as_tensor(visual_positions, device=device, dtype=torch.int64).detach().clone()
        if vp.ndim != 1 or not vp.numel() or vp.unique().numel() != vp.numel():
            _fail("visual_positions must be nonempty unique original 1D positions")
        if int(vp.min()) < 0 or int(vp.max()) >= prefix_length or not torch.equal(vp, vp.sort().values):
            _fail("visual positions must be sorted and inside native prefix")
        if visual_owners is not None and len(visual_owners) != len(pairs):
            _fail("visual owner layer count mismatch")
        text_mask = torch.ones(prefix_length, device=device, dtype=torch.bool)
        text_mask[vp] = False
        tp = torch.arange(prefix_length, device=device)[text_mask]
        layers = []
        for index, (k, v) in enumerate(pairs):
            if k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1 or k.shape[-2] != prefix_length:
                _fail("native prefix must have uniform non-sliding batch-one cache layers")
            if k.dtype != torch.float16 or v.dtype != torch.float16:
                _fail("this acceptance scope requires FP16 native weights and text KV")
            if k.device != device:
                _fail("multi-device cache is unsupported")
            owner = (FP16VisualOwner.encode(k.index_select(-2, vp), v.index_select(-2, vp), vp)
                     if visual_owners is None else visual_owners[index])
            owner_tensors = (owner.named_tensors() if hasattr(owner, "named_tensors") else owner.owner_tensors())
            native_ptrs = {k.untyped_storage().data_ptr(), v.untyped_storage().data_ptr()}
            if any(t.untyped_storage().data_ptr() in native_ptrs for t in owner_tensors.values()):
                _fail("visual owner aliases full native prefix storage")
            owner_positions = getattr(owner, "positions", vp)
            if not bool(torch.isin(owner_positions.to(device), vp).all()):
                _fail("owner contains positions outside declared visual candidates")
            layers.append(HybridVisualLayer(k[:, :, text_mask].detach().contiguous(),
                                            v[:, :, text_mask].detach().contiguous(), tp.clone(), owner, vp, prefix_length))
        super().__init__(layers=layers, offloading=False)
        self.visual_positions = vp
        self.prefix_length = prefix_length
        self.reader_backend = reader_backend
        self.reader_implementation = "method_owner_reader_plus_exact_text"
        if consume_native:
            if hasattr(native_prefix_cache, "layers"):
                for layer in native_prefix_cache.layers:
                    layer.keys = layer.values = None
                    layer.is_initialized = False
                native_prefix_cache.layers.clear()
            elif hasattr(native_prefix_cache, "key_cache"):
                native_prefix_cache.key_cache.clear()
                native_prefix_cache.value_cache.clear()
            elif isinstance(native_prefix_cache, list):
                native_prefix_cache.clear()
            else:
                _fail("cannot consume immutable prefix container; use DynamicCache/list or consume_native=False")

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].logical_length if layer_idx < len(self.layers) else 0

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_max_cache_shape(self, layer_idx=0):
        return -1

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    @property
    def max_batch_size(self):
        return 1

    @property
    def max_cache_len(self):
        return -1

    def batch_repeat_interleave(self, repeats):
        if repeats != 1:
            _fail("batch expansion is unsupported")

    def batch_select_indices(self, indices):
        if indices.numel() != 1 or int(indices[0]) != 0:
            _fail("batch expansion/selection is unsupported")

    def named_tensors(self):
        result = {"original_visual_positions": self.visual_positions}
        for index, layer in enumerate(self.layers):
            base = f"layers.{index}."
            result.update({base + "text_keys": layer.keys, base + "text_values": layer.values,
                           base + "text_positions": layer.text_positions})
            owner = layer.visual_owner
            tensors = owner.named_tensors() if hasattr(owner, "named_tensors") else owner.owner_tensors()
            result.update({base + "visual." + name: tensor for name, tensor in tensors.items()})
        return result

    @staticmethod
    def _mask(attention_mask, q, layer):
        n, length = q.shape[-2], layer.logical_length
        causal = torch.arange(length, device=q.device)[None, :] <= layer.query_positions[:, None]
        if attention_mask is None:
            return causal[None, None], True
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
            _fail("only native 4D tensor masks or None are supported")
        if attention_mask.shape[0] != 1 or attention_mask.shape[1] not in (1, q.shape[1]) or attention_mask.shape[-2:] != (n, length):
            _fail("native mask shape does not cover the logical complete key axis")
        mask = attention_mask.to(q.device)
        if mask.dtype == torch.bool:
            standard = torch.equal(mask, causal[None, None].expand_as(mask))
        elif mask.is_floating_point():
            if not bool((torch.isfinite(mask) | torch.isneginf(mask)).all()):
                _fail("nonfinite attention bias")
            c = causal[None, None].expand_as(mask)
            blocked = torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min / 2)
            standard = bool((mask[c] == 0).all() and blocked[~c].all())
        else:
            _fail("attention mask must be boolean or floating point")
        return mask, standard

    def read(self, layer_idx, q, module, attention_mask, scaling, original_sdpa, **kwargs):
        layer = self.layers[layer_idx]
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-2] != layer.last_update_length:
            _fail("Q does not match this layer's last logical cache append")
        if kwargs.get("dropout", 0.0) != 0 or getattr(module, "training", False):
            _fail("only inference without attention dropout is supported")
        if kwargs.get("sliding_window") not in (None, 0):
            _fail("sliding attention is unsupported")
        if kwargs.get("is_causal") is False or not getattr(module, "is_causal", True):
            _fail("bidirectional language attention is unsupported")
        mask, standard_causal = self._mask(attention_mask, q, layer)
        scale = q.shape[-1] ** -0.5 if scaling is None else scaling
        owner = layer.visual_owner
        if isinstance(owner, FP16VisualOwner):
            # Reconstruct only for this call, preserving native precision/order.
            shape = (1, layer.keys.shape[1], layer.logical_length, layer.keys.shape[-1])
            k = torch.empty(shape, device=q.device, dtype=layer.dtype)
            v = torch.empty_like(k)
            k[:, :, owner.positions] = owner.keys
            v[:, :, owner.positions] = owner.values
            k[:, :, layer.text_positions] = layer.keys
            v[:, :, layer.text_positions] = layer.values
            self.reader_implementation = "merge_original_order_then_original_HF_SDPA"
            native_mask = attention_mask
            if native_mask is None and q.shape[-2] > 1:
                native_mask = mask
            return original_sdpa(module, q, k, v, native_mask, scaling=scale, **kwargs)
        if hasattr(owner, "attend"):
            if not standard_causal:
                _fail("this method's attend reader supports standard causal masks only; extra mask/bias rejected")
            call_kwargs = dict(exact_keys=layer.keys, exact_values=layer.values,
                               exact_positions=layer.text_positions, query_positions=layer.query_positions, scale=scale)
            if self.reader_backend is not None and "backend" in inspect.signature(owner.attend).parameters:
                call_kwargs["backend"] = self.reader_backend
            output = owner.attend(q, **call_kwargs)
            return output.transpose(1, 2).contiguous(), None
        if not (hasattr(owner, "read") and hasattr(owner, "weighted_value")):
            _fail("owner has neither attend nor calibrated read/weighted_value interface")
        # CalibQuant read already includes both scale and visual-only calibration.
        visual_logits = owner.read(q, softmax_scale=scale).float()
        text_logits = (q.float() @ _repeat_heads(layer.keys, q.shape[1]).float().transpose(-1, -2)) * scale
        vp = layer.visual_positions
        if vp.ndim != 1:
            _fail("calibrated read requires dense ordered visual positions")
        visual_mask = mask.index_select(-1, vp)
        text_mask = mask.index_select(-1, layer.text_positions)
        if mask.dtype == torch.bool:
            visual_logits = visual_logits.masked_fill(~visual_mask, -torch.inf)
            text_logits = text_logits.masked_fill(~text_mask, -torch.inf)
        else:
            visual_logits = visual_logits + visual_mask
            text_logits = text_logits + text_mask
        probabilities = torch.softmax(torch.cat((visual_logits, text_logits), dim=-1), dim=-1).to(q.dtype)
        count = visual_logits.shape[-1]
        output = owner.weighted_value(probabilities[..., :count])
        output = output + probabilities[..., count:] @ _repeat_heads(layer.values, q.shape[1])
        return output.transpose(1, 2).contiguous(), None
