# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA kvpress PyramidKV selector projected to a visual-only cache budget.

Score and layer budget equations are ported from the pinned NVIDIA source.
Only visual candidates are evicted, while all text remains exact outside owner.
The observation window and query-known condition are explicit, mandatory data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import torch
from torch.nn import functional as F

from .turboquant_adapter import grouped_attention, storage_bytes

UPSTREAM_REPO = "https://github.com/NVIDIA/kvpress"
UPSTREAM_COMMIT = "6cf5e324be90ea9cedce66e843699b6dc655c0bd"
PAPER = "https://arxiv.org/abs/2406.02069"


def pyramid_layer_budget(q_len, compression_ratio, window_size, beta, layer_idx, num_layers):
    """Exact NVIDIA get_layer_budget equation (including documented SnapKV fallback)."""
    max_capacity_prompt = window_size + q_len * (1 - compression_ratio)
    min_num = (max_capacity_prompt - window_size) / beta
    max_num = (max_capacity_prompt - window_size) * 2 - min_num
    if max_num >= q_len - window_size:
        max_num = q_len - window_size
        min_num = (max_capacity_prompt - window_size) * 2 - max_num
    if not (q_len >= max_num >= min_num >= window_size):
        return max(1, round(q_len * (1 - compression_ratio)))
    steps = (max_num - min_num) / (num_layers - 1) if num_layers > 1 else 0.0
    return round(max_num - layer_idx * steps)


def window_scores(keys: torch.Tensor, window_queries: torch.Tensor, *, window_size: int,
                  kernel_size: int, attentions: torch.Tensor | None = None):
    """Exact SnapKV score equation, with already native post-RoPE Q as input.

    K [B,Hkv,N,D], Q [B,Hq,W,D]. softmax covers *all* prefix keys before
    removing the observation window. Group averaging follows kvpress.
    """
    b, hkv, n, d = keys.shape
    if n <= window_size or window_queries.shape[-2] != window_size:
        raise ValueError("Full prefix must exceed the exact declared observation window")
    hq = window_queries.shape[1]
    if hq % hkv:
        raise ValueError("Invalid grouped-query geometry")
    if attentions is None:
        repeat_keys = keys[:, :, None].expand(b, hkv, hq // hkv, n, d).reshape(b, hq, n, d)
        weights = torch.matmul(window_queries, repeat_keys.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones_like(weights) * float("-inf"), diagonal=n - window_size + 1)
        weights = torch.softmax(weights + mask, dim=-1, dtype=torch.float32).to(window_queries.dtype)
        weights = weights[..., :-window_size]
    else:
        weights = attentions[..., -window_size:, :-window_size]
    scores = weights.mean(dim=-2)
    scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    scores = scores.view(b, hkv, hq // hkv, n - window_size).mean(2)
    return F.pad(scores, (0, window_size), value=scores.max().item() + 1)


@dataclass
class EvictionOwner:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    original_visual_tokens: int
    full_prefix_tokens: int
    layer_idx: int
    config: dict[str, Any]

    def named_tensors(self, include_constants=True):
        return {"keys": self.keys, "values": self.values, "positions": self.positions}

    def owner_report(self):
        payload = self.keys.untyped_storage().nbytes() + self.values.untyped_storage().nbytes()
        pos = self.positions.untyped_storage().nbytes()
        scalar_count = 2 * self.keys.shape[0] * self.keys.shape[1] * self.original_visual_tokens * self.keys.shape[-1]
        return {"method": "NVIDIA_kvpress_PyramidKV_visual_port", "method_paper": PAPER,
                "upstream_repo": UPSTREAM_REPO, "upstream_commit": UPSTREAM_COMMIT,
                "faithfulness_status": "PINNED_SELECTOR_AND_BUDGET_VISUAL_CANDIDATE_PORT",
                "bits_nominal": None, "stored_visual_KV_bytes": payload + pos,
                "coder_metadata_bytes": 0, "positions_bytes": pos, "encoder_constants_bytes": 0,
                "lookup_bytes": 0, "actual_effective_visual_bits": 8 * (payload + pos) / scalar_count,
                "full_kv_byte_denominator": scalar_count * self.keys.element_size(),
                "retained_visual_tokens_per_head": self.keys.shape[-2],
                "original_visual_tokens": self.original_visual_tokens,
                "temporary_full_dequant_KV_bytes": 0, "config": self.config,
                "storage": storage_bytes(self.named_tensors())}

    def materialize(self, dtype=None):
        return self.keys.to(dtype or self.keys.dtype), self.values.to(dtype or self.values.dtype)

    def attend(self, query, **kwargs):
        return grouped_attention(query, self.keys, self.values, self.positions, **kwargs)


class PyramidKVVisualAdapter:
    def __init__(self, *, compression_ratio=0.875, window_size=64, kernel_size=5, beta=20,
                 num_layers=28, information_condition="query_unknown"):
        if not 0 <= compression_ratio < 1 or beta < 1 or kernel_size % 2 != 1:
            raise ValueError("Invalid original selector parameters")
        if information_condition not in ("query_unknown", "query_known"):
            raise ValueError("Declare query_unknown or query_known")
        self.compression_ratio, self.window_size = compression_ratio, window_size
        self.kernel_size, self.beta, self.num_layers = kernel_size, beta, num_layers
        self.information_condition = information_condition

    def encode(self, keys, values, *, visual_positions, window_queries=None, attentions=None,
               layer_idx=0, question_in_observation=False, prefix_is_question_free=False,
               positions=None, **kwargs):
        """Input is the *complete* original prefix K/V, not a visual-only slice.

        No query substitute is accepted. Q must be native post-RoPE; provided
        attention must be the actual matching full-prefix attention. Unknown
        requires a question-free prefix, known requires question observation.
        """
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Expected matching [B,Hkv,Nprefix,D] K/V")
        if self.information_condition == "query_unknown":
            if question_in_observation or not prefix_is_question_free:
                raise ValueError("INPUT_MISMATCH: query_unknown requires verified question-free prefix")
        elif not question_in_observation:
            raise ValueError("INPUT_MISMATCH: query_known requires current question in observation")
        vp = torch.as_tensor(visual_positions, device=keys.device, dtype=torch.int64)
        if vp.ndim != 1 or len(vp) == 0 or vp.unique().numel() != vp.numel():
            raise ValueError("Visual positions must be nonempty unique 1D indices")
        if vp.min() < 0 or vp.max() >= keys.shape[-2]:
            raise ValueError("Visual position outside original prefix")
        if self.compression_ratio == 0:
            indices = vp.view(1, 1, -1).expand(keys.shape[0], keys.shape[1], -1)
        else:
            if window_queries is None and attentions is None:
                raise ValueError("METHOD_PORT_UNVERIFIED: native window Q or full attention required")
            if window_queries is None:
                window_queries = keys.new_empty((keys.shape[0], attentions.shape[1], self.window_size, keys.shape[-1]))
            scores = window_scores(keys, window_queries, window_size=self.window_size,
                                   kernel_size=self.kernel_size, attentions=attentions)
            # NVIDIA's full-prefix budget is unchanged. Text preservation is the
            # visual port's explicit scope restriction, subtracting exact text.
            total_budget = pyramid_layer_budget(keys.shape[-2], self.compression_ratio,
                                               self.window_size, self.beta, layer_idx, self.num_layers)
            text_count = keys.shape[-2] - len(vp)
            protected_visual = int((vp >= keys.shape[-2] - self.window_size).sum())
            keep = min(len(vp), max(protected_visual, total_budget - text_count, 0))
            selected = scores.index_select(-1, vp).topk(keep, dim=-1).indices
            indices = vp[selected]
        gather_idx = indices.unsqueeze(-1).expand(*indices.shape, keys.shape[-1])
        kept_keys = keys.gather(-2, gather_idx).detach().contiguous()
        kept_values = values.gather(-2, gather_idx).detach().contiguous()
        original_positions = torch.arange(keys.shape[-2], device=keys.device) if positions is None else positions
        original_positions = torch.as_tensor(original_positions, device=keys.device, dtype=torch.int64)
        kept_positions = original_positions[indices].clone()
        config = {"compression_ratio": self.compression_ratio, "window_size": self.window_size,
                  "kernel_size": self.kernel_size, "beta": self.beta, "num_layers": self.num_layers,
                  "information_condition": self.information_condition,
                  "question_in_observation": question_in_observation,
                  "prefix_is_question_free": prefix_is_question_free,
                  "budget_scope": "original_full_prefix_budget_minus_all_exact_text",
                  "protected_visual_window_tokens": int((vp >= keys.shape[-2] - self.window_size).sum())}
        return EvictionOwner(kept_keys, kept_values, kept_positions, len(vp), keys.shape[-2], layer_idx, config)


def create_adapter(**config):
    return PyramidKVVisualAdapter(**config)
