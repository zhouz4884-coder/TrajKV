"""Question-unknown image-only selectors for Round13 physical row owners.

Inputs are the complete native image-prefix KV and original visual positions.
No task question, answer, generated token, or dense N-by-N attention is used.
Returned row IDs are transient encoder workspace, not retained reader state.
"""
from __future__ import annotations

import hashlib
import time

import torch

SCHEMA = 'trajkv.round13.image_only_selector.v1'
SELECTORS = ('spatial', 'importance_diversity', 'random')


def require(value, message):
    if not value:
        raise ValueError('ROUND13_SELECTOR: ' + message)


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _thumbnail(n, device, thumbnail_indices, required):
    if thumbnail_indices is None:
        require(not required or n == 256, 'explicit thumbnail visual-row IDs required for multi-page sources')
        result = torch.arange(n, device=device) if required else torch.empty(0, device=device, dtype=torch.long)
    else:
        result = torch.as_tensor(thumbnail_indices, dtype=torch.long, device=device).reshape(-1)
        require(result.numel() == 256 and torch.equal(result, result.unique(sorted=True))
                and int(result[0]) >= 0 and int(result[-1]) < n, 'thumbnail must identify exactly 256 sorted original rows')
    return result if required else result.new_empty(0)


def _spatial_order(available, n):
    """Equal coverage of real 16x16 tile cells, using a fixed bit-reversed order.

    All source rows are native 256-row tiles; no inferred image saliency is used.
    The order visits each tile at the same local grid cell before refining it.
    """
    require(n % 256 == 0, 'spatial layout requires explicit 256-row native tile units')
    rows = available.detach().cpu().tolist()
    # Bit reversal on Morton-interleaved row/col coordinates gives progressive
    # coverage rather than the top-left-first order of flattened token IDs.
    def key(row):
        cell = row % 256
        y, x = divmod(cell, 16)
        morton = sum(((x >> bit) & 1) << (2 * bit) | ((y >> bit) & 1) << (2 * bit + 1)
                     for bit in range(4))
        reverse = int(f'{morton:08b}'[::-1], 2)
        return reverse, row // 256
    return torch.tensor(sorted(rows, key=key), device=available.device, dtype=torch.long)


@torch.no_grad()
def image_only_features(native_prefix_cache, visual_positions, *, layers=(0, 13, 27)):
    """O(Lsample*N*D) features from native image states, with no question Q.

    Importance is the geometric mean of centered K and V RMS, normalized per
    layer. Diversity uses fixed four-channel averaging, head averaging, and
    per-row normalization. This is a declared heuristic, not a task oracle.
    """
    require(len(native_prefix_cache.layers) == 28, 'M1 L28 native cache required')
    require(layers and all(type(i) is int and 0 <= i < 28 for i in layers), 'valid fixed image-state sample layers')
    features, strengths = [], []
    for index in layers:
        layer = native_prefix_cache.layers[index]
        key = layer.keys.index_select(-2, visual_positions).float()[0]
        value = layer.values.index_select(-2, visual_positions).float()[0]
        key = key - key.mean(-2, keepdim=True)
        value = value - value.mean(-2, keepdim=True)
        krms = key.square().mean((0, 2)).sqrt()
        vrms = value.square().mean((0, 2)).sqrt()
        krms = krms / krms.median().clamp_min(1e-8)
        vrms = vrms / vrms.median().clamp_min(1e-8)
        strengths.append((krms * vrms).sqrt())
        # 32 K + 32 V coordinates per layer. These are encoder-only features.
        feature = torch.cat((key.mean(0).reshape(-1, 32, 4).mean(-1),
                             value.mean(0).reshape(-1, 32, 4).mean(-1)), -1)
        features.append(torch.nn.functional.normalize(feature, dim=-1))
    feature = torch.nn.functional.normalize(torch.cat(features, -1), dim=-1).contiguous()
    importance = torch.stack(strengths).mean(0).contiguous()
    require(bool(torch.isfinite(feature).all()) and bool(torch.isfinite(importance).all()), 'finite image-only features')
    return feature, importance


def _diverse_order(features, importance, available, count, max_anchors):
    """Bounded anchor clustering + within-cluster importance round robin.

    At most max_anchors streaming N-by-D distances; no N-by-N allocation.
    Stable ID tie-breaking is preserved throughout.
    """
    f = features.index_select(0, available)
    weight = importance.index_select(0, available)
    anchors = min(int(max_anchors), count, int(available.numel()))
    require(anchors > 0, 'positive diversity anchors')
    # Mild priority weighting: diversity is not overwhelmed by one norm outlier.
    priority = weight / weight.mean().clamp_min(1e-8)
    priority = priority.clamp(.25, 4.).sqrt()
    chosen = int(torch.argmax(priority))
    best = torch.full_like(weight, float('inf'))
    labels = torch.zeros_like(available)
    anchor_ids = []
    selected = torch.zeros_like(available, dtype=torch.bool)
    for anchor in range(anchors):
        anchor_ids.append(chosen)
        selected[chosen] = True
        distance = (f - f[chosen]).square().sum(-1)
        improve = distance < best
        labels = torch.where(improve, torch.full_like(labels, anchor), labels)
        best = torch.minimum(best, distance)
        candidate_score = (best * priority).masked_fill(selected, -1.)
        chosen = int(torch.argmax(candidate_score))
    # Build ordering once on CPU (O(N log N)); only row IDs cross back to GPU.
    rows, clusters, weights = (x.detach().cpu().tolist() for x in (available, labels, weight))
    buckets = [[] for _ in range(anchors)]
    for row, cluster, strength in zip(rows, clusters, weights):
        buckets[cluster].append((float(strength), int(row)))
    for bucket in buckets:
        bucket.sort(key=lambda pair: (-pair[0], pair[1]))
    # One strongest entry per semantic cluster, then a second, and so on.
    result = []
    for depth in range(max(map(len, buckets))):
        for bucket in buckets:
            if depth < len(bucket):
                result.append(bucket[depth][1])
    return torch.tensor(result[:count], dtype=torch.long, device=available.device)


@torch.no_grad()
def rank_visual_rows(native_prefix_cache, visual_positions, *, selector='importance_diversity',
                     seed=0, config=None):
    """Complete PRIORITY permutation for exception/tier assignment.

    Unlike select_visual_rows(count=N), this never takes the identity shortcut
    and never sorts the chosen rows by original position. No thumbnail row is
    forced into high precision; those callers retain every row at some tier.
    """
    started = time.perf_counter()
    require(selector in SELECTORS and visual_positions.ndim == 1
            and visual_positions.dtype == torch.long, 'known selector and original int64 position vector')
    n, device = int(visual_positions.numel()), visual_positions.device
    require(n > 0, 'nonempty native visual prefix')
    available = torch.arange(n, device=device)
    config = {} if config is None else dict(config)
    require(set(config) <= {'feature_layers', 'max_anchors'}, 'unknown selector configuration')
    if selector == 'random':
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        priority = torch.randperm(n, generator=generator).to(device)
    elif selector == 'spatial':
        priority = _spatial_order(available, n)
    else:
        feature, importance = image_only_features(native_prefix_cache, visual_positions,
                         layers=tuple(config.get('feature_layers', (0, 13, 27))))
        priority = _diverse_order(feature, importance, available, n, int(config.get('max_anchors', 64)))
    require(torch.equal(priority.sort().values, available), 'priority is not a complete original-row permutation')
    return priority, {'schema': SCHEMA, 'selector': selector, 'N': n, 'seed': int(seed), 'config': config,
        'return_order': 'priority_not_original_position', 'priority_visual_row_ids': priority.detach().cpu().tolist(),
        'priority_ids_sha256': _digest(priority), 'mandatory_thumbnail_count': 0,
        'question_or_answer_used': False, 'dense_N_by_N_attention_allocated': False,
        'feature_source': 'native_question_free_prefix_KV' if selector == 'importance_diversity' else 'native_256_row_tile_layout_or_seed',
        'selection_host_elapsed_seconds': time.perf_counter() - started,
        'timing_is_encoder_overhead_not_a_controlled_speed_benchmark': True}


@torch.no_grad()
def select_visual_rows(native_prefix_cache, visual_positions, count, *, selector='importance_diversity',
                       thumbnail_indices=None, thumbnail_required=True, seed=0, config=None):
    started = time.perf_counter()
    require(selector in SELECTORS and type(count) is int, 'known selector and integer count required')
    require(visual_positions.ndim == 1 and visual_positions.dtype == torch.long,
            'original visual positions must be int64 vector')
    n, device = int(visual_positions.numel()), visual_positions.device
    require(0 <= count <= n, 'count inside original visual rows')
    mandatory = _thumbnail(n, device, thumbnail_indices, thumbnail_required)
    require(count >= mandatory.numel(), 'budget cannot drop mandatory complete thumbnail')
    mask = torch.ones(n, device=device, dtype=torch.bool)
    mask[mandatory] = False
    available = torch.arange(n, device=device)[mask]
    take = count - mandatory.numel()
    config = {} if config is None else dict(config)
    require(set(config) <= {'feature_layers', 'max_anchors'}, 'unknown selector configuration')
    if take == 0:
        selected = mandatory
    elif count == n:
        selected = torch.arange(n, device=device)
    elif selector == 'random':
        # CPU RNG makes the declared seed repeatable across CUDA devices.
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        order = torch.randperm(available.numel(), generator=generator)[:take].to(device)
        selected = torch.cat((mandatory, available[order]))
    elif selector == 'spatial':
        selected = torch.cat((mandatory, _spatial_order(available, n)[:take]))
    else:
        layers = tuple(config.get('feature_layers', (0, 13, 27)))
        feature, importance = image_only_features(native_prefix_cache, visual_positions, layers=layers)
        chosen = _diverse_order(feature, importance, available, take, int(config.get('max_anchors', 64)))
        selected = torch.cat((mandatory, chosen))
    selected = selected.sort().values.contiguous()
    require(selected.numel() == count and selected.unique().numel() == count, 'exact unique selected-row count')
    metadata = {'schema': SCHEMA, 'selector': selector, 'N': n, 'M': count, 'seed': int(seed),
                'thumbnail_required': bool(thumbnail_required), 'mandatory_thumbnail_count': int(mandatory.numel()),
                'selected_visual_row_ids': selected.detach().cpu().tolist(), 'selected_ids_sha256': _digest(selected),
                'question_or_answer_used': False, 'dense_N_by_N_attention_allocated': False,
                'feature_source': 'native_question_free_prefix_KV' if selector == 'importance_diversity' else 'native_256_row_tile_layout_or_seed',
                'config': config, 'selection_host_elapsed_seconds': time.perf_counter() - started,
                'timing_is_encoder_overhead_not_a_controlled_speed_benchmark': True}
    return selected, metadata
