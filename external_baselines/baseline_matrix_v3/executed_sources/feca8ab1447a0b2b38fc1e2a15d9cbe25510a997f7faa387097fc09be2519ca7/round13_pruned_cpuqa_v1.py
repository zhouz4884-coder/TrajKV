"""CPU format/position QA only; no claim of CUDA or task equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import round12_static_bank_v1 as full_bank
import round12_static_codec_v1 as codec
from round13_pruned_bank_v1 import (budget_plan, compact_full_a_owners,
                                    validate_sparse_causal_positions)
from round13_selector_v1 import rank_visual_rows, select_visual_rows


def check(value, message):
    if not value:
        raise AssertionError(message)


def fake_full_a(n):
    """Synthetic exact-shape owners, including a correctly priced constant pool."""
    generator = torch.Generator().manual_seed(7301)
    vp = torch.arange(9, 9 + n, dtype=torch.long)
    book = torch.randn((256, 4), generator=generator).half()
    lookup = torch.full((28, 2, 4, 2), -1, dtype=torch.int32)
    shared = {'book': book, 'global_residual_lookup': lookup,
              'qa_frozen_constant_padding': torch.zeros(1797, dtype=torch.uint8),
              'round12_format': torch.tensor([1, 4, 8, 0], dtype=torch.uint8)}
    result = []
    left = 43
    for layer in range(28):
        # Different coverage across layers exercises both present/absent pools.
        slots = min(left, 2)
        left -= slots
        bits = torch.randint(0, 2, (1, 4, 128, n), generator=generator)
        state = SimpleNamespace(
            direction_code=torch.randint(0, 256, (1, 4, n, 32), dtype=torch.uint8, generator=generator),
            direction_gain=torch.rand((1, 4, n, 1), generator=generator).half(),
            direction_mean=torch.randn((1, 4, 1, 128), generator=generator).half(),
            v_code=codec.pack_channel_codes(bits, 1),
            v_scale=torch.rand((1, 4, 128, 1), generator=generator).half(),
            v_min=torch.randn((1, 4, 128, 1), generator=generator).half(),
            visual_pos=vp, depth_residual_lookup=lookup[layer], round12_vmode=0)
        if slots:
            state.depth_residual_code=torch.randint(-2147483648, 2147483647, (slots, n, 4), dtype=torch.int32, generator=generator)
            state.depth_residual_alpha=torch.randn((slots, n), generator=generator).half()
        result.append(full_bank.Round12Owner(layer, state, shared, None, n + 9))
    check(left == 0, 'synthetic43 residual count')
    check(full_bank.unique_capacity(result) == 6382 * n + 91657, 'synthetic exact A formula')
    return result, vp


def run_qa():
    n, m = 768, 544
    owners, vp = fake_full_a(n)
    fields = ('direction_code', 'direction_gain', 'direction_mean', 'v_code', 'v_scale', 'v_min',
              'depth_residual_code', 'depth_residual_alpha')
    original = [{name: getattr(o.state, name).clone() for name in fields if hasattr(o.state, name)} for o in owners]
    pointers = [{name: getattr(o.state, name).data_ptr() for name in fields if hasattr(o.state, name)} for o in owners]
    identity = compact_full_a_owners(owners, torch.arange(n), vp)
    check(identity is owners, 'M=N must return identical owner list')
    check(all(getattr(o.state, name).data_ptr() == ptr for o, row in zip(owners, pointers)
              for name, ptr in row.items()), 'identity changed a packed field/storage')
    thumbnail = torch.arange(n - 256, n)
    # Noncontiguous gather deliberately crosses token-packed V word boundaries.
    available = torch.arange(n - 256)
    rows = torch.cat((available[::2], available[1:64:2], thumbnail)).sort().values
    check(rows.numel() == m, 'QA planned row count')
    compact = compact_full_a_owners(owners, rows, vp, selector_id=0, seed=0)
    for owner, previous in zip(compact, original):
        state = owner.state
        for name, value in previous.items():
            actual = getattr(state, name)
            if name in ('direction_code', 'direction_gain'):
                expected = value.index_select(2, rows)
            elif name == 'v_code':
                expected = codec.pack_channel_codes(codec.unpack_channel_codes(value, 1, n).index_select(-1, rows), 1)
            elif name in ('depth_residual_code', 'depth_residual_alpha'):
                expected = value.index_select(1, rows)
            else:
                expected = value
            check(torch.equal(actual, expected), 'full-N encoded field gather mismatch: ' + name)
            check(actual.untyped_storage().nbytes() == actual.numel() * actual.element_size(), 'compact field retains oversized storage: ' + name)
        vfull = codec.half_value(previous['v_code'], previous['v_scale'], previous['v_min'])
        vcompact = codec.half_value(state.v_code, state.v_scale, state.v_min)
        check(torch.equal(vcompact, vfull.index_select(-1, rows)), 'exact Half V reconstruction gather failed')
        kfull = codec.decode_key(previous['direction_code'], previous['direction_gain'], owner.shared['book'])
        kcompact = codec.decode_key(state.direction_code, state.direction_gain, owner.shared['book'])
        check(torch.equal(kcompact, kfull.index_select(2, rows)), 'exact Half K reconstruction gather failed')
    actual = full_bank.unique_capacity(compact)
    expected = 6382 * m + 91657 + 8 * n + 16
    check(actual == expected, f'actual unique capacity mismatch {actual} vs {expected}')
    qpos = torch.arange(n + 9, n + 12)
    text = torch.cat((torch.arange(9), qpos))
    logical = validate_sparse_causal_positions(vp[rows], vp, text, qpos, n + 9)
    check(logical == n + 12, 'logical original prefix length changed')
    blocked = False
    try:
        validate_sparse_causal_positions(vp[rows], vp, torch.cat((text, vp[1:2])), qpos, n + 9)
    except ValueError:
        blocked = True
    check(blocked, 'pruned image row was incorrectly accepted as exact text')
    # Use synthetic native KV for the image-only feature path.
    generator = torch.Generator().manual_seed(902)
    native = SimpleNamespace(layers=[SimpleNamespace(
        keys=torch.randn((1, 4, n + 9, 128), generator=generator).half(),
        values=torch.randn((1, 4, n + 9, 128), generator=generator).half()) for _ in range(28)])
    selectors = {}
    for method in ('spatial', 'random', 'importance_diversity'):
        chosen, metadata = select_visual_rows(native, vp, m, selector=method, thumbnail_indices=thumbnail, seed=4)
        repeated, _ = select_visual_rows(native, vp, m, selector=method, thumbnail_indices=thumbnail, seed=4)
        check(torch.equal(chosen, repeated), 'selector not deterministic: ' + method)
        check(bool(torch.isin(thumbnail, chosen).all()), 'selector dropped thumbnail: ' + method)
        check(chosen.numel() == m and torch.equal(chosen, chosen.unique(sorted=True)), 'selector count/order')
        selectors[method] = metadata
    small, smallmeta = select_visual_rows(native, vp, 32, selector='importance_diversity',
                                         thumbnail_indices=thumbnail, thumbnail_required=False)
    check(small.numel() == 32, 'exception-compatible selector count')
    ranking_results = {}
    for method in ('spatial', 'random', 'importance_diversity'):
        priority, info = rank_visual_rows(native, vp, selector=method, seed=4)
        repeated, _ = rank_visual_rows(native, vp, selector=method, seed=4)
        check(torch.equal(priority, repeated), 'priority not deterministic: ' + method)
        check(torch.equal(priority.sort().values, torch.arange(n)), 'priority permutation incomplete: ' + method)
        check(not torch.equal(priority, torch.arange(n)), 'priority incorrectly took identity shortcut: ' + method)
        ranking_results[method] = info
    plans = [budget_plan(k) for k in (256, 768, 1280, 1792, 2304, 4352)]
    check(all(p['strict_cap_satisfied'] and (p['M'] % 32 == 0) for p in plans), 'budget cap/alignment')
    check(plans[0]['fallback_original_low'] and plans[3]['M'] == 1280, 'thumbnail fallback or reference M wrong')
    return {'schema': 'trajkv.round13.pruned_cpu_format_qa.v1', 'status': 'PASS',
            'CPU_only_no_CUDA_or_task_quality_claim': True,
            'identity_all_packed_fields_and_pointers': True, 'exact_gather_K_V_and_residuals': True,
            'Half_reconstruction_exact_on_retained_rows': True,
            'actual_compact_unique_owner_bytes': actual, 'expected_compact_owner_bytes': expected,
            'no_oversized_underlying_storage_retained': True,
            'original_absolute_positions_and_logical_length_preserved': True,
            'pruned_visual_as_exact_text_rejected': True, 'budget_plans': plans,
            'selectors': selectors, 'exception32_selector': smallmeta, 'priority_rankings': ranking_results,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), Path(__file__).with_name('round13_pruned_bank_v1.py'),
                 Path(__file__).with_name('round13_selector_v1.py'))}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    check(not target.exists(), 'QA result must be a new immutable path')
    result = run_qa()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': str(target), 'bytes': result['actual_compact_unique_owner_bytes']}))
