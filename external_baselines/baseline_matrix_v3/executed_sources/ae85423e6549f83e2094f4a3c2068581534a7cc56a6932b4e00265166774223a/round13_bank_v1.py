"""Round13 factory router: deployable question-unknown image-only encoders.

Only native image-prefix KV and physical visual positions enter selection.
Question suffixes, reference answers and diagnostic question queries cannot
enter this interface. All candidate factories return real packed owners.
"""
from __future__ import annotations

from copy import deepcopy

import torch

MODES = ('P_best', 'P_random', 'EX_best', 'TOK_best', 'CAL_AFF')


def require(value, message):
    if not value:
        raise ValueError('ROUND13_ROUTER: ' + message)


def build_round13_owners(native_prefix_cache, visual_positions, variant, *, candidate,
                         legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
                         expected_asset_sha256=None, expected_legacy_source_sha256=None,
                         native_tile_layout=None):
    mode = candidate.get('mode')
    require(mode in MODES, 'unknown deployable Round13 mode')
    require(candidate.get('technical_only', False) is False
            and not candidate.get('allow_over_budget', False)
            and not candidate.get('technical_keep_all', False), 'technical/oracle variants cannot enter production runner')
    n = int(visual_positions.numel())
    require(n > 0 and n % 256 == 0, 'actual complete native256-row image tiles required')
    if native_tile_layout is not None:
        require(native_tile_layout['actual_visual_tokens'] == n
                and native_tile_layout['native_tokens_per_tile'] == 256,
                'prepared native tile/token geometry differs')
    common = dict(legacy_a2=legacy_a2, legacy_experiments=legacy_experiments,
                  legacy_scripts=legacy_scripts, kivi_source=kivi_source,
                  expected_asset_sha256=expected_asset_sha256,
                  expected_legacy_source_sha256=expected_legacy_source_sha256)
    current = deepcopy(candidate)
    if mode.startswith('P_'):
        from round13_pruned_bank_v1 import build_pruned_owners
        current['selector'] = 'random' if mode == 'P_random' else 'importance_diversity'
        current['thumbnail_indices'] = list(range(n - 256, n))
        owners, metadata = build_pruned_owners(native_prefix_cache, visual_positions, variant,
                                               candidate=current, **common)
    elif mode in ('EX_best', 'TOK_best'):
        from round13_selector_v1 import rank_visual_rows
        ranking, selection = rank_visual_rows(native_prefix_cache, visual_positions,
            selector='importance_diversity', seed=int(candidate.get('selector_seed', 0)),
            config=candidate.get('selector_config'))
        if mode == 'EX_best':
            from round13_exception_bank_v1 import budget_plan, build_round13_exception_owners
            stages = int(candidate.get('base_stages', 43))
            requested_h = candidate.get('requested_h')
            if n <= int(candidate.get('small_n_fallback_max_tokens', 0)):
                require(stages == 43, 'small-N fallback requires low43 donor')
                requested_h = 0
            plan = budget_plan(n, stages, requested_h=requested_h)
            high = ranking[:plan['exception_rows']].sort().values.contiguous()
            current['selection_name'] = 'importance_diversity'
            owners, metadata = build_round13_exception_owners(native_prefix_cache, visual_positions,
                variant, candidate=current, selected_indices=high, **common)
        else:
            from round13_precision_bank_v1 import build_round13_precision_owners
            current.update(mode='token_tier', selector_name='importance_diversity')
            owners, metadata = build_round13_precision_owners(native_prefix_cache, visual_positions,
                variant, candidate=current, token_ranking=ranking, **common)
        metadata['round13_image_only_selection'] = selection
    else:
        from round13_precision_bank_v1 import build_round13_precision_owners
        current['mode'] = 'CalTransport'
        owners, metadata = build_round13_precision_owners(native_prefix_cache, visual_positions,
                                                         variant, candidate=current, **common)
    metadata['round13_router'] = {'mode': mode, 'query_unknown': True,
        'native_question_free_prefix_only': True, 'selection_uses_test_question_or_reference': False,
        'original_visual_count': n, 'owner_visual_counts': [int(o.positions.numel()) for o in owners],
        'native_thumbnail_visual_row_range': [n - 256, n],
        'thumbnail_mapping_contract': 'native crop_to_patches image tiles followed by final256 global thumbnail; N256 is thumbnail-only',
        'native_tile_layout': native_tile_layout,
        'native_logical_positions_not_renumbered': True,
        'technical_only': False}
    return owners, metadata
