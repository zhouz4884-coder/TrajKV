"""Preserve a K2 reference stage layout while billing actual K1/K2 bytes."""
import math

from round6_depth_codec import plan_depth_stages


def plan_key_stages(*, layout_reference=None, key_index_bytes=32, book_bytes=2048,
                    reference_factor_bytes=None, **kwargs):
    """Reference only the old K2 allocation; never reuse its byte total.

    The same full uint16 priority table is retained once. The high bit of the
    existing five-byte policy descriptor declares this reference-layout mode.
    Actual book, payload, metric/factor constants, lookup and stages are billed.
    An unaffordable requested reference layout fails rather than truncating it.
    """
    if layout_reference in (None, 'actual'):
        return plan_depth_stages(**kwargs)
    if layout_reference != 'k2':
        raise ValueError('residual_layout_reference must be actual/k2')
    layers, heads, nv = kwargs['layers'], kwargs['heads'], kwargs['visual_tokens']
    if kwargs.get('selected_stages') is not None or len(kwargs.get('selected_units', ())) != 2 * layers * heads:
        raise ValueError('K2 reference layout requires the same complete named unit ranking, not explicit stages')
    if key_index_bytes not in (16, 32) or book_bytes not in (128, 2048):
        raise ValueError('Reference layout is defined only for actual G4_16/G4_256')
    reference_base = kwargs['base_bytes'] + layers * heads * nv * (32 - key_index_bytes) + 2048 - book_bytes
    reference_args = dict(kwargs, base_bytes=reference_base,
                          factor_bytes=kwargs['factor_bytes'] if reference_factor_bytes is None else reference_factor_bytes)
    plan = plan_depth_stages(**reference_args)
    actual_fixed = (kwargs['base_bytes'] + kwargs['factor_bytes'] + plan['lookup_bytes']
                    + len(plan['ranking_constant']) + len(plan['policy_constant']))
    actual_allowed = math.floor(kwargs.get('budget_bits', 1.8) * kwargs['visual_full_bytes'] / 16)
    if actual_fixed > actual_allowed:
        raise RuntimeError('Actual base/constants exceed cap under preserved K2 layout')
    actual_affordable = (actual_allowed - actual_fixed) // plan['stage_bytes']
    if kwargs.get('stage_limit') is not None:
        actual_affordable = min(actual_affordable, max(0, int(kwargs['stage_limit'])))
    if len(plan['selected_stages']) > actual_affordable:
        raise RuntimeError('Preserved K2 reference stages exceed the actual complete owner cap')
    predicted = actual_fixed + len(plan['selected_stages']) * plan['stage_bytes']
    # Retain no separate reference rank or flag tensor; mark the existing policy.
    policy = plan['policy_constant']
    plan.update(layout_reference='k2', layout_reference_base_bytes=reference_base,
        layout_reference_factor_bytes=reference_args['factor_bytes'],
        layout_reference_fixed_bytes=plan['fixed_bytes'],
        layout_reference_predicted_bytes=plan['predicted_bytes'],
        layout_reference_affordable_stages=plan['affordable_stages'],
        layout_reference_incumbent_stage_cap=plan['incumbent_stage_cap'],
        layout_actual_affordable_stages=actual_affordable,
        layout_actual_base_bytes=kwargs['base_bytes'],
        policy_constant=bytes((policy[0] | 128,)) + policy[1:],
        fixed_bytes=actual_fixed, predicted_bytes=predicted, allowed_bytes=actual_allowed,
        affordable_stages=actual_affordable,
        actual_budget_bits=16 * predicted / kwargs['visual_full_bytes'])
    return plan
