"""Exact frozen metric fields and explicitly distinct group-storage contracts.

D96 adapters store the original reference group after merging metric output.
Confirmation/expansion adapters let metric output supply canonical group.
Neither labels nor numeric scores are modified or accepted approximately.
"""
from __future__ import annotations

RAW_REFERENCE_GROUP = 'frozen_D96_raw_reference_group_after_metric_merge'
CANONICAL_METRIC_GROUP = 'frozen_confirmation_metric_canonical_group'


def require(value, message):
    if not value: raise ValueError('ROUND11_FROZEN_METRIC_CONTRACT: '+message)


def verify_metric_fields(module, dataset, prediction, reference, stored, group_contract):
    require(group_contract in (RAW_REFERENCE_GROUP, CANONICAL_METRIC_GROUP), 'unknown frozen group-storage contract')
    metric = module.score_question(dataset, prediction, reference)
    for field in ('score', 'metric_id', 'reference_texts'):
        require(metric.get(field) == stored.get(field), 'exact frozen metric/reference field differs: '+field)
    # This normalization function is part of the byte-pinned metric source.
    _, canonical_group = module._dataset_and_group(dataset, reference)
    require(metric.get('group') == canonical_group, 'metric canonical group differs from pinned scorer normalization')
    expected_stored_group = reference.get('group') if group_contract == RAW_REFERENCE_GROUP else canonical_group
    require(stored.get('group') == expected_stored_group, 'stored group differs from its exact frozen adapter contract')
    return metric
