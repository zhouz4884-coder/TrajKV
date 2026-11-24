"""CPU-only admission and deterministic decisions for future R12 policies.

This module neither chooses a threshold/action nor reads scores. It creates no
real policy, confirmation sample, model call or remote job. Selection assets
must be produced and frozen by a separate observed-development decision.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys

POLICY_SCHEMA = 'trajkv.round12.margin_policy_asset.v1'
RANDOM_SCHEMA = 'trajkv.round12.frozen_random_decisions.v1'
RANDOM_SPEC_SCHEMA = 'trajkv.round12.random_allocator_spec.v1'
POLICY_FIELDS = {
    'schema', 'policy_id', 'feature', 'threshold', 'action', 'comparison',
    'probe_max_new_tokens', 'selection_scope', 'development_input_manifest_sha256',
    'selection_report_sha256', 'frozen_before_evaluation', 'uses_current_answer_GT',
}
RANDOM_FIELDS = {
    'schema', 'policy_asset_sha256', 'input_sha256', 'ordered_ids_sha256',
    'allocator_spec_sha256', 'native_bindings_sha256', 'decisions', 'generation',
}
RANDOM_SPEC_FIELDS = {
    'schema', 'policy_asset_sha256', 'seed', 'stratum_field', 'quota_rule',
    'upgrade_fractions', 'selection_scope', 'development_input_manifest_sha256',
    'selection_report_sha256', 'frozen_before_evaluation', 'uses_current_answer_GT',
}
GENERATION_FIELDS = {
    'method', 'seed', 'stratum_field', 'upgrade_counts',
    'uses_correctness_or_reference_answers', 'frozen_before_terminal_outputs',
}
FEATURES = ('first_token_margin', 'first3_min_margin')
ACTIONS = ('P', 'Z')


def require(condition, message):
    if not condition:
        raise ValueError('ROUND12_POLICY_CONTRACT: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def recursive_size(value):
    """Actual CPython object sizes, de-duplicated; no claim of process RSS."""
    seen = set()
    def visit(x):
        if id(x) in seen:
            return 0
        seen.add(id(x))
        size = sys.getsizeof(x)
        if isinstance(x, dict):
            size += sum(visit(k) + visit(v) for k, v in x.items())
        elif isinstance(x, (tuple, list, set)):
            size += sum(map(visit, x))
        return size
    return visit(value)


def validate_policy_asset(asset):
    require(isinstance(asset, dict) and set(asset) == POLICY_FIELDS,
            'policy must contain exactly declared numerical/provenance fields')
    require(asset['schema'] == POLICY_SCHEMA, 'unknown policy schema')
    require(isinstance(asset['policy_id'], str) and
            re.fullmatch('[A-Za-z0-9_.-]{1,128}', asset['policy_id']) is not None,
            'simple bounded policy identifier required')
    require(asset['feature'] in FEATURES and asset['action'] in ACTIONS,
            'only one frozen first/min3 margin and one P/Z action supported')
    require(finite_number(asset['threshold']) and asset['threshold'] >= 0,
            'nonnegative finite margin threshold required')
    require(asset['comparison'] == 'lt', 'strict margin<threshold comparison required')
    require(type(asset['probe_max_new_tokens']) is int and asset['probe_max_new_tokens'] == 3,
            'first version pays and uses the accepted native cap3 probe')
    require(asset['selection_scope'] == 'observed_development_only_before_evaluation',
            'policy must be selected only on observed development outputs')
    require(valid_sha(asset['development_input_manifest_sha256']) and
            valid_sha(asset['selection_report_sha256']), 'selection provenance hashes required')
    require(asset['frozen_before_evaluation'] is True and asset['uses_current_answer_GT'] is False,
            'policy must be frozen before current evaluation and cannot use current GT')
    return asset


def read_asset(path, expected_sha256, *, max_bytes):
    path = Path(path)
    require(path.is_file() and path.suffix.lower() == '.json', 'existing JSON asset required')
    require(valid_sha(expected_sha256), 'explicit asset SHA required')
    require(0 < path.stat().st_size <= max_bytes, 'asset size outside bound')
    body = path.read_bytes()
    require(0 < len(body) <= max_bytes and hashlib.sha256(body).hexdigest() == expected_sha256,
            'asset bytes/size/hash differ')
    value = json.loads(body.decode('utf-8-sig'))
    return value, dict(path=str(path), sha256=expected_sha256, serialized_bytes=len(body),
        loaded_python_object_bytes=recursive_size(value),
        python_size_scope='recursive sys.getsizeof with identity de-duplication; not RSS',
        policy_GPU_tensor_bytes=0)


def load_policy_asset(path, expected_sha256):
    asset, evidence = read_asset(path, expected_sha256, max_bytes=1024 * 1024)
    validate_policy_asset(asset)
    evidence.update(schema=POLICY_SCHEMA, policy_id=asset['policy_id'],
        feature=asset['feature'], action=asset['action'], threshold=asset['threshold'])
    return asset, evidence


def evaluate_margin_decision(asset, probe):
    validate_policy_asset(asset)
    value = probe.get(asset['feature'])
    require(finite_number(value) and value >= 0, 'finite current-low margin required')
    require(probe.get('native_generation_processors_used') is True and
            probe.get('low_probe_text_KV_rolled_back_to_prefix') is True,
            'decision must use the accepted actual low probe after rollback')
    require(type(probe.get('actual_probe_token_count')) is int and
            1 <= probe['actual_probe_token_count'] <= 3, 'invalid native probe count')
    upgrade = value < asset['threshold']
    return dict(decision_mode='frozen_margin_threshold', selected_action=asset['action'] if upgrade else 'a0',
        upgrade=upgrade, feature=asset['feature'], feature_value=value,
        threshold=asset['threshold'], comparison='lt',
        features_scope='current_low_probe_only', current_GT_or_other_branch_output_used=False)


def random_strata(ids, metadata_by_id, field):
    require(len(ids) == len(set(ids)) and all(isinstance(x, str) for x in ids),
            'random manifest needs exact unique input IDs')
    result = {}
    for ident in ids:
        if field == 'none':
            key = 'all'
        else:
            require(field == 'actual_visual_token_count' and ident in metadata_by_id,
                    'random strata must use admitted native input token-count metadata')
            n = metadata_by_id[ident].get(field)
            require(type(n) is int and n > 0, 'invalid native token count for random stratum')
            key = str(n)
        result.setdefault(key, []).append(ident)
    return result


def validate_random_allocator_spec(spec, policy_asset_sha256):
    require(isinstance(spec, dict) and set(spec) == RANDOM_SPEC_FIELDS and
            spec['schema'] == RANDOM_SPEC_SCHEMA, 'unknown random allocator spec fields')
    require(spec['policy_asset_sha256'] == policy_asset_sha256, 'random allocator binds another policy')
    require(isinstance(spec['seed'], str) and 0 < len(spec['seed']) <= 256 and
            spec['stratum_field'] in ('none', 'actual_visual_token_count'), 'frozen seed/input-only strata required')
    require(spec['quota_rule'] == 'floor_calibrated_fraction_per_stratum', 'unsupported random quota rule')
    fractions = spec['upgrade_fractions']
    require(isinstance(fractions, dict) and 'all' in fractions and
            all((key == 'all' or (isinstance(key, str) and key.isdecimal() and int(key) > 0))
                and finite_number(value) and 0 <= value <= 1 for key, value in fractions.items()),
            'calibration fractions with all-strata fallback required')
    require(spec['stratum_field'] != 'none' or set(fractions) == {'all'}, 'unstratified allocator accepts only all quota')
    require(spec['selection_scope'] == 'observed_development_only_before_evaluation' and
            valid_sha(spec['development_input_manifest_sha256']) and valid_sha(spec['selection_report_sha256']) and
            spec['frozen_before_evaluation'] is True and spec['uses_current_answer_GT'] is False,
            'allocator rule must be selected and frozen on development only before this evaluation')
    return spec


def load_random_allocator_spec(path, expected_sha256, policy_asset_sha256):
    spec, evidence = read_asset(path, expected_sha256, max_bytes=1024 * 1024)
    validate_random_allocator_spec(spec, policy_asset_sha256)
    evidence.update(schema=RANDOM_SPEC_SCHEMA, seed=spec['seed'],
                    stratum_field=spec['stratum_field'], quota_rule=spec['quota_rule'])
    return spec, evidence


def validate_random_manifest(value, *, policy_asset_sha256, input_sha256,
                             ordered_ids, metadata_by_id, allocator_spec,
                             allocator_spec_sha256, native_bindings_sha256):
    require(isinstance(value, dict) and set(value) == RANDOM_FIELDS and
            value['schema'] == RANDOM_SCHEMA, 'unknown random manifest fields/schema')
    require(value['policy_asset_sha256'] == policy_asset_sha256 and
            value['input_sha256'] == input_sha256 and
            value['ordered_ids_sha256'] == canonical_sha(ordered_ids) and
            value['allocator_spec_sha256'] == allocator_spec_sha256 and
            value['native_bindings_sha256'] == native_bindings_sha256,
            'random decisions are not bound to this policy and exact no-answer cohort')
    require(valid_sha(allocator_spec_sha256) and valid_sha(native_bindings_sha256),
            'allocator spec and input-native binding hashes required')
    validate_random_allocator_spec(allocator_spec, policy_asset_sha256)
    decisions = value['decisions']
    require(isinstance(decisions, dict) and set(decisions) == set(ordered_ids) and
            all(type(v) is bool for v in decisions.values()), 'one explicit boolean for every input ID required')
    generation = value['generation']
    require(isinstance(generation, dict) and set(generation) == GENERATION_FIELDS,
            'random generation provenance fields differ')
    require(generation['method'] == 'seeded_hash_rank_per_stratum' and
            isinstance(generation['seed'], str) and 0 < len(generation['seed']) <= 256,
            'deterministic seeded hash ranking required')
    require(generation['uses_correctness_or_reference_answers'] is False and
            generation['frozen_before_terminal_outputs'] is True,
            'random decisions must be frozen without current terminal correctness')
    require(generation['seed'] == allocator_spec['seed'] and
            generation['stratum_field'] == allocator_spec['stratum_field'], 'random generation changes frozen allocator spec')
    strata = random_strata(ordered_ids, metadata_by_id, generation['stratum_field'])
    counts = generation['upgrade_counts']
    require(isinstance(counts, dict) and set(counts) == set(strata), 'upgrade count for each admitted stratum required')
    expected = {ident: False for ident in ordered_ids}
    for key, members in strata.items():
        count = counts[key]
        require(type(count) is int and 0 <= count <= len(members), 'invalid random stratum count')
        fraction = allocator_spec['upgrade_fractions'].get(key, allocator_spec['upgrade_fractions']['all'])
        require(count == math.floor(fraction * len(members)), 'random count differs from pre-data calibrated quota rule')
        order = sorted(members, key=lambda ident: (
            hashlib.sha256((generation['seed'] + '|' + key + '|' + ident).encode()).hexdigest(), ident))
        for ident in order[:count]:
            expected[ident] = True
    require(decisions == expected, 'declared decisions differ from frozen randomization rule')
    return value


def load_random_manifest(path, expected_sha256, **binding):
    value, evidence = read_asset(path, expected_sha256, max_bytes=64 * 1024 * 1024)
    validate_random_manifest(value, **binding)
    evidence.update(schema=RANDOM_SCHEMA, action_count=sum(value['decisions'].values()),
        input_count=len(value['decisions']), stratum_field=value['generation']['stratum_field'],
        matching_scope='explicit frozen action-count/input-metadata strata; actual resources must still be measured')
    return value, evidence


def evaluate_random_decision(asset, manifest, ident):
    require(ident in manifest['decisions'], 'input ID missing from admitted random manifest')
    upgrade = manifest['decisions'][ident]
    return dict(decision_mode='frozen_random_manifest', selected_action=asset['action'] if upgrade else 'a0',
        upgrade=upgrade, feature=None, feature_value=None, threshold=None,
        random_manifest_decision_ID=ident, features_scope='frozen_no_answer_input_identity_only',
        current_GT_or_other_branch_output_used=False)
