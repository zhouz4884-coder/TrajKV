"""Synthetic-only CPU checks for policy admission/decision boundaries.

No real threshold/action is selected and no real policy or input cohort is
created. The temporary JSON below is explicitly synthetic and deleted before
the receipt is written. No torch/model/GPU/remote module is imported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import round12_policy_contract_v1 as c


def expect_reject(fn, name):
    try:
        fn()
    except (ValueError, TypeError, KeyError):
        return {'name': name, 'status': 'PASS_REJECTED'}
    raise AssertionError('Bad synthetic fixture admitted: ' + name)


def checks(output):
    asset = dict(schema=c.POLICY_SCHEMA, policy_id='SYNTHETIC_ONLY_NOT_A_REAL_POLICY',
        feature='first_token_margin', threshold=1.0, action='P', comparison='lt',
        probe_max_new_tokens=3, selection_scope='observed_development_only_before_evaluation',
        development_input_manifest_sha256='a' * 64, selection_report_sha256='b' * 64,
        frozen_before_evaluation=True, uses_current_answer_GT=False)
    c.validate_policy_asset(asset)
    probe = dict(first_token_margin=1.0, first3_min_margin=.5,
        native_generation_processors_used=True, low_probe_text_KV_rolled_back_to_prefix=True,
        actual_probe_token_count=3)
    tests = []
    assert c.evaluate_margin_decision(asset, probe)['selected_action'] == 'a0'
    assert c.evaluate_margin_decision(asset, {**probe, 'first_token_margin': .999})['selected_action'] == 'P'
    alt = {**asset, 'feature': 'first3_min_margin', 'action': 'Z'}
    assert c.evaluate_margin_decision(alt, probe)['selected_action'] == 'Z'
    assert c.evaluate_margin_decision({**asset, 'threshold': 0.0}, {**probe, 'first_token_margin': 0.0})['selected_action'] == 'a0'
    tests.append({'name': 'strict_threshold_boundary_and_single_declared_action', 'status': 'PASS'})
    for value, name in [(math.nan, 'NaN'), (math.inf, 'infinity'), (-1.0, 'negative'), (True, 'boolean')]:
        tests.append(expect_reject(lambda value=value: c.validate_policy_asset({**asset, 'threshold': value}),
                                   'invalid_threshold_' + name))
    tests.append(expect_reject(lambda: c.validate_policy_asset({**asset, 'answers': ['not permitted']}), 'extra_reference_field'))
    tests.append(expect_reject(lambda: c.validate_policy_asset({**asset, 'action': 'oracle_best'}), 'undeclared_action'))
    tests.append(expect_reject(lambda: c.validate_policy_asset({**asset, 'feature': 'high_score'}), 'future_feature'))
    tests.append(expect_reject(lambda: c.evaluate_margin_decision(asset, {**probe, 'low_probe_text_KV_rolled_back_to_prefix': False}), 'probe_not_rolled_back'))
    tests.append(expect_reject(lambda: c.evaluate_margin_decision(asset, {**probe, 'first_token_margin': math.nan}), 'nonfinite_current_feature'))
    ids = [f'SYNTHETIC:{i}' for i in range(8)]
    metadata = {ident: {'actual_visual_token_count': 256 if i < 4 else 768} for i, ident in enumerate(ids)}
    policy_sha = 'c' * 64
    spec = dict(schema=c.RANDOM_SPEC_SCHEMA, policy_asset_sha256=policy_sha,
        seed='SYNTHETIC-SEED-NOT-A-REAL-ALLOCATOR', stratum_field='actual_visual_token_count',
        quota_rule='floor_calibrated_fraction_per_stratum', upgrade_fractions={'all': .5},
        selection_scope='observed_development_only_before_evaluation',
        development_input_manifest_sha256='a' * 64, selection_report_sha256='b' * 64,
        frozen_before_evaluation=True, uses_current_answer_GT=False)
    c.validate_random_allocator_spec(spec, policy_sha)
    strata = c.random_strata(ids, metadata, spec['stratum_field'])
    decisions = dict.fromkeys(ids, False)
    counts = {}
    for key, members in strata.items():
        counts[key] = len(members) // 2
        ordered = sorted(members, key=lambda ident: (
            hashlib.sha256((spec['seed'] + '|' + key + '|' + ident).encode()).hexdigest(), ident))
        for ident in ordered[:counts[key]]:
            decisions[ident] = True
    random = dict(schema=c.RANDOM_SCHEMA, policy_asset_sha256=policy_sha,
        input_sha256='d' * 64, ordered_ids_sha256=c.canonical_sha(ids),
        allocator_spec_sha256='e' * 64, native_bindings_sha256='f' * 64, decisions=decisions,
        generation=dict(method='seeded_hash_rank_per_stratum', seed=spec['seed'],
            stratum_field=spec['stratum_field'], upgrade_counts=counts,
            uses_correctness_or_reference_answers=False, frozen_before_terminal_outputs=True))
    binding = dict(policy_asset_sha256=policy_sha, input_sha256='d' * 64,
        ordered_ids=ids, metadata_by_id=metadata, allocator_spec=spec,
        allocator_spec_sha256='e' * 64, native_bindings_sha256='f' * 64)
    c.validate_random_manifest(random, **binding)
    assert sum(c.evaluate_random_decision(asset, random, ident)['upgrade'] for ident in ids) == 4
    tests.append({'name': 'frozen_random_rule_input_binding_and_metadata_strata', 'status': 'PASS'})
    bad = deepcopy(random)
    bad['decisions'][ids[0]] = not bad['decisions'][ids[0]]
    tests.append(expect_reject(lambda: c.validate_random_manifest(bad, **binding), 'tampered_random_choice'))
    tests.append(expect_reject(lambda: c.validate_random_manifest(random, **{**binding, 'ordered_ids': ids[::-1]}), 'different_input_order'))
    bad_count = deepcopy(random)
    bad_count['generation']['upgrade_counts']['256'] += 1
    tests.append(expect_reject(lambda: c.validate_random_manifest(bad_count, **binding), 'changed_frozen_quota'))
    bad_seed = deepcopy(random)
    bad_seed['generation']['seed'] = 'POSTHOC'
    tests.append(expect_reject(lambda: c.validate_random_manifest(bad_seed, **binding), 'changed_allocator_seed'))
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = output.with_name('SYNTHETIC_ONLY_TEMPORARY_ASSET.json')
    assert not fixture.exists() and not output.exists()
    try:
        fixture.write_text(json.dumps(asset) + '\n', encoding='utf-8')
        digest = c.sha(fixture)
        admitted, evidence = c.load_policy_asset(fixture, digest)
        assert admitted == asset and evidence['serialized_bytes'] == fixture.stat().st_size
        tests.append(expect_reject(lambda: c.load_policy_asset(fixture, '0' * 64), 'wrong_asset_hash'))
        tests.append({'name': 'asset_content_hash_and_serialized_host_charge', 'status': 'PASS'})
    finally:
        # A single explicitly named file, never a recursive removal.
        if fixture.exists():
            fixture.unlink()
    result = dict(schema='trajkv.round12.policy_contract_cpu_check.v1',
        status='PASS_SYNTHETIC_CPU_CONTRACT_ONLY', tests=tests,
        checks=len(tests), real_policy_or_threshold_selected=False,
        real_policy_asset_created=False, synthetic_asset_fixture_deleted=not fixture.exists(),
        real_confirmation_sources_created=False, model_imports=0, model_forward_calls=0,
        GPU_calls=0, remote_calls=0,
        source_sha256={p.name: c.sha(p) for p in (Path(__file__), Path(c.__file__))},
        qualification='Asset and decision admission only; does not establish GPU a0 numerical equivalence, quality or resource gains')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(checks(args.output), ensure_ascii=False, indent=2))
