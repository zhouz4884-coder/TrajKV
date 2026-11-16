"""CPU-only replay of matched static-batch aliases; never reads confirmation.

Reconstructs a complete priority from frozen initial scores under the identical
candidate partition and at-most-one-positive-pair-per-batch rule. It compares
deployment parameters and recomputed bytes with the preserved conditional run.
"""
import argparse
import json
from pathlib import Path

from round7_exchange_policy import apply_swaps, sha, stages, unit

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    'k2_gt': 'results/round7_v1/exchange_cal8_pairs8',
    'k1_gt': 'results/round7_v1/exchange_key_cal8/gt',
    'k1_self': 'results/round7_v1/exchange_key_cal8/self',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_asset(path):
    value = Path(path)
    if value.is_file():
        return value
    marker = '/TrajKV/'
    if marker in str(path):
        candidate = ROOT / str(path).split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def replay_run(label, relative):
    from round7_exchange_key import depth_plan
    folder = ROOT / relative
    paths = [folder / name for name in ('summary.json', 'frozen_candidates.json',
        'policy_incumbent.json', 'policy_conditional_batches.json', 'packed_scores.jsonl')]
    frozen_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in paths}
    summary, frozen, initial_policy, final_policy = [read(path) for path in paths[:4]]
    initial_priority = [unit(value) for value in initial_policy['selected_units_priority']]
    static = {row['pair']['id']: row for row in summary['static_pair_scores']}
    pairs = {row['id']: row for row in frozen['pairs']}
    endpoints = [unit(pair[key]) for pair in pairs.values() for key in ('remove', 'add')]
    assert len(endpoints) == len(set(endpoints)), 'Replay assumes the actual frozen disjoint endpoints'
    initial_allocation = summary['initial']['allocation_id']
    assert all(row['baseline_allocation'] == initial_allocation for row in static.values())
    partitions = [[pair['id'] for pair in batch] for batch in frozen['batches']]
    assert partitions == [trace['candidate_ids'] for trace in summary['trace']]
    assert [pair for batch in partitions for pair in batch] == list(pairs)
    raw = [json.loads(line) for line in paths[4].read_text(encoding='utf-8').splitlines() if line.strip()]
    final_id = summary['final']['conditional_batches']['allocation_id']
    observed = [row for row in raw if row['allocation_id'] == final_id]
    checks = []
    for alias, field, initial_interventions in (
        ('matched_static_batch_additive', 'predicted_improvement_additive', 2 * len(pairs)),
        ('matched_static_batch_joint', 'actual_improvement_joint', len(pairs))):
        accepted, decisions = [], []
        for index, batch in enumerate(partitions):
            ordered = sorted((static[pair_id] for pair_id in batch),
                             key=lambda row: (-row[field], row['pair']['id']))
            best = ordered[0]
            chosen = best['pair'] if best[field] > 0 else None
            if chosen is not None:
                accepted.append(chosen)
            decisions.append({'batch': index, 'frozen_candidate_ids': batch,
                'initial_state_scores': {row['pair']['id']: row[field] for row in ordered},
                'selected_pair_id': None if chosen is None else chosen['id'],
                'state_after_acceptance_was_not_used_for_scoring': True})
        reconstructed = apply_swaps(initial_priority, accepted)
        reconstructed_variant = dict(initial_policy['variant'], selected_stages=stages(reconstructed))
        expected_variant = dict(final_policy['variant'])
        # Display names are labels, not deployment parameters. No other field is
        # omitted, including source/calibration identity and stage/bit limits.
        reconstructed_variant.pop('name', None)
        expected_variant.pop('name', None)
        assert reconstructed_variant == expected_variant
        assert stages(reconstructed) == final_policy['variant']['selected_stages']
        assert [list(value) for value in reconstructed] == final_policy['selected_units_priority']
        actual_accepted = [pair['id'] for pair in summary['accepted_pairs']['conditional_batches']]
        assert [pair['id'] for pair in accepted] == actual_accepted
        settings = dict(final_policy['variant'], selected_units=[list(value) for value in reconstructed])
        for key in ('direction_book_path', 'metadata_factor_path', 'key_metric_path'):
            if settings.get(key):
                settings[key] = str(local_asset(settings[key]))
        base = read(settings['metadata_factor_path'])
        by_n = {}
        # This project's frozen model has L=28,Hkv=4. Verify the full universe
        # before using those dimensions in exact complete-owner recomputation.
        assert len(reconstructed) == 224
        assert set(reconstructed) == {(axis, layer, head) for axis in ('k', 'v')
                                     for layer in range(28) for head in range(4)}
        for row in observed:
            n = row['visual_tokens']
            plan = depth_plan(settings, base, 28, 4, n, reconstructed)
            assert plan['predicted_bytes'] == row['complete_bytes']
            assert abs(plan['actual_budget_bits'] - row['actual_budget_bits']) <= 1e-14
            assert [list(value) for value in plan['selected_stages']] == row['selected_stages']
            assert len(plan['ranking_constant']) == row['ranking_bytes'] == 448
            assert len(plan['policy_constant']) == row['policy_bytes'] == 5
            by_n[str(n)] = {'selected_stages': len(plan['selected_stages']),
                'complete_bytes': plan['predicted_bytes'], 'bits_per_scalar': plan['actual_budget_bits'],
                'ranking_bytes': 448, 'policy_bytes': 5, 'lookup_bytes': plan['lookup_bytes']}
        target_forwards_per_allocation = sum(len(row['scores']) for row in observed)
        objective_forwards_per_allocation = len(observed)
        checks.append({'alias': alias, 'alias_of_frozen_policy': str(paths[3].relative_to(ROOT)).replace('\\', '/'),
            'alias_of_frozen_policy_sha256': sha(paths[3]),
            'existing_free_generation_variant': 'r7_budget_k2_cond' if label == 'k2_gt' else 'r7_joint_k1_gt_cond',
            'selected_pair_ids': actual_accepted, 'decisions': decisions,
            'full_224_unit_ranking_identical': True, 'full_selected_stages_order_identical': True,
            'all_deployment_parameters_except_display_name_identical': True,
            'complete_owner_recomputed_from_actual_codec_assets': True,
            'owner_by_visual_tokens': by_n,
            'initial_intervention_allocations_needed': initial_interventions,
            'initial_plus_one_baseline_allocations_needed': initial_interventions + 1,
            'packed_target_forwards_to_form_policy_using_both_targets': (initial_interventions + 1) * target_forwards_per_allocation,
            'packed_target_forwards_to_form_policy_using_only_selection_objective': (initial_interventions + 1) * objective_forwards_per_allocation,
            'optional_final_validation_adds_allocations': 1,
            'reuse_existing_actual_free_answers': True, 'new_independent_generation_records_created': 0})
    for relative_path, expected_hash in frozen_hashes.items():
        assert sha(ROOT / relative_path) == expected_hash
    return {'run': relative, 'selection_objective': summary['selection_objective'],
        'question_count': summary['questions'], 'independent_sources': summary['independent_sources'],
        'frozen_candidate_count': len(pairs), 'frozen_partitions': partitions,
        'no_candidate_reordering': True, 'no_current_state_score_used_by_aliases': True,
        'global_endpoint_disjointness_verified': True,
        'original_controller_logical_allocation_cap_per_method': summary['logical_allocation_evaluations_per_method'],
        'original_controller_logical_packed_target_forward_cap_per_method': summary['logical_target_forwards_per_method'],
        'original_controller_physical_packed_target_forwards_for_shared_campaign': summary['physical_target_forwards'],
        'original_controller_unique_allocations': summary['unique_allocations'],
        'teacher_generation_and_Full_native_controls_excluded_from_these_counts': True,
        'checks': checks, 'frozen_input_sha256': frozen_hashes, 'frozen_inputs_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'audit/a2/round7_matched_static_batch_equivalence_v1.json')
    args = parser.parse_args()
    if args.output.exists():
        previous = read(args.output)
    else:
        previous = None
    result = {'schema': 'trajkv.round7.matched-static-batch-equivalence.v1', 'status': 'PASS',
        'scope': 'CPU audit of completed calibration only; no confirmation predictions accessed',
        'algorithmic_confound': 'Existing global-top-B static controls differ from conditional in per-batch selection constraint; updated-state scoring is not needed to reproduce any of these final policies',
        'alias_semantics': 'Same deployed policy and same real answers, not an independent replication',
        'runs': {label: replay_run(label, relative) for label, relative in RUNS.items()},
        'check_source_sha256': sha(__file__)}
    if previous is not None and previous != result:
        raise RuntimeError('Refusing to overwrite a different preserved equivalence audit; choose a new output')
    if previous is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'runs': len(result['runs']), 'aliases': 6,
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
