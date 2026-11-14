"""Finite within-layer K/V interaction policies for a fixed-Full-q objective.

This solves a separable SUM of local layer output SSEs. It does not solve the
end-to-end model problem: cross-layer state feedback is absent. Every policy
uses the same calibration Gram and exact residual-unit count for comparison.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'trajkv_round5_interaction_policies_v1'
STATES = ['none', 'K', 'V', 'both']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_grams(rows, layers=28, sample_lookup=None):
    """Average events within document/phase, then phases, then documents."""
    sums, counts = {}, defaultdict(int)
    cluster_sources = set()
    for row in rows:
        if 'interaction_gram' not in row:
            raise ValueError('An input metric row has no interaction_gram')
        sid = row['session_id']
        linked = (sample_lookup or {}).get(sid, {})
        document = row.get('source_document_id') or linked.get('source_document_id')
        if document:
            cluster_sources.add('source_document_id')
        else:
            document = row.get('image_id') or linked.get('image_id')
            cluster_sources.add('image_id' if document else 'session_id_fallback')
            document = document or sid
        layer, phase = int(row['layer']), row['phase']
        if not 0 <= layer < layers or not phase:
            raise ValueError('Invalid layer or phase')
        gram = np.asarray(row['interaction_gram'], dtype=np.float64)
        if gram.shape != (16, 16) or not np.isfinite(gram).all():
            raise ValueError('interaction_gram must be finite 16x16')
        scale = max(float(np.abs(gram).max()), 1.)
        if np.abs(gram-gram.T).max() > 1e-5*scale:
            raise ValueError('interaction_gram is materially asymmetric')
        gram = (gram+gram.T)*.5
        key = (str(document), layer, str(phase))
        if key not in sums:
            sums[key] = np.zeros((16, 16), dtype=np.float64)
        sums[key] += gram
        counts[key] += 1
    if not sums:
        raise ValueError('No Gram rows')
    documents = sorted({key[0] for key in sums})
    result = np.zeros((layers, 16, 16), dtype=np.float64)
    coverage = []
    for doc in documents:
        phases_by_layer = [{phase for d, l, phase in sums if d == doc and l == layer}
                           for layer in range(layers)]
        if not phases_by_layer[0] or any(phases != phases_by_layer[0] for phases in phases_by_layer):
            raise ValueError('Document has incomplete layer/phase coverage: '+doc)
        for layer, phases in enumerate(phases_by_layer):
            phase_means = [sums[(doc, layer, phase)]/counts[(doc, layer, phase)]
                           for phase in sorted(phases)]
            result[layer] += np.mean(phase_means, axis=0)/len(documents)
        coverage.append(dict(document=doc, phases=sorted(phases_by_layer[0]),
            counts_by_phase={phase: [counts[(doc, layer, phase)] for layer in range(layers)]
                             for phase in sorted(phases_by_layer[0])}))
    return result, dict(documents=len(documents), input_rows=len(rows),
        cluster_sources=sorted(cluster_sources), coverage=coverage,
        weighting='events mean within source document/layer/phase; phase means equal; document means equal; sum layers')


def combination_tables(grams):
    combos = np.arange(256, dtype=np.int64)
    costs = np.array([int(combo).bit_count() for combo in combos], dtype=np.int64)
    indexes = np.array([[4*h+((int(combo) >> (2*h)) & 3) for h in range(4)]
                        for combo in combos], dtype=np.int64)
    losses = np.empty((len(grams), 256), dtype=np.float64)
    for layer, gram in enumerate(grams):
        losses[layer] = gram[indexes[:, :, None], indexes[:, None, :]].sum(axis=(1, 2))
    return losses, costs, indexes


def singleton_policy(losses, max_budget):
    layers = len(losses)
    order = sorted(((float(losses[layer, 0]-losses[layer, 1 << bit]), layer, bit)
                    for layer in range(layers) for bit in range(8)),
                   key=lambda row: (-row[0], row[1], row[2]))
    policy = np.zeros((max_budget+1, layers), dtype=np.uint8)
    current = np.zeros(layers, dtype=np.uint8)
    for budget in range(1, max_budget+1):
        _, layer, bit = order[budget-1]
        current[layer] |= np.uint8(1 << bit)
        policy[budget] = current
    return policy, order


def conditional_policy(losses, max_budget):
    layers = len(losses)
    policy = np.zeros((max_budget+1, layers), dtype=np.uint8)
    current = np.zeros(layers, dtype=np.uint8)
    path = []
    for budget in range(1, max_budget+1):
        choices = []
        for layer in range(layers):
            old = int(current[layer])
            for bit in range(8):
                if not old & (1 << bit):
                    gain = float(losses[layer, old]-losses[layer, old | (1 << bit)])
                    choices.append((gain, layer, bit))
        gain, layer, bit = min(choices, key=lambda row: (-row[0], row[1], row[2]))
        current[layer] |= np.uint8(1 << bit)
        policy[budget] = current
        path.append(dict(budget=budget, layer=layer, head=bit//2,
                         axis='K' if bit % 2 == 0 else 'V', conditional_gain=gain))
    return policy, path


def dp_policy(losses, costs, max_budget):
    layers = len(losses)
    best_loss = np.empty((layers, 9), dtype=np.float64)
    best_combo = np.empty((layers, 9), dtype=np.uint8)
    for layer in range(layers):
        for cost in range(9):
            options = np.flatnonzero(costs == cost)
            choice = int(options[np.argmin(losses[layer, options])])
            best_combo[layer, cost] = choice
            best_loss[layer, cost] = losses[layer, choice]
    objective = np.full((layers+1, max_budget+1), np.inf)
    previous = np.full((layers+1, max_budget+1), -1, dtype=np.int16)
    objective[0, 0] = 0.
    for number in range(1, layers+1):
        for budget in range(min(max_budget, 8*number)+1):
            for cost in range(min(8, budget)+1):
                value = objective[number-1, budget-cost]+best_loss[number-1, cost]
                if value < objective[number, budget]:
                    objective[number, budget] = value
                    previous[number, budget] = cost
    policies = np.zeros((max_budget+1, layers), dtype=np.uint8)
    for budget in range(max_budget+1):
        remaining = budget
        for number in range(layers, 0, -1):
            cost = int(previous[number, remaining])
            if cost < 0:
                raise RuntimeError('Unreachable exact budget in DP')
            policies[budget, number-1] = best_combo[number-1, cost]
            remaining -= cost
        if remaining:
            raise RuntimeError('DP reconstruction has nonzero remaining budget')
    return policies, objective[-1], best_loss, best_combo


def policy_objectives(losses, policies):
    return np.array([sum(losses[layer, int(combo)] for layer, combo in enumerate(policy))
                     for policy in policies], dtype=np.float64)


def verify_tables(losses, costs, tables):
    curves = {name: policy_objectives(losses, policy) for name, policy in tables.items()}
    expected = np.arange(len(next(iter(tables.values()))))
    for name, table in tables.items():
        if table.dtype != np.uint8 or table.shape != (len(expected), len(losses)):
            raise RuntimeError('Wrong policy dtype/shape: '+name)
        actual = costs[table.astype(np.int64)].sum(axis=1)
        if not np.array_equal(actual, expected):
            raise RuntimeError('Policy is not exact-budget: '+name)
    scale = max(1., float(np.abs(losses).max())*len(losses))
    tolerance = 1e-10*scale
    for name in ('singleton', 'conditional_greedy'):
        if np.any(curves['dp'] > curves[name]+tolerance):
            raise RuntimeError('DP lost to a feasible heuristic under its own objective')
    return curves, dict(exact_budget_every_row=True, uint8_tables=True,
        dp_not_worse_than_both_heuristics=True,
        conditional_greedy_vs_singleton_order_not_assumed=True, numeric_tolerance=tolerance)


def math_controls():
    rng = np.random.default_rng(1212)
    vectors = rng.normal(size=(16, 11))
    gram = vectors@vectors.T
    losses, costs, indexes = combination_tables(gram[None])
    direct = np.array([float(np.square(vectors[index].sum(axis=0)).sum()) for index in indexes])
    if not np.allclose(direct, losses[0], rtol=1e-12, atol=1e-12):
        raise RuntimeError('Gram and direct vector-composition SSE differ')
    synthetic = []
    for _ in range(2):
        x = rng.normal(size=(16, 9))
        synthetic.append(x@x.T)
    loss2, costs, _ = combination_tables(np.stack(synthetic))
    dp, optimum, _, _ = dp_policy(loss2, costs, 16)
    brute_loss = loss2[0, :, None]+loss2[1, None, :]
    brute_cost = costs[:, None]+costs[None, :]
    brute = np.array([brute_loss[brute_cost == budget].min() for budget in range(17)])
    if not np.allclose(optimum, brute, rtol=1e-12, atol=1e-12):
        raise RuntimeError('DP differs from complete two-layer enumeration')
    single, _ = singleton_policy(loss2, 16)
    greedy, _ = conditional_policy(loss2, 16)
    _, checks = verify_tables(loss2, costs, dict(singleton=single, conditional_greedy=greedy, dp=dp))
    rows = []
    for sid, phase, value, repetitions in [('a', 'prefill', 10., 1), ('a', 'decode', 20., 4),
                                          ('b', 'prefill', 4., 1), ('b', 'decode', 8., 1)]:
        rows.extend(dict(session_id=sid, layer=0, phase=phase,
                         interaction_gram=(np.eye(16)*value).tolist()) for _ in range(repetitions))
    averaged, _ = aggregate_grams(rows, layers=1)
    if not np.allclose(averaged[0], np.eye(16)*10.5):
        raise RuntimeError('Document/phase equal-weight control failed')
    return dict(status='PASS', gram_direct_combinations=256,
        dp_bruteforce_combinations=256**2, dp_bruteforce_layers=2,
        document_phase_equal_weight=True, **checks)


def summarize_interactions(grams):
    nonlinear = []
    for layer, gram in enumerate(grams):
        for head in range(4):
            coefficient = np.zeros(16)
            coefficient[4*head:4*head+4] = [1., -1., -1., 1.]
            nonlinear.append(dict(layer=layer, kv_head=head,
                same_head_kv_nonadditivity_sse=float(coefficient@gram@coefficient)))
    return nonlinear


def serialize_cpu_artifacts(output, metadata, tables, grams):
    """Keep NumPy/MKL solving separate from Torch's CPU runtime on Windows.

    This uses no CUDA and avoids loading two Intel OpenMP runtimes into one
    process. The child only serializes/roundtrips CPU tensors, never solves.
    """
    writer = """import json,sys
from pathlib import Path
import torch
data=json.load(sys.stdin);output=Path(sys.argv[1])
tables={name:torch.tensor(values,dtype=torch.uint8,device='cpu') for name,values in data['policies'].items()}
torch.save({'format':data['format'],'metadata':data['metadata'],'policies':tables},output/'policies.pt')
torch.save({'format':'trajkv_round5_interaction_gram_v1','metadata':data['metadata'],
 'gram':torch.tensor(data['gram'],dtype=torch.float64,device='cpu')},output/'interaction_gram.pt')
loaded=torch.load(output/'policies.pt',map_location='cpu',weights_only=False)
for name,value in tables.items():
 if not torch.equal(loaded['policies'][name],value):raise RuntimeError('Saved CPU policy table roundtrip failed')
print(json.dumps({'roundtrip':'PASS','tensor_bytes':{name:t.numel()*t.element_size() for name,t in tables.items()}}))
"""
    payload = dict(format=FORMAT, metadata=metadata,
                   policies={name: table.tolist() for name, table in tables.items()},
                   gram=grams.tolist())
    result = subprocess.run([sys.executable, '-c', writer, str(output)],
        input=json.dumps(payload), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('CPU Torch artifact serialization failed: '+result.stderr[-1500:])
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--samples', type=Path)
    p.add_argument('--layers', type=int, default=28)
    p.add_argument('--max-budget', type=int, default=48)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    controls = math_controls()
    if a.self_test:
        print(json.dumps(controls))
        return
    if a.metrics is None or a.output is None:
        p.error('--metrics and --output are required unless --self-test')
    if not 0 <= a.max_budget <= 8*a.layers:
        raise ValueError('Budget must be between 0 and 8*layers')
    value = json.loads(a.metrics.read_text(encoding='utf-8-sig'))
    rows = value['local_metrics'] if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError('Metrics must be a list or an object containing local_metrics')
    sample_lookup = None
    if a.samples:
        sample_lookup = {row['id']: row for row in json.loads(a.samples.read_text(encoding='utf-8-sig'))}
    grams, membership = aggregate_grams(rows, layers=a.layers, sample_lookup=sample_lookup)
    eigenvalues = np.linalg.eigvalsh(grams)
    if eigenvalues.min() < -1e-5*max(1., float(np.abs(grams).max())):
        raise ValueError('Aggregated Gram is not positive semidefinite within numeric tolerance')
    losses, costs, indexes = combination_tables(grams)
    single, singleton_order = singleton_policy(losses, a.max_budget)
    conditional, conditional_path = conditional_policy(losses, a.max_budget)
    exact, optimum, layer_loss, layer_combo = dp_policy(losses, costs, a.max_budget)
    tables = dict(singleton=single, conditional_greedy=conditional, dp=exact)
    curves, checks = verify_tables(losses, costs, tables)
    if not np.allclose(curves['dp'], optimum, rtol=1e-11, atol=1e-9):
        raise RuntimeError('DP backtracking does not match DP objective')
    baseline = float(losses[:, 0].sum())
    source_sha = sha(a.metrics)
    row_bytes = (a.max_budget+1)*a.layers  # one uint8 combo per layer/budget
    metadata = dict(source_metrics_sha256=source_sha,
        samples_sha256=sha(a.samples) if a.samples else None,
        analyzer_sha256=sha(__file__), states=STATES,
        flatten_order='index=4*kv_head+state',
        layer_combo_encoding='state_h=(combo>>(2*h))&3',
        residual_unit_cost='popcount(combo), one K or V residual unit per bit',
        budgets=list(range(a.max_budget+1)), layers=a.layers, kv_heads_per_layer=4,
        strategy_tensor_bytes=row_bytes, deployment_table_dtype='uint8',
        objective='sum over layers of equal-document/equal-phase fixed-Full-q local W_O output SSE',
        deployment_scope='retain only the chosen strategy table; if all tables are retained, charge all three',
        omitted_mechanisms='cross-layer state feedback, end-to-end answer loss, and finite-precision full-model nonlinearities')
    output = a.output
    output.mkdir(parents=True, exist_ok=False)
    serialized = serialize_cpu_artifacts(output, metadata, tables, grams)
    rows_curve = []
    gain_lookup = {(layer, bit): gain for gain, layer, bit in singleton_order}
    for budget in range(a.max_budget+1):
        chosen_at_most = int(np.argmin(optimum[:budget+1]))
        strategies = {}
        for name, table in tables.items():
            policy = table[budget]
            predicted = baseline-sum(gain_lookup[(layer, bit)]
                for layer, combo in enumerate(policy) for bit in range(8) if int(combo) & (1 << bit))
            diagonal_head = sum(grams[layer, index, index]
                for layer, combo in enumerate(policy) for index in indexes[int(combo)])
            strategies[name] = dict(objective=float(curves[name][budget]),
                gain_over_none=baseline-float(curves[name][budget]),
                singleton_additive_prediction=float(predicted),
                additivity_gap=float(curves[name][budget]-predicted),
                selected_cross_kv_head_term=float(curves[name][budget]-diagonal_head))
        rows_curve.append(dict(budget=budget, strategies=strategies,
            dp_improvement_over_singleton=float(curves['singleton'][budget]-optimum[budget]),
            dp_improvement_over_conditional=float(curves['conditional_greedy'][budget]-optimum[budget]),
            at_most_dp_objective=float(optimum[chosen_at_most]),
            at_most_dp_used_units=chosen_at_most))
    result = dict(status='COMPLETE', format=FORMAT, analysis_role='calibration_design',
        scope='exact within-layer finite-state interactions at fixed Full q/text; globally optimal only for the separable local sum-SSE surrogate',
        metadata=metadata, membership=membership, controls=controls, direct_checks=checks,
        serialization_controls=serialized,
        baseline_objective=baseline, objective_curves=rows_curve,
        same_head_interactions=summarize_interactions(grams),
        layer_cost_best_combo=layer_combo.tolist(), layer_cost_best_objective=layer_loss.tolist(),
        conditional_greedy_path=conditional_path,
        singleton_order=[dict(layer=layer, head=bit//2, axis='K' if bit % 2 == 0 else 'V',
                              marginal_gain=gain) for gain, layer, bit in singleton_order],
        minimum_gram_eigenvalue=float(eigenvalues.min()),
        policy_tensor_bytes_per_strategy=row_bytes,
        total_policy_tensor_bytes=3*row_bytes,
        policy_artifact_file_bytes=(output/'policies.pt').stat().st_size,
        policy_artifact_sha256=sha(output/'policies.pt'),
        budget_semantics='deployed tables are exact-budget; at-most-DP is reported only, not included as a deployment table',
        optimization_guarantee='DP <= each feasible heuristic under the same local objective; no universal order between conditional greedy and singleton')
    (output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output/'complete.json').write_text(json.dumps(dict(status='COMPLETE', strategies=3,
        budgets=a.max_budget+1, layers=a.layers, source_metrics_sha256=source_sha)))
    print(json.dumps(dict(status='COMPLETE', source_metrics_sha256=source_sha,
        tensor_bytes_per_strategy=row_bytes, baseline_objective=baseline,
        final_budget=rows_curve[-1], controls=controls['status'])))


if __name__ == '__main__':
    main()
