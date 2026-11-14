"""Stage a strict calibration-only matched-DeltaMSE residual swap.

Run locally after both calibration artifacts arrive. No model, task-score,
reference-answer, SSH, or GPU code is imported. Existing outputs are never
overwritten. Insufficient strict matches produce an audit, not a runnable job.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIR_COUNT = 8
COMMON_COUNT = 16
RATIO_MIN, RATIO_MAX = .9, 1.1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local_project_path(path):
    direct = Path(path)
    if direct.is_file():
        return direct
    marker = '/TrajKV/'
    if marker not in str(path):
        raise ValueError('Cannot map project asset to local checkout: '+str(path))
    mapped = ROOT/str(path).rsplit(marker, 1)[1]
    if not mapped.is_file():
        raise FileNotFoundError(mapped)
    return mapped


def build_candidates(finite, gradient):
    if gradient.get('status') != 'COMPLETE' or gradient.get('gold_answers_loaded') is not False:
        raise ValueError('Completed calibration-only MSE artifact required')
    stats = gradient['per_axis']['v']
    nl, nh = len(stats['mean_mse_gain']), len(stats['mean_mse_gain'][0])
    if (nl, nh) != (28, 4):
        raise ValueError('This design is frozen for 28 layers x 4 KV heads')
    data = {}
    for row in finite['ranks']['v']:
        unit = tuple(row['unit'])
        if unit in data or not (len(unit) == 2 and 0 <= unit[0] < nl and 0 <= unit[1] < nh):
            raise ValueError('Duplicate or invalid finite-gain unit')
        layer, head = unit
        item = dict(unit=unit, roi=float(row['estimated_local_gain']),
                    mse_gain=float(stats['mean_mse_gain'][layer][head]),
                    base_mse=float(stats['mean_base_mse'][layer][head]),
                    refined_mse=float(stats['mean_refined_mse'][layer][head]))
        if not all(math.isfinite(item[key]) for key in ('roi', 'mse_gain', 'base_mse', 'refined_mse')):
            raise ValueError('Nonfinite calibration statistic')
        if not math.isclose(item['mse_gain'], item['base_mse']-item['refined_mse'], rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError('Reported MSE decrease disagrees with base/refined values')
        data[unit] = item
    if len(data) != nl*nh:
        raise ValueError('Expected complete 112-unit calibration coverage')
    candidates = []
    units = sorted(data)
    for index, first in enumerate(units):
        for second in units[index+1:]:
            high, low = sorted((data[first], data[second]), key=lambda x: (-x['roi'], x['unit']))
            if high['roi'] <= low['roi'] or min(high['mse_gain'], low['mse_gain']) <= 0:
                continue
            ratio = high['mse_gain']/low['mse_gain']
            if RATIO_MIN <= ratio <= RATIO_MAX:
                candidates.append(dict(high=high, low=low, mse_gain_ratio=ratio,
                                       roi_gap=high['roi']-low['roi']))
    candidates.sort(key=lambda x: (x['high']['unit'], x['low']['unit']))
    return data, candidates


def select_exact_pairs(data, candidates):
    # NetworkX is available in the local authoring environment. Using its
    # Blossom matcher avoids greedy failure when eight legal pairs do exist.
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(sorted(data))
    lookup = {}
    tie_scale = (len(candidates)+1)*PAIR_COUNT+1
    for index, pair in enumerate(candidates):
        high, low = pair['high']['unit'], pair['low']['unit']
        # Integer weights make ties deterministic at a 1e-9 ROI resolution.
        weight = round(pair['roi_gap']*1e9)*tie_scale+len(candidates)-index
        graph.add_edge(high, low, weight=weight)
        lookup[frozenset((high, low))] = pair
    maximum = nx.max_weight_matching(graph, maxcardinality=True, weight='weight')
    if len(maximum) < PAIR_COUNT:
        partial = [lookup[frozenset(edge)] for edge in maximum]
        return sorted(partial, key=lambda x: (-x['roi_gap'], x['high']['unit'])), len(maximum)
    # n-2k dummy vertices must each match an original vertex. A perfect
    # matching then contains exactly k real-real edges, optimizing their sum.
    augmented = graph.copy()
    dummy = [('dummy', i) for i in range(len(data)-2*PAIR_COUNT)]
    for node in dummy:
        for unit in sorted(data):
            augmented.add_edge(node, unit, weight=0)
    matching = nx.max_weight_matching(augmented, maxcardinality=True, weight='weight')
    if len(matching)*2 != augmented.number_of_nodes():
        raise RuntimeError('Fixed-cardinality matching construction failed')
    pairs = [lookup[frozenset(edge)] for edge in matching if frozenset(edge) in lookup]
    if len(pairs) != PAIR_COUNT:
        raise RuntimeError('Matching did not select exactly eight real pairs')
    return sorted(pairs, key=lambda x: (-x['roi_gap'], x['high']['unit'])), len(maximum)


def arm_totals(units, data):
    return dict(units=[list(unit) for unit in units],
                mse_gain_sum=sum(data[u]['mse_gain'] for u in units),
                base_mse_sum=sum(data[u]['base_mse'] for u in units),
                refined_mse_sum=sum(data[u]['refined_mse'] for u in units),
                finite_local_gain_sum=sum(data[u]['roi'] for u in units))


def csv_rows(pairs):
    return [dict(pair=index+1, high_layer=p['high']['unit'][0], high_head=p['high']['unit'][1],
                 low_layer=p['low']['unit'][0], low_head=p['low']['unit'][1],
                 high_roi=p['high']['roi'], low_roi=p['low']['roi'], roi_gap=p['roi_gap'],
                 high_mse_gain=p['high']['mse_gain'], low_mse_gain=p['low']['mse_gain'],
                 mse_gain_ratio=p['mse_gain_ratio'],
                 high_base_mse=p['high']['base_mse'], low_base_mse=p['low']['base_mse'],
                 high_refined_mse=p['high']['refined_mse'], low_refined_mse=p['low']['refined_mse'])
            for index, p in enumerate(pairs)]


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        json.dump(data, output, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finite', type=Path, default=ROOT/'configs/a2/round5_v1/sparse_rankings.json')
    parser.add_argument('--mse', type=Path, default=ROOT/'results/round5_v1/gradient_queue/calibration16/ranking_scores.json')
    parser.add_argument('--template', type=Path, default=ROOT/'configs/a2/round5_v1/sparse_dev24_s0.json')
    parser.add_argument('--output', type=Path, default=ROOT/'configs/a2/round5_v1/matched_swap_dev64.json')
    parser.add_argument('--audit', type=Path, default=ROOT/'audit/a2/round5_v1/matched_swap_design.json')
    parser.add_argument('--pairs-csv', type=Path, default=ROOT/'audit/a2/round5_v1/matched_swap_pairs.csv')
    args = parser.parse_args()
    for path in (args.output, args.audit, args.pairs_csv):
        if path.exists():
            raise FileExistsError('Refusing to overwrite: '+str(path))
    finite, gradient, template = read(args.finite), read(args.mse), read(args.template)
    variant = copy.deepcopy(next(v for v in template['variants'] if v['name'] == 'v_impact_max'))
    if variant.get('functional_v_factor_path') or variant.get('v_group_size') != 'all' or variant['residual_axis'] != 'v':
        raise ValueError('This swap uses the unchanged whole-image V1 frozen base')
    base_path = local_project_path(variant['metadata_factor_path'])
    book_path = local_project_path(variant['direction_book_path'])
    if sha(base_path) != gradient['fixed_metadata_sha256'] or sha(book_path) != gradient['encoder_book_sha256']:
        raise ValueError('MSE calibration and runtime use different K code/factors')
    data, candidates = build_candidates(finite, gradient)
    pairs, maximum_count = select_exact_pairs(data, candidates)
    audit = dict(schema='trajkv_matched_delta_mse_swap_v1',
                 created_at_utc=datetime.now(timezone.utc).isoformat(),
                 status='READY' if len(pairs) == PAIR_COUNT else 'INSUFFICIENT_MATCHES',
                 requested_pairs=PAIR_COUNT, selected_pairs=len(pairs),
                 strict_mse_gain_ratio=[RATIO_MIN, RATIO_MAX],
                 eligible_edges=len(candidates), maximum_nonoverlapping_pairs=maximum_count,
                 matching='exact eight-edge maximum total finite-gain-gap matching; ROI integer resolution 1e-9; no MSE-caliper relaxation',
                 calibration_only=True, development_or_confirmation_answers_read=False,
                 inputs={str(path): sha(path) for path in (args.finite, args.mse, args.template, base_path, book_path)},
                 finite_source_sha256=finite.get('source_sha256'), finite_scope=finite.get('scope'),
                 mse_calibration_cases=gradient['cases'], mse_calibration_images=gradient['images'],
                 mse_calibration_ids=gradient['calibration_ids'],
                 pairs=csv_rows(pairs),
                 matching_scope='MSE reduction is matched within 10%; base MSE is reported but not a second matching restriction; summed finite gains are a local nonadditive proxy, not a predicted ANLS')
    if len(pairs) == PAIR_COUNT:
        excluded = {p[side]['unit'] for p in pairs for side in ('high', 'low')}
        common = sorted((u for u in data if u not in excluded), key=lambda u: (-data[u]['roi'], u))[:COMMON_COUNT]
        high = common+[p['high']['unit'] for p in pairs]
        low = common+[p['low']['unit'] for p in pairs]
        if len(high) != 24 or len(low) != 24 or len(set(high) & set(low)) != COMMON_COUNT:
            raise RuntimeError('The two arms must share exactly sixteen units and swap eight')
        variants = []
        for name, units in [('v_matched_high_gain_24', high), ('v_matched_low_gain_24', low)]:
            item = copy.deepcopy(variant)
            item.update(name=name, selected_units=[list(unit) for unit in units],
                        residual_units_limit=24, budget_bits=1.8,
                        selection_calibration_sha256=finite.get('source_sha256'),
                        selection_mse_calibration_sha256=sha(args.mse),
                        selection_protocol='strict matched-DeltaMSE eight-pair swap, sixteen common units; calibration-only')
            variants.append(item)
        config = copy.deepcopy(template)
        config.update(start=0, count=64, variants=variants,
                      analysis_role='open-development matched-DeltaMSE mechanism intervention')
        hi, lo = arm_totals(high, data), arm_totals(low, data)
        audit.update(common_units=[list(u) for u in common], high_arm=hi, low_arm=lo,
                     mse_gain_sum_difference=hi['mse_gain_sum']-lo['mse_gain_sum'],
                     mse_gain_sum_ratio=hi['mse_gain_sum']/lo['mse_gain_sum'] if lo['mse_gain_sum'] else None,
                     finite_local_gain_sum_difference=hi['finite_local_gain_sum']-lo['finite_local_gain_sum'])
        # N>=256 covers this processor's image lengths. Actual runtime also
        # checks owner bytes and must retain all 24 units in each image.
        base = read(base_path)
        budgets = []
        for nv in (256, 768, 1792, 4352):
            fixed = (28*4*nv*50+28*4*128*2+4*28*4*128+2048
                     + 28*4*4+8*nv+4*base['parameter_count']+2*24)
            actual = fixed+24*nv*18
            bits = 16*actual/(4*28*4*nv*128)
            if bits > 1.8:
                raise RuntimeError('Twenty-four units would exceed the complete owner budget')
            budgets.append(dict(visual_tokens=nv, units=24, bits=bits, complete_owner_bytes=actual))
        audit.update(expected_owner_budgets=budgets, runtime_requirement='verify both arms retain exactly24 units and equal actual bytes per image',
                     output_config=str(args.output), output_config_sha256=hashlib.sha256(json.dumps(config, indent=2, ensure_ascii=False).encode()).hexdigest())
        write_new(args.output, config)
    else:
        audit['reason'] = 'Fewer than eight disjoint positive-DeltaMSE pairs meet the strict ratio; no runnable config written.'
    write_new(args.audit, audit)
    args.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = csv_rows(pairs)
    fields = list(rows[0]) if rows else ['pair', 'high_layer', 'high_head', 'low_layer', 'low_head', 'mse_gain_ratio', 'roi_gap']
    with args.pairs_csv.open('x', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: audit[key] for key in ('status', 'selected_pairs', 'eligible_edges', 'maximum_nonoverlapping_pairs')}))
    if audit['status'] != 'READY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
