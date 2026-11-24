"""Read-only post-hoc diagnostics of already closed Round12 development.

No model invocation, fixed500 access, algorithm selection or training occurs.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, '${PROJECT_ROOT}/code')
from metrics import score_question, METRIC_CONTRACT_VERSION

OUT = Path(__file__).resolve().parent
CLOSED = OUT.parent / 'development_final_v1/closure_attempt_v2'
PANELS = Path('${EVIDENCE_ROOT}/results/discussion/20261124_round12_proposal_v1/ROUND12_PROPOSED_DEVELOPMENT_PANELS_v1.json')
INPUTS = Path('${EVIDENCE_ROOT}/results/round12_v1/bootstrap_inputs_v1')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def rows(p):
    with Path(p).open('r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(p, value):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def mean(values):
    return statistics.mean(values) if values else None


def compact(raw, refs, expected):
    assert raw['id'] == expected['id']
    for key in ('dataset', 'source_cluster_id', 'original_image_sha256', 'question_sha256',
                'input_ids_sha256', 'full_prompt_hash', 'native_position_ids_sha256'):
        assert raw[key] == expected[key], (raw['id'], key)
    assert raw['status'] == 'PASS'
    actual = score_question(raw['dataset'], raw['raw_prediction'], refs[raw['id']])
    stages = raw.get('r7_bridge', {}).get('selected_stages', [])
    numeric = [bool(re.fullmatch(r'[+-]?\d+(?:\.\d+)?%?', str(v).strip())) for v in refs[raw['id']]['answers']]
    return {
        'id': raw['id'], 'dataset': raw['dataset'], 'score': float(actual['score']),
        'metric_id': actual['metric_id'], 'N': int(raw['actual_visual_token_count']),
        'm': len(stages), 'K_slots': sum(v[0] == 'k' for v in stages),
        'V_slots': sum(v[0] == 'v' for v in stages),
        'bits': raw['actual_effective_visual_bits'], 'bytes': raw['complete_visual_owner_bytes'],
        'full_bytes': raw['full_kv_byte_denominator'], 'prediction': raw['raw_prediction'],
        'question': raw['question'], 'references': refs[raw['id']]['answers'],
        'answer_heuristic': 'all_refs_strict_numeric' if numeric and all(numeric) else
                            'mixed_numeric_and_text' if any(numeric) else 'no_strict_numeric_refs',
    }


def comparison(label_a, label_b, domain, data):
    left, right = data[(label_a, domain)], data[(label_b, domain)]
    ids = sorted(left.keys() & right.keys())
    if not ids:
        return None
    aa, bb = [left[x] for x in ids], [right[x] for x in ids]
    deltas = [a['score'] - b['score'] for a, b in zip(aa, bb)]
    am, bm = mean([x['score'] for x in aa]), mean([x['score'] for x in bb])
    oracle = mean([max(a['score'], b['score']) for a, b in zip(aa, bb)])
    return {
        'a': label_a, 'b': label_b, 'dataset': domain, 'n': len(ids), 'ids': ids,
        'a_mean_percent': 100*am, 'b_mean_percent': 100*bm,
        'a_minus_b_pp': 100*mean(deltas),
        'a_better': sum(x > 1e-12 for x in deltas), 'b_better': sum(x < -1e-12 for x in deltas),
        'equal_score': sum(abs(x) <= 1e-12 for x in deltas),
        'different_raw_predictions': sum(a['prediction'] != b['prediction'] for a,b in zip(aa,bb)),
        'both_perfect': sum(a['score'] == 1 and b['score'] == 1 for a,b in zip(aa,bb)),
        'only_a_perfect': sum(a['score'] == 1 and b['score'] < 1 for a,b in zip(aa,bb)),
        'only_b_perfect': sum(b['score'] == 1 and a['score'] < 1 for a,b in zip(aa,bb)),
        'both_zero': sum(a['score'] == 0 and b['score'] == 0 for a,b in zip(aa,bb)),
        'posthoc_answer_selected_two_method_max_percent': 100*oracle,
        'oracle_gain_over_better_standalone_pp': 100*(oracle-max(am,bm)),
        'mean_a_bits': mean([x['bits'] for x in aa]), 'mean_b_bits': mean([x['bits'] for x in bb]),
        'mean_sum_two_banks_complete_bytes_no_sharing': mean([a['bytes']+b['bytes'] for a,b in zip(aa,bb)]),
        'oracle_is_deployable': False,
        'oracle_cost_scope': 'Requires answer-aware choice; two banks have not been implemented with sharing or measured joint time. Sum is an explicit no-sharing storage diagnostic, not the cost of a deployed method.',
    }


def main():
    source = read(CLOSED/'SOURCE_BINDINGS_v1.json')
    panel = read(PANELS)['panels']['main160']
    lookup = {r['id']: r for r in panel}
    refs = {}
    for d in ('infovqa', 'docvqa', 'textvqa'):
        refs.update({r['id']: r for r in rows(INPUTS/f'main160_{d}_refs.jsonl')})
    data, proofs = defaultdict(dict), []
    for cell in source['new_cells']:
        if cell['category'] != 'production' or cell['mode'] not in ('A','R','G8','C'):
            continue
        raw_path = Path(cell['raw'])/'records.jsonl'
        assert sha(raw_path) == cell['raw_hashes']['records.jsonl']
        assert sha(Path(cell['score'])/'summary.json') == cell['score_summary_sha256']
        scored = {r['id']: r for r in rows(Path(cell['score'])/'scored_records.jsonl')}
        loaded = 0
        for raw in rows(raw_path):
            value = compact(raw, refs, lookup[raw['id']])
            assert value['score'] == scored[raw['id']]['score']
            target = data[(cell['mode'], cell['domain'])]
            assert raw['id'] not in target
            target[raw['id']] = value
            loaded += 1
        assert loaded == cell['expected_QA']
        proofs.append({'path': str(raw_path), 'sha256': cell['raw_hashes']['records.jsonl'], 'selected_rows': loaded})
    reference_requests = defaultdict(dict)
    for item in panel:
        for mode in ('FP16','R7low','R7high'):
            binding = item['reference_'+mode]
            reference_requests[(mode, binding['file'], binding['file_sha256'])][item['id']] = item
    for (mode, path, digest), wanted in reference_requests.items():
        assert sha(path) == digest
        found = 0
        for raw in rows(path):
            if raw['id'] not in wanted:
                continue
            value = compact(raw, refs, wanted[raw['id']])
            data[(mode, raw['dataset'])][raw['id']] = value
            found += 1
        assert found == len(wanted)
        proofs.append({'path': path, 'sha256': digest, 'selected_rows': found})
    comparisons = [v for d in ('infovqa','docvqa','textvqa')
                   for a,b in itertools.combinations(('A','R','G8','C','R7low','R7high','FP16'),2)
                   if (v := comparison(a,b,d,data)) is not None]
    strata = []
    for d in ('infovqa','docvqa','textvqa'):
        rdata = data[('R',d)]
        for stratum_field in ('N','answer_heuristic'):
            for level in sorted({str(v[stratum_field]) for v in rdata.values()}):
                ids = [i for i,v in rdata.items() if str(v[stratum_field]) == level]
                for other in ('A','G8','C','R7low','R7high','FP16'):
                    shared = sorted(set(ids) & data[(other,d)].keys())
                    if not shared:
                        continue
                    delta = [rdata[i]['score']-data[(other,d)][i]['score'] for i in shared]
                    strata.append({'dataset':d,'stratum':stratum_field,'level':level,'reference':other,
                        'n':len(shared),'R_minus_reference_pp':100*mean(delta),
                        'R_better':sum(v>1e-12 for v in delta),'R_worse':sum(v<-1e-12 for v in delta),
                        'posthoc_exploratory_only':True})
    four_way = []
    for d in ('infovqa','docvqa','textvqa'):
        ids = sorted(set.intersection(*(set(data[(x,d)]) for x in ('A','R','G8','C'))))
        means = {m:100*mean([data[(m,d)][i]['score'] for i in ids]) for m in ('A','R','G8','C')}
        oracle = 100*mean([max(data[(m,d)][i]['score'] for m in ('A','R','G8','C')) for i in ids])
        four_way.append({'dataset':d,'n':len(ids),'ids':ids,'means_percent':means,
                         'four_method_answer_selected_max_percent':oracle,
                         'gain_over_best_standalone_pp':oracle-max(means.values()),'deployable':False})
    result = {'schema':'trajkv.round12.closed_development_error_patterns.v1',
        'status':'PASS_POSTHOC_READONLY_CLOSED_DEVELOPMENT', 'metric_contract':METRIC_CONTRACT_VERSION,
        'source_bindings_sha256':sha(CLOSED/'SOURCE_BINDINGS_v1.json'), 'panels_sha256':sha(PANELS),
        'new_GPU_calls':0,'new_free_answers':0,'fixed500_outputs_accessed':False,
        'no_algorithm_or_admission_changed':True, 'comparisons':comparisons,
        'four_way_common_panels':four_way,'R_strata':strata,'source_checks':proofs,
        'warnings':['All analyses are post-hoc and development-only; small strata are not confirmation.',
                    'Answer-type groups use reference-string regex, not human semantic annotation.',
                    'Answer-selected unions are unattainable oracle diagnostics, not a method or upper bound on all KV compression.',
                    'Density N and m are deterministically linked by the budget formula; their associations are confounded.']}
    write(OUT/'ERROR_PATTERNS_v1.json',result)
    with (OUT/'COMPACT_ANALYSIS_ROWS_v1.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        for (method,d), values in sorted(data.items()):
            for i,row in sorted(values.items()):
                f.write(json.dumps({'method':method,**row},ensure_ascii=False)+'\n')
    print(json.dumps({'status':result['status'],'comparisons':len(comparisons),'four_way':four_way},ensure_ascii=False))


if __name__ == '__main__':
    main()
