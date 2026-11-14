"""Complete matched round5 analysis, with an explicit development/confirm role.

Score real free-running sessions, including explicitly labelled materialized
mechanism ports. Their retained experimental owners are part of actual bytes.
Gaussian/T0 reference statistics are deliberately not treated as task scores.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/analysis'))
from analyze_external import score_docvqa
from analyze_cycle import cluster_bootstrap
from analyze import paired, write_csv
from analyze_docvqa_confirm64 import holm
from costs import visual_owner, role


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def actual_visual_owner(record):
    """Include this experiment's extra owner, absent from frozen costs.py."""
    base, _ = visual_owner(record)
    return base + int(record['owned_initial'].get('spectrum_experiment_reference', 0))


def validate_sample_source(manifest, samples_path, selected_ids):
    if manifest['samples_sha256'] == sha(samples_path):
        return 'same_input_manifest'
    # A runner may read full dev64 and select a shard while this analyzer takes
    # the exact smaller subset manifest. Verify the original file when local.
    source = manifest['config']['arguments']['samples']
    normalized = source.replace('\\', '/')
    marker = '/TrajKV/'
    local = ROOT / normalized.split(marker, 1)[1] if marker in normalized else Path(source)
    if not local.is_file() or sha(local) != manifest['samples_sha256']:
        raise ValueError('Cannot verify runner sample source against the supplied subset')
    original = {r['id']: r for r in load(local)}
    subset = {r['id']: r for r in load(samples_path)}
    for sid in selected_ids:
        if sid not in subset or sid not in original or subset[sid] != original[sid]:
            raise ValueError('Subset differs from the verified original sample: ' + sid)
    return 'verified_subset_of_original_input'


def spectrum_summary(records):
    buckets = defaultdict(list)
    for record in records:
        for row in record.get('spectrum_score_statistics', []):
            buckets[(record['variant'], row['phase'])].append(row)
    result = []
    for (variant, phase), rows in sorted(buckets.items()):
        total = lambda key: sum(float(r.get(key, 0)) for r in rows)
        elements = total('score_elements')
        queries = total('sampled_queries')
        error, reference, centered = [total(k) for k in (
            'score_error_sse', 'score_reference_sse', 'score_reference_centered_sse')]
        result.append(dict(variant=variant, phase=phase, sampled_events=len(rows),
            sampled_queries=queries, score_elements=elements, score_error_sse=error,
            score_reference_sse=reference, score_reference_centered_sse=centered,
            score_error_mse=error/elements if elements else None,
            score_relative_mse=error/reference if reference else None,
            score_centered_relative_mse=error/centered if centered else None,
            top1_same_fraction=sum(float(r['top1_same_fraction'])*float(r['sampled_queries'])
                for r in rows)/queries if queries else None,
            scope='same student q, original visual K, final position per sampled event; descriptive development metric'))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--references', type=Path, default=ROOT/'configs/data/docvqa_next_references.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-variants', type=int, required=True)
    p.add_argument('--analysis-role',choices=['development','confirmation'],default='development')
    p.add_argument('--baselines', nargs='+', default=[
        'full8', 'native4_8', 'base2_8', 'cold_meta_all_kv', 'cold_meta_g128_kv', 'fwht_global'])
    p.add_argument('--bootstrap-repeats', type=int, default=8000)
    a = p.parse_args()
    samples = load(a.samples)
    source_rows = {r['id']: r for r in samples}
    if not samples or len(source_rows) != len(samples):
        raise ValueError('Samples must be nonempty and unique by session ID')
    ids = set(source_rows)
    references = {}
    for line in a.references.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['id'] in ids:
            if row['id'] in references:
                raise ValueError('Duplicate reference session')
            references[row['id']] = {int(t['turn']): t['answer'] for t in row['references']}
    if set(references) != ids:
        raise ValueError('Missing references for supplied subset')

    records, manifests = [], []
    seen, variants, by, costs = set(), {}, defaultdict(dict), defaultdict(list)
    scored = []
    for name in a.runs:
        folder = Path(name)
        if not folder.is_absolute():
            folder = ROOT / 'results' / folder
        if load(folder/'complete.json')['status'] != 'COMPLETE':
            raise ValueError('Run is not complete: ' + str(folder))
        quality = folder/'quality' if (folder/'quality').is_dir() else folder
        manifest = load(quality/'manifest.json')
        config_variants = {v['name']: v for v in manifest['config']['variants']}
        selected = manifest['selected_ids']
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= ids:
            raise ValueError('Run selected IDs differ from the analysis subset')
        check = validate_sample_source(manifest, a.samples, selected)
        manifests.append(dict(run=str(folder), manifest_sha256=sha(quality/'manifest.json'),
            records_sha256=sha(quality/'records.jsonl'), sample_source_check=check))
        for name_, value in config_variants.items():
            if name_ in variants and variants[name_] != value:
                raise ValueError('Same variant name has different configs across runs: ' + name_)
            variants[name_] = value
        run_seen = set()
        for line in (quality/'records.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sid, variant = r['session_id'], r['variant']
            key = (sid, variant)
            if sid not in selected or variant not in config_variants or key in seen:
                raise ValueError('Unexpected or duplicate session/variant: ' + str(key))
            if r['protocol'] != 'image_boundary_free_running' or r.get('task_accuracy_eligible') is False:
                raise ValueError('Run is not eligible free-running answer data')
            if r['variant_config'] != config_variants[variant] or not r.get('read_ledger_exact') or not r.get('global_extra_budget_ok', True):
                raise ValueError('Run config/read ledger check failed')
            if {int(t['turn']) for t in r['turns']} != set(references[sid]) or len(r['turns']) != len(references[sid]):
                raise ValueError('Incomplete or duplicate answer turns for ' + sid)
            sample = source_rows[sid]
            document = str(sample.get('source_document_id') or sample.get('image_id') or r['image_id'])
            seen.add(key); run_seen.add(key); records.append(r); costs[variant].append(r)
            for turn in r['turns']:
                index = int(turn['turn'])
                targets = references[sid][index]
                result = score_docvqa(turn['answer'], targets)
                row = dict(session_id=sid, image_id=r['image_id'], source_document_id=document,
                    variant=variant, turn=index, score=result['score'], exact=result['exact'],
                    prediction=turn['answer'], reference=json.dumps(targets, ensure_ascii=False),
                    numeric=any(any(c.isdigit() for c in target) for target in targets),
                    visual_tokens=r['visual_tokens'], analysis_role=a.analysis_role)
                scored.append(row)
                by[variant][(document, sid, index)] = result['score']
        if run_seen != {(sid, v) for sid in selected for v in config_variants}:
            raise ValueError('Run does not completely cover its manifest IDs and variants')
    if len(costs) != a.expected_variants or seen != {(sid, v) for sid in ids for v in costs}:
        raise ValueError('Incomplete subset x expected variant coverage')

    table = []
    for variant, rows in sorted(costs.items()):
        answers = [r for r in scored if r['variant'] == variant]
        estimate = cluster_bootstrap(((r['source_document_id'], r['score']) for r in answers),
            repeats=a.bootstrap_repeats)
        denominator = sum(r['owned_initial']['visual_fp16_denominator'] for r in rows)
        owner = sum(actual_visual_owner(r) for r in rows)
        encoder = sum(r.get('encoder_calibration_parameters_bytes', 0) for r in rows)
        experiment = sum(r['owned_initial'].get('spectrum_experiment_reference', 0) for r in rows)
        first = rows[0]
        materialized = any(role(r) == 'materialized quality prototype' for r in rows)
        table.append(dict(variant=variant, analysis_role=a.analysis_role, images=len(rows),
            source_documents=len({r['source_document_id'] for r in answers}), questions=len(answers),
            anls=estimate['mean'], ci_low=estimate['ci_lower'], ci_high=estimate['ci_upper'],
            exact=float(np.mean([r['exact'] for r in answers])),
            effective_visual_bits=16*owner/denominator,
            budget_bits_including_encoder_constants=16*(owner+encoder)/denominator,
            visual_bytes_total=owner, encoder_constants_bytes_per_session=encoder/len(rows),
            spectrum_experiment_reference_bytes_total=experiment,
            materialized_quality_port=materialized, role=role(first),
            eligible_lowbit_frontier=role(first) == 'actual packed runtime' and not materialized
                and all(r.get('lowbit_deployment_candidate', True) for r in rows)
                and all(16*(actual_visual_owner(r)+r.get('encoder_calibration_parameters_bytes',0))
                    /r['owned_initial']['visual_fp16_denominator'] <= 1.8+1e-12 for r in rows),
            extra_reads_total=sum(r['extra_logical_read_bytes'] for r in rows),
            logical_cache_reads_total=sum(sum(sum(e.get(k, 0) for k in
                ('base_visual_bytes', 'residual_bytes', 'text_bytes', 'reference_bytes', 'selection_bytes', 'position_bytes'))
                for e in r['read_events']) for r in rows),
            free_session_seconds_median=float(np.median([r['session_ms']/1000 for r in rows])),
            peak_allocated_max=max(r['peak_allocated'] for r in rows),
            spectrum_transform=first.get('spectrum_transform'), spectrum_rank=first.get('spectrum_rank'),
            spectrum_encoding=first.get('spectrum_encoding'),
            spectrum_artifact_sha256=first.get('spectrum_artifact_sha256'),
            functional_v_channel_groups=first.get('functional_v_channel_groups'),
            visual_bias_bytes=first.get('visual_bias_bytes', 0)))

    pairs = {(variant, baseline) for variant in by for baseline in a.baselines if baseline in by and variant != baseline}
    # Also give direct same-rank mechanism comparisons, independent of names.
    for left in table:
        for right in table:
            if left['spectrum_transform'] and right['spectrum_transform'] and left['variant'] != right['variant']:
                if (left['spectrum_rank'], left['spectrum_encoding']) == (right['spectrum_rank'], right['spectrum_encoding']):
                    if left['spectrum_transform'].startswith('query') and not right['spectrum_transform'].startswith('query'):
                        pairs.add((left['variant'], right['variant']))
    functional = {(r['functional_v_channel_groups'], bool(r['visual_bias_bytes'])): r['variant']
        for r in table if r['functional_v_channel_groups'] is not None}
    for left, right in [((8, False), (1, False)), ((8, True), (1, True)),
                        ((1, True), (1, False)), ((8, True), (8, False))]:
        if left in functional and right in functional:
            pairs.add((functional[left], functional[right]))
    comparisons = []
    for left, right in sorted(pairs):
        if by[left].keys() != by[right].keys():
            raise ValueError('Paired candidates do not share identical inputs')
        comparisons.append(dict(variant=left, baseline=right, analysis_role=a.analysis_role,
            cluster_unit='source_document_id', **paired(by[left], by[right], repeats=a.bootstrap_repeats)))
    adjusted = holm([r['p'] for r in comparisons]) if comparisons else []
    for row, value in zip(comparisons, adjusted):
        row['holm_all_exploratory_comparisons'] = value
    continuous = spectrum_summary(records)
    a.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('scores.csv', scored), ('table.csv', table),
                       ('compare_baselines.csv', comparisons), ('comparisons.csv', comparisons),
                       ('spectrum_score_statistics.csv', continuous)]:
        write_csv(a.output/name, rows)
        if not rows:
            (a.output/name).write_text('', encoding='utf-8')
    summary = dict(status='COMPLETE', analysis_role=a.analysis_role,
        scope=('reused development documents' if a.analysis_role=='development' else 'frozen new-source confirmation documents')+'; complete matched coverage; materialized ports are mechanism references, not lowbit deployment claims',
        images=len(ids), source_documents=len({r['source_document_id'] for r in scored}),
        sessions=len(records), variants=len(costs), scored_answers=len(scored),
        questions_per_variant=sum(len(x) for x in references.values()),
        samples_sha256=sha(a.samples), references_sha256=sha(a.references),
        analyzer_sha256=sha(__file__), runs=manifests, table=table,
        comparisons=comparisons, spectrum_score_statistics=continuous,
        cost_scope='actual visual owner plus spectrum_experiment_reference; folded encoder constants charged once; visual bias already in runtime owner',
        uncertainty_scope='question-weighted source-document cluster bootstrap and document-paired sign flip; exploratory comparisons',
        time_scope='descriptive free-session medians with differing output lengths, no fixed-output speed or HBM claim')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE', images=len(ids), sessions=len(records), table=table)))


if __name__ == '__main__':
    main()
