"""CPU-only R8 quality analysis, preserving frozen R7 scoring/statistics semantics.

One or more methods, primary sessions or second-model real-N first questions.
Running/incomplete inputs return NOT_COMPLETE_NO_SCORING before opening labels.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/analysis'))
from analyze_external import score_docvqa, PROTOCOLS
from analyze_cycle import cluster_bootstrap
from analyze_docvqa_confirm64 import holm
from textvqa_metric import TextVQAAccuracyEvaluator
import round8_quality_owner as r8_owner
from round8_quality_owner import validate_sample_source
from costs import VISUAL_KEYS, visual_owner as canonical_visual_owner, role as canonical_role


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def same(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-10)


def local_path(value):
    return r8_owner.local_path(value)


def csv_write(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value
                             for key, value in row.items()} for row in rows)


def selected_references(path, ids):
    """Decode selected reference objects only, including a shared JSONL bank."""
    text = Path(path).read_text(encoding='utf-8-sig')
    decoder, result = json.JSONDecoder(), {}
    for sid in sorted(ids):
        matches = list(re.finditer(r'"id"\s*:\s*' + re.escape(json.dumps(sid)), text))
        require(len(matches) == 1, 'Missing/duplicate selected reference: ' + sid)
        start = text.rfind('{', 0, matches[0].start())
        value, _ = decoder.raw_decode(text, start)
        require(value['id'] == sid, 'Selected reference object framing mismatch')
        result[sid] = value
    return result


def factor_values_hash(data):
    fields = {key: data.get(key) for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets',
                                           'key_gain_mode', 'value_group_size', 'parameter_count')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def independent_plan(variant, layers, heads, nv, base_position, factor_bytes, metric_bytes, words, full):
    """Scalar legal-prefix oracle, independent of the runtime planner module."""
    explicit = variant.get('selected_stages') is not None
    entries = [tuple(x) for x in variant.get('selected_stages' if explicit else 'selected_units', [])]
    seen = set()
    ids = []
    for entry in entries:
        require(len(entry) == (4 if explicit else 3), 'Malformed ranking entry')
        axis, layer, head = entry[:3]
        require(axis in ('k', 'v') and 0 <= layer < layers and 0 <= head < heads and entry not in seen,
                'Illegal/duplicate ranking entry')
        if explicit:
            require(entry[3] in (0, 1) and (entry[3] == 0 or (*entry[:3], 0) in seen), 'Illegal stage prefix')
        seen.add(entry)
        value = ((int(axis == 'v') * layers + layer) * heads + head)
        ids.append(value * 2 + entry[3] if explicit else value)
    ranking = struct.pack('<' + 'H' * len(ids), *ids)
    policy = 'explicit' if explicit else variant.get('depth_policy', 'incumbent')
    require(policy in ('explicit', 'incumbent', 'v_pair', 'v_mixed'), 'Unknown residual policy')
    fraction = {'explicit': 0., 'incumbent': 0., 'v_pair': 1., 'v_mixed': float(variant.get('v_mixed_pair_fraction', .5))}[policy]
    descriptor = struct.pack('<Bf', {'incumbent': 0, 'v_pair': 1, 'v_mixed': 2, 'explicit': 3}[policy], fraction)
    fraction = struct.unpack('<f', descriptor[1:])[0]
    lookup, stage_bytes = 16 * layers * heads, 18 * nv
    allowed = math.floor(float(variant.get('budget_bits', 1.8)) * full / 16)
    actual_fixed = base_position + factor_bytes + metric_bytes + lookup + len(ranking) + len(descriptor)
    reference = variant.get('residual_layout_reference') == 'k2'
    if reference:
        require(not explicit and len(entries) == 2 * layers * heads, 'K2 layout requires full named ranking')
        reference_base = base_position + layers * heads * nv * (32 - (16 if words == 16 else 32)) + 2048 - words * 8
        plan_base, plan_factors = reference_base, factor_bytes
        descriptor = bytes((descriptor[0] | 128,)) + descriptor[1:]
    else:
        reference_base = None
        plan_base, plan_factors = base_position, factor_bytes + metric_bytes
    fixed = plan_base + plan_factors + lookup + len(ranking) + 5
    require(actual_fixed <= allowed and fixed <= allowed, 'Fixed complete owner exceeds cap')
    affordable = (allowed - fixed) // stage_bytes
    actual_affordable = (allowed - actual_fixed) // stage_bytes
    limit = variant.get('res_stages_limit', variant.get('res_units_limit', variant.get('residual_units_limit')))
    if limit is not None:
        affordable, actual_affordable = min(affordable, max(0, int(limit))), min(actual_affordable, max(0, int(limit)))
    if explicit:
        selected = entries[:affordable]
        incumbent_cap = None
    else:
        old_fixed = plan_base + plan_factors + layers * 2 * heads * 4 + 4 * layers * heads
        incumbent_cap = min(len(entries), max(0, (allowed - old_fixed) // stage_bytes))
        key = [unit for unit in entries[:incumbent_cap] if unit[0] == 'k']
        values = [unit for unit in entries if unit[0] == 'v']
        require(len(key) <= affordable, 'Fixed incumbent K subset exceeds cap')
        vb = min(affordable - len(key), len(values))
        selected = [(*unit, 0) for unit in key]
        if policy == 'incumbent':
            selected += [(*unit, 0) for unit in values[:vb]]
        elif policy == 'v_pair':
            selected += [(*unit, stage) for unit in values[:vb // 2] for stage in (0, 1)]
            if vb % 2:
                selected.append((*values[vb // 2], 0))
        else:
            pairs = min(len(values), int(vb * fraction) // 2)
            width = min(len(values), vb - pairs)
            pairs = min(width, vb - width)
            selected += [(*unit, 0) for unit in values[:width]] + [(*unit, 1) for unit in values[:pairs]]
    require(len(selected) <= actual_affordable, 'Preserved K2 mask exceeds actual owner cap')
    return dict(stages=selected, fixed=actual_fixed, complete=actual_fixed + len(selected) * stage_bytes,
        allowed=allowed, lookup=lookup, ranking=len(ranking), policy=5, stage_bytes=stage_bytes,
        ranking_sha256=hashlib.sha256(ranking).hexdigest(), policy_sha256=hashlib.sha256(descriptor).hexdigest(),
        k2_reference=reference, reference_base=reference_base,
        reference_complete=fixed + len(selected) * stage_bytes, reference_affordable=affordable,
        actual_affordable=actual_affordable, incumbent_cap=incumbent_cap)


from round8_quality_owner import audit_owner


def paired_stats(left, right, repeats):
    require(set(left) == set(right), 'Paired methods do not have identical question identities')
    groups = defaultdict(list)
    for key in sorted(left):
        groups[key[0]].append(float(left[key] - right[key]))
    values = [(source, value) for source, differences in groups.items() for value in differences]
    estimate = cluster_bootstrap(values, repeats=repeats)
    sums = np.asarray([sum(v) for v in groups.values()], dtype=np.float64)
    observed = abs(float(sums.sum()))
    if len(groups) <= 16:
        indexes = np.arange(1 << len(groups), dtype=np.uint64)
        signs = (((indexes[:, None] >> np.arange(len(groups), dtype=np.uint64)) & 1).astype(np.float64) * 2 - 1)
        p = float(np.mean(np.abs(signs @ sums) >= observed - 1e-12))
        method, draws = 'exact source-cluster sign enumeration', len(indexes)
    else:
        rng, extreme, draws = np.random.default_rng(20261116), 0, 100000
        for start in range(0, draws, 2048):
            signs = rng.integers(0, 2, size=(min(2048, draws - start), len(groups)), dtype=np.int8) * 2 - 1
            extreme += int((np.abs(signs @ sums) >= observed - 1e-12).sum())
        p, method = (extreme + 1) / (draws + 1), 'Monte Carlo source-cluster sign flip, plus-one correction'
    return dict(delta=estimate['mean'], ci_low=estimate['ci_lower'], ci_high=estimate['ci_upper'], p=p,
        sources=len(groups), questions=len(left), bootstrap_status=estimate['status'], signflip_method=method,
        signflip_draws=draws, scope='question-weighted paired effect; complete original-source clusters are resampled/flipped together')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--dataset', choices=('doc', 'text', 'infovqa'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-variants', type=int, required=True)
    p.add_argument('--analysis-role', required=True)
    p.add_argument('--bootstrap-repeats', type=int, default=8000)
    p.add_argument('--budget-bits', type=float, default=1.8)
    p.add_argument('--candidate')
    p.add_argument('--variants', nargs='+', help='Analysis-only method filter; original run manifests and full record coverage remain verified')
    p.add_argument('--quiet', action='store_true', help='Print compact completion metadata only; complete tables/comparisons remain in output files')
    p.add_argument('--artifact-map', type=Path, help='Optional JSON files/prefixes map; preserve original execution hashes')
    args = p.parse_args()
    r8_owner.configure_path_map(args.artifact_map)
    folders = []
    for value in args.runs:
        folder = Path(value)
        if not folder.is_absolute():
            folder = ROOT / folder if (ROOT / folder).is_dir() else ROOT / 'results' / folder
        for child in ('quality', 'transfer32'):
            if (folder / child / 'records.jsonl').is_file():
                folder = folder / child
                break
        complete = folder / 'complete.json'
        if not complete.is_file() or read(complete).get('status') != 'COMPLETE':
            print(json.dumps(dict(status='NOT_COMPLETE_NO_SCORING', run=str(folder), references_opened=False, output_written=False)))
            return
        folders.append(folder)
    all_samples_list = read(args.samples)
    all_samples = {row['id']: row for row in all_samples_list}
    require(all_samples and len(all_samples) == len(all_samples_list), 'Duplicate/empty supplied samples')
    manifests = [read(folder / 'manifest.json') for folder in folders]
    selected_ids = set()
    variants, records, seen, receipts, cache = {}, [], set(), [], {}
    for folder, manifest in zip(folders, manifests):
        if any(v.get('method') == 'round8_joint' for v in manifest['config']['variants']):
            post_path = folder / 'complete_round8.json'
            if not post_path.is_file():
                print(json.dumps(dict(status='NOT_COMPLETE_NO_SCORING', run=str(folder), reason='round8 wrapper postcheck pending', references_opened=False, output_written=False)))
                return
            post = read(post_path)
            require(post.get('status') == 'COMPLETE' and post.get('encoder_inputs_unchanged') is True
                    and post['engine_complete_sha256'] == sha(folder / 'complete.json')
                    and post['manifest_sha256'] == sha(folder / 'manifest.json'), 'Round8 producer postcheck changed/failed')
        require(manifest.get('answers_visible_to_runner') is False, 'Inference manifest does not establish answer-free generation')
        selected = manifest['selected_ids']
        require(selected and len(selected) == len(set(selected)) and set(selected) <= set(all_samples), 'Manifest source IDs differ from supplied samples')
        source_check = validate_sample_source(manifest, args.samples, selected)
        local_variants = {v['name']: v for v in manifest['config']['variants']}
        require(len(local_variants) == len(manifest['config']['variants']), 'Duplicate variant names')
        for name, variant in local_variants.items():
            require(name not in variants or variants[name] == variant, 'Same variant name has different settings across runs')
            variants[name] = variant
        external = manifest.get('external_codec_inputs', {})
        for value, expected in external.items():
            require(local_path(value).is_file() and sha(local_path(value)) == expected, 'External codec evidence missing/changed: ' + value)
        local_seen = set()
        for line in (folder / 'records.jsonl').read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            sid, name = record['session_id'], record['variant']
            key = (sid, name)
            require(key not in seen and sid in selected and name in local_variants, 'Duplicate/unexpected session/variant')
            require(record['variant_config'] == local_variants[name], 'Record settings differ from frozen manifest')
            require(record.get('task_accuracy_eligible') is not False and record.get('primary_quality_eligible') is not False,
                    'Diagnostic record cannot enter task quality')
            require(record['image_id'] == all_samples[sid]['image_id'], 'Record image identity mismatch')
            protocol = record['protocol']
            require(protocol in ('image_boundary_free_running', 'first_question_image_boundary_free_running', 'native_hf_first_question_free_running'), 'Unsupported generation protocol')
            if record['method'] != 'native_hf':
                require(record.get('read_ledger_exact') is True and record.get('global_extra_budget_ok', True), 'Read ledger/budget audit failed')
                total = sum(int(e.get('selection_bytes', 0)) + int(e.get('residual_bytes', 0)) for e in record.get('read_events', []))
                if 'extra_logical_read_bytes' in record:
                    require(total == int(record['extra_logical_read_bytes']), 'Logical extra-read ledger sum mismatch')
            cost = audit_owner(record, local_variants[name], external, cache)
            record['_round7_owner_audit'] = cost
            record['_round7_first_question_only'] = protocol != 'image_boundary_free_running'
            records.append(record)
            seen.add(key); local_seen.add(key)
        require(local_seen == {(sid, name) for sid in selected for name in local_variants}, 'Incomplete per-run source x variant coverage')
        selected_ids.update(selected)
        complete = read(folder / 'complete.json')
        for key in ('records', 'expected'):
            if key in complete:
                require(int(complete[key]) == len(local_seen), 'Producer completion count differs from its complete original records')
        receipts.append(dict(run=str(folder), manifest_sha256=sha(folder / 'manifest.json'), records_sha256=sha(folder / 'records.jsonl'),
            complete_sha256=sha(folder / 'complete.json'), config_sha256=manifest.get('config_sha256'),
            sample_source_check=source_check, external_codec_inputs=external,
            producer_recorded_source_hashes=manifest.get('sources', {}),
            source_inventory_scope='producer-recorded source inventory; not proof every listed module executed',
            original_selected_sessions=len(selected), original_variant_names=list(local_variants), original_record_count=len(local_seen),
            model_geometry=manifest.get('geometry'), model_config_sha256=manifest.get('model_config_sha256')))
    producer_variant_names, producer_record_count, producer_selected_sessions = list(variants), len(records), len(selected_ids)
    if args.variants:
        require(len(args.variants) == len(set(args.variants)) and set(args.variants) <= set(variants), 'Duplicate/missing requested analysis-filter variants')
        variants = {name: variants[name] for name in args.variants}
        records = [record for record in records if record['variant'] in variants]
        seen = {(record['session_id'], record['variant']) for record in records}
        selected_ids = {record['session_id'] for record in records}
    require(len(variants) == args.expected_variants and selected_ids
            and seen == {(sid, name) for sid in selected_ids for name in variants}, 'Incomplete analysis source union x expected filtered methods')
    for receipt in receipts:
        original = set(receipt['original_variant_names'])
        receipt['analysis_selected_variant_names'] = [name for name in variants if name in original]
        receipt['analysis_selected_record_count'] = receipt['original_selected_sessions'] * len(receipt['analysis_selected_variant_names'])
    refs = selected_references(args.references, selected_ids)
    metric = TextVQAAccuracyEvaluator() if args.dataset == 'text' else None
    if metric:
        for count, expected in [(0, 0.), (1, .3), (2, .6), (3, .9), (4, 1.)]:
            require(same(metric.eval_pred_list([dict(pred_answer='two', gt_answers=['2'] * count + ['other'] * (10 - count))]), expected), 'Official TextVQA metric control failed')
    scores, by, costs = [], defaultdict(dict), defaultdict(list)
    for record in records:
        sid, name = record['session_id'], record['variant']
        sample, ref = all_samples[sid], refs[sid]
        first = record['_round7_first_question_only']
        expected_turns = {1} if first else set(range(1, min(int(sample['turns']), int(next(m['config']['arguments'].get('turns', sample['turns']) for m in manifests if sid in m['selected_ids']))) + 1))
        actual_turns = [int(turn['turn']) for turn in record['turns']]
        require(len(actual_turns) == len(set(actual_turns)) and set(actual_turns) == expected_turns, 'Missing/duplicate expected free-answer turns')
        cluster = str(sample.get('source_document_id') or sample.get('source_original_image_id') or sample['image_id'])
        if args.dataset in ('doc', 'infovqa'):
            require(sample.get('dataset') in (('docvqa_grouped',) if args.dataset == 'doc' else ('infovqa', 'infographicvqa')), 'Expected DocVQA source')
            targets = {int(t['turn']): t.get('answer', t.get('answers')) for t in ref['references']}
            if ref.get('source_document_id'):
                require(str(ref['source_document_id']) == str(sample.get('source_document_id')), 'Reference source document identity mismatch')
        else:
            require(sample.get('dataset') == 'textvqa' and expected_turns == {1}
                    and ref['image_id'] == sample['image_id'] and len(ref['answers']) == 10, 'Invalid TextVQA one-question/ten-label source')
            targets = {1: ref['answers']}
        for turn in record['turns']:
            index, answer = int(turn['turn']), turn['answer']
            require(isinstance(answer, str) and index in targets, 'Missing reference/free answer')
            if args.dataset in ('doc', 'infovqa'):
                result = score_docvqa(answer, targets[index])
                value, exact, normalized = result['score'], result['exact'], result['normalized_prediction']
            else:
                value = float(metric.eval_pred_list([dict(pred_answer=answer, gt_answers=targets[index])]))
                normalized = metric.answer_processor(answer)
                exact = float(normalized in [metric.answer_processor(x) for x in targets[index]])
            scores.append(dict(session_id=sid, source_cluster=cluster, image_id=sample['image_id'], variant=name,
                turn=index, score=value, exact=exact, prediction=answer, normalized_prediction=normalized,
                reference=json.dumps(targets[index], ensure_ascii=False), visual_tokens=record['visual_tokens'],
                question_scope='first_question_only' if first else 'all_declared_session_turns', analysis_role=args.analysis_role))
            by[name][(cluster, sid, index)] = value
        costs[name].append(record)
    table = []
    for name in variants:
        rows = [row for row in scores if row['variant'] == name]
        estimate = cluster_bootstrap([(row['source_cluster'], row['score']) for row in rows], repeats=args.bootstrap_repeats)
        group, owner_rows = costs[name], [r['_round7_owner_audit'] for r in costs[name]]
        denominator = sum(row['visual_fp16_denominator'] for row in owner_rows)
        total = sum(row['complete_visual_owner_bytes'] for row in owner_rows)
        encoder = sum(row['encoder_parameters_bytes'] for row in owner_rows)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['source_cluster']].append(row['score'])
        table.append(dict(variant=name, method=variants[name]['method'], role=owner_rows[0]['role'], sessions=len(group),
            sources=len(grouped), questions=len(rows), score=estimate['mean'], ci_low=estimate['ci_lower'], ci_high=estimate['ci_upper'],
            source_macro_mean=float(np.mean([np.mean(x) for x in grouped.values()])), exact=float(np.mean([row['exact'] for row in rows])),
            anls=estimate['mean'] if args.dataset in ('doc', 'infovqa') else None, accuracy=estimate['mean'] if args.dataset == 'text' else None,
            budget_bits_including_encoder_constants=16 * total / denominator,
            maximum_per_session_complete_bits=max(x['actual_visual_bits'] for x in owner_rows),
            complete_visual_bytes_total=total, encoder_constants_bytes_per_session=encoder / len(group),
            key_metric_bytes_per_session=max(x['key_metric_bytes'] for x in owner_rows),
            physical_tail_storage_bytes_total=sum(x['physical_tail_storage_bytes'] for x in owner_rows),
            eligible_lowbit_frontier=all(x['role'] == 'actual packed runtime' and x['actual_packed_owner_reconstructed']
                and x['actual_visual_bits'] <= args.budget_bits + 1e-12 for x in owner_rows),
            visual_tokens_min=min(x['visual_tokens'] for x in owner_rows), visual_tokens_max=max(x['visual_tokens'] for x in owner_rows),
            first_question_only=all(x['_round7_first_question_only'] for x in group),
            session_seconds_median=float(np.median([r['session_ms'] / 1000 for r in group])), analysis_role=args.analysis_role))
    pairs = [(args.candidate, x) for x in variants if x != args.candidate] if args.candidate else list(itertools.combinations(variants, 2))
    require(not args.candidate or args.candidate in variants, 'Requested candidate missing')
    comparisons = [dict(candidate=left, comparator=right, **paired_stats(by[left], by[right], args.bootstrap_repeats)) for left, right in pairs]
    for row, adjusted in zip(comparisons, holm([row['p'] for row in comparisons])):
        row['holm_reported_comparisons'] = adjusted
        row['comparison_family'] = 'requested candidate versus others' if args.candidate else 'exploratory all pairs'
    args.output.mkdir(parents=True, exist_ok=False)
    audits = [record['_round7_owner_audit'] for record in records]
    for filename, data in [('scores.csv', scores), ('table.csv', table), ('comparisons.csv', comparisons), ('owner_audit.csv', audits)]:
        csv_write(args.output / filename, data)
    summary = dict(status='COMPLETE', dataset=args.dataset, analysis_role=args.analysis_role, table=table, comparisons=comparisons,
        selected_sessions=len(selected_ids), supplied_samples=len(all_samples), variants=len(variants), records=len(records),
        analysis_method_filter=dict(applied=bool(args.variants), selected_variants=list(variants),
            producer_variant_names=producer_variant_names, producer_record_count=producer_record_count,
            producer_selected_sessions=producer_selected_sessions,
            scope='post hoc analysis selection only; no new execution manifest; each original run retains its original complete counts and fingerprints'),
        metric=PROTOCOLS['docvqa'] if args.dataset in ('doc', 'infovqa') else 'official TextVQA normalized ten-annotator leave-one-out soft accuracy',
        statistics='question-weighted source-cluster bootstrap; paired cluster sign flips; Holm over reported comparisons; source-macro descriptive mean also reported',
        scope='R8 independent actual-owner and inherited scoring/statistics; caller supplies honest exposed-dev/fresh-source scope; no blind or speed claim from filenames',
        samples_sha256=sha(args.samples), references_sha256=sha(args.references), analyzer_sha256=sha(__file__), round8_owner_helper_sha256=sha(Path(__file__).with_name('round8_quality_owner.py')),
        audit=dict(status='PASS', complete_expected_coverage=True, complete_owner_reconstructed=True,
            total_unique_never_added_to_visual_owner=True, encoder_json_file_size_never_charged_as_tensor_payload=True,
            physical_tail_bits_charged_without_semantic_padding=True,
            rank1_metric_shape_payload_bytes=29120 if any(x['key_metric_bytes'] == 29120 for x in audits) else None,
            k2_reference_masks_verified=sum(x['k2_reference_mask_verified'] is True for x in audits), runs=receipts))
    for name, value in [('summary.json', summary), ('complete.json', dict(status='COMPLETE', records=len(records), variants=len(variants)))]:
        with (args.output / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    compact = dict(status='COMPLETE', output=str(args.output), variants=len(variants), sources=table[0]['sources'],
                   scored_answers=len(scores), producer_records_verified=producer_record_count,
                   analysis_filter_applied=bool(args.variants))
    if not args.quiet:
        compact['table'] = [{key: row[key] for key in ('variant', 'score', 'sources', 'questions', 'budget_bits_including_encoder_constants')} for row in table]
    print(json.dumps(compact))


if __name__ == '__main__':
    main()
