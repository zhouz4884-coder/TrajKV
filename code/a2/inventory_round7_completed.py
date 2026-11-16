"""CPU workload inventory from completed raw producers, never partial predicts.

Analysis outputs/queue parents are excluded. Repeated physical runs count as
work, not new independent sources; declared logical aliases are never expanded.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREE_PROTOCOLS = {'image_boundary_free_running', 'first_question_image_boundary_free_running',
                  'native_hf_first_question_free_running', 'native_history_free_running'}
MAIN_PARENTS = ('confirm_s0', 'confirm_s1', 'second_confirm64')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def complete(path):
    return path.is_file() and read(path).get('status') in ('COMPLETE', 'PASS')


def kind(manifest, raw_name, row):
    schema = str(manifest.get('schema', ''))
    if raw_name == 'timing_records.jsonl':
        if row.get('quality') != 'not_scored' or 'fixed_tokens' not in row.get('runner_result', {}).get('protocol', ''):
            raise ValueError('Non-forced record in a quiet timing producer')
        return 'quiet_forced_trial'
    if 'instrumented_profile' in schema or 'component_profile' in row:
        return 'instrumented_profile'
    if 'key_metric_collect' in schema or 'actual_query_error_diagnostics' in row:
        return 'metric_query_diagnostic'
    if 'conditional_exchange' in schema:
        return 'cached_prefix_calibration'
    if row.get('protocol') in FREE_PROTOCOLS and isinstance(row.get('turns'), list):
        return 'diagnostic_free_generation' if row.get('task_accuracy_eligible') is False else 'free_generation'
    raise ValueError('Unrecognized raw producer type: ' + schema + ' / ' + raw_name)


def dataset_name(value):
    text = str(value or 'unknown').lower()
    return 'doc' if text.startswith('doc') else 'text' if text.startswith('text') else text


def manifest_images(manifest):
    result = {}
    for entry in manifest.get('selected_inputs', []):
        if entry.get('id'):
            result[entry['id']] = entry.get('source_image_id', entry.get('image_id'))
    return result


def summarize_producer(folder, manifest, receipt, raw_name):
    raw = rows(folder / raw_name)
    if not raw:
        raise ValueError('Empty completed raw producer: ' + str(folder))
    types = {kind(manifest, raw_name, row) for row in raw}
    if len(types) != 1:
        raise ValueError('Mixed producer record kinds: ' + str(folder))
    category = next(iter(types))
    counts = Counter(raw_records=len(raw))
    images, sources, sessions, variants = set(), set(), set(), Counter()
    images_by_sid = manifest_images(manifest)
    alias_views = 0
    for row in raw:
        # Explicit reused views are not another execution. Current producers
        # normally omit these entirely; the inventory never expands aliases.
        if row.get('execution_reused') is True or row.get('alias_of') or row.get('execution_alias_of'):
            alias_views += 1
            continue
        data = row.get('runner_result', row)
        sid = row.get('session_id', row.get('id'))
        if sid is not None:
            sessions.add(str(sid))
        dataset = dataset_name(data.get('dataset', row.get('dataset')))
        image = data.get('image_id', images_by_sid.get(sid))
        if image is not None:
            images.add((dataset, str(image)))
        # Prefer the same image identity across primary/secondary backbones;
        # mixing a document ID on one view with its image hash on another
        # would otherwise make a reused source appear twice.
        source = image if image is not None else row.get('source_id', data.get('source_document_id', data.get('source_original_image_id')))
        if source is not None:
            sources.add((dataset, str(source)))
        variant = row.get('variant', data.get('variant'))
        if variant:
            variants[str(variant)] += 1
        if category in ('free_generation', 'diagnostic_free_generation'):
            counts['executed_sessions'] += 1
            counts['generated_answer_turns'] += sum(isinstance(turn.get('answer'), str) for turn in data['turns'])
            counts['generated_output_tokens'] += sum(len(turn.get('token_ids', [])) for turn in data['turns'])
            counts['unscored_initial_description_turns'] += int(data.get('first_turn_scored') is False)
        elif category == 'instrumented_profile':
            counts['instrumented_sessions'] += 1
            counts['instrumented_generated_tokens'] += int(row.get('generated_tokens', 0))
        elif category == 'metric_query_diagnostic':
            counts['diagnostic_sessions'] += 1
            counts['diagnostic_generated_tokens'] += int(row.get('generated_tokens', 0))
            counts['query_diagnostic_layer_phase_rows'] += len(row.get('actual_query_error_diagnostics', []))
        elif category == 'quiet_forced_trial':
            counts['forced_trials'] += 1
            counts['warmup_trials' if row.get('phase') == 'warmup' else 'measurement_trials'] += 1
            counts['forced_answer_turns'] += len(data.get('turns', []))
            counts['forced_output_tokens'] += sum(len(turn.get('token_ids', [])) for turn in data.get('turns', []))
        elif category == 'cached_prefix_calibration':
            counts['cached_prefix_question_cases'] += 1
    if category == 'cached_prefix_calibration':
        if receipt.get('questions') != len(raw):
            raise ValueError('Cached-prefix case count differs from completion receipt')
        scores = folder / 'packed_scores.jsonl'
        if not scores.is_file():
            raise ValueError('Cached-prefix raw score ledger missing')
        packed = rows(scores)
        forwards = sum(len(row['scores']) for row in packed)
        if forwards != receipt.get('physical_target_forwards'):
            raise ValueError('Physical target-forward count differs from raw cached-prefix ledger')
        counts['physical_allocation_case_evaluations'] = len(packed)
        counts['physical_target_forwards'] = forwards
    if category == 'quiet_forced_trial' and receipt.get('trials') != len(raw):
        raise ValueError('Quiet producer completion count differs from raw trial ledger')
    if category == 'quiet_forced_trial' and (folder / 'length_controls.json').is_file():
        counts['Full_canonical_control_sessions'] = len(read(folder / 'length_controls.json'))
    if category == 'metric_query_diagnostic' and receipt.get('cases') != len(raw):
        raise ValueError('Metric producer completion count differs from cases')
    if category in ('free_generation', 'diagnostic_free_generation', 'instrumented_profile') and 'records' in receipt and receipt['records'] != len(raw):
        raise ValueError('Producer completion count differs from raw records')
    aliases = manifest.get('config', {}).get('execution_aliases', manifest.get('execution_aliases', {}))
    hashes = {name: sha(folder / name) for name in ('manifest.json', 'complete.json', raw_name)}
    for name in ('packed_scores.jsonl', 'length_controls.json'):
        if (folder / name).is_file():
            hashes[name] = sha(folder / name)
    return dict(category=category, counts=dict(counts), unique_logged_sessions=len(sessions),
        unique_original_image_ids=len(images), unique_source_identity_keys=len(sources),
        actual_variants=dict(variants), declared_execution_aliases=aliases,
        explicit_reused_alias_rows_excluded=alias_views, answers_added_for_manifest_aliases=0,
        original_image_identity_keys=[list(item) for item in sorted(images)],
        source_identity_keys=[list(item) for item in sorted(sources)], hashes=hashes)


def inventory(root, require_main=False, quiet_run='quiet3_frozen6_v1'):
    root = Path(root)
    required = [root / name / 'complete.json' for name in (*MAIN_PARENTS, quiet_run)]
    if require_main:
        missing = [str(path.relative_to(root)) for path in required if not complete(path)]
        if missing:
            raise RuntimeError('Main collection incomplete; no raw prediction file opened: ' + ', '.join(missing))
    producers, skipped = [], []
    totals = defaultdict(Counter)
    image_unions, source_unions = defaultdict(set), defaultdict(set)
    # Receipt gates are evaluated before any raw JSONL is opened.
    for path in sorted(root.rglob('complete.json')):
        folder = path.parent
        relative = folder.relative_to(root)
        if any('analysis' in part.lower() for part in relative.parts):
            skipped.append(dict(run=str(relative), reason='analysis_output_not_a_producer'))
            continue
        if not complete(path):
            skipped.append(dict(run=str(relative), reason='incomplete_receipt_raw_not_opened'))
            continue
        formal = relative.parts[0] if relative.parts else ''
        if formal in MAIN_PARENTS and not complete(root / formal / 'complete.json'):
            skipped.append(dict(run=str(relative), reason='formal_parent_incomplete_raw_not_opened'))
            continue
        if not (folder / 'manifest.json').is_file():
            skipped.append(dict(run=str(relative), reason='no_manifest_no_free_generation_count'))
            continue
        raw_names = [name for name in ('timing_records.jsonl', 'records.jsonl', 'cases.jsonl') if (folder / name).is_file()]
        if not raw_names:
            skipped.append(dict(run=str(relative), reason='queue_or_analysis_or_asset_without_raw_producer_ledger'))
            continue
        if len(raw_names) != 1:
            raise ValueError('Ambiguous multiple primary raw ledgers: ' + str(relative))
        manifest = read(folder / 'manifest.json')
        if 'analysis' in str(manifest.get('schema', '')).lower() or manifest.get('analysis_role') or manifest.get('copied_from'):
            skipped.append(dict(run=str(relative), reason='declared_analysis_or_copied_view_not_execution'))
            continue
        row = summarize_producer(folder, manifest, read(path), raw_names[0])
        row['run'] = str(relative)
        producers.append(row)
        totals[row['category']].update(row['counts'])
        image_unions[row['category']].update(map(tuple, row['original_image_identity_keys']))
        source_unions[row['category']].update(map(tuple, row['source_identity_keys']))
    summary = {category: dict(producer_runs=sum(row['category'] == category for row in producers),
        **counts, union_unique_original_image_ids=len(image_unions[category]),
        union_unique_source_identity_keys=len(source_unions[category])) for category, counts in totals.items()}
    same_hash_paths = defaultdict(list)
    for row in producers:
        for name in ('records.jsonl', 'cases.jsonl', 'timing_records.jsonl'):
            if name in row['hashes']:
                same_hash_paths[(name, row['hashes'][name])].append(row['run'])
    return dict(schema='trajkv_round7_completed_raw_work_inventory_v1', status='COMPLETE',
        root=str(root), require_main_complete=require_main,
        required_receipts={str(path.relative_to(root)): sha(path) if complete(path) else None for path in required},
        categories=summary, producer_count=len(producers), producers=producers, skipped=skipped,
        identical_raw_hash_paths_for_review=[dict(raw_file=name, sha256=digest, runs=paths)
            for (name, digest), paths in same_hash_paths.items() if len(paths) > 1],
        accounting_scope='Actual completed producer rows count repeated physical work. Aliases are not expanded; analysis/queue copies are not executions. Source unions deduplicate methods/runs/backbones and are descriptive identities, not a statistical independent sample size.',
        not_a_total_GPU_forward_or_GPU_hour_count=True,
        excluded_auxiliary_work='Reader math checks without session ledgers, codebook fitting/assets, and analysis outputs are not session/answer producers.',
        no_partial_raw_prediction_files_opened=True, no_quality_scores_or_predictions_exported=True,
        code_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT / 'results/round7_v1')
    parser.add_argument('--require-main-complete', action='store_true')
    parser.add_argument('--quiet-run', default='quiet3_frozen6_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inventory(args.root, args.require_main_complete, args.quiet_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], producer_count=report['producer_count'], categories=report['categories'])), flush=True)


if __name__ == '__main__':
    main()
