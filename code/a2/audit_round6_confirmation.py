"""Read-only freeze/coverage/owner audit; never scores or selects fresh methods.

The full audit requires complete Doc54 and Text256 shards. --assets-only checks
frozen inputs/transport/immutable launch receipts without opening fresh records.
All report files are new, exclusive writes; inference sources stay unchanged.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from analyze_round5 import actual_visual_owner
from analyze_round6_depth import audit_record as audit_depth_record
from summarize_round5_confirmation import representation_check, checked_runner_seed
from round6_depth_codec import plan_depth_stages

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = {'full', 'native4', 'base2'}
LOWBIT = {'round6_depth', 'mixed_residual', 'sparse_v_residual'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(value):
    value = str(value).replace('\\', '/')
    if '/TrajKV/' in value:
        return ROOT / value.split('/TrajKV/', 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sources(mapping):
    """Accept timing's bare names and canonical code/a2 aliases consistently."""
    result = {}
    for name, digest in mapping.items():
        name = name.replace('\\', '/')
        if '/' not in name:
            name = 'code/a2/' + name
        if name in result and result[name] != digest:
            raise ValueError('Conflicting source SHA aliases: ' + name)
        result[name] = digest
    return result


def content_ids(row):
    values = set(row.get('hashes', []))
    for key in ('image_id', 'source_image_id'):
        value = str(row.get(key, ''))
        if value.startswith('sha256:'):
            values.add(value[7:])
    return values


def legacy_exposure(pool):
    """Rebuild old identity/content unions from the frozen inspection registry."""
    initial_path = ROOT / 'audit/a2/next_docvqa_source_split_v1.json'
    old_doc = load(ROOT / 'audit/a2/round5_doc64_membership.json')
    require(sha(initial_path) == old_doc['registry_sha256'][initial_path.relative_to(ROOT).as_posix()],
            'Legacy Doc source split identity changed')
    doc_ids = set(load(initial_path)['excluded_source_documents'])
    doc_hashes, text_ids, text_hashes = set(), set(), set()
    registry_hashes = {initial_path.relative_to(ROOT).as_posix():sha(initial_path)}

    def add(rows):
        for row in rows:
            if row.get('dataset') == 'docvqa_grouped':
                if row.get('source_document_id') is not None:
                    doc_ids.add(str(row['source_document_id']))
                doc_hashes.update(content_ids(row))
            elif row.get('dataset') == 'textvqa':
                if row.get('source_original_image_id') is not None:
                    text_ids.add(str(row['source_original_image_id']))
                text_hashes.update(content_ids(row))

    for item in pool['registry_inputs']:
        path = local_path(item['path'])
        require(sha(path) == item['sha256'], 'Preselection registry changed: ' + str(path))
        registry_hashes[str(path)] = item['sha256']
        kind = item['kind']
        if kind.startswith('registered subset'):
            add(load(path))
        elif kind.startswith('manifest selected-ID'):
            manifest = load(path)
            source = {row['id']:row for row in load(local_path(manifest['config']['arguments']['samples']))}
            add([source[sid] for sid in manifest.get('selected_ids', [])])
        elif kind.startswith('legacy identities'):
            decoder = json.JSONDecoder()
            paths = set()
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                identity = {}
                for field in ('image_id', 'image_path'):
                    match = re.search(r'"' + field + r'"\s*:\s*', line)
                    require(match is not None, 'Missing historical Text identity')
                    identity[field] = decoder.raw_decode(line, match.end())[0]
                text_ids.add(str(identity['image_id']))
                filename = identity['image_path'].replace('\\', '/').rsplit('/', 1)[-1]
                paths.add(ROOT.parent / 'benchmark_samples/textvqa_images' / filename)
            text_hashes.update(sha(path) for path in paths)
        elif kind.startswith('existing membership'):
            for image in load(path).get('images_manifest', []):
                text_hashes.add(image['sha256'])
                if image.get('source_image_id') is not None:
                    text_ids.add(str(image['source_image_id']))
        else:
            raise ValueError('Unrecognized preselection registry kind: ' + kind)
    return dict(doc_ids=doc_ids, doc_hashes=doc_hashes, text_ids=text_ids,
                text_hashes=text_hashes, registry_hashes=registry_hashes)


def check_memberships(marker, memberships, transport_path, pool_path):
    pool, transport = load(pool_path), load(transport_path)
    require(pool['model_scores_read'] is False and pool['annotation_files_read'] is False,
            'Pool was not fixed independently of outcomes/answers')
    old = legacy_exposure(pool)
    cohorts, all_images = {}, {}
    for path in memberships:
        membership = load(path)
        cohort = membership['cohort']
        require(cohort in ('doc54', 'text256') and cohort not in cohorts, 'Unexpected/duplicate fresh cohort')
        require(membership['method_freeze_sha256'] == sha(marker), 'Membership method-freeze hash differs')
        require(membership['pool_inspection_sha256'] == sha(pool_path), 'Preinspected pool identity differs')
        require(membership['selection_fixed_before_answers_accessed'] is True and
                membership['annotations_used_for_selection'] is False and
                membership['model_outcomes_used_for_selection'] is False and
                membership['source_content_overlap'] == 0 and membership['answers_uploaded'] is False,
                'Fresh membership provenance rule differs')
        samples_path, refs_path = ROOT / membership['samples'], ROOT / membership['references']
        require(sha(samples_path) == membership['samples_sha256'] and sha(refs_path) == membership['references_sha256'],
                'Fresh sample/reference bytes differ from membership')
        rows = load(samples_path)
        expected_size = 54 if cohort == 'doc54' else 256
        require(len(rows) == expected_size == membership['images'] and
                len({row['id'] for row in rows}) == expected_size, 'Fresh cohort identities/counts differ')
        require(sum(row['turns'] for row in rows) == membership['questions'], 'Fresh question count mismatch')
        candidates = pool['doc']['candidates'] if cohort == 'doc54' else pool['text']['candidates256']
        require(len(candidates) == len(rows), 'Preinspected candidate size mismatch')
        source_ids, hashes = set(), set()
        for row, candidate in zip(rows, candidates):
            require(not any(key in row for key in ('answer', 'answers', 'references')), 'Answers leaked into runner')
            digest = candidate['sha256']
            require(content_ids(row) == {digest} and row['image_id'] == 'sha256:' + digest,
                    'Fresh encoded image identity mismatch')
            if cohort == 'doc54':
                sid = str(row['source_document_id'])
                require(sid == candidate['source_document_id'] and row['source_page_id'] == candidate['source_page_id']
                        and row['source_question_ids'] == candidate['source_question_ids'], 'Doc source/page/question identity mismatch')
                require(sid not in old['doc_ids'] and digest not in old['doc_hashes'], 'Doc source/content overlap')
            else:
                sid = str(row['source_original_image_id'])
                require(sid == candidate['source_image_id'] and row['source_question_id'] == candidate['source_question_id'],
                        'Text source/question identity mismatch')
                require(sid not in old['text_ids'] and digest not in old['text_hashes'], 'Text source/content overlap')
            require(sid not in source_ids and digest not in hashes, 'Fresh cohort repeats source/content')
            source_ids.add(sid); hashes.add(digest)
        require(len(membership['images_manifest']) == expected_size, 'Staged image inventory count mismatch')
        inventory = {}
        for image in membership['images_manifest']:
            require(image['remote'] not in all_images and image['sha256'] not in inventory,
                    'Staged image inventory repeats identity')
            local = local_path(image['relative'])
            require(sha(local) == image['sha256'] and local.stat().st_size == image['bytes'], 'Staged raw image changed')
            inventory[image['sha256']] = image
            all_images[image['remote']] = (image['sha256'], image['bytes'])
        require(set(inventory) == hashes, 'Runner/staged image inventory differs')
        cohorts[cohort] = dict(membership=membership, membership_path=path, samples_path=samples_path,
            rows=rows, ids={row['id'] for row in rows}, by_id={row['id']:row for row in rows},
            source_ids=source_ids, content_hashes=hashes)
    require(set(cohorts) == {'doc54', 'text256'}, 'Both fresh cohorts are required')
    require(not (cohorts['doc54']['content_hashes'] & cohorts['text256']['content_hashes']), 'Doc/Text content overlap')
    require(transport['status'] == 'PASS' and transport['answers_uploaded'] is False
            and transport['method_freeze_sha256'] == sha(marker) and transport['images'] == 310,
            'Remote transport receipt is incomplete/different')
    remote = {item['remote']:(item['sha256'], item['bytes']) for item in transport['remote_images']}
    require(len(remote) == len(transport['remote_images']) == 310 and remote == all_images,
            'Remote image hashes/bytes differ from both local memberships')
    return cohorts, old['registry_hashes']


def check_sources(frozen, manifest, job, receipts):
    launch_path = receipts / (job + '.launch.json')
    snapshot_root = receipts / (job + '.source')
    snapshot_path = snapshot_root / 'sha256.json'
    launch, snapshot = load(launch_path), canonical_sources(load(snapshot_path))
    require(launch['name'] == job and launch.get('immutable_code_execution') is True,
            'Wrong/nonimmutable launch receipt')
    require(any(str(arg).startswith(launch['source_snapshot'] + '/') and str(arg).endswith('.py')
                for arg in launch['argv_executed']), 'Launch did not execute its immutable source snapshot')
    sources = canonical_sources(manifest['sources']) if manifest is not None else None
    copied = {}
    for name, expected in frozen['inference_source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Frozen local inference source drift: ' + name)
        require(snapshot.get(name) == expected, 'Frozen launch source SHA mismatch: ' + name)
        if name.startswith('code/a2/') and sources is not None:
            require(sources.get(name) == expected, 'Run-used a2 source SHA mismatch: ' + name)
        if not name.startswith('code/a2/'):
            path = snapshot_root / name
            require(path.is_file() and sha(path) == expected, 'Missing/wrong immutable auxiliary source bytes: ' + name)
            copied[name] = expected
    seed_key = 'code/a2/run_representation_variants.py'
    seed_path = ROOT / seed_key
    require(snapshot.get(seed_key) == sha(seed_path), 'Seeded transitive runner not in immutable source inventory')
    if sources is not None:
        require(sources.get(seed_key) == sha(seed_path), 'Run manifest seeded runner differs')
    checked_runner_seed(seed_path.read_text(encoding='utf-8'))
    return dict(job=job, launch_sha256=sha(launch_path), snapshot_sha256=sha(snapshot_path),
        frozen_source_count=len(frozen['inference_source_sha256']), auxiliary_source_bytes=copied,
        seed=1212, canonical_source_aliases_supported=True)


def record_cost(record, config, factor_bytes, layers=28, heads=4):
    """Same visual-owner helper as scorers, plus reused independent validators."""
    method, owned, nv = record['method'], record['owned_initial'], int(record['visual_tokens'])
    require(method == config['method'] and record['variant_config'] == config, 'Record differs from frozen variant')
    require(method in REFERENCES | LOWBIT, 'Unaccounted confirmation codec')
    require(record['protocol'] == 'image_boundary_free_running' and record.get('task_accuracy_eligible') is not False
            and record.get('primary_quality_eligible') is not False and record.get('initial_logits_shared_fp16') is False,
            'Record is not eligible image-boundary free generation')
    require(record.get('read_ledger_exact') is True and record.get('global_extra_budget_ok', True), 'Read/budget flag failed')
    require(not record.get('materialized_quality_port', False) and not owned.get('diagnostic_fp16_reference', 0)
            and not owned.get('full_visual_retained', False), 'Materialized/diagnostic source owner retained')
    if method in LOWBIT:
        require(record.get('visual_fp16_materialization') is False, 'Lowbit path materializes source visual FP16')
    complete = actual_visual_owner(record) + int(record.get('encoder_calibration_parameters_bytes', 0))
    require(owned['visual_fp16_denominator'] == 4 * layers * heads * nv * 128 and owned['visual_position'] == 8 * nv,
            'Visual denominator/position owner differs from actual geometry')
    categories = ('visual_base', 'visual_residual', 'text_capacity', 'position',
                  'kvarn_packed_reference', 'diagnostic_fp16_reference', 'selector_metadata')
    require(sum(int(owned.get(k, 0)) for k in categories) == owned['total_unique'], 'Whole owner category sum differs')
    require(owned['position'] == owned['visual_position'] + owned['text_position_capacity'], 'Position owner double-count/missing owner')
    raw_reads = sum(event.get('selection_bytes', 0) + event['residual_bytes']
        + (event['base_visual_bytes'] + event['text_bytes'] if event['pass_kind'] in ('replay', 'repair', 'probe') else 0)
        for event in record['read_events'])
    require(raw_reads == record['extra_logical_read_bytes'], 'Independent read ledger sum differs')
    if method != 'round6_depth':
        result = representation_check(record, config, factor_bytes)
        require(result['complete_visual_bytes'] == complete, 'Shared/scorer visual owner differs')
        stages = int(record.get('mixed_residual_units', record.get('sparse_residual_units', 0)))
    else:
        result, _, _, _, _ = audit_depth_record(record)
        require(result['passed'], 'Depth record validator failed: ' + result['errors'])
        vg = nv if config.get('v_group_size', 'all') == 'all' else int(config['v_group_size'])
        expected_base = (layers * heads * nv * 50 + layers * heads * 128 * 2
                         + 4 * layers * heads * 128 * (nv // vg) + 2048)
        require(owned['visual_base'] == expected_base, 'Independent PQ/V1 base owner formula mismatch')
        plan = plan_depth_stages(layers=layers, heads=heads, visual_tokens=nv,
            visual_full_bytes=owned['visual_fp16_denominator'], base_bytes=expected_base + 8 * nv,
            factor_bytes=factor_bytes, budget_bits=float(config.get('budget_bits', 1.8)),
            selected_units=config.get('selected_units', []), selected_stages=config.get('selected_stages'),
            depth_policy=config.get('depth_policy', 'incumbent'),
            v_mixed_pair_fraction=float(config.get('v_mixed_pair_fraction', .5)),
            stage_limit=config.get('res_stages_limit', config.get('res_units_limit', config.get('residual_units_limit'))))
        require([tuple(x) for x in record['depth_residual_selected_stages']] == plan['selected_stages'], 'Selected stages differ from frozen affordable policy')
        require(owned['selector_metadata'] == plan['lookup_bytes'] and
                record['encoder_calibration_parameters_bytes'] == factor_bytes + len(plan['ranking_constant']) + len(plan['policy_constant']),
                'Independent lookup/ranking/policy/factor encoder owner differs')
        require(record['depth_residual_encoder_ranking_sha256'] == hashlib.sha256(plan['ranking_constant']).hexdigest()
                and record['depth_residual_encoder_policy_sha256'] == hashlib.sha256(plan['policy_constant']).hexdigest(),
                'Actual depth ranking/descriptor bytes differ from frozen config')
        require(complete == plan['predicted_bytes'], 'Depth complete owner differs from affordable exact plan')
        stages = len(plan['selected_stages'])
    bits = 16 * complete / owned['visual_fp16_denominator']
    if method in LOWBIT:
        require(bits <= 1.8 + 1e-12, 'Eligible method exceeds complete 1.8bit cap')
    return dict(complete_visual_bytes=complete, encoder_bytes=int(record.get('encoder_calibration_parameters_bytes', 0)),
        actual_bits=bits, stages=stages, eligible_lowbit=method in LOWBIT,
        reference_above_budget=method in REFERENCES and bits > 1.8,
        visual_tokens=nv, method=method)


def check_run(folder, frozen, marker, cohorts, receipts, factor_bytes):
    complete = load(folder / 'complete.json')
    require(complete['status'] == 'COMPLETE', 'Fresh run is not complete: ' + str(folder))
    manifest = load(folder / 'manifest.json')
    config = manifest['config']
    require(config['variants'] == frozen['variants'], 'Run does not contain identical ordered frozen eight variants')
    require(config.get('frozen_contract_sha256') == sha(marker), 'Run contract SHA differs from pre-outcome freeze')
    matched = [key for key, cohort in cohorts.items() if local_path(config['arguments']['samples']).resolve() == cohort['samples_path'].resolve()]
    require(len(matched) == 1, 'Run is not on a frozen fresh cohort')
    cohort_name, cohort = matched[0], cohorts[matched[0]]
    require(manifest['samples_sha256'] == cohort['membership']['samples_sha256']
            and manifest.get('answers_visible_to_runner') is False, 'Run sample bytes/answer visibility differs')
    planned_paths = [ROOT / f'configs/a2/round6_v1/confirm_{cohort_name}_s{shard}.json' for shard in (0, 1)]
    matches = [path for path in planned_paths if sha(path) == manifest['config_sha256'] and load(path) == config]
    require(len(matches) == 1, 'Manifest config has no unique byte-identical prepared shard')
    job = 'round6_' + matches[0].stem + '_v1'
    expected_ids = [row['id'] for row in cohort['rows'][config.get('start', 0):config.get('start', 0) + config['count']]]
    require(manifest['selected_ids'] == expected_ids and len(set(expected_ids)) == len(expected_ids), 'Shard selected-ID order differs from prepared input')
    sources = check_sources(frozen, manifest, job, receipts)
    require(manifest['external_codec_inputs'] == frozen['external_codec_inputs_sha256'], 'Run external three-file SHA inventory differs')
    require(config['arguments'].get('compress_before_question') is True and
            config['arguments']['turns'] == (6 if cohort_name == 'doc54' else 1), 'Run inference boundary/turn protocol differs')
    variants = {v['name']:v for v in frozen['variants']}
    seen, costs, answers = set(), [], 0
    with (folder / 'records.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sid, name = record['session_id'], record['variant']
            key = (sid, name)
            require(key not in seen and sid in expected_ids and name in variants, 'Duplicate/unexpected fresh record')
            sample = cohort['by_id'][sid]
            require(record['image_id'] == sample['image_id'], 'Record source image differs from frozen sample')
            turns = [turn['turn'] for turn in record['turns']]
            require(turns == list(range(1, sample['turns'] + 1)) and
                    all(isinstance(turn['answer'], str) and isinstance(turn.get('token_ids'), list) for turn in record['turns']),
                    'Missing/repeated/out-of-order question outputs')
            costs.append(dict(session_id=sid, variant=name, **record_cost(record, variants[name], factor_bytes)))
            seen.add(key); answers += len(turns)
    expected = {(sid, name) for sid in expected_ids for name in variants}
    require(seen == expected and complete['records'] == len(expected)
            and complete.get('expected', len(expected)) == len(expected), 'Run COMPLETE counts or per-image eight-arm coverage differs')
    return dict(cohort=cohort_name, job=job, ids=expected_ids, seen=seen, cost_rows=costs,
        answers=answers, records=len(seen), source_audit=sources,
        manifest_sha256=sha(folder / 'manifest.json'), records_sha256=sha(folder / 'records.jsonl'),
        complete_sha256=sha(folder / 'complete.json'), configuration_sha256=manifest['config_sha256'])


def self_test():
    require(canonical_sources({'round6_timing.py':'x', 'code/a2/round6_timing.py':'x'}) == {'code/a2/round6_timing.py':'x'}, 'Canonical source alias test failed')
    rejected = 0
    try:
        canonical_sources({'round6_timing.py':'x', 'code/a2/round6_timing.py':'y'})
    except ValueError:
        rejected += 1
    nv, full = 128, 4 * 28 * 4 * 128 * 128
    config = dict(name='synthetic_full_reference', method='full')
    record = dict(session_id='synthetic', variant=config['name'], variant_config=config, method='full',
        protocol='image_boundary_free_running', initial_logits_shared_fp16=False, read_ledger_exact=True,
        visual_tokens=nv, turns=[], read_events=[], extra_logical_read_bytes=0,
        owned_initial=dict(visual_base=full, visual_residual=0, selector_metadata=0,
            visual_position=8 * nv, text_position_capacity=0, text_capacity=0, position=8 * nv,
            total_unique=full + 8 * nv, visual_fp16_denominator=full))
    cost = record_cost(record, config, 0)
    require(cost['reference_above_budget'] and not cost['eligible_lowbit'], 'Full reference wrongly subjected to 1.8 cap')
    bad = copy.deepcopy(record); bad['owned_initial']['total_unique'] += 1
    try:
        record_cost(bad, config, 0)
    except ValueError:
        rejected += 1
    bad = copy.deepcopy(record); bad['materialized_quality_port'] = True
    try:
        record_cost(bad, config, 0)
    except ValueError:
        rejected += 1
    units = [(axis, layer, head) for layer in range(28) for head in range(4) for axis in ('k', 'v')]
    stages = []
    for n in (256, 1792):
        base = 28 * 4 * n * 50 + 28 * 4 * 128 * 6 + 2048 + 8 * n
        plan = plan_depth_stages(layers=28, heads=4, visual_tokens=n,
            visual_full_bytes=4 * 28 * 4 * n * 128, base_bytes=base, factor_bytes=1344,
            selected_units=units, depth_policy='incumbent')
        stages.append(len(plan['selected_stages']))
    require(stages[0] != stages[1], 'Variable-N affordable counts were incorrectly fixed')
    require(rejected == 3, 'Synthetic corruption rejection failed')
    return dict(status='PASS_SELF_TEST', synthetic_only=True, records_scored=0,
        corruptions_rejected=rejected, source_aliases=True, reference_above_budget_allowed=True,
        variable_N_stages=stages)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', type=Path, default=ROOT / 'audit/a2/round6_confirmation_v1.json')
    p.add_argument('--memberships', nargs=2, type=Path, default=[ROOT / 'audit/a2/round6_doc54_membership.json', ROOT / 'audit/a2/round6_text_membership_v1.json'])
    p.add_argument('--transport', type=Path, default=ROOT / 'audit/a2/round6_confirmation_transport_v1.json')
    p.add_argument('--pool', type=Path, default=ROOT / 'audit/a2/round6_unseen_pool_inspection_v1.json')
    p.add_argument('--receipts', type=Path, default=ROOT / 'results/round6_v1/receipts')
    p.add_argument('--runs', nargs=4, type=Path)
    p.add_argument('--timing-runs', nargs='*', type=Path, default=[])
    p.add_argument('--output', type=Path)
    p.add_argument('--assets-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    frozen = load(args.marker)
    require(len(frozen['variants']) == len({v['name'] for v in frozen['variants']}) == 8, 'Expected eight distinct frozen variants')
    require(len(frozen['inference_source_sha256']) == 24 and len(frozen['external_codec_inputs_sha256']) == 3,
            'Frozen inference/codec inventory shape differs')
    require(frozen['answers_uploaded'] is False and frozen['primary'] in {v['name'] for v in frozen['variants']}, 'Invalid method freeze')
    for name, expected in frozen['inference_source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Local frozen inference source drift: ' + name)
    for name, expected in frozen['external_codec_inputs_sha256'].items():
        require(sha(local_path(name)) == expected, 'Local frozen external codec drift: ' + name)
    cohorts, registry_hashes = check_memberships(args.marker, args.memberships, args.transport, args.pool)
    launches = [check_sources(frozen, None, f'round6_confirm_{cohort}_s{shard}_v1', args.receipts)
                for cohort in ('doc54', 'text256') for shard in (0, 1)]
    factor_path = local_path(next(v['metadata_factor_path'] for v in frozen['variants'] if v.get('metadata_factor_path')))
    factors = load(factor_path)
    counted = sum(len(row) for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets') for row in factors.get(key, []))
    require(counted == factors['parameter_count'], 'Independent base encoder factor element count differs')
    factor_bytes = 4 * counted
    runs, tables = [], []
    if not args.assets_only:
        folders = args.runs or [ROOT / f'results/round6_v1/confirm_{cohort}_s{shard}' for cohort in ('doc54', 'text256') for shard in (0, 1)]
        runs = [check_run(folder, frozen, args.marker, cohorts, args.receipts, factor_bytes) for folder in folders]
        for cohort_name, cohort in cohorts.items():
            selected = [run for run in runs if run['cohort'] == cohort_name]
            require(len(selected) == 2 and not (set(selected[0]['ids']) & set(selected[1]['ids']))
                    and set(selected[0]['ids']) | set(selected[1]['ids']) == cohort['ids'], 'Fresh shards overlap or do not cover whole cohort')
            require(sum(run['answers'] for run in selected) == cohort['membership']['questions'] * 8, 'Fresh complete answer coverage differs')
            rows = [row for run in selected for row in run['cost_rows']]
            for variant in frozen['variants']:
                group = [row for row in rows if row['variant'] == variant['name']]
                tables.append(dict(cohort=cohort_name, variant=variant['name'], sessions=len(group),
                    complete_bytes_total=sum(row['complete_visual_bytes'] for row in group),
                    encoder_bytes_per_session=sorted({row['encoder_bytes'] for row in group}),
                    maximum_per_image_bits=max(row['actual_bits'] for row in group),
                    minimum_stages=min(row['stages'] for row in group), maximum_stages=max(row['stages'] for row in group),
                    eligible_lowbit=group[0]['eligible_lowbit'], reference_above_budget=any(row['reference_above_budget'] for row in group)))
    timing_checks = []
    for folder in args.timing_runs:
        require(load(folder / 'complete.json')['status'] == 'COMPLETE', 'Timing is not complete')
        manifest = load(folder / 'manifest.json')
        sources = canonical_sources(manifest['sources'])
        require(all(sources.get(name) == digest for name, digest in frozen['inference_source_sha256'].items()), 'Timing frozen source hashes differ')
        require(all(frozen['external_codec_inputs_sha256'].get(name) == digest for name, digest in manifest['external_codec_inputs'].items()), 'Timing codec source hashes differ')
        timing_checks.append(dict(folder=str(folder), manifest_sha256=sha(folder / 'manifest.json'), frozen_sources_checked=24))
    result = dict(status='PASS_ASSETS_ONLY' if args.assets_only else 'PASS',
        schema='trajkv.round6.confirmation-independent-audit.v1', no_scores_computed=True,
        method_reselection=False, fresh_records_opened=not args.assets_only,
        marker_sha256=sha(args.marker), transport_sha256=sha(args.transport), pool_sha256=sha(args.pool),
        membership_sha256={str(path):sha(path) for path in args.memberships},
        frozen_inference_source_sha256=frozen['inference_source_sha256'], external_codec_inputs_sha256=frozen['external_codec_inputs_sha256'],
        preselection_registry_sha256=registry_hashes, launch_receipts=launches,
        factor_parameter_bytes=factor_bytes, shared_owner_helpers=['analyze_round5.actual_visual_owner',
            'summarize_round5_confirmation.representation_check', 'analyze_round6_depth.audit_record'],
        coverage=dict(doc_images=54, doc_questions=205, text_images=256, text_questions=256, variants=8,
                      expected_records=2480, expected_answers=3688),
        cost_table=tables, run_hashes=[{key:value for key, value in run.items() if key not in ('ids', 'seen', 'cost_rows')} for run in runs],
        timing_source_checks=timing_checks, auditor_sha256=sha(__file__))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(dict(status=result['status'], no_scores_computed=True, frozen_sources=24,
        external_inputs=3, launches=len(launches), fresh_records_opened=result['fresh_records_opened'],
        output=str(args.output) if args.output else None, cost_table=tables)))


if __name__ == '__main__':
    main()
