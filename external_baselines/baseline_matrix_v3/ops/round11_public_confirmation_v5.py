"""R128-only public v3 source/selection/data/score closure.

No model, GPU, credentials or network. D96 subset evidence is never projected
into a confirmation run. Portable loader replay accounts explicitly for path
redaction and original versus public byte identities, without opening images.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

MANIFEST_SHA = 'acdfb720120d55793a4f6e7dc2b43e5681c0727aff28e8c906ecbc6696578380'
D96_SHA = '663076f21fc7df7d00253d06604e9f46da06ffe9d62532cd66d82794c05fa9b4'
COUNTS = dict(textvqa=48, infovqa=48, docvqa=16, chartqa=16)
DEVELOPMENT_COUNTS = dict(textvqa=32, infovqa=32, docvqa=16, chartqa=16)
REFS = ('FP16', 'R7low', 'R7high')
CANDIDATES = ('JointV', 'Spatial', 'JointSpatial', 'Shuffled', 'SpatialWO', 'JointSpatialWO')
PLAN_SCHEMA = 'trajkv.round11.confirmation_harvest_plan.v2'
RAW_SCHEMA = 'trajkv.round11_confirmation.raw.v2'
SCRIPT = 'code/round11_confirmation_runner_v2.py'
ADAPTER = 'ops/round11_confirmation_harvest_v2.py'
PROOF = 'source_retrieval_receipt_round11_confirmation_v2.json'
PINS = {
    SCRIPT: '3db28f74add8ba98b5ea1cff928b4aada775408cde69c5629dbe0b2b75e25aca',
    ADAPTER: '6435fee9fffc38b873c73b82d9dcc82010c73c1acb123c2d7f6c47f46ea4ba9e',
    'code/round11_runner_v2.py': 'ec87b1270bfae11d4a2d98fcafc544fcf42a3c132c72bc30d41fe4176ee73ce1',
    'code/round11_wo_runner_v1.py': 'c83846fce9e41ef4308160e9e48dae78ba143c0ef9843e387e041ec632907517',
    'code/round11_wo_value_v1.py': 'b0f62f1c8ddbad41336b161891cf0943876a0087e9dcdec63e9bc6e312aaf69e',
    'code/round11_confirmation_data_v1.py': '44297a3c58a493c26a17084a72c4800303b0b70143543217fe0ede3db02548cf',
    'code/round10_data_builder_v1.py': '914dfa258e6fbcfc00c709df6e78cfba6cba5c385db7c51f1e15ebb5d0f59936',
    'code/round10_inputs_v1.py': '61cc6dca53d95f0f24ab0a3b160eeb7be5fbac179e5ded510a2c2852e8e3dd77',
    'code/round10_sequence_v1.py': '2d0dce9da80156fb0224a04d48a9a3e3869ede5e9885885a6a25073ab511fe00',
}
D96_RUNNERS = {
    'code/round11_runner_v2.py': {'raw_schema': 'trajkv.round11_research.raw.v1', 'method_family': 'trajkv_round11_research', 'proof_name': 'source_retrieval_receipt_round11_v1.json', 'proof_status': 'PASS_ROUND11_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA', 'modes': ['JointV', 'Spatial', 'JointSpatial', 'Shuffled']},
    'code/round11_wo_runner_v1.py': {'raw_schema': 'trajkv.round11_wo_research.raw.v1', 'method_family': 'trajkv_round11_wo_research', 'proof_name': 'source_retrieval_receipt_round11_wo_v1.json', 'proof_status': 'PASS_ROUND11_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA', 'modes': ['SpatialWO', 'JointSpatialWO']},
}


def d96_runner_contract(script):
    require(script in D96_RUNNERS, 'unreviewed D96 runner/schema requires a new public contract version')
    return D96_RUNNERS[script]


def require(value, message):
    if not value: raise ValueError('ROUND11_PUBLIC_R128: '+message)


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def selection_modes(selection, development_summary_shas):
    require(set(selection) == {'schema', 'selected_modes', 'selection_scope', 'confirmation_results_read', 'primary_source_group_sha256', 'development_harvest_summary_sha256'}, 'selection v2 must retain exactly six decision-provenance fields')
    modes = selection['selected_modes']
    hashes = selection['development_harvest_summary_sha256']
    require(isinstance(hashes, dict) and set(hashes) == {'primary', 'wo'} and all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None and value != '0'*64 for value in hashes.values()) and hashes == development_summary_shas, 'selection requires both exact nonzero completed primary and WO summary identities')
    require(selection['schema'] == 'trajkv.round11.confirmation_selection.v2' and isinstance(modes, list) and len(modes) <= 2 and len(set(modes)) == len(modes) and set(modes) <= set(CANDIDATES) and selection['selection_scope'] == 'D96_only_before_R128' and selection['confirmation_results_read'] is False and selection['primary_source_group_sha256'] == D96_SHA, 'selection must be fixed from exact completed primary+WO D96 sources before R128 outputs')
    return modes


def contrasts(selected):
    return [[mode, ref] for mode in selected for ref in REFS] + ([['Shuffled', 'Spatial']] if {'Shuffled', 'Spatial'} <= set(selected) else []) + ([['JointSpatialWO', 'SpatialWO']] if {'JointSpatialWO', 'SpatialWO'} <= set(selected) else [])


def validate_development_summary(role, summary):
    require(role in ('primary', 'wo'), 'unknown development selection-basis role')
    views, fresh, schema, status, candidates = (28, 384, 'trajkv.round11.harvest_summary.v1', 'PASS_ALL_28_PROVEN_VIEWS', CANDIDATES[:4]) if role == 'primary' else (20, 192, 'trajkv.round11_wo.harvest_summary.v1', 'PASS_ALL_20_PROVEN_VIEWS', CANDIDATES[4:])
    require(summary.get('schema') == schema and summary.get('status') == status and summary.get('expected_cells_or_views') == summary.get('closed_cells_or_views') == summary.get('successful_complete_cells_or_views') == views and summary.get('expected_new_free_generation_QA') == summary.get('actual_closed_new_free_generation_QA') == fresh and summary.get('expected_reused_reference_views') == summary.get('actual_closed_reused_reference_views') == 288 and summary.get('source_group_manifest_sha256') == D96_SHA and summary.get('canonical_SourceRun_admission_claimed') is False, 'selection basis must be the exact complete primary384 or WO192 D96 scope')
    points = summary.get('points', []); keys = {(p['dataset'], p['mode']) for p in points}
    require(len(points) == len(keys) == views and keys == {(d, m) for d in COUNTS for m in (*REFS, *candidates)} and all(p['status'] == 'PASS_FULL_SOURCE_GROUP_SCORE' and p['primary_mean'] is not None and p['expected_QA'] == DEVELOPMENT_COUNTS[p['dataset']] and p['fresh_free_generation_QA'] == (DEVELOPMENT_COUNTS[p['dataset']] if p['mode'] in candidates else 0) and p['reused_reference_views'] == (DEVELOPMENT_COUNTS[p['dataset']] if p['mode'] in REFS else 0) for p in points), 'selection basis contains incomplete, duplicated or wrong-mode development points')


def plan_shape(plan, selected):
    modes = (*REFS, *selected)
    require(plan['schema'] == PLAN_SCHEMA and plan['input_manifest_sha256'] == MANIFEST_SHA, 'exact R128 plan/input manifest required')
    entries = {(e['dataset'], e['mode']): e for e in plan['entries']}
    require(len(entries) == len(plan['entries']) == 4*len(modes) and set(entries) == {(d, m) for d in COUNTS for m in modes}, 'all3 fresh references and exactly selected candidates required')
    for (ds, mode), e in entries.items():
        cell = e['item']['cell']
        require(e['kind'] == 'fresh' and not e.get('source_raw_dir') and not e.get('reuse_from') and cell['dataset'] == ds and cell['mode'] == mode and cell['expected_QA'] == COUNTS[ds] and cell['script'] == SCRIPT and cell['script_sha256'] == PINS[SCRIPT] and e['item']['sequence_script_sha256'] == PINS['code/round10_sequence_v1.py'] and e['item']['input_manifest_sha256'] == MANIFEST_SHA and e['item']['selection_manifest_sha256'] == plan['selection_manifest_sha256'], 'reused/subset/wrong-source confirmation item')
    return entries


def collect_data(group, add):
    """Collect only the explicitly pinned 41-file readback inventory."""
    closure = group['confirmation_data']; inventory = closure['fetch_inventory']; remote = closure['remote_sha_receipt']
    add(inventory['path'], 'confirmation_data_fetch_inventory', inventory['sha256'])
    add(remote['path'], 'confirmation_data_remote_sha', remote['sha256'])
    records = load(inventory['path']); proof = load(remote['path'])
    require(proof['schema'] == 'trajkv.round11.confirm128_independent_remote_sha.v1' and proof['no_model_or_GPU_calls'] is True, 'independent data source readback required')
    require(len(records) == len({r['remote_path'] for r in records}), 'duplicate data source paths')
    indexed = {r['remote_path']: r for r in records}
    require(set(indexed) == {r['path'] for r in proof['files']}, 'public data inventory must cover every remote proof file')
    for p in proof['files']:
        e = indexed[p['path']]; require(e['local_sha256'] == p['sha256'] == sha(e['local_path']) and e['bytes'] == p['bytes'] == Path(e['local_path']).stat().st_size, 'independent data proof/local SHA or original size differs')
        name = Path(e['local_path']).name
        role = 'source_inputs' if name.endswith('_inputs.jsonl') else 'reference_answers' if name.endswith('_refs.jsonl') else 'input_manifest' if name == 'CONFIRM128_INPUT_MANIFEST_v1.json' else 'source_lock' if name == 'confirm128_source_lock_v1.json' else 'frozen_data_job_log' if Path(e['local_path']).suffix.lower() == '.log' else 'confirmation_data_source'
        public_path = 'evidence/data_job_logs/'+e['local_sha256'][:20]+'_'+name+'.txt' if role == 'frozen_data_job_log' else None
        add(e['local_path'], role, e['local_sha256'], public_path=public_path, aliases=[e['remote_path']])
    def local(remote_path, digest=None):
        require(remote_path in indexed, 'missing exact data closure alias: '+remote_path)
        e = indexed[remote_path]; require(digest is None or e['local_sha256'] == digest, 'data closure source identity differs'); return Path(e['local_path'])
    return local


def collect_group(g, adapter, plan, modes, add, deep, fresh_sources, project_root, all_groups):
    require(PINS[SCRIPT] is not None and PINS[ADAPTER] is not None, 'confirmation-v2 exact source epochs have not been frozen')
    require(sha(project_root/ADAPTER) == PINS[ADAPTER] and sha(project_root/SCRIPT) == PINS[SCRIPT], 'frozen R128 adapter/driver epoch differs')
    bases = g['selection_basis']; require(set(bases) == {'primary', 'wo'}, 'selection basis requires both development groups')
    basis_hashes = {}
    for role, basis in bases.items():
        require(set(basis) == {'group_id', 'path', 'sha256'}, 'explicit development group/path/SHA basis required')
        development = next((x for x in all_groups if x['group_id'] == basis['group_id']), None)
        require(development is not None and development['execution_disposition'] == 'EXECUTED' and Path(development['summary_root']).resolve()/'summary.json' == Path(basis['path']).resolve(), 'selection must bind both actual included completed D96 groups')
        require(sha(basis['path']) == basis['sha256'], 'selection development summary bytes changed')
        validate_development_summary(role, load(basis['path'])); add(basis['path'], 'confirmation_D96_'+role+'_selection_basis', basis['sha256']); basis_hashes[role] = basis['sha256']
    add(plan['selection_manifest_path'], 'confirmation_selection_manifest', plan['selection_manifest_sha256'])
    selected = selection_modes(load(plan['selection_manifest_path']), basis_hashes); entries = plan_shape(plan, selected)
    require(tuple(modes) == (*REFS, *selected) and g['counts'] == COUNTS and g['modes'] == list(modes) and g['paired_contrasts'] == contrasts(selected), 'actual R128 counts/modes/contrasts differ')
    local = collect_data(g, add); manifest = load(plan['input_manifest_path'])
    require(sha(plan['input_manifest_path']) == MANIFEST_SHA and manifest['schema'] == 'trajkv.round11.confirm128_inputs.v1' and manifest['counts'] == COUNTS and manifest['total_count'] == 128 and manifest['answers_in_encoder_inputs'] is False, 'frozen R128 source manifest differs')
    for key in ('disjoint_proof', 'raw_source_proof', 'source_locators', 'source_lock'):
        local(manifest[key]['path'], manifest[key]['sha256'])
    for rel, digest in PINS.items(): add(project_root/rel, 'frozen_confirmation_source', digest)
    for rel, digest in adapter.check_pins().items(): add(project_root/rel, 'frozen_confirmation_scoring_source', digest)
    summary_root = Path(g['summary_root']); summary = load(summary_root/'summary.json'); pairs = load(summary_root/'paired_comparisons.json')
    require(summary['schema'] == 'trajkv.round11.confirmation_harvest_summary.v2' and summary['plan_sha256'] == g['plan']['sha256'] and summary['input_manifest_sha256'] == MANIFEST_SHA and summary['selection_manifest_sha256'] == plan['selection_manifest_sha256'] and summary['expected_new_QA'] == 128*len(modes) and summary['expected_fresh_reference_QA'] == 384 and summary['reused_reference_views'] == 0, 'R128 summary source/actual-scope accounting differs')
    for name in ('summary.json', 'paired_comparisons.json'): add(summary_root/name, 'frozen_confirmation_group_summary')
    frozen = {(p['dataset'], p['mode']): p for p in summary['points']}; rows = {(r['dataset'], r['mode']): r for r in g['cells']}
    require(set(rows) == set(entries) == set(frozen) and len(rows) == len(g['cells']), 'R128 every actual cell disposition required')
    bundles = {}; outrows = []; counts = Counter()
    for key, entry in entries.items():
        ds, mode = key; row = rows[key]; n = COUNTS[ds]; state = row['disposition']
        require(row['kind'] == 'fresh' and row['expected_QA'] == n and state in ('CLOSED_COMPLETE', 'CLOSED_WITH_FAILURES', 'STOPPED', 'NOT_RUN'), 'fresh R128 disposition differs')
        if state in ('STOPPED', 'NOT_RUN'):
            require(row.get('primary_mean') is None and row.get('reason') and frozen[key]['primary_mean'] is None, 'unrun confirmation quality must remain null and cannot hide an existing primary score'); e = row['disposition_evidence']; add(e['path'], 'confirmation_noncompletion_disposition', e['sha256']); outrows.append({**row, 'primary_mean': None}); continue
        raw = Path(g['harvest_root'])/'raw'/entry['item']['cell']['name']; scored = Path(g['harvest_root'])/'scored'/entry['item']['cell']['name']
        require(all((scored/name).is_file() for name in ('summary.json', 'scored_records.jsonl')), 'existing frozen R128 score required; export never creates one')
        proof = load(raw/PROOF)
        for e in proof['files']: add(raw/e['file'], 'confirmation_actual_producer_source_or_raw', e['sha256'], aliases=[e['remote_file']])
        add(raw/PROOF, 'independent_confirmation_source_proof')
        for rel, digest in load(raw/'manifest.json')['source_hashes'].items(): add(raw/'source_files'/rel, 'executed_confirmation_runtime_source', digest, rel)
        for name in ('summary.json', 'scored_records.jsonl'): add(scored/name, 'frozen_confirmation_score')
        point = load(scored/'summary.json'); require(canonical(point) == canonical(frozen[key]) and point['expected_QA'] == n and point['reused_reference_views'] == 0 and (point['primary_mean'] is not None) == (state == 'CLOSED_COMPLETE'), 'R128 frozen score/failure denominator differs')
        if deep:
            adapter.check_source(raw, entry['item']); actual, bundle = adapter.score_cell(entry, raw, plan)
            require(canonical(actual) == canonical(point), 'deep R128 source/physical/primary recomputation differs')
            if bundle is not None: bundles[key] = bundle
        identity = tuple(sha(raw/name) for name in ('manifest.json', 'records.jsonl', 'complete.json'))
        require(identity not in fresh_sources, 'confirmation reuses an existing inference producer'); fresh_sources.add(identity)
        counts['actual_view_answer_events'] += n; counts['actual_fresh_answer_events'] += n
        counts['fresh_reference_answer_events' if mode in REFS else 'fresh_candidate_answer_events'] += n
        counts['fresh_successful_generation_QA'] += point['fresh_QA']; counts['fresh_failure_status_QA'] += n-point['fresh_QA']
        if state == 'CLOSED_COMPLETE': counts['closed_quality_answer_events'] += n
        outrows.append({**row, 'raw': str(raw), 'scored': str(scored), 'proof_name': PROOF, 'primary_mean': point['primary_mean'], 'original_producer_QA': n})
    expected_pairs = {ds+'__'+candidate+'_minus_'+anchor for ds in COUNTS for candidate, anchor in g['paired_contrasts'] if (ds, candidate) in bundles and (ds, anchor) in bundles} if deep else set(pairs)
    require(set(pairs) == expected_pairs, 'R128 paired comparison coverage differs')
    if deep:
        for name in expected_pairs:
            ds, pair = name.split('__', 1); candidate, anchor = pair.split('_minus_', 1)
            require(canonical(pairs[name]) == canonical(adapter.shared.paired(bundles[ds, anchor], bundles[ds, candidate])), 'R128 paired CI/runtime/source recomputation differs')
    return {'group_id': g['group_id'], 'kind': g['kind'], 'cohort_label': g['cohort_label'], 'execution_disposition': 'EXECUTED', 'protocol': 'fresh_R128_confirmation', 'selected_modes': selected, 'declared_new_answer_events': 128*len(modes), 'fresh_reference_modes': list(REFS), 'view_cells': len(entries), 'paired_comparisons': len(pairs), 'reused_reference_view_events': 0, **dict(counts), 'cells': outrows}


def virtual_image_path(value):
    if value.startswith('${DATA_ROOT}/'): return '/PUBLIC/baseline_matrix_v3/'+value[len('${DATA_ROOT}/'):]
    if value.startswith('${PROJECT_ROOT}/'): return '/PUBLIC/baseline_matrix_v3/'+value[len('${PROJECT_ROOT}/'):]
    if value.startswith('${DATA_DISK}/'): return '/PUBLIC_DATA_DISK/'+value[len('${DATA_DISK}/'):]
    require(value.startswith('/media/'), 'unknown image-path redaction; no arbitrary relocation fallback')
    return value


@contextmanager
def scoped_driver_sha(driver, resolver):
    """Bind all driver wrappers' hash queries to explicit archived inputs."""
    pending = [driver]; seen = set(); saved = []
    try:
        while pending:
            module = pending.pop()
            if id(module) in seen: continue
            seen.add(id(module))
            if hasattr(module, 'sha'):
                saved.append((module, module.sha)); module.sha = resolver
            for name in ('primary', 'wo', 'frozen', 'base'):
                child = getattr(module, name, None)
                if child is not None and hasattr(child, '__file__'): pending.append(child)
        yield
    finally:
        for module, value in reversed(saved): module.sha = value


def verify_runtime_graph(driver, resolver, actual):
    with scoped_driver_sha(driver, resolver):
        require(actual == driver.runtime_source_manifest(), 'complete executed runtime role set and pinned epochs differ from the archived driver inventory')


def replay_loader(source, loader, inputs, manifest, expected_sha):
    """Exact archived no-answer loader, with indexed source identities only.

    The wrapper supplies original sizes/SHA from verified index entries. It
    relocates only reviewed machine prefixes in image metadata; image suffix,
    source IDs, question bytes and all other admission checks stay unchanged.
    """
    saved = {key: getattr(loader, key) for key in ('Path', 'read', 'rows', 'sha')}
    class IndexedPath:
        def __init__(self, value): self.member = value.member if isinstance(value, IndexedPath) else source.source(value)
        @property
        def name(self): return PurePosixPath(self.member).name
        def stat(self): return SimpleNamespace(st_size=source.entries[self.member]['source_bytes'])
    def member(value): return value.member if isinstance(value, IndexedPath) else source.source(value)
    def digest(value):
        rel = member(value); e = source.entries[rel]; require(sha(source.path(rel)) == e['public_sha256'], 'indexed loader input public bytes changed'); return e['source_sha256']
    def rows(value):
        values = [json.loads(line) for line in source.path(member(value)).read_text(encoding='utf-8-sig').splitlines() if line]
        for row in values: row['image_path'] = virtual_image_path(row['image_path'])
        return values
    try:
        loader.Path = IndexedPath; loader.read = lambda value: source.doc(member(value)); loader.rows = rows; loader.sha = digest
        return loader.load_confirmation_inputs(inputs, manifest, expected_sha)
    finally:
        for key, value in saved.items(): setattr(loader, key, value)


def verify_data(source, group, plan, loader):
    closure = group['confirmation_data']; inventory = closure['fetch_inventory']; remote = closure['remote_sha_receipt']
    source.original(inventory['path'], inventory['sha256']); source.original(remote['path'], remote['sha256'])
    records = source.doc(inventory['path']); proof = source.doc(remote['path'])
    require(proof['schema'] == 'trajkv.round11.confirm128_independent_remote_sha.v1' and proof['no_model_or_GPU_calls'] is True, 'independent frozen data proof required')
    require(len(records) == len({r['remote_path'] for r in records}) and {r['remote_path'] for r in records} == {r['path'] for r in proof['files']}, 'complete explicit data readback source closure differs')
    remote_map = {r['path']: r for r in proof['files']}
    for e in records:
        rel = source.original(e['local_path'], e['local_sha256']); p = remote_map[e['remote_path']]
        require(source.source(e['remote_path']) == rel or source.entries[source.source(e['remote_path'])]['source_sha256'] == e['local_sha256'], 'remote data source alias differs')
        require(e['local_sha256'] == p['sha256'] and e['bytes'] == p['bytes'] == source.entries[rel]['source_bytes'], 'independent data proof source SHA/size differs')
    for rel, digest in PINS.items(): source.original(rel, digest)
    source.original(plan['input_manifest_path'], MANIFEST_SHA); manifest = source.doc(plan['input_manifest_path'])
    require(manifest['schema'] == 'trajkv.round11.confirm128_inputs.v1' and manifest['answers_in_encoder_inputs'] is False and manifest['counts'] == COUNTS and manifest['total_count'] == 128, 'R128 manifest/cohort source identity differs')
    for key in ('source_locators', 'disjoint_proof', 'raw_source_proof', 'source_lock'):
        ref = manifest[key]; rel = source.original(ref['path'], ref['sha256']); require(source.entries[rel]['source_bytes'] == ref['bytes'], 'manifest evidence original size differs')
    disjoint = source.doc(manifest['disjoint_proof']['path']); raw = source.doc(manifest['raw_source_proof']['path'])
    require(disjoint['counts'] == COUNTS and disjoint['selected_count'] == 128 and disjoint['known_history_and_used_and_formal500_overlap'] == 0 and disjoint['Round10_all_cal64_D160_reserve224_overlap'] == 0 and disjoint['selection_reads_model_outcomes'] is False and disjoint['reference_text_decoded_after_complete_source_selection'] is True and disjoint['no_pretraining_unseen_or_absolute_history_independence_claim'] is True and raw['all_pinned_bytes_verified'] is True and raw['downloads_performed'] == 0, 'bounded data-disjoint/source-selection proof differs')
    lock = source.doc(manifest['source_lock']['path']); loaded = {}
    for ds, n in COUNTS.items():
        domain = plan['domains'][ds]
        for key in ('inputs', 'refs', 'source_lock'): source.original(domain[key], domain[key+'_sha256'])
        require(domain['source_lock_sha256'] == manifest['source_lock']['sha256'], 'R128 domain source lock differs')
        for key in ('inputs', 'refs'):
            rel = source.source(domain[key]); item = lock['files'][PurePosixPath(rel).name]
            require(item['sha256'] == domain[key+'_sha256'] and item['bytes'] == source.entries[rel]['source_bytes'], 'frozen input/reference source lock differs')
        values = replay_loader(source, loader, domain['inputs'], plan['input_manifest_path'], domain['inputs_sha256'])
        require(len(values) == n and {r['dataset'] for r in values} == {ds}, 'exact archived no-answer R128 loader admission differs')
        loaded[ds] = [json.loads(line) for line in source.path(domain['inputs']).read_text(encoding='utf-8-sig').splitlines() if line]
    all_rows = [r for ds in COUNTS for r in loaded[ds]]
    # Recompute the declared history/content exclusions from the archived lists.
    blocked = loader.Identities()
    for ref in disjoint['exclusion_coverage']:
        source.original(ref['path'], ref['sha256']); values = [json.loads(line) for line in source.path(ref['path']).read_text(encoding='utf-8-sig').splitlines() if line]
        require(len(values) == ref['count'], 'exclusion coverage count differs')
        for row in values: blocked.add(row)
    require(len({r['id'] for r in all_rows}) == len({r['source_cluster_id'] for r in all_rows}) == len({r['original_image_sha256'] for r in all_rows}) == 128 and all(not blocked.reasons(r) for r in all_rows), 'actual R128 source/content/history intersection failed')
    locators = [json.loads(line) for line in source.path(manifest['source_locators']['path']).read_text(encoding='utf-8-sig').splitlines() if line]
    require([r['id'] for r in locators] == [r['id'] for r in all_rows] and all(all(a[k] == b[k] for k in ('dataset', 'source_id', 'source_cluster_id', 'original_image_sha256')) for a, b in zip(locators, all_rows)), 'pre-reference frozen source locators differ from actual R128 rows')
    return manifest, loaded


def verify_producer(source, raw, entry, state, selection, manifest, inputs):
    under = lambda name: source.under(raw, name); doc = lambda name: source.doc(under(name))
    m, done, cfg, proof = map(doc, ('manifest.json', 'complete.json', 'source_config.json', PROOF)); item = entry['item']; cell = item['cell']; ds = entry['dataset']; mode = entry['mode']; n = COUNTS[ds]
    expected_cfg_mode = 'original' if mode in ('R7low', 'R7high') else mode
    require(entry['kind'] == 'fresh' and cell['script'] == SCRIPT and cell['script_sha256'] == PINS[SCRIPT] and m['schema'] == 'baseline_matrix.raw_inference.v1' and m['runner_schema'] == cfg['schema'] == RAW_SCHEMA and cfg['method']['family'] == 'trajkv_round11_confirmation' and cfg['candidate']['mode'] == expected_cfg_mode and cfg['variant_source']['name'] == ('r7_budget_k2_cond' if mode == 'R7high' else 'r7_joint_k1_gt_cond') and cfg['candidate'].get('technical_only', False) is False, 'R128 requires actual fresh confirmation wrapper and original reference/candidate lineage')
    require(cfg['input_role'] == cfg['cohort_kind'] == 'confirmation_holdout' and cfg['start'] == 0 and cfg['count'] == cfg['input_count'] == n and m['expected_QA'] == done['expected_QA'] == done['recorded_QA'] == n, 'R128 cannot accept a development or subset producer')
    require(m['references_read'] is False and m['prediction_cache_used'] is False and done['quality_scored'] is False and done['coverage_complete'] is True and m['all_reference_predictions_must_be_fresh'] is True and done['all_reference_predictions_fresh'] is True, 'fresh uncached/unscored confirmation raw3 required')
    require(m['config'] == cfg and m['config_sha256'] == cell['config_sha256'] and proof['schema'] == 'trajkv.round11.confirmation_source_proof.v2' and proof['status'] == 'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA' and proof['remote_hashes_computed_independently'] is True and proof['SourceRun_admission_established'] is False and proof['parent_producer'] == item['job']['name'] and proof['cell_name'] == cell['name'] and proof['actual_driver'] == SCRIPT, 'R128 independent actual producer proof differs')
    require(len(proof['files']) == len({e['file'] for e in proof['files']}), 'duplicate confirmation proof member')
    for e in proof['files']:
        rel = source.original(under(e['file']), e['sha256']); require(source.entries[rel]['source_bytes'] == e['bytes'], 'source proof original length differs')
    for name, digest in (('manifest.json', done['manifest_sha256']), ('records.jsonl', done['records_jsonl_sha256']), ('source_config.json', cell['config_sha256']), ('source_job.json', item['job_sha256']), ('source_sequence_config.json', item['sequence_sha256']), ('source_parent_driver.py', PINS['code/round10_sequence_v1.py']), ('input_manifest.json', MANIFEST_SHA), ('selection_manifest.json', item['selection_manifest_sha256'])): source.original(under(name), digest)
    require(cfg['input_manifest_sha256'] == item['input_manifest_sha256'] == MANIFEST_SHA and cfg['selection_manifest_sha256'] == item['selection_manifest_sha256'] and doc('selection_manifest.json') == selection, 'R128 data/selection source binding differs')
    job, seq, launch, inventory = map(doc, ('source_job.json', 'source_sequence_config.json', 'launch_receipt.json', 'source_inventory.json'))
    require(job == item['job'] and seq['schema'] == 'trajkv.round10.sequence.v1' and [r for r in seq['entries'] if r['name'] == cell['name']] == [cell], 'actual confirmation parent job/sequence differs')
    root = item['root'].rstrip('/'); snapshot = root+'/jobs/'+job['name']+'.source'; expected = [snapshot+'/code/'+a.rsplit('/code/', 1)[1] if a.startswith(root+'/code/') and a.endswith('.py') else a for a in job['argv']]
    require(launch['job_sha256'] == item['job_sha256'] and launch['name'] == job['name'] and type(launch['pid']) is int and launch['pid'] > 0 and launch['source_snapshot'] == snapshot == m['runtime_source_root'] and launch['argv_executed'] == expected and expected[2] == snapshot+'/code/round10_sequence_v1.py', 'actual parent PID/snapshot/argv differs')
    source.original(under('source_inventory.json'), launch['source_inventory_sha256'])
    event = proof['actual_child_start']; start = event['event']
    require(json.loads(event['exact_line_utf8']) == start and start['name'] == cell['name'] and start['config_sha256'] == cell['config_sha256'] and type(start['pid']) is int and start['pid'] > 0 and start['argv'] == [expected[0], '-B', snapshot+'/'+SCRIPT, '--config', cell['config'], '--output', cell['output']] and m['runtime_process']['PID'] == start['pid'] and m['runtime_process']['parent_PID'] == launch['pid'] and m['runtime_process']['source_snapshot_root'] == snapshot, 'actual child PID/parent PID/START_CELL differs')
    require(m['source_hashes'] == cfg['expected_source_hashes'] and len(m['source_hashes']) == 24 and inventory['code/round10_sequence_v1.py'] == PINS['code/round10_sequence_v1.py'], 'complete 24-file confirmation runtime graph required')
    for rel, digest in m['source_hashes'].items():
        require(inventory[rel] == digest, 'snapshot inventory/source graph differs'); source.original(under('source_files/'+rel), digest); source.original(rel, digest)
    for rel, digest in PINS.items():
        source.original(rel, digest)
        if rel.startswith('code/') and rel != 'code/round10_sequence_v1.py': require(m['source_hashes'][rel] == digest, 'executed frozen R128 source pin differs')
    parent = json.loads(proof['parent_state_snapshot']['body_utf8']); require(parent['source_root'] == snapshot and parent['plan_sha256'] == item['sequence_sha256'], 'parent state scope differs'); closed = [r for r in parent['completed_cells'] if r['name'] == cell['name']]
    if proof['parent_cell_success']:
        require(len(closed) == 1 and closed[0]['exit_code'] == 0 and closed[0]['recorded_QA'] == n and closed[0]['output'] == cell['output'] and closed[0]['complete_sha256'] == source.entries[under('complete.json')]['source_sha256'] and done['status'] == 'PASS_RAW_COMPLETE', 'parent exit0 does not bind actual complete confirmation')
    else:
        failed = parent.get('failed_cell', {}); require(not closed and failed.get('name') == cell['name'] and failed['exit_code'] != 0 and failed['output'] == cell['output'] and done['status'] == 'COMPLETE_WITH_FAILURES', 'preserved failed confirmation parent differs')
    require(proof['parent_cell_success'] or state == 'CLOSED_WITH_FAILURES', 'quality-complete confirmation lacks parent exit0')
    if source.entries[under(PROOF)]['transformation'] == 'EXACT': require(hashlib.sha256(event['exact_line_utf8'].encode()).hexdigest() == event['exact_line_sha256'] and hashlib.sha256(proof['parent_state_snapshot']['body_utf8'].encode()).hexdigest() == proof['parent_state_snapshot']['sha256'], 'original embedded START/state snapshot bytes differ')
    identity = m['confirmation_identity']; require(identity['schema'] == 'trajkv.round11.confirm128_identity.v1' and identity['manifest_sha256'] == MANIFEST_SHA and identity['dataset'] == ds and identity['count'] == n and identity['ids'] == [r['id'] for r in inputs] and identity['role'] == 'confirmation_holdout' and identity['fresh_inference_required'] is True and identity['source_disjoint_proof_sha256'] == manifest['disjoint_proof']['sha256'] and identity['raw_source_proof_sha256'] == manifest['raw_source_proof']['sha256'] and identity['source_lock_sha256'] == manifest['source_lock']['sha256'], 'actual no-answer confirmation loader identity differs')
    require(m['confirmation_selection'] == {'manifest_sha256': item['selection_manifest_sha256'], 'selected_modes': selection['selected_modes'], 'selection_scope': selection['selection_scope'], 'development_harvest_summary_sha256': selection['development_harvest_summary_sha256']}, 'actual confirmation selected-mode identity differs')
    source.original(under('source_inputs.jsonl'), cfg['inputs_sha256']); source.original(under('assets/original_profile.json'), cfg['variant_source']['sha256'])
    profile = doc('assets/original_profile.json'); variants = [v for v in profile['variants'] if v['name'] == cfg['variant_source']['name']]; numeric = lambda v: {k: value for k, value in v.items() if k not in cfg['expected_asset_sha256']}
    require(len(variants) == 1 and numeric(variants[0]) == numeric(cfg['variant']), 'original R7 base numeric lineage changed')
    for role, digest in cfg['expected_asset_sha256'].items(): source.original(under('assets/'+role+'.json'), digest)
    proven = {(e['remote_file'], e['sha256']) for e in proof['files'] if e['file'].startswith('legacy_sources/')}; require(set(done.get('executed_legacy_source_sha256', {}).items()) <= proven, 'executed legacy source closure omitted')
    rows = [json.loads(line) for line in source.path(under('records.jsonl')).read_text(encoding='utf-8-sig').splitlines() if line]
    require([r['id'] for r in rows] == m['expected_ids'] == identity['ids'] and len(rows) == n and done['counts'] == dict(Counter(r['status'] for r in rows)), 'whole R128 producer order/status/coverage differs')
    return m, done, cfg, proof, rows


def verify_group(source, group, plan, phase, scorer, metrics, physical, public_module, resolver, fresh_producers, all_groups):
    """Independent portable all-fresh R128 source/score/physical/paired replay."""
    require(PINS[SCRIPT] is not None and PINS[ADAPTER] is not None, 'confirmation-v2 exact source epochs remain unbound')
    bases = group['selection_basis']; require(set(bases) == {'primary', 'wo'}, 'selection basis requires both included development groups')
    basis_hashes = {}
    for role, basis in bases.items():
        require(set(basis) == {'group_id', 'path', 'sha256'}, 'explicit development group/path/SHA basis required')
        development = next((g for g in all_groups if g['group_id'] == basis['group_id']), None)
        require(development is not None and development['execution_disposition'] == 'EXECUTED' and source.source(development['summary_root']+'/summary.json') == source.source(basis['path']), 'selection basis must bind both included complete D96 groups')
        source.original(basis['path'], basis['sha256']); validate_development_summary(role, source.doc(basis['path'])); basis_hashes[role] = basis['sha256']
    source.original(plan['selection_manifest_path'], plan['selection_manifest_sha256']); selection = source.doc(plan['selection_manifest_path']); selected = selection_modes(selection, basis_hashes); entries = plan_shape(plan, selected)
    require(group['adapter']['sha256'] == PINS[ADAPTER] and group['counts'] == COUNTS and group['modes'] == [*REFS, *selected] and group['paired_contrasts'] == contrasts(selected), 'explicit actual R128 adapter/counts/modes/contrasts differ')
    for rel, digest in PINS.items(): source.original(rel, digest)
    loader = public_module('round11_confirmation_data_v1', source.path('code/round11_confirmation_data_v1.py'))
    manifest, input_rows = verify_data(source, group, plan, loader)
    driver = public_module('round11_confirmation_runner_v2', source.path(SCRIPT))
    patched = []
    for mod in (driver, driver.primary, driver.wo, driver.frozen):
        patched.append((mod, mod.sha)); mod.sha = resolver
    try:
        summary = source.doc(group['summary_root']+'/summary.json'); frozen_pairs = source.doc(group['summary_root']+'/paired_comparisons.json')
        require(summary['schema'] == 'trajkv.round11.confirmation_harvest_summary.v2' and summary['plan_sha256'] == group['plan']['sha256'] and summary['input_manifest_sha256'] == MANIFEST_SHA and summary['selection_manifest_sha256'] == plan['selection_manifest_sha256'] and summary['expected_new_QA'] == 128*(3+len(selected)) and summary['expected_fresh_reference_QA'] == 384 and summary['reused_reference_views'] == 0, 'frozen R128 summary source/accounting differs')
        for rel, digest in summary['pins'].items(): source.original(rel, digest)
        rows = {(r['dataset'], r['mode']): r for r in group['cells']}; frozen_points = {(p['dataset'], p['mode']): p for p in summary['points']}
        require(len(rows) == len(group['cells']) and set(rows) == set(entries) == set(frozen_points), 'every R128 actual cell needs a final disposition and frozen point')
        declared_phase = next(g for g in phase['groups'] if g['group_id'] == group['group_id']); quality = []; bundles = {}; counts = Counter(); physical_count = 0
        for key, entry in entries.items():
            ds, mode = key; row = rows[key]; n = COUNTS[ds]; state = row['disposition']
            require(row['kind'] == 'fresh' and row['expected_QA'] == n and state in ('CLOSED_COMPLETE', 'CLOSED_WITH_FAILURES', 'STOPPED', 'NOT_RUN'), 'reused/partial/pending confirmation cell')
            if state in ('STOPPED', 'NOT_RUN'):
                require(row.get('reason') and row.get('primary_mean') is None and frozen_points[key]['primary_mean'] is None, 'unrun confirmation must have null quality/reason and cannot hide an existing primary score'); e = row['disposition_evidence']; source.original(e['path'], e['sha256']); quality.append({'group_id': group['group_id'], **row, 'scored_QA': 0, 'primary_recomputed': False}); continue
            phase_row = next(r for r in declared_phase['cells'] if (r['dataset'], r['mode']) == key); raw, scored = phase_row['raw'], phase_row['scored']
            require(phase_row['proof_name'] == PROOF and phase_row['original_producer_QA'] == n, 'R128 producer proof/type/denominator changed')
            m, done, cfg, proof, actual = verify_producer(source, raw, entry, state, selection, manifest, input_rows[ds])
            verify_runtime_graph(driver, resolver, m['source_hashes'])
            domain = plan['domains'][ds]; require(cfg['inputs_sha256'] == m['inputs_sha256'] == domain['inputs_sha256'] and m['confirmation_identity']['input_sha256'] == domain['inputs_sha256'], 'actual no-answer input/loader digest differs')
            if mode != 'FP16': physical.r7_verify_executed_roles(cfg, done.get('executed_legacy_source_sha256', {}))
            point = source.doc(scored+'/summary.json'); require(canonical(point) == canonical(frozen_points[key]), 'per-cell and group frozen R128 score differ')
            source.original(raw+'/'+PROOF, point['source_receipt_sha256']); source.original(scored+'/scored_records.jsonl', point['scored_records_sha256'])
            scores = [json.loads(line) for line in source.path(scored+'/scored_records.jsonl').read_text(encoding='utf-8-sig').splitlines() if line]; identities = source.line_ids(raw+'/records.jsonl'); refs = {r['id']: r for r in map(json.loads, source.path(domain['refs']).read_text(encoding='utf-8-sig').splitlines())}
            require(len(scores) == len(actual) == len(identities) == n and point['expected_QA'] == n and point['reused_reference_views'] == 0 and point['canonical_SourceRun_admission_claimed'] is False and point['failed_QA_remain_in_primary_denominator'] is True, 'R128 exact full scoring denominator/admission differs')
            failed = 0; capacities = []; pair_identities = {}
            for number, (r, source_row, s, identity) in enumerate(zip(actual, input_rows[ds], scores, identities), 1):
                require(r['id'] == source_row['id'] == s['id'] and s['raw_line_number'] == number == identity['line'] and s['raw_line_sha256'] == identity['source_sha256'], 'R128 score is not bound to exact original raw line')
                require(all(r.get(k) == source_row.get(k) for k in ('dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id', 'question_id', 'original_image_sha256', 'question_sha256')) and refs[r['id']]['source_cluster_id'] == source_row['source_cluster_id'] and r.get('official_score') is None and r['status'] in scorer.FINISHED_STATUSES, 'R128 source identity/reference separation/raw status differs')
                if r['status'] != 'PASS': require(s['score'] is None and s['score_status'] == 'NOT_SCORED_RAW_FAILURE', 'R128 raw failure received a primary score'); failed += 1; continue
                require(r['raw_question'] == source_row['question'] and r['question_suffix'] == driver.frozen.SUFFIX and isinstance(r.get('raw_prediction'), str) and isinstance(r.get('output_token_ids'), list) and r['output_token_ids'] and r['question_unknown_prefix_verified'] is True and r['original_full_prefix_released'] is True and r['original_FP16_visual_alias_retained'] is False and r['first_generation_call']['native_first_logits_injected'] is False, 'R128 genuine native full-suffix/free-generation/source-release differs')
                amount = physical.physical_complete_bytes(r, m['model_spec']); physical_count += 1; capacities.append(amount)
                require(r['owner_after_generation']['stored_visual_KV_bytes'] == amount and math.isclose(r['actual_effective_visual_bits'], 16*amount/r['full_kv_byte_denominator'], rel_tol=1e-12, abs_tol=1e-12), 'R128 actual complete owner/bits differs')
                require(r['round10_budget_vs_original_R7_low'] == driver.physical_budget_guard(cfg['candidate']['mode'], r['actual_visual_token_count'], amount, r['r7_bridge'], cfg['variant_source']['name']), 'R128 exact deployed complete physical budget guard differs')
                if mode != 'FP16': physical.r7_verify_record_role_pins(cfg, r)
                from round11_public_metric_contracts_v5 import verify_metric_fields, CANONICAL_METRIC_GROUP
                metric = verify_metric_fields(metrics, ds, r['raw_prediction'], refs[r['id']], s, CANONICAL_METRIC_GROUP)
                require(s['score_status'] == 'SCORED_PASS' and s['output_token_ids'] == r['output_token_ids'] and s['raw_prediction'] == r['raw_prediction'], 'R128 frozen primary metric/reference/decoded answer differs')
                pair_identities[r['id']] = {**physical.input_identity(r), 'raw_question_sha256': r['raw_question_sha256']}
            aggregate = scorer.summarize_group(scores, 2000, 20261123)
            require(canonical(point['groups']['overall']) == canonical(aggregate) and point['primary_mean'] == aggregate['primary']['mean'] and point['ci95'] == aggregate['primary']['ci95'] and point['fresh_QA'] == n-failed, 'R128 primary full-denominator mean/cluster CI differs')
            if failed: require(state == 'CLOSED_WITH_FAILURES' and point['primary_mean'] is None and point['resources'] is None, 'failed confirmation produced successful-subset quality')
            else:
                require(state == 'CLOSED_COMPLETE' and done['status'] == 'PASS_RAW_COMPLETE' and point['status'] == 'PASS_FRESH_R128_SOURCE_SCORE', 'R128 closed complete disposition differs')
                expected = {'n': n, 'mean': sum(capacities)/n, 'min': min(capacities), 'max': max(capacities)}
                require(all(math.isclose(point['resources']['complete_bytes'][k], v, rel_tol=1e-12, abs_tol=1e-12) for k, v in expected.items()), 'R128 full complete-owner summary differs')
                bundles[key] = {'context': physical.native_context(m), 'identities': pair_identities, 'scores': {r['id']: r for r in scores}}
            producer = tuple(source.entries[source.source(raw+'/'+name)]['source_sha256'] for name in ('manifest.json', 'records.jsonl', 'complete.json')); require(producer not in fresh_producers, 'R128 reused an earlier inference producer'); fresh_producers.add(producer)
            counts['actual_view_answer_events'] += n; counts['actual_fresh_answer_events'] += n; counts['fresh_reference_answer_events' if mode in REFS else 'fresh_candidate_answer_events'] += n
            counts['fresh_successful_generation_QA'] += n-failed; counts['fresh_failure_status_QA'] += failed
            if not failed: counts['closed_quality_answer_events'] += n
            quality.append({'group_id': group['group_id'], 'dataset': ds, 'mode': mode, 'disposition': state, 'kind': 'fresh', 'original_producer_QA': n, 'selected_QA': n, 'scored_QA': n-failed, 'primary_mean': point['primary_mean'], 'primary_recomputed': True})
        comparisons = {}; pairs_out = []
        for ds in COUNTS:
            for candidate, anchor in group['paired_contrasts']:
                if (ds, candidate) not in bundles or (ds, anchor) not in bundles: continue
                left, right = bundles[ds, anchor], bundles[ds, candidate]; require(left['context'] == right['context'] and left['identities'] == right['identities'], 'R128 paired source/runtime/positions differ')
                deltas = [{'id': i, 'source_cluster_id': r['source_cluster_id'], 'delta': right['scores'][i]['score']-r['score']} for i, r in left['scores'].items()]; changes = [r['delta'] for r in deltas]
                value = {'schema': 'trajkv.round11.paired_source_group.v1', 'status': 'PASS_PAIRED_COMPLETE', **scorer.cluster_bootstrap(deltas, 'delta', 2000, 20261123), 'improved': sum(v > 0 for v in changes), 'degraded': sum(v < 0 for v in changes), 'equal': sum(v == 0 for v in changes), 'gross_gain_pp': 100*sum(max(0, v) for v in changes)/len(changes), 'gross_loss_pp': 100*sum(max(0, -v) for v in changes)/len(changes), 'descriptive_unadjusted_interval': True, 'source_identity_and_runtime_exactly_matched': True}
                name = ds+'__'+candidate+'_minus_'+anchor; comparisons[name] = value; pairs_out.append({'group_id': group['group_id'], 'comparison': name, 'paired_recomputed': True, 'runtime_source_positions_exact': True})
        require(canonical(comparisons) == canonical(frozen_pairs), 'R128 all declared paired comparisons/CI differ')
        counts['view_cells'] = len(entries); counts['paired_comparisons'] = len(comparisons)
        for key, value in counts.items(): require(declared_phase.get(key, 0) == value, 'R128 phase actual accounting differs: '+key)
        require(declared_phase['reused_reference_view_events'] == 0 and declared_phase['selected_modes'] == selected and declared_phase['declared_new_answer_events'] == 128*(3+len(selected)) and summary['actual_closed_fresh_QA'] == sum(q['scored_QA'] for q in quality), 'R128 fresh references/selection/quality event accounting differs')
        return {'counts': dict(counts), 'quality': quality, 'pairs': pairs_out, 'physical_count': physical_count, 'group': {'group_id': group['group_id'], 'protocol': 'fresh_R128_confirmation', 'execution_disposition': 'EXECUTED', 'selected_modes': selected, 'reused_reference_view_events': 0, **dict(counts)}, 'data': {'exact_pinned_manifest_verified': True, 'archived_no_answer_loader_replayed': True, 'known_history_and_source_disjointness_recomputed': True, 'images_weights_parquet_or_GPU_replayed': False}}
    finally:
        for mod, old in reversed(patched): mod.sha = old
