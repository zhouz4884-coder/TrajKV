"""Prospective, score-blind Round12 independent-source data materializer.

This module does not select a final N by itself. A completed frozen decision,
counts, seed, exact source/exclusion bytes and fresh-output requirement are
mandatory before --build. No model import, job launch or download exists here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True
SCHEMA = 'trajkv.round12.independent_confirmation_inputs.v1'
PLAN_SCHEMA = 'trajkv.round12.independent_confirmation_data_plan.v1'
DECISION_SCHEMA = 'trajkv.round12.confirmation_selection.v1'
ROLE = 'round12_independent_confirmation'
DOMAINS = ('infovqa', 'docvqa', 'textvqa')
REQUIRED_EXCLUSION_ROLES = (
    'round12_development', 'round12_calibration_fit', 'round12_calibration_select',
    'formal500', 'round11_original128', 'round11_additional384', 'registered_history',
)
DEPENDENCY_PINS = {
    'round10_data_builder_v1.py': '914dfa258e6fbcfc00c709df6e78cfba6cba5c385db7c51f1e15ebb5d0f59936',
    'round10_inputs_v1.py': '61cc6dca53d95f0f24ab0a3b160eeb7be5fbac179e5ded510a2c2852e8e3dd77',
}
FIELDS = {'id', 'dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id',
          'question_id', 'image_path', 'original_image_sha256', 'width', 'height',
          'question', 'group', 'image_format', 'image_bytes', 'original_split_row', 'question_sha256'}
ALLOWED = {'infovqa': ('539088ef8a8ada01ac8e2e6d4e372586748a265e', {'train', 'validation'}),
           'docvqa': ('539088ef8a8ada01ac8e2e6d4e372586748a265e', {'train', 'validation'}),
           'textvqa': ('9c0699cd19768ac5ab97568f6b3cbac4c0062884', {'validation'})}
IMAGE_HASH_FIELDS = ('original_image_sha256', 'image_sha256', 'sha256', 'reencoded_image_sha256', 'legacy_image_sha256')


def require(value, message):
    if not value:
        raise ValueError('ROUND12_CONFIRM_DATA: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as s:
        return [json.loads(line) for line in s if line.strip()]


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == body, 'existing evidence differs; use a new run: ' + str(path))
    else:
        with path.open('xb') as s:
            s.write(body)
            s.flush()
            os.fsync(s.fileno())
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def freeze_json(path, value):
    return freeze(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def freeze_rows(path, values):
    return freeze(path, ''.join(json.dumps(v, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False) + '\n' for v in values).encode())


def priority(seed, kind, value):
    return hashlib.sha256((seed + '\0' + kind + '\0' + str(value)).encode()).hexdigest()


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def output_path(path, project):
    path = Path(path).resolve()
    require(inside(path, project), 'output escapes named project')
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(ancestor.stat().st_dev == project.stat().st_dev, 'output/cache is not on project data device')
    return path


def pinned(item, plan):
    path = Path(item['path']).resolve()
    roots = [Path(p).resolve() for p in plan.get('read_only_source_roots', [plan['project_root']])]
    require(any(inside(path, r) for r in roots), 'source escapes explicitly declared read-only roots')
    require(path.is_file() and sha(path) == item['sha256'], 'pinned source missing/changed: ' + str(path))
    require('bytes' not in item or path.stat().st_size == item['bytes'], 'pinned source size changed')
    return path


class Identities:
    """Metadata-only known exposure, including Doc parent and explicit SHA aliases."""
    def __init__(self):
        self.ids, self.questions, self.sources, self.clusters, self.shas = set(), set(), set(), set(), set()
        self.partial_records = 0

    def add(self, row):
        ds = row.get('dataset')
        ident = row.get('id')
        if ds is None and isinstance(ident, str):
            ds = ident.split(':', 1)[0]
        if isinstance(ds, str):
            ds = ds.lower()
        if ident and not str(ident).startswith('UNRESOLVED_'):
            self.ids.add(str(ident))
        qid = row.get('question_id')
        if qid is not None and ds:
            self.questions.add((str(ds), str(qid)))
        source = row.get('source_id')
        if source and not str(source).startswith('UNRESOLVED_') and ds:
            self.sources.add((str(ds), str(source)))
            self.clusters.add(('docvqa:ucsf:' + str(source).rsplit('/', 1)[0]) if ds == 'docvqa'
                              else str(ds) + ':image:' + str(source))
        cluster = row.get('source_cluster_id', row.get('cluster'))
        if cluster and not str(cluster).startswith('UNRESOLVED_'):
            self.clusters.add(str(cluster))
        hashes = [row.get(k) for k in IMAGE_HASH_FIELDS]
        hashes += list(row.get('image_sha_aliases', []))
        for value in hashes:
            if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value):
                self.shas.add(value)
        recognized = bool((ident and not str(ident).startswith('UNRESOLVED_'))
            or (source and ds and not str(source).startswith('UNRESOLVED_'))
            or (cluster and not str(cluster).startswith('UNRESOLVED_'))
            or any(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in hashes))
        require(recognized, 'exclusion record has no usable question/source/cluster/image identity')
        if not (ds and source and cluster and any(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in hashes)):
            self.partial_records += 1

    def reasons(self, row):
        ds, source = row.get('dataset'), row.get('source_id')
        result = []
        if row.get('id') in self.ids or (ds, str(row.get('question_id'))) in self.questions:
            result.append('question_identity')
        if (ds, str(source)) in self.sources:
            result.append('source_identity')
        cluster = row.get('source_cluster_id', row.get('cluster'))
        derived = ('docvqa:ucsf:' + str(source).rsplit('/', 1)[0]) if ds == 'docvqa' and source else None
        if cluster in self.clusters or derived in self.clusters:
            result.append('source_or_document_parent')
        if any(row.get(k) in self.shas for k in IMAGE_HASH_FIELDS if row.get(k)):
            result.append('original_or_registered_image_SHA')
        return result


def exclusion_rows(path, item):
    if item.get('format', 'jsonl') == 'jsonl':
        value = rows(path)
    else:
        require(item.get('format') == 'json' and isinstance(item.get('identity_json_key'), str),
                'JSON exclusion requires an explicit metadata list key (empty means root list)')
        value = read(path)
        if item['identity_json_key']:
            for key in item['identity_json_key'].split('.'):
                value = value[key]
    require(isinstance(value, list) and all(isinstance(r, dict) for r in value), 'identity JSON key is not a record list')
    if item.get('adapter') == 'legacy_grouped_docvqa_v1':
        # Only identity metadata is read from the grouped historical sessions;
        # nested questions/turns/reference text do not participate in selection.
        normalized = []
        for old in value:
            require(old.get('dataset') in ('docvqa_grouped', 'docvqa'), 'legacy adapter is DocVQA only')
            doc = old.get('source_document_id')
            require(isinstance(doc, str) and bool(doc) and not doc.startswith('UNRESOLVED_'),
                    'legacy grouped Doc parent unresolved')
            hashes = list(old.get('hashes', []))
            for field in ('source_image_id', 'image_id'):
                v = old.get(field)
                if isinstance(v, str) and v.startswith('sha256:'):
                    hashes.append(v.split(':', 1)[1])
            qids = old.get('source_question_ids') or [None]
            for qid in qids:
                normalized.append({'dataset': 'docvqa', 'id': old.get('id'),
                    'question_id': str(qid) if qid is not None else None,
                    'source_cluster_id': 'docvqa:ucsf:' + doc,
                    'image_sha_aliases': sorted(set(hashes))})
        return normalized
    require(item.get('adapter') in (None, 'identity_fields'), 'unknown exclusion metadata adapter')
    return value


def ordered_sources(candidates, dataset, seed):
    groups = defaultdict(list)
    for c in candidates:
        groups[c['source_cluster_id']].append(c)
    for cluster in sorted(groups, key=lambda k: (priority(seed, 'confirm:source:' + dataset, k), k)):
        # One question per source/document. Tie-breaking does not read text.
        yield min(groups[cluster], key=lambda c: (priority(seed, 'question:' + dataset,
            cluster + '\0' + c['question_id']), c['question_id'], c['kind'], str(c.get('source_basename', ''))))


def validate_input_rows(values, expected_count=None):
    require(values and (expected_count is None or len(values) == expected_count), 'wrong no-answer input count')
    for row in values:
        require(FIELDS - {'group'} <= set(row) <= FIELDS, 'input contains labels/unknown or missing fields')
        ds = row['dataset']
        require(ds in ALLOWED and row['dataset_revision'] == ALLOWED[ds][0] and row['split'] in ALLOWED[ds][1], 'revision/split differs')
        require(row['id'] == f"{ds}:{row['split']}:{row['question_id']}", 'question/source ID differs')
        require(hashlib.sha256(row['question'].encode()).hexdigest() == row['question_sha256'], 'question hash differs')
        require(row['source_cluster_id'] == ('docvqa:ucsf:' + row['source_id'].rsplit('/', 1)[0]
                if ds == 'docvqa' else ds + ':image:' + row['source_id']), 'source/document cluster differs')
        p = PurePosixPath(row['image_path'])
        require(p.is_absolute() and '..' not in p.parts and '\\' not in row['image_path']
                and '/baseline_matrix_v3/' in row['image_path'] and p.stem == row['original_image_sha256'], 'noncanonical data image path')
        require(re.fullmatch('[0-9a-f]{64}', row['original_image_sha256']) is not None, 'invalid image SHA')
        require(all(type(row[k]) is int and row[k] > 0 for k in ('width', 'height', 'image_bytes')), 'invalid image dimensions')
    for field in ('id', 'source_cluster_id', 'original_image_sha256'):
        require(len({r[field] for r in values}) == len(values), 'duplicate question/source/content: ' + field)
    return values


def validate_plan(plan):
    require(plan.get('schema') == PLAN_SCHEMA, 'unknown plan schema')
    counts = plan.get('counts')
    require(isinstance(counts, dict) and {'infovqa', 'docvqa'} <= set(counts) <= set(DOMAINS)
            and all(type(n) is int and n > 0 for n in counts.values()), 'counts must be fixed after method selection; no default N')
    require(isinstance(plan.get('seed'), str) and bool(plan['seed']), 'explicit frozen seed required')
    for key in ('project_root', 'runtime_project_root', 'sources_root', 'output_root', 'inputs_root', 'images_root', 'cache_root'):
        require(isinstance(plan.get(key), str) and bool(plan[key]), 'explicit configured path required: ' + key)
    require(plan.get('allow_download') is False and plan.get('read_reference_text_after_selection_only') is True
            and plan.get('selection_reads_model_predictions_or_correctness') is False
            and plan.get('selection_reads_question_text') is False, 'score-blind/no-download flags required')
    require(plan.get('native_hashes_from_old_predictions') is False
            and plan.get('fresh_native_baselines_and_candidate_required') is True, 'fresh native inference contract required')
    require(isinstance(plan.get('historical_coverage_limits'), list) and plan['historical_coverage_limits'], 'explicit history limitations required')
    roles = {r for item in plan.get('exclusions', []) for r in item.get('roles', [])}
    require(set(REQUIRED_EXCLUSION_ROLES) <= roles, 'missing required exposure groups: ' + str(set(REQUIRED_EXCLUSION_ROLES) - roles))
    require(isinstance(plan.get('frozen_decision'), dict), 'candidate/size decision must be frozen before build')
    for spec in plan.get('parquet_sources', []):
        if spec.get('dataset') in counts and not spec.get('calibration_only'):
            require(spec['dataset'] in ('infovqa', 'docvqa'),
                    'TextVQA is supported through pinned canonical no-answer catalogs, not this parquet schema')
    return plan


def decision_identity(plan):
    p = pinned(plan['frozen_decision'], plan)
    decision = read(p)
    require(decision.get('schema') == DECISION_SCHEMA and decision.get('confirmation_outputs_read') is False
            and decision.get('selection_scope') == 'observed_development_only'
            and decision.get('counts') == plan['counts'] and decision.get('seed') == plan['seed']
            and decision.get('methods_and_assets_frozen_before_confirmation') is True,
            'decision does not freeze methods/size/seed before confirmation outputs')
    methods, comparators = decision.get('selected_methods'), decision.get('comparators')
    require(isinstance(methods, list) and bool(methods)
            and all(isinstance(m, str) and m for m in methods) and len(set(methods)) == len(methods),
            'freeze explicit nonduplicated selected methods')
    require(isinstance(comparators, list) and {'FP16', 'R7low', 'R7high'} <= set(comparators), 'fresh FP16/R7 baseline comparators required')
    locks = decision.get('method_asset_locks')
    require(isinstance(locks, list) and locks, 'source/config/parameter lock assets required')
    for item in locks:
        pinned(item, plan)
    return {'path': str(p), 'sha256': sha(p), 'selected_methods': methods,
            'comparators': comparators, 'method_asset_locks': locks}


def metadata_catalog(path, pool):
    result = []
    for row in rows(path):
        require(set(row) <= FIELDS and row['dataset'] == pool['dataset'], 'canonical catalog is not no-answer metadata')
        # Drop question text before source ordering. It is opened again only
        # after SELECTED_SOURCE_LOCATORS is durably frozen.
        metadata = {k: v for k, v in row.items() if k != 'question'}
        result.append({'kind': 'canonical', 'input_metadata': metadata, 'inputs_file': str(path),
                       'refs_file': pool['refs']['path'], **{k: metadata[k] for k in (
                           'id', 'dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id', 'question_id')}})
    return result


class Materializer:
    def __init__(self, plan, sources, project):
        self.plan, self.sources, self.project = plan, sources, project
        self.cache_key = self.cache_table = None
        self.input_cache, self.ref_cache, self.images = {}, {}, {}

    def raw(self, c, columns):
        import pyarrow.parquet as pq
        p = self.sources[c['source_basename']]
        key = (str(p), c['row_group'], tuple(columns))
        if key != self.cache_key:
            self.cache_table = pq.ParquetFile(p).read_row_group(c['row_group'], columns=columns)
            self.cache_key = key
        return self.cache_table.slice(c['row_in_group'], 1).to_pylist()[0]

    def inspect_image(self, c):
        if c['kind'] == 'canonical':
            row = dict(c['input_metadata'])
            p = Path(row['image_path']).resolve()
            require(inside(p, self.project) and p.is_file() and sha(p) == row['original_image_sha256']
                    and p.stat().st_size == row['image_bytes'], 'canonical original image differs')
            return row, None
        from PIL import Image
        image = self.raw(c, ['image'])['image']
        data = image['bytes'] if isinstance(image, dict) else image
        require(isinstance(data, bytes), 'pinned source lacks original image bytes')
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as im:
            width, height, fmt = *im.size, im.format.lower()
        suffix = 'jpg' if fmt == 'jpeg' else fmt
        p = output_path(Path(self.plan['images_root']) / (digest + '.' + suffix), self.project)
        row = {k: c[k] for k in ('id', 'dataset', 'dataset_revision', 'split', 'source_id',
                'source_cluster_id', 'question_id', 'original_split_row')}
        row.update(original_image_sha256=digest, image_path=str(p), width=width, height=height,
                   image_bytes=len(data), image_format=fmt, group=c.get('group'))
        return row, data

    def commit_image(self, row, body):
        if body is not None:
            self.images[row['image_path']] = freeze(row['image_path'], body)

    def finish(self, c, selected):
        if c['kind'] == 'canonical':
            if c['inputs_file'] not in self.input_cache:
                self.input_cache[c['inputs_file']] = {r['id']: r for r in rows(c['inputs_file'])}
            if c['refs_file'] not in self.ref_cache:
                self.ref_cache[c['refs_file']] = {r['id']: r for r in rows(c['refs_file'])}
            row = dict(self.input_cache[c['inputs_file']][c['id']])
            reference = self.ref_cache[c['refs_file']][c['id']]
            ref = {k: reference[k] for k in ('id', 'source_cluster_id', 'answers')}
        else:
            raw = self.raw(c, ['questionId', 'question', 'answers'])
            require(str(raw['questionId']) == c['question_id'], 'source question locator changed')
            question = raw['question']
            row = dict(selected, question=question, question_sha256=hashlib.sha256(question.encode()).hexdigest())
            ref = {'id': row['id'], 'source_cluster_id': row['source_cluster_id'], 'answers': raw['answers']}
        validate_input_rows([row], 1)
        require(ref['id'] == row['id'] and ref['source_cluster_id'] == row['source_cluster_id']
                and isinstance(ref['answers'], list) and ref['answers'], 'reference identity or values invalid')
        return row, ref


def build(plan, plan_sha256, *, plan_source_bytes=None):
    validate_plan(plan)
    builder_sha_at_start = sha(__file__)
    require(plan_source_bytes is None or hashlib.sha256(plan_source_bytes).hexdigest() == plan_sha256,
            'parsed plan and exact source bytes differ')
    project = Path(plan['project_root']).resolve()
    require(os.name != 'nt' and str(project).startswith('/media/') and project.name == 'baseline_matrix_v3'
            and plan['runtime_project_root'] == str(project), 'materialize only on named remote data-disk project')
    for root in plan.get('read_only_source_roots', [str(project)]):
        r = Path(root).resolve()
        require(inside(r, project.parent), 'read-only roots must stay in named data-disk workspace')
        ancestor = r
        while not ancestor.exists():
            ancestor = ancestor.parent
        require(ancestor.stat().st_dev == project.stat().st_dev, 'read-only source root is not on project data device')
    for name, digest in DEPENDENCY_PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, 'frozen catalog dependency changed')
    output, inputs = output_path(plan['output_root'], project), output_path(plan['inputs_root'], project)
    require(not output.exists() and not inputs.exists(), 'fresh output and input directories required; use a new run')
    for name in ('images_root', 'cache_root'):
        output_path(plan[name], project)
    decision = decision_identity(plan)
    output.mkdir(parents=True)
    inputs.mkdir(parents=True)
    cache = Path(plan['cache_root'])
    for env in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME',
                'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PYTHONPYCACHEPREFIX'):
        p = output_path(cache / env.lower(), project)
        p.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(p)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    freeze_json(output / 'FROZEN_PLAN.json', plan)
    if plan_source_bytes is not None:
        freeze(output / 'FROZEN_PLAN_EXACT.json', plan_source_bytes)
    freeze(output / 'BUILDER_EXECUTED_SOURCE.py', Path(__file__).read_bytes())
    freeze_json(output / 'BUILDER_IDENTITY.json', {'builder_sha256': builder_sha_at_start,
        'dependency_pins': DEPENDENCY_PINS, 'plan_sha256': plan_sha256, 'decision': decision,
        'model_GPU_calls': 0, 'downloads_performed': 0})
    blocked, occupied = Identities(), Identities()
    coverage = []
    for item in plan['exclusions']:
        p = pinned(item, plan)
        values = exclusion_rows(p, item)
        before = blocked.partial_records
        for row in values:
            blocked.add(row)
        require('count' not in item or len(values) == item['count'], 'exclusion count changed')
        coverage.append({'path': str(p), 'sha256': sha(p), 'count': len(values), 'roles': item['roles'],
                         'partial_identity_records': blocked.partial_records - before})
    original_history_partial_records = blocked.partial_records
    catalogs, sources, source_proof = defaultdict(list), {}, []
    for pool in plan.get('canonical_pools', []):
        p, refs = pinned(pool['inputs'], plan), pinned(pool['refs'], plan)
        if pool['dataset'] in plan['counts']:
            catalogs[pool['dataset']].extend(metadata_catalog(p, pool))
        source_proof.append({'kind': 'canonical', 'dataset': pool['dataset'], 'inputs_sha256': sha(p),
                             'refs_sha256_verified_without_reading_reference_text': sha(refs)})
    import round10_data_builder_v1 as inherited
    offsets = defaultdict(int)
    for spec in plan.get('parquet_sources', []):
        if spec['dataset'] not in plan['counts'] or spec.get('calibration_only'):
            continue
        item = {'path': spec.get('path') or str(Path(plan['sources_root']) / spec['dataset'] / Path(spec['upstream_path']).name),
                'sha256': spec['sha256'], 'bytes': spec['bytes']}
        p = pinned(item, plan)
        require(p.name not in sources, 'ambiguous parquet basename')
        sources[p.name] = p
        key = (spec['dataset'], spec['split'])
        effective = dict(spec, original_offset=spec.get('original_offset', offsets[key]))
        values = inherited.parquet_catalog(p, effective)
        offsets[key] += len(values)
        catalogs[spec['dataset']].extend(values)
        source_proof.append(dict(effective, actual_path=str(p), verified_bytes=True, catalog_rows=len(values)))
    recovered = 0
    for values in catalogs.values():
        for candidate in values:
            if candidate['id'] in blocked.ids or (candidate['dataset'], candidate['question_id']) in blocked.questions:
                before = len(blocked.clusters)
                blocked.add(candidate)
                recovered += len(blocked.clusters) > before
    materializer = Materializer(plan, sources, project)
    selected, rejections, pool_counts = {}, [], {}
    for dataset, count in plan['counts'].items():
        chosen = []
        ordered = list(ordered_sources(catalogs[dataset], dataset, plan['seed']))
        pool_counts[dataset] = {'QA_rows': len(catalogs[dataset]), 'source_or_document_clusters': len(ordered), 'requested': count}
        for candidate in ordered:
            if blocked.reasons(candidate) or occupied.reasons(candidate):
                continue
            row, body = materializer.inspect_image(candidate)
            why = blocked.reasons(row) + occupied.reasons(row)
            if why:
                rejections.append({'id': row['id'], 'source_cluster_id': row['source_cluster_id'],
                                   'image_sha256': row['original_image_sha256'], 'reasons': why})
                continue
            materializer.commit_image(row, body)
            occupied.add(row)
            chosen.append((candidate, row))
            if len(chosen) == count:
                break
        selected[dataset] = chosen
        pool_counts[dataset]['selected'] = len(chosen)
        if len(chosen) != count:
            freeze_json(output / 'INSUFFICIENT_SOURCE_POOL.json', {'status': 'INSUFFICIENT_DISJOINT_POOL',
                'dataset': dataset, 'requested': count, 'selected': len(chosen), 'pool_counts': pool_counts,
                'reference_text_not_opened': True, 'exclusions_not_relaxed': True})
            raise ValueError('insufficient disjoint pool; do not relax exclusions or replace run')
    locators = [{**{k: row[k] for k in ('dataset', 'id', 'source_id', 'source_cluster_id', 'original_image_sha256')},
                 'locator': {k: v for k, v in c.items() if k not in ('input_metadata', 'inputs_file', 'refs_file')}}
                for dataset in plan['counts'] for c, row in selected[dataset]]
    selected_source = freeze_rows(output / 'SELECTED_SOURCE_LOCATORS.jsonl', locators)
    selection = freeze_json(output / 'SELECTION_FROZEN_BEFORE_REFERENCES.json', {
        'selected_sources': sum(plan['counts'].values()), 'counts': plan['counts'], 'seed': plan['seed'],
        'decision_sha256': decision['sha256'], 'source_locators': selected_source,
        'reference_text_not_opened': True, 'question_text_not_used_for_ordering': True,
        'canonical_catalog_question_strings_decoded_then_discarded': True,
        'selection_reads_predictions_or_correctness': False})
    finished = {ds: [materializer.finish(c, row) for c, row in values] for ds, values in selected.items()}
    all_rows = [row for ds in plan['counts'] for row, _ in finished[ds]]
    validate_input_rows(all_rows, sum(plan['counts'].values()))
    require(all(not blocked.reasons(row) for row in all_rows), 'final source/history overlap survived')
    entries, references = {}, {}
    cohorts = {ds + '_confirm' + str(plan['counts'][ds]): values for ds, values in finished.items()}
    for key, values in cohorts.items():
        model_rows, refs = [r for r, _ in values], [r for _, r in values]
        entry = freeze_rows(inputs / (key + '_inputs.jsonl'), model_rows)
        entries[key] = dict(entry, count=len(model_rows), domains=dict(Counter(r['dataset'] for r in model_rows)),
                            role=ROLE, answers_included=False)
        ref = freeze_rows(inputs / (key + '_refs.jsonl'), refs)
        references[key] = dict(ref, count=len(refs), role='CPU_SCORING_ONLY')
    disjoint = freeze_json(output / 'SOURCE_DISJOINT_PROOF.json', {'status': 'PASS_DECLARED_EXCLUSION_IDENTITIES',
        'selected_count': len(all_rows), 'known_exclusion_overlap': 0, 'coverage': coverage,
        'registered_identities_partial_records': original_history_partial_records, 'parent_recoveries_from_catalog_question_ID': recovered,
        'pool_counts': pool_counts, 'SHA_rejections': rejections,
        'historical_coverage_limits': plan['historical_coverage_limits'],
        'no_absolute_history_or_pretraining_unseen_claim': True})
    source_lock = freeze_json(inputs / 'source_lock.json', {'schema': 'trajkv.round12.confirmation_source_lock.v1',
        'files': {Path(e['path']).name: {k: e[k] for k in ('sha256', 'bytes')}
                  for e in [*entries.values(), *references.values()]}})
    manifest = {'schema': SCHEMA, 'role': ROLE, 'status': 'FROZEN_NEW_SOURCE_INPUTS_NO_MODEL_INFERENCE',
        'input_files': entries, 'answers_in_encoder_inputs': False, 'counts': plan['counts'],
        'total_count': len(all_rows), 'seed': plan['seed'], 'decision': decision,
        'plan_sha256': plan_sha256, 'builder_source_sha256': builder_sha_at_start, 'dependency_pins': DEPENDENCY_PINS,
        'selection_before_references': selection, 'disjoint_proof': disjoint, 'source_lock': source_lock,
        'native_hashes_borrowed_from_old_predictions': False, 'fresh_native_hashes_required': True,
        'fresh_inference_required_for': decision['comparators'] + decision['selected_methods'],
        'old_prediction_reuse_admitted': False, 'historical_coverage_limits': plan['historical_coverage_limits'],
        'materialization_complete_marker': 'MATERIALIZATION_COMPLETE_v1.json',
        'model_or_GPU_calls': 0, 'downloads_performed': 0}
    mf = freeze_json(inputs / 'ROUND12_CONFIRMATION_INPUT_MANIFEST_v1.json', manifest)
    freeze_json(inputs / 'ROUND12_CONFIRMATION_SCORING_MANIFEST_v1.json', {
        'schema': 'trajkv.round12.confirmation_refs.v1', 'input_manifest_sha256': mf['sha256'],
        'reference_files': references, 'role': 'CPU_SCORING_ONLY_NOT_ENCODER_INPUT'})
    freeze_json(output / 'RAW_SOURCE_PROOF.json', {'sources': source_proof, 'all_declared_bytes_checked': True, 'downloads': 0})
    freeze_json(output / 'IMAGE_MATERIALIZATION.json', {'images': list(materializer.images.values()), 'original_bytes_preserved': True})
    # End-to-end source stability, including files opened again after selection.
    for item in plan['exclusions']:
        pinned(item, plan)
    for pool in plan.get('canonical_pools', []):
        pinned(pool['inputs'], plan)
        pinned(pool['refs'], plan)
    for spec in plan.get('parquet_sources', []):
        if spec['dataset'] in plan['counts'] and not spec.get('calibration_only'):
            pinned({'path': spec.get('path') or str(Path(plan['sources_root']) / spec['dataset'] / Path(spec['upstream_path']).name),
                    'sha256': spec['sha256'], 'bytes': spec['bytes']}, plan)
    decision_identity(plan)
    require(sha(__file__) == builder_sha_at_start, 'builder source changed during materialization')
    require(all(sha(Path(__file__).with_name(name)) == digest for name, digest in DEPENDENCY_PINS.items()),
            'frozen dependency changed during materialization')
    require(all(sha(r['image_path']) == r['original_image_sha256'] for r in all_rows), 'materialized image bytes changed')
    result = {'status': 'PASS_FROZEN_ROUND12_NEW_CONFIRMATION_DATA_ONLY', 'manifest': mf,
              'counts': plan['counts'], 'total': len(all_rows), 'model_or_GPU_calls': 0, 'jobs_launched': 0}
    freeze_json(output / 'COMPLETE.json', result)
    freeze_json(inputs / 'MATERIALIZATION_COMPLETE_v1.json', result)
    return result


def load_confirmation_inputs(path, manifest_path, expected_sha256=None, expected_role=ROLE):
    """Fresh native-run loader; it opens no references and has no old-logit map."""
    require(expected_role == ROLE, 'wrong confirmation role')
    manifest = read(manifest_path)
    require(manifest.get('schema') == SCHEMA and manifest.get('answers_in_encoder_inputs') is False
            and manifest.get('old_prediction_reuse_admitted') is False
            and manifest.get('fresh_native_hashes_required') is True, 'unknown/faulty fresh-source contract')
    require(manifest.get('materialization_complete_marker') == 'MATERIALIZATION_COMPLETE_v1.json', 'bad completion marker name')
    marker = Path(manifest_path).parent / manifest['materialization_complete_marker']
    require(marker.is_file(), 'materialization not closed')
    done = read(marker)
    require(done.get('status') == 'PASS_FROZEN_ROUND12_NEW_CONFIRMATION_DATA_ONLY'
            and done['manifest']['sha256'] == sha(manifest_path), 'completion marker/manifest mismatch')
    name = Path(path).name
    require(name.endswith('_inputs.jsonl'), 'only no-answer input file')
    key = name[:-len('_inputs.jsonl')]
    entry = manifest['input_files'][key]
    require(entry['sha256'] == sha(path) and entry['bytes'] == Path(path).stat().st_size
            and entry['role'] == ROLE and entry['answers_included'] is False, 'input bytes/binding differ')
    require(expected_sha256 is None or sha(path) == expected_sha256, 'explicit input SHA differs')
    values = validate_input_rows(rows(path), entry['count'])
    require(dict(Counter(r['dataset'] for r in values)) == entry['domains'], 'domain count differs')
    return values


def self_check():
    blocked = Identities()
    blocked.add({'dataset': 'docvqa', 'id': 'docvqa:train:old', 'source_id': 'docA/1',
                 'source_cluster_id': 'docvqa:ucsf:docA', 'original_image_sha256': 'a' * 64})
    require('source_or_document_parent' in blocked.reasons({'dataset': 'docvqa', 'id': 'new',
        'source_id': 'docA/2', 'source_cluster_id': 'docvqa:ucsf:docA'}), 'Doc sibling-page exclusion failed')
    require('original_or_registered_image_SHA' in blocked.reasons({'dataset': 'textvqa',
        'id': 'new', 'source_id': 'new', 'original_image_sha256': 'a' * 64}), 'cross-dataset image SHA exclusion failed')
    fixture = [{'kind': 'parquet', 'dataset': 'infovqa', 'source_cluster_id': c, 'question_id': q}
               for c in ('a', 'b', 'c') for q in ('0', '1')]
    require(list(ordered_sources(fixture, 'infovqa', 'synthetic')) ==
            list(ordered_sources(fixture[::-1], 'infovqa', 'synthetic')), 'ordering depends on catalog enumeration')
    return {'status': 'PASS_SYNTHETIC_METADATA_CONTRACT', 'real_source_selection': False,
            'final_N_chosen': False, 'Doc_parent_blocking': True, 'cross_dataset_SHA_blocking': True,
            'hash_order_permutation_invariant': True, 'model_GPU_or_download_calls': 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path)
    p.add_argument('--build', action='store_true')
    p.add_argument('--validate-plan', action='store_true')
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check:
        require(a.plan is None and not a.build, 'synthetic check cannot select real sources')
        result = self_check()
    else:
        require(a.plan is not None and a.build != a.validate_plan, 'explicit --plan and exactly one of --build/--validate-plan required')
        body = a.plan.read_bytes()
        plan = json.loads(body.decode('utf-8-sig'))
        plan_sha = hashlib.sha256(body).hexdigest()
        validate_plan(plan)
        result = build(plan, plan_sha, plan_source_bytes=body) if a.build else {'status': 'PLAN_STRUCTURE_VALID_ONLY',
            'plan_sha256': plan_sha, 'real_source_selection': False, 'model_GPU_or_download_calls': 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
