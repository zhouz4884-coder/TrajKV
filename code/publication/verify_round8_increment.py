"""Verify derived Round8 archive/content refs and independently recount scores/owners.

No Git/network/model execution. Optional tree JSON comes from the publisher's
actual staged/committed Git tree, rather than a working-directory inventory.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/a2'), str(ROOT / 'code/analysis')]
from analyze_external import score_docvqa
from textvqa_metric import TextVQAAccuracyEvaluator
from prepare_round5_delivery import secret_scan
from round8_content_index import verify as verify_content, safe_name, file_digest
import verify_round7_evidence as r7_evidence
import round8_quality_owner as r8_owner
import round8_postcal_owner as postcal_owner
import analyze_round8_postcal_quality as completed_quality
import numpy as np


def local_path(value):
    return r8_owner.local_path(value)


def calibration(path):
    return r7_evidence.calibration(path)


def postcheck(folder, read_value=None, digest_file=file_digest):
    read_value = read_value or (lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig')))
    value = read_value(folder / 'complete_round8.json')
    require(value.get('status') == 'COMPLETE' and value.get('encoder_inputs_unchanged') is True,
            'R8 postcheck did not complete')
    require(value['engine_complete_sha256'] == digest_file(folder / 'complete.json')
            and value['manifest_sha256'] == digest_file(folder / 'manifest.json'), 'R8 postcheck identity mismatch')
    return value


def final_completion(folder, get=None, entries=None):
    """Apply the same dedicated gates used before scoring, also in an archive.

    Embedded completion digests identify original producer bytes. Public bodies
    can be path-redacted, so archive gates use source_sha256, not public SHA.
    """
    if get is None:
        result = completed_quality.completion_gate(folder)
    else:
        class Member:
            def __init__(self, name):
                self.name = str(name).replace('\\', '/')
            def __truediv__(self, name):
                return Member(self.name + '/' + str(name))
            def __str__(self):
                return self.name
            def is_file(self):
                return self.name in entries
            def exists(self):
                return self.is_file()
        base = completed_quality.base
        old_read, old_sha = base.read, base.sha
        try:
            base.read = lambda path: json.loads(get(str(path)))
            base.sha = lambda path: entries[str(path)]['source_sha256']
            result = completed_quality.completion_gate(Member(folder),
                read_record_rows=lambda path:[json.loads(line) for line in get(str(path)).splitlines() if line.strip()])
        finally:
            base.read, base.sha = old_read, old_sha
    require(result is None or result['status'] == 'PASS',
            'Dedicated final producer marker is pending/invalid: ' + str(folder))
    return result


def local_quality(path):
    """Frozen official scoring plus the R8 owner oracle and producer postcheck."""
    path = path / 'summary.json' if path.is_dir() else path
    summary = load(path)
    variants = {}
    for item in summary['audit']['runs']:
        folder = local_path(item['run']); final_completion(folder); postcheck(folder)
        for variant in load(folder / 'manifest.json')['config']['variants']:
            name = variant['name']
            require(name not in variants or variants[name] == variant, 'Conflicting variant settings')
            variants[name] = variant
    if summary['dataset'] == 'infovqa':
        return local_infovqa_quality(path, summary)
    previous_owner, previous_path = r7_evidence.actual_owner, r7_evidence.local_path
    def owner(record, external):
        value = postcal_owner.audit_owner(record, variants[record['variant']], external, {})
        return dict(complete=value['complete_visual_owner_bytes'], denominator=value['visual_fp16_denominator'],
            metric_bytes=value.get('key_metric_bytes', 0), physical_tail_bytes=value.get('physical_tail_storage_bytes', 0))
    try:
        r7_evidence.actual_owner, r7_evidence.local_path = owner, local_path
        result = r7_evidence.quality(path)
    finally:
        r7_evidence.actual_owner, r7_evidence.local_path = previous_owner, previous_path
    result['owner_oracle'] = 'round8_postcal_owner.audit_owner delegating preserved families to round8_quality_owner.audit_owner'
    result['complete_round8_postchecks_verified'] = True
    result['dedicated_final_markers_verified_before_reference_access'] = True
    return result


def local_infovqa_quality(path, summary):
    """Recount InfoVQA as its own ANLS domain without altering frozen R7 code.

    The preserved R7 local verifier accepts only doc/text. Replay the same
    publication member verifier for InfoVQA, using original local bytes as
    sealed members. No domain alias or summary rewrite is performed.
    """
    entries, members = {}, {}
    def add(member):
        member = Path(member).resolve()
        require(member.is_relative_to(ROOT), 'InfoVQA dependency escaped the project')
        name = member.relative_to(ROOT).as_posix()
        require(member.is_file(), 'Missing InfoVQA evidence member: ' + name)
        digest = file_digest(member)
        entries[name] = dict(source_sha256=digest, public_sha256=digest)
        members[name] = member
        return name
    sample_paths = [member for member in (ROOT / 'configs/a2').rglob('*.json')
                    if file_digest(member) == summary['samples_sha256']]
    require(sample_paths, 'Missing original InfoVQA sample fingerprint')
    samples = add(sample_paths[0])
    references = add(r7_evidence.reference_file(summary['references_sha256']))
    raw_records = []
    for receipt in summary['audit']['runs']:
        folder = local_path(receipt['run'])
        require(file_digest(folder / 'manifest.json') == receipt['manifest_sha256']
            and file_digest(folder / 'records.jsonl') == receipt['records_sha256'],
            'InfoVQA audited producer identities changed')
        for member in folder.glob('*.json'):
            add(member)
        raw_records.append(add(folder / 'records.jsonl'))
        for name, expected in load(folder / 'manifest.json').get('external_codec_inputs', {}).items():
            member_name = add(local_path(name))
            require(entries[member_name]['source_sha256'] == expected, 'InfoVQA external dependency changed')
    add(ROOT / 'code/a2/round8_postcal_fields.py')
    spec = dict(dataset='infovqa', analysis_role=summary['analysis_role'], summary=add(path),
        scores=add(path.parent / 'scores.csv'), samples=samples, references=references,
        raw_records=raw_records, selected_variants=[row['variant'] for row in summary['table']])
    checked = quality_check(spec, lambda name: members[str(name).replace('\\', '/')].read_bytes(), entries)
    checked.update(status='PASS', summary=str(path), summary_sha256=file_digest(path),
        scores_sha256=file_digest(path.parent / 'scores.csv'),
        owner_oracle='round8_postcal_owner.audit_owner delegating preserved families to round8_quality_owner.audit_owner',
        complete_round8_postchecks_verified=True, dedicated_final_markers_verified_before_reference_access=True,
        domain_scope='InfoVQA remains a separate ANLS domain; no DocVQA aggregation or frozen verifier mutation')
    return checked


def verify_books(manifests, get_json, original_sha):
    expected = {(asset, mode) for asset in ('k_global4_16', 'k_global8_256', 'k_position4_16', 'v_global8_256')
                for mode in ('decoder-only', 'reencode')}
    found, inputs = set(), set()
    for filename in manifests:
        value = get_json(filename)
        require(value.get('status') == 'COMPLETE', 'Incomplete fitted-book bank')
        for item in value['books']:
            key = item['asset'], item['mode']; require(key not in found, 'Duplicate fitted endpoint')
            found.add(key)
            name = str(Path(filename).parent / item['file']).replace('\\', '/')
            require(original_sha(name) == item['sha256'], 'Original fitted-book bytes changed')
            book = get_json(name)
            centers = np.asarray(book['centers'], dtype='<f2')
            require(list(centers.shape) == item['shape'] and np.isfinite(centers).all()
                and centers.nbytes == item['actual_book_bytes'] == book['actual_book_bytes'], 'Fitted-book shape/bytes mismatch')
            require(hashlib.sha256(centers.tobytes()).hexdigest() == item['centers_fp16_le_sha256']
                == book['centers_fp16_le_sha256'], 'Exact FP16 center identity mismatch')
            require(not set(book['training_source_ids']) & set(book['select_source_ids']), 'Fitted-book source leakage')
            for field in ('source_split_sha256', 'dump_manifest_sha256', 'old_encoder_book_sha256'):
                inputs.add((field, book[field]))
    require(found == expected, 'The eight declared K4/K8/K4pos/V8 endpoints are not complete')
    return dict(status='PASS', endpoints=8, source_input_fingerprints=[dict(field=x, sha256=y) for x,y in sorted(inputs)])


def verify_runtime_collections(index, get):
    """Recheck original alias/inventory to derived public object closure."""
    results = []
    for item in index.get('runtime_collection_derivations',index.get('runtime_collections',[])):
        mapping = json.loads(get(item['public_mapping']))
        require(mapping['source_runtime_index_sha256'] == item['source_index_sha256'], 'Runtime derivation bound to another original index')
        blobs = {row['source_sha256']:row['content_path'] for row in mapping['public_source_objects']}
        excluded = {row['source_sha256'] for row in mapping['private_source_exclusions']}
        require(not set(blobs)&excluded and len(blobs)+len(excluded) == mapping['original_objects'] == item['original_objects'],
                'Runtime object/exclusion membership changed')
        for original,name in blobs.items():
            require(name in index['files'] and index['files'][name]['source_sha256'] == original,
                    'Runtime object source/public mapping mismatch')
        inventory_count = manifest_count = 0
        for job in mapping['jobs']:
            name = job['public_inventory']; inventory = json.loads(get(name))
            require(index['files'][name]['source_sha256'] == job['source_inventory_sha256'], 'Original inventory fingerprint changed')
            for key,expected in inventory.items():
                require(job['full_source_objects'].get(key) == expected and expected in set(blobs)|excluded,
                        'Public snapshot inventory-to-source-object gap')
                inventory_count += 1
            for expected in job.get('narrow_source_objects',{}).values():
                require(expected in set(blobs)|excluded,'Public manifest-to-source-object gap'); manifest_count += 1
        for metadata in mapping['captured_metadata']:
            require(index['files'][metadata['public_content_path']]['source_sha256'] == metadata['source_sha256'],
                    'Captured receipt/log metadata source fingerprint differs')
        require(inventory_count == mapping['inventory_mappings'] == item['inventory_mappings']
                and manifest_count == mapping['manifest_mappings'] == item['manifest_mappings'], 'Runtime mapping count changed')
        results.append(dict(status='PASS',source_runtime_index_sha256=item['source_index_sha256'],jobs=len(mapping['jobs']),
            original_objects=mapping['original_objects'],public_objects=len(blobs),private_exclusions=len(excluded),
            inventory_mappings=inventory_count,manifest_mappings=manifest_count,
            scope='source/receipt tracking closure, not executed-module or task-quality proof'))
    return results

OWNER = ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position', 'hadamard_constant',
         'axis_packed_reference', 'calib_packed_reference', 'hybrid_packed_reference', 'kvarn_packed_reference',
         'rotation_packed_reference', 'anchor_visual_auxiliary', 'spectrum_experiment_reference')


def require(value, message):
    if not value:
        raise ValueError(message)


def close(left, right, context):
    require(math.isfinite(float(left)) and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-10), context)


def rel(value):
    text = str(value).replace('\\', '/')
    return text.split('/TrajKV/', 1)[1] if '/TrajKV/' in text else text


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def recount_owner(record, manifest, get, entries, artifact_aliases=None):
    """Reuse the R8 scalar owner oracle on exact, redacted archive bodies.

    Original hashes come from the source-public mapping, never from hashing a
    redacted body against an original expected artifact SHA.
    """
    artifact_aliases = artifact_aliases or {}
    def canonical_name(value):
        name = rel(value)
        return artifact_aliases.get(name, name)
    class Artifact:
        def __init__(self, value):
            self.name = canonical_name(value)
        def __truediv__(self, value):
            return Artifact(str(PurePosixPath(self.name) / str(value)))
        def __str__(self):
            return self.name
        @property
        def parent(self):
            return Artifact(str(PurePosixPath(self.name).parent))
        def resolve(self):
            # Archive names have already passed safe_name. Never resolve an
            # archive dependency against the machine's working filesystem.
            safe_name(self.name)
            return self
        def is_relative_to(self, other):
            return PurePosixPath(self.name).is_relative_to(PurePosixPath(str(other)))
        def is_file(self):
            return self.name in entries
    def locate(value):
        return Artifact(value)
    def original_hash(value):
        value = value if isinstance(value, Artifact) else locate(value)
        require(value.name in entries, 'Unindexed owner dependency: ' + value.name)
        require(hashlib.sha256(get(value.name)).hexdigest() == entries[value.name]['public_sha256'],
                'Missing/changed exact archived owner dependency: ' + value.name)
        return entries[value.name]['source_sha256']
    def read_artifact(value):
        require(isinstance(value, Artifact), 'Owner attempted an unindexed filesystem read')
        return json.loads(get(value.name))
    variant = next(x for x in manifest['config']['variants'] if x['name'] == record['variant'])
    external = manifest.get('external_codec_inputs', {})
    for value, expected in external.items():
        name = canonical_name(value)
        require(name in entries and entries[name]['source_sha256'] == expected, 'Original external artifact mapping mismatch')
    old = r8_owner.local_path, r8_owner.r7.local_path, r8_owner.r7.sha, r8_owner.r7.read
    old_postcal = postcal_owner.sha, postcal_owner.read, postcal_owner.local_path
    try:
        r8_owner.local_path = r8_owner.r7.local_path = locate
        r8_owner.r7.sha, r8_owner.r7.read = original_hash, read_artifact
        postcal_owner.sha, postcal_owner.read, postcal_owner.local_path = original_hash, read_artifact, locate
        result = postcal_owner.audit_owner(record, variant, external, {})
    finally:
        r8_owner.local_path, r8_owner.r7.local_path, r8_owner.r7.sha, r8_owner.r7.read = old
        postcal_owner.sha, postcal_owner.read, postcal_owner.local_path = old_postcal
    return result['complete_visual_owner_bytes'], result['visual_fp16_denominator']


def quality_check(spec, get, entries, artifact_aliases=None):
    summary = json.loads(get(spec['summary']))
    require(summary['status'] == 'COMPLETE' and summary['audit']['status'] == 'PASS', 'Incomplete quality summary')
    require(spec['dataset'] in ('doc', 'text', 'infovqa') and summary['dataset'] == spec['dataset'],
            'Unknown/mismatched publication metric domain')
    # Gate every producer before opening any reference content or raw answers.
    final_markers = []
    for filename in spec['raw_records']:
        folder = str(Path(filename).parent).replace('\\', '/')
        final = final_completion(folder, get, entries)
        if final:
            final_markers.append(final)
        postcheck(Path(folder), lambda path: json.loads(get(str(path).replace('\\', '/'))),
                  lambda path: entries[str(path).replace('\\', '/')]['source_sha256'])
    require(entries[spec['samples']]['source_sha256'] == summary['samples_sha256']
            and entries[spec['references']]['source_sha256'] == summary['references_sha256'],
            'Original source/answer-table fingerprints differ from the quality analysis')
    text = get(spec['references']).decode('utf-8-sig')
    references = (json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()])
    refs = {row['id']: row for row in references}
    scores = list(csv.DictReader(io.StringIO(get(spec['scores']).decode('utf-8-sig'))))
    metric, scored, groups = TextVQAAccuracyEvaluator(), {}, defaultdict(list)
    for row in scores:
        key = (row['session_id'], row['variant'], int(row['turn']))
        require(key not in scored and row['variant'] in spec['selected_variants'], 'Duplicate/unselected scored answer')
        target = (refs[key[0]]['answers'] if spec['dataset'] == 'text' else
                  next(value.get('answer', value.get('answers')) for value in refs[key[0]]['references'] if int(value['turn']) == key[2]))
        require(json.loads(row['reference']) == target, 'Scored references differ from original selected labels')
        actual = (metric.eval_pred_list([dict(pred_answer=row['prediction'], gt_answers=target)]) if spec['dataset'] == 'text'
                  else score_docvqa(row['prediction'], target)['score'])
        close(row['score'], actual, 'Official score recomputation differs')
        groups[row['variant']].append(float(actual)); scored[key] = row
    stores, denominators, raw_keys, original_records = defaultdict(int), defaultdict(int), set(), 0
    for filename in spec['raw_records']:
        folder = str(Path(filename).parent).replace('\\', '/')
        manifest = json.loads(get(folder + '/manifest.json'))
        require('cached-prefix' not in str(manifest.get('quality_scope', '')).lower(), 'Cached-prefix calibration mixed into quality')
        require(json.loads(get(folder + '/complete.json'))['status'] == 'COMPLETE', 'Incomplete producer')
        post = json.loads(get(folder + '/complete_round8.json'))
        require(post.get('status') == 'COMPLETE' and post.get('encoder_inputs_unchanged') is True
            and post['engine_complete_sha256'] == entries[folder + '/complete.json']['source_sha256']
            and post['manifest_sha256'] == entries[folder + '/manifest.json']['source_sha256'], 'R8 producer postcheck is absent/changed')
        for line in get(filename).splitlines():
            if not line.strip():
                continue
            record = json.loads(line); original_records += 1
            require('free_running' in record.get('protocol', '') and record.get('task_accuracy_eligible') is not False
                    and record.get('primary_quality_eligible') is not False,
                    'Non-free diagnostic mixed into quality run')
            amount, denominator = recount_owner(record, manifest, get, entries, artifact_aliases)
            if record['variant'] not in spec['selected_variants']:
                continue
            stores[record['variant']] += amount; denominators[record['variant']] += denominator
            for answer in record['turns']:
                key = (record['session_id'], record['variant'], int(answer['turn']))
                require(key in scored and key not in raw_keys and answer['answer'] == scored[key]['prediction'], 'Raw/scored answer coverage or identity mismatch')
                raw_keys.add(key)
    require(raw_keys == set(scored), 'Raw/scored question union differs')
    for row in summary['table']:
        name = row['variant']
        close(statistics.mean(groups[name]), row.get('score', row.get('anls') if spec['dataset'] in ('doc', 'infovqa') else row.get('accuracy')), 'Quality mean mismatch')
        close(16 * stores[name] / denominators[name], row['budget_bits_including_encoder_constants'], 'Aggregate complete owner mismatch')
    return dict(dataset=spec['dataset'], analysis_role=spec['analysis_role'], variants=len(groups),
        scored_answers=len(scored), original_producer_records_verified=original_records, score_and_owner_recomputed=True,
        dedicated_final_markers=final_markers, final_markers_verified_before_reference_access=True)


def verify(assets, tree_map=None):
    delivery = load(assets / 'DELIVERY.json')
    archive_path = assets / delivery['archive']; index_path = assets / 'ROUND8_SOURCE_INDEX.json'
    content_path = assets / 'ROUND8_CONTENT_REFS.json'; index = load(index_path)
    require(index['schema'] == 'trajkv.round8.public.increment.v1' and index['status'] == 'PASS', 'Wrong/incomplete R8 source index')
    require(file_digest(archive_path) == delivery['archive_sha256'] and file_digest(index_path) == delivery['source_index_sha256']
            and file_digest(content_path) == delivery['content_index_sha256'], 'Prepared delivery identity changed')
    content = verify_content(SimpleNamespace(verify_index=content_path, archive=archive_path,
        payload=assets / 'payload', git_tree_map=tree_map))
    source_verified, derived_verified = 0, 0
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP members')
        require(set(archive.namelist()) == set(index['files']) | {'ROUND8_SOURCE_INDEX.json'}, 'Archive/index inventory mismatch')
        require(archive.read('ROUND8_SOURCE_INDEX.json') == index_path.read_bytes(), 'Inner/outer source index differs')
        def get(name):
            name = rel(name)
            require(name in index['files'], 'Unindexed evidence dependency: ' + name)
            return archive.read(name)
        for name, entry in index['files'].items():
            safe_name(name); data = get(name); secret_scan(data, name)
            require(hashlib.sha256(data).hexdigest() == entry['public_sha256'] and len(data) == entry['bytes'], 'Public exact content changed')
            source = ROOT / entry.get('source_path', name)
            if source.is_file():
                actual = file_digest(source)
                if actual == entry['source_sha256']:
                    source_verified += 1
                elif actual in (entry['public_sha256'], load(content_path)['files'][name]['git_normalized_body_sha256']):
                    derived_verified += 1
                else:
                    raise ValueError('Existing source/public materialization differs from all pinned identities: ' + name)
        aliases = {}
        alias_path = index['publication_plan'].get('artifact_map')
        if alias_path:
            alias_path = rel(alias_path)
            require(alias_path in index['files'], 'Publication artifact map is absent from the archive')
            expected = index['publication_plan'].get('artifact_map_sha256')
            require(not expected or index['files'][alias_path]['source_sha256'] == expected,
                    'Original publication artifact-map fingerprint differs')
            mapping = json.loads(get(alias_path))
            require(not mapping.get('prefixes'), 'Archive aliases require explicit file identities')
            for original, target in mapping.get('files', {}).items():
                original, target = rel(original), rel(target)
                safe_name(target)
                require(target in index['files'] and (original not in aliases or aliases[original] == target),
                        'Missing/ambiguous explicit archived artifact alias')
                aliases[original] = target
        quality = [quality_check(spec, get, index['files'], aliases) for spec in index['checks']['quality_analyses']]
        books = verify_books(index['checks']['book_manifests'], lambda name: json.loads(get(name)),
                             lambda name: index['files'][name]['source_sha256'])
        fingerprints = {entry['source_sha256'] for entry in index['files'].values()}
        require(all(item['sha256'] in fingerprints for item in books['source_input_fingerprints']),
                'Fitted-book split/dump/reference input fingerprint is missing from the archive')
        source_blobs = {item['source_sha256'] for item in index.get('runtime_source_blobs', [])}
        require(set(index.get('quality_manifest_source_hashes', [])) <= source_blobs,
                'Original quality source inventory lacks its content objects')
        runtime = verify_runtime_collections(index,get)
    return dict(status='PASS', archive_sha256=file_digest(archive_path), source_index_sha256=file_digest(index_path),
        content_index_sha256=file_digest(content_path), indexed_files=len(index['files']),
        original_source_files_verified=source_verified, derived_public_files_verified=derived_verified,
        quality_recomputations=quality, fitted_books=books, runtime_source_tracking=runtime,
        explicit_artifact_aliases_verified=aliases,
        R8_owner_and_postcheck_verified=True, content_identity_check=content,
        cached_prefix_calibration_not_E2E=True, failed_smokes_excluded_from_quality=True,
        credential_findings=0, Git_operations=0, network_operations=0, GPU_used=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--git-tree-map', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); result = verify(a.assets.resolve(), a.git_tree_map)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(status=result['status'], indexed_files=result['indexed_files'],
        quality_analyses=len(result['quality_recomputations']), actual_git_tree_verified=result['content_identity_check']['actual_git_tree_verified'])))
