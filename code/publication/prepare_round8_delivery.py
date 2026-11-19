"""Prepare a plan-driven Round8 public increment locally, without Git/network writes."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from datetime import datetime, timezone
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/a2'), str(ROOT / 'code/analysis')]
from prepare_public_snapshot import Exporter
from prepare_round5_delivery import secret_scan, zip_info
from verify_round8_increment import local_quality as quality, calibration, local_path
from round8_content_index import build as build_content_index, safe_name

SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.svg', '.png', '.pdf', '.log'}
PRIVATE = {'chat.md', 'cha.md', 'AGENTS.md', 'remote.py', 'check_github_access.py'}
PUBLICATION_TOOLS = {'prepare_round8_delivery.py', 'publish_round8_update.py', 'verify_round8_increment.py',
    'round8_content_index.py', 'prepare_public_snapshot.py', 'prepare_round5_delivery.py'}


class Round8Exporter(Exporter):
    """Also redact bare mount roots and known routing paths inside metadata text.

    Human answer/question content retains ordinary IP/numeric strings. Only
    exact known private project/mount roots are removed in protected fields.
    """
    def __init__(self, destination):
        super().__init__(destination)
        extra = []
        for original, _ in self.roots:
            if original.startswith('/media/') and '/VLAtest/TrajKV' in original:
                extra.append((original.split('/VLAtest/TrajKV', 1)[0], '/path/to/storage'))
        for spelling in {str(ROOT.parent), ROOT.parent.as_posix(), str(ROOT.parent).replace('\\', '\\\\')}:
            extra.append((spelling, '/path/to/research'))
        self.roots = sorted(set(self.roots + extra), key=lambda pair: -len(pair[0]))

    def clean_string(self, text, protect=False):
        text = super().clean_string(text, protect)
        if protect:
            for original, target in self.roots:
                text = text.replace(original, target)
        return text


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def allowed(path):
    path = Path(path)
    parts = path.relative_to(ROOT).parts
    if 'maintenance_20261118' in parts:
        return False
    if parts[0] == 'outputs' and 'split_root_fixture' in parts:
        # These are deliberately invalid text payloads with .py names for
        # wrong-counterpart hash tests. Keep the CPU check receipts and the
        # real instrument source, rather than publishing them as Python code.
        return False
    if set(parts) & {'.git', '__pycache__', 'images', 'models', 'weights', '.ssh', 'private', 'chats', 'conversations'}:
        return False
    name = path.name
    if name in PRIVATE or re.search(r'^chat|^DS(?:[_. -]|\d|$)|credential|auth[_-]?helper|connection', name, re.I):
        return False
    if name.startswith('EXECUTION_CHECKPOINT') or path.suffix.lower() not in SUFFIXES:
        return False
    if tuple(parts[:2]) == ('code', 'publication') and name not in PUBLICATION_TOOLS:
        return False
    if path.suffix.lower() in {'.png', '.pdf'} and parts[0] != 'figures':
        return False
    return True


def find_fingerprint(folder, expected):
    matches = [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl') and sha(p) == expected]
    require(matches, 'Missing original frozen fingerprint: ' + expected)
    return matches[0]


def runtime_collection(source_root):
    """Validate the frozen collect_round8_runtime_sources.py object format."""
    source_root = Path(source_root).resolve()
    index_path = source_root if source_root.is_file() else source_root/'runtime_index.json'
    source_root = index_path.parent
    index = load(index_path)
    receipt_path = (source_root/'COLLECTION.json' if index_path.name == 'runtime_index.json'
                    else source_root/('COLLECTION_'+index['capture']+'.json'))
    receipt = load(receipt_path)
    require(index.get('schema') == 'trajkv.round8.runtime-source-index.v1'
        and receipt.get('status') == 'PASS' and not index.get('source_gaps'), 'Incomplete/wrong runtime collection')
    require(receipt['runtime_index_sha256'] == sha(index_path), 'Frozen runtime index SHA changed')
    require(receipt.get('all_local_manifest_python_hashes_covered') is True
        and all(not item['missing'] for item in index.get('local_quality_manifest_coverage',[])), 'Runtime manifest-source gaps')
    all_objects = {path.stem:path for path in (source_root/'objects').glob('*.py')}
    pinned = {expected for job in index['jobs'] for field in ('full_source_objects','narrow_source_objects')
              for expected in job.get(field,{}).values()}
    # Future immutable captures may add objects to the shared directory. Pin
    # this capture's referenced set rather than absorbing later source bytes.
    require(len(pinned) == index['objects'] == receipt['objects'] and pinned <= set(all_objects),
            'Frozen capture object membership is incomplete/ambiguous')
    objects = {expected:all_objects[expected] for expected in pinned}
    for expected,path in objects.items():
        require(re.fullmatch(r'[0-9a-f]{64}',expected) and sha(path) == expected, 'Original runtime object SHA changed')
    originals,origin_names = {},{}
    inventory_mappings = manifest_mappings = 0
    for job in index['jobs']:
        relative_name = job['original_inventory']; safe_name(relative_name)
        path = source_root/relative_name
        require(sha(path) == job['source_inventory_sha256'], 'Original snapshot inventory changed')
        inventory = load(path)
        require(len(inventory) == job['inventory_entries'], 'Original inventory entry count changed')
        for name,expected in inventory.items():
            require(job['full_source_objects'].get(name) == expected and expected in objects,
                    'Snapshot inventory-to-object gap')
            origin_names.setdefault(expected,set()).add(name); inventory_mappings += 1
        for name,expected in job.get('full_source_objects',{}).items():
            require(expected in objects, 'Full snapshot object missing')
            origin_names.setdefault(expected,set()).add(name)
        for name,expected in job.get('narrow_source_objects',{}).items():
            require(expected in objects, 'Manifest-to-original-source gap')
            origin_names.setdefault(expected,set()).add(name); manifest_mappings += 1
    for item in index['files']:
        name = item['local_relative']; safe_name(name)
        path = source_root/name
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Captured runtime file changed')
        originals[name] = item
    return dict(root=source_root,index_path=index_path,receipt_path=receipt_path,
        index=index,receipt=receipt,objects=objects,originals=originals,
        origin_names=origin_names,inventory_mappings=inventory_mappings,manifest_mappings=manifest_mappings,
        source_index_sha256=sha(index_path),collection_sha256=sha(receipt_path))


def append_runtime_collection(value, exporter, archive, entries, collection_id):
    """Derive only public bodies; source aliases and exclusions stay explicit."""
    root,index = value['root'],value['index']
    prefix = 'runtime_sources/collections/'+value['source_index_sha256']
    metadata,blobs,inventories,excluded = [],[],[],[]
    def add(path,name,role):
        if name in entries:
            return
        data = exporter.transform(path,Path(name)); secret_scan(data,name)
        archive.writestr(zip_info(name),data)
        entries[name] = dict(source_path=relative(path),source_sha256=sha(path),source_bytes=path.stat().st_size,
            public_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),browsable=False,role=role)
    for path,name in ((value['index_path'],'runtime_index.json'),(value['receipt_path'],'COLLECTION.json')):
        add(path,prefix+'/'+name,'frozen source collection metadata; no quality eligibility inferred')
    for name,item in sorted(value['originals'].items()):
        if name.startswith('objects/'):
            continue
        public = prefix+'/'+name
        add(root/name,public,'original captured '+item.get('category','metadata')+'; tracking only, not task quality')
        metadata.append(dict(original_relative=name,source_sha256=item['sha256'],public_content_path=public))
        if name.endswith('/sha256.json'):
            inventories.append(dict(source_sha256=item['sha256'],content_path=public,entries=len(load(root/name))))
    for expected,path in sorted(value['objects'].items()):
        aliases = sorted(value['origin_names'].get(expected,[]))
        private_name = any(Path(name).name in PRIVATE
            or re.search(r'credential|auth[_-]?helper|connection',Path(name).name,re.I) for name in aliases)
        text = path.read_text(encoding='utf-8-sig',errors='replace')
        private_body = any(marker in text for marker in ('contractkv_ssh_credential','win32cred','CredReadW','CredRead('))
        if private_name or private_body:
            excluded.append(dict(source_sha256=expected,reason='private authentication/routing helper; original fingerprint retained only'))
            continue
        public = 'runtime_sources/'+expected+'.py'
        add(path,public,'original runtime source object; path-redacted public body, not proof of module execution')
        blobs.append(dict(source_sha256=expected,content_path=public))
    mapping = dict(schema='trajkv.round8.runtime-public-content-mapping.v1',source_runtime_index_sha256=value['source_index_sha256'],
        original_objects=len(value['objects']),public_source_objects=blobs,private_source_exclusions=excluded,
        captured_metadata=metadata,jobs=[dict(job=job['job'],running_at_capture=job['running_at_capture'],
            source_inventory_sha256=job['source_inventory_sha256'],
            public_inventory=prefix+'/'+job['original_inventory'],
            full_source_objects=job['full_source_objects'],narrow_source_objects=job.get('narrow_source_objects',{}),
            completion_markers=job.get('completion_markers',{}),scope='available source/receipt mapping only; not quality evidence') for job in index['jobs']],
        inventory_mappings=value['inventory_mappings'],manifest_mappings=value['manifest_mappings'],
        scope='Original source aliases map to exact derived public content; failed/running/unexecuted jobs remain tracking, and inventory does not prove every module ran')
    name = prefix+'/PUBLIC_CONTENT_MAPPING.json'
    data = (json.dumps(exporter.clean_object(mapping),indent=2,ensure_ascii=False)+'\n').encode('utf-8'); secret_scan(data,name)
    archive.writestr(zip_info(name),data)
    entries[name] = dict(source_path=relative(value['index_path']),source_sha256=value['source_index_sha256'],
        source_bytes=value['index_path'].stat().st_size,public_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
        browsable=False,role='explicit metadata-to-content derivation from frozen runtime index; no producer/task completion inferred')
    return dict(blobs=blobs,inventories=inventories,exclusions=excluded,public_mapping=name,
        source_index_sha256=value['source_index_sha256'],jobs=len(index['jobs']),original_objects=len(value['objects']),
        inventory_mappings=value['inventory_mappings'],manifest_mappings=value['manifest_mappings'])


def scan_plan(plan):
    """Readiness inventory only; COMPLETE is not promoted to task quality."""
    rows = []
    for path in sorted((ROOT / 'results/round8_v1').rglob('complete*.json')):
        if path.name not in ('complete.json', 'complete_round8.json', 'complete_books.json',
                             'complete_postcal.json', 'complete_postcal_train.json',
                             'complete_final_measurement.json'):
            continue
        value = load(path)
        rows.append(dict(path=relative(path), sha256=sha(path), status=value.get('status'),
                         scope='completion/diagnostic tracking; quality requires owner, labels and postcheck recount'))
    pending = [key for key in ('execution_receipt','final_report','readme_entry_file','method_freeze')
               if not plan.get(key) or not local_path(plan[key]).is_file()]
    return dict(status='PLAN_ONLY_NOT_PACKAGED', tag=plan['tag_name'], pending_final_inputs=pending,
        discovered_completions=rows, public_write_performed=False, Git_operations=0, network_operations=0,
        scope='No quality victory inferred; failed, incomplete and unexecuted work stays trace-only')


def preflight(plan):
    from verify_round8_increment import verify_books
    import round8_quality_owner
    require(plan.get('schema') == 'trajkv.round8.publication-plan.v1'
        and plan.get('tag_name') == 'research-round8-20261118', 'Explicit fixed-tag R8 plan required')
    if plan.get('artifact_map'):
        round8_quality_owner.configure_path_map(local_path(plan['artifact_map']))
    required = set()
    for key in ('execution_receipt','final_report','readme_entry_file','method_freeze'):
        require(plan.get(key) and local_path(plan[key]).is_file(), 'Pending final-stage input: '+key)
        required.add(local_path(plan[key]))
    receipt = load(local_path(plan['execution_receipt']))
    require(receipt.get('research_execution_completed') is True and receipt.get('github_sync_pending') is True
        and receipt.get('status') == 'RESEARCH_COMPLETE_READY_FOR_PUBLICATION', 'Research-complete/pending-GitHub receipt required')
    if plan.get('method_freeze_sha256'):
        require(sha(local_path(plan['method_freeze'])) == plan['method_freeze_sha256'], 'Final method freeze receipt changed')
    frozen = load(local_path(plan['method_freeze']))
    require(frozen.get('frozen_files'), 'Final method freeze must pin concrete source/input evidence')
    for item in frozen['frozen_files']:
        path = local_path(item['path']); require(sha(path) == item['sha256'], 'Frozen input changed')
        required.add(path)
    summaries = list(plan.get('quality_analyses', []))
    if plan.get('include_completed_development_analyses', True):
        known = {str(x.get('summary', x) if isinstance(x,dict) else x) for x in summaries}
        for path in sorted((ROOT/'results/round8_v1').rglob('summary.json')):
            value = load(path)
            if relative(path) not in known and value.get('status') == 'COMPLETE' and value.get('audit',{}).get('status') == 'PASS':
                summaries.append(dict(summary=relative(path)))
    specs, quality_checks, quality_runs = [], [], set()
    for item in summaries:
        item = dict(summary=item) if isinstance(item,str) else item
        path = local_path(item['summary'])
        if item.get('summary_sha256'):
            require(sha(path) == item['summary_sha256'], 'Pinned quality summary changed')
        check = quality(path); value = load(path); quality_checks.append(check)
        sample = find_fingerprint('configs/a2', value['samples_sha256'])
        refs = find_fingerprint('configs/data', value['references_sha256'])
        required.update({path,path.parent/'scores.csv',sample,refs})
        runs = []
        for run in value['audit']['runs']:
            folder = local_path(run['run']); quality_runs.add(folder.resolve())
            manifest = load(folder/'manifest.json')
            for name in ('manifest.json','complete.json','complete_round8.json','records.jsonl'):
                required.add(folder/name)
            for name in ('complete_candidate.json','candidate_subset_postcheck.json',
                         'complete_postres.json','complete_final_ray.json','complete_mean.json','complete_postcal.json',
                         'audit/backend_manifest_before_candidate.json',
                         'audit/backend_complete_round8_before_candidate.json'):
                if (folder/name).is_file():
                    required.add(folder/name)
            for name in manifest.get('external_codec_inputs',{}):
                required.add(local_path(name))
            required.add(local_path(manifest['config']['arguments']['samples']))
            runs.append(relative(folder/'records.jsonl'))
        specs.append(dict(dataset=check['dataset'],analysis_role=value.get('analysis_role'),summary=relative(path),
            scores=relative(path.parent/'scores.csv'),samples=relative(sample),references=relative(refs),
            samples_source_sha256=sha(sample),references_source_sha256=sha(refs),raw_records=runs,
            selected_variants=[row['variant'] for row in value['table']],analysis_method_filter=value.get('analysis_method_filter',{})))
    require(set(plan.get('required_quality_domains',['doc','text'])) <= {x['dataset'] for x in specs}, 'Required quality domains missing')
    book_manifests = plan.get('book_manifests', ['results/round8_v1/native_books_k_v1/complete_books.json',
                                               'results/round8_v1/native_books_v_v1/complete_books.json'])
    books = verify_books(book_manifests, lambda name: load(local_path(name)), lambda name: sha(local_path(name)))
    for filename in book_manifests:
        path = local_path(filename); required.add(path)
        for item in load(path)['books']:
            required.add(path.parent/item['file'])
    for entry in books['source_input_fingerprints']:
        folder = 'configs' if entry['field'] == 'source_split_sha256' else 'results'
        required.add(find_fingerprint(folder, entry['sha256']))
    failed = []
    for path in sorted((ROOT/'results/round8_v1').rglob('records.jsonl')):
        if path.parent.resolve() not in quality_runs:
            failed.append(dict(path=relative(path.parent),scope='diagnostic/failed/partial or unselected trace; excluded from task quality'))
    if plan.get('maintenance_public_summary'):
        path = local_path(plan['maintenance_public_summary'])
        require('maintenance_20261118' not in path.parts, 'Never export raw maintenance directory')
        value = load(path)
        whitelist = {'schema','status','runtime_checks_passed','gpu_devices_available','system_free_bytes_after',
                     'data_volume_free_bytes_after','migration_accepted','caches_on_data_volume','private_receipt_sha256','scope'}
        require(set(value) <= whitelist and value.get('migration_accepted') is True
            and all(isinstance(x,(str,bool,int,float,type(None))) for x in value.values()), 'Use a narrow capability-only maintenance projection')
        required.add(path)
    checks = dict(status='PASS',quality_analyses=specs,numeric_recounts=quality_checks,
        book_manifests=book_manifests,fitted_book_verification=books,calibration_diagnostics=[],failed_smokes=failed,
        research_execution_completed=True,github_sync_pending_at_preparation=True,
        execution_goal_completed_at_preparation=False,analysis_views_may_share_raw_runs=True,
        required_owner_oracle='round8_postcal_owner.audit_owner delegating preserved families to round8_quality_owner.audit_owner',
        complete_round8_postchecks_required=True,
        scope='Actual official scores and R8 shape-based owners/postchecks recounted; overlapping development views are not independent confirmation; other traces are not quality')
    return checks, required


def transformed(exporter, path, name, failed=False):
    if path.suffix.lower() != '.jsonl':
        raw = path.read_bytes()
        if path.suffix.lower() in {'.txt', '.log'} and raw.startswith((b'\xff\xfe', b'\xfe\xff')):
            data = exporter.clean_string(raw.decode('utf-16')).encode('utf-8')
        else:
            data = exporter.transform(path, Path(name))
        secret_scan(data, name)
        return data, {}
    output, complete_rows, partial_rows = [], 0, 0
    text = path.read_bytes().decode('utf-8-sig', errors='replace' if failed else 'strict')
    for line in text.splitlines(keepends=True):
        if not line.strip():
            output.append(line)
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            require(failed, 'Incomplete JSONL outside an explicitly preserved failed run: ' + name)
            output.append(exporter.clean_string(line)); partial_rows += 1
        else:
            clean = exporter.clean_object(value)
            output.append(line if clean == value else json.dumps(clean, ensure_ascii=False) + '\n')
            complete_rows += 1
    data = ''.join(output).encode('utf-8')
    secret_scan(data, name)
    return data, dict(complete_json_rows=complete_rows, unparsed_partial_rows=partial_rows,
        parse_scope='diagnostic/failed/partial or unselected trace, excluded from task quality' if failed else 'complete JSONL')


def prepare(args):
    plan = load(args.plan)
    require(plan.get('schema') == 'trajkv.round8.publication-plan.v1', 'Explicit Round8 publication plan required')
    require(not args.output.exists(), 'Use a new immutable preparation directory')
    artifact_root = (getattr(args, 'artifact_root', None) or ROOT / 'artifacts').resolve()
    require(args.output.resolve() != artifact_root and args.output.resolve().is_relative_to(artifact_root),
            'Prepare under the explicitly selected artifact data directory')
    if os.name == 'nt':
        require(artifact_root.drive.casefold() != os.environ.get('SystemDrive', 'C:').casefold(),
                'Publication packages must use a data drive rather than the Windows system drive')
    checks, required = preflight(plan)
    required.add(args.plan.resolve())
    baseline = load(args.public_baseline / 'PUBLIC_EXPORT_MANIFEST.json')
    runtime_roots = [path.resolve().parent if path.is_file() else path.resolve() for path in args.runtime_sources]
    candidates = set(required)
    for folder in ('code', 'configs', 'docs', 'figures', 'audit/a2'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and allowed(path) and ('round8' in relative(path).lower()) and baseline.get('files', {}).get(relative(path), {}).get('source_sha256') != sha(path):
                candidates.add(path)
    for folder in [ROOT / 'results/round8_v1', *[local_path(x) for x in plan.get('extra_roots', [])]]:
        require(folder.is_relative_to(ROOT), 'Extra evidence root must stay in this project')
        for path in folder.rglob('*'):
            if (path.is_file() and allowed(path) and not path.is_relative_to(ROOT/'results/round8_v1/runtime_sources')
                    and not any(path.resolve().is_relative_to(root) for root in runtime_roots)):
                candidates.add(path)
    for path in required:
        require(path.is_file() and allowed(path), 'Required public evidence is missing/prohibited: ' + str(path))
    args.output.mkdir(parents=True)
    payload = args.output / 'payload'; payload.mkdir()
    exporter = Round8Exporter(payload)
    failed_roots = [local_path(x['path']).resolve() for x in checks['failed_smokes']]
    failed_roots += [local_path(x).resolve() for x in plan.get('failed_evidence_roots', [])]
    archive_path = args.output / plan.get('archive_name', 'round8_increment_20261118.zip')
    entries, runtime = {}, []
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(candidates):
            name = relative(path)
            original_sha = sha(path)
            failed = any(path.resolve().is_relative_to(folder) for folder in failed_roots)
            # Completed primary dependencies must always parse completely. Old
            # aborted diagnostics may have only a partial final line; archive
            # them with explicit scope instead of inventing a complete record.
            formal_raw = {value for spec in checks['quality_analyses'] for value in spec['raw_records']}
            if path.suffix == '.jsonl' and name.startswith('results/') and name not in formal_raw:
                folder = path.parent
                if not (folder / 'complete.json').is_file():
                    failed = True
            data, extra = transformed(exporter, path, name, failed)
            require(sha(path) == original_sha, 'Source changed during preparation; preserve this attempt and use a new version: ' + name)
            browsable = not (name.startswith('code/') and 'round7' in name.lower()) and not (name.startswith('results/') and path.suffix.lower() in {'.jsonl', '.log'}) and len(data) < 90 * 1024 * 1024
            if browsable:
                target = payload / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            archive.writestr(zip_info(name), data)
            entries[name] = dict(source_path=name, source_sha256=original_sha, source_bytes=path.stat().st_size,
                public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=browsable,
                role='preserved diagnostic/failed/partial or unselected trace; excluded from task quality' if failed else
                     'cached-prefix calibration diagnostic' if any(path.resolve().is_relative_to(local_path(x['path'])) for x in checks['calibration_diagnostics']) else
                     'derived research source/configuration/result evidence', **extra)
        require(args.runtime_sources, 'Supply original isolated per-job snapshots or an exact source-object collection')
        seen_runtime = set()
        inventories = []
        runtime_collections = []
        for collection_id, source_root in enumerate(args.runtime_sources):
            require(source_root.exists(), 'Supplied runtime source collection is missing')
            candidate = source_root if source_root.is_file() else source_root/'runtime_index.json'
            if candidate.is_file() and load(candidate).get('schema') == 'trajkv.round8.runtime-source-index.v1':
                derived = append_runtime_collection(runtime_collection(source_root),exporter,archive,entries,collection_id)
                runtime.extend(item for item in derived['blobs'] if item['source_sha256'] not in seen_runtime)
                seen_runtime.update(item['source_sha256'] for item in derived['blobs'])
                inventories.extend(derived['inventories']); runtime_collections.append(derived)
                continue
            require(source_root.is_dir(), 'Unrecognized runtime source index')
            collection = source_root / 'COLLECTION.json'
            if collection.is_file():
                require(load(collection).get('status') in ('PASS', 'COMPLETE'), 'Runtime source collection did not pass')
            # Preserve the launcher's actual immutable snapshot inventories.
            # These are evidence of available source versions, not proof that
            # every listed Python module executed in a worker.
            for metadata_path in sorted(source_root.rglob('sha256.json')):
                inventory = load(metadata_path)
                require(isinstance(inventory, dict) and all(re.fullmatch(r'[0-9a-f]{64}', value)
                    for value in inventory.values()), 'Malformed original job source inventory')
                original = sha(metadata_path)
                name = 'runtime_sources/inventories/' + original + '.json'
                if name in entries:
                    continue
                data = exporter.transform(metadata_path, Path(name)); secret_scan(data, name)
                archive.writestr(zip_info(name), data)
                entries[name] = dict(source_sha256=original, source_bytes=metadata_path.stat().st_size,
                    public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=False,
                    role='original isolated SR/jobs/*.source inventory; inventory is not executed-module evidence')
                inventories.append(dict(source_sha256=original, content_path=name,
                    entries=len(inventory), source_snapshot=exporter.clean_string(str(metadata_path.parent))))
            for metadata_name in ('runtime_index.json', 'COLLECTION.json'):
                metadata_path = source_root / metadata_name
                if not metadata_path.is_file():
                    continue
                name = f'runtime_sources/collection{collection_id}_' + metadata_name
                data = exporter.transform(metadata_path, Path(name)); secret_scan(data, name)
                archive.writestr(zip_info(name), data)
                entries[name] = dict(source_sha256=sha(metadata_path), source_bytes=metadata_path.stat().st_size,
                    public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=False,
                    role='per-job source inventory and narrow source mapping; inventory presence is not proof of execution')
            for path in sorted(source_root.rglob('*.py')):
                if path.name in PRIVATE or re.search(r'credential|auth[_-]?helper|connection', path.name, re.I):
                    continue
                text = path.read_text(encoding='utf-8-sig', errors='replace')
                if any(marker in text for marker in ('contractkv_ssh_credential', 'win32cred', 'CredReadW', 'CredRead(')):
                    continue
                original = sha(path)
                if re.fullmatch(r'[0-9a-f]{64}', path.stem):
                    require(path.stem == original, 'Runtime blob filename differs from original source SHA')
                if original in seen_runtime:
                    continue
                seen_runtime.add(original)
                name = 'runtime_sources/' + original + '.py'
                data = exporter.transform(path, Path(name)); secret_scan(data, name)
                archive.writestr(zip_info(name), data)
                entries[name] = dict(source_sha256=original, source_bytes=path.stat().st_size,
                    public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=False,
                    role='original runtime source version deduplicated by private source SHA; public bytes are path-redacted')
                runtime.append(dict(source_sha256=original, content_path=name))
        required_source_hashes = set()
        for spec in checks['quality_analyses']:
            for filename in spec['raw_records']:
                manifest = load(ROOT / Path(filename).parent / 'manifest.json')
                required_source_hashes.update(value for name, value in manifest.get('sources', {}).items()
                    if name.endswith('.py') and Path(name).name not in PRIVATE
                    and not re.search(r'credential|auth[_-]?helper|connection', Path(name).name, re.I))
        require(required_source_hashes <= seen_runtime,
                'Missing original consumed source objects; do not substitute current working-tree versions: '
                + str(sorted(required_source_hashes - seen_runtime)))
        require(inventories or runtime_collections or any((folder / 'runtime_index.json').is_file() for folder in args.runtime_sources if folder.is_dir()),
                'Original source inventory/index is required')
        index = dict(schema='trajkv.round8.public.increment.v1', status='PASS',
            generated_at_utc=datetime.now(timezone.utc).isoformat(), files=entries, checks=checks,
            publication_plan=plan, publication_plan_sha256=sha(args.plan),
            baseline_manifest_sha256=sha(args.public_baseline / 'PUBLIC_EXPORT_MANIFEST.json'),
            runtime_source_blobs=runtime,
            runtime_original_inventories=inventories,
            runtime_collection_derivations=runtime_collections,
            quality_manifest_source_hashes=sorted(required_source_hashes),
            exclusions='private chats/checkpoints/authentication/connection helpers, original images, model weights, all tensor/checkpoint/moments.pt files, synthetic split_root_fixture payloads (CPU receipts and real instrument source retained)',
            hash_scope='source SHA fixes preserved original bytes; public SHA fixes derived archive bytes; fixed content_ref and LF Git identities are recorded separately',
            snapshot_scope='a recorded source inventory does not prove every module executed; historical source versions are retained when supplied',
            public_write_performed=False)
        encoded = (json.dumps(exporter.clean_object(index), indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        secret_scan(encoded, 'ROUND8_SOURCE_INDEX.json')
        archive.writestr(zip_info('ROUND8_SOURCE_INDEX.json'), encoded)
    index_path = args.output / 'ROUND8_SOURCE_INDEX.json'; index_path.write_bytes(encoded)
    content_path = args.output / 'ROUND8_CONTENT_REFS.json'
    content = build_content_index(SimpleNamespace(output=content_path, source_index=index_path, payload=payload, archive=archive_path))
    delivery = dict(status='PREPARED_LOCAL_ONLY', archive=archive_path.name, archive_sha256=sha(archive_path),
        archive_bytes=archive_path.stat().st_size, source_index_sha256=sha(index_path), content_index_sha256=sha(content_path),
        files=len(entries), browsable_files=sum(x['browsable'] for x in entries.values()),
        force_add_metadata_paths=content['git_stage_plan']['force_paths'], public_write_performed=False,
        source_plan=str(args.plan.resolve()), source_plan_sha256=sha(args.plan))
    (args.output / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in delivery.items() if key != 'force_add_metadata_paths'}))


def runtime_plan_test(args):
    """Real collection derivation test, intentionally not a research package."""
    require(args.output and not args.output.exists() and args.runtime_sources,'New output and real runtime collections required')
    require(args.output.resolve().is_relative_to(ROOT/'artifacts'),'Runtime-only test stays under project artifacts/')
    args.output.mkdir(parents=True); payload = args.output/'payload'; payload.mkdir()
    exporter = Round8Exporter(payload); entries,runtime = {},[]
    archive_path = args.output/'RUNTIME_DERIVATION_TEST.zip'
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for number,root in enumerate(args.runtime_sources):
            runtime.append(append_runtime_collection(runtime_collection(root),exporter,archive,entries,number))
        source = dict(schema='trajkv.round8.runtime-derivation-test.v1',status='RUNTIME_PLAN_ONLY_NOT_RESEARCH_PACKAGE',
            CPU_fixture_only=True,files=entries,runtime_collections=runtime,research_execution_completed=False,
            final_inputs_pending=['execution_receipt','final_report','readme_entry_file','method_freeze'],public_write_performed=False)
        encoded = (json.dumps(exporter.clean_object(source),indent=2,ensure_ascii=False)+'\n').encode('utf-8')
        archive.writestr(zip_info('ROUND8_RUNTIME_PLAN_SOURCE_INDEX.json'),encoded)
    source_index = args.output/'ROUND8_RUNTIME_PLAN_SOURCE_INDEX.json';source_index.write_bytes(encoded)
    content_path = args.output/'ROUND8_RUNTIME_PLAN_CONTENT_REFS.json'
    content = build_content_index(SimpleNamespace(output=content_path,source_index=source_index,payload=payload,archive=archive_path))
    from round8_content_index import verify as verify_refs
    checked = verify_refs(SimpleNamespace(verify_index=content_path,archive=archive_path,payload=payload,git_tree_map=None))
    from verify_round8_increment import verify_runtime_collections
    with zipfile.ZipFile(archive_path) as archive:
        tracking = verify_runtime_collections(source,archive.read)
    for root,value in zip(args.runtime_sources,runtime):
        path = root if root.is_file() else root/'runtime_index.json'
        require(sha(path) == value['source_index_sha256'],'Frozen runtime index mutated during test')
    result = dict(status='RUNTIME_DERIVATION_TEST_PASS_NOT_STAGE_COMPLETE',archive_sha256=sha(archive_path),
        source_index_sha256=sha(source_index),content_index_sha256=sha(content_path),runtime=runtime,
        files=len(entries),content_identity_check=checked,runtime_source_tracking=tracking,final_inputs_pending=source['final_inputs_pending'],
        Git_operations=0,network_operations=0,SSH_operations=0,GPU_used=False,public_checkout_touched=False,
        publishable=False,scope='Real frozen sources sanitized and content-indexed only; no final delivery/quality completion asserted')
    (args.output/'RUNTIME_PLAN_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],archive_sha256=result['archive_sha256'],files=len(entries),
        runtime=[{key:value[key] for key in ('source_index_sha256','jobs','original_objects','inventory_mappings','manifest_mappings')}
                 for value in runtime],content_identity_check=checked,final_inputs_pending=source['final_inputs_pending'],
        publishable=False,public_checkout_touched=False),indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--artifact-root', type=Path,
        help='Explicit local data-drive package root; default is this project artifacts/; source evidence stays in the project')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--plan-only', action='store_true', help='Inventory pending final inputs without packaging, Git or network')
    p.add_argument('--runtime-plan-only', action='store_true', help='Test real source collection derivation only; cannot be published')
    p.add_argument('--public-baseline', type=Path, default=ROOT.parent / 'TrajKV_public_20261114')
    p.add_argument('--runtime-sources', nargs='*', type=Path, default=[])
    args = p.parse_args()
    args.plan = args.plan.resolve(); args.public_baseline = args.public_baseline.resolve()
    if args.runtime_plan_only:
        require(not args.plan_only and not args.preflight_only,'Select exactly one planning mode')
        args.output = args.output.resolve() if args.output else None
        runtime_plan_test(args); raise SystemExit(0)
    if args.plan_only:
        value = scan_plan(load(args.plan))
        value['runtime_sources'] = [dict(source_index_sha256=item['source_index_sha256'],jobs=len(item['index']['jobs']),
            objects=len(item['objects']),inventory_mappings=item['inventory_mappings'],manifest_mappings=item['manifest_mappings'],
            status='FROZEN_COLLECTION_VERIFIED_NOT_QUALITY') for item in [runtime_collection(root) for root in args.runtime_sources]]
        print(json.dumps(value, indent=2)); raise SystemExit(0)
    if args.preflight_only:
        checks, required = preflight(load(args.plan))
        print(json.dumps(dict(status='PASS', quality_analyses=len(checks['quality_analyses']),
            research_execution_completed=checks['research_execution_completed'],
            execution_goal_completed_at_preparation=checks['execution_goal_completed_at_preparation'],
            github_sync_pending_at_preparation=checks['github_sync_pending_at_preparation'],
            required_input_files=len(required), writes=0, network_operations=0)))
    else:
        require(args.output, '--output required for actual local preparation')
        args.output = args.output.resolve(); prepare(args)
