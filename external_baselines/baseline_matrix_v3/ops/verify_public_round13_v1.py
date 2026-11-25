"""Portable R13 public identity/security/archive checks plus Root semantics.

No network, GPU, credentials or private tensor access. Production requires an
explicit indexed Root semantic verifier; a transport fixture cannot publish.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile

from round12_public_core_v1 import NAMESPACE, PublicSource, canonical, require, safe_relative
from round12_public_security_v1 import findings

SCHEMA = 'trajkv.round13.public_content_refs.v1'
STATUS = 'PASS_PUBLIC_ROUND13_EXPORT_VERIFIED_V1'
BROWSER_LIMIT = 16 * 1024 ** 2


def _hash_body(relative, path, entry, identity_path=None):
    """One body pass for SHA/LF/blob/privacy/JSON and optional line binding."""
    raw_h, lf_h = hashlib.sha256(), hashlib.sha256()
    expected_lf = entry['git_lf']
    blob_h = hashlib.sha1(f"blob {expected_lf['bytes']}\0".encode())
    raw_size = lf_size = lines = 0
    def consume(body):
        nonlocal raw_size, lf_size
        raw_h.update(body); raw_size += len(body)
        normalized = body.replace(b'\r\n', b'\n')
        lf_h.update(normalized); blob_h.update(normalized); lf_size += len(normalized)
    if path.suffix.lower() == '.jsonl':
        ids = identity_path.open('rb') if identity_path else None
        try:
            with path.open('rb') as stream:
                for number, body in enumerate(stream, 1):
                    require(body.strip(), 'R13 blank public JSONL row: ' + relative)
                    value = json.loads(body)
                    require(not findings(relative, body), 'R13 public JSONL privacy findings: ' + relative)
                    consume(body); lines += 1
                    if ids is not None:
                        identity = json.loads(ids.readline())
                        require(identity['line'] == number and identity.get('id') == value.get('id')
                                and identity['public_sha256'] == hashlib.sha256(body).hexdigest()
                                and re.fullmatch('[0-9a-f]{64}', identity['source_sha256']) is not None,
                                'R13 JSONL source/public identity differs: ' + relative)
            require(ids is None or not ids.readline(), 'R13 extra JSONL line identities: ' + relative)
        finally:
            if ids is not None:
                ids.close()
        if identity_path is not None:
            require(lines == entry['line_count'], 'R13 JSONL line count differs')
    else:
        # Whole-body semantics deliberately preserve cross-line privacy rules.
        body = path.read_bytes()
        require(not findings(relative, body), 'R13 public body privacy findings: ' + relative)
        consume(body)
    require(raw_size == entry['public_bytes'] and raw_h.hexdigest() == entry['public_sha256'],
            'R13 public body SHA/size differs: ' + relative)
    actual_lf = {'sha256': lf_h.hexdigest(), 'bytes': lf_size, 'blob_oid_sha1': blob_h.hexdigest()}
    require(actual_lf == expected_lf, 'R13 Git LF/blob identity differs: ' + relative)
    return {'bytes': raw_size, 'JSONL_rows': lines}


def load_semantic_callback(root, index):
    descriptor = index.get('semantic_verifier')
    require(isinstance(descriptor, dict), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    relative = safe_relative(descriptor['relative_path'])
    name = descriptor.get('function', 'verify_semantics')
    require(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', name) is not None,
            'R13 safe semantic function name required')
    require(relative in index['files'] and descriptor['public_sha256'] == index['files'][relative]['public_sha256'],
            'R13 semantic module not bound by public index')
    for folder in (root / 'ops', root / 'code'):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('_r13_root_semantics_' + descriptor['public_sha256'][:16], root / relative)
    require(spec is not None and spec.loader is not None, 'R13 semantic module cannot load')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    callback = getattr(module, name, None)
    require(callable(callback), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    return callback


def verify(root, archive=None, *, semantic_callback=None, allow_fixture=False):
    root = Path(root).resolve()
    index_path = root / 'PUBLIC_CONTENT_REFS.json'
    index_body = index_path.read_bytes()
    require(not findings('PUBLIC_CONTENT_REFS.json', index_body), 'R13 index privacy findings')
    index = json.loads(index_body)
    fixture = index.get('fixture_only') is True
    require(index.get('schema') == SCHEMA and index.get('namespace') == NAMESPACE,
            'R13 actual content index required')
    require((allow_fixture and fixture) or (not fixture and index.get('publication_eligible') is True),
            'R13 fixture/unsealed export cannot become production')
    require(fixture or isinstance(index.get('semantic_verifier'), dict), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    entries = index['files']
    require(entries and len(entries) == len(set(entries)), 'R13 empty/duplicate inventory')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(entries) | {'PUBLIC_CONTENT_REFS.json'}, 'R13 unindexed/missing public file')
    checked_lines = 0
    for relative, entry in entries.items():
        safe_relative(relative)
        line_path = entry.get('line_identities_file')
        require(line_path is None or line_path in entries, 'R13 unindexed JSONL identity sidecar')
        identity = root / safe_relative(line_path) if line_path else None
        checked = _hash_body(relative, root / relative, entry, identity)
        checked_lines += checked['JSONL_rows']
        if entry['transformation'] == 'EXACT':
            require(entry['source_sha256'] == entry['public_sha256'], 'R13 false EXACT identity')
        channel = 'BROWSER_AND_ARCHIVE' if entry['expected_in_git'] else 'ARCHIVE_ONLY'
        require(entry['delivery_channel'] == channel, 'R13 delivery channel differs')
        require(not entry['expected_in_git'] or entry['public_bytes'] <= BROWSER_LIMIT, 'R13 oversized browser body')
        require(entry['content_ref'] == {'archive_asset': index['archive_asset'],
                'member': NAMESPACE + '/' + relative, 'public_body_sha256': entry['public_sha256']},
                'R13 fixed content reference differs')
    require(len(index_body) <= BROWSER_LIMIT, 'R13 content index exceeds browser limit')
    browser = {NAMESPACE + '/' + rel for rel, entry in entries.items() if entry['expected_in_git']}
    all_paths = {NAMESPACE + '/' + rel for rel in entries} | {NAMESPACE + '/PUBLIC_CONTENT_REFS.json'}
    channels = index['delivery_channels']
    require(channels['browser_max_file_bytes'] == BROWSER_LIMIT
            and channels['browser_bytes_excluding_self_index'] == sum(
                entry['git_lf']['bytes'] for entry in entries.values() if entry['expected_in_git'])
            and channels['archive_uncompressed_bytes_excluding_self_index'] == sum(
                entry['public_bytes'] for entry in entries.values())
            and channels['archive_only_files'] == sum(not entry['expected_in_git'] for entry in entries.values()),
            'R13 channel byte/count accounting differs')
    require(set(index['git_stage_plan']['force_paths']) == browser
            and set(channels['browser_paths']) == browser | {NAMESPACE + '/PUBLIC_CONTENT_REFS.json'}
            and len(channels['browser_paths']) == len(browser) + 1,
            'R13 explicit browser/force plan differs')
    require(set(channels['archive_paths']) == all_paths and len(channels['archive_paths']) == len(all_paths),
            'R13 archive member inventory differs')
    for rel in index['required_reference_files']:
        require(rel in entries and entries[rel]['expected_in_git'] and NAMESPACE + '/' + rel in browser,
                'R13 required refs/input hidden by ignore')
    require(set(index['required_reference_files']) == {rel for rel, entry in entries.items()
                if entry.get('evidence_role') in ('inputs', 'refs', 'source_lock')},
            'R13 reference role/index coverage differs')
    source = PublicSource(root, index)
    scope = source.doc(index['scope_file'])
    require(scope.get('schema') == 'trajkv.round13.public_scope.v1' and scope.get('fixture_only') is fixture,
            'R13 scope/index schema or fixture differs')
    seal = scope.get('root_seal')
    payload = {k: v for k, v in scope.items() if k != 'root_seal'}
    require(scope.get('stage_state') == 'ROOT_SEALED' and isinstance(seal, dict)
            and seal.get('schema') == 'trajkv.round13.root_seal.v1'
            and seal.get('public_payload_sha256') == canonical(payload), 'R13 public Root seal differs')
    originals = {item['public_path']: item for item in scope['files']}
    require(len(originals) == len(scope['files']), 'R13 duplicate sealed source path')
    for relative, item in originals.items():
        require(relative in entries and entries[relative]['source_sha256'] == item['sha256']
                and entries[relative].get('evidence_role') == item['role'], 'R13 sealed source mapping differs')
    generated = {'README_PUBLIC.md', '.gitattributes', 'evidence/ROUND13_SCOPE.json', 'evidence/ROUND13_PHASE.json'}
    sidecars = {entry['line_identities_file'] for entry in entries.values() if entry.get('line_identities_file')}
    require(set(entries) <= set(originals) | generated | sidecars, 'R13 unsealed public source/member')
    archive_verified = False
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and set(names) == all_paths, 'R13 ZIP member set/duplicates differ')
            require(z.read(NAMESPACE + '/PUBLIC_CONTENT_REFS.json') == index_body, 'R13 ZIP index differs')
            for relative, entry in entries.items():
                h = hashlib.sha256()
                with z.open(NAMESPACE + '/' + relative) as stream:
                    for part in iter(lambda: stream.read(1024 ** 2), b''):
                        h.update(part)
                require(h.hexdigest() == entry['public_sha256'], 'R13 ZIP body SHA differs: ' + relative)
        archive_verified = True
    # Only after every body/JSONL/ZIP member/privacy check is semantic code run.
    if semantic_callback is None:
        semantic_callback = load_semantic_callback(root, index)
    require(callable(semantic_callback), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    semantic = semantic_callback(scope, source, allow_fixture=allow_fixture)
    require(isinstance(semantic, dict), 'R13 semantic verifier returned no result')
    if not fixture:
        require(semantic.get('publication_eligible') is True
                and semantic.get('actual_producer_binding_required') is True
                and semantic.get('private_tensors_or_GPU_replayed') is False
                and semantic.get('whole_external_library_completion_claimed') is False
                and isinstance(semantic.get('counts'), dict), 'R13 complete Root semantic closure required')
    return {**semantic, 'status': 'PASS_R13_PUBLIC_FIXTURE_TRANSPORT_ONLY' if fixture else STATUS,
        'fixture_only': fixture, 'publication_eligible': not fixture,
        'archive_verified': archive_verified, 'public_files_verified': len(entries),
        'all_public_JSONL_rows_verified': checked_lines, 'expected_Git_payload_files': len(browser) + 1,
        'actual_producer_binding_required': False if fixture else True,
        'private_tensors_or_GPU_replayed': False, 'whole_external_library_completion_claimed': False,
        'GitHub_mutation': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--allow-fixture', action='store_true')
    a = p.parse_args()
    result = verify(a.root, a.archive, allow_fixture=a.allow_fixture)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
