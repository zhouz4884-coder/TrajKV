"""Cold verify an R12 text/scalar package, optionally its exact ZIP.

Portable: Python + numpy, no network/GPU/credentials/private transport needed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from round12_public_core_v1 import (NAMESPACE, PublicSource, canonical,
    check_header, payload, require, safe_relative, sha, verify_semantics)


def lf_identity(path):
    body = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return {'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
            'blob_oid_sha1': hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()}


def verify(root, archive=None, *, allow_fixture=False):
    root = Path(root).resolve()
    indexpath = root / 'PUBLIC_CONTENT_REFS.json'
    index = json.loads(indexpath.read_text(encoding='utf-8-sig'))
    require(index.get('schema') == 'trajkv.round12.public_content_refs.v1'
            and index.get('namespace') == NAMESPACE, 'actual R12 content index required')
    require(allow_fixture or (index.get('fixture_only') is False and index.get('publication_eligible') is True),
            'fixture/unsealed package cannot become production delivery')
    entries = index['files']
    require(entries and len(entries) == len(set(entries)), 'empty/duplicate public inventory')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(entries) | {'PUBLIC_CONTENT_REFS.json'}, 'extra or missing public file')
    for rel, entry in entries.items():
        safe_relative(rel)
        path = root / rel
        require(path.stat().st_size == entry['public_bytes'] and sha(path) == entry['public_sha256'],
                'public body changed: ' + rel)
        require(lf_identity(path) == entry['git_lf'], 'Git LF/blob identity differs: ' + rel)
        require(entry.get('delivery_channel') == ('BROWSER_AND_ARCHIVE' if entry['expected_in_git'] else 'ARCHIVE_ONLY'),
                'delivery channel disagrees with actual expected Git plan')
        if entry['transformation'] == 'EXACT':
            require(entry['source_sha256'] == entry['public_sha256'], 'false exact-source identity')
        ref = entry['content_ref']
        require(ref['member'] == NAMESPACE + '/' + rel
                and ref['public_body_sha256'] == entry['public_sha256']
                and ref['archive_asset'] == index['archive_asset'], 'fixed content_ref differs')
    force = set(index['git_stage_plan']['force_paths'])
    require(force == {NAMESPACE + '/' + rel for rel, entry in entries.items() if entry['expected_in_git']},
            'expected Git tree differs')
    channels = index['delivery_channels']
    require(set(channels['browser_paths']) == force | {NAMESPACE + '/PUBLIC_CONTENT_REFS.json'}
            and len(channels['browser_paths']) == len(set(channels['browser_paths'])), 'browser delivery inventory differs')
    require(set(channels['archive_paths']) == {NAMESPACE + '/' + rel for rel in entries}
            | {NAMESPACE + '/PUBLIC_CONTENT_REFS.json'}
            and len(channels['archive_paths']) == len(entries) + 1, 'archive delivery inventory differs')
    limit = channels['browser_max_file_bytes']
    require(type(limit) is int and 0 < limit <= 16 * 1024 ** 2 and indexpath.stat().st_size <= limit,
            'browser safety limit/index size differs')
    require(all(entry['public_bytes'] <= limit for entry in entries.values() if entry['expected_in_git']),
            'oversized browser payload')
    require(channels['browser_bytes_excluding_self_index'] == sum(entry['git_lf']['bytes'] for entry in entries.values() if entry['expected_in_git'])
            and channels['archive_uncompressed_bytes_excluding_self_index'] == sum(entry['public_bytes'] for entry in entries.values())
            and channels['archive_only_files'] == sum(not entry['expected_in_git'] for entry in entries.values()),
            'browser/archive byte accounting differs')
    for rel in index['required_reference_files']:
        require(rel in entries and entries[rel]['expected_in_git']
                and NAMESPACE + '/' + rel in force, 'input/reference hidden by ignore rule')
    source = PublicSource(root, index)
    scope = source.doc(index['scope_file'])
    fixture = check_header(scope, sealed=True, allow_fixture=allow_fixture)
    require(scope['root_seal']['public_payload_sha256'] == canonical(payload(scope)), 'public Root cutoff seal differs')
    original_paths = {item['public_path'] for item in scope['files']}
    require(len(original_paths) == len(scope['files']), 'duplicate sealed original path')
    for item in scope['files']:
        rel = item['public_path']
        require(rel in entries and entries[rel]['source_sha256'] == item['sha256']
                and entries[rel].get('evidence_role') == item['role'],
                'archive original file differs from Root-sealed inventory: ' + rel)
    generated = {'README_PUBLIC.md', '.gitattributes', 'evidence/ROUND12_SCOPE.json',
                 'evidence/ROUND12_PHASE.json', 'ops/round12_public_core_v1.py',
                 'ops/verify_public_round12_v1.py'}
    line_files = {entry['line_identities_file'] for entry in entries.values() if entry.get('line_identities_file')}
    require(set(entries) <= original_paths | generated | line_files, 'unsealed indexed public member')
    require(all(entry['source_sha256'] is None or rel in original_paths
                for rel, entry in entries.items()), 'unsealed original source in package')
    require(index['fixture_only'] == fixture and index['source_scope_sha256'], 'scope/index fixture identity differs')
    result = verify_semantics(scope, source, allow_fixture=allow_fixture)
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            require(len(names) == len(set(names)), 'duplicate ZIP member')
            expected = {NAMESPACE + '/' + n for n in entries} | {NAMESPACE + '/PUBLIC_CONTENT_REFS.json'}
            require(set(names) == expected, 'archive member scope differs')
            require(z.read(NAMESPACE + '/PUBLIC_CONTENT_REFS.json') == indexpath.read_bytes(), 'archive index differs')
            for rel, entry in entries.items():
                h = hashlib.sha256()
                with z.open(NAMESPACE + '/' + rel) as f:
                    for chunk in iter(lambda: f.read(1024 ** 2), b''):
                        h.update(chunk)
                require(h.hexdigest() == entry['public_sha256'], 'archive body differs: ' + rel)
    return {**result, 'status': 'PASS_PUBLIC_R12_FIXTURE_ONLY' if fixture else 'PASS_PUBLIC_ROUND12_EXPORT_VERIFIED_V1',
            'archive_verified': archive is not None, 'archive_sha256': sha(archive) if archive else None,
            'public_files': len(entries), 'expected_Git_payload_files': len(force) + 1,
            'archive_only_files': channels['archive_only_files'],
            'browser_payload_bytes_with_index': channels['browser_bytes_excluding_self_index'] + indexpath.stat().st_size,
            'source_public_Git_LF_archive_identities_separate': True,
            'network_calls': 0, 'GitHub_mutation': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path)
    p.add_argument('--allow-fixture', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = verify(a.root, a.archive, allow_fixture=a.allow_fixture)
    if a.output:
        require(not a.output.exists(), 'new immutable cold receipt required')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
