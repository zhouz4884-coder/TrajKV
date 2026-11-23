"""Exact, explicitly enumerated local source-path aliases for public v4.

Inventory/export proves OS samefile plus source SHA for every added spelling.
Cold verification never case-folds, guesses a path or opens unindexed input.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

SCHEMA = 'trajkv.explicit_samefile_source_aliases.v1'
PROOF = 'OS_SAMEFILE_AND_STREAMED_SOURCE_SHA256'


def require(value, message):
    if not value: raise ValueError('ROUND11_EXACT_ALIAS: '+message)


def normalize(value):
    value = str(value).replace('\\', '/')
    while '//' in value: value = value.replace('//', '/')
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def verify_local_alias(requested_path, canonical_path, expected_sha):
    requested = Path(requested_path).absolute(); canonical = Path(canonical_path).resolve()
    require(requested.is_file() and canonical.is_file(), 'explicit local alias must name an existing file')
    require(requested.samefile(canonical), 'local alias is a distinct physical file')
    require(sha(requested) == sha(canonical) == expected_sha, 'local alias/source original SHA differs')
    require(requested.stat().st_size == canonical.stat().st_size, 'local alias original size differs')
    return {'requested_path': str(requested), 'canonical_path': str(canonical), 'source_sha256': expected_sha,
        'source_bytes': canonical.stat().st_size, 'same_physical_file': True, 'verification': PROOF}


def append_local_alias(records, record):
    key = (record['requested_path'], record['canonical_path'], record['source_sha256'])
    if not any((r['requested_path'], r['canonical_path'], r['source_sha256']) == key for r in records): records.append(record)


def build_aliases(selected, entries, sanitizer):
    aliases = {}; equivalences = []; verified = []
    def add(value, member):
        alias = normalize(sanitizer.string(str(value)))
        if alias in aliases and aliases[alias] != member:
            previous = aliases[alias]; left, right = entries[previous], entries[member]
            require(left['source_sha256'] == right['source_sha256'] and left['public_sha256'] == right['public_sha256'] and left['source_bytes'] == right['source_bytes'], 'one exact alias names ambiguous source/public bodies')
            item = {'alias': alias, 'representative': previous, 'equivalent_copy': member,
                'source_sha256': right['source_sha256'], 'public_sha256': right['public_sha256']}
            if item not in equivalences: equivalences.append(item)
        else: aliases[alias] = member
        return alias
    for rel, item in sorted(selected.items()):
        for value in [str(item['path']), *item.get('aliases', [])]: add(value, rel)
        for record in item.get('verified_local_aliases', []):
            actual = verify_local_alias(record['requested_path'], record['canonical_path'], entries[rel]['source_sha256'])
            require(actual == record, 'inventory local alias proof changed before export')
            alias = add(record['requested_path'], rel); canonical_alias = add(record['canonical_path'], rel); member = aliases[alias]
            bound = {'alias': alias, 'canonical_alias': canonical_alias, 'member': member,
                'source_sha256': entries[member]['source_sha256'], 'public_sha256': entries[member]['public_sha256'],
                'source_bytes': entries[member]['source_bytes'], 'same_physical_file': True, 'verification': PROOF}
            if bound not in verified: verified.append(bound)
    return aliases, equivalences, verified


def validate_alias_bindings(index):
    require(index.get('source_alias_schema') == SCHEMA and index.get('source_alias_lookup_is_exact') is True,
            'explicit samefile alias schema and exact-lookup declaration required')
    entries = index['files']; aliases = index['source_aliases']; records = index['verified_local_source_aliases']
    require(isinstance(records, list) and all(normalize(key) == key and member in entries for key, member in aliases.items()), 'unsafe or unindexed exact alias mapping')
    seen = set()
    for record in records:
        alias, canonical = record['alias'], record['canonical_alias']; member = record['member']
        require(alias == normalize(alias) and canonical == normalize(canonical) and alias in aliases and canonical in aliases and member in entries and aliases[alias] == member, 'verified local alias is not its exact indexed member')
        require(record['same_physical_file'] is True and record['verification'] == PROOF, 'missing physical/source-identity alias verification')
        for target in (entries[member], entries[aliases[canonical]]):
            require(target['source_sha256'] == record['source_sha256'] and target['public_sha256'] == record['public_sha256'] and target['source_bytes'] == record['source_bytes'], 'verified alias source/public identity is ambiguous')
        key = (alias, canonical, member)
        require(key not in seen, 'duplicate local alias binding'); seen.add(key)
    return {'status': 'PASS_EXPLICIT_LOCAL_SAMEFILE_SOURCE_ALIAS_BINDINGS', 'verified_local_aliases': len(records), 'casefold_or_unindexed_fallback': False}
