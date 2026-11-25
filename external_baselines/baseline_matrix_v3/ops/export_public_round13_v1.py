"""R13 explicit sealed export using frozen streaming/sanitizer primitives.

No authentication, network, GPU or Git writes. Root must supply the final
indexed semantic verifier; no unfinished scope can create a public stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile

from export_public import Sanitizer
from export_public_round9_formal_v6 import write_public_file, generated_entry, json_bytes, io_path
from export_public_round12_v2 import exact_scanner
from round12_public_core_v1 import NAMESPACE, canonical, require, sha, safe_relative
from round12_public_security_v1 import findings
import round13_public_adapter_v1 as admission
from verify_public_round13_v1 import verify, BROWSER_LIMIT

ROOT, EVIDENCE = admission.ROOT, admission.EVIDENCE
PORTABLE_CORE = ('ops/verify_public_round13_v1.py', 'ops/round13_public_adapter_v1.py',
    'ops/round12_public_core_v1.py', 'ops/round12_public_legacy_v1.py',
    'ops/round12_public_security_v1.py', 'ops/export_public.py')


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json_bytes(value))


def _local_callback(source, descriptor):
    relative = safe_relative(descriptor['relative_path'])
    name = descriptor.get('function', 'verify_semantics')
    require(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', name), 'R13 semantic function identifier')
    path = source.path(relative)
    spec = importlib.util.spec_from_file_location('_r13_local_root_semantics_' + sha(path)[:16], path)
    require(spec is not None and spec.loader is not None, 'R13 semantic module loader')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    callback = getattr(module, name, None)
    require(callable(callback), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    return callback


def export(scope_path, stage, tag, *, semantic_verifier, semantic_callback=None):
    require(re.fullmatch('[A-Za-z0-9_-]+', stage) and re.fullmatch('[A-Za-z0-9_.-]+', tag),
            'R13 new safe stage/tag required')
    scope_path = Path(scope_path)
    scope_sha = sha(scope_path)
    scope = json.loads(scope_path.read_text(encoding='utf-8-sig'))
    admission.check_final_header(scope)
    seal = scope.get('root_seal')
    require(scope['stage_state'] == 'ROOT_SEALED' and isinstance(seal, dict)
            and seal.get('schema') == 'trajkv.round13.root_seal.v1', 'R13 unsealed scope cannot export')
    require(seal['source_payload_sha256'] == canonical(admission.payload(scope))
            and seal['public_payload_sha256'] == canonical(Sanitizer().tree(admission.payload(scope))),
            'R13 Root source/public seal differs')
    source, _ = admission.inventory_preflight(scope)
    require(isinstance(semantic_verifier, dict), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    semantic_relative = safe_relative(semantic_verifier['relative_path'])
    source.path(semantic_relative)
    for relative in PORTABLE_CORE:
        source.path(relative)
    callback = semantic_callback or _local_callback(source, semantic_verifier)
    require(callable(callback), 'R13_ROOT_SEMANTIC_VERIFIER_REQUIRED')
    closure = callback(scope, source, allow_fixture=False)
    require(closure.get('publication_eligible') is True and closure.get('actual_producer_binding_required') is True
            and closure.get('private_tensors_or_GPU_replayed') is False
            and closure.get('whole_external_library_completion_claimed') is False,
            'R13 complete Root semantic closure required before stage creation')
    target = EVIDENCE / 'public_staging' / stage
    require(not target.exists(), 'R13 preserve prior exports; use new unique stage')
    target.mkdir(parents=True)
    public = io_path(target / 'tree' / NAMESPACE)
    public.mkdir(parents=True)
    sanitizer = Sanitizer()
    entries, unsafe = {}, []
    for relative, item in sorted(source.files.items()):
        path = item['path']; stat = path.stat()
        origin = ROOT if path.is_relative_to(ROOT) else EVIDENCE
        selected = {'path': path, 'origin': 'project' if origin == ROOT else 'evidence_mirror',
            'source_path': path.relative_to(origin).as_posix(),
            'stat': (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), 'large': True}
        if relative == 'ops/export_public.py':
            entry, side, bad = exact_scanner(relative, selected, public)
        else:
            entry, side, bad = write_public_file(relative, selected, public, sanitizer)
        require(entry['source_sha256'] == item['sha256'], 'R13 sealed source changed during export: ' + relative)
        entry['evidence_role'] = item['role']
        if 'expected_in_git' in item:
            entry['expected_in_git'] = bool(item['expected_in_git'])
        if item.get('delivery_channel') is not None:
            require(item['delivery_channel'] in ('BROWSER_AND_ARCHIVE', 'ARCHIVE_ONLY'), 'R13 channel name')
            entry['expected_in_git'] = item['delivery_channel'] == 'BROWSER_AND_ARCHIVE'
        if item['role'] in ('inputs', 'refs', 'source_lock'):
            require(item.get('delivery_channel', 'BROWSER_AND_ARCHIVE') == 'BROWSER_AND_ARCHIVE'
                    and item.get('expected_in_git', True) is True, 'R13 references cannot be hidden')
            entry['expected_in_git'] = True
        raw_only = Path(relative).name in ('records.jsonl', 'scored_records.jsonl') or '/records/' in relative
        if raw_only:
            require(item.get('delivery_channel') != 'BROWSER_AND_ARCHIVE' and item.get('expected_in_git') is not True,
                    'R13 complete owner/raw records must be archive-only')
            entry['expected_in_git'] = False
        require(not entry['expected_in_git'] or entry['public_bytes'] <= BROWSER_LIMIT,
                'R13 browser body exceeds16MiB: ' + relative)
        entries[relative] = entry
        if side:
            require(side not in entries, 'R13 line sidecar collision')
            entries[side] = generated_entry(public / side, False)
        if bad:
            unsafe.append({'path': relative, 'findings': bad})
    public_scope = sanitizer.tree(scope)
    require(public_scope['root_seal']['public_payload_sha256'] == canonical(admission.payload(public_scope)),
            'R13 sanitized Root seal changed')
    generated = {
        'evidence/ROUND13_SCOPE.json': json_bytes(public_scope),
        'evidence/ROUND13_PHASE.json': json_bytes(sanitizer.tree(closure)),
        'README_PUBLIC.md': ('# Round13 Root-sealed research cutoff\n\n'
            'Development, reused references, actual free answers, failed attempts, calibration and QA0 remain separate.\n'
            'The release ZIP contains the explicit complete text/scalar evidence; browser files are a declared subset.\n'
            'Images, weights, private tensors, connection routes, credentials and private chat are excluded.\n\n'
            'Cold verification: `python -B ops/verify_public_round13_v1.py --root . --archive <ZIP>`. '
            'The index binds the Root semantic verifier. Python and numpy suffice; no network or GPU is used.\n'
            'Local preparation is not GitHub upload. The final publisher must verify commit/tree/body/ZIP members and prior history.\n').encode(),
        '.gitattributes': b'* text=auto eol=lf\n'}
    for relative, body in generated.items():
        require(relative not in entries, 'R13 generated path collision')
        path = public / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
        bad = findings(relative, body)
        if bad:
            unsafe.append({'path': relative, 'findings': bad})
        entries[relative] = generated_entry(path)
    archive_url = f'https://github.com/zhouz4884-coder/TrajKV/releases/download/{tag}/{stage}.zip'
    for relative, entry in entries.items():
        entry['delivery_channel'] = 'BROWSER_AND_ARCHIVE' if entry['expected_in_git'] else 'ARCHIVE_ONLY'
        entry['content_ref'] = {'archive_asset': archive_url, 'member': NAMESPACE + '/' + relative,
                                'public_body_sha256': entry['public_sha256']}
    browser = [NAMESPACE + '/' + rel for rel, entry in entries.items() if entry['expected_in_git']]
    archive_paths = [NAMESPACE + '/' + rel for rel in entries] + [NAMESPACE + '/PUBLIC_CONTENT_REFS.json']
    index = {'schema': 'trajkv.round13.public_content_refs.v1', 'namespace': NAMESPACE,
        'release_tag': tag, 'fixture_only': False, 'publication_eligible': not unsafe,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_scope_sha256': scope_sha,
        'scope_file': 'evidence/ROUND13_SCOPE.json', 'archive_asset': archive_url, 'files': entries,
        'required_reference_files': [rel for rel, entry in entries.items()
            if entry.get('evidence_role') in ('inputs', 'refs', 'source_lock')],
        'semantic_verifier': {'relative_path': semantic_relative,
            'function': semantic_verifier.get('function', 'verify_semantics'),
            'source_sha256': entries[semantic_relative]['source_sha256'],
            'public_sha256': entries[semantic_relative]['public_sha256']},
        'delivery_channels': {'browser_paths': browser + [NAMESPACE + '/PUBLIC_CONTENT_REFS.json'],
            'archive_paths': archive_paths, 'browser_max_file_bytes': BROWSER_LIMIT,
            'browser_bytes_excluding_self_index': sum(x['git_lf']['bytes'] for x in entries.values() if x['expected_in_git']),
            'archive_uncompressed_bytes_excluding_self_index': sum(x['public_bytes'] for x in entries.values()),
            'archive_only_files': sum(not x['expected_in_git'] for x in entries.values())},
        'git_stage_plan': {'force_paths': browser, 'configs_data_ignore_rule_must_not_hide_required_files': True},
        'private_tensor_payloads_archived': False, 'GitHub_mutation': False}
    index_path = public / 'PUBLIC_CONTENT_REFS.json'
    _write_new(index_path, index)
    require(index_path.stat().st_size <= BROWSER_LIMIT, 'R13 index exceeds16MiB; explicit sharding required')
    _write_new(target / 'PRIVATE_EXPORT_AUDIT.json', {'status': 'BLOCKED_UNSAFE_R13_EXPORT' if unsafe else 'PASS_LOCAL_R13_EXPORT',
        'unsafe': unsafe, 'selected_files': len(source.files), 'GitHub_mutation': False})
    require(not unsafe, 'R13 unsafe export; preserve failure stage and inspect audit')
    require(sha(scope_path) == scope_sha, 'R13 sealed scope changed during export')
    archive = target / (stage + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for relative in sorted(entries):
            z.write(public / relative, NAMESPACE + '/' + relative)
        z.write(index_path, NAMESPACE + '/PUBLIC_CONTENT_REFS.json')
    cold = verify(public, archive, semantic_callback=callback)
    _write_new(target / 'LOCAL_COLD_VERIFICATION.json', cold)
    portable = {rel: {'source_sha256': source.files[rel]['source_sha256'],
        'public_sha256': entries[rel]['public_sha256']} for rel in (*PORTABLE_CORE, semantic_relative)}
    delivery = {'schema': 'trajkv.round13.public_delivery.v1', 'status': 'PREPARED_NOT_PUBLISHED',
        'stage': stage, 'namespace': NAMESPACE, 'release_tag': tag, 'fixture_only': False,
        'publication_eligible': True, 'archive': archive.name, 'archive_bytes': archive.stat().st_size,
        'archive_sha256': sha(archive), 'content_index': 'tree/' + NAMESPACE + '/PUBLIC_CONTENT_REFS.json',
        'content_index_sha256': sha(index_path), 'phase': closure, 'cold_verification_status': cold['status'],
        'portable_verifier_sha256': portable, 'expected_git_payload_files': cold['expected_Git_payload_files'],
        'delivery_channels': {**index['delivery_channels'], 'browser_index_bytes': index_path.stat().st_size,
            'archive_member_count': len(entries) + 1},
        'preserve_prior_namespace_release_tag_history': True, 'GitHub_mutation': False}
    _write_new(target / 'DELIVERY.json', delivery)
    return delivery


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--semantic-file', required=True)
    p.add_argument('--semantic-function', default='verify_semantics')
    a = p.parse_args()
    result = export(a.scope, a.stage, a.tag,
        semantic_verifier={'relative_path': a.semantic_file, 'function': a.semantic_function})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
