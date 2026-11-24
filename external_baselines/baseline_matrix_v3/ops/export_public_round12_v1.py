"""Root-sealed explicit R12 export and archive; never publishes or authenticates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile
from export_public import Sanitizer, security_findings
from export_public_round9_formal_v6 import write_public_file, generated_entry, json_bytes, io_path
from round12_public_contract_v1 import ROOT, EVIDENCE, validate
from round12_public_core_v1 import NAMESPACE, canonical, payload, require, sha

STREAMING_SHA = '70a28113325e686240970956f610296a809a5157ec67ae0b8e62a50e0476a1cd'
SANITIZER_SHA = 'ae711282f004b6704d76e2f09901ef840480d505706b50bf44a3fd8a02a030ef'
BROWSER_LIMIT = 16 * 1024 ** 2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--allow-fixture', action='store_true')
    a = p.parse_args(argv)
    require(sha(ROOT / 'ops/export_public_round9_formal_v6.py') == STREAMING_SHA
            and sha(ROOT / 'ops/export_public.py') == SANITIZER_SHA,
            'reviewed frozen streaming/sanitizer dependency changed')
    require(re.fullmatch('[A-Za-z0-9_-]+', a.stage) and re.fullmatch('[A-Za-z0-9_.-]+', a.tag), 'safe unique stage/tag required')
    source_scope_sha = sha(a.scope)
    scope = json.loads(a.scope.read_text(encoding='utf-8-sig'))
    source, phase = validate(scope, require_seal=True, allow_fixture=a.allow_fixture)
    fixture = scope.get('fixture_only') is True
    target = EVIDENCE / 'public_staging' / a.stage
    require(not target.exists(), 'new unique stage required; preserve earlier exports')
    target.mkdir(parents=True)
    public = io_path(target / 'tree' / NAMESPACE)
    public.mkdir(parents=True)
    sanitizer = Sanitizer()
    entries, unsafe = {}, []
    for rel, item in sorted(source.files.items()):
        path = item['path']
        st = path.stat()
        origin = ROOT if path.is_relative_to(ROOT) else EVIDENCE
        selected = {'path': path, 'origin': 'project' if origin == ROOT else 'evidence_mirror',
                    'source_path': path.relative_to(origin).as_posix(),
                    'stat': (st.st_size, st.st_mtime_ns, st.st_ctime_ns), 'large': True}
        entry, side, findings = write_public_file(rel, selected, public, sanitizer)
        require(entry['source_sha256'] == item['sha256'], 'sealed source changed after semantic validation: ' + rel)
        entry['evidence_role'] = item['role']
        if 'expected_in_git' in item:
            entry['expected_in_git'] = bool(item['expected_in_git'])
        if item['role'] in ('inputs', 'refs', 'source_lock'):
            entry['expected_in_git'] = True
        archive_required = (Path(rel).name in ('records.jsonl', 'scored_records.jsonl')
                            or '/records/' in rel or rel.startswith('evidence/line_identities/'))
        if archive_required:
            require(item.get('expected_in_git') is not True
                    and item.get('delivery_channel') != 'BROWSER_AND_ARCHIVE',
                    'full raw/owner records must be explicitly archive-only: ' + rel)
            entry['expected_in_git'] = False
        if item.get('delivery_channel'):
            require(item['delivery_channel'] in ('BROWSER_AND_ARCHIVE', 'ARCHIVE_ONLY'), 'unknown delivery channel')
            require(item['role'] not in ('inputs', 'refs', 'source_lock')
                    or item['delivery_channel'] == 'BROWSER_AND_ARCHIVE', 'reference/input must be browser-visible')
            entry['expected_in_git'] = item['delivery_channel'] == 'BROWSER_AND_ARCHIVE'
        require(not entry['expected_in_git'] or entry['public_bytes'] <= BROWSER_LIMIT,
                'browser body exceeds16MiB; use explicit archive-only channel or reviewed compact shards: ' + rel)
        entries[rel] = entry
        if side:
            require(side not in entries, 'generated JSONL identity collision')
            entries[side] = generated_entry(public / side, False)
        if findings:
            unsafe.append({'path': rel, 'findings': findings})
    public_scope = sanitizer.tree(scope)
    require(public_scope['root_seal']['public_payload_sha256'] == canonical(payload(public_scope)), 'sanitized seal changed')
    generated = {
        'evidence/ROUND12_SCOPE.json': json_bytes(public_scope),
        'evidence/ROUND12_PHASE.json': json_bytes(sanitizer.tree(phase)),
        'README_PUBLIC.md': (
            '# Round12 actual research cutoff\n\n'
            + ('CPU fixture only; not publication eligible or real GPU evidence.\n\n' if fixture else '')
            + 'This package records only the Root-sealed actual scope. Fresh tasks, successful terminal answers, failure-status rows, pilots, reused reference views, probes, calibration and QA0 are separate.\n\n'
            + 'The archive contains explicitly selected text/scalar predictions, source proofs, configs, references, metrics, physical-owner records and report. It excludes original images, weights, private tensors/PT, connection routes, credentials and private chat. No GPU or private tensor replay is claimed.\n\n'
            + 'Cold verification: `python -B ops/verify_public_round12_v1.py --root . --archive <ZIP>` (add `--allow-fixture` only for synthetic self-tests). Python and numpy are sufficient; no network or GPU is used.\n\n'
            + '`PUBLIC_CONTENT_REFS.json` separates original source SHA, sanitized body SHA, Git LF/blob identity and fixed archive member references. The publisher must still verify actual remote commit/tree/blob bodies, downloaded release assets and preservation of earlier history. A local package is not an upload.\n'
        ).encode('utf-8'),
        '.gitattributes': b'* text=auto eol=lf\n',
    }
    for name in ('round12_public_core_v1.py', 'verify_public_round12_v1.py'):
        rel = 'ops/' + name
        if rel not in entries:
            generated[rel] = (ROOT / rel).read_bytes()
    for rel, body in generated.items():
        require(rel not in entries, 'generated public path collision: ' + rel)
        path = public / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        findings = security_findings(rel, body)
        if findings:
            unsafe.append({'path': rel, 'findings': findings})
        entries[rel] = generated_entry(path)
    asset_url = f'https://github.com/zhouz4884-coder/TrajKV/releases/download/{a.tag}/{a.stage}.zip'
    for rel, entry in entries.items():
        entry['delivery_channel'] = 'BROWSER_AND_ARCHIVE' if entry['expected_in_git'] else 'ARCHIVE_ONLY'
        entry['content_ref'] = {'archive_asset': asset_url, 'member': NAMESPACE + '/' + rel,
                                'public_body_sha256': entry['public_sha256']}
    refs = [rel for rel, entry in entries.items() if entry.get('evidence_role') in ('inputs', 'refs', 'source_lock')]
    index = {'schema': 'trajkv.round12.public_content_refs.v1', 'namespace': NAMESPACE,
             'release_tag': a.tag, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'fixture_only': fixture, 'publication_eligible': not fixture and not unsafe,
             'source_scope_sha256': source_scope_sha, 'scope_file': 'evidence/ROUND12_SCOPE.json',
             'archive_asset': asset_url, 'files': entries, 'required_reference_files': refs,
             'GitHub_mutation': False, 'private_tensor_payloads_archived': False,
             'delivery_channels': {
                 'browser_paths': [NAMESPACE + '/' + rel for rel, entry in entries.items() if entry['expected_in_git']]
                                  + [NAMESPACE + '/PUBLIC_CONTENT_REFS.json'],
                 'archive_paths': [NAMESPACE + '/' + rel for rel in entries]
                                  + [NAMESPACE + '/PUBLIC_CONTENT_REFS.json'],
                 'browser_max_file_bytes': BROWSER_LIMIT,
                 'browser_bytes_excluding_self_index': sum(entry['git_lf']['bytes'] for entry in entries.values() if entry['expected_in_git']),
                 'archive_uncompressed_bytes_excluding_self_index': sum(entry['public_bytes'] for entry in entries.values()),
                 'archive_only_files': sum(not entry['expected_in_git'] for entry in entries.values()),
                 'all_archive_members_have_public_SHA_and_fixed_content_ref': True,
                 'archive_only_is_not_omitted_or_unpublished_after_release_upload': True},
             'git_stage_plan': {'force_paths': [NAMESPACE + '/' + rel for rel, entry in entries.items() if entry['expected_in_git']],
                                'plus_generated_index': NAMESPACE + '/PUBLIC_CONTENT_REFS.json',
                                'configs_data_ignore_rule_must_not_hide_required_files': True}}
    (public / 'PUBLIC_CONTENT_REFS.json').write_bytes(json_bytes(index))
    require((public / 'PUBLIC_CONTENT_REFS.json').stat().st_size <= BROWSER_LIMIT,
            'browsable identity index exceeds16MiB; create reviewed explicit shards')
    require(sha(a.scope) == source_scope_sha, 'sealed scope file changed during export')
    audit = {'status': 'BLOCKED_UNSAFE_R12_EXPORT' if unsafe else 'PASS_LOCAL_R12_TEXT_EXPORT',
             'unsafe': unsafe, 'selected_files': len(source.files), 'phase': phase,
             'GitHub_mutation': False, 'fixture_only': fixture}
    (target / 'PRIVATE_EXPORT_AUDIT.json').write_bytes(json_bytes(audit))
    require(not unsafe, 'unsafe selected/generated public content; inspect private export audit')
    archive = target / (a.stage + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for path in sorted(public.rglob('*')):
            if path.is_file():
                z.write(path, (Path(NAMESPACE) / path.relative_to(public)).as_posix())
    from verify_public_round12_v1 import verify
    cold = verify(public, archive, allow_fixture=a.allow_fixture)
    (target / 'LOCAL_COLD_VERIFICATION.json').write_bytes(json_bytes(cold))
    delivery = {'schema': 'trajkv.round12.public_delivery.v1', 'status': 'PREPARED_NOT_PUBLISHED',
                'stage': a.stage, 'namespace': NAMESPACE, 'release_tag': a.tag,
                'fixture_only': fixture, 'publication_eligible': not fixture,
                'archive': archive.name, 'archive_bytes': archive.stat().st_size,
                'archive_sha256': sha(archive),
                'content_index': 'tree/' + NAMESPACE + '/PUBLIC_CONTENT_REFS.json',
                'content_index_sha256': sha(public / 'PUBLIC_CONTENT_REFS.json'),
                'phase': phase, 'cold_verification_status': cold['status'],
                'cold_verifier_sha256': sha(ROOT / 'ops/verify_public_round12_v1.py'),
                'core_verifier_sha256': sha(ROOT / 'ops/round12_public_core_v1.py'),
                'expected_git_payload_files': cold['expected_Git_payload_files'],
                'delivery_channels': {**index['delivery_channels'],
                    'browser_index_bytes': (public / 'PUBLIC_CONTENT_REFS.json').stat().st_size,
                    'archive_member_count': len(entries) + 1,
                    'browser_payload_bytes_with_index': index['delivery_channels']['browser_bytes_excluding_self_index']
                                                       + (public / 'PUBLIC_CONTENT_REFS.json').stat().st_size},
                'preserve_prior_namespace_release_tag_history': True,
                'expected_base_must_be_refreshed_before_actual_publication': True,
                'GitHub_mutation': False}
    (target / 'DELIVERY.json').write_bytes(json_bytes(delivery))
    print(json.dumps(delivery, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
