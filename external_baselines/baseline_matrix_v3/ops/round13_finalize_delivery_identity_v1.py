"""Root-authorized publication identity repair; scientific cutoff unchanged.

Clone the already Root-sealed scope into a new version. Only the R13 adapter
source identity and explicit publication/cache/failure evidence are updated.
No model, result, metric, report, source row, old scope or old stage is changed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import round13_public_adapter_v1 as adapter
from export_public import Sanitizer

BASE = Path('${EVIDENCE_ROOT}/results/round13')
OLD = BASE / 'final_scope_v2/ROUND13_SCOPE_SEALED_v1.json'
OUTPUT = BASE / 'final_scope_v3'
OLD_ADAPTER = '4769b3653d3ce5a4517fe64b7f6b278593f0e18273344b5ea136653880a46045'
NEW_ADAPTER = '9a671aefdcab1e638782a47f296cc1d3faba9021988f0ea4173a35401e7f0031'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    require = adapter.require
    require(not OUTPUT.exists(), 'preserve prior scope versions; use a new output folder')
    old = json.loads(OLD.read_text(encoding='utf-8-sig'))
    adapter.check_final_header(old)
    require(adapter.frozen.canonical(adapter.payload(old)) == old['root_seal']['source_payload_sha256'],
            'original scientific Root seal differs')
    scope = deepcopy(old)
    old_files = {item['public_path']: item for item in old['files']}
    files = {item['public_path']: item for item in scope['files']}
    relative = 'ops/round13_public_adapter_v1.py'
    require(files[relative]['sha256'] == OLD_ADAPTER and adapter.sha(ROOT / relative) == NEW_ADAPTER,
            'explicit reviewed identity-repair epoch required')
    files[relative]['sha256'] = NEW_ADAPTER
    added = []
    def add(path, public, role):
        path = Path(path)
        require(path.is_file() and public not in files, 'new publication-support member required: ' + public)
        item = {'source_path': str(path.resolve()), 'public_path': public, 'sha256': adapter.sha(path),
                'role': role, 'delivery_channel': 'BROWSER_AND_ARCHIVE', 'expected_in_git': True}
        files[public] = item
        added.append(item)
    for name in ('export_public_round13_v2.py', 'round13_export_cache_v1.py', 'round13_finalize_delivery_identity_v1.py'):
        add(ROOT / 'ops' / name, 'ops/' + name, 'code')
    prep = BASE / 'publication_preparation_v1'
    for name, role in (('ROUND13_STAGE1_REUSE_PIN_v1.json', 'support'),
                       ('EXPORT_STAGE1_FAILURE_TRACE_v1.txt', 'attempt'),
                       ('PUBLIC_V1_DATA_FIXED_SIZE_SEMANTIC_TEST_v1.json', 'technical'),
                       ('ORIGINAL_PUBLIC_SIZE_NEGATIVE_CONTROLS_v1.json', 'technical')):
        add(prep / name, 'evidence/publication_identity/' + name, role)
    add(prep / 'cache_controls_v1/CONTROLS.json',
        'evidence/publication_identity/cache_controls_v1/CONTROLS.json', 'technical')
    preserved = prep / 'previous_sources' / OLD_ADAPTER / 'round13_public_adapter_v1.py'
    require(adapter.sha(preserved) == OLD_ADAPTER, 'old failed adapter not preserved byte-exact')
    add(preserved, 'evidence/publication_identity/previous_sources/' + OLD_ADAPTER + '/round13_public_adapter_v1.py', 'code')
    identity = {'schema': 'trajkv.round13.publication_identity_repair.v1',
        'authorization': 'Root delegated the explicit user-authorized stage delivery and identity-only repair',
        'original_scientific_scope_sha256': adapter.sha(OLD),
        'original_Root_source_payload_sha256': old['root_seal']['source_payload_sha256'],
        'old_adapter_sha256': OLD_ADAPTER, 'new_adapter_sha256': NEW_ADAPTER,
        'repair': 'PublicSource uses indexed original source_bytes; LocalSource keeps actual original file size. All SHA and JSONL bindings stay required.',
        'actual_scientific_new_answers': 1396, 'scientific_result_or_report_modified': False,
        'old_failed_stage_and_ZIP_preserved': True, 'fresh_full_public_cold_required': True,
        'new_GPU_or_model_work': False, 'GitHub_mutation_by_scope_repair': False,
        'source_finalizer_sha256': adapter.sha(__file__)}
    identity_path = OUTPUT / 'PUBLICATION_IDENTITY_REPAIR_v1.json'
    write_new(identity_path, identity)
    add(identity_path, 'evidence/publication_identity/PUBLICATION_IDENTITY_REPAIR_v1.json', 'support')
    scope['files'] = list(files.values())
    for key in old:
        if key not in ('files', 'root_seal'):
            require(scope[key] == old[key], 'scientific cutoff/report/ledger changed: ' + key)
    for public, item in old_files.items():
        require(files[public] == item if public != relative else
                {key: value for key, value in files[public].items() if key != 'sha256'} ==
                {key: value for key, value in item.items() if key != 'sha256'}, 'unexpected old file entry changed')
    payload = adapter.payload(scope)
    scope['root_seal'] = {**old['root_seal'],
        'created_hkt': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        'source_payload_sha256': adapter.frozen.canonical(payload),
        'public_payload_sha256': adapter.frozen.canonical(Sanitizer().tree(payload)),
        'publication_identity_finalizer_sha256': adapter.sha(__file__),
        'scientific_Root_scope_v2_sha256': adapter.sha(OLD),
        'identity_only_repair_scientific_results_unchanged': True}
    target = OUTPUT / 'ROUND13_SCOPE_SEALED_v1.json'
    write_new(target, scope)
    receipt = {'schema': 'trajkv.round13.identity_only_reseal_receipt.v1',
        'status': 'ROOT_AUTHORIZED_IDENTITY_RESEAL_NOT_PUBLICATION',
        'old_scope': str(OLD), 'old_scope_sha256': adapter.sha(OLD),
        'new_scope': str(target), 'new_scope_sha256': adapter.sha(target),
        'source_payload_sha256': scope['root_seal']['source_payload_sha256'],
        'public_payload_sha256': scope['root_seal']['public_payload_sha256'],
        'changed_old_source_paths': [relative], 'added_publication_support': added,
        'all_scientific_fields_unchanged': True, 'actual_new_answers': 1396,
        'GitHub_mutation': False, 'publication_verified': False}
    write_new(OUTPUT / 'IDENTITY_RESEAL_RECEIPT_v1.json', receipt)
    print(json.dumps({'scope': str(target), 'sha256': receipt['new_scope_sha256'],
        'added_files': len(added), 'scientific_fields_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
