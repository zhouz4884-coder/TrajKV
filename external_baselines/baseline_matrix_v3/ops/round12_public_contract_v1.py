"""Explicit local R12 cutoff validation and Root-invoked immutable seal."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from export_public import Sanitizer
from round12_public_core_v1 import (SCHEMA, NAMESPACE, LocalSource, canonical,
    check_header, payload, require, sha, verify_semantics)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('${EVIDENCE_ROOT}').resolve()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def validate(scope, *, require_seal=False, allow_fixture=False, require_producer=True):
    fixture = check_header(scope, sealed=require_seal, allow_fixture=allow_fixture)
    source = LocalSource(scope, (ROOT, EVIDENCE))
    if not fixture:
        roles = {x['role'] for x in scope['files']}
        require({'report', 'quality_table'} <= roles and scope.get('report_summary'),
                'final science report, browsable quality table and recomputable report summary required')
        for rel in ('ops/round12_public_core_v1.py', 'ops/verify_public_round12_v1.py'):
            source.path(rel)
    if require_seal:
        seal = scope['root_seal']
        require(seal['source_payload_sha256'] == canonical(payload(scope)), 'Root original cutoff seal differs')
        public = Sanitizer().tree(payload(scope))
        require(seal['public_payload_sha256'] == canonical(public), 'Root sanitized cutoff seal differs')
    return source, verify_semantics(scope, source, allow_fixture=allow_fixture,
                                    require_producer=require_producer)


def seal_scope(scope_path, output, *, allow_fixture=False):
    scope_path, output = Path(scope_path), Path(output)
    original = json.loads(scope_path.read_text(encoding='utf-8-sig'))
    require(original.get('stage_state') == 'READY_FOR_ROOT_SEAL' and original.get('root_seal') is None,
            'Root must explicitly mark final actual scope READY_FOR_ROOT_SEAL')
    scope = deepcopy(original)
    scope['stage_state'] = 'ROOT_SEALED'
    scope['root_seal'] = None
    _, result = validate(scope, allow_fixture=allow_fixture)
    source_payload = payload(scope)
    public_payload = Sanitizer().tree(source_payload)
    scope['root_seal'] = {
        'schema': 'trajkv.round12.root_seal.v1',
        'source_payload_sha256': canonical(source_payload),
        'public_payload_sha256': canonical(public_payload),
        'original_unsealed_scope_sha256': sha(scope_path),
        'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'actual_deep_validation_status': result['status'],
        'fixture_only': scope.get('fixture_only') is True,
    }
    validate(scope, require_seal=True, allow_fixture=allow_fixture)
    write_new(output, scope)
    return {'status': 'SEALED_FIXTURE_NOT_PUBLICATION_ELIGIBLE' if scope.get('fixture_only') else
            'PASS_R12_ROOT_SEALED_ACTUAL_CUTOFF', 'sealed_scope': str(output),
            'sealed_scope_sha256': sha(output), 'counts': result['counts'],
            'publication_eligible': not scope.get('fixture_only', False),
            'GitHub_mutation': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--seal-output', type=Path)
    p.add_argument('--allow-fixture', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.seal_output:
        result = seal_scope(a.scope, a.seal_output, allow_fixture=a.allow_fixture)
    else:
        scope = json.loads(a.scope.read_text(encoding='utf-8-sig'))
        _, result = validate(scope, require_seal=scope.get('stage_state') == 'ROOT_SEALED',
                             allow_fixture=a.allow_fixture)
    if a.output:
        write_new(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
