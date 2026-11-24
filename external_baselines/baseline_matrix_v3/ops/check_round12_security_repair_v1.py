"""Local exact-source scanner and pinned-cache checks; no research inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from round12_public_security_v1 import findings, file_findings, reviewed_scanner, SCANNER_RELATIVE, SCANNER_SHA256
from export_public_round12_v2 import PinnedReuse, exact_scanner
from round12_public_contract_v1 import ROOT, EVIDENCE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reuse-stage', required=True, type=Path)
    p.add_argument('--reuse-config', required=True, type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    original = (ROOT / SCANNER_RELATIVE).read_bytes()
    checks = []
    assert hashlib.sha256(original).hexdigest() == SCANNER_SHA256
    assert findings(SCANNER_RELATIVE, original) == []
    checks.append('exact_reviewed_source_only_regex_self_matches_masked')
    assert findings('ops/renamed_scanner.py', original)
    assert findings(SCANNER_RELATIVE, original + b'\n')
    checks.extend(['same_body_wrong_relative_not_exempt', 'changed_body_same_relative_not_exempt'])
    tests = {
        'literal_credential': ('api' + '_key' + '=' + chr(34) + 'x' * 30 + chr(34)).encode(),
        'ssh_endpoint': ('ssh' + ':' + '/' * 2 + 'example.invalid').encode(),
        'signed_download_url': ('?' + 'Signature' + '=' + 'example').encode(),
        'github_credential': ('gh' + 'p_' + 'X' * 30).encode(),
    }
    for label, injection in tests.items():
        assert label in findings(SCANNER_RELATIVE, original + b'\n' + injection + b'\n')
        assert label in findings('results/support.txt', injection)
        checks.append('reject_injected_' + label + '_in_scanner_and_other_file')
    spanning = ('s' + 'sh' + ' \n -p \n 2222 \n synthetic' + '@' + '10.' + '0.0.1').encode()
    spanning_path = a.output / 'synthetic_multiline.txt'
    spanning_path.write_bytes(spanning)
    assert 'ssh_endpoint' in findings('results/support.txt', spanning)
    assert 'ssh_endpoint' in file_findings('results/support.txt', spanning_path)
    checks.append('non_JSONL_cross_line_whitespace_pattern_rejected_with_original_wholebody_semantics')
    stage = a.output / 'exact_source'; stage.mkdir()
    path = ROOT / SCANNER_RELATIVE
    entry, side, bad = exact_scanner(SCANNER_RELATIVE,
        {'path': path, 'origin': 'project', 'source_path': SCANNER_RELATIVE}, stage)
    assert not bad and side is None and entry['transformation'] == 'EXACT'
    assert (stage / SCANNER_RELATIVE).read_bytes() == original
    checks.append('exact_scanner_export_preserves_all_bytes_and_source_semantics')
    reuse = PinnedReuse(a.reuse_stage, a.reuse_config)
    candidates = [(rel, item) for rel, item in reuse.old_source.items()
                  if rel.endswith('records.jsonl') and reuse.index['files'][rel]['public_bytes'] < 10 * 1024 ** 2]
    rel, item = min(candidates, key=lambda pair: reuse.index['files'][pair[0]]['public_bytes'])
    copied = reuse.copy(rel, item, a.output / 'reused')
    assert copied and copied[0]['source_sha256'] == item['sha256'] and not copied[2]
    checks.append('actual_cached_raw_and_line_identity_bytes_verified_and_copied')
    assert reuse.copy(SCANNER_RELATIVE, reuse.old_source[SCANNER_RELATIVE], a.output / 'never') is None
    checks.append('corrupted_cached_scanner_forbidden_even_when_original_SHA_matches')
    result = {'status': 'PASS_R12_EXACT_SCANNER_AND_PINNED_REUSE_CHECKS',
        'checks': checks, 'frozen_scanner_source_sha256': SCANNER_SHA256,
        'actual_cached_raw_smoke_member': rel, 'reused_raw_is_not_new_inference': True,
        'all_injection_values_synthetic': True, 'actual_free_QA': 0, 'GitHub_mutation': False,
        'checked_utc': datetime.now(timezone.utc).isoformat()}
    (a.output / 'CHECKS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
