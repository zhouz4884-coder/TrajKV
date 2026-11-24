"""CPU/local-Git safety checks; never authenticates or contacts GitHub."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from publish_public_round12_v1 import (EVIDENCE, NAMESPACE, Git,
    browser_entries, prepare, preserved_tree, write_new)
from verify_public_round12_v1 import lf_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path,
                   default=EVIDENCE / 'results/round12/publication_preparation_v1'
                   / ('publisher_checks_' + str(time.time_ns())))
    a = p.parse_args(); out = a.output.resolve()
    if out.exists():
        raise ValueError('unique immutable check directory required')
    out.mkdir(parents=True)
    passed = []
    def rejected(label, call):
        try:
            call()
        except ValueError:
            passed.append(label)
        else:
            raise AssertionError('unsafe input accepted: ' + label)
    index_path = out / 'PUBLIC_CONTENT_REFS.json'
    index_path.write_bytes(b'{"synthetic":true}\r\n')
    browser = {'expected_in_git': True, 'git_lf': {'sha256': 'a'}}
    archive = {'expected_in_git': False, 'git_lf': {'sha256': 'b'}}
    expected = {NAMESPACE + '/configs/data/input.jsonl', NAMESPACE + '/PUBLIC_CONTENT_REFS.json'}
    index = {'files': {'configs/data/input.jsonl': browser, 'results/raw/records.jsonl': archive},
             'delivery_channels': {'browser_paths': sorted(expected)}}
    entries = browser_entries(index, index_path)
    assert set(entries) == expected and not any('records.jsonl' in x for x in entries)
    assert entries[NAMESPACE + '/PUBLIC_CONTENT_REFS.json']['git_lf'] == lf_identity(index_path)
    passed.append('archive_only_not_staged_and_generated_index_LF_bound')
    bad = deepcopy(index); bad['delivery_channels']['browser_paths'] = []
    rejected('missing_browser_inventory_rejected', lambda: browser_entries(bad, index_path))
    prior = {NAMESPACE + '/configs/old.json': 'a', NAMESPACE + '/results/old.json': 'b',
             NAMESPACE + '/code/previous.py': 'c', 'docs/old.md': 'd'}
    valid = {**prior, NAMESPACE + '/code/previous.py': 'e', NAMESPACE + '/results/round12/new.json': 'f'}
    preserved_tree(prior, valid); passed.append('new_versioned_evidence_preserves_old_frozen_tree')
    for label, path in [('frozen_config', NAMESPACE + '/configs/old.json'),
                        ('frozen_result', NAMESPACE + '/results/old.json'),
                        ('outside_namespace', 'docs/old.md')]:
        bad = {**valid, path: 'wrong'}
        rejected(label + '_replacement_rejected', lambda b=bad: preserved_tree(prior, b))
    bad = dict(valid); del bad[NAMESPACE + '/code/previous.py']
    rejected('old_tree_path_deletion_rejected', lambda: preserved_tree(prior, bad))
    fixture = EVIDENCE / 'public_staging' / ('R12_PUBLISHER_REJECT_FIXTURE_' + str(time.time_ns()))
    fixture.mkdir(parents=True)
    write_new(fixture / 'DELIVERY.json', {'schema': 'trajkv.round12.public_delivery.v1',
              'status': 'PREPARED_NOT_PUBLISHED', 'namespace': NAMESPACE,
              'fixture_only': True, 'publication_eligible': False})
    rejected('fixture_rejected_before_authentication_or_network', lambda: prepare(fixture, 'a' * 40))
    repo = out / 'local_git'; repo.mkdir()
    run = subprocess.run(['git', 'init', '--initial-branch=main', str(repo)], capture_output=True)
    assert run.returncode == 0
    git = Git(repo)
    (repo / '.gitignore').write_text('configs/data/\n', encoding='utf-8')
    (repo / '.gitattributes').write_text('* text=auto eol=lf\n', encoding='utf-8')
    ref = repo / 'configs/data/input.jsonl'; ref.parent.mkdir(parents=True)
    ref.write_bytes(b'{"id":1}\r\n')
    paths = ['.gitignore', '.gitattributes', 'configs/data/input.jsonl']
    git('add', '-f', '--', *paths)
    actual = git.tree(); wanted = {'configs/data/input.jsonl': {'git_lf': lf_identity(ref)}}
    git.verify_bodies(wanted, actual)
    passed.append('ignored_reference_force_added_and_real_Git_LF_body_verified')
    bad = deepcopy(wanted); bad['configs/data/input.jsonl']['git_lf']['sha256'] = '0' * 64
    rejected('actual_Git_body_tamper_rejected', lambda: git.verify_bodies(bad, actual))
    result = {'status': 'PASS_R12_PUBLISHER_LOCAL_SAFETY_CHECKS', 'checks': passed,
              'checks_count': len(passed), 'actual_local_Git_used': True,
              'GitHub_mutations': 0, 'network_calls': 0, 'authentication_accessed': False,
              'fixture_is_not_actual_research_or_publication': True,
              'checked_utc': datetime.now(timezone.utc).isoformat()}
    write_new(out / 'CHECKS.json', result)
    print(json.dumps({'output': str(out), **result}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
