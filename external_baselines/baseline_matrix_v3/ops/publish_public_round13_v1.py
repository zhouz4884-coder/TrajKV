"""R13 incremental publisher using unchanged R12 Git/GitHub transport.

Local preflight is the default. Root alone invokes execute after final closure.
Remote verification reuses the reviewed R12 implementation and labels that
reuse explicitly instead of pretending it is a separately rewritten verifier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import publish_public_round12_v1 as transport
from round12_public_core_v1 import NAMESPACE, require, sha
from verify_public_round13_v1 import verify, STATUS

Git, GitHub = transport.Git, transport.GitHub
ROOT, EVIDENCE, REPOSITORY, API = transport.ROOT, transport.EVIDENCE, transport.REPOSITORY, transport.API
load, write_new, write_journal, longpath = transport.load, transport.write_new, transport.write_journal, transport.longpath
browser_entries, preserved_tree = transport.browser_entries, transport.preserved_tree
HISTORICAL_ROUND12_COMMIT = 'c310da201e92187862aa88c1421347ecbfedd4a3'


def prepare(stage, expected_base):
    stage = Path(stage).resolve()
    require(stage.is_relative_to(EVIDENCE / 'public_staging') and re.fullmatch('[0-9a-f]{40}', expected_base),
            'R13 named data-disk stage and explicit refreshed base required')
    delivery = load(stage / 'DELIVERY.json')
    require(delivery.get('schema') == 'trajkv.round13.public_delivery.v1'
            and delivery.get('status') == 'PREPARED_NOT_PUBLISHED'
            and delivery.get('publication_eligible') is True and delivery.get('fixture_only') is False
            and delivery.get('cold_verification_status') == STATUS and delivery.get('namespace') == NAMESPACE,
            'R13 nonfixture complete Root export required')
    require(re.fullmatch('[A-Za-z0-9_.-]+', delivery['release_tag'])
            and Path(delivery['archive']).name == delivery['archive'], 'R13 safe tag/archive name')
    public = longpath(stage / 'tree' / NAMESPACE)
    archive = stage / delivery['archive']
    index_path = public / 'PUBLIC_CONTENT_REFS.json'
    require(sha(archive) == delivery['archive_sha256'] and sha(index_path) == delivery['content_index_sha256'],
            'R13 prepared payload changed')
    index = load(index_path)
    require(index['schema'] == 'trajkv.round13.public_content_refs.v1'
            and index['release_tag'] == delivery['release_tag'], 'R13 delivery/index epoch differs')
    reviewed = delivery['portable_verifier_sha256']
    require(reviewed and index['semantic_verifier']['relative_path'] in reviewed, 'R13 portable semantic graph omitted')
    for relative, pin in reviewed.items():
        require(sha(ROOT / relative) == pin['source_sha256']
                and index['files'][relative]['source_sha256'] == pin['source_sha256']
                and sha(public / relative) == pin['public_sha256']
                and index['files'][relative]['public_sha256'] == pin['public_sha256'],
                'R13 reviewed original/public verifier differs: ' + relative)
    cold = verify(public, archive)
    require(cold['status'] == STATUS and cold['publication_eligible'] is True
            and cold['archive_verified'] is True and cold['actual_producer_binding_required'] is True,
            'R13 full source/score/physical/archive cold closure required')
    require(all(sha(ROOT / relative) == pin['source_sha256'] for relative, pin in reviewed.items()),
            'R13 reviewed verifier changed during cold check')
    entries = browser_entries(index, index_path)
    require(len(entries) == delivery['expected_git_payload_files'], 'R13 browser inventory count differs')
    return {'stage': stage, 'delivery': delivery, 'index': index, 'archive': archive,
        'public': public, 'index_path': index_path, 'entries': entries,
        'reviewed': reviewed, 'cold': cold, 'expected_base': expected_base}


def verify_remote(ctx, journal, github, git):
    result = transport.verify_remote(ctx, journal, github, git)
    require(result['status'] == 'PASS_GITHUB_ROUND12_INCREMENT_DELIVERED_V1', 'R13 transport verifier did not close')
    return {**result, 'status': 'PASS_GITHUB_ROUND13_INCREMENT_DELIVERED_V1',
        'transport_verifier_original_status': result['status'],
        'transport_verifier_source': 'ops/publish_public_round12_v1.py',
        'transport_verifier_sha256': sha(Path(transport.__file__)),
        'publisher_sha256': sha(__file__), 'verified_utc': datetime.now(timezone.utc).isoformat()}


def publish(ctx, checkout, receipt, *, verify_only=False):
    checkout, receipt = Path(checkout).resolve(), Path(receipt)
    require(checkout.is_relative_to(EVIDENCE) and (checkout / '.git').exists(), 'R13 existing data-disk public checkout required')
    require(not receipt.exists(), 'R13 new immutable final receipt required')
    git, github = Git(checkout), GitHub()
    try:
        user = github.identity()
        require(git('remote', 'get-url', 'origin') == REPOSITORY + '.git'
                and git('symbolic-ref', '--short', 'HEAD') == 'main', 'R13 intended origin/main checkout required')
        stage, delivery = ctx['stage'], ctx['delivery']
        journal_path = stage / 'PUBLISH_ROUND13_JOURNAL_V1.json'
        journal = load(journal_path) if journal_path.exists() else None
        if journal is None:
            require(not verify_only and not git('status', '--porcelain'), 'R13 no resume journal or unrelated checkout changes')
            tag = delivery['release_tag']
            require(tag not in github.tags(), 'R13 tag exists; never replace')
            check = github.request('GET', API + '/releases/tags/' + tag)
            require(check.status_code == 404, 'R13 release exists; never replace'); check.close()
            git('fetch', 'origin', 'main'); git('merge', '--ff-only', 'origin/main')
            prior = git('rev-parse', 'HEAD')
            require(prior == ctx['expected_base'], 'R13 refreshed expected base differs')
            git('merge-base', '--is-ancestor', HISTORICAL_ROUND12_COMMIT, prior)
            journal = {'schema': 'trajkv.round13.publish_journal.v1', 'prior_head': prior,
                'prior_tree': git.tree(prior), 'prior_releases': github.releases(), 'prior_tags': github.tags(),
                'namespace': NAMESPACE, 'release_tag': tag, 'archive_sha256': delivery['archive_sha256'],
                'index_sha256': delivery['content_index_sha256'], 'reviewed': ctx['reviewed'],
                'publisher_sha256': sha(__file__), 'transport_sha256': sha(Path(transport.__file__)),
                'commit': None, 'tree': None}
            write_journal(journal_path, journal)
        require(journal['schema'] == 'trajkv.round13.publish_journal.v1'
                and journal['prior_head'] == ctx['expected_base'] and journal['namespace'] == NAMESPACE
                and journal['release_tag'] == delivery['release_tag'] and journal['reviewed'] == ctx['reviewed']
                and journal['publisher_sha256'] == sha(__file__)
                and journal['transport_sha256'] == sha(Path(transport.__file__))
                and journal['archive_sha256'] == delivery['archive_sha256']
                and journal['index_sha256'] == delivery['content_index_sha256'], 'R13 resume epoch differs')
        if verify_only:
            require(journal['commit'] and journal['tree'], 'R13 no committed stage to verify')
        elif journal['commit'] is None:
            require(git('rev-parse', 'HEAD') == journal['prior_head'], 'R13 checkout moved before commit')
            for relative in ctx['entries']:
                source = ctx['public'] / relative[len(NAMESPACE) + 1:]
                target = longpath(checkout / relative)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            pathspec = stage / 'GIT_PATHSPEC_ROUND13.nul'
            pathspec.write_bytes(b''.join(relative.encode() + b'\0' for relative in ctx['entries']))
            git('add', '-f', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul')
            changed = {row.decode() for row in git('diff', '--cached', '--name-only', '-z', binary=True).split(b'\0') if row}
            require(changed and changed <= set(ctx['entries']), 'R13 unexpected/empty staged file set')
            staged = git.tree(); preserved_tree(journal['prior_tree'], staged); git.verify_bodies(ctx['entries'], staged)
            message = stage / 'ROUND13_COMMIT_MESSAGE.txt'
            message.write_text('Add Round13 frozen research evidence\n\n'
                'Publish the explicit Root-sealed methods, task results, complete budgets and raw-evidence archive.\n'
                'Keep reused development, free answers, calibration, QA0 and failed attempts separate.\n'
                'Preserve previous results, configs, releases, tags and history.\n', encoding='utf-8', newline='\n')
            git('-c', 'user.name=zhouz4884-coder', '-c', 'user.email=269925009+zhouz4884-coder@users.noreply.github.com',
                'commit', '--file', str(message))
            journal.update(commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
            write_journal(journal_path, journal)
        require(git('rev-parse', 'HEAD') == journal['commit'], 'R13 checkout diverged from journal')
        git.verify_bodies(ctx['entries'], git.tree(journal['commit']))
        preserved_tree(journal['prior_tree'], git.tree(journal['commit']))
        if not verify_only:
            remote = github.doc('GET', API + '/git/ref/heads/main')['object']['sha']
            require(remote in (journal['prior_head'], journal['commit']), 'R13 concurrent remote move')
            releases = github.releases()
            require(all(releases.get(k) == v for k, v in journal['prior_releases'].items()), 'R13 earlier releases changed')
            if remote != journal['commit']:
                git('push', 'origin', 'HEAD:main')
            require(github.doc('GET', API + '/git/ref/heads/main')['object']['sha'] == journal['commit'], 'R13 remote push not observed')
            response = github.request('GET', API + '/releases/tags/' + journal['release_tag'])
            if response.status_code == 404:
                response.close()
                release = github.doc('POST', API + '/releases', codes=(201,), json={
                    'tag_name': journal['release_tag'], 'target_commitish': journal['commit'],
                    'name': 'Round13 frozen research evidence', 'draft': False, 'prerelease': False,
                    'body': 'Root-sealed actual research cutoff. Counts: ' + json.dumps(ctx['cold']['counts'], sort_keys=True)
                    + '\n\nIncludes code, configs, official task scores, full physical-byte budgets and raw text/scalar evidence. '
                    'Reused development, free answers, calibration, QA0 and failed attempts are distinguished. '
                    'No original images, model weights, private tensors, full external-library completion or controlled speed claim. '
                    'Previous history, releases, tags and assets are preserved.'})
            else:
                require(response.status_code == 200, 'R13 release lookup failed')
                release = response.json(); response.close()
            require(release['target_commitish'] == journal['commit'], 'R13 existing release target differs')
            upload = release['upload_url'].split('{', 1)[0]
            existing = {row['name']: row for row in release['assets']}
            for path in (ctx['archive'], ctx['index_path'], stage / 'DELIVERY.json'):
                if path.name in existing:
                    require(existing[path.name]['size'] == path.stat().st_size, 'R13 existing asset differs; never overwrite')
                else:
                    require(github.doc('GET', API + '/git/ref/heads/main')['object']['sha'] == journal['commit'], 'R13 head changed before upload')
                    with path.open('rb') as stream:
                        github.doc('POST', upload, codes=(201,), params={'name': path.name}, data=stream,
                            headers={'Content-Type': 'application/octet-stream'}, timeout=(30, 600))
        result = verify_remote(ctx, journal, github, git)
        result.update(account=user['login'], visibility='public')
        write_new(receipt, result)
        journal['status'] = 'VERIFIED_COMPLETE'; write_journal(journal_path, journal)
        return result
    finally:
        github.client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage-dir', type=Path, required=True)
    p.add_argument('--expected-base', required=True)
    p.add_argument('--checkout', type=Path, default=EVIDENCE / 'public_checkout')
    p.add_argument('--receipt', type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--verify-remote', action='store_true')
    a = p.parse_args()
    ctx = prepare(a.stage_dir, a.expected_base)
    if not (a.execute or a.verify_remote):
        print(json.dumps({'status': 'READY_FOR_ROOT_ROUND13_EXECUTION',
            'expected_git_payload_files': len(ctx['entries']), 'archive_sha256': ctx['delivery']['archive_sha256'],
            'GitHub_mutation': False}))
        return
    require(a.receipt is not None, 'R13 explicit new final receipt path required')
    print(json.dumps(publish(ctx, a.checkout, a.receipt, verify_only=a.verify_remote), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
