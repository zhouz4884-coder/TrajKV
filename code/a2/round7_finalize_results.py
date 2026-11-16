"""Collect completed Round7 runs and score their preserved free answers locally.

No model runs locally, no job is restarted, no reference answer is uploaded.
Optional finite waiting is only for the already-authorized current campaign.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from remote import connect, REMOTE_ROOT

COHORTS = ('confirm_s0', 'confirm_s1', 'second_confirm64', 'quiet3_frozen6_v1')


def collect():
    client = connect()
    try:
        sftp = client.open_sftp()
        statuses = {}
        for name in COHORTS:
            try:
                with sftp.open(REMOTE_ROOT + '/results/round7_v1/' + name + '/complete.json') as stream:
                    statuses[name] = json.load(stream)['status']
            except FileNotFoundError:
                statuses[name] = 'PENDING'
        if any(value not in ('PENDING', 'PASS', 'COMPLETE') for value in statuses.values()):
            raise RuntimeError('Explicit failed completion receipt: ' + str(statuses))
        if 'PENDING' in statuses.values():
            return statuses

        def copy_tree(remote, local):
            local.mkdir(parents=True, exist_ok=True)
            for item in sftp.listdir_attr(remote):
                source, target = remote + '/' + item.filename, local / item.filename
                if stat.S_ISDIR(item.st_mode):
                    copy_tree(source, target)
                elif target.suffix in ('.json', '.jsonl', '.csv'):
                    if not target.exists():
                        sftp.get(source, str(target))
                    elif target.stat().st_size != item.st_size:
                        raise RuntimeError('Preserve existing differing local evidence: ' + str(target))

        for name in COHORTS:
            copy_tree(REMOTE_ROOT + '/results/round7_v1/' + name, ROOT / 'results/round7_v1' / name)
        return statuses
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=14400)
    args = parser.parse_args()
    started, previous, transient_errors = time.monotonic(), None, 0
    while True:
        try:
            statuses = collect()
            transient_errors = 0
        except (OSError, EOFError):
            transient_errors += 1
            if transient_errors > 3:
                raise
            # An observation failure never restarts the remote worker.
            time.sleep(10)
            continue
        if statuses != previous:
            print(json.dumps(statuses), flush=True)
            previous = statuses
        if 'PENDING' not in statuses.values():
            break
        if not args.wait:
            return
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError('Finite observation expired; inspect existing jobs, do not restart automatically')
        time.sleep(30)
    for dataset in ('doc', 'text'):
        target = ROOT / f'results/round7_v1/confirm_{dataset}128_analysis'
        if target.exists():
            raise FileExistsError(target)
        references = ('configs/data/round7_doc_confirm128_references.jsonl' if dataset == 'doc'
                      else 'configs/data/round7_text_confirm128_references.json')
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'code/a2/analyze_round7_quality.py'),
            '--runs', 'results/round7_v1/confirm_s0/' + dataset, 'results/round7_v1/confirm_s1/' + dataset,
            '--samples', f'configs/a2/round7_v1/{dataset}_confirm128.json', '--references', references,
            '--dataset', dataset, '--output', str(target), '--expected-variants', '9',
            '--analysis-role', 'frozen-project-fresh-source-confirmation', '--quiet'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'code/a2/analyze_round7_quality.py'),
        '--runs', 'results/round7_v1/second_confirm64',
        '--samples', 'configs/a2/round7_v1/second_confirm64_first_questions.json',
        '--references', 'configs/data/round7_second_confirm64_references.jsonl', '--dataset', 'doc',
        '--output', 'results/round7_v1/second_confirm64_analysis', '--expected-variants', '3',
        '--analysis-role', 'frozen-secondary-first-question-confirmation-shared-source-subset', '--quiet'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'code/a2/analyze_round7_timing.py'),
        '--run', 'results/round7_v1/quiet3_frozen6_v1', '--output', 'results/round7_v1/quiet_timing_analysis'], cwd=ROOT, check=True)
    print(json.dumps(dict(status='ALL_FINITE_RUNS_COLLECTED_AND_SCORED', models_run_locally=False,
        reference_answers_uploaded=False, primary_sources=256, secondary_sources_subset=64)), flush=True)


if __name__ == '__main__':
    main()
