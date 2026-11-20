"""Serial GPU lanes for immutable jobs. Never retries or restarts a worker.

The caller supplies the separately SHA-locked original launcher source. This
supervisor only schedules jobs; all inference remains in the declared runtime.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def observed(receipt):
    p = Path('/proc') / str(receipt['pid'])
    if not p.exists(): return 'gone'
    state = (p / 'stat').read_text().rsplit(')', 1)[1].strip().split()[0]
    if state == 'Z': return 'gone'
    actual = [v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v]
    if actual != receipt['argv_executed']: return 'gone_pid_reused'
    return 'live'


def output_complete(receipt):
    argv = receipt['argv_executed']
    output = Path(argv[argv.index('--output') + 1])
    complete = output / 'complete.json'
    if not complete.exists(): return {'status': 'PROCESS_EXITED_WITHOUT_COMPLETE', 'output': str(output)}
    value = read(complete)
    # The inference runner owns this marker. Validate its raw identities before
    # using it as a prerequisite; do not infer completion from a log tail.
    for name, key in [('manifest.json', 'manifest_sha256'), ('records.jsonl', 'records_jsonl_sha256')]:
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == value[key], name
    return {key: value[key] for key in ('status', 'expected_QA', 'recorded_QA', 'counts', 'coverage_complete')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan); root = Path(plan['root']); volume = Path(plan['data_volume'])
    assert str(root.resolve()).startswith(str(volume.resolve()).rstrip('/') + '/')
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = Path(plan['launcher_source'])
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() == plan['launcher_sha256']
    leases = []
    for gpu in sorted(plan['lanes']):
        lease = (root / 'jobs' / f'queue_gpu{gpu}.lock').open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(lease)
    state = {'status': 'RUNNING', 'started_unix': time.time(), 'lanes': {
        gpu: {'next': 0, 'active': None, 'completed': [], 'status': 'RUNNING'} for gpu in plan['lanes']}}
    def flush():
        temporary = args.output / 'state.json.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        os.replace(temporary, args.output / 'state.json')
    for gpu, initial in plan.get('wait_for_existing', {}).items():
        receipt = read(root / 'jobs' / (initial + '.launch.json'))
        assert str(receipt['CUDA_VISIBLE_DEVICES']) == str(gpu)
        state['lanes'][gpu]['active'] = {'name': initial, 'initial_external': True, 'receipt': receipt}
    flush()
    while any(lane['status'] == 'RUNNING' for lane in state['lanes'].values()):
        for gpu, lane in state['lanes'].items():
            if lane['status'] != 'RUNNING': continue
            active = lane['active']
            if active:
                if observed(active['receipt']) == 'live': continue
                complete = output_complete(active['receipt'])
                lane['completed'].append({'name': active['name'], 'initial_external': active.get('initial_external', False),
                                          'observed_complete': complete, 'observed_unix': time.time()})
                lane['active'] = None
                if complete['status'] != 'PASS_RAW_COMPLETE':
                    lane.update(status='STOPPED_ON_ACTUAL_JOB_FAILURE', failure=complete)
                    flush(); continue
            if lane['next'] >= len(plan['lanes'][gpu]):
                lane['status'] = 'COMPLETE'; flush(); continue
            item = plan['lanes'][gpu][lane['next']]
            config_path = root / item['job']
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == item['job_sha256']
            job = read(config_path)
            assert str(job['env']['CUDA_VISIBLE_DEVICES']) == str(gpu)
            prerequisites = item.get('require_pass', [])
            if any(not (root / condition).exists() for condition in prerequisites): continue
            failed_gate = next((condition for condition in prerequisites
                                if read(root / condition)['status'] != 'PASS_RAW_COMPLETE'), None)
            if failed_gate:
                lane.update(status='STOPPED_ON_ACTUAL_PREREQUISITE_FAILURE', failure=failed_gate)
                flush(); continue
            # This is the original source-snapshot launcher with a fixed SHA,
            # not an alternate inference implementation or a fallback backend.
            run = subprocess.run(['/usr/bin/python3', '-B', str(launcher), str(root), str(config_path), str(volume)],
                                 capture_output=True, text=True, check=True)
            receipt = json.loads(run.stdout)
            lane['active'] = {'name': job['name'], 'receipt': receipt}
            lane['next'] += 1
            print(json.dumps({'gpu': gpu, 'job': job['name'], 'pid': receipt['pid'], 'status': 'SCHEDULED'}), flush=True)
            flush()
        if any(lane['status'] == 'RUNNING' for lane in state['lanes'].values()): time.sleep(10)
    state['status'] = 'PASS_QUEUE_COMPLETE' if all(lane['status'] == 'COMPLETE' for lane in state['lanes'].values()) else 'QUEUE_COMPLETE_WITH_FAILURES'
    state['finished_unix'] = time.time(); flush()
    with (args.output / 'complete.json').open('x', encoding='utf-8') as stream: json.dump(state, stream, indent=2)
    print(json.dumps({'status': state['status']}), flush=True)


if __name__ == '__main__': main()
