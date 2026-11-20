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


def record_finished(lane, active, complete):
    """Default is still stop. Only explicitly optional new jobs may continue."""
    expected = active.get('completion_contract', {}).get('accepted_status', 'PASS_RAW_COMPLETE')
    policy = active.get('on_failure', 'stop')
    assert policy in ('stop', 'continue')
    row = {'name': active['name'], 'initial_external': active.get('initial_external', False),
           'observed_complete': complete, 'observed_unix': time.time(), 'on_failure': policy}
    lane['completed'].append(row)
    lane['active'] = None
    if complete['status'] != expected:
        if policy == 'continue':
            lane.setdefault('optional_failures', []).append(row)
        else:
            lane.update(status='STOPPED_ON_ACTUAL_JOB_FAILURE', failure=complete)
    return lane['status'] == 'RUNNING'


def json_prerequisite_state(root, condition, completed):
    """Missing markers wait only while their declared producer is nonterminal."""
    path = root / condition['path']
    if not path.exists():
        terminal = any(row['name'] == condition.get('producer_name') for row in completed)
        return 'failed' if terminal else 'pending'
    try:
        value = read(path)
        valid = value.get('status') == condition['accepted_status'] and all(
            value.get(key) == expected for key, expected in condition.get('required_fields', {}).items())
        return 'passed' if valid else 'failed'
    except Exception:
        return 'failed'


def existing_active(root, initial):
    descriptor = {'name': initial} if isinstance(initial, str) else initial
    assert set(descriptor) <= {'name', 'completion_contract', 'on_failure'}
    assert descriptor.get('on_failure', 'stop') in ('stop', 'continue')
    name = descriptor['name']
    receipt = read(root / 'jobs' / (name + '.launch.json'))
    return {'name': name, 'initial_external': True, 'receipt': receipt,
            'completion_contract': descriptor.get('completion_contract', {}),
            'on_failure': descriptor.get('on_failure', 'stop')}


def observed(receipt):
    p = Path('/proc') / str(receipt['pid'])
    if not p.exists(): return 'gone'
    state = (p / 'stat').read_text().rsplit(')', 1)[1].strip().split()[0]
    if state == 'Z': return 'gone'
    actual = [v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v]
    if actual != receipt['argv_executed']: return 'gone_pid_reused'
    return 'live'


def output_complete(receipt, contract=None):
    argv = receipt['argv_executed']
    output = Path(argv[argv.index('--output') + 1])
    complete = output if output.suffix == '.json' and contract and contract.get('type') == 'json_status' else output / 'complete.json'
    if not complete.exists(): return {'status': 'PROCESS_EXITED_WITHOUT_COMPLETE', 'output': str(output)}
    value = read(complete)
    if contract and contract.get('type') == 'json_status':
        status=value['status']
        fields_ok=all(value.get(key)==expected for key,expected in contract.get('required_fields', {}).items())
        for name,key in contract.get('hash_fields', {}).items():
            assert hashlib.sha256((complete.parent/name).read_bytes()).hexdigest()==value[key], name
        if status==contract['accepted_status'] and not fields_ok:status='JSON_STATUS_CONTRACT_MISMATCH'
        return {'status':status, 'required_fields_match':fields_ok,
                'report_sha256':hashlib.sha256(complete.read_bytes()).hexdigest(),
                'quality_score_or_full_matrix_cell':False}
    if contract and contract.get('type') == 'diagnostic_json':
        cases=value.get('cases', [])
        valid=(len(cases)==contract['expected_cases'] and
               all(case.get('status')=='PASS' and len(case.get('arms', []))==contract['expected_arms'] and
                   all(arm.get('status')=='PASS' for arm in case['arms']) for case in cases))
        status=value['status']
        if status==contract['accepted_status'] and not valid:status='DIAGNOSTIC_COVERAGE_MISMATCH'
        return {'status':status, 'cases':len(cases), 'all_expected_arms_executed':valid,
                'quality_score_or_full_matrix_cell':False}
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
    assert all(item.get('on_failure', 'stop') in ('stop', 'continue')
               for items in plan['lanes'].values() for item in items)
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
        active = existing_active(root, initial)
        assert str(active['receipt']['CUDA_VISIBLE_DEVICES']) == str(gpu)
        state['lanes'][gpu]['active'] = active
    flush()
    while any(lane['status'] == 'RUNNING' for lane in state['lanes'].values()):
        for gpu, lane in state['lanes'].items():
            if lane['status'] != 'RUNNING': continue
            active = lane['active']
            if active:
                if observed(active['receipt']) == 'live': continue
                try:
                    complete = output_complete(active['receipt'], active.get('completion_contract'))
                except Exception as error:
                    if active.get('on_failure') != 'continue': raise
                    complete = {'status': 'INVALID_OR_INCOMPLETE_OPTIONAL_RAW',
                                'error_type': type(error).__name__, 'error': str(error)[-1800:]}
                if not record_finished(lane, active, complete):
                    flush(); continue
            if lane['next'] >= len(plan['lanes'][gpu]):
                lane['status'] = 'COMPLETE'; flush(); continue
            item = plan['lanes'][gpu][lane['next']]
            config_path = root / item['job']
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == item['job_sha256']
            job = read(config_path)
            if item.get('execution_resource') == 'cpu':
                assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
            else:assert str(job['env']['CUDA_VISIBLE_DEVICES']) == str(gpu)
            prerequisites = item.get('require_pass', [])
            if any(not (root / condition).exists() for condition in prerequisites): continue
            json_prerequisites = item.get('require_json_status', [])
            json_states = [(condition, json_prerequisite_state(root, condition, lane['completed']))
                           for condition in json_prerequisites]
            if any(status == 'pending' for condition, status in json_states): continue
            failed_gate = next((condition for condition in prerequisites
                                if read(root / condition)['status'] != 'PASS_RAW_COMPLETE'), None)
            failed_json = next((condition['path'] for condition, status in json_states if status == 'failed'), None)
            failed_gate = failed_gate or failed_json
            if failed_gate:
                if item.get('on_failure') == 'continue':
                    lane.setdefault('optional_prerequisite_failures', []).append({
                        'job': item['job'], 'failure': failed_gate, 'observed_unix': time.time()})
                    lane['next'] += 1
                else:
                    lane.update(status='STOPPED_ON_ACTUAL_PREREQUISITE_FAILURE', failure=failed_gate)
                flush(); continue
            # This is the original source-snapshot launcher with a fixed SHA,
            # not an alternate inference implementation or a fallback backend.
            run = subprocess.run(['/usr/bin/python3', '-B', str(launcher), str(root), str(config_path), str(volume)],
                                 capture_output=True, text=True, check=True)
            receipt = json.loads(run.stdout)
            lane['active'] = {'name': job['name'], 'receipt': receipt,
                              'completion_contract':item.get('completion_contract', {}),
                              'on_failure':item.get('on_failure', 'stop')}
            lane['next'] += 1
            print(json.dumps({'gpu': gpu, 'job': job['name'], 'pid': receipt['pid'], 'status': 'SCHEDULED'}), flush=True)
            flush()
        if any(lane['status'] == 'RUNNING' for lane in state['lanes'].values()): time.sleep(10)
    state['status'] = ('PASS_QUEUE_COMPLETE' if all(lane['status'] == 'COMPLETE' and
        not lane.get('optional_failures') and not lane.get('optional_prerequisite_failures')
        for lane in state['lanes'].values()) else 'QUEUE_COMPLETE_WITH_FAILURES')
    state['finished_unix'] = time.time(); flush()
    with (args.output / 'complete.json').open('x', encoding='utf-8') as stream: json.dump(state, stream, indent=2)
    print(json.dumps({'status': state['status']}), flush=True)


if __name__ == '__main__': main()
