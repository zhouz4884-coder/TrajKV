"""Serial, immutable-source GPU cells; the parent records every child result."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(path, value):
    temporary = path.with_suffix('.writing')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = Path(os.environ['BASELINE_DATA_ROOT']).resolve()
    source = Path(os.environ['BASELINE_SOURCE_ROOT']).resolve()
    assert a.output.resolve().is_relative_to(data)
    plan_sha = sha(a.config)
    plan = json.loads(a.config.read_text(encoding='utf-8-sig'))
    assert plan['schema'] == 'trajkv.round10.sequence.v1' and plan['entries']
    names = [x['name'] for x in plan['entries']]
    assert len(names) == len(set(names))
    assert len({x['output'] for x in plan['entries']}) == len(names)
    a.output.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'trajkv.round10.sequence_execution.v1',
              'status': 'RUNNING', 'plan_sha256': plan_sha,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'source_root': str(source), 'started_unix': time.time(),
              'expected_cells': len(names), 'completed_cells': [], 'active': None}
    state(a.output / 'state.json', record)
    try:
        for item in plan['entries']:
            assert sha(a.config) == plan_sha
            relative = PurePosixPath(item['script'])
            assert (not relative.is_absolute() and '..' not in relative.parts and
                    relative.parts[0] == 'code' and relative.suffix == '.py')
            script = source.joinpath(*relative.parts)
            assert sha(script) == item['script_sha256']
            config, output = Path(item['config']), Path(item['output'])
            assert config.resolve().is_relative_to(data)
            assert output.resolve().is_relative_to(data) and not output.exists()
            assert sha(config) == item['config_sha256']
            log = a.output / (item['name'] + '.log')
            argv = [sys.executable, '-B', str(script), '--config', str(config),
                    '--output', str(output)]
            begun = time.time()
            with log.open('xb', buffering=0) as handle:
                child = subprocess.Popen(argv, cwd=source, env=os.environ.copy(),
                                         stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT)
                record['active'] = {'name': item['name'], 'pid': child.pid,
                                    'argv': argv, 'started_unix': begun,
                                    'config_sha256': item['config_sha256']}
                state(a.output / 'state.json', record)
                print(json.dumps({'event': 'START_CELL', **record['active']}), flush=True)
                code = child.wait()
            result = {'name': item['name'], 'exit_code': code,
                      'wall_seconds': time.time() - begun,
                      'output': str(output), 'log': str(log)}
            assert sha(config) == item['config_sha256']
            if code != 0:
                record['failed_cell'] = result
                raise RuntimeError('Child failed; original output preserved: ' + item['name'])
            closed = json.loads((output / 'complete.json').read_text(encoding='utf-8'))
            assert closed['status'] == 'PASS_RAW_COMPLETE'
            assert closed['coverage_complete'] is True
            assert closed['expected_QA'] == closed['recorded_QA'] == item['expected_QA']
            result['recorded_QA'] = closed['recorded_QA']
            result['complete_sha256'] = sha(output / 'complete.json')
            record['completed_cells'].append(result)
            record['active'] = None
            state(a.output / 'state.json', record)
            print(json.dumps({'event': 'COMPLETE_CELL', **result}), flush=True)
        record['status'] = 'PASS_SEQUENCE_COMPLETE'
    except Exception as exc:
        record['status'] = 'FAILED_SEQUENCE_PRESERVED'
        record['error_type'] = type(exc).__name__
        record['error'] = str(exc)
        raise
    finally:
        record['finished_unix'] = time.time()
        state(a.output / 'state.json', record)
        with (a.output / 'complete.json').open('x', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
