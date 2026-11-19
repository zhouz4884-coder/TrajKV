"""Bounded final-backend measurements through the preserved R8 wrappers.

The original diagnostic/timer code and numerical backends stay unchanged.
This outer adapter extends the explicit allowlist for the final Mean/postcal
backends, pins its own source, and binds the original final sidecar/marker.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import round8_backend_support as support


EXTRA = {'round8_mean_runner_v2': {'mean'}, 'round8_postcal_runner': {'postcal'}}


@contextmanager
def final_backend_scope():
    previous = dict(support.ALLOWED)
    support.ALLOWED = {**previous, **EXTRA}
    try:
        yield
    finally:
        support.ALLOWED = previous


def execute(kind, config_path, output, references=None, plan_only=False):
    config_path, output = Path(config_path), Path(output)
    source_before = support.sha(__file__)
    config = support.read(config_path)
    name, family = config.get('backend_module'), config.get('backend_family')
    if name not in EXTRA or family not in EXTRA[name]:
        raise ValueError('This final adapter accepts only the named Mean v2/postcal backends')
    if kind not in ('continuous', 'timing'):
        raise ValueError('Choose continuous or timing')
    marker_name = 'complete_backend_diagnostic.json' if kind == 'continuous' else 'complete_backend_timing.json'
    inner_name = 'round8_backend_continuous' if kind == 'continuous' else 'round8_backend_timing'
    inner = importlib.import_module(inner_name)
    engine_argv = [inner_name, '--config', str(config_path), '--output', str(output)]
    if references is not None:
        if kind != 'continuous':
            raise ValueError('Timing never consumes GT references')
        engine_argv += ['--references', str(references)]
    if plan_only:
        engine_argv += ['--plan-only']
    previous_argv = sys.argv
    try:
        sys.argv = engine_argv
        with final_backend_scope():
            if kind == 'continuous':
                result = inner.execute(config_path, output, references, plan_only)
            else:
                result = inner.execute(config_path, output, plan_only)
    finally:
        sys.argv = previous_argv
    if support.sha(__file__) != source_before:
        raise RuntimeError('Final measurement adapter changed during execution')
    if plan_only:
        return result
    inner_marker = output / marker_name
    if support.read(inner_marker).get('status') != 'COMPLETE':
        raise RuntimeError('Original measurement wrapper did not complete')
    final = dict(status='COMPLETE', schema='trajkv.round8.final-measurement-adapter.v1',
        kind=kind, backend_module=name, backend_family=family,
        adapter_sha256=source_before, config_sha256=support.sha(config_path),
        preserved_wrapper_sha256=support.sha(inner.__file__),
        preserved_support_sha256=support.sha(support.__file__),
        original_final_marker=marker_name, original_final_marker_sha256=support.sha(inner_marker),
        numerical_backend_not_modified=True, original_engine_not_modified=True,
        primary_quality_eligible=False, task_accuracy_eligible=False,
        end_to_end_timing=kind == 'timing')
    support.write_new(output / 'complete_final_measurement.json', final)
    print(json.dumps(final), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('continuous', 'timing'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    execute(args.kind, args.config, args.output, args.references, args.plan_only)


if __name__ == '__main__':
    main()
