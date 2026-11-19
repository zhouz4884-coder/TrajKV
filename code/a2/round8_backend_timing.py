"""Selected-backend wrapper around frozen quiet same-output8-token timing.

No profiling instrumentation is installed. The original five/six-method,
three-source, warm1/measure3 timer and its records/manifest remain unchanged.
"""
import argparse
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path

from round8_backend_support import (read,validate_backend_config,source_snapshot,input_snapshot,
    registered_backend,supported_round8,complete_wrapper,wrapper_failure)


def execute(config_path,output,plan_only=False,original=None):
    config=read(config_path);module,family,declarations=validate_backend_config(config)
    original=original or importlib.import_module('round7_timing')
    # An explicit exact-variant contract is required for these new winner costs.
    if not config.get('frozen_variant_contract'):
        raise ValueError('Backend timing requires an explicit frozen quality-variant contract')
    with supported_round8(original):
        protocol=original.validate(config)
        if plan_only:
            stream=io.StringIO()
            with redirect_stdout(stream):original.main()
            plan=dict(status='PLAN_ONLY',backend_module=module.__name__,backend_family=family,
                candidate_subset=True,factorial_complete=False,variant_declarations=declarations,
                original_quiet_same_output_plan=json.loads(stream.getvalue()),
                sources=source_snapshot(module,family,__file__,'round7_timing.py'),
                inputs=input_snapshot(config_path,config,'round7_timing.py',strict=False),
                runtime_registration_not_executed=True,GPU_executed=False,task_accuracy_eligible=False,
                quality='not_scored',instrumentation='none')
            print(json.dumps(plan),flush=True);return plan
        importlib.import_module('round8_runner')
        before=dict(sources=source_snapshot(module,family,__file__,'round7_timing.py'),
            inputs=input_snapshot(config_path,config,'round7_timing.py',strict=True))
        after=None
        try:
            with registered_backend(module,family,config['variants']) as (_,identity):original.main()
            after=dict(sources=source_snapshot(module,family,__file__,'round7_timing.py'),
                inputs=input_snapshot(config_path,config,'round7_timing.py',strict=True))
            result=complete_wrapper(output,'timing',module,family,declarations,before,after,identity,__file__)
            print(json.dumps(dict(status='COMPLETE',backend_family=family,complete_marker='complete_backend_timing.json',
                quiet_same_output=True,scheduled_trials=protocol['scheduled_trials'],quality='not_scored')),flush=True)
            return result
        except BaseException as error:
            wrapper_failure(output,'timing',error,before,after);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();execute(args.config,args.output,args.plan_only)


if __name__=='__main__':main()
