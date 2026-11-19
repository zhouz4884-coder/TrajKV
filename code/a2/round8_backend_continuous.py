"""Selected-backend wrapper around frozen complete-Full-prefix diagnostics."""
import argparse
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path

from round8_backend_support import (read,validate_backend_config,source_snapshot,input_snapshot,
    registered_backend,complete_wrapper,wrapper_failure)


def execute(config_path,output,reference_arg=None,plan_only=False,original=None):
    config=read(config_path);module,family,declarations=validate_backend_config(config)
    if plan_only:
        original=original or importlib.import_module('round8_continuous')
        stream=io.StringIO()
        with redirect_stdout(stream):original.main()
        protocol=json.loads(stream.getvalue())
        plan=dict(status='PLAN_ONLY',backend_module=module.__name__,backend_family=family,
            candidate_subset=True,factorial_complete=False,variant_declarations=declarations,
            original_fixed_prefix_plan=protocol,
            sources=source_snapshot(module,family,__file__,'round8_continuous.py'),
            inputs=input_snapshot(config_path,config,'round8_continuous.py',reference_arg,strict=False),
            runtime_registration_not_executed=True,GPU_executed=False,primary_quality_eligible=False,end_to_end_timing=False)
        print(json.dumps(plan),flush=True);return plan
    # Preimport before registration. The original main's import is then cached.
    importlib.import_module('round8_runner')
    original=original or importlib.import_module('round8_continuous')
    before=dict(sources=source_snapshot(module,family,__file__,'round8_continuous.py'),
        inputs=input_snapshot(config_path,config,'round8_continuous.py',reference_arg,strict=True))
    after=None
    try:
        with registered_backend(module,family,config['variants']) as (_,identity):original.main()
        after=dict(sources=source_snapshot(module,family,__file__,'round8_continuous.py'),
            inputs=input_snapshot(config_path,config,'round8_continuous.py',reference_arg,strict=True))
        result=complete_wrapper(output,'diagnostic',module,family,declarations,before,after,identity,__file__)
        print(json.dumps(dict(status='COMPLETE',backend_family=family,complete_marker='complete_backend_diagnostic.json',
            primary_quality_eligible=False,end_to_end_timing=False)),flush=True)
        return result
    except BaseException as error:
        wrapper_failure(output,'diagnostic',error,before,after);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--references',type=Path);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();execute(args.config,args.output,args.references,args.plan_only)


if __name__=='__main__':main()
