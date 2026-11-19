"""CPU real-config subset/negative/source-scope/postcheck controls, no GPU."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import torch  # CPU import before sys.modules snapshots; do not reinitialize its C extension.

from round8_candidate_runner import (sha, load_backend, validate_subset_config,
    scoped_backend, backend_registration, finish_subset_metadata)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    return False


def real_subset(family, dataset):
    root=Path(__file__).resolve().parents[2]
    prefix='joint_cd_select5_v1_' if family=='spherical' else 'field_joint_select5_v1_'
    source=root/'configs/a2/round8_v1'/(prefix+dataset+'.json')
    config=json.loads(source.read_text(encoding='utf-8-sig'))
    config['backend_family']=family
    config['variants']=[v for v in config['variants'] if v['method']=='full' or (
        v['method']=='round8_joint' and (v.get('key_assignment')=='row_cosine_cd' if family=='spherical'
                                      else v.get('joint_gain_refit') is True and v.get('value_joint_refit') is True))]
    return source,config


def controls(output):
    root=Path(__file__).resolve().parents[2]
    old_paths=[root/'code/a2'/name for name in ('round8_spherical_joint_runner.py',
        'round8_field_joint_runner.py','round8_spherical_codec.py','round8_value_joint_refit.py','round8_bank.py','round8_codec.py')]
    old_hashes={p.name:sha(p) for p in old_paths}
    real_plans=[]
    for family in ('spherical','field'):
        backend=load_backend(family)
        for dataset in ('doc','text'):
            source,config=real_subset(family,dataset)
            configpath=output/(family+'_'+dataset+'_subset.json')
            configpath.write_text(json.dumps(config,indent=2),encoding='utf-8')
            pretend_output=output/(family+'_'+dataset+'_unused_engine_output')
            process=subprocess.run([sys.executable,str(Path(__file__).with_name('round8_candidate_runner.py')),
                '--config',str(configpath),'--output',str(pretend_output),'--plan-only'],
                text=True,capture_output=True,check=True)
            plan=json.loads(process.stdout)
            require(plan['candidate_subset'] is True and plan['factorial_complete'] is False
                and plan['backend_family']==family and plan['GPU_executed'] is False,'Wrong subset plan declaration')
            require(plan['source_sha256']['round8_candidate_runner.py']==sha(Path(__file__).with_name('round8_candidate_runner.py')),'Wrapper source not pinned')
            require(len(plan['variants'])==len(config['variants']) and not pretend_output.exists(),'Plan modified engine output or invented candidates')
            real_plans.append(dict(family=family,dataset=dataset,original_config=str(source.relative_to(root)),
                original_config_sha256=sha(source),subset_config_sha256=sha(configpath),
                variants=list(plan['variants']),plan_only_exit=process.returncode,
                factorial_complete=False,candidate_subset=True,GPU_executed=False))
        _,config=real_subset(family,'doc')
        original_validator,original_sources=backend.validate_config,backend.source_fingerprints
        try:
            with scoped_backend(family) as active:
                require(active.validate_config(config) and 'round8_candidate_runner.py' in active.source_fingerprints(),'Scoped functions not installed')
                raise RuntimeError('intentional source/validator scope failure')
        except RuntimeError as error:
            require(str(error)=='intentional source/validator scope failure','Wrong scoped error')
        require(backend.validate_config is original_validator and backend.source_fingerprints is original_sources,'Backend hooks did not restore')
        bad=deepcopy(config);bad.pop('backend_family')
        require(rejected(lambda:validate_subset_config(bad,family,backend)),'Missing family accepted')
        bad=deepcopy(config);bad['variants'][-1]['backend_family']='field' if family=='spherical' else 'spherical'
        require(rejected(lambda:validate_subset_config(bad,family,backend)),'Cross-family variant accepted')
        bad=deepcopy(config)
        if family=='spherical':bad['variants'][-1]['value_joint_refit']=True
        else:bad['variants'][-1]['key_assignment']='row_cosine_cd'
        require(rejected(lambda:validate_subset_config(bad,family,backend)),'Unimplemented cross-family intervention accepted')
        # Add a second format candidate if the field subset has only one.
        bad=deepcopy(config);extra=deepcopy(bad['variants'][-1]);extra['name']='mismatched_price';extra['budget_bits']=1.7;bad['variants'].append(extra)
        require(rejected(lambda:validate_subset_config(bad,family,backend)),'Mismatched candidate budget accepted')
        # Stub only import-dependent registry surfaces; numerical class factories
        # and source functions remain the real frozen family implementations.
        factory,executor,bank,runner=(ModuleType(name) for name in ('factory','run_representation_variants','round8_bank','round8_runner'))
        class Base:pass
        bank.Round8Bank=Base;bank.encode_key=lambda *args,**kwargs:None
        previous_factory=lambda method:('original',method)
        previous_executor=lambda method:('executor-original',method)
        factory.bank_class,executor.bank_class=previous_factory,previous_executor
        replacements={module.__name__:module for module in (factory,executor,bank,runner)}
        try:
            with patch.dict(sys.modules,replacements):
                with backend_registration(family) as cls:
                    require(issubclass(cls,Base) and factory.bank_class('round8_joint') is cls
                        and executor.bank_class('round8_joint') is cls,'Wrong diagnostic class registration')
                    require(factory.bank_class('full')==('original','full'),'Baseline dispatch changed')
                    require('round8_candidate_runner.py' in backend.source_fingerprints(),'Diagnostic class scope lacks wrapper source')
                    raise RuntimeError('intentional registration failure')
        except RuntimeError as error:
            require(str(error)=='intentional registration failure','Wrong registration error')
        require(factory.bank_class is previous_factory and executor.bank_class is previous_executor
            and backend.validate_config is original_validator and backend.source_fingerprints is original_sources,'Registration/source scope did not restore')
        with scoped_backend(family) as active:
            sources=active.source_fingerprints();declarations=active.validate_config(config)
            fixture=output/(family+'_postcheck_stub');fixture.mkdir()
            (fixture/'manifest.json').write_text(json.dumps(dict(config=config,config_sha256='CPU-config',external_codec_inputs={})),encoding='utf-8')
            (fixture/'complete.json').write_text(json.dumps(dict(status='COMPLETE',records=2,expected=2)),encoding='utf-8')
            (fixture/'records.jsonl').write_text('{"CPU_stub":0}\n{"CPU_stub":1}\n',encoding='utf-8')
            raw_sha=sha(fixture/'records.jsonl')
            active.finalize_run(fixture,{},sources,'CPU-config',declarations)
            marker=finish_subset_metadata(fixture,family,sources)
            manifest=json.loads((fixture/'manifest.json').read_text())
            require(marker['candidate_subset'] and marker['factorial_complete'] is False and manifest['factorial_complete'] is False,'Final subset declaration false')
            require(marker['complete_round8_sha256']==sha(fixture/'complete_round8.json') and marker['manifest_sha256']==sha(fixture/'manifest.json') and raw_sha==sha(fixture/'records.jsonl'),'Final candidate hash binding/raw preservation failed')
            require((fixture/'complete_candidate.json').is_file() and (fixture/'audit/backend_manifest_before_candidate.json').is_file()
                and not (fixture/'audit/manifest.json').exists() and not (fixture/'audit/complete.json').exists(),'Unsafe before-copy naming')
            # A wrong wrapper/source hook cannot leave an accepted candidate marker.
            failed=output/(family+'_bad_source_stub');failed.mkdir()
            (failed/'manifest.json').write_text(json.dumps(dict(config=config,config_sha256='CPU-config',external_codec_inputs={})),encoding='utf-8')
            (failed/'complete.json').write_text(json.dumps(dict(status='COMPLETE',records=2,expected=2)),encoding='utf-8')
            (failed/'records.jsonl').write_text('{"CPU_stub":0}\n{"CPU_stub":1}\n',encoding='utf-8')
            active.finalize_run(failed,{},sources,'CPU-config',declarations)
            wrong=dict(sources);wrong['round8_candidate_runner.py']='0'*64
            require(rejected(lambda:finish_subset_metadata(failed,family,wrong)) and not (failed/'complete_candidate.json').exists()
                and not (failed/'complete_round8.json').exists(),'Bad source retained accepted marker')
    require(rejected(lambda:load_backend('mixed')),'Unknown family accepted')
    require({p.name:sha(p) for p in old_paths}==old_hashes,'Frozen numerical/backend sources changed')
    return dict(status='PASS',schema='trajkv.round8.candidate-subset-cpu-check.v1',real_config_subset_plans=real_plans,
        original_source_sha256=old_hashes,wrapper_sha256=sha(Path(__file__).with_name('round8_candidate_runner.py')),
        checker_sha256=sha(__file__),both_source_validator_scopes_restored=True,
        both_diagnostic_registration_scopes_restored=True,old_numerical_sources_unchanged=True,
        both_backend_postchecks_rebound=True,complete_candidate_written_last=True,
        before_audit_names_noncanonical=True,raw_records_unchanged=True,negative_source_does_not_leave_complete_marker=True,
        validation_negative_controls=9,source_negative_controls=2,GPU_used=False,SSH_used=False,
        scope='CPU real-config selection/metadata/source-scope controls only; no new codec/model/quality/timing measurement')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=controls(args.output)
    (args.output/'check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_used=False,real_subset_plans=4,
        report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',real_subset_plans=4,GPU_used=False)))


if __name__=='__main__':main()
