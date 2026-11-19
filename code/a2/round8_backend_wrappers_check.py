"""CPU plans/stub dispatch/restoration/immutable sidecar controls; no model."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import torch  # Load once before sys.modules test snapshots.
import round8_spherical_codec,round8_value_joint_refit,round8_postres_direction_v2,round8_final_ray
import round8_backend_continuous as diagnostic
import round8_backend_timing as timing
from round8_backend_support import read,sha,validate_backend_config,input_snapshot


def require(value,message):
    if not value:raise RuntimeError(message)


def rejected(call):
    try:call()
    except (ValueError,RuntimeError,FileNotFoundError):return True
    return False


def fixture_inputs(output):
    folder=output/'inputs';folder.mkdir()
    model=folder/'model';model.mkdir();(model/'config.json').write_text('{"CPU_fixture":true}',encoding='utf-8')
    legacy=folder/'legacy';legacy.mkdir();(legacy/'fixture.py').write_text('# CPU fixture source\n',encoding='utf-8')
    packed=folder/'packed.py';packed.write_text('# CPU upstream fixture\n',encoding='utf-8')
    book=folder/'book.json';book.write_text(json.dumps(dict(centers=torch.ones(16,4).tolist())),encoding='utf-8')
    reference=folder/'reference_book.json';reference.write_bytes(book.read_bytes())
    factor=folder/'factor.json';factor.write_text('{"CPU_fixture":true}',encoding='utf-8')
    rows=[]
    for index in range(3):
        image=folder/f'image_stub{index}.dat';image.write_bytes(b'CPU metadata-only image stub '+str(index).encode())
        rows.append(dict(id=f'CPU{index}',image_id=f'image{index}',source_image_id=f'source{index}',dataset='doc',
            images=[str(image)],remote_images=[str(image)],questions=['Synthetic question; no actual inference'],content=[dict(type='image')]))
    samples=folder/'samples.json';samples.write_text(json.dumps(rows),encoding='utf-8')
    refs=folder/'refs.json';refs.write_text(json.dumps([dict(id=r['id'],answers=['Synthetic target']) for r in rows]),encoding='utf-8')
    priority=[[axis,layer,head,0] for layer in range(28) for head in range(4) for axis in ('k','v')]
    candidate=dict(name='selected_backend',method='round8_joint',cap=8,key_assignment='l2',key_amplitude='norm',key_encoding='reencode',
        value_encoding='binary',v_group_size='all',direction_book_path=str(book),metadata_reference_book_path=str(reference),metadata_factor_path=str(factor),
        round8_factor_policy='inherit_from_reference_book',selected_stages=priority,res_stages_limit=43,budget_bits=1.8,joint_gain_refit=False)
    args=dict(samples=str(samples),model=str(model),legacy_scripts=str(legacy),kivi_source=str(packed),max_patches=8,turns=1)
    return candidate,args,refs,reference


def make_config(output,family,kind,candidate,args):
    pairs={'spherical':'round8_candidate_runner','field':'round8_candidate_runner','postres':'round8_postres_runner','final_ray':'round8_final_ray_runner'}
    c=deepcopy(candidate)
    if family=='spherical':c.update(key_assignment='row_cosine_cd',joint_gain_refit=True,row_cosine_sweeps=1,row_cosine_hard_accept=True)
    elif family=='field':c.update(joint_gain_refit=True,value_joint_refit=True,value_joint_rounds=2)
    elif family=='postres':c.update(postres_direction=True,postres_row_chunk=4096)
    else:c.update(final_ray_mode='norm')
    config=dict(backend_module=pairs[family],backend_family=family,arguments=deepcopy(args),variants=[c])
    if kind=='diagnostic':config.update(count=2,balance_datasets=False,questions_per_source=1,target_tokens=8)
    else:
        config['variants']=[dict(name='Full',method='full',cap=8),dict(name='native4',method='native4',cap=8),
            dict(name='base2',method='base2',cap=8),dict(name='oldK1',method='round7_key',cap=8),dict(name='oldK2',method='round7_key',cap=8),c]
        config.update(images=3,warmups=1,repeats=3,tokens_per_turn=8)
        contract=output/(family+'_quality_contract.json');contract.write_text(json.dumps(dict(variants=config['variants'])),encoding='utf-8')
        config['frozen_variant_contract']=dict(path=str(contract),sha256=sha(contract))
    return config


def modules_fixture():
    names=('round8_runner','round7_runner','round6_runner','round8_bank','factory','run_representation_variants','session_boundary')
    modules={name:ModuleType(name) for name in names}
    class DefaultBank:settings={'original':'default'}
    class OldBank:settings={'original':'old'}
    modules['round8_bank'].Round8Bank=DefaultBank;modules['round8_bank'].encode_key=lambda *a,**k:None
    old_factory=lambda method:OldBank
    old_executor=lambda method:OldBank
    modules['factory'].bank_class=old_factory;modules['run_representation_variants'].bank_class=old_executor
    modules['session_boundary'].ProgressiveBank=OldBank
    return modules,DefaultBank,OldBank,old_factory,old_executor


def stub_original_main(config,output,kind,modules,default_bank,old_bank,probe,fail=False,mutate=None,config_path=None,references=None):
    # This is the same late-binding/cached-import shape as both frozen mains.
    import round8_runner,round7_runner  # noqa: F401
    from factory import bank_class
    selected=bank_class('round8_joint')
    require(selected is not default_bank and issubclass(selected,default_bank),'Original main bound default bank, ignoring flags')
    probe.append(selected)
    selected.settings={'temporarily':'changed'};old_bank.settings={'temporarily':'old-changed'}
    modules['session_boundary'].ProgressiveBank=selected
    output.mkdir(parents=True,exist_ok=False)
    if fail:raise RuntimeError('intentional original-main failure')
    consumed=dict(config_sha256=sha(config_path),samples_sha256=sha(config['arguments']['samples']),
        model_config_sha256=sha(Path(config['arguments']['model'])/'config.json'),references_sha256=sha(references) if references else None)
    files={'manifest.json':dict(CPU_stub=True,config=config,**consumed),'complete.json':dict(status='COMPLETE',CPU_stub=True),
        'summary.json':dict(status='COMPLETE',CPU_stub=True)}
    if kind=='diagnostic':
        files.update({'forced_controls.json':[],'receipt.json':dict(status='COMPLETE',CPU_stub=True)})
        raw=('records.jsonl','passes.jsonl','teacher_references.jsonl')
    else:
        files.update({'schedule.json':[],'length_controls.json':{}});raw=('timing_records.jsonl','quiet_boundaries.jsonl')
    for name,value in files.items():(output/name).write_text(json.dumps(value),encoding='utf-8')
    for name in raw:(output/name).write_text('{"CPU_stub":true}\n',encoding='utf-8')
    probe.append({p.name:sha(p) for p in output.iterdir() if p.is_file()})
    if mutate:mutate.write_text('{"changed_in_CPU_stub":true}',encoding='utf-8')


def controls(output):
    candidate,args,refs,reference=fixture_inputs(output);plans=[];stubs=[];failures=[]
    old_source_names=('round8_continuous.py','round7_timing.py','round8_candidate_runner.py','round8_spherical_joint_runner.py',
        'round8_field_joint_runner.py','round8_postres_runner.py','round8_final_ray_runner.py','round8_bank.py','round8_codec.py')
    old_hashes={name:sha(Path(__file__).with_name(name)) for name in old_source_names}
    configs={}
    for family in ('spherical','field','postres','final_ray'):
        for kind,wrapper in (('diagnostic',diagnostic),('timing',timing)):
            config=make_config(output,family,kind,candidate,args);configs[family,kind]=config
            cp=output/(family+'_'+kind+'.json');cp.write_text(json.dumps(config),encoding='utf-8')
            command=[sys.executable,str(Path(wrapper.__file__)),'--config',str(cp),'--output',str(output/(family+'_'+kind+'_unused_plan')),'--plan-only']
            if kind=='diagnostic':command += ['--references',str(refs)]
            process=subprocess.run(command,text=True,capture_output=True,check=True);plan=json.loads(process.stdout)
            require(plan['GPU_executed'] is False and plan['backend_family']==family and plan['runtime_registration_not_executed'],'Plan scope false')
            if kind=='diagnostic':require(plan['original_fixed_prefix_plan']['optional_GT_separate_passes_per_QA']==2 and plan['original_fixed_prefix_plan']['planned_outer_sessions']==10,'Original reference/forward plan was not preserved')
            else:require(plan['original_quiet_same_output_plan']['scheduled_trials']==72 and plan['instrumentation']=='none','Quiet protocol changed')
            require(not plan['inputs']['missing'],'CPU fixture plan omitted an input')
            plans.append(dict(family=family,kind=kind,exit=process.returncode,original_protocol_preserved=True))
            engine=importlib.import_module('round8_continuous' if kind=='diagnostic' else 'round7_timing')
            previous_supported=getattr(engine,'SUPPORTED',None)
            modules,base,old,previous_factory,previous_executor=modules_fixture();original_settings=old.settings
            probe=[];run=output/(family+'_'+kind+'_stub')
            body=lambda:stub_original_main(config,run,kind,modules,base,old,probe,config_path=cp,references=refs if kind=='diagnostic' else None)
            with patch.dict(sys.modules,modules),patch.object(engine,'main',body):
                result=wrapper.execute(cp,run,refs if kind=='diagnostic' else False,False,engine) if kind=='diagnostic' else wrapper.execute(cp,run,False,engine)
            require(modules['factory'].bank_class is previous_factory and modules['run_representation_variants'].bank_class is previous_executor,'Registry pointers leaked')
            require(modules['session_boundary'].ProgressiveBank is old and old.settings is original_settings and 'settings' not in probe[0].__dict__,'Settings/ProgressiveBank leaked')
            if kind=='timing':require(engine.SUPPORTED is previous_supported and 'round8_joint' not in engine.SUPPORTED,'Timer support set leaked')
            require(all(sha(run/name)==expected for name,expected in probe[1].items()),'Original artifacts were modified')
            marker='complete_backend_diagnostic.json' if kind=='diagnostic' else 'complete_backend_timing.json'
            require((run/marker).is_file() and result['original_artifacts_unchanged'] and result['accepted_only_after_this_marker'],'New completion gate missing')
            side=read(run/('backend_'+kind+'_sidecar.json'))
            require(side['before']==side['after'] and side['class_identity']['qualname']!='DefaultBank','Wrong source/class sidecar')
            stubs.append(dict(family=family,kind=kind,selected_class=side['class_identity'],original_bytes_preserved=True))
            # Actual failure inside original main must restore all scopes and
            # never produce the wrapper's accepted-complete marker.
            modules,base,old,pf,pe=modules_fixture();old_settings=old.settings;probe=[];failed=output/(family+'_'+kind+'_failed_stub')
            body=lambda:stub_original_main(config,failed,kind,modules,base,old,probe,fail=True,config_path=cp,references=refs if kind=='diagnostic' else None)
            with patch.dict(sys.modules,modules),patch.object(engine,'main',body):
                require(rejected(lambda:wrapper.execute(cp,failed,refs,False,engine) if kind=='diagnostic' else wrapper.execute(cp,failed,False,engine)),'Original failure was swallowed')
            require(modules['factory'].bank_class is pf and modules['run_representation_variants'].bank_class is pe and modules['session_boundary'].ProgressiveBank is old and old.settings is old_settings,'Failure did not restore globals')
            require(not (failed/marker).exists(),'Failure left an accepted wrapper marker')
            if kind=='timing':require(engine.SUPPORTED is previous_supported,'Failure leaked support set')
            failures.append(dict(family=family,kind=kind,restored=True,accepted_marker_absent=True))
    bad=deepcopy(configs['spherical','diagnostic']);bad['backend_module']='os'
    require(rejected(lambda:validate_backend_config(bad)),'Unknown module accepted')
    bad=deepcopy(configs['spherical','diagnostic']);bad['backend_family']='postres'
    require(rejected(lambda:validate_backend_config(bad)),'Unknown module/family pairing accepted')
    bad=deepcopy(configs['spherical','diagnostic']);bad['variants'][0]['value_joint_refit']=True
    require(rejected(lambda:validate_backend_config(bad)),'An ignored V flag was accepted')
    bad=deepcopy(configs['postres','diagnostic']);bad['variants'][0]['final_ray_mode']='norm'
    require(rejected(lambda:validate_backend_config(bad)),'An ignored ray flag was accepted')
    # Missing exact quality contract is not accepted for the new winner costs.
    engine=importlib.import_module('round7_timing');bad=deepcopy(configs['field','timing']);bad.pop('frozen_variant_contract')
    bp=output/'bad_no_contract.json';bp.write_text(json.dumps(bad),encoding='utf-8')
    require(rejected(lambda:timing.execute(bp,output/'unused_bad_contract',True,engine)),'Missing frozen variant contract accepted')
    bad=deepcopy(configs['field','timing']);bad['variants'][0]['selector']='changed'
    require(rejected(lambda:input_snapshot(bp,bad,'round7_timing.py',strict=True)),'Changed frozen quality variant accepted')
    # Detect the metadata-reference book omitted by the old timer's own list.
    config=configs['final_ray','timing'];cp=output/'final_ray_timing.json';run=output/'reference_changed_timing_stub'
    modules,base,old,pf,pe=modules_fixture();probe=[];saved=reference.read_bytes()
    body=lambda:stub_original_main(config,run,'timing',modules,base,old,probe,mutate=reference,config_path=cp)
    try:
        with patch.dict(sys.modules,modules),patch.object(engine,'main',body):
            require(rejected(lambda:timing.execute(cp,run,False,engine)),'Reference-book change was not rejected')
    finally:reference.write_bytes(saved)
    require(not (run/'complete_backend_timing.json').exists() and (run/'backend_timing_failure.json').is_file(),'Changed input retained accepted timing marker')
    require({name:sha(Path(__file__).with_name(name)) for name in old_source_names}==old_hashes,'Old sources changed')
    return dict(status='PASS',schema='trajkv.round8.backend-wrapper-cpu-check.v1',real_original_plan_cases=plans,
        stub_selected_backend_cases=stubs,exception_restore_cases=failures,validation_negative_controls=6,
        metadata_reference_before_after_negative_control=True,old_source_sha256=old_hashes,
        new_source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('round8_backend_support.py','round8_backend_continuous.py','round8_backend_timing.py','round8_backend_wrappers_check.py')},
        old_manifest_complete_records_unchanged=True,quiet_timer_SUPPORTED_restored=True,GPU_used=False,SSH_used=False,
        scope='CPU synthetic source/plan/late-factory-binding/sidecar controls; stub mains perform no model forwards and are not diagnostic/timing measurements')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=controls(args.output);(args.output/'check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_used=False,plan_cases=8,stub_cases=8,exception_cases=8,report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',plan_cases=8,stub_cases=8,exception_cases=8,GPU_used=False)),flush=True)


if __name__=='__main__':main()
