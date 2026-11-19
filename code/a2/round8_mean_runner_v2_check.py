"""CPU-only mean runner config/registration/owner/final-gate fixtures."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch

import round8_mean_runner_v2 as runner
from round8_mean_correction_check import fixture,fields,bytes_owned
from round8_mean_correction import final_packed_key_mean,compensated_mean_header


def require(value,message):
    if not value:raise RuntimeError(message)


def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError):return True
    return False


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)


def check(output):
    book=output/'book.json';write(book,dict(centers=torch.ones(16,4).tolist()))
    factor=output/'factor.json';write(factor,dict(CPU_stub=True))
    sample=output/'samples.json';write(sample,[dict(id='CPU_stub')])
    priority=[[axis,layer,head,0] for layer in range(28) for head in range(4) for axis in ('k','v')]
    base=dict(method='round8_joint',cap=8,key_assignment='l2',key_amplitude='norm',key_encoding='reencode',value_encoding='binary',
        v_group_size='all',direction_book_path=str(book),metadata_reference_book_path=str(book),metadata_factor_path=str(factor),
        round8_factor_policy='inherit_from_reference_book',selected_stages=priority,res_stages_limit=43,budget_bits=1.8,
        joint_gain_refit=False,value_joint_refit=False,mean_decode_chunk=4096)
    variants=[dict(base,name='off',visual_mean_correction=False),dict(base,name='on',visual_mean_correction=True)]
    config=dict(backend_family='mean',arguments=dict(samples=str(sample),max_patches=8),variants=variants)
    declarations=runner.validate_config(config)
    require(len(declarations)==2,'off/on config rejected')
    for items in ([variants[0]],[variants[1]],[dict(name='Full',method='full')]+variants,[dict(name='old',method='round7_key')]+variants):
        require(len(runner.validate_config(dict(config,variants=items)))==len(items),'Arbitrary subset or old control rejected')
    invalid=[]
    for change in (dict(visual_mean_correction='true'),dict(joint_gain_refit=True),dict(value_joint_refit=True),
        dict(postres_direction=True),dict(final_ray_mode='norm'),dict(key_assignment='row_cosine_cd'),dict(mean_decode_chunk=0),
        dict(selected_stages=priority[:43])):
        require(rejected(lambda:runner.validate_variant(dict(variants[1],**change))),'Ignored forbidden intervention')
        invalid.append(change)
    require(rejected(lambda:runner.validate_variant(dict(name='Full',method='full',visual_mean_correction=True))), 'Full ignored true flag')
    # Actual helper chunk changes must preserve final FP16 rows/mean/header.
    chunks=[]
    for shape in ((16,4),(256,8),(32,16,4)):
        state,b=fixture(shape)
        means=[final_packed_key_mean(state.direction_code,state.direction_gain,b,state.depth_residual_lookup[0],
            state.depth_residual_code,state.depth_residual_alpha,chunk=c) for c in (1,7,256,4096)]
        require(all(torch.equal(means[0],m) for m in means),'Chunk-dependent final mean')
        headers=[compensated_mean_header(state.direction_mean,m)[0] for m in means]
        require(all(torch.equal(headers[0],h) for h in headers),'Chunk-dependent header')
        chunks.append(dict(book_shape=list(shape),chunks=[1,7,256,4096],mean_header_bitwise=True))
    class FixtureBase:
        settings={}
        def _initialize(self):
            self.states=[fixture((16,4),100+i)[0] for i in range(28)]
            self.book=fixture((16,4),100)[1]
            self.joint_gain_enabled=False;self.ranking_constant_bytes=448
            self.requested_residual_stages=priority;self.selected_residual_stages=priority[:43]
            return 'BASE'
        def owned(self):return bytes_owned(self.states,self.book)
        def result_metadata(self):return dict(CPU_stub=True)
    cls=runner.mean_class(FixtureBase);records=[];initializations=[]
    for variant in variants:
        cls.settings=variant;bank=cls();require(bank._initialize()=='BASE','super result lost')
        metadata=bank.result_metadata();integrity=metadata['mean_header_integrity']
        require(integrity['actual_owner_before']==integrity['actual_owner_after'] and integrity['fields_before']==integrity['fields_after'],
            'Nonmean fields/owner differ')
        require(metadata['mean_actual_decode_chunk']==4096 and len(metadata['mean_correction_statistics'])==(28 if variant['visual_mean_correction'] else 0),
            'Actual chunk/layer coverage mismatch')
        require(not any(isinstance(value,torch.Tensor) for value in vars(bank).values() if value is not bank.book),
            'New top-level retained tensor introduced')
        records.append(dict(metadata,session_id='CPU_stub',variant=variant['name'],variant_config=variant,method='round8_joint',visual_tokens=32))
        initializations.append(dict(enabled=variant['visual_mean_correction'],layers=28,actual_decode_chunk=4096,
            nonmean_SHA_and_pointer_unchanged=True,owner_unchanged=True,apply_count_each_enabled_layer=1 if variant['visual_mean_correction'] else 0))
    # Real context-manager code, with CPU module doubles replacing GPU imports.
    modules={name:ModuleType(name) for name in ('round8_runner','round8_bank','factory','run_representation_variants','session_boundary')}
    modules['round8_bank'].Round8Bank=FixtureBase
    previous_factory=lambda method:FixtureBase;previous_executor=lambda method:FixtureBase
    modules['factory'].bank_class=previous_factory;modules['run_representation_variants'].bank_class=previous_executor
    modules['session_boundary'].ProgressiveBank=FixtureBase;old_settings=FixtureBase.settings
    restored=[]
    for fail in (False,True):
        try:
            with patch.dict(sys.modules,modules),runner.backend_registration('mean') as selected:
                require(modules['factory'].bank_class('round8_joint') is selected and modules['run_representation_variants'].bank_class('round8_joint') is selected,
                    'Actual registered class differs')
                require(modules['factory'].bank_class('full') is FixtureBase,'Full route changed')
                selected.settings={'mutated':'selected'};FixtureBase.settings={'mutated':'base'}
                modules['session_boundary'].ProgressiveBank=selected
                if fail:raise RuntimeError('intentional CPU registration failure')
        except RuntimeError:
            if not fail:raise
        require(modules['factory'].bank_class is previous_factory and modules['run_representation_variants'].bank_class is previous_executor
            and modules['session_boundary'].ProgressiveBank is FixtureBase and FixtureBase.settings is old_settings
            and 'settings' not in selected.__dict__,'Registry/settings exception leak')
        restored.append(dict(intentional_failure=fail,factory_executor_session_settings_restored=True))
    sources=runner.source_fingerprints();cp=output/'config.json';write(cp,config);external=runner.external_fingerprints(config)
    def engine(folder,rows):
        folder.mkdir();write(folder/'complete.json',dict(status='COMPLETE',records=len(rows),expected=2))
        write(folder/'manifest.json',dict(config=config,config_sha256=runner.sha(cp),samples_sha256=runner.sha(sample),selected_ids=['CPU_stub']))
        with (folder/'records.jsonl').open('x',encoding='utf-8') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
    successful=output/'successful_stub';engine(successful,records)
    final=runner.finalize_run(successful,external,sources,runner.sha(cp),declarations)
    require(final['status']=='COMPLETE' and (successful/'complete_mean.json').is_file(),'Final gate failed')
    bad=copy.deepcopy(records);bad[1]['mean_correction_statistics'][0]['apply_count']=2
    failed=output/'failed_double_apply_stub';engine(failed,bad)
    require(rejected(lambda:runner.finalize_run(failed,external,sources,runner.sha(cp),declarations))
        and not (failed/'complete_mean.json').exists() and not (failed/'complete_round8.json').exists(),'Double apply evidence accepted')
    duplicate=output/'failed_duplicate_stub';engine(duplicate,[records[0],records[0]])
    require(rejected(lambda:runner.finalize_run(duplicate,external,sources,runner.sha(cp),declarations))
        and not (duplicate/'complete_mean.json').exists(),'Duplicate producer records accepted')
    require(rejected(lambda:runner.finalize_run(successful,external,sources,runner.sha(cp),declarations)),'Existing final marker overwritten')
    return dict(status='PASS',GPU_used=False,SSH_used=False,source_sha256=sources,arbitrary_subsets_and_Full_R7_controls=True,
        rejected_invalid_variants=invalid,chunk_parity=chunks,bank_initializations=initializations,registration=restored,
        successful_final_marker_SHA256=runner.sha(successful/'complete_mean.json'),real_producer_session_id_only_fixture=True,
        final_postcheck_tests=['source/input/config/sample/manifest/record binding','28 layer fullNv evidence','double apply rejected','duplicate records rejected','existing marker immutable'],
        limitations='CPU synthetic subclass/module/engine fixtures only; no real model/GPU/owner deployment or task metric',
        new_persistent_tensor_bytes=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','Hide CUDA for CPU-only check')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    report=check(a.output);write(a.output/'check.json',report)
    write(a.output/'complete.json',dict(status='PASS',GPU_used=False,SSH_used=False,check_sha256=runner.sha(a.output/'check.json')))
    print(json.dumps(dict(status='PASS',output=str(a.output),GPU_used=False,SSH_used=False,new_persistent_tensor_bytes=0)))


if __name__=='__main__':main()
