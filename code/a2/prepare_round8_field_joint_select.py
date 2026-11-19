"""Prepare local-only L2 Kjoint x Vjoint select factorial plus Full.

The prior select QA, old book/factors, full224 ranking and43-stage limit remain
fixed. V refit runs exactly two rounds in its existing physical slots.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re

from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path
from round8_field_joint_runner import validate_config,source_fingerprints


def require(value,message):
    if not value: raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='field_joint_select5_v1')
    a=p.parse_args(); require(re.fullmatch(r'[A-Za-z0-9_]+',a.name),'Use a new simple versioned name')
    runtime_path=ROOT/'configs/a2/round8_v1/job_joint_cd_select5_v1_doc.json'; runtime=read(runtime_path)
    source_root=runtime['round8_source_root']; python=runtime['argv'][0]
    global_root=runtime['argv'][runtime['argv'].index('--output')+1].split('/results/',1)[0]
    shared=global_root+'/cache/round8_v1/shared'
    require(source_root.startswith(global_root+'/stages/round8_') and python.startswith(global_root.rsplit('/',1)[0]+'/_runtime/'),
        'Keep isolated sourceRoot and validated data-volume P72')
    views_path=ROOT/'audit/a2/round8_exposed_evaluation_views_v1.json'; views=read(views_path)['files']
    sources=source_fingerprints()
    require(sources['round8_field_joint_runner.py']=='2781cabfedaad35b5db909e6ba0d1dcde5edd53d72dd8c13ca410fd60cb1dd09',
        'Expected root-approved final field-joint runner version')
    payloads=[]; outputs=[]; parent_hashes={}; consumed={}; images=set()
    receipt_path=ROOT/f'audit/a2/round8_{a.name}_prepared.json'
    for kind,gpu,questions in [('doc',0,38),('text',1,22)]:
        parent_path=ROOT/f'configs/a2/round8_v1/joint_cd_select5_v1_{kind}.json'; parent=read(parent_path)
        original_path=ROOT/f'configs/a2/round8_v1/postfit_select11_v2_{kind}.json'; original=read(original_path)
        parent_hashes[parent_path.relative_to(ROOT).as_posix()]=sha(parent_path)
        parent_hashes[original_path.relative_to(ROOT).as_posix()]=sha(original_path)
        by={v['name']:v for v in parent['variants']}; base=copy.deepcopy(by['r8_l2_joint0'])
        require(base['direction_book_path']==base['metadata_reference_book_path'] and base['value_encoding']=='binary'
            and base['v_group_size']=='all' and base['key_amplitude']=='norm' and base['key_encoding']=='reencode',
            'Expected same old Gaussian16/norm and binary/all frame')
        require(len(base['selected_stages'])==224 and base['res_stages_limit']==43,'Frozen priority contract changed')
        variants=[copy.deepcopy(by['full8'])]; cells=[]
        for k,v in [(False,False),(True,False),(False,True),(True,True)]:
            variant=copy.deepcopy(base)
            for key in list(variant):
                if key.startswith('row_cosine_'): variant.pop(key)
            variant.update(name='r8_kjoint'+str(int(k))+'_vjoint'+str(int(v)),key_assignment='l2',
                joint_gain_refit=k,value_joint_refit=v,value_joint_rounds=2)
            variants.append(variant); cells.append(dict(variant=variant['name'],key_joint=k,value_joint=v))
        config=copy.deepcopy(parent); config.pop('joint_cd_factorial_design',None)
        config.update(variants=variants,scope='Exposed selection QA: Full and complete L2 K-field x V-field refit factorial; no confirmation/blind claim')
        require(config['arguments']==original['arguments'] and config['count']==questions and config.get('start',0)==0,
            'Do not change original postfit selection QA/arguments')
        samples=local_path(config['arguments']['samples']); rows=read(samples)
        require(config['arguments']['samples'].startswith(global_root+'/configs/') and sha(samples)==views[kind]['samples_sha256']
            and len(rows)==questions and len({row['image_id'] for row in rows})==16 and all(row['turns']==1 for row in rows),
            'Expected exact original independent-QA selection bytes')
        require(sha(ROOT/views[kind]['references'])==views[kind]['references_sha256'],'Official old references changed')
        for row in rows:
            require(not any(key in row for key in ('answer','answers','references')),'Answers in inference stimuli')
            for path,digest in zip(row['images'],row['hashes']):
                require(sha(local_path(path))==digest,'Existing image SHA mismatch'); images.add(digest)
        declarations=validate_config(config)
        require(len(declarations)==5,'Field runner rejected complete four cells plus Full')
        config['field_joint_factorial_design']=dict(cells=cells,Full_variant='full8',L2_only=True,value_joint_rounds=2,
            new_retained_tensor_fields=0,old_full224_ranking_bytes=448,actual_stage_limit=43,
            same_arguments_and_stimuli=True,samples_sha256=sha(samples),parent_config_sha256=sha(original_path),
            scope='Native fixed-index/sign field refit in original gain/mean/scale/alpha slots; actual hard reconstruction acceptance is not a task-quality guarantee.')
        cp=ROOT/f'configs/a2/round8_v1/{a.name}_{kind}.json'; jp=ROOT/f'configs/a2/round8_v1/job_{a.name}_{kind}.json'
        job=copy.deepcopy(runtime); job.update(name='round8_'+a.name+'_'+kind,round8_source_root=source_root,cwd=source_root)
        job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
        for key,value in job['env'].items():
            if key in ('TMPDIR','TMP','TEMP','HF_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX') or 'CACHE' in key:
                require(value.startswith(shared+'/'),'Cache/temp must stay under shared project data-disk root')
        output=global_root+'/results/round8_v1/'+a.name+'_'+kind
        job['argv']=[python,source_root+'/code/a2/round8_field_joint_runner.py','--config',global_root+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
        job['prerequisites']=['Root runs actual-reader GPU checker and verifies final source/asset/config SHA before launch',
            'Root waits for idle GPUs; config/sample/output use globalR, code uses isolated sourceRoot',
            'L2/binary/all, oldGaussian16/factors/full224priority/43mask/budget/cap are unchanged; no new confirmation GT']
        for variant in variants:
            for key,value in variant.items():
                if key.endswith('_path') and isinstance(value,str): consumed[value]=sha(local_path(value))
        payloads.extend([(cp,config),(jp,job)])
        outputs.append(dict(dataset=kind,gpu=gpu,config=cp.relative_to(ROOT).as_posix(),job=jp.relative_to(ROOT).as_posix(),
            output=output,samples=samples.relative_to(ROOT).as_posix(),samples_sha256=sha(samples),references_local_only=views[kind]['references'],
            references_sha256=views[kind]['references_sha256'],sources=16,independent_questions=questions,variants=5,
            expected_records=questions*5,expected_answers=questions*5,
            launch_command='python code/launch_round8_data_disk.py '+jp.relative_to(ROOT).as_posix()))
    require(not any(path.exists() for path,_ in payloads) and not receipt_path.exists(),'Existing files preserved; use new name')
    require(source_fingerprints()==sources,'Runner/source changed during preparation')
    for path,value in payloads: new(path,value)
    for output in outputs:
        for key in ('config','job'): output[key+'_sha256']=sha(ROOT/output[key])
    receipt=dict(schema='trajkv.round8.select-field-joint-preparation.v1',status='PREPARED_NOT_EXECUTED',
        created_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),outputs=outputs,cells=cells,
        runner_contract_validated=True,runner_source_sha256_at_preparation=sources,consumed_codec_inputs_sha256=consumed,
        parent_config_sha256=parent_hashes,parent_runtime_job_sha256=sha(runtime_path),views_sha256=sha(views_path),
        independent_questions=60,total_answers_expected=300,existing_source_images=len(images),
        source_root=source_root,global_config_sample_asset_result_root=global_root,worker_python=python,shared_cache=shared,
        original224_priority_limit43_preserved=True,value_joint_rounds=2,L2_only=True,expected_retained_extra_tensor_bytes=0,
        new_book_fit_performed=False,new_source_count_claimed=0,future_confirmation_answers_opened=False,
        scope='Full and four predeclared field-refit cells on reused exposed selection QA; no task results yet and no blind-confirmation claim.',
        GPU_used=False,SSH_used=False,uploaded=False)
    new(receipt_path,receipt)
    print(json.dumps(dict(status=receipt['status'],outputs=outputs,questions=60,expected_answers=300,GPU_used=False,SSH_used=False)))


if __name__=='__main__': main()
