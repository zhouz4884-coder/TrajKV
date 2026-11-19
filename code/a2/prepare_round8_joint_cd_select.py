"""Prepare exposed select QA 2x2 L2/CD x joint-gain plus Full, locally only.

Uses the exact prior postfit selection samples/arguments, the old Gaussian16
book and frozen factors/full224 priority/limit43. No fitting or inference.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re

import numpy as np

from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path
from round8_spherical_joint_runner import validate_config,source_fingerprints


def require(value,message):
    if not value: raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='joint_cd_select5_v1')
    a=p.parse_args(); require(re.fullmatch(r'[A-Za-z0-9_]+',a.name),'Use a new simple versioned name')
    views_path=ROOT/'audit/a2/round8_exposed_evaluation_views_v1.json'; views=read(views_path)['files']
    runtime_path=ROOT/'configs/a2/round8_v1/job_natural_dev64_firstq3_v1_doc.json'; runtime=read(runtime_path)
    source_root=runtime['round8_source_root']; python=runtime['argv'][0]
    global_root=runtime['argv'][runtime['argv'].index('--output')+1].split('/results/',1)[0]
    shared=global_root+'/cache/round8_v1/shared'
    require(source_root.startswith(global_root+'/stages/round8_') and python.startswith(global_root.rsplit('/',1)[0]+'/_runtime/'),
        'Expected isolated source tree and validated data-volume worker')
    sources=source_fingerprints()
    payloads=[]; outputs=[]; declarations={}; consumed={}; inputs={}; image_hashes=set()
    receipt_path=ROOT/f'audit/a2/round8_{a.name}_prepared.json'
    for kind,gpu,questions in [('doc',0,38),('text',1,22)]:
        parent_path=ROOT/f'configs/a2/round8_v1/postfit_select11_v2_{kind}.json'; parent=read(parent_path)
        inputs[str(parent_path.relative_to(ROOT))]=sha(parent_path)
        by={v['name']:v for v in parent['variants']}
        base=copy.deepcopy(by['r8_k4old_vbinary'])
        require(base['method']=='round8_joint' and base['key_encoding']=='reencode' and base['key_amplitude']=='norm'
            and base['value_encoding']=='binary' and base['v_group_size']=='all','Unexpected old-book/binary-V parent')
        reference=base['metadata_reference_book_path']
        require(sha(local_path(base['direction_book_path']))==sha(local_path(reference)), 'Parent decoder differs from the frozen reference book')
        centers=np.asarray(read(local_path(reference))['centers'],dtype=np.float16)
        require(centers.shape==(16,4) and np.isfinite(centers).all(),'Expected finite old globalGaussian16x4')
        base['direction_book_path']=reference
        variants=[copy.deepcopy(by['full8'])]
        cells=[]
        for cd,joint in [(False,False),(False,True),(True,False),(True,True)]:
            variant=copy.deepcopy(base)
            for key in ('encoder_book_path','value_joint_book_path','key_metric_path','joint_gain_refit'):
                variant.pop(key,None)
            variant.update(name='r8_'+('cd1' if cd else 'l2')+'_joint'+str(int(joint)),
                key_assignment='row_cosine_cd' if cd else 'l2',joint_gain_refit=joint,
                row_cosine_sweeps=1,row_cosine_hard_accept=True,row_cosine_chunk=8192)
            require(len(variant['selected_stages'])==224 and variant['res_stages_limit']==43,
                'Keep all224 priority entries and unchanged actual43 stage prefix')
            variants.append(variant); cells.append(dict(variant=variant['name'],cd=cd,joint=joint))
        config=copy.deepcopy(parent)
        config.pop('postfit_provenance',None)
        config.update(variants=variants,scope='Previously exposed selection QA: full 2x2 L2/CD x existing joint gain plus Full; no blind-confirmation claim')
        require(config['arguments']==parent['arguments'] and config.get('start',0)==0 and config['count']==questions,
            'Do not change prior selection stimuli/arguments/count')
        samples=local_path(config['arguments']['samples']); rows=read(samples)
        require(config['arguments']['samples'].startswith(global_root+'/configs/') and sha(samples)==views[kind]['samples_sha256']
            and len(rows)==questions and len({row['image_id'] for row in rows})==16 and all(row['turns']==1 for row in rows),
            'Expected exact original independent-QA selection sample bytes')
        require(sha(ROOT/views[kind]['references'])==views[kind]['references_sha256'],'Original official reference bytes changed')
        for row in rows:
            require(not any(key in row for key in ('answer','answers','references')),'Answer leaked into inference sample')
            for path,digest in zip(row['images'],row['hashes']):
                require(sha(local_path(path))==digest,'Existing selection image changed')
                image_hashes.add(digest)
        declarations[kind]=validate_config(config)
        require(len(declarations[kind])==5,'Runner rejected complete five-arm factorial')
        config['joint_cd_factorial_design']=dict(cells=cells,Full_variant='full8',
            fixed_CD_sweeps=1,hard_FP16_base_SSE_guard=True,new_retained_tensor_fields=0,
            old_book_sha256=sha(local_path(reference)),samples_sha256=sha(samples),
            parent_config_sha256=sha(parent_path),same_arguments_and_stimuli=True,refitting_performed=False,
            scope='Guard applies to CD base before residual regeneration; existing joint-gain acceptance applies at fixed J/sign. No cross-cell task or final-residual monotonicity claim.')
        cp=ROOT/f'configs/a2/round8_v1/{a.name}_{kind}.json'
        jp=ROOT/f'configs/a2/round8_v1/job_{a.name}_{kind}.json'
        job=copy.deepcopy(runtime)
        job.update(name='round8_'+a.name+'_'+kind,round8_source_root=source_root,cwd=source_root)
        job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
        for key,value in job['env'].items():
            if key in ('TMPDIR','TMP','TEMP','HF_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX') or 'CACHE' in key:
                require(value.startswith(shared+'/'),'Every cache/temp must stay under the shared R8 data-disk cache')
        output=global_root+'/results/round8_v1/'+a.name+'_'+kind
        job['argv']=[python,source_root+'/code/a2/round8_spherical_joint_runner.py','--config',
            global_root+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
        job['prerequisites']=['Root verifies new joint-runner checker PASS and immutable source/spherical_codec SHA before launch',
            'Root waits for an idle assigned GPU; configs/samples/assets are under globalR and code alone under sourceRoot',
            'OldGaussian16/factor/224ranking43mask retained; no final fitted book or new confirmation GT required']
        for variant in variants:
            for key,value in variant.items():
                if key.endswith('_path') and isinstance(value,str): consumed[value]=sha(local_path(value))
        payloads.extend([(cp,config),(jp,job)])
        outputs.append(dict(dataset=kind,gpu=gpu,config=cp.relative_to(ROOT).as_posix(),job=jp.relative_to(ROOT).as_posix(),
            output=output,samples=str(samples.relative_to(ROOT)).replace('\\','/'),samples_sha256=sha(samples),
            references_local_only=views[kind]['references'],references_sha256=views[kind]['references_sha256'],
            sources=16,independent_questions=questions,variants=5,expected_records=questions*5,expected_answers=questions*5,
            launch_command='python code/launch_round8_data_disk.py '+jp.relative_to(ROOT).as_posix()))
    require(not any(path.exists() for path,_ in payloads) and not receipt_path.exists(),'Existing files preserved; use a new version')
    require(source_fingerprints()==sources,'Runner sources changed during preparation; retry after checker finalization')
    for path,value in payloads: new(path,value)
    for output in outputs:
        for key in ('config','job'): output[key+'_sha256']=sha(ROOT/output[key])
    receipt=dict(schema='trajkv.round8.select-CD-joint-factorial-preparation.v1',status='PREPARED_NOT_EXECUTED',
        created_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),outputs=outputs,
        cells=cells,runner_contract_validated=True,runner_source_sha256_at_preparation=sources,
        independent_questions=60,total_generated_answers_expected=300,existing_source_images=len(image_hashes),
        consumed_codec_inputs_sha256=consumed,parent_config_sha256=inputs,parent_runtime_job_sha256=sha(runtime_path),
        views_sha256=sha(views_path),source_root=source_root,global_config_sample_asset_result_root=global_root,
        worker_python=python,shared_cache=shared,Full_and_four_factorial_cells=True,
        expected_retained_extra_tensor_bytes=0,old_full224_priority_limit43_preserved=True,
        new_fit_performed=False,new_source_count_claimed=0,
        scope='Exposed selection development already used to inform this design. Same38/22 independent QA; 300 answers do not mean300 independent sources. No quality or E2E result yet.',
        future_confirmation_answers_opened=False,GPU_used=False,SSH_used=False,uploaded=False)
    new(receipt_path,receipt)
    print(json.dumps(dict(status=receipt['status'],outputs=outputs,questions=60,expected_answers=300,
        existing_source_images=len(image_hashes),GPU_used=False,SSH_used=False)))


if __name__=='__main__': main()
