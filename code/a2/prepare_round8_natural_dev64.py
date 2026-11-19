"""Prepare exposed natural-dev64 fixed-first-question three-arm jobs locally.

Use every predeclared pilot64 source, its stored first question, original
references and existing image bytes. No fitting, inference, SSH or publishing.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from round8_data_prepare import ROOT,read,sha,new
from round7_data_prepare import selected_json_references
from inspect_round6_unseen_pools import project_path


def require(value,message):
    if not value: raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='natural_dev64_firstq3_v1')
    a=p.parse_args(); require(re.fullmatch(r'[A-Za-z0-9_]+',a.name),'Use a new simple versioned name')
    template_path=ROOT/'configs/a2/round8_v1/postfit_select11_v2_doc.json'
    template=read(template_path)
    runtime_path=ROOT/'configs/a2/round8_v1/job_postfit_select11_v2_doc.json'
    runtime=read(runtime_path)
    global_root=runtime['argv'][runtime['argv'].index('--output')+1].split('/results/',1)[0]
    source_root=runtime['round8_source_root']; python=runtime['argv'][0]
    require(source_root.startswith(global_root+'/stages/round8_') and source_root!=global_root,'Independent source root required')
    require(python.startswith(global_root.rsplit('/',1)[0]+'/_runtime/') and '/envs/p72_modern_vlm/bin/python' in python,
        'Use the validated data-volume P72 worker')
    by={v['name']:v for v in template['variants']}
    variants=[copy.deepcopy(by['full8']),copy.deepcopy(by['r7_frozen_joint43_control']),
              copy.deepcopy(by['r8_k4old_vbinary_jointgain'])]
    variants[2]['name']='r8_oldbook_binaryV_jointgain'
    frozen_path=ROOT/'configs/a2/round7_v1/confirm_doc128_s0.json'
    frozen=next(v for v in read(frozen_path)['variants'] if v['name']=='r7_joint_k1_gt_cond')
    for v in variants[1:]:
        require(v['selected_stages']==frozen['selected_stages'] and len(v['selected_stages'])==224
            and v['res_stages_limit']==43,'Frozen full ranking or actual residual limit changed')
    require(variants[2]['key_encoding']=='reencode' and variants[2]['value_encoding']=='binary'
        and variants[2]['joint_gain_refit'] is True and variants[2]['key_amplitude']=='norm',
        'Expected old-book/binary-V zero-field joint-gain control')
    from round8_quality_owner import local_path
    require(sha(local_path(variants[2]['direction_book_path']))==sha(local_path(frozen['direction_book_path'])),
        'Joint-gain control must use the exact old decoder book')
    consumed={value:sha(local_path(value)) for variant in variants for key,value in variant.items()
        if key.endswith('_path') and isinstance(value,str)}
    preparation_path=ROOT/'audit/a2/round8_data_preparation_v1.json'; prepared=read(preparation_path)
    grouped_cal={role:{row['image_id'] for row in read(ROOT/f'configs/a2/round8_v1/calibration_{role}32.json')}
        for role in ('train','select')}
    payloads=[]; outputs=[]; image_manifest=[]; source_inputs={}; reference_audits=[]
    shared=global_root+'/cache/round8_v1/shared'
    receipt_path=ROOT/f'audit/a2/round8_{a.name}_prepared.json'
    for kind,gpu in [('doc',0),('text',1)]:
        pilot_spec=prepared['pilots'][kind+'64']; pilot_path=ROOT/pilot_spec['samples']
        require(sha(pilot_path)==pilot_spec['sha256'],'Pilot64 frozen view changed')
        pilot=read(pilot_path); require(len(pilot)==64 and len({r['image_id'] for r in pilot})==64,'Expected exactly64 original sources')
        reference_path=ROOT/pilot_spec['references']
        refs_by=selected_json_references(reference_path,{r['id'] for r in pilot})
        samples=[]; refs=[]
        for i,original in enumerate(pilot):
            row=copy.deepcopy(original); ref=copy.deepcopy(refs_by[original['id']])
            original_id=original['id']; original_question=original['questions'][0]
            sid='r8_'+a.name+'_'+kind+'_'+row['image_id'].removeprefix('sha256:')
            row['id']=row['session_id']=sid
            row.update(questions=[original_question],turns=1,content=[{'type':'image'},{'type':'text','text':original_question}],
                split='round8_historically_exposed_development_first_question_only',
                protocol='single_source_first_question_independent_QA',
                round8_analysis_role='historically_exposed_development_not_confirmation',
                source_id=row['image_id'],original_pilot_session_id=original_id,
                historical_result_reused=False,first_question_policy='stored first question; no outcome/answer based question selection')
            row.pop('historical_use',None)
            require(not any(key in row for key in ('answers','answer','references')),'Answers in runner input')
            ref['id']=sid
            if kind=='doc':
                require(original['dataset']=='docvqa_grouped' and ref['source_document_id']==row['source_document_id'],
                    'Doc source-reference mismatch')
                first=next(item for item in ref['references'] if int(item['turn'])==1)
                require(str(first['question_id'])==str(original['source_question_ids'][0]),'Doc first-question identity mismatch')
                ref['references']=[copy.deepcopy(first)]
                row['source_question_ids']=original['source_question_ids'][:1]
                require(ref['references'][0]['answer']==refs_by[original_id]['references'][0]['answer'],
                    'Doc first-question reference list changed')
                answers=ref['references'][0]['answer']
            else:
                require(original['dataset']=='textvqa' and ref['image_id']==row['image_id'] and len(ref['answers'])==10,
                    'TextVQA original ten-annotator reference required')
                require(ref['answers']==refs_by[original_id]['answers'],'Text raw votes/order changed')
                answers=ref['answers']
            reference_audits.append(dict(dataset=kind,session_id=sid,parent_id=original_id,
                source_image_id=row['image_id'],question_id=row.get('source_question_id',row.get('source_question_ids',[None])[0]),
                retained_answer_count=len(answers),answers_list_preserved=True,
                answer_list_sha256=hashlib.sha256(json.dumps(answers,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))
            require(len(row['images'])==len(row['remote_images'])==len(row['hashes'])==1,'Expected single unchanged image')
            local=project_path(row['images'][0]); digest=row['hashes'][0]
            require(local.is_file() and sha(local)==digest and row['image_id']=='sha256:'+digest,'Existing image SHA mismatch')
            require(row['remote_images'][0].startswith(global_root+'/data/'),'Existing image is not under global data root')
            with Image.open(local) as image:
                dimensions=list(image.size); format=image.format; image.verify()
            image_manifest.append(dict(dataset=kind,session_id=sid,local=str(local),remote=row['remote_images'][0],
                sha256=digest,bytes=local.stat().st_size,image_size=dimensions,image_format=format,
                local_existing_asset=True,local_sha_and_decode_check='PASS',remote_path_preserved=True,remote_existence_checked=False))
            samples.append(row); refs.append(ref)
        source_inputs[kind]=dict(pilot_samples=pilot_spec['samples'],pilot_sha256=sha(pilot_path),
            original_references=pilot_spec['references'],original_references_sha256=sha(reference_path),
            source_order_preserved=True,first_question_fixed=True,
            overlaps_with_R8_calibration={role:len({r['image_id'] for r in samples}&ids) for role,ids in grouped_cal.items()})
        samples_path=ROOT/f'configs/a2/round8_v1/{a.name}_{kind}_samples.json'
        references_path=ROOT/f'configs/data/round8_{a.name}_{kind}_references.json'
        cp=ROOT/f'configs/a2/round8_v1/{a.name}_{kind}.json'
        jp=ROOT/f'configs/a2/round8_v1/job_{a.name}_{kind}.json'
        config=copy.deepcopy(template)
        config.pop('postfit_provenance',None)
        config.update(start=0,count=64,variants=variants,
            scope='Previously exposed development64, stored first independent QA/source; no new confirmation/blind claim')
        config['arguments'].update(samples=global_root+'/'+samples_path.relative_to(ROOT).as_posix(),turns=1)
        config['development_protocol']=dict(original_pilot=pilot_spec['samples'],original_pilot_sha256=sha(pilot_path),
            first_question_only=True,questions=64,sources=64,new_source_claim=False,
            method_choice='Joint-gain selected after observed R8 select results; development only',
            no_refitting=True,no_current_answer_or_outcome_filtering=True)
        job=dict(schema='trajkv.round8.job.v2',name='round8_'+a.name+'_'+kind,
            round8_source_root=source_root,cwd=source_root,env=copy.deepcopy(runtime['env']))
        job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
        for key,leaf in [('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),
                ('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('PYTHONPYCACHEPREFIX','pycache'),
                ('TORCH_HOME','torch'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('PIP_CACHE_DIR','pip'),
                ('UV_CACHE_DIR','uv'),('XDG_CACHE_HOME','xdg')]:
            job['env'][key]=shared+'/'+leaf
        hf=shared+'/huggingface'
        job['env'].update(HF_HOME=hf,HF_HUB_CACHE=hf+'/hub',HUGGINGFACE_HUB_CACHE=hf+'/hub',HF_MODULES_CACHE=hf+'/modules')
        output=global_root+'/results/round8_v1/'+a.name+'_'+kind
        job['argv']=[python,source_root+'/code/a2/round8_runner.py','--config',global_root+'/'+cp.relative_to(ROOT).as_posix(),
            '--output',output]
        job['runtime_policy']='Validated data-volume P72; sourceRoot code only; globalR configs/samples/output; shared project data-disk caches'
        job['prerequisites']=['Root verifies immutable source snapshot, uploaded config/sample SHA and current idle GPU',
            'Local128 existing images verified by SHA and image decode; remote existence/readability is verified by root']
        payloads.extend([(samples_path,samples),(references_path,refs),(cp,config),(jp,job)])
        outputs.append(dict(dataset=kind,gpu=gpu,samples=samples_path.relative_to(ROOT).as_posix(),
            references_local_only=references_path.relative_to(ROOT).as_posix(),config=cp.relative_to(ROOT).as_posix(),
            job=jp.relative_to(ROOT).as_posix(),output=output,sources=64,independent_questions=64,
            variants=3,expected_records=192,expected_answers=192,
            launch_command='python code/launch_round8_data_disk.py '+jp.relative_to(ROOT).as_posix(),
            upload_to_globalR=[samples_path.relative_to(ROOT).as_posix(),cp.relative_to(ROOT).as_posix()],
            refs_must_not_be_uploaded=True))
    require(not any(path.exists() for path,_ in payloads) and not receipt_path.exists(),'Use a new version; existing files are preserved')
    require(len({row['sha256'] for row in image_manifest})==128,'Domain pools share raw content')
    for path,value in payloads: new(path,value)
    for output in outputs:
        for key in ('samples','config','job'):
            output[key+'_sha256']=sha(ROOT/output[key])
        output['references_sha256']=sha(ROOT/output['references_local_only'])
    receipt=dict(schema='trajkv.round8.exposed-natural-development-preparation.v1',status='PREPARED_NOT_EXECUTED',
        created_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),outputs=outputs,
        source_inputs=source_inputs,images=image_manifest,images_verified=128,
        image_bytes=sum(row['bytes'] for row in image_manifest),reference_integrity=reference_audits,
        Text_all_original_ten_votes_preserved=True,Doc_first_question_answer_lists_preserved=True,
        consumed_codec_inputs_sha256=consumed,requested_priority_sha256=canonical_sha(frozen['selected_stages']),
        actual_43_mask_sha256=canonical_sha(frozen['selected_stages'][:43]),
        fixed_priority_entries=224,ranking_bytes=448,stage_limit=43,K_stages=23,V_stages=20,
        parent_config_sha256=sha(template_path),parent_runtime_job_sha256=sha(runtime_path),
        frozen_baseline_config_sha256=sha(frozen_path),data_preparation_sha256=sha(preparation_path),
        source_root=source_root,global_config_sample_asset_result_root=global_root,worker_python=python,shared_cache=shared,
        independent_new_source_count_claimed=0,new_fit_performed=False,model_outputs_opened=False,
        exposure_scope='Historically exposed development cohorts; first question fixed from prior pilot64, all64 retained; chosen after select results; no confirmation/blind claim',
        future_confirmation_answers_opened=False,model_or_GPU_executed=False,SSH_used=False,uploaded=False)
    new(receipt_path,receipt)
    print(json.dumps(dict(status=receipt['status'],outputs=outputs,images_verified=128,image_bytes=receipt['image_bytes'],
        shared_cache=shared,GPU_used=False,SSH_used=False)))


if __name__=='__main__': main()
