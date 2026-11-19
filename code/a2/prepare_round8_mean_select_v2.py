"""Prepare only explicit cap8 mean-off/on on old exposed select QA; no SSH."""
import argparse
import copy
from datetime import datetime,timezone
import json

from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path
import round8_mean_runner_v2 as backend


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='mean_select2_v2');a=p.parse_args()
    require(a.name.replace('_','').isalnum(),'Simple new versioned preparation name required')
    runtime_path=ROOT/'configs/a2/round8_v1/job_joint_cd_select5_v1_doc.json';runtime=read(runtime_path)
    source_root,worker=runtime['round8_source_root'],runtime['argv'][0]
    global_root=runtime['argv'][runtime['argv'].index('--output')+1].split('/results/',1)[0]
    shared=global_root+'/cache/round8_v1/shared'
    require(source_root.startswith(global_root+'/stages/round8_') and worker.startswith(global_root.rsplit('/',1)[0]+'/_runtime/'),
        'Use isolated code and actual data-volume P72')
    views_path=ROOT/'audit/a2/round8_exposed_evaluation_views_v1.json';views=read(views_path)['files']
    sources=backend.source_fingerprints();payloads=[];outputs=[];inputs={};codec={}
    receipt=ROOT/f'audit/a2/round8_{a.name}_prepared.json'
    for dataset,gpu,questions in [('doc',0,38),('text',1,22)]:
        parent_path=ROOT/f'configs/a2/round8_v1/joint_cd_select5_v1_{dataset}.json';parent=read(parent_path)
        base=copy.deepcopy(next(v for v in parent['variants'] if v['name']=='r8_l2_joint0'))
        for key in list(base):
            if key.startswith('row_cosine_'):base.pop(key)
        base.update(key_assignment='l2',key_amplitude='norm',joint_gain_refit=False,value_joint_refit=False,
            postres_direction=False,final_ray_mode='off',mean_decode_chunk=4096,cap=8)
        require(base['direction_book_path']==base['metadata_reference_book_path'] and base['value_encoding']=='binary'
            and base['v_group_size']=='all' and len(base['selected_stages'])==224 and base['res_stages_limit']==43,
            'OldGaussian16/L2/norm/binary/all/full224/actual43 required')
        variants=[]
        for enabled in (False,True):
            variant=copy.deepcopy(base);variant.update(name='r8_visual_mean_'+('on' if enabled else 'off'),visual_mean_correction=enabled)
            variants.append(variant)
        config=copy.deepcopy(parent);config.pop('joint_cd_factorial_design',None)
        config.update(variants=variants,backend_family='mean',backend_module='round8_mean_runner_v2',candidate_subset=True,
            factorial_complete=False,scope='Historically exposed select QA; one final-decoded visual mean-header off/on control')
        config['arguments']['max_patches']=8
        require(config['arguments']==dict(parent['arguments'],max_patches=8) and config['count']==questions
            and config.get('start',0)==0 and all(v['cap']==8 for v in variants),'Explicit effective cap8 alignment required')
        samples=local_path(config['arguments']['samples'])
        require(config['arguments']['samples'].startswith(global_root+'/configs/') and sha(samples)==views[dataset]['samples_sha256'],
            'Use exact existing select sample bytes')
        require(sha(ROOT/views[dataset]['references'])==views[dataset]['references_sha256'],'Existing reference bytes changed')
        require(len(backend.validate_config(config))==2,'Exactly off/on candidate declarations required')
        config['selected_control_design']=dict(family='mean',declared_modes=[False,True],
            initialization='oldGaussian16 L2/norm with original factor fold and signMAD; only existing direction_mean corrected after final packed decode',
            all_real_Nv=True,native128_sample_not_used=True,original224_ranking_limit43=True,no_K_joint=True,no_V_joint=True,no_CD=True,
            no_postres=True,no_final_ray=True,added_persistent_tensor_bytes=0,explicit_max_patches=8,all_variant_cap=8,
            same_original_select_questions=True,required_final_marker='complete_mean.json',parent_config_sha256=sha(parent_path),
            source_scope='Historically exposed selection,16sources/domain; no fresh confirmation claim')
        cp=ROOT/f'configs/a2/round8_v1/{a.name}_{dataset}.json';jp=ROOT/f'configs/a2/round8_v1/job_{a.name}_{dataset}.json'
        job=copy.deepcopy(runtime);job.update(name='round8_'+a.name+'_'+dataset,round8_source_root=source_root,cwd=source_root)
        job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
        for key,value in job['env'].items():
            if key in ('TMPDIR','TMP','TEMP','HF_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME') or 'CACHE' in key:
                require(value.startswith(shared+'/'),'Project shared data-disk cache/temp required')
        output=global_root+'/results/round8_v1/'+a.name+'_'+dataset
        job['argv']=[worker,source_root+'/code/a2/round8_mean_runner_v2.py','--config',global_root+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
        job['prerequisites']=['Root verifies actual CUDA round8_mean_gpu_check PASS before launch',
            'Root grants idle GPU lease; prepared jobs do not enforce scheduling',
            'Only complete_mean.json after complete_round8 and full input/source/record postcheck permits scoring',
            'Explicit max_patches8/all cap8; sourceRoot code/cwd only; globalR config/sample/results; data-volume P72/sharedcache']
        job['dependency_enforcement']='Manual root scheduling; preparation and launcher do not enforce prerequisites'
        for variant in variants:
            for key,value in variant.items():
                if key.endswith('_path') and isinstance(value,str):codec[value]=sha(local_path(value))
        payloads.extend([(cp,config),(jp,job)]);inputs[parent_path.relative_to(ROOT).as_posix()]=sha(parent_path)
        outputs.append(dict(dataset=dataset,gpu=gpu,config=cp.relative_to(ROOT).as_posix(),job=jp.relative_to(ROOT).as_posix(),
            output=output,samples=samples.relative_to(ROOT).as_posix(),samples_sha256=sha(samples),
            references_local_only=views[dataset]['references'],references_sha256=views[dataset]['references_sha256'],
            sources=16,questions=questions,variants=2,expected_records=questions*2,required_final_marker='complete_mean.json',explicit_max_patches=8,all_cap=8))
    require(not receipt.exists() and not any(path.exists() for path,_ in payloads),'Preserve existing outputs; select a new name')
    require(backend.source_fingerprints()==sources,'Sources changed during preparation')
    for path,value in payloads:new(path,value)
    for out in outputs:
        for key in ('config','job'):out[key+'_sha256']=sha(ROOT/out[key])
    new(receipt,dict(status='PREPARED_NOT_EXECUTED',schema='trajkv.round8.mean-select-preparation.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(__file__),source_sha256=sources,outputs=outputs,parent_input_sha256=inputs,views_sha256=sha(views_path),
        consumed_codec_inputs_sha256=codec,source_root=source_root,global_config_sample_asset_result_root=global_root,
        worker_python=worker,shared_cache=shared,explicit_max_patches=8,all_cap=8,original224_priority_limit43_preserved=True,
        total_expected_records=120,added_persistent_tensor_bytes=0,GPU_gate='CUDA8/8 numerical helper/reader PASS retained; v2 changes only real session_id finalization; root GPU lease pending',
        old_inputs_and_configs_modified=False,future_confirmation_answers_opened=False,GPU_used=False,SSH_used=False,uploaded=False))
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',outputs=outputs,total_expected_records=120,GPU_used=False,SSH_used=False)),flush=True)


if __name__=='__main__':main()
