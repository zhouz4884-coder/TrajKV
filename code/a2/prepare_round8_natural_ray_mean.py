"""Local-only final Ray-norm/Mean-on natural-dev64 preparation; no refit."""
import argparse
import copy
from datetime import datetime,timezone
import importlib
import json

from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path


SPECS={
    'final_ray':dict(stem='natural_dev64_raynorm1_v1',module='round8_final_ray_runner',
        parent='final_ray_select5_v1',variant='r8_final_ray_norm',marker='complete_final_ray.json'),
    'mean':dict(stem='natural_dev64_mean1_v1',module='round8_mean_runner_v2',
        parent='mean_select2_v2',variant='r8_visual_mean_on',marker='complete_mean.json'),
}


def require(value,message):
    if not value:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--families',nargs='+',choices=list(SPECS),default=list(SPECS))
    args=parser.parse_args();prepared_path=ROOT/'audit/a2/round8_natural_dev64_firstq3_v1_prepared.json'
    prepared={row['dataset']:row for row in read(prepared_path)['outputs']}
    runtime_path=ROOT/'configs/a2/round8_v1/job_natural_dev64_firstq3_v1_doc.json';runtime=read(runtime_path)
    sr,worker=runtime['round8_source_root'],runtime['argv'][0]
    gr=runtime['argv'][runtime['argv'].index('--output')+1].split('/results/',1)[0];shared=gr+'/cache/round8_v1/shared'
    require(sr.startswith(gr+'/stages/round8_') and worker.startswith(gr.rsplit('/',1)[0]+'/_runtime/')
        and worker.endswith('/envs/p72_modern_vlm/bin/python'),'Isolated SR and validated8TB P72 required')
    launcher=ROOT/'code/launch_round8_data_disk.py';generic=ROOT/'code/launch_remote_data_disk.py'
    payloads=[];outputs=[];inputs={};source_maps={};codec={};families={}
    receipt=ROOT/'audit/a2/round8_natural_ray_mean_v1_prepared.json'
    for family in args.families:
        spec=SPECS[family];backend=importlib.import_module(spec['module']);sources=backend.source_fingerprints()
        source_maps[family]=sources;families[family]=spec
        for dataset,gpu in [('doc',0),('text',1)]:
            original_path=ROOT/f'configs/a2/round8_v1/natural_dev64_firstq3_v1_{dataset}.json';original=read(original_path)
            parent_path=ROOT/f'configs/a2/round8_v1/{spec["parent"]}_{dataset}.json';parent=read(parent_path)
            variant=copy.deepcopy(next(v for v in parent['variants'] if v['name']==spec['variant']))
            baseline_names=[v['name'] for v in original['variants']]
            require(baseline_names==['full8','r7_frozen_joint43_control','r8_oldbook_binaryV_jointgain']
                and spec['variant'] not in baseline_names,'Preserve original three-arm control graph')
            require(original['variants'][2]['joint_gain_refit'] is True,'Original third control is joint-gain; do not relabel as off')
            reference=original['variants'][1]
            require(variant['direction_book_path']==variant['metadata_reference_book_path']==reference['direction_book_path']
                and variant['metadata_factor_path']==reference['metadata_factor_path']
                and variant['selected_stages']==reference['selected_stages'] and len(variant['selected_stages'])==224
                and variant['res_stages_limit']==43 and variant['budget_bits']==reference['budget_bits'],
                'OldGaussian16/f0/full224/43/price must match original natural control')
            require(variant.get('key_assignment','l2')=='l2' and variant['key_amplitude']=='norm'
                and variant.get('key_encoding','reencode')=='reencode' and variant['value_encoding']=='binary'
                and variant['v_group_size']=='all' and variant['cap']==8
                and variant.get('joint_gain_refit',False) is False and variant.get('value_joint_refit',False) is False
                and variant.get('postres_direction',False) is False,'No unrelated intervention allowed')
            if family=='final_ray':
                require(variant['final_ray_mode']=='norm' and variant.get('visual_mean_correction',False) is False,'Ray norm only')
            else:
                require(variant['visual_mean_correction'] is True and variant.get('final_ray_mode','off')=='off'
                    and variant['mean_decode_chunk']==4096,'Mean on only with actual4096 chunk')
            config=copy.deepcopy(original);config.pop('development_protocol',None)
            config.update(variants=[variant],backend_family=family,backend_module=spec['module'],candidate_subset=True,
                factorial_complete=False,start=0,count=64,scope='Historically exposed natural-dev64 fixed first-QA final candidate validation; no new-source confirmation')
            config['arguments']['max_patches']=8
            require(config['arguments']==dict(original['arguments'],max_patches=8) and config['arguments']['turns']==1,
                'Only effective argument max_patches changes16 to8; candidate cap8 is explicit')
            samples=local_path(config['arguments']['samples']);rows=read(samples);old=prepared[dataset]
            require(config['arguments']['samples'].startswith(gr+'/configs/') and sha(samples)==old['samples_sha256']
                and len(rows)==64 and len({r['id'] for r in rows})==64 and len({r['image_id'] for r in rows})==64,
                'Exact prior64 unique first-QA samples required')
            require(all(r['turns']==1 and len(r['questions'])==1 and not any(k in r for k in ('answer','answers','references')) for r in rows),
                'Answer-free first-QA input contract violated')
            reference_path=ROOT/old['references_local_only']
            require(sha(reference_path)==old['references_sha256'],'Existing reference SHA changed; references are not parsed')
            declarations=backend.validate_config(config);require(len(declarations)==1,'Exactly one candidate per job')
            name='round8_'+spec['stem']+'_'+dataset
            config['natural_final_candidate_extension']=dict(parent_controls='natural_dev64_firstq3_v1_'+dataset,
                variant=spec['variant'],same_original64_first_questions=True,required_final_marker=spec['marker'],
                expected_records=64,expected_variants_after_individual_merge=4,original_three_control_names=baseline_names,
                explicit_max_patches=8,all_cap=8,full224_ranking_limit43=True,new_source_count_claimed=0,
                selected_variant_contract=dict(path=parent_path.relative_to(ROOT).as_posix(),sha256=sha(parent_path),variant_name=spec['variant']))
            cp=ROOT/f'configs/a2/round8_v1/{spec["stem"]}_{dataset}.json';jp=ROOT/f'configs/a2/round8_v1/job_{spec["stem"]}_{dataset}.json'
            output=gr+'/results/round8_v1/'+spec['stem']+'_'+dataset
            job=copy.deepcopy(runtime);job.update(name=name,round8_source_root=sr,cwd=sr)
            job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
            for key,value in job['env'].items():
                if key in ('TMPDIR','TMP','TEMP','HF_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME') or 'CACHE' in key:
                    require(value.startswith(shared+'/'),'Use shared project data-disk cache/temp')
            job['argv']=[worker,sr+'/code/a2/'+spec['module']+'.py','--config',gr+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
            snapshot=sr+'/jobs/'+name+'.source';actual=snapshot+'/code/a2/'+spec['module']+'.py'
            job['runtime_policy']='Required dedicated code/launch_round8_data_disk.py; actual code argv rewritten to SR/jobs/name.source; code/cwdSR, model/config/samples/resultsglobalR,8TBP72/sharedcache'
            job['required_launcher']='code/launch_round8_data_disk.py'
            job['expected_launch_receipt']=dict(job=sr+'/jobs/'+name+'.json',source_snapshot=snapshot,
                launch_json=sr+'/jobs/'+name+'.launch.json',log=sr+'/logs/'+name+'.log',argv_executed=[worker,actual,'--config',gr+'/'+cp.relative_to(ROOT).as_posix(),'--output',output])
            job['prerequisites']=['Root grants this GPU only after its current natural64_g8budget2 lease releases',
                'Exclusive source/config upload and exact hashes; do not overwrite frozen source lists/runs',
                'Use dedicated launcher and verify actual receipt argv points inside SR/jobs/name.source/code/a2 before accepting worker',
                'Accept only final '+spec['marker']+' with complete canonical/source/input/records coverage; no intermediate-marker scoring']
            job['dependency_enforcement']='Manual root scheduling; prepared jobs do not enforce dependencies or leases'
            analysis_output='results/round8_v1/'+spec['stem']+'_'+dataset+'_merged4_analysis_v1'
            sample_local=samples.relative_to(ROOT).as_posix();baseline='results/round8_v1/natural_dev64_firstq3_v1_'+dataset
            analysis_argv=['python','code/a2/analyze_round8_final_quality.py','--runs',baseline,'results/round8_v1/'+spec['stem']+'_'+dataset,
                '--samples',sample_local,'--references',old['references_local_only'],'--dataset',dataset,'--output',analysis_output,
                '--expected-variants','4','--analysis-role','historically_exposed_natural_dev64_final_candidate','--candidate',spec['variant'],
                '--artifact-map','configs/a2/round8_v1/mean_select2_v2_analysis_artifact_map.json']
            for key,value in variant.items():
                if key.endswith('_path') and isinstance(value,str):codec[value]=sha(local_path(value))
            payloads.extend([(cp,config),(jp,job)]);inputs[original_path.relative_to(ROOT).as_posix()]=sha(original_path);inputs[parent_path.relative_to(ROOT).as_posix()]=sha(parent_path)
            outputs.append(dict(family=family,dataset=dataset,gpu=gpu,config=cp.relative_to(ROOT).as_posix(),job=jp.relative_to(ROOT).as_posix(),
                output=output,variant=spec['variant'],samples=sample_local,samples_sha256=sha(samples),sources=64,questions=64,variants=1,expected_records=64,
                references_local_only=old['references_local_only'],references_sha256=old['references_sha256'],required_final_marker=spec['marker'],
                expected_variants_after_individual_merge=4,analysis_argv=analysis_argv,launch_command='python code/launch_round8_data_disk.py '+jp.relative_to(ROOT).as_posix(),
                expected_launch_receipt=job['expected_launch_receipt']))
    require(not receipt.exists() and not any(path.exists() for path,_ in payloads),'Use new names; existing files preserved')
    for family in args.families:require(importlib.import_module(SPECS[family]['module']).source_fingerprints()==source_maps[family],'Sources changed during preparation')
    for path,value in payloads:new(path,value)
    for output in outputs:
        for key in ('config','job'):output[key+'_sha256']=sha(ROOT/output[key])
    new(receipt,dict(status='PREPARED_NOT_EXECUTED',schema='trajkv.round8.natural-final-ray-mean-preparation.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(__file__),outputs=outputs,total_expected_records=64*len(outputs),families=families,source_sha256=source_maps,
        consumed_codec_inputs_sha256=codec,parent_input_sha256=inputs,natural_preparation_sha256=sha(prepared_path),
        launcher_sha256={'code/launch_round8_data_disk.py':sha(launcher),'code/launch_remote_data_disk.py':sha(generic)},
        source_root=sr,global_config_sample_asset_result_root=gr,worker_python=worker,shared_cache=shared,explicit_max_patches=8,all_cap=8,
        original224_ranking43_mask_preserved=True,expected_added_persistent_tensor_bytes=0,original_three_controls_reused=True,
        analysis_design='Separate Ray+old3 and Mean+old3 four-way analyses per domain; do not silently merge into five methods',
        lease_status='PENDING root assignment after natural64_g8budget2 per-card release',new_source_count_claimed=0,
        old_sources_runs_and_launch_receipts_modified=False,GT_references_parsed=False,GPU_used=False,SSH_used=False,uploaded=False,published=False))
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',outputs=[{k:v for k,v in x.items() if k in ('family','dataset','config','job','variant','expected_records','required_final_marker','launch_command')} for x in outputs],
        total_expected_records=64*len(outputs),GPU_used=False,SSH_used=False)),flush=True)


if __name__=='__main__':main()
