"""Prepare one fixed200-step mixed-calTrain post336 calibration job locally."""
import copy
import json

from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path
from round8_postcal_schedule import build_schedule
from round8_postcal_train import validate_config


def main():
    parent_path=ROOT/'configs/a2/round8_v1/mean_select2_v2_doc.json';parent=read(parent_path)
    base=copy.deepcopy(next(v for v in parent['variants'] if v['name']=='r8_visual_mean_off'))
    base.update(name='r8_postcal_R42_base_collection',res_stages_limit=42,visual_mean_correction=False,final_ray_mode='off',postres_direction=False)
    base.pop('mean_decode_chunk',None)
    runtime_path=ROOT/'configs/a2/round8_v1/job_natural_dev64_mean1_v1_doc.json';runtime=read(runtime_path)
    gr=runtime['argv'][-1].split('/results/',1)[0];sr=runtime['round8_source_root'];worker=runtime['argv'][0]
    samples='configs/a2/round8_v1/calibration_train32.json';grouped=read(ROOT/samples);flat,schedule,meta=build_schedule(grouped,steps=200,seed=1212)
    args=copy.deepcopy(parent['arguments']);args.update(samples=gr+'/'+samples,max_patches=8,compress_before_question=True,turns=1)
    config=dict(schema='trajkv.round8.postcal-train.v1',arguments=args,bank_settings=base,steps=200,seed=1212,
        objective='full_teacher_KL',checkpoint='final_step200',target_tokens=8,text_capacity=512,parameter_shape=[3,28,4],encoder_constant_bytes=1344,
        learning_rate=.003,regularization=.01,shadow_absolute_tolerance=.003,shadow_relative_tolerance=.003,
        source_balanced_schedule=dict(canonical_sha256=meta['schedule_sha256'],sources=32,questions=70,Doc=16,Text=16,
            finite_domain_steps=meta['domain_step_counts'],source_visits='6or7; six full32-source epochs plus8 balanced tail',
            per_step_loss='one scheduled question tokenmean KL, weight1',summary_loss='equal domains/sources/Qwithin source/tokenmean'),
        old_f0_scope='Original Doc32/83case G4_256 teacherKL fit inherited unchanged to K1/R43; controls intentionally frozen, not a claim of new-codec joint optimality or an original implementation bug',
        method_scope='New final-state coherent K/V field fold plus mixed-domain Fullteacher-aware calibration; no nativeMSE objective, select iteration choice, GT, Mean, Ray or book update',
        acceptance='Actual first-Q zero-theta field/packed identity, dense shadow tolerance, finite nonzero q/post gradients, then fixed200 updates and strict complete70Q source-balanced trainKL decrease; final marker required')
    contract=validate_config(config);config['priority_contract']=contract
    cp=ROOT/'configs/a2/round8_v1/postcal_train32_v2.json';jp=ROOT/'configs/a2/round8_v1/job_postcal_train32_v2.json'
    job=copy.deepcopy(runtime);name='round8_postcal_train32_v1';output=gr+'/results/round8_v1/postcal_train32_v1'
    job.update(name=name,round8_source_root=sr,cwd=sr);job['env'].update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
    job['argv']=[worker,sr+'/code/a2/round8_postcal_train.py','--config',gr+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
    job['required_launcher']='code/launch_round8_data_disk.py'
    job['expected_launch_receipt']=dict(job=sr+'/jobs/'+name+'.json',launch_json=sr+'/jobs/'+name+'.launch.json',log=sr+'/logs/'+name+'.log',
        source_snapshot=sr+'/jobs/'+name+'.source',argv_executed=[worker,sr+'/jobs/'+name+'.source/code/a2/round8_postcal_train.py','--config',gr+'/'+cp.relative_to(ROOT).as_posix(),'--output',output])
    job['prerequisites']=['Root GPU0 lease; exclusive new sources/config/output and actual snapshot argv verified',
        'Shared field helper CPU and actual CUDA reader controls pass; theta0 identity and actual q/post gradient gate enforced within trainer before remaining collection/updates',
        'Calibration-only no select/GT;200steps fixed1212; full336FP32 charged; R42=23K19V; original224ranking/f0/J/sign fixed']
    job['dependency_enforcement']='Root/agent manual source+GPU gate; trainer itself enforces actual first-Q and final train gates'
    job['runtime_policy']='DedicatedSR snapshot; globalR model/samples/results and explicit8TB P72/sharedcache; all future artifacts new immutable paths'
    if cp.exists() or jp.exists():raise ValueError('Prepared files already exist; preserve immutable names')
    new(cp,config);new(jp,job)
    inputs={parent_path.relative_to(ROOT).as_posix():sha(parent_path),runtime_path.relative_to(ROOT).as_posix():sha(runtime_path),samples:sha(ROOT/samples)}
    codec={value:sha(local_path(value)) for key,value in base.items() if key.endswith('_path') and isinstance(value,str)}
    receipt=dict(status='PREPARED_NOT_EXECUTED',schema='trajkv.round8.postcal-train-prepared.v1',config=cp.relative_to(ROOT).as_posix(),config_sha256=sha(cp),
        job=jp.relative_to(ROOT).as_posix(),job_sha256=sha(jp),source_samples=32,questions=70,steps=200,seed=1212,
        contract=contract,schedule_canonical_sha256=meta['schedule_sha256'],inputs_sha256=inputs,codec_inputs_sha256=codec,code_sha256=sha(__file__),
        output=output,factors=output+'/factors.json',final_marker=output+'/complete_postcal_train.json',source_root=sr,global_root=gr,
        model_weights_frozen=True,new_post_FP32_bytes=1344,old_f0_still_billed=True,GT_references_opened=False,select_opened=False,GPU_used=False,SSH_used=False,
        launch_command='python code/launch_round8_data_disk.py '+jp.relative_to(ROOT).as_posix(),actual_GPU_first_probe_pending=True,actual_trainKL_decrease_pending=True)
    path=ROOT/'audit/a2/round8_postcal_train32_v2_prepared.json';new(path,receipt)
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',config=receipt['config'],job=receipt['job'],config_sha256=receipt['config_sha256'],job_sha256=receipt['job_sha256'],
        questions=70,steps=200,output=output,final_marker=receipt['final_marker'],GPU_used=False,SSH_used=False)),flush=True)


if __name__=='__main__':main()
