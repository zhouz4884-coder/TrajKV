"""Prepare only the authorized split postcal select4 and natural2 endpoints.

No GPU, SSH, model import, labels or training is performed here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from round8_postcal_runner import validate_config,source_fingerprints
from round8_postcal_owner import EXTRA_BYTES,identity_table,table_values_sha256
ROOT=Path(__file__).resolve().parents[2]
REMOTE='/path/to/TrajKV'
SOURCE=REMOTE+'/stages/round8_20261118_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    output=ROOT/'results/round8_v1/postcal_quality_split_v1_preparation';assert not output.exists()
    files=[];jobs=[];declarations={}
    train=REMOTE+'/results/round8_v1/postcal_train32_v1'
    for d in ('doc','text'):
        prior=read(ROOT/'configs/a2/round8_v1'/('postfit_cap8_aligned11_v3_'+d+'.json'))
        ordinary=deepcopy(next(v for v in prior['variants'] if v['name']=='r8_k4old_vbinary'))
        identity=deepcopy(ordinary);identity.update(name='postcal_identity_R42',res_stages_limit=42,postcal_mode='identity',
            joint_gain_refit=False,value_joint_refit=False,postres_direction=False)
        learned=deepcopy(identity);learned.update(name='postcal_learned_R42',postcal_mode='learned',
            postcal_factor_path=train+'/factors.json',postcal_train_complete_path=train+'/complete_postcal_train.json')
        controls=[deepcopy(next(v for v in prior['variants'] if v['name']=='full8')),
            deepcopy(next(v for v in prior['variants'] if v['name']=='r7_frozen_joint43_control')),identity]
        natural=read(ROOT/'configs/a2/round8_v1'/('natural_dev64_firstq3_v1_'+d+'.json'))
        for label,base,variants,count,role in (
            ('postcal_select_control3_v1',prior,controls,38 if d=='doc' else 22,'historically_exposed_select_postcal_controls'),
            ('postcal_select_learned1_v1',prior,[learned],38 if d=='doc' else 22,'historically_exposed_select_postcal_frozen_teacher_KL'),
            ('postcal_natural64_identity1_v1',natural,[identity],64,'historically_exposed_natural64_development_postcal_identity'),
            ('postcal_natural64_learned1_v1',natural,[learned],64,'historically_exposed_natural64_development_postcal_frozen_teacher_KL')):
            arguments=deepcopy(base['arguments']);arguments['max_patches']=8
            cp=ROOT/'configs/a2/round8_v1'/(label+'_'+d+'.json')
            config=dict(schema='trajkv.a2.variant-config.v1',arguments=arguments,start=0,count=count,variants=deepcopy(variants),
                backend_family='postcal',candidate_subset=True,factorial_complete=False,scope=role,
                postcal_design=dict(original_R42_first_then_fields=True,extra_absolute_FP32_parameters=336,added_persistent_bytes=EXTRA_BYTES,
                    original_f0_full224_ranking_and_policy_retained=True,selected_stages=42,K_stages=23,V_stages=19,
                    identity_and_learned_same_price=True,complete_owner_delta_vs_old43='1344-18*Nv',
                    no_J_sign_mean_Vcode_change=True,no_residual_reencoding=True,learned_requires_final_train_marker=True,
                    samples_and_sources_reused=True,new_reference_sources=0,
                    split_scope='Select control3+learned1 totals4 methods; natural identity1+learned1 adds2 methods and reuses oldnatural3 baselines'))
            decl=validate_config(config);write(cp,config);files.append(cp)
            job=read(ROOT/'configs/a2/round8_v1'/('job_natural_dev64_g8budget2_v1_'+d+'.json'))
            job['name']='round8_'+label+'_'+d;job['env']['CUDA_VISIBLE_DEVICES']='1'
            job['argv']=[job['argv'][0],SOURCE+'/code/a2/round8_postcal_runner.py','--config',REMOTE+'/'+cp.relative_to(ROOT).as_posix(),
                '--output',REMOTE+'/results/round8_v1/'+label+'_'+d]
            job['prerequisites']=['Root GPU1 bounded postcal split scheduling; actual source snapshot argv must be verified',
                'CPU physical/deployment stub and actual CUDA8reader checks PASS; no newkernel',
                'For learned: final200-step complete_postcal_train hash graph and frozen336FP32 factors must pass before launch',
                'No reused output/job/snapshot; complete_postcal final marker and independent full owner/source/coverage before references']
            job['prepared_control']='Authorized postcal select4/natural2 split only; no new method grid or GT'
            jp=cp.with_name('job_'+cp.name);write(jp,job);files.append(jp);jobs.append(jp.relative_to(ROOT).as_posix())
            declarations[cp.relative_to(ROOT).as_posix()]=dict(config_sha256=sha(cp),expected_answers=count*len(variants),declarations=decl)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='PREPARED',GPU_started=False,references_opened=False,source_sha256=source_fingerprints(),jobs=jobs,
        endpoints=declarations,expected_new_select_answers=240,expected_new_natural_answers=256,
        identity_table_values_sha256=table_values_sha256(identity_table()),training_paths_frozen=dict(factors=train+'/factors.json',complete=train+'/complete_postcal_train.json'),
        owner_proof=[dict(nv=n,old43=4590*n+89733,postcal42=4572*n+89733+1344,delta=1344-18*n) for n in (256,768,1280,1792,2304)],
        exact_files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write(output/'plan.json',report)
    print(json.dumps(dict(status='PREPARED',jobs=len(jobs),select_answers=240,natural_answers=256,plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'))))
if __name__=='__main__':main()
