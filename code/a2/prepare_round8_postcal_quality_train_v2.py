"""New learned endpoints for final train32_v2; preserve every v1 artifact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from round8_postcal_runner import validate_config
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
    prior_path=ROOT/'results/round8_v1/postcal_quality_split_v1_preparation/plan.json';old=read(prior_path);plan=deepcopy(old)
    plan.update(status='PREPARED_TRAIN_V2',prior_preparation_sha256=sha(prior_path),prior_v1_training_attempt_preserved=True)
    for key in ('factors','complete'):plan['training_paths_frozen'][key]=plan['training_paths_frozen'][key].replace('/postcal_train32_v1/','/postcal_train32_v2/')
    endpoints={};files={};jobs=[]
    for relative,entry in old['endpoints'].items():
        cp=ROOT/relative;jp=cp.with_name('job_'+cp.name)
        if 'learned1_v1_' in relative:
            config=read(cp);job=read(jp);newcp=cp.with_name(cp.name.replace('learned1_v1_','learned1_v2_'));newjp=newcp.with_name('job_'+newcp.name)
            for v in config['variants']:
                v['postcal_factor_path']=v['postcal_factor_path'].replace('/postcal_train32_v1/','/postcal_train32_v2/')
                v['postcal_train_complete_path']=v['postcal_train_complete_path'].replace('/postcal_train32_v1/','/postcal_train32_v2/')
            config['postcal_training_v2_route']=dict(previous_v1_config_sha256=sha(cp),reason='Trainingv1 execution-gradient gate failed; new immutable final training run required; physical method unchanged')
            job['name']=job['name'].replace('learned1_v1_','learned1_v2_');job['argv'][3]=job['argv'][3].replace('learned1_v1_','learned1_v2_');job['argv'][5]=job['argv'][5].replace('learned1_v1_','learned1_v2_')
            job['prerequisites'].append('Consume only postcal_train32_v2 final artifacts; preserve failed trainingv1 and unused learnedv1 configs')
            validate_config(config);write(newcp,config);write(newjp,job);cp,jp=newcp,newjp
            entry=dict(entry,config_sha256=sha(cp),declarations=validate_config(config))
        endpoints[cp.relative_to(ROOT).as_posix()]=entry;files[cp.relative_to(ROOT).as_posix()]=sha(cp);files[jp.relative_to(ROOT).as_posix()]=sha(jp);jobs.append(jp.relative_to(ROOT).as_posix())
    plan.update(endpoints=endpoints,exact_files_sha256=files,jobs=jobs,GPU_started=False,references_opened=False)
    out=ROOT/'results/round8_v1/postcal_quality_split_v2_preparation';out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    print(json.dumps(dict(status='PREPARED_TRAIN_V2',plan=str(out/'plan.json'),sha256=sha(out/'plan.json'),new_learned_jobs=4,identity_control_endpoints_unchanged=True)))
if __name__=='__main__':main()
