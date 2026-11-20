"""CPU-only actual-producer/hash gate for canonical E64 Text2 -> Text16."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def inside(path,root):
    try:path.relative_to(root);return True
    except ValueError:return False
def relative(root,value):
    path=(root/value).resolve();require(inside(path,root.resolve()),'Gate path escapes data root');return path


def expected_executed_argv(argv,root,snapshot):
    result=list(argv)
    for index,value in enumerate(result):
        if isinstance(value,str) and value.startswith(str(root)+'/code/') and value.endswith('.py'):
            result[index]=str(snapshot/Path(value).relative_to(root))
    return result


def verify(root,contract):
    root=Path(root).resolve()
    require(contract['schema']=='baseline_matrix.canonical_text2_evidence_gate.v1','Wrong gate contract')
    jobfile=relative(root,contract['producer_job']);job=read(jobfile)
    require(sha(jobfile)==contract['producer_job_sha256'],'Actual canonical producer job SHA differs')
    require(job['name']==contract['producer_name'] and job['name'].endswith('_GPU0_affinity_v20'),'Not the actual v20 canonical alias')
    require(str(job['env']['CUDA_VISIBLE_DEVICES'])=='0','Canonical producer GPU affinity differs')
    cfgfile=relative(root,contract['producer_speed_config']);cfg=read(cfgfile)
    require(sha(cfgfile)==contract['producer_speed_config_sha256'],'Canonical Text2 config SHA differs')
    require(job['argv'][job['argv'].index('--config')+1]==str(cfgfile),'Producer did not execute this speed config')
    output=relative(root,contract['producer_output'])
    require(job['argv'][job['argv'].index('--output')+1]==str(output),'Producer output differs')
    launchfile=relative(root,'jobs/'+contract['producer_name']+'.launch.json');launch=read(launchfile)
    require(launch['name']==contract['producer_name'] and launch['job_sha256']==sha(jobfile),'Launch does not bind exact canonical alias')
    snapshot=Path(launch['source_snapshot']).resolve()
    require(snapshot==(root/'jobs'/(contract['producer_name']+'.source')).resolve(),'Producer source snapshot identity differs')
    require(inside(snapshot,(root/'jobs').resolve()),'Producer source snapshot escapes project jobs')
    expected_argv=expected_executed_argv(job['argv'],root,snapshot)
    require(launch['argv_executed']==expected_argv and str(launch['CUDA_VISIBLE_DEVICES'])=='0','Actual launched argv/affinity differs from exact launcher rule')
    inventoryfile=snapshot/'sha256.json';inventory=read(inventoryfile)
    require(sha(inventoryfile)==launch['source_inventory_sha256'],'Producer source-inventory identity differs')
    for name,digest in contract['numerical_source_sha256'].items():
        require(inventory[name]==digest==sha(relative(snapshot,name)),'Executed numerical source differs: '+name)
    complete=read(output/'complete.json');manifest=read(output/'manifest.json');plan=read(output/'plan.json');summary=read(output/'summary.json')
    require(complete['status']=='PASS_TIMED_AND_TOKEN_VALIDATED' and complete['GPU_used'] is True,'Canonical producer did not pass actual GPU timing')
    for field,value in contract['required_fields'].items():require(complete.get(field)==value,'Canonical40 field differs: '+field)
    raw_hashes={}
    for name,key in contract['hash_fields'].items():
        digest=sha(output/name);require(digest==complete[key],'Canonical40 raw hash differs: '+name);raw_hashes[name]=digest
    raw_hashes['complete.json']=sha(output/'complete.json')
    require(manifest['plan_sha256']==sha(output/'plan.json'),'Actual manifest/plan binding differs')
    require(plan['job_sha256']==sha(cfgfile),'Actual plan did not use canonical Text2 config')
    require(cfg['selection']=={'mode':'first','count':2} and cfg['warmup_per_qa']==1 and cfg['repeat']==3,'Canonical smoke schedule differs')
    require(plan['expected_ids']==contract['expected_ids'] and plan['method_ids']==contract['profile_method_ids'],'Actual canonical cohort/methods differ')
    for path,digest in contract['eval_config_sha256'].items():
        require(sha(relative(root,path))==digest,'Frozen nested eval config changed: '+path)
        require(plan['original_config_hashes'].get(str(relative(root,path)))==digest,'Actual plan uses another nested eval: '+path)
    require({c['method']['id'] for c in plan['effective_configs']}==set(contract['profile_method_ids']),'Canonical method profile differs')
    E=[c for c in plan['effective_configs'] if c['method']['family']=='pyramidkv']
    require(len(E)==1 and E[0]['method']['constructor_args']['window_size']==64 and E[0]['method']['constructor_args']['compression_ratio']==.888,'Historical E8 cannot satisfy canonical E64 gate')
    expectedfile=relative(root,contract['expected_predictions']);expected=read(expectedfile)
    require(sha(expectedfile)==contract['expected_predictions_sha256']==manifest['expected_predictions_sha256'],'MODEL expected identity differs')
    require(expected['references_read'] is False,'Expected file must contain MODEL outputs only')
    lookup={(r['method'],r['id']):r for r in expected['records']};require(len(lookup)==10,'Expected complete 5x2 MODEL outputs')
    records=[json.loads(line) for line in (output/'records.jsonl').open(encoding='utf-8')]
    require(len(records)==len(plan['schedule'])==40,'Actual40 trial count differs')
    for record,schedule in zip(records,plan['schedule']):
        require(all(record[key]==value for key,value in schedule.items()),'Actual trial schedule differs')
        require(record['status']=='PASS' and record['canonical_speed_eligible'] is True and record['reference_answers_read'] is False,'Actual trial not token-validated PASS')
        source=lookup[(record['method'],record['id'])]
        for key in ('output_token_ids','input_ids_sha256','full_prompt_hash','stored_visual_KV_bytes','original_image_sha256'):
            require(record[key]==source[key],'Actual trial differs from true MODEL expected: '+key)
    require(summary['status']=='PASS_TIMED_AND_TOKEN_VALIDATED','Actual summary status differs')
    return {'status':'PASS_CANONICAL_E64_TEXT2_PREREQUISITE','CPU_only':True,'GPU_used':False,
        'producer_name':contract['producer_name'],'producer_job_sha256':sha(jobfile),
        'verified_exact_producer':True,'verified_hash_fields':True,'verified_required_fields':True,
        'recorded_trials':40,'expected_trials':40,'method_profile':'canonical_E64_source_v2',
        'raw_file_sha256':raw_hashes,'producer_launch_sha256':sha(launchfile),'producer_inventory_sha256':sha(inventoryfile),
        'producer_plan_sha256':sha(output/'plan.json'),'expected_predictions_sha256':sha(expectedfile),
        'new_GPU_trials':0,'source_modified':False,'root':str(root)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'Use an immutable new CPU gate output')
    try:
        contract=read(args.contract);require(contract.get('CPU_fixture') is not True,'CPU fixtures cannot run as a production gate')
        report=verify(args.root,contract)
    except Exception as exc:
        report={'status':'FAIL_CANONICAL_E64_TEXT2_PREREQUISITE','CPU_only':True,'GPU_used':False,
            'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'new_GPU_trials':0,'source_modified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'status':report['status'],'new_GPU_trials':0}))


if __name__=='__main__':main()
