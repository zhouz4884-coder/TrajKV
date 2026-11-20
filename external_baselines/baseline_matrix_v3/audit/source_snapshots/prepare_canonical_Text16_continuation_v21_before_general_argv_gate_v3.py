"""Prepare only: one CPU evidence gate + canonical Text16 formal320.

Root supplies an actual verified pause receipt later. This tool never calls
connect/pause/stage/launch, signals a worker, or edits an existing plan/job.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import re

from remote_control import ROOT,REMOTE_ROOT,cache_environment

V20_PLAN='configs/queues/External_baselines_P2_low_budget_canonical_speed_queue_v20.json'
V20_PLAN_SHA='2b2110a1a94ccf97ff7c4747ed69c0e6fa937ab62d4caedb3f7a9f0508047a56'
SOURCE_JOB='configs/jobs/bm_M1_source_representatives_textvqa_16_formal_fixed_cohort_v2_actualK32_v1.json'
TIMING_MATRIX='configs/matrix/SPEED_M1_source_representatives_targets_v2_actualK32_v1.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    if path.exists():assert path.read_bytes()==body,'Frozen preparation differs: '+str(path)
    else:
        with path.open('xb') as stream:stream.write(body)


def additions(version):
    assert re.fullmatch(r'v[0-9]+',version)
    assert sha(ROOT/V20_PLAN)==V20_PLAN_SHA
    original_plan=read(ROOT/V20_PLAN)
    smoke=next(n for n in original_plan['lanes']['0'] if n.get('extension_role')=='canonical_E64_profile_hardware_smoke_separate_from_historical_E8_40')
    smokejob=read(ROOT/smoke['job']);smokecfg_rel=smokejob['argv'][smokejob['argv'].index('--config')+1].split(REMOTE_ROOT+'/',1)[1]
    assert sha(ROOT/smoke['job'])==smoke['job_sha256'] and smokejob['name'].endswith('_GPU0_affinity_v20')
    cfg=read(ROOT/smokecfg_rel)
    matrix=read(ROOT/TIMING_MATRIX);formal=next(r for r in matrix['rows'] if r['dataset']=='textvqa' and r['QA']==16)
    assert formal['status']=='ACTUAL_MODEL_EXPECTED_COMPLETE_NOT_LAUNCHED' and formal['total_trials']==320
    assert formal['job']==SOURCE_JOB
    source=read(ROOT/SOURCE_JOB);alias=deepcopy(source)
    alias['name']+='_GPU0_affinity_'+version;alias['env']['CUDA_VISIBLE_DEVICES']='0'
    stripped=lambda j:{**j,'name':None,'env':{**j['env'],'CUDA_VISIBLE_DEVICES':None}}
    assert stripped(alias)==stripped(source) and alias['argv']==source['argv']
    alias_rel='configs/jobs/'+alias['name']+'.json'
    nested={p.split(REMOTE_ROOT+'/',1)[1]:sha(ROOT/p.split(REMOTE_ROOT+'/',1)[1]) for p in cfg['eval_configs']}
    expected_rel=cfg['expected_predictions'].split(REMOTE_ROOT+'/',1)[1]
    numerical={name:sha(ROOT/name) for name in ('code/benchmark_speed.py','code/evaluate.py','code/cache_bridge.py',
        'code/attention_bridge.py','code/ownership.py','code/m1_protocol.py')}
    contract={'schema':'baseline_matrix.canonical_text2_evidence_gate.v1','producer_name':smokejob['name'],
        'producer_job':smoke['job'],'producer_job_sha256':sha(ROOT/smoke['job']),
        'producer_speed_config':smokecfg_rel,'producer_speed_config_sha256':sha(ROOT/smokecfg_rel),
        'producer_output':smokejob['argv'][smokejob['argv'].index('--output')+1].split(REMOTE_ROOT+'/',1)[1],
        'required_fields':deepcopy(smoke['completion_contract']['required_fields']),
        'hash_fields':deepcopy(smoke['completion_contract']['hash_fields']),
        'expected_ids':['textvqa:validation:39023','textvqa:validation:36209'],
        'profile_method_ids':[read(ROOT/p)['method']['id'] for p in nested],
        'eval_config_sha256':nested,'expected_predictions':expected_rel,'expected_predictions_sha256':sha(ROOT/expected_rel),
        'numerical_source_sha256':numerical,'historical_E8_smoke_forbidden':True}
    gate_contract='configs/gates/M1_canonical_E64_Text2_prerequisite_'+version+'.json'
    gate_name='bm_M1_canonical_E64_Text2_evidence_CPU_'+version+'_sourcev2'
    gate_output='results/canonical_source_timing_evidence/'+version+'/Text2_gate_sourcev2.json'
    gate_job={'name':gate_name,'env':cache_environment(''),'argv':['/usr/bin/python3','-B',
        REMOTE_ROOT+'/code/gates/canonical_text2_prerequisite_v2.py','--root',REMOTE_ROOT,
        '--contract',REMOTE_ROOT+'/'+gate_contract,'--output',REMOTE_ROOT+'/'+gate_output]}
    gate_rel='configs/jobs/'+gate_name+'.json'
    gate_node={'job':gate_rel,'job_sha256':None,'execution_resource':'cpu','on_failure':'continue',
        'extension_role':'CPU_exact_canonical_Text2_producer_four_raw_hash_gate_no_GPU',
        'completion_contract':{'type':'json_status','accepted_status':'PASS_CANONICAL_E64_TEXT2_PREREQUISITE',
            'required_fields':{'CPU_only':True,'GPU_used':False,'verified_exact_producer':True,
                'verified_hash_fields':True,'verified_required_fields':True,'recorded_trials':40,'expected_trials':40}}}
    formal_node={'job':alias_rel,'job_sha256':None,'on_failure':'continue',
        'extension_role':'canonical_E64_Text16_formal320_only_not_legacy_or_historical_E8',
        'require_json_status':[{'path':gate_output,'producer_name':gate_name,
            'accepted_status':'PASS_CANONICAL_E64_TEXT2_PREREQUISITE',
            'required_fields':{'producer_name':smokejob['name'],'producer_job_sha256':sha(ROOT/smoke['job']),
                'verified_exact_producer':True,'verified_hash_fields':True,'verified_required_fields':True,
                'recorded_trials':40,'expected_trials':40,'method_profile':'canonical_E64_source_v2'}}],
        'completion_contract':{'type':'json_status','accepted_status':'PASS_TIMED_AND_TOKEN_VALIDATED',
            'required_fields':{'GPU_used':True,'recorded_trials':320,'expected_trials':320,'warmup_and_measure_status_counts':{'PASS':320}},
            'hash_fields':{'manifest.json':'manifest_sha256','records.jsonl':'records_jsonl_sha256','summary.json':'summary_sha256'}}}
    production={alias_rel:alias,gate_rel:gate_job,gate_contract:contract}
    stages={'code/gates/canonical_text2_prerequisite_v2.py',*production}
    speedrel=source['argv'][source['argv'].index('--config')+1].split(REMOTE_ROOT+'/',1)[1]
    speed=read(ROOT/speedrel);stages.add(speedrel);stages.add(speed['expected_predictions'].split(REMOTE_ROOT+'/',1)[1])
    for p in speed['eval_configs']:stages.add(p.split(REMOTE_ROOT+'/',1)[1])
    return {'production':production,'stage':stages,'new_nodes':[gate_node,formal_node],
        'producer':{'job':smoke['job'],'name':smokejob['name'],'completion_contract':smoke['completion_contract']},
        'formal_original':source,'formal_alias':alias_rel,'gate_report':gate_output}


def build_plan(old,paused,new_nodes,producer,formal_original,job_loader,allow_fixture=False):
    assert paused['status']=='CPU_SCHEDULER_STOPPED_GPU_WORKERS_UNTOUCHED'
    assert allow_fixture or paused.get('CPU_fixture') is not True
    assert set(paused['state']['lanes'])==set(old['lanes'])=={'0','1'}
    plan=deepcopy(old);plan['wait_for_existing']={};pending={};active={}
    producer_places=[]
    for gpu,lane in paused['state']['lanes'].items():
        index=lane['next'];assert type(index) is int and 0<=index<=len(old['lanes'][gpu])
        pending[gpu]=deepcopy(old['lanes'][gpu][index:]);plan['lanes'][gpu]=deepcopy(pending[gpu])
        current=lane.get('active')
        if current:
            descriptor={k:deepcopy(current[k]) for k in ('name','completion_contract','on_failure') if k in current}
            plan['wait_for_existing'][gpu]=descriptor;active[gpu]=descriptor
            assert current['name']!=formal_original['name'] and not current['name'].startswith(formal_original['name']+'_GPU')
            if current['name']==producer['name']:
                assert gpu=='0' and current.get('completion_contract',{})==producer['completion_contract']
                producer_places.append('active')
        if any(r['name']==producer['name'] for r in lane.get('completed',[])):
            assert gpu=='0';producer_places.append('completed')
        for i,node in enumerate(old['lanes'][gpu]):
            job=job_loader(node['job'])
            assert job['argv']!=formal_original['argv'],'Formal output/argv already scheduled; do not duplicate'
            if node['job']==producer['job'] and i>=index:
                assert gpu=='0';producer_places.append('pending')
    assert len(producer_places)==1,'Canonical v20 producer must be exactly pending, active or terminal in the pause state'
    where=producer_places[0]
    insertion=next((i+1 for i,n in enumerate(plan['lanes']['0']) if n['job']==producer['job']),0) if where=='pending' else 0
    plan['lanes']['0'][insertion:insertion]=deepcopy(new_nodes)
    for gpu in pending:
        old_keys={n['job'] for n in pending[gpu]}
        assert [n for n in plan['lanes'][gpu] if n['job'] in old_keys]==pending[gpu]
    assert plan['lanes']['1']==pending['1'] and plan['wait_for_existing']==active
    return plan,{'insertion_mode':where,'insertion_index_GPU0':insertion,
        'old_pending_nodes':sum(len(v) for v in pending.values()),
        'old_pending_by_lane_sha256':{g:canonical(v) for g,v in pending.items()},
        'active_descriptors':active,'new_GPU_trials':320,'new_CPU_evidence_nodes':1,
        'all_old_node_objects_and_relative_order_preserved':True,'gate_failure_skips_only_formal320':True}


def prepare(version,pause_path=None,prepare_only=False):
    info=additions(version)
    for relative,value in info['production'].items():write(ROOT/relative,value)
    for node in info['new_nodes']:node['job_sha256']=sha(ROOT/node['job'])
    dependencies={relative:sha(ROOT/relative) for relative in sorted(info['stage'])}
    dependency_report={'status':'CANONICAL_TEXT320_CPU_DEPENDENCIES_PREPARED_NOT_STAGED_OR_LAUNCHED',
        'version':version,'formal_original':SOURCE_JOB,'formal_GPU0_alias':info['formal_alias'],
        'only_formal_alias_changes':['name','env.CUDA_VISIBLE_DEVICES'],'stage_files_sha256':dependencies,
        'new_GPU_trials':320,'new_CPU_evidence_nodes':1,'canonical_producer':info['producer'],
        'no_Text2_duplicate':True,'Supervisor_unchanged_sha256':sha(ROOT/'code/queue_supervisor.py'),
        'helper_source_sha256':sha(__file__),'stage_launch_pause_or_queue_mutation':False,
        'failure_semantics':'CPU gate runs after the actual canonical producer even if no marker; it writes PASS/FAIL. Optional formal requires the CPU gate PASS, so failed/missing gate skips only formal and cannot wait forever.'}
    write(ROOT/f'audit/canonical_Text320_{version}_dependencies_gatev2.json',dependency_report)
    if prepare_only:return dependency_report
    assert pause_path is not None
    paused=read(pause_path);assert re.fullmatch(r'[A-Za-z0-9_-]+',paused['controller'])
    oldrel='configs/queues/'+paused['controller']+'.json';old=read(ROOT/oldrel)
    for nodes in old['lanes'].values():
        for node in nodes:assert sha(ROOT/node['job'])==node['job_sha256']
    plan,proof=build_plan(old,paused,info['new_nodes'],info['producer'],info['formal_original'],lambda p:read(ROOT/p))
    name='External_baselines_P2_canonical_Text16_continuation_'+version
    plan['preserved_from']={'plan':oldrel,'sha256':sha(ROOT/oldrel),'pause_receipt':str(pause_path),
        'pause_receipt_sha256':sha(pause_path),'source_controller':paused['controller']}
    planrel='configs/queues/'+name+'.json';write(ROOT/planrel,plan)
    controller=deepcopy(read(ROOT/('configs/jobs/bm_'+paused['controller']+'.json')))
    controller['name']='bm_'+name;controller['argv'][controller['argv'].index('--plan')+1]=REMOTE_ROOT+'/'+planrel
    controller['argv'][controller['argv'].index('--output')+1]=REMOTE_ROOT+'/results/queues/'+name
    ctrlrel='configs/jobs/bm_'+name+'.json';write(ROOT/ctrlrel,controller)
    dependency_report.update(status='PASS_CANONICAL_TEXT320_CONTINUATION_PREPARED_NOT_STAGED_OR_LAUNCHED',
        controller=name,plan=planrel,plan_sha256=sha(ROOT/planrel),controller_job=ctrlrel,
        pause_receipt=str(pause_path),pause_receipt_sha256=sha(pause_path),conservation=proof)
    dependency_report['stage_files_sha256'].update({planrel:sha(ROOT/planrel),ctrlrel:sha(ROOT/ctrlrel)})
    write(ROOT/f'audit/canonical_Text320_{version}_preparation_gatev2.json',dependency_report)
    return dependency_report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v21');parser.add_argument('--pause-receipt',type=Path)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    result=prepare(args.version,args.pause_receipt,args.prepare_only)
    print(json.dumps({k:result[k] for k in ('status','version','new_GPU_trials','new_CPU_evidence_nodes','stage_launch_pause_or_queue_mutation')}))
