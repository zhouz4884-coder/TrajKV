"""Prepare confirmation descriptors only after both completed D96 decisions.

Default operation reports readiness. --prepare is an explicit local JSON-only
commit after every admission/source check succeeds. This generator does not
choose modes, inspect R128 outcomes, use a model/GPU/network, stage or launch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ops'),str(ROOT/'code')]
import round11_jobs_v1 as jobs
import round11_confirmation_runner_v2 as runner
import round11_confirmation_harvest_v2 as adapter

E = Path('${EVIDENCE_ROOT}/results/round11')
DATA = E/'confirmation_inputs_v1/remote_materialization_v1/inputs'
BOOTSTRAP = E/'bootstrap_v1'
REFERENCE_PLAN = E/'driver_cpu_v6/reference_only_plan.json'
GPU_DOMAINS = {0: ('textvqa','docvqa'), 1: ('infovqa','chartqa')}
PINS = {
    'code/round11_confirmation_runner_v2.py': '3db28f74add8ba98b5ea1cff928b4aada775408cde69c5629dbe0b2b75e25aca',
    'ops/round11_confirmation_harvest_v2.py': '6435fee9fffc38b873c73b82d9dcc82010c73c1acb123c2d7f6c47f46ea4ba9e',
    'code/round10_sequence_v1.py': '2d0dce9da80156fb0224a04d48a9a3e3869ede5e9885885a6a25073ab511fe00',
}
PROTOCOL_KEYS = ('model_id','model_revision','model_path','model_source_lock','max_new_tokens',
    'question_suffix','preprocessing','seed','position_owner_layout','evaluation_contract',
    'variant','variant_source','expected_asset_sha256','legacy_a2','legacy_experiments','legacy_scripts','kivi_source')


def require(value, message):
    if not value:raise ValueError('CONFIRMATION_PREPARATION: '+message)


def canonical_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def validate_development(role, path):
    summary=jobs.read(path); counts=dict(textvqa=32,infovqa=32,docvqa=16,chartqa=16)
    modes=runner.primary.MODES if role=='primary' else runner.wo.MODES
    views,fresh,schema,status=(28,384,'trajkv.round11.harvest_summary.v1','PASS_ALL_28_PROVEN_VIEWS') if role=='primary' else (20,192,'trajkv.round11_wo.harvest_summary.v1','PASS_ALL_20_PROVEN_VIEWS')
    require(summary.get('schema')==schema and summary.get('status')==status
        and summary.get('expected_cells_or_views')==summary.get('closed_cells_or_views')==summary.get('successful_complete_cells_or_views')==views
        and summary.get('expected_new_free_generation_QA')==summary.get('actual_closed_new_free_generation_QA')==fresh
        and summary.get('expected_reused_reference_views')==summary.get('actual_closed_reused_reference_views')==288
        and summary.get('source_group_manifest_sha256')==runner.D96_SOURCE_GROUP_SHA
        and summary.get('canonical_SourceRun_admission_claimed') is False, role+' D96 must be completely source-proven before any writes')
    planpath=BOOTSTRAP/('HARVEST_PLAN_v2.json' if role=='primary' else 'WO_HARVEST_PLAN_v1.json')
    require(summary['plan_sha256']==jobs.sha(planpath), role+' summary must bind the actual frozen execution plan')
    points=summary['points'];keys={(r['dataset'],r['mode']) for r in points}
    require(len(points)==len(keys)==views and keys=={(d,m) for d in counts for m in (*runner.REFERENCES,*modes)}
        and all(r['status']=='PASS_FULL_SOURCE_GROUP_SCORE' and r['expected_QA']==counts[r['dataset']]
                and r['primary_mean'] is not None
                and r['fresh_free_generation_QA']==(counts[r['dataset']] if r['mode'] in modes else 0)
                and r['reused_reference_views']==(counts[r['dataset']] if r['mode'] in runner.REFERENCES else 0)
                for r in points), role+' D96 contains an incomplete/wrong-domain/mode score')
    return {'path':str(path),'sha256':jobs.sha(path),'schema':schema,'actual_new_QA':fresh,'reused_views':288}


def reference_templates():
    plan=jobs.read(REFERENCE_PLAN)
    entries={(e['dataset'],e['mode']):e for e in plan['entries']}
    require(len(entries)==12 and all(e['kind']=='reused' for e in entries.values()), 'twelve original reference producer entries required')
    result={};provenance={}
    for key,e in entries.items():
        raw=Path(e['source_raw_dir']);cfgpath=raw/'source_config.json';m=jobs.read(raw/'manifest.json')
        require(jobs.sha(raw/'source_retrieval_receipt_round10_v1.json')==e['source_receipt_sha256']
            and jobs.sha(cfgpath)==e['source_item']['cell']['config_sha256']==m['config_sha256'], 'original reference producer/config SHA changed')
        cfg=jobs.read(cfgpath);require(cfg==m['config'], 'reference producer config snapshot differs')
        require(cfg['max_new_tokens']==64 and cfg['preprocessing']['max_patches']==8
            and cfg['question_suffix']==runner.frozen.SUFFIX and cfg['seed']==0 and cfg['model_id']=='M1', 'original native protocol differs')
        result[key]=cfg;provenance[key]={'source_config':str(cfgpath),'source_config_sha256':jobs.sha(cfgpath),
            'producer_manifest_sha256':jobs.sha(raw/'manifest.json'),'source_receipt_sha256':e['source_receipt_sha256']}
    return result,provenance


def readiness(selection_path, selection_sha, primary_summary, wo_summary):
    # This entire function is read-only. No output directory, selection copy,
    # config, sequence or job is created until all of it has passed.
    for path,digest in PINS.items():require(jobs.sha(ROOT/path)==digest, 'frozen source epoch differs: '+path)
    graph=runner.runtime_source_manifest();require(len(graph)==24, 'exact24 confirmation runtime roles required')
    require(all(re.fullmatch('[0-9a-f]{64}',h) and h!='0'*64 and jobs.sha(ROOT/p)==h for p,h in graph.items()), 'placeholder/missing runtime source SHA')
    basis={'primary':validate_development('primary',primary_summary), 'wo':validate_development('wo',wo_summary)}
    require(re.fullmatch('[0-9a-f]{64}',selection_sha) is not None and jobs.sha(selection_path)==selection_sha, 'Root selection exact SHA required')
    selection=jobs.read(selection_path)
    identity=runner.selection_identity({'selection_manifest_path':str(selection_path),'selection_manifest_sha256':selection_sha,
        'candidate':{'mode':'FP16'},'variant_source':{'name':'r7_joint_k1_gt_cond'}})
    require(identity['development_harvest_summary_sha256']=={key:value['sha256'] for key,value in basis.items()}, 'Root selection must bind both actual completed D96 summary bytes')
    selected=identity['selected_modes'];require(1<=len(selected)<=2, 'this generator prepares4 or5 methods, requiring1 or2 Root-selected candidates')
    manifest_path=DATA/'CONFIRM128_INPUT_MANIFEST_v1.json';manifest=jobs.read(manifest_path)
    require(jobs.sha(manifest_path)==runner.MANIFEST_SHA and manifest['counts']==runner.COUNTS and manifest['total_count']==128, 'frozen R128 no-answer manifest differs')
    domains={}
    for ds,n in runner.COUNTS.items():
        entry=manifest['input_files'][f'{ds}_confirm{n}'];ref=manifest['reference_files'][f'{ds}_confirm{n}']
        local_inputs=DATA/Path(entry['path']).name;local_refs=DATA/Path(ref['path']).name
        values=runner.data.load_confirmation_inputs(local_inputs,manifest_path,entry['sha256'])
        require(len(values)==n and {row['dataset'] for row in values}=={ds}, 'R128 domain input identity/count differs')
        require(jobs.sha(local_refs)==ref['sha256'] and local_refs.stat().st_size==ref['bytes'], 'offline frozen references differ')
        lock=DATA/Path(manifest['source_lock']['path']).name
        require(jobs.sha(lock)==manifest['source_lock']['sha256'], 'R128 source lock differs')
        domains[ds]={'inputs':str(local_inputs),'inputs_sha256':entry['sha256'],'refs':str(local_refs),'refs_sha256':ref['sha256'],
                     'source_lock':str(lock),'source_lock_sha256':manifest['source_lock']['sha256']}
    templates,provenance=reference_templates()
    return {'selection':selection,'selection_bytes':Path(selection_path).read_bytes(),'selection_sha256':selection_sha,
        'selected_modes':selected,'development_basis':basis,'manifest':manifest,'manifest_path':str(manifest_path),
        'domains':domains,'runtime':graph,'templates':templates,'template_provenance':provenance}


def prepare(ready):
    modes=(*runner.REFERENCES,*ready['selected_modes']);expected=128*len(modes)
    require(expected in (512,640), 'balanced confirmation count must be512 or640')
    selected_relative='configs/round11_v1/CONF2_SELECTION_MANIFEST_v2.json'
    selected_local=ROOT/selected_relative;selected_remote=jobs.REMOTE_ROOT+'/'+selected_relative
    writes={selected_local:ready['selection_bytes']};cells_by_gpu={};configs=[];sequences={};job_descriptors={}
    for gpu,datasets in GPU_DOMAINS.items():
        cells=[]
        for ds in datasets:
            n=runner.COUNTS[ds];entry=ready['manifest']['input_files'][f'{ds}_confirm{n}']
            for mode in modes:
                original=ready['templates'][ds,mode if mode in runner.REFERENCES else 'R7low'];config=deepcopy(original)
                logical= 'original' if mode in ('R7low','R7high') else mode
                config.update(schema=runner.SCHEMA,inputs=entry['path'],inputs_sha256=entry['sha256'],start=0,count=n,input_count=n,
                    input_manifest_path=jobs.REMOTE_ROOT+'/configs/round11_inputs_v1/CONFIRM128_INPUT_MANIFEST_v1.json',
                    input_manifest_sha256=runner.MANIFEST_SHA,input_role='confirmation_holdout',cohort_kind='confirmation_holdout',
                    candidate={'mode':logical},expected_source_hashes=ready['runtime'],
                    selection_manifest_path=selected_remote,selection_manifest_sha256=ready['selection_sha256'],
                    scope='R128 confirmation; Root-selected modes frozen from both completed D96 groups before outcomes')
                config['method']={**original['method'],'id':'Round11_R128_'+mode+'_v2','family':'trajkv_round11_confirmation'}
                require(all(config[key]==original[key] for key in PROTOCOL_KEYS), 'original model/template/native protocol/variant changed')
                name='CONF2_'+mode+'_'+ds+'_v1';relative='configs/round11_v1/'+name+'.json';body=canonical_bytes(config)
                writes[ROOT/relative]=body;digest=hashlib.sha256(body).hexdigest()
                cell={'name':name,'dataset':ds,'mode':mode,'script':'code/round11_confirmation_runner_v2.py',
                    'script_sha256':ready['runtime']['code/round11_confirmation_runner_v2.py'],
                    'config':jobs.REMOTE_ROOT+'/'+relative,'config_sha256':digest,
                    'output':jobs.REMOTE_ROOT+'/results/round11_v1/confirmation/'+name,'expected_QA':n}
                cells.append(cell);configs.append({'path':str(ROOT/relative),'sha256':digest,
                    'template':ready['template_provenance'][ds,mode if mode in runner.REFERENCES else 'R7low']})
        require(sum(cell['expected_QA'] for cell in cells)==64*len(modes), 'balanced whole-domain allocation differs')
        require(len(cells)==2*len(modes) and len({cell['name'] for cell in cells})==len(cells), 'duplicate baseline or candidate cell')
        cells_by_gpu[gpu]=cells;seqrel=f'configs/round10_v1/R11_confirm_gpu{gpu}_v2.json'
        seq={'schema':'trajkv.round10.sequence.v1','scope':'R128 fresh confirmation with Root-frozen selection v2','expected_QA':64*len(modes),'entries':cells}
        writes[ROOT/seqrel]=canonical_bytes(seq);seqsha=hashlib.sha256(writes[ROOT/seqrel]).hexdigest()
        sequences[gpu]={'path':str(ROOT/seqrel),'remote':jobs.REMOTE_ROOT+'/'+seqrel,'sha256':seqsha}
        name=f'bm_R11_confirm_gpu{gpu}_v2';job={'name':name,'env':jobs.env(gpu),'argv':[jobs.MAIN_PYTHON,'-B',jobs.REMOTE_ROOT+'/code/round10_sequence_v1.py',
            '--config',jobs.REMOTE_ROOT+'/'+seqrel,'--output',jobs.REMOTE_ROOT+f'/results/round11_v1/queues/confirm_gpu{gpu}_v2']}
        path=ROOT/'configs/jobs'/(name+'.json');writes[path]=canonical_bytes(job)
        job_descriptors[gpu]={'name':name,'path':str(path),'sha256':hashlib.sha256(writes[path]).hexdigest(),'job':job}
    require(sum(len(v) for v in cells_by_gpu.values())==4*len(modes), 'each domain/method must occur exactly once')
    entries=[]
    for gpu,cells in cells_by_gpu.items():
        for cell in cells:
            entries.append({'kind':'fresh','dataset':cell['dataset'],'mode':cell['mode'],'item':{
                'root':jobs.REMOTE_ROOT,'job':job_descriptors[gpu]['job'],'job_sha256':job_descriptors[gpu]['sha256'],
                'cell':cell,'sequence_sha256':sequences[gpu]['sha256'],'sequence_script_sha256':PINS['code/round10_sequence_v1.py'],
                'input_manifest_sha256':runner.MANIFEST_SHA,'selection_manifest_sha256':ready['selection_sha256']}})
    plan={'schema':'trajkv.round11.confirmation_harvest_plan.v2','input_manifest_path':ready['manifest_path'],
        'input_manifest_sha256':runner.MANIFEST_SHA,'selection_manifest_path':str(selected_local),
        'selection_manifest_sha256':ready['selection_sha256'],'domains':ready['domains'],'entries':entries,
        'expected_fresh_reference_QA':384,'expected_total_new_QA':expected,'reused_reference_views':0}
    planpath=BOOTSTRAP/'CONFIRM_HARVEST_PLAN_v2.json';writes[planpath]=canonical_bytes(plan)
    plan_sha=hashlib.sha256(writes[planpath]).hexdigest()
    receipt={'schema':'trajkv.round11.confirmation_preparation.v2','status':'PASS_LOCAL_DESCRIPTORS_NO_LAUNCH',
        'source_script_sha256':jobs.sha(__file__),'selected_modes':ready['selected_modes'],'modes':list(modes),
        'selection_manifest_sha256':ready['selection_sha256'],'development_admission':ready['development_basis'],
        'input_manifest_sha256':runner.MANIFEST_SHA,'runtime_source_hashes':ready['runtime'],
        'harvest_adapter':{'path':str(ROOT/'ops/round11_confirmation_harvest_v2.py'),'sha256':PINS['ops/round11_confirmation_harvest_v2.py']},
        'expected_new_QA':expected,'fresh_reference_QA':384,'reused_reference_views':0,'balanced_new_QA_per_GPU':64*len(modes),
        'queues':[{'GPU':gpu,'domains':list(GPU_DOMAINS[gpu]),'sequence':sequences[gpu],'job':job_descriptors[gpu],'cells':cells_by_gpu[gpu]} for gpu in GPU_DOMAINS],
        'config_receipts':configs,'harvest_plan_path':str(planpath),'harvest_plan_sha256':plan_sha,
        'selection_performed_by_generator':False,'R128_outcomes_read':False,'new_GPU_jobs_started':0,'SSH_used':False,'Git_used':False}
    path=BOOTSTRAP/'CONFIRM_JOB_PLAN_v2.json';writes[path]=canonical_bytes(receipt)
    digest=hashlib.sha256(writes[path]).hexdigest()
    # Include both final plan artifacts in the same preflight as every input
    # copy/config/sequence/job. A stale final plan must fail before any write.
    for target,body in writes.items():
        require(not target.exists() or target.read_bytes()==body, 'immutable descriptor conflict: '+str(target))
    for target,body in writes.items():
        if target==selected_local:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with target.open('xb') as stream:stream.write(body)
        else:
            require(jobs.freeze(target,json.loads(body))==hashlib.sha256(body).hexdigest(), 'root JSON freeze differs')
    # Root helpers and the frozen adapter independently validate the committed
    # JSON. Every possible existing-target conflict was checked above.
    for gpu in GPU_DOMAINS:
        actual=jobs.job(job_descriptors[gpu]['name'],'code/round10_sequence_v1.py',job_descriptors[gpu]['job']['argv'][3:],gpu)
        require(actual==job_descriptors[gpu], 'root job helper descriptor differs')
    adapter.load_plan(planpath)
    return {'status':receipt['status'],'new_QA':expected,'methods':len(modes),'plan':str(path),'sha256':digest,'harvest_plan_sha256':plan_sha}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256',required=True)
    p.add_argument('--primary-summary',type=Path,default=E/'summary_final_primary_v4/summary.json')
    p.add_argument('--wo-summary',type=Path,default=E/'wo_summary_final_v1/summary.json')
    p.add_argument('--prepare',action='store_true',help='Write local descriptors only after all read-only admission checks pass')
    a=p.parse_args();ready=readiness(a.selection,a.selection_sha256,a.primary_summary,a.wo_summary)
    if a.prepare:result=prepare(ready)
    else:result={'status':'READY_NO_WRITES','selected_mode_count':len(ready['selected_modes']),'expected_new_QA':128*(3+len(ready['selected_modes'])),'descriptor_files_written':0,'new_GPU_jobs_started':0}
    print(json.dumps(result,ensure_ascii=False))
