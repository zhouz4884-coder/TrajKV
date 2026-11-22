"""Explicit Round10 stage cutoff, custom-driver closure and privacy inventory.

Preparation only: a pending cell cannot become a score or a publishable stage.
Root seals each of the original 28 dispositions, including stopped/unrun work.
The source inventory is explicit; mutable project directories are never swept.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path('${EVIDENCE_ROOT}')
SCHEMA='trajkv.round10.public_stage_contract.v1'
SEAL_SCHEMA='trajkv.round10.public_stage_seal.v1'
BASE_COMMIT='77df2e0f1f950c061be5a190a2d472dae6ae880a'
BASE_TREE='f0b31ea6fcb613c1ed21c3eee3ba0c4e684444be'
BASE_ARCHIVE_SHA='eddec7b99a9c412c8c986e2eb4a03136d10a92503d99f54b702629200579810a'
DOMAINS=('textvqa','infovqa','docvqa','chartqa')
COUNTS={'textvqa':64,'infovqa':64,'docvqa':16,'chartqa':16}
MODES=('FP16','R7low','R7high','LS-FZ','LS-RE','FB0','FB-W')
FINAL_STATES=('CLOSED_COMPLETE','CLOSED_WITH_FAILURES','STOPPED','NOT_RUN')
SAFE_SUFFIXES={'.py','.json','.jsonl','.csv','.md','.txt','.toml','.yaml','.yml'}
REQUIRED_ROLES={'scope_report','quality_bytes_table','instrumented_timing_table','input_manifest','source_lock',
                'reference_answers','source_inputs','frozen_metric_source','frozen_scoring_source'}


def require(value,message):
    if not value:raise ValueError('ROUND10_PUBLIC: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def inside(path,root):return Path(path).resolve().is_relative_to(Path(root).resolve())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def declared_rows():
    return [{'dataset':ds,'mode':mode,'expected_QA':COUNTS[ds],'disposition':'PENDING'} for ds in DOMAINS for mode in MODES]


def candidate():
    return {'schema':SCHEMA,'phase_label':'ROUND10_V_REENCODING_WINDOW_FEEDBACK','stage_state':'PENDING_REAL_RESULTS',
        'expected_cells':28,'expected_free_answer_events':1120,'expected_distinct_development_sources':160,
        'SourceRun_admission':False,'whole_Round10_goal_complete_claimed':False,
        'source_public_Git_LF_archive_identities_separate':True,'scope':'D160 development, not official500 or blind confirmation.',
        'base_public_delivery':{'commit':BASE_COMMIT,'tree':BASE_TREE,'archive_sha256':BASE_ARCHIVE_SHA,
            'history_and_all_previous_namespaces_releases_must_be_preserved':True},
        'root_seal':None,'plans':[],'harvest_root':None,'refs_root':None,'cells':declared_rows(),
        'calibration':{'disposition':'PENDING','raw':None,'independent_source_proof':None},
        'files':[],'no_private_tensors_images_weights_parquet_chat_or_credentials':True,
        'packager_transport_dependencies_publicly_rerunnable':False,
        'offline_verifier_requires_no_private_transport':True}


def file_allowed(path,role):
    path=Path(path);lower=path.name.lower()
    require(path.suffix.lower() in SAFE_SUFFIXES or lower.startswith(('license','copying','notice')),'non-text tensor/image/weight/parquet/archive file is not public support')
    require(not any(x.lower() in {'.git','__pycache__','models','weights','images','cache','tmp','private_fit_shards','private_select_teachers'} for x in path.parts),'private or cache directory')
    require(lower not in ('chat.md','gpt40_reply.md') and not any(x in lower for x in ('对话记录','doubao','私聊')),'private discussion is not public evidence')
    require(isinstance(role,str) and role and not role.startswith('private_'),'explicit public evidence role required')


def validate(contract_path,deep=False):
    contract_path=Path(contract_path);c=load(contract_path)
    require(c.get('schema')==SCHEMA and re.fullmatch('[A-Za-z0-9_.-]+',c['phase_label']),'explicit Round10 public contract')
    require(c.get('SourceRun_admission') is False and c.get('expected_cells')==28 and c.get('expected_free_answer_events')==1120 and
            c.get('whole_Round10_goal_complete_claimed') is False,'stage scope/admission differs')
    rows=c['cells'];require(len(rows)==28 and {(r['dataset'],r['mode']) for r in rows}=={(d,m) for d in DOMAINS for m in MODES},'exact original28 scope required')
    require(all(r['expected_QA']==COUNTS[r['dataset']] and r['disposition'] in (*FINAL_STATES,'PENDING','RUNNING') for r in rows),'invalid cell count or disposition')
    missing=[]
    if c.get('stage_state')!='ROOT_SEALED':missing.append('Root final stage seal')
    for r in rows:
        if r['disposition'] not in FINAL_STATES:missing.append(r['dataset']+'/'+r['mode']+':actual final disposition')
    if c['calibration']['disposition'] not in ('CLOSED_COMPLETE','STOPPED'):missing.append('calibration actual final disposition')
    if not c.get('root_seal'):missing.append('root_seal path/SHA')
    readiness={'status':'WAIT_ROUND10_ROOT_SEAL_AND_REAL_RESULTS' if missing else 'READY_FOR_DEEP_LOCAL_VALIDATION',
        'missing':missing,'publication_eligible':False,'declared_cells':28,'declared_free_answer_events':1120,
        'dispositions':dict(Counter(r['disposition'] for r in rows)),'quality_QA_inferred_from_plans':0,'SourceRun_admission':False}
    if missing:return readiness,{},c
    sealref=c['root_seal'];require(sha(sealref['path'])==sealref['sha256'],'Root seal bytes changed');seal=load(sealref['path'])
    require(seal['schema']==SEAL_SCHEMA and seal['status']=='SEALED_FOR_LOCAL_PUBLIC_PACKAGE' and seal['phase_label']==c['phase_label'] and
            seal['cells_disposition_sha256']==canonical(rows) and seal['calibration_disposition_sha256']==canonical(c['calibration']),
            'Root seal does not bind complete disposition lists')
    require(c.get('no_private_tensors_images_weights_parquet_chat_or_credentials') is True,'public privacy declaration required')
    files={};roles=set()
    def add(path,role,expected=None,public_path=None,aliases=()):
        path=Path(path).resolve();require(inside(path,ROOT) or inside(path,EVIDENCE),'source must be project or E mirror')
        file_allowed(path,role);require(path.is_file(),'missing required source: '+str(path))
        digest=sha(path);require(expected is None or digest==expected,'declared source SHA differs: '+str(path))
        origin=ROOT if inside(path,ROOT) else EVIDENCE
        rel=public_path or path.relative_to(origin).as_posix();p=PurePosixPath(rel)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in rel,'unsafe public relative path')
        info=path.stat()
        require(path.suffix=='.jsonl' or info.st_size<=64*1024**2,'large non-JSONL support requires a separate review')
        item={'path':path,'role':role,'origin':'project' if origin==ROOT else 'evidence_mirror',
            'source_path':path.relative_to(origin).as_posix(),'stat':(info.st_size,info.st_mtime_ns,info.st_ctime_ns),
            'large':path.suffix=='.jsonl','aliases':list(aliases)}
        if rel in files:
            require(sha(files[rel]['path'])==digest,'different sources at same public path');files[rel]['aliases']+=list(aliases)
        else:files[rel]=item
        roles.add(role);return rel
    add(contract_path,'sealed_stage_contract',public_path='evidence/ROUND10_STAGE_CONTRACT.json')
    add(sealref['path'],'root_stage_disposition_seal',sealref['sha256'],public_path='evidence/ROUND10_STAGE_SEAL.json')
    for f in c['files']:add(f['path'],f['role'],f['sha256'],f.get('public_path'),f.get('source_aliases',[]))
    require(REQUIRED_ROLES<=roles,'required source/refs/report roles missing: '+str(sorted(REQUIRED_ROLES-roles)))
    require(c.get('refs_root') and sha(Path(c['refs_root'])/'source_lock.json') in [sha(v['path']) for v in files.values() if v['role']=='source_lock'],'scoring refs root is not archived')
    sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'code')]
    import round10_harvest_score_v1 as harvest
    harvest.check_pins()
    for p in c['plans']:require(sha(p['path'])==p['sha256'],'declared execution plan changed');add(p['path'],'execution_plan',p['sha256'])
    items,origins=harvest.load_plans([Path(p['path']) for p in c['plans']]) if c['plans'] else ({},[])
    closed=[];failures=[];dispositions=[]
    for r in rows:
        state=r['disposition'];key=(r['dataset'],r['mode']);n=r['expected_QA']
        if state in ('STOPPED','NOT_RUN'):
            evidence=r.get('disposition_evidence');require(evidence and isinstance(r.get('reason'),str) and r['reason'],'stopped/unrun needs a factual public reason and hashed evidence')
            ref=add(evidence['path'],'actual_stop_or_nonexecution_disposition',evidence['sha256'])
            require(r.get('primary_mean') is None and r.get('quality_QA') in (None,0),'stopped/unrun cannot expose partial primary quality')
            dispositions.append({**r,'disposition_evidence_file':ref,'quality_mean':None});continue
        require(key in items and c.get('harvest_root'),'actual executed child plan required')
        item=items[key];name=item['cell']['name'];raw=Path(c['harvest_root'])/'raw'/name;scored=Path(c['harvest_root'])/'scored'/name
        m,d,cfg,proof=harvest.check_local_proof(raw,item)
        require(d['expected_QA']==d['recorded_QA']==n,'whole-domain raw counts differ')
        for e in proof['files']:add(raw/e['file'],'executed_child_source_or_raw',e['sha256'],aliases=[e['remote_file']])
        for name in ('source_retrieval_receipt_round10_v1.json','retrieval_receipt.json'):add(raw/name,'independent_remote_source_proof')
        if state=='CLOSED_COMPLETE':
            require(d['status']=='PASS_RAW_COMPLETE' and d['counts']=={'PASS':n},'complete disposition is not actual all-QA PASS')
        else:require(d['status']=='COMPLETE_WITH_FAILURES','failure disposition without actual failures')
        require((scored/'summary.json').is_file(),'actual frozen score required, no export-time invented score')
        summary=load(scored/'summary.json')
        require(summary['expected_run_questions']==n and summary['integrity']['scorer_sha256']==harvest.PINS['code/score.py'],'wrong scorer or count')
        require((summary['status']=='PASS_SCORED_COMPLETE')==(state=='CLOSED_COMPLETE') and
                (summary['primary_mean'] is not None)==(state=='CLOSED_COMPLETE'),'failure/complete primary mismatch')
        for name in ('summary.json',*summary['derived_file_hashes']):
            add(scored/name,'frozen_offline_score',None if name=='summary.json' else summary['derived_file_hashes'][name])
        if deep:
            point,bundle=harvest.closed_cell(raw,scored,item,Path(c['refs_root']))
            require((point['primary_mean'] is not None)==(state=='CLOSED_COMPLETE'),'deep source/score/physical gate differs')
        ref={'dataset':r['dataset'],'mode':r['mode'],'disposition':state,'expected_QA':n,
            'raw':str(raw),'scored':str(scored),'primary_mean':summary['primary_mean'],'config_sha256':m['config_sha256']}
        (closed if state=='CLOSED_COMPLETE' else failures).append(ref)
    calibration=c['calibration'];cal=Path(calibration['raw'])
    proofref=calibration.get('independent_source_proof');require(proofref,'calibration source retrieval proof required')
    proof=load(proofref['path']);add(proofref['path'],'calibration_independent_source_proof',proofref['sha256'])
    require(proof.get('remote_hashes_computed_independently') is True or proof.get('independent_remote_SHA') is True,'calibration was not independently retrieved')
    for f in calibration['public_scalar_and_source_files']:
        add(f['path'],'teacher_calibration_scalar_or_source',f['sha256'],f.get('public_path'),f.get('source_aliases',[]))
    if calibration['disposition']=='CLOSED_COMPLETE':
        require(load(cal/'complete.json')['status']=='PASS_LS48_AND_ACTUAL_MODEL_SELECT16','false completed selector')
        require(all(any(v['path']==(cal/name).resolve() for v in files.values()) for name in
                    ('manifest.json','records.jsonl','complete.json','fit_collection_complete.json','value_theta_selected.json','window_selected.json')),'calibration scalar closure incomplete')
    else:require(calibration.get('disposition_evidence') and calibration.get('selected_parameters_claimed') is False,'stopped calibration cannot claim a selected theta/window')
    report={**readiness,'status':'PASS_ROUND10_ROOT_SEALED_LOCAL_STAGE','missing':[],'publication_eligible':True,
        'closed_cells':len(closed),'failed_cells':len(failures),'stopped_or_unrun_cells':len(dispositions),
        'closed_quality_answer_events':sum(x['expected_QA'] for x in closed),'closed':closed,'failures':failures,
        'disposition_only':dispositions,'calibration_disposition':calibration['disposition'],'plans':origins,
        'no_canonical_SourceRun_admission':True,'not_official500_or_independent_confirmation':True,
        'whole_Round10_goal_complete_claimed':False}
    return report,files,c


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path);p.add_argument('--candidate-output',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deep',action='store_true');a=p.parse_args()
    require(inside(a.output,EVIDENCE) and not a.output.exists(),'new E output required')
    if a.candidate_output:
        require(a.contract is None and inside(a.candidate_output,EVIDENCE),'candidate only on E')
        write_new(a.candidate_output,candidate());a.contract=a.candidate_output
    require(a.contract is not None,'explicit contract required');report,_,_=validate(a.contract,a.deep)
    write_new(a.output,report);print(json.dumps(report,ensure_ascii=False));return 0 if report['publication_eligible'] else 3


if __name__=='__main__':raise SystemExit(main())
