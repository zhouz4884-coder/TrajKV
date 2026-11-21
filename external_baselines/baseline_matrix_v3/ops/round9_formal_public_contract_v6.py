"""Formal Drop4 V6: primary4/2000 fresh; twenty same500 references reused.

Explicitly dispatches R9 custom, original R7 custom, and canonical SourceRun
closure without aliasing any custom runner to evaluate.py. No live operations.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path('${EVIDENCE_ROOT}')
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'code')];sys.dont_write_bytecode=True
from public_phase_contract_v2 import require,load,sha,resolve,inside,validate_scored
import public_phase_contract_v2 as canonical_phase
import round9_public_contract_v3 as original_r9
import custom_driver_phase_v1 as original_r7
import round9_summarize_formal_Drop4_v1 as same500

FAMILY='round9_selected_Drop4_formal4_v6'
R7_FAMILY=original_r7.FAMILY
CANONICAL_FAMILY='round9_formal_same500_canonical12_v6'
PLAN=EVIDENCE/'results/round9/formal_Drop4_draft_v1/PLAN.json'
PLAN_SHA='1d33889629eaf1b75e5bc3cce0a88b6c53617c71c15d7b00e7d4a5cc13219553'
SELECTION=PLAN.parent/'SELECTION_FROZEN_v1.json'
SELECTION_SHA='25cd37961793dbfbe4305ce8761e3388389a2aaa2899cd4ac3ce3d493491501e'
RUNNER='code/round9_runner_v4.py'
RUNNER_SHA='cf4885b77b03fd94aadd5bbabc5a1e6d62e3259ba6037b7a708fabab71758961'
PROFILE_SHA=original_r9.BRPE_PROFILE_SHA
POLICY=EVIDENCE/'results/publication/R9_selected_formal4_support_v6/FORMAL_SOURCE_POLICY.json'
POLICY_REL='results/publication/R9_selected_formal4_support_v6/FORMAL_SOURCE_POLICY.json'
DOMAINS=('textvqa','infovqa','docvqa','chartqa')
LABELS=('F','R7_K1','R7_K2','K2_g32R32','C2_bias32')
HELPER_PINS={'ops/round9_public_contract_v3.py':'cfb12373d24ef35875ad964626f2af1d1a0d8342c3b7c32314ae068a261fe9c6',
 'ops/custom_driver_phase_v1.py':'cc0f80bd0578f377b0149b67910b177afc0200359f047e8296214c645c20478a',
 'ops/public_phase_contract_v2.py':'b92493b9509c1f7d8e02170b1a548be12e651a39b00067ea3ba7c5f135b928eb',
 'ops/round9_summarize_formal_Drop4_v1.py':'74f79a2bdeed5aabcf4f31c22d1fbf4fd5f3bc8d0d0958e33fc4ce53c6248e68'}

def canonical(value):return original_r9.canonical(value)

def write(path,value):return original_r9.write(path,value)

def source_declarations():
    for name,digest in HELPER_PINS.items():require(sha(ROOT/name)==digest,'Frozen reused admission helper changed: '+name)
    require(sha(PLAN)==PLAN_SHA and sha(SELECTION)==SELECTION_SHA,'Exact pre-formal selection/plan changed')
    plan,selected=load(PLAN),load(SELECTION)
    require(plan['expected_cells']==4 and plan['expected_model_answers']==2000 and plan['not400'] is True and
        plan['runner_source_sha256']==RUNNER_SHA and plan['frozen_profile_sha256']==PROFILE_SHA,'Exact4x500 historical plan required')
    require(selected['status']=='SELECTED_FROZEN_FOR_FOUR_DOMAIN_FORMAL_EVALUATION' and selected['mode']=='BRPE_Drop4' and
        selected['profile_sha256']==PROFILE_SHA and selected['runner_sha256']==RUNNER_SHA and selected['kept_V0'] is True and
        selected['dropped_k_units']==original_r9.BRPE_DONORS and selected['questions_per_domain']==500 and selected['expected_fresh_model_answers']==2000 and
        set(selected['expected_domains'])==set(DOMAINS) and selected['new_source_was_replication_and_candidate_selection_not_blind_confirm'] is True,
        'Selected formal point cannot be changed after looking at formal scores')
    for item in selected['evidence']:require(sha(resolve(item['path']))==item['sha256'],'Prelaunch selection evidence changed')
    require(len(plan['rows'])==4 and {r['dataset'] for r in plan['rows']}==set(DOMAINS),'Four distinct domains required')
    for row in plan['rows']:
        cfg=load(ROOT/row['config'])
        require(sha(ROOT/row['config'])==row['config_sha256'] and sha(ROOT/row['job'])==row['job_sha256'] and
            cfg['cohort_kind']=='formal500' and cfg['start']==0 and cfg['count']==cfg['input_count']==row['count']==500 and
            cfg['candidate']['mode']=='BRPE_Drop4' and cfg['candidate']['profile_sha256']==PROFILE_SHA,
            'Frozen executable config/job/500 source differs')
    return plan,selected

def references():
    refs,common,_=same500.load_refs()
    require(len(refs)==20 and set(refs)=={(d,k) for d in DOMAINS for k in LABELS},'Exact five same500 anchors, no official-full substitution')
    return refs,common

def expected_reference_rows():
    refs,_=references();r7=[];other=[]
    declared,ready=original_r7.exact_ready8()
    old={(r['dataset'],'R7_'+r['profile']):(r,c['cell']) for r,c in zip(declared,ready['cells'])}
    for (ds,key),cell in sorted(refs.items()):
        if key in ('R7_K1','R7_K2'):
            frozen,proof=old[ds,key]
            require(resolve(cell['view'])==resolve(proof['raw']) and resolve(cell['scored'])==resolve(proof['scored']), 'R7 reference table must refer to the exact frozen source')
            cell={**cell,'config':frozen['config'],'config_sha256':frozen['config_sha256']}
        row={'cell':Path(cell['config']).stem,'model':'M1','dataset':ds,'reference_key':key,'count':500,
            'config':cell['config'],'config_sha256':cell['config_sha256'],'view':cell['view'],'scored':cell['scored']}
        (r7 if key in ('R7_K1','R7_K2') else other).append(row)
    require(len(r7)==8 and len(other)==12,'R7custom8 and canonical12 are different admission families')
    return r7,other

def validate_declared_reference(contract):
    require(contract['family'] in (R7_FAMILY,CANONICAL_FAMILY),'Unknown auxiliary producer family')
    want=expected_reference_rows()[0 if contract['family']==R7_FAMILY else 1]
    matrix=load(resolve(contract['matrix']));rows=matrix['rows']
    identity=lambda row:{k:row[k] for k in ('config','config_sha256','view','scored','count','dataset','model')}
    require(len(rows)==len(want) and {canonical(identity(r)) for r in rows}=={canonical(identity(r)) for r in want},
        'Only frozen exact20 same500 reference identities, never full-F or another method')
    for row in rows:require(sha(resolve(row['config']))==row['config_sha256'],'Reference configuration changed')
    return original_r7.validate_phase(contract) if contract['family']==R7_FAMILY else canonical_phase.validate_phase(contract)

def validate_primary_item(row,common):
    from score import validate_raw
    raw,scored=resolve(row['view']),resolve(row['scored'])
    item=original_r9.validate_inference(raw);m,cfg=item['manifest'],item['config']
    ds=row['dataset'];plan,selected=source_declarations();source=next(r for r in plan['rows'] if r['dataset']==ds)
    require(cfg==load(ROOT/source['config']) and row['config_sha256']==source['config_sha256']==m['config_sha256'] and
        sha(raw/'source_job.json')==source['job_sha256'] and m['source_hashes']==original_r9.expected_sources(RUNNER),
        'Actual selected formal producer/config/source differs')
    require(m['started_unix']>=datetime.fromisoformat(selected['observed_utc']).timestamp(),'Selection must predate actual formal run')
    require(cfg['candidate']['mode']=='BRPE_Drop4' and cfg['candidate']['profile_sha256']==PROFILE_SHA and cfg['cohort_kind']=='formal500' and
        cfg['count']==cfg['input_count']==500 and cfg['start']==0,'No old dev32/64 or formal400 may enter')
    frozen=next(x for x in common['rows'] if x['dataset']==ds)
    require(m['inputs_sha256']==source['inputs_sha256']==frozen['input_sha256'] and m['expected_ids']==frozen['selected_ids'],'Exact same500 order required')
    context=common['common_contracts_by_dataset'][ds]['context']
    require(same500.frozen.native_context(m)=={k:context[k] for k in same500.frozen.native_context(m)} and m['model_spec']==context['model_spec'],
        'Actual checkpoint/processor/generation/runtime differs')
    identities={r['id']:same500.frozen.input_identity(r) for r in item['rows']}
    require(same500.frozen.canonical(identities)==common['parent_full500_alignment'][ds]['per_ID_identity_fingerprint'],
        'Every input/image/prompt/tokens/RoPE/visual position/boundary must match same500 anchors')
    refs=ROOT/'configs/datasets/v3'/Path(cfg['inputs']).name.replace('_inputs.jsonl','_refs.jsonl')
    validated=validate_raw(raw,refs);summary,failures=validate_scored(scored,validated)
    require(not failures and summary['status']=='PASS_SCORED_COMPLETE' and summary['expected_run_questions']==500,'Formal500 complete PASS scores required')
    return item,summary

def validate_phase(contract):
    if contract.get('family')!=FAMILY:return validate_declared_reference(contract)
    require(contract.get('completion_scope')=='declared_matrix_only','No whole-Round9 completion')
    plan,_=source_declarations();matrix_path=resolve(contract['matrix']);rows=load(matrix_path)['rows']
    expected={r['config']:r for r in plan['rows']}
    require(len(rows)==4 and {r['config'] for r in rows}==set(expected) and sum(r['count'] for r in rows)==2000,'Exact4 formal500 primary required')
    refs,common=references();artifacts=[];observed=[];missing=[]
    for row in rows:
        old=expected[row['config']]
        require(row['count']==500 and row['config_sha256']==old['config_sha256'] and resolve(row['view'])==resolve(old['raw']) and
            resolve(row['scored'])==resolve(old['scored']) and row['dataset']==old['dataset'],'Frozen primary source paths differ')
        raw,scored=resolve(row['view']),resolve(row['scored'])
        if not all((raw/name).is_file() for name in ('manifest.json','records.jsonl','complete.json','retrieval_receipt.json','source_retrieval_receipt_round9_v2.json')) or not(scored/'summary.json').is_file():
            missing.append({'cell':row['cell'],'missing':'actual_complete500_source_and_score','primary_mean':None});continue
        if load(raw/'complete.json')['status']!='PASS_RAW_COMPLETE' or load(scored/'summary.json')['status']!='PASS_SCORED_COMPLETE':
            require(load(scored/'summary.json')['primary_mean'] is None,'Failed formal primary must remain null')
            missing.append({'cell':row['cell'],'missing':'all500_QA_PASS','primary_mean':None});continue
        item,summary=validate_primary_item(row,common)
        artifacts.append({'cell':row['cell'],'config':resolve(row['config']),'raw':raw,'scored':scored,'kind':'custom_R9_original_fresh_inference_run',
            'expected_QA':500,'raw_status':'PASS_RAW_COMPLETE','origin_counts':{'fresh':500}})
        observed.append({'cell':row['cell'],'questions':500,'kind':'custom_R9_original_fresh_inference_run','raw_status':'PASS_RAW_COMPLETE','score_status':summary['status']})
    phase={'status':'PASS_DECLARED_MATRIX_COMPLETE' if not missing else 'INCOMPLETE_DECLARED_MATRIX_PREVIEW','phase_label':contract['phase_label'],
        'family':FAMILY,'coverage_scope':contract['coverage_scope'],'completion_scope':'declared_matrix_only','whole_external_library_completion_assessed':False,
        'declared_matrix_source_sha256':sha(matrix_path),'expected_cells':4,'verified_scored_cells':len(observed),'closed_artifact_cells':len(artifacts),
        'expected_questions':2000,'verified_scored_questions':len(observed)*500,'missing':missing,'observed':observed,'SourceRun_admission':False,
        'primary_actual_fresh_runs':4,'primary_actual_fresh_QA':2000,'required_auxiliary_same500_points':20,'auxiliary_reused_view_QA':10000,'auxiliary_new_GPU_QA':0,
        'old704_and_new384_recounted_as_fresh':0,'whole_Round9_complete':False}
    return phase,artifacts,matrix_path

def source_closure(artifacts):
    custom9=[];others=[]
    for artifact in artifacts:
        schema=load(artifact['raw']/'manifest.json').get('runner_schema')
        if schema==original_r9.SCHEMA:custom9.append(artifact)
        else:others.append(artifact)
    nodes=original_r7.source_closure(others) if others else []
    if not custom9:return nodes
    require(len(custom9)==4,'Primary closure must contain all4, not a partial formal score')
    plan,selected=source_declarations();refs,common=references()
    cal=original_r9.validate_brpe_calibration()['files']+original_r9.validate_calibration_metadata()['files']
    files=[PLAN,SELECTION,same500.COMMON,same500.REF]+[ROOT/name for name in HELPER_PINS]
    # The selected formal configs/jobs live outside the legacy collector globs.
    # Archive their exact frozen source bytes explicitly, never a fabricated alias.
    files += [ROOT/r[key] for r in plan['rows'] for key in ('config','job')]
    for ds in DOMAINS:
        files += [ROOT/('configs/datasets/v3/'+ds+'_p2_'+part+'.jsonl') for part in ('inputs','refs')]
    files += [ROOT/'configs/datasets/v3/source_lock.json']+[resolve(x['path']) for x in selected['evidence']]
    policy={'schema':'trajkv.round9.selected_formal_source_policy.v6','SourceRun_admission':False,'runner_schema':original_r9.SCHEMA,
        'drivers':{RUNNER:RUNNER_SHA},'source_epochs':{RUNNER:original_r9.expected_sources(RUNNER)},
        'selection_file':SELECTION.relative_to(EVIDENCE).as_posix(),'selection_sha256':SELECTION_SHA,
        'plan_file':PLAN.relative_to(EVIDENCE).as_posix(),'plan_sha256':PLAN_SHA,'required_primary_configs':{r['config']:r['config_sha256'] for r in plan['rows']},
        'required_auxiliary_configs':{cell['config']:cell['config_sha256'] for rows in expected_reference_rows() for cell in rows},
        'same500_common_file':same500.COMMON.relative_to(EVIDENCE).as_posix(),'same500_common_sha256':same500.frozen.REF_PINS['INPUT_AND_COMMON_CONTRACTS500.json'],
        'BRPE_profile_sha256':PROFILE_SHA,'BRPE_calibration_root':original_r9.BRPE_PROOF.relative_to(EVIDENCE).as_posix(),
        'calibration_metadata_root':original_r9.CALIB_META.relative_to(EVIDENCE).as_posix(),'calibration_complete_sha256':original_r9.CALIB_COMPLETE_SHA,
        'calibration_query_source':'compressed_query_same_student_text','native_query_captured':False,'private_tensor_bytes_archived':False,
        'primary_actual_fresh_QA':2000,'auxiliary_reused_view_QA':10000,'auxiliary_new_GPU_QA':0}
    if POLICY.exists():require(canonical(load(POLICY))==canonical(policy),'New policy cutoff needed for changed selected formal scope')
    else:write(POLICY,policy)
    for artifact in custom9:
        item=original_r9.validate_inference(artifact['raw'])
        nodes.append({'path':artifact['raw'],'kind':'custom_R9_original_fresh_inference_run','files':list(dict.fromkeys(item['source_files']+cal+files+[POLICY]))})
    return nodes

def contract_from_arguments(args):
    require(args.contract is not None,'Explicit formal V6 primary contract required')
    path=resolve(args.contract);value=load(path)
    require(value['schema']=='baseline_matrix.public_phase_contract.v2' and value['family']==FAMILY,'No unknown custom producer fallback')
    return value,path

def require_phase_groups(groups):
    require(len(groups)==3 and [g['contract']['family'] for g in groups]==[FAMILY,R7_FAMILY,CANONICAL_FAMILY],
        'Primary4 plus separately admitted R7custom8 and canonical12 references required')
    require([(g['phase']['expected_cells'],g['phase']['expected_questions']) for g in groups]==[(4,2000),(8,4000),(12,6000)],
        'Primary2000 fresh and auxiliary10000 reused are separate domains')

def prepare_bundle(output):
    plan,_=source_declarations();r7,canon=expected_reference_rows();output=resolve(output);require(not output.exists(),'Use new formal V6 preparation directory')
    primary=[{'cell':Path(r['config']).stem,'model':'M1','dataset':r['dataset'],'count':500,'config':r['config'],'config_sha256':r['config_sha256'],
        'view':r['raw'],'scored':r['scored']} for r in plan['rows']]
    output.mkdir(parents=True);contracts=[]
    for label,family,rows in [('ROUND9_SELECTED_DROP4_FORMAL500',FAMILY,primary),('FORMAL_SAME500_R7_ANCHORS8_REUSED',R7_FAMILY,r7),('FORMAL_SAME500_CANONICAL12_REUSED',CANONICAL_FAMILY,canon)]:
        matrix=output/(label+'_matrix.json');write(matrix,{'schema':'baseline_matrix.fixed_matrix.v1','rows':rows,'expected_cells':len(rows),'expected_QA':500*len(rows)})
        contract={'schema':'baseline_matrix.public_phase_contract.v2','family':family,'phase_label':label,'matrix':str(matrix),'view_root':str(EVIDENCE/'results'),
            'score_root':str(EVIDENCE/'results/scored'),'completion_scope':'declared_matrix_only','completion_policy':'all_expected_QA_pass',
            'coverage_scope':'Selected Drop4 formal4x500 fresh primary; same500 R7low/high, FP16, sourceKIVIg32R32 and common-FP16 numerical-port C2 references are separately reused. No fullofficial-F/old704/new384/wholeRound9 completion.'}
        path=output/(label+'_contract.json');write(path,contract);contracts.append(str(path))
    result={'status':'PREPARED_NOT_COMPLETE_OR_EXPORTED','primary_contract':contracts[0],'auxiliary_contracts':contracts[1:],
        'primary_requested_QA':2000,'auxiliary_reused_view_QA':10000,'auxiliary_new_GPU_QA':0}
    write(output/'BUNDLE.json',result);return result

def inspect_current(output):
    plan,_=source_declarations();rows=[]
    for r in plan['rows']:
        raw,scored=resolve(r['raw']),resolve(r['scored'])
        ready=all((raw/n).is_file() for n in ('complete.json','source_retrieval_receipt_round9_v2.json')) and (scored/'summary.json').is_file()
        rows.append({'dataset':r['dataset'],'metadata_present_only_not_deep_verified':ready,'primary_mean':None})
    value={'status':'WAIT_REAL_ALL4_COMPLETE500_SOURCE_SCORE' if not all(r['metadata_present_only_not_deep_verified'] for r in rows) else 'METADATA_ALL4_PRESENT_DEEP_VALIDATION_REQUIRED',
        'rows':rows,'primary_actual_verified_QA':0,'formal_positive_or_export_claimed':False,'expected_primary_QA':2000,'auxiliary_reused_view_QA':10000,'GPU_or_network':False}
    write(output,value);return value

def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare-bundle',type=Path);a.add_argument('--inspect-current',type=Path);x=a.parse_args()
    require(bool(x.prepare_bundle)^bool(x.inspect_current),'Choose one local preparation/readiness action')
    print(json.dumps(prepare_bundle(x.prepare_bundle) if x.prepare_bundle else inspect_current(x.inspect_current)))
if __name__=='__main__':main()
