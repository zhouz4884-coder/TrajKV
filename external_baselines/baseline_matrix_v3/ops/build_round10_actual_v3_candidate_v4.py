"""Collect actual closed development scope; leave reserve results unopened.

No seal, public export, task scorer, GPU, Git or remote operation. Root's
actual sealed primary contract is reused; extension selection is fixed only
by its hashed decision/plan and all extension cells remain pending here.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'code')]
import round10_public_contract_v1 as primary
import round10_public_contract_v3 as public
E=primary.EVIDENCE
SEALED_PRIMARY_SHA='ec646facb78f00e2922a1514ea4367e67207d26dc399a95f403590413451fab5'
DECISION_SHA='6b515c946ec8d87bec7b3d41dd0e293eef2a27586b9450532a577927cc80d43b'


def build(a):
    out=a.output.resolve();primary.require(primary.inside(out,E) and not out.exists(),'new E candidate directory required')
    out.mkdir(parents=True)
    primary.require(primary.sha(a.primary_contract)==SEALED_PRIMARY_SHA,'Root sealed primary changed')
    pc=primary.load(a.primary_contract)
    # All scored summaries exist before calling deep; no scorer can be invoked.
    import round10_harvest_score_v1 as ph
    pitems,_=ph.load_plans([Path(p['path']) for p in pc['plans']])
    primary.require(all((Path(pc['harvest_root'])/'scored'/x['cell']['name']/'summary.json').is_file() for x in pitems.values()),'pre-existing primary score required')
    pp,pfiles,_=primary.validate(a.primary_contract,deep=True)
    primary.require(pp['closed_cells']==28 and pp['closed_quality_answer_events']==1120,'Root sealed primary deep closure differs')
    c=public.candidate();c['primary_contract']={'path':str(a.primary_contract.resolve()),'sha256':SEALED_PRIMARY_SHA}
    c.update(stage_state='PENDING_ROOT_FINAL_SCOPE_SEAL',root_seal=None)
    extras={};closure=dict(pfiles);watched={};group_audits=[]
    def add(path,role,digest=None,aliases=()):
        rel=public.add_file(extras,path,role,digest,aliases=aliases)
        public.add_file(closure,path,role,digest,aliases=aliases)
        watched[Path(path).resolve()]=primary.sha(path);return rel
    for kind,summary_path,expected_status in (
            ('middle_development',a.middle_summary,'PASS_ALL20_D160_MIDDLE_CUSTOM_SOURCE_SCORE_PHYSICAL'),
            ('kra_development',a.kra_summary,'PASS_ALL16_D160_KRA_CUSTOM_SOURCE_SCORE_PHYSICAL')):
        s=primary.load(summary_path);g=next(g for g in c['groups'] if g['kind']==kind)
        modes=(*public.REFS,*g['candidate_modes']);expected={(d,m) for d in public.DOMAINS for m in modes}
        primary.require(s['status']==expected_status and s['closed_cells']==len(expected) and
                        {(r['dataset'],r['mode']) for r in s['points']}==expected and
                        all(r['primary_mean'] is not None for r in s['points']),'actual full development summary required')
        adapter=importlib.import_module(public.ADAPTERS[kind]);adapter.check_pins()
        primary.require(primary.sha(adapter.__file__)==public.ADAPTER_PINS[kind],'actual approved development adapter differs')
        g.update(execution_disposition='EXECUTED',plans=s['plan_sources'],harvest_root=str(a.harvest_root.resolve()),refs_root=str(a.refs_root.resolve()))
        for r in g['cells']:r['disposition']='CLOSED_COMPLETE'
        items,_=adapter.load_plans([Path(p['path']) for p in g['plans']])
        points={(p['dataset'],p['mode']):p for p in s['points']};fresh=0
        for key,item in items.items():
            ds,mode=key
            if mode in public.REFS:continue
            name=item['cell']['name'];raw=a.harvest_root/'raw'/name;scored=a.harvest_root/'scored'/name
            primary.require((scored/'summary.json').is_file(),'no new scorer allowed; actual scored development cell required')
            m,done,cfg,proof=adapter.check_local_proof(raw,item)
            for e in proof['files']:
                public.add_file(closure,raw/e['file'],'actual_development_source_or_raw',e['sha256'],aliases=[e['remote_file']])
            for n in ('source_retrieval_receipt_round10_v1.json','retrieval_receipt.json'):
                public.add_file(closure,raw/n,'actual_development_independent_proof')
            score=primary.load(scored/'summary.json')
            public.add_file(closure,scored/'summary.json','frozen_development_score')
            for n,digest in score['derived_file_hashes'].items():public.add_file(closure,scored/n,'frozen_development_score',digest)
            point,_=adapter.closed_cell(raw,scored,item,a.refs_root)
            primary.require(point['primary_mean']==points[key]['primary_mean'] and done['status']=='PASS_RAW_COMPLETE' and
                            proof['parent_cell_success'] is True,'development source/score/physical closure differs')
            fresh+=point['expected_QA']
        primary.require(fresh==(320 if kind=='middle_development' else 160),'new development event scope differs')
        for n,role in (('summary.json','closed_development_summary'),('quality_complete_bytes.csv','development_quality_bytes_table'),
                       ('instrumented_timing_separate.json','development_instrumented_timing'),('paired_comparisons.json','development_paired_comparisons')):
            add(summary_path.parent/n,role)
        group_audits.append({'kind':kind,'view_cells':len(expected),'fresh_answer_events':fresh,'reused_primary_reference_events':480,
                             'all_real_source_score_physical_cells_closed':True})
    # Explicit KRA calibration proof list, never private fit or model payloads.
    cal=a.kra_calibration_root.resolve();proofpath=cal/'remote_kra_calibration_retrieval_v1.json';proof=primary.load(proofpath)
    primary.require(proof['status']=='PASS_REMOTE_KRA_CALIBRATION_LAUNCH_SOURCE_AND_SCALAR_SHA' and
                    proof['remote_hashes_computed_independently'] is True and proof['identity_only'] is False,'actual full KRA selector proof required')
    calfiles=[]
    for e in proof['files']:
        path=cal/e['file'];public.add_file(closure,path,'actual_kra_calibration_source_or_scalar',e['sha256'],aliases=[e['remote_file']])
        primary.require(path.stat().st_size==e['bytes'],'KRA actual retrieved scalar/source byte size differs')
        calfiles.append({'path':str(path),'sha256':e['sha256'],'source_aliases':[e['remote_file']]})
    public.add_file(closure,proofpath,'actual_kra_calibration_independent_proof')
    c['kra_calibration']={'disposition':'CLOSED_COMPLETE','raw':str(cal),
        'independent_source_proof':{'path':str(proofpath),'sha256':primary.sha(proofpath)},'public_scalar_and_source_files':calfiles}
    done=primary.load(cal/'complete.json')
    primary.require(done['status']=='PASS_KRA_FIT48_REUSED_AND_ACTUAL_MODEL_SELECT16' and done['fit_sources_reused']==48 and
                    done['select_sources']==16 and done['new_fit_source_captures']==0,'KRA actual selector scope differs')
    for n,digest in {'records.jsonl':done['records_sha256'],'selected_metric.json':done['metric_sha256'],
                     'selection.json':done['selection_sha256'],'identity.json':done['identity_sha256']}.items():
        primary.require(primary.sha(cal/n)==digest,'KRA actual scalar closure differs')
    add(ROOT/'ops/round10_kra_fetch_calibration_v1.py','actual_kra_calibration_retrieval_source')
    analysis_root=a.development_analysis_root
    analysis_base=primary.load(analysis_root/'complete.json')
    analysis_kra=primary.load(analysis_root/'kra_addendum_v1/complete.json')
    primary.require(analysis_base['status']=='PASS_D160_DEVELOPMENT_ANALYSIS' and
        analysis_kra['status']=='PASS_KRA_D160_ADDENDUM_AND_COMBINED_DEVELOPMENT_SUMMARY' and
        analysis_base['reserved224_predictions_or_scores_opened'] is False and
        analysis_kra['reserved224_predictions_or_scores_opened'] is False,'actual completed development-only analysis required')
    for receipt,relative,script in ((analysis_base,'complete.json','round10_development_analysis_v1.py'),
                                    (analysis_kra,'kra_addendum_v1/complete.json','round10_development_kra_addendum_v1.py')):
        add(analysis_root/relative,'completed_development_analysis_receipt')
        add(ROOT/'ops'/script,'actual_development_analysis_source',receipt['source_sha256'])
        for name,digest in receipt['outputs'].items():
            role='development_findings_report' if name.endswith('.md') else 'development_analysis_public_table_or_evidence'
            add(analysis_root/Path(name),role,digest)
    support_audits=[]
    for inventory_path in a.support_file_lists:
        inventory=primary.load(inventory_path)
        primary.require(inventory['schema']=='trajkv.round10.public_support_filelist.v1' and
            inventory['new_GPU_jobs_started']==0 and inventory['task_scoring_performed'] is False and
            inventory['private_tensors_weights_images_included'] is False,'only completed text/source QA0 support inventories')
        add(inventory_path,'completed_QA0_support_inventory')
        for f in inventory['files']:add(f['path'],f['role'],f['sha256'],f.get('source_aliases',[]))
        shared_path=inventory_path.with_name('SHARED_LEGACY_CLOSURE.json')
        shared_count=0
        if shared_path.is_file():
            shared=primary.load(shared_path)
            primary.require(shared['schema']=='trajkv.round10.middle_shared_legacy_closure.v1' and
                primary.sha(shared['independent_original_QA_source_proof'])==shared['independent_original_QA_source_proof_sha256'],
                'QA0 shared legacy source proof differs')
            add(shared_path,'QA0_shared_legacy_exact_source_closure')
            for f in shared['files']:add(f['path'],f['role'],f['sha256'],f.get('source_aliases',[]))
            shared_count=len(shared['files'])
        support_audits.append({'inventory':str(inventory_path),'sha256':primary.sha(inventory_path),
            'explicit_support_files':len(inventory['files']),'exact_shared_legacy_sources':shared_count,
            'free_generation_quality_events_added':0})
    add(Path(__file__),'actual_scope_candidate_preparation_source')
    # Only selection/plan metadata is opened for reserve224; no raw or score.
    primary.require(primary.sha(a.decision)==DECISION_SHA,'Root replication selection decision changed')
    decision=primary.load(a.decision);primary.require(decision['selected_modes']==['V128-R43'],'only frozen V128 extension authorized')
    add(a.decision,'root_replication_selection_freeze',DECISION_SHA)
    plan=primary.load(a.extension_plan);primary.require(plan['selected_modes']==['V128-R43'] and plan['phase']=='extension' and
        len(plan['cells'])==16 and sum(r['expected_QA'] for r in plan['cells'])==896,'frozen extension plan scope differs')
    ext=next(g for g in c['groups'] if g['kind']=='extension_retest')
    ext.update(execution_disposition='PENDING',candidate_modes=['V128-R43'],
        plans=[{'path':str(a.extension_plan.resolve()),'sha256':primary.sha(a.extension_plan)}],
        harvest_root=str(a.extension_harvest_root.resolve()),refs_root=str(a.refs_root.resolve()),
        cells=[{'dataset':d,'mode':m,'expected_QA':public.EXTENSION_COUNTS[d],'disposition':'PENDING','reuse_from':None}
               for d in public.DOMAINS for m in (*public.REFS,'V128-R43')])
    es=primary.load(a.extension_summary)
    primary.require(es['status']=='PASS_ALL_RESERVE224_REPLICATION_SOURCE_SCORE_PHYSICAL' and
        es['closed_cells']==es['expected_cells']==16 and es['closed_answer_events']==es['expected_answer_events']==896 and
        es['actual_closed_unique_IDs']==224 and es['selected_modes']==['V128-R43'],'exact completed fixed reserve224 scope required')
    import round10_extension_harvest_v1 as eh
    eh.check_pins();extension_items,_,chosen=eh.load_plans([a.extension_plan])
    primary.require(chosen==['V128-R43'] and len(extension_items)==16,'actual fixed extension plans differ')
    roots={Path(p['raw']).resolve().parent.parent for p in es['points']}
    primary.require(len(roots)==1,'one explicit actual extension harvest root required')
    actual_extension_root=roots.pop();ext.update(execution_disposition='EXECUTED',harvest_root=str(actual_extension_root))
    extension_points={(p['dataset'],p['mode']):p for p in es['points']}
    for r in ext['cells']:r['disposition']='CLOSED_COMPLETE'
    for key,item in extension_items.items():
        name=item['cell']['name'];raw=actual_extension_root/'raw'/name;scored=actual_extension_root/'scored'/name
        primary.require((scored/'summary.json').is_file(),'pre-existing complete extension score required; no scorer may be created')
        m,done,cfg,proof=eh.check_local_proof(raw,item)
        for e in proof['files']:public.add_file(closure,raw/e['file'],'actual_extension_source_or_raw',e['sha256'],aliases=[e['remote_file']])
        for n in ('source_retrieval_receipt_round10_v1.json','retrieval_receipt.json'):public.add_file(closure,raw/n,'actual_extension_independent_proof')
        score=primary.load(scored/'summary.json');public.add_file(closure,scored/'summary.json','frozen_extension_score')
        for n,digest in score['derived_file_hashes'].items():public.add_file(closure,scored/n,'frozen_extension_score',digest)
        point,_=eh.closed_cell(raw,scored,item,a.refs_root)
        primary.require(point['primary_mean']==extension_points[key]['primary_mean'] and proof['parent_cell_success'] is True,
                        'complete extension source/score/physical evidence differs')
    for n,role in (('summary.json','closed_extension_replication_summary'),('quality_complete_bytes.csv','replication_quality_bytes_table'),
                   ('instrumented_timing_separate.json','replication_instrumented_timing'),('paired_comparisons.json','replication_paired_comparisons')):
        add(a.extension_summary.parent/n,role)
    final_receipt=primary.load(a.final_analysis_root/'complete.json')
    primary.require(final_receipt['status']=='PASS_FINAL_EXECUTED_SCOPE_ANALYSIS' and final_receipt['actual_new_free_QA']==2496 and
        final_receipt['actual_runs']==56 and final_receipt['declared_views']==80 and final_receipt['declared_view_events']==3456 and
        final_receipt['reference_view_reuses']==960,'complete final independent analysis required')
    add(a.final_analysis_root/'complete.json','final_executed_analysis_receipt')
    add(ROOT/'ops/round10_final_analysis_v1.py','final_independent_analysis_source',final_receipt['source_sha256'])
    for n,digest in final_receipt['outputs'].items():add(a.final_analysis_root/n,'final_executed_analysis_output',digest)
    detail_root=a.final_analysis_root/'detail_addendum_v1';detail=primary.load(detail_root/'complete.json')
    primary.require(detail['status']=='PASS_FINAL_DETAIL_SOURCE_BOUND' and detail['original_outputs_unmodified'] is True,
                    'completed independent final-detail addendum required')
    add(detail_root/'complete.json','final_detail_analysis_receipt')
    add(ROOT/'ops/round10_final_detail_v1.py','final_detail_analysis_source',detail['source_sha256'])
    for n,digest in detail['outputs'].items():add(detail_root/n,'final_detail_analysis_output',digest)
    standalone_audits=[]
    for folder,receipt_name in ((E/'results/round10/caltech_v1','INDEPENDENT_REMOTE_SOURCE_FETCH_v1.json'),
                                (E/'results/round10/kra_identity_retrieved_v1','remote_kra_calibration_retrieval_v1.json')):
        receipt_path=folder/receipt_name;receipt=primary.load(receipt_path)
        primary.require(receipt.get('remote_hashes_computed_independently') is True or
                        receipt.get('status')=='PASS_INDEPENDENT_REMOTE_METADATA_SOURCE_SHA','actual standalone technical source proof required')
        add(receipt_path,'standalone_QA0_independent_source_proof')
        for e in receipt['files']:
            original=folder/e['file']
            primary.require(primary.sha(original)==e['sha256'] and original.stat().st_size==e['bytes'],'standalone actual source/scalar differs')
            if original.suffix.lower()=='.log':
                target=out/'text_mirrors'/(folder.name+'_producer_log.txt');target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(original.read_bytes())
                add(target,'standalone_QA0_stdout_exact_text_mirror',e['sha256'],[str(original),e['remote_file']])
            else:add(original,'standalone_QA0_actual_text_source',e['sha256'],[e['remote_file']])
        td=primary.load(folder/'complete.json')
        primary.require(td['actual_GPU'] is True and td.get('new_free_generation_task_answers',0)==0,'standalone technical controls add zero free answers')
        standalone_audits.append({'raw':str(folder),'receipt_sha256':primary.sha(receipt_path),'files':len(receipt['files']),
            'sequence_parent_exit0_claimed':False,'free_answer_events_added':0})
    add(a.math_root/'NOTE.zh.md','final_math_synthesis','bdd33e8c649d5adf3df187c7db9e5adb3b64058d3efcaee310a0a20a8816d23f')
    add(a.math_root/'EVIDENCE_INDEX.json','final_math_evidence_index','3ada2504c37de9d00de53fcc40d5d798fdeac569e5a6a5be7cc4c5095fcdbb69')
    for n in ('KRA_EXISTING_RAW_AGGREGATES.json','KRA_PER_SOURCE_RAW_SCALARS.json'):add(a.math_root/n,'existing_KRA_raw_mathematical_scalars')
    for path in a.math_scripts:add(path,'existing_raw_mathematical_aggregation_source')
    math_recompute=primary.load(a.math_recompute_root/'RECOMPUTATION_RECEIPT.json')
    add(ROOT/'ops/round10_kra_reaggregate_existing_v1.py','new_portable_existing_raw_math_recomputation_source',
        '27e9540ee23b0ec70cfa9edd28c492319f1e80311dc06b10419aae57939f67cf')
    for n,digest in {'RECOMPUTATION_RECEIPT.json':'16018bc03a89d09ae5960faa0345d66551dabb2a4d5302d27163df06518c11bd',
                     'RECOMPUTED_AGGREGATES.json':'f3514e9b1c65fb47ff99c81ce15ed0e01ee6f869c7c04936d76d1496c93f33a7',
                     'RECOMPUTED_PER_SOURCE_SCALARS.json':'09f1f31953f9753dc46c895a20fb4511155744939cb9a4a2cda223117c7c1393'}.items():
        add(a.math_recompute_root/n,'new_independent_existing_raw_math_recomputation',digest)
    for n in ('summary.json','layer_records.jsonl','retrieval_receipt.json','launch_receipt.json','source_config.json',
              'source_driver.py','source_inventory.json','source_job.json'):add(a.mechanism_root/n,'existing_CPU_mechanism8_QA0_support')
    add(ROOT/'code/round10_mechanism_v1.py','existing_CPU_mechanism8_source')
    for n in ('runtime_gpu0_v1_complete.json','runtime_gpu1_v1_complete.json','window_cuda_v1_complete.json'):
        add(E/'results/round10/bootstrap_v1'/n,'technical_controls')
    add(E/'results/round10/extension_observation_completion_v1/EXTENSION_COMPLETION_RECEIPT.json','actual_extension_observation_completion',
        '38f6bca0a9222e5c1e5d81f37e126d9301522f8d2d77121380aabe73f40e7591')
    add(E/'results/round10/extension_observation_completion_v1/EXACT_METHOD_DOMAIN_MEANS.csv','actual_extension_exact_domain_means')
    proposal=primary.load(a.proposal_root/'SOURCE_IDENTITY.json')
    primary.require(proposal['source_sha256']==proposal['mirror_sha256']=='4a5a0ca78ba33fb790198d35d2850db8620cc443abd9d2456d4e3e6e63ad8112',
                    'Root next-proposal exact mirror identity differs')
    add(proposal['mirror'],'next_proposal_unexecuted',proposal['mirror_sha256'],[proposal['source']])
    add(a.proposal_root/'SOURCE_IDENTITY.json','next_proposal_source_identity')
    designs=primary.load(a.design_root/'SOURCE_IDENTITIES.json')
    primary.require(designs['originals_overwritten'] is False and designs['private_chat_included'] is False,'original design freezes required without chat')
    for d in designs['documents']:
        primary.require(d['byte_exact'] is True and d['source_sha256']==d['mirror_sha256'],'original design exact-byte identity differs')
        add(d['mirror'],'execution_design_freeze',d['mirror_sha256'],[d['source']])
    add(a.design_root/'SOURCE_IDENTITIES.json','execution_design_freeze')
    primary.require(len(a.authoritative_report_sha256)==64,'explicit Root authoritative report SHA required')
    add(a.authoritative_report,'root_authoritative_final_research_report',a.authoritative_report_sha256)
    add(E/'results/round10/final_report_v1/SOURCE_IDENTITIES.json','root_authoritative_report_mirror_identities')
    c['replication_selection_freeze']={'path':str(a.decision.resolve()),'sha256':DECISION_SHA}
    c['files']=[{'path':str(e['path']),'role':e['role'],'sha256':primary.sha(e['path']),'public_path':rel,
                 **({'source_aliases':e['aliases']} if e.get('aliases') else {})} for rel,e in extras.items()]
    primary.require(all(primary.sha(p)==digest for p,digest in watched.items()),'closed support file changed during candidate collection')
    target=out/'ACTUAL_V3_CONTRACT_CANDIDATE.json';primary.write_new(target,c)
    readiness,_,_=public.validate(target,deep=False)
    primary.require(not readiness['publication_eligible'] and set(readiness['missing'])=={'Root v3 scope seal','root_seal path/SHA'},
                    'final candidate must wait only for Root seal')
    primary.write_new(out/'READINESS.json',readiness)
    rows=[{'public_path':rel,'path':str(e['path']),'role':e['role'],'source_sha256':primary.sha(e['path']),
           'source_aliases':e.get('aliases',[])} for rel,e in closure.items()]
    primary.write_new(out/'ALL_ACTUAL_STAGE_FILE_CLOSURE.json',{'schema':'trajkv.round10.final_actual_file_closure.v1',
        'files':rows,'file_count':len(rows),'extension_model_results_included':True})
    audit={'status':'PASS_ALL_ACTUAL_2496_FRESH_3456_VIEWS_CANDIDATE_WAIT_ROOT_SEAL_ONLY',
        'primary_contract_sha256':SEALED_PRIMARY_SHA,'actual_primary_closed_cells':28,'actual_primary_answer_events':1120,
        'closed_development_groups':group_audits,'total_development_fresh_answer_events':1600,'total_development_view_events':2560,
        'KRA_calibration_text_source_files':len(calfiles),'all_actual_stage_closure_files':len(rows),
        'completed_QA0_support':support_audits,
        'standalone_QA0_support':standalone_audits,
        'development_analysis_base_and_KRA_receipts_PASS':True,
        'development_reports_public_prefix':'results/round10/development_analysis_v1/',
        'extension_selected_modes':['V128-R43'],'extension_closed_cells':16,'extension_actual_fresh_events':896,
        'stage_actual_unique_runs':56,'stage_actual_fresh_events':2496,'stage_actual_view_events':3456,'reference_view_reuses':960,
        'extension_model_raw_or_scores_opened':True,'candidate_sha256':primary.sha(target),'builder_sha256':primary.sha(__file__),
        'root_seals_created':0,'exports_created':0,'new_task_scores':0,'new_GPU_QA':0,'Git_GitHub_SSH_mutations':False,
        'publication_eligible':False,'whole_Round10_goal_complete_claimed':False}
    primary.write_new(out/'PREPARATION_AUDIT.json',audit)
    print(json.dumps({k:audit[k] for k in ('status','stage_actual_fresh_events','stage_actual_view_events','all_actual_stage_closure_files','candidate_sha256')}))
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    defaults={'primary_contract':'primary_sealed_v1/CONTRACT.json','middle_summary':'middle_watch_snapshot_v2/summary.json',
        'kra_summary':'kra_watch_snapshot_v3/summary.json','harvest_root':'harvest_v1','refs_root':'input_preparation_v1/inputs',
        'kra_calibration_root':'kra_calibration_retrieved_v1','decision':'selection_freeze_v1/DECISION.json',
        'extension_plan':'bootstrap_v1/EXTENSION_V128_JOB_PLAN_v1.json','extension_harvest_root':'extension_harvest_v1',
        'development_analysis_root':'development_analysis_v1','extension_summary':'extension_D224_closed_v1/summary.json',
        'final_analysis_root':'final_analysis_v1','math_root':'math_synthesis_v1','proposal_root':'next_proposal_v1','design_root':'design_freezes_v1',
        'math_recompute_root':'math_recompute_v1','mechanism_root':'mechanism8_v1'}
    for name,value in defaults.items():p.add_argument('--'+name.replace('_','-'),type=Path,default=E/'results/round10'/value)
    p.add_argument('--support-file-lists',nargs='*',type=Path,default=[E/'results/round10/middle_technical_retrieved_v1/PUBLIC_SUPPORT_FILES.json'])
    p.add_argument('--math-scripts',nargs='*',type=Path,default=[])
    p.add_argument('--authoritative-report',type=Path,required=True)
    p.add_argument('--authoritative-report-sha256',required=True)
    build(p.parse_args())


if __name__=='__main__':main()
