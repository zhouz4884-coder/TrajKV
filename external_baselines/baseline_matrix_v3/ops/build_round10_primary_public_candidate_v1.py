"""Build an actual primary28 public contract candidate, never a Root seal.

Only closed D160/cal64 development evidence is read. Existing raw/score files
must already exist and stay byte-identical. No scoring, export, remote access,
Git operation, model invocation, reserve result read or publication occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'code')]
import round10_public_contract_v1 as public
import round10_harvest_score_v1 as harvest

E=public.EVIDENCE


def seal_payload(contract):
    """Pure helper for Root's later explicit seal; this builder never writes it."""
    return {'schema':public.SEAL_SCHEMA,'status':'SEALED_FOR_LOCAL_PUBLIC_PACKAGE',
        'phase_label':contract['phase_label'],
        'cells_disposition_sha256':public.canonical(contract['cells']),
        'calibration_disposition_sha256':public.canonical(contract['calibration'])}


def write_json(path,value):
    public.write_new(path,value)


def build(output, *, primary_summary, harvest_root, refs_root, calibration_root, report_source, plans):
    output=Path(output).resolve()
    public.require(public.inside(output,E) and not output.exists(),'new unique E preparation directory required')
    summary=public.load(primary_summary)
    public.require(summary['schema']=='trajkv.round10.development_summary.v1' and
        summary['status']=='PASS_ALL28_D160_CUSTOM_SOURCE_SCORE_PHYSICAL' and summary['closed_cells']==28 and
        summary['closed_answer_events']==1120 and summary['actual_closed_unique_IDs']==160,'actual closed primary28/1120 summary required')
    expected={(d,m) for d in public.DOMAINS for m in public.MODES}
    points={(p['dataset'],p['mode']):p for p in summary['points']}
    public.require(set(points)==expected and len(summary['points'])==28 and
        all(p['status']=='CLOSED_FULL_DOMAIN_CUSTOM_SOURCE_SCORE_PHYSICAL' and p['primary_mean'] is not None and
            p['expected_QA']==public.COUNTS[p['dataset']] for p in points.values()),'primary summary is partial or contains another scope')
    harvest.check_pins();items,origins=harvest.load_plans(plans)
    public.require(set(items)==expected,'actual plans must contain exact original primary28 only')
    public.require({str(Path(x['path']).resolve()):x['sha256'] for x in summary['plan_sources']}==
                   {str(Path(x['path']).resolve()):x['sha256'] for x in origins},'primary summary/actual execution plans differ')
    files=[];listed={};roles=Counter();watched={};metric_pins={};cell_proofs=[]
    def watch(path):
        path=Path(path).resolve();digest=public.sha(path)
        public.require(path not in watched or watched[path]==digest,'evidence changed during collection')
        watched[path]=digest;return digest
    def add(path,role,expected_sha=None,public_path=None,aliases=()):
        path=Path(path).resolve();public.require(public.inside(path,ROOT) or public.inside(path,E),'explicit public source outside ROOT/E')
        public.file_allowed(path,role);digest=watch(path)
        public.require(expected_sha is None or digest==expected_sha,'explicit source identity differs: '+str(path))
        key=(str(path),public_path)
        row={'path':str(path),'role':role,'sha256':digest}
        if public_path:row['public_path']=public_path
        if aliases:row['source_aliases']=list(aliases)
        if key in listed:
            old=listed[key];public.require(old['sha256']==digest,'different source at explicit inventory path')
            if aliases:old['source_aliases']=list(dict.fromkeys([*old.get('source_aliases',[]),*aliases]))
        else:files.append(row);listed[key]=row;roles[role]+=1
        return row
    # Check existing bindings without invoking the scorer. Every file in each
    # already retrieved proof is watched; only v1's later deep validation will
    # add that actual source/raw closure to the publication selection.
    for ds,mode in [(d,m) for d in public.DOMAINS for m in public.MODES]:
        item=items[ds,mode];name=item['cell']['name'];raw=Path(harvest_root)/'raw'/name;scored=Path(harvest_root)/'scored'/name
        public.require((scored/'summary.json').is_file(),'missing pre-existing primary score; no scoring allowed')
        manifest,done,cfg,proof=harvest.check_local_proof(raw,item)
        public.require(proof['parent_cell_success'] is True and done['status']=='PASS_RAW_COMPLETE' and
            done['counts']=={'PASS':public.COUNTS[ds]},'actual complete parent/source/raw binding required')
        for e in proof['files']:
            path=raw/e['file'];public.require(watch(path)==e['sha256'] and path.stat().st_size==e['bytes'],'retrieved primary proof file differs')
        for n in ('source_retrieval_receipt_round10_v1.json','retrieval_receipt.json'):watch(raw/n)
        scored_summary=public.load(scored/'summary.json');watch(scored/'summary.json')
        public.require(scored_summary['status']=='PASS_SCORED_COMPLETE' and scored_summary['expected_run_questions']==public.COUNTS[ds] and
            scored_summary['primary_mean']==points[ds,mode]['primary_mean'] and
            scored_summary['integrity']['scorer_sha256']==harvest.PINS['code/score.py'] and
            scored_summary['integrity']['raw_files']=={n:watch(raw/n) for n in ('manifest.json','records.jsonl','complete.json')},
            'actual primary raw/score/summary binding differs')
        public.require(scored_summary['integrity']['refs_sha256']==watch(Path(refs_root)/PurePosixPath(cfg['inputs']).name.replace('_inputs.jsonl','_refs.jsonl')) and
            scored_summary['integrity']['dataset_source_lock_sha256']==watch(Path(refs_root)/'source_lock.json'), 'frozen reference/source lock binding differs')
        for n,digest in scored_summary['derived_file_hashes'].items():public.require(watch(scored/n)==digest,'derived score file differs')
        for p,digest in scored_summary['integrity']['metric_code_hashes'].items():
            public.require(p not in metric_pins or metric_pins[p]==digest,'metric epoch differs across primary cells');metric_pins[p]=digest
        cell_proofs.append({'dataset':ds,'mode':mode,'expected_QA':public.COUNTS[ds],'raw':str(raw),'scored':str(scored),
            'primary_mean':scored_summary['primary_mean'],'raw_status':done['status'],'source_proof_status':proof['status'],
            'actual_parent_committed_exit0':True,'scored_summary_sha256':watch(scored/'summary.json')})
    cal=Path(calibration_root).resolve();proof_path=cal/'remote_calibration_retrieval_full_v1.json';proof=public.load(proof_path)
    public.require(proof['status']=='PASS_REMOTE_CALIBRATION_SOURCE_AND_RAW_SHA' and proof['remote_hashes_computed_independently'] is True,
                   'actual full cal64 independent source retrieval required')
    calfiles=[]
    for e in proof['files']:
        rel=PurePosixPath(e['file']);public.require(not rel.is_absolute() and '..' not in rel.parts,'unsafe cal64 proof member')
        p=cal.joinpath(*rel.parts);public.file_allowed(p,'teacher_calibration_scalar_or_source')
        public.require(watch(p)==e['sha256'] and p.stat().st_size==e['bytes'],'actual cal64 proof file differs')
        calfiles.append({'path':str(p),'sha256':e['sha256'],'source_aliases':[e['remote_file']]})
    cal_by_name={e['file']:e for e in proof['files']}
    required_cal={'manifest.json','records.jsonl','complete.json','fit_collection_complete.json','value_theta_selected.json',
                  'window_selected.json','value_theta_LS_proposal.json','source_config.json','source_job.json','launch_receipt.json','source_inventory.json'}
    public.require(required_cal<=set(cal_by_name),'cal64 full scalar/source proof incomplete')
    done=public.load(cal/'complete.json');manifest=public.load(cal/'manifest.json');inventory=public.load(cal/'source_inventory.json')
    public.require(done['status']=='PASS_LS48_AND_ACTUAL_MODEL_SELECT16' and done['actual_GPU'] is True and
        done['references_read'] is False and done['fit_sources']==48 and done['select_sources']==16 and
        done['source_hashes']==manifest['source_hashes'],'actual cal64 selected parameter closure differs')
    for n,digest in {'records.jsonl':done['records_sha256'],'value_theta_selected.json':done['theta_sha256'],'window_selected.json':done['window_sha256']}.items():
        public.require(watch(cal/n)==digest,'cal64 actual scalar identity differs')
    for rel,digest in done['source_hashes'].items():
        public.require(inventory[rel]==digest and cal_by_name['source_files/'+rel]['sha256']==digest,'cal64 actual source graph missing/different')
    # Create only new exact copies and inventory outputs after all source checks.
    output.mkdir(parents=True)
    report_source=Path(report_source).resolve();report_sha=public.sha(report_source)
    mirror=output/'mirrored_reports'/report_source.name;mirror.parent.mkdir()
    mirror.write_bytes(report_source.read_bytes());public.require(public.sha(mirror)==report_sha,'scope report exact-byte mirror differs')
    add(mirror,'scope_report',report_sha,aliases=[str(report_source)])
    base=Path(primary_summary).parent
    for n,role in [('summary.json','primary_closed_summary'),('quality_complete_bytes.csv','quality_bytes_table'),
                   ('instrumented_timing_separate.json','instrumented_timing_table'),('paired_comparisons.json','primary_paired_comparisons')]:add(base/n,role)
    add(Path(refs_root)/'INPUT_MANIFEST.json','input_manifest')
    add(Path(refs_root)/'CAL_INPUT_MANIFEST_v1.json','calibration_input_manifest')
    add(Path(refs_root)/'source_lock.json','source_lock')
    for ds in public.DOMAINS:
        n=public.COUNTS[ds]
        add(Path(refs_root)/f'{ds}_dev{n}_inputs.jsonl','source_inputs')
        add(Path(refs_root)/f'{ds}_dev{n}_refs.jsonl','reference_answers')
    for n in ('cal_fit48_inputs.jsonl','cal_select16_inputs.jsonl'):add(Path(refs_root)/n,'calibration_source_inputs')
    add(ROOT/'code/score.py','frozen_scoring_source',harvest.PINS['code/score.py'])
    for rel,digest in sorted(metric_pins.items()):add(ROOT/rel,'frozen_metric_source',digest)
    add(ROOT/'code/compare_runs.py','frozen_pairwise_comparison_source',harvest.PINS['code/compare_runs.py'])
    add(ROOT/'ops/round10_summarize_calibration_v1.py','frozen_calibration_scalar_recompute_source')
    add(ROOT/'ops/round10_harvest_score_v1.py','actual_primary_harvest_source')
    add(ROOT/'ops/round10_fetch_calibration_v1.py','actual_calibration_retrieval_source')
    add(Path(__file__),'primary_scope_preparation_source')
    for n in ('license.txt','processors.txt','evaluators.txt'):add(ROOT/'third_party/textvqa_metric_source'/n,'frozen_metric_license_and_upstream_text')
    dependency=output/'PUBLIC_COLD_DEPENDENCIES.txt'
    dependency.write_text('Cold public verification/scoring uses Python 3.10+ and NumPy. It reads archived text/scalars only; no private transport, GPU model, PT/W_O, image or weight payload is required. The exporter and remote producer are not claimed publicly rerunnable.\n',encoding='utf-8',newline='\n')
    add(dependency,'public_cold_dependency_note')
    public.require(public.REQUIRED_ROLES<=set(roles),'explicit public source role gap: '+str(public.REQUIRED_ROLES-set(roles)))
    c=public.candidate();c.update(phase_label='ROUND10_PRIMARY_D160_CLOSED_28',stage_state='PENDING_ROOT_SEAL',root_seal=None,
        plans=[{'path':str(Path(p).resolve()),'sha256':watch(p)} for p in plans],harvest_root=str(Path(harvest_root).resolve()),
        refs_root=str(Path(refs_root).resolve()),cells=[{'dataset':d,'mode':m,'expected_QA':public.COUNTS[d],'disposition':'CLOSED_COMPLETE'}
        for d in public.DOMAINS for m in public.MODES],files=files,
        calibration={'disposition':'CLOSED_COMPLETE','raw':str(cal),
            'independent_source_proof':{'path':str(proof_path),'sha256':watch(proof_path)},
            'public_scalar_and_source_files':calfiles,'new_free_generation_task_answers':0},
        scope='Exact original primary28/1120 D160 development only; separate middle context in the mirrored stage report adds no primary quality cells or events.')
    candidate_path=output/'PRIMARY_V1_CONTRACT_CANDIDATE.json';write_json(candidate_path,c)
    readiness,_,_=public.validate(candidate_path,deep=False)
    public.require(readiness['publication_eligible'] is False and set(readiness['missing'])=={'Root final stage seal','root_seal path/SHA'},
                   'candidate must be waiting only for Root seal')
    write_json(output/'READINESS.json',readiness)
    write_json(output/'EXPLICIT_FILE_ROLES.json',{'schema':'trajkv.round10.primary_public_file_roles.v1','files':files,
        'calibration_files':calfiles,'role_counts':dict(roles),'calibration_files_role':'teacher_calibration_scalar_or_source'})
    write_json(output/'ROOT_SEAL_BINDING_INPUTS.json',{'schema':'trajkv.round10.root_seal_binding_inputs.v1','status':'PENDING_ROOT_SEAL_NOT_A_SEAL',
        'candidate_path':str(candidate_path),'candidate_sha256':public.sha(candidate_path),'phase_label':c['phase_label'],
        'cells_disposition_sha256':public.canonical(c['cells']),'calibration_disposition_sha256':public.canonical(c['calibration']),
        'pure_seal_payload_helper':'ops/build_round10_primary_public_candidate_v1.py:seal_payload','root_seals_written':0})
    public.require(all(public.sha(p)==h for p,h in watched.items()) and public.sha(report_source)==report_sha,
                   'frozen source/raw/score/report changed during preparation')
    result={'status':'PASS_PRIMARY28_ACTUAL_SOURCE_SCORE_BINDINGS_PREPARED_WAIT_ROOT_SEAL','primary_cells':28,'actual_primary_answer_events':1120,
        'actual_distinct_development_IDs':160,'calibration_source_files':len(done['source_hashes']),'calibration_archived_text_files':len(calfiles),
        'explicit_support_files':len(files),'role_counts':dict(roles),'cell_proofs':cell_proofs,
        'mirrored_report':{'source':str(report_source),'mirror':str(mirror),'source_sha256':report_sha,'mirror_sha256':public.sha(mirror),'bytes':mirror.stat().st_size},
        'contract_sha256':public.sha(candidate_path),'builder_sha256':public.sha(__file__),
        'raw_source_and_existing_scores_unchanged':True,'new_scores_created':0,'new_GPU_QA':0,'reserve224_model_outputs_read':False,
        'primary_only_middle_and_KRA_cells_added':0,'root_seals_written':0,'exports_created':0,'Git_GitHub_SSH_mutations':False,
        'publication_eligible':False,'whole_Round10_goal_complete_claimed':False}
    write_json(output/'PRESEAL_AUDIT.json',result)
    print(json.dumps({k:result[k] for k in ('status','primary_cells','actual_primary_answer_events','calibration_source_files','explicit_support_files','contract_sha256')},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primary-summary',type=Path,default=E/'results/round10/primary_D160_closed_v1/summary.json')
    p.add_argument('--harvest-root',type=Path,default=E/'results/round10/harvest_v1')
    p.add_argument('--refs-root',type=Path,default=E/'results/round10/input_preparation_v1/inputs')
    p.add_argument('--calibration-root',type=Path,default=E/'results/round10/cal64_v1')
    p.add_argument('--report-source',type=Path,default=ROOT.parent/'TrajKV/docs/REPORT_2026-11-22_Round10_D160_首批与中档结果_阶段记录.md')
    p.add_argument('--plans',nargs='+',type=Path,default=[E/'results/round10/bootstrap_v1'/n for n in ('REFERENCE_JOB_PLAN_v1.json','CANDIDATE_JOB_PLAN_v1.json')])
    a=p.parse_args();build(a.output,primary_summary=a.primary_summary,harvest_root=a.harvest_root,refs_root=a.refs_root,
        calibration_root=a.calibration_root,report_source=a.report_source,plans=a.plans)


if __name__=='__main__':main()
