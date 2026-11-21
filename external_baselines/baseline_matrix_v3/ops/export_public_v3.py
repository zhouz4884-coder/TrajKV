"""Streaming export of one primary phase and independently declared auxiliaries.

No GitHub calls. P1 exporter/stages/journals are read-only and unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile
from export_public import Sanitizer,security_findings,TEXT_SUFFIXES,BLOCK_DIRS,AUDIT_ROOT_GLOBS,AUDIT_DIRS,NAMESPACE
from public_phase_contract_v2 import ROOT,EVIDENCE,load,sha,require,resolve,inside,contract_from_arguments,validate_phase,source_closure

sys.dont_write_bytecode=True


def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def io_path(value):
    path=Path(value).resolve();prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(path)) if os.name=='nt' and not str(path).startswith(prefix) else path


def text_chunks(path,normalize_lf=False):
    with Path(path).open('rb') as stream:
        pending=b''
        for part in iter(lambda:stream.read(1024**2),b''):
            part=pending+part;pending=b''
            if normalize_lf and part.endswith(b'\r'):part,pending=part[:-1],b'\r'
            yield part.replace(b'\r\n',b'\n') if normalize_lf else part
        if pending:yield pending


def identity(path,binary=False):
    normal=hashlib.sha256();size=0
    for part in text_chunks(path,not binary):normal.update(part);size+=len(part)
    blob=hashlib.sha1(f'blob {size}\0'.encode())
    for part in text_chunks(path,not binary):blob.update(part)
    return {'sha256':normal.hexdigest(),'blob_oid_sha1':blob.hexdigest(),'bytes':size}


def files(folder,technical=False):
    if not folder.exists():return
    for path in folder.rglob('*'):
        if not path.is_file() or set(path.relative_to(folder).parts)&BLOCK_DIRS:continue
        if technical:
            if path.suffix.lower() not in ('.json','.jsonl') or path.stat().st_size>16*1024**2:continue
        elif path.suffix.lower() not in TEXT_SUFFIXES|{'.npy'} and not path.name.upper().startswith(('LICENSE','COPYING','NOTICE')):continue
        if path.name=='artifact_pointer.json':continue
        yield path


def collect(contract,contract_path,matrix_path,artifacts,closure):
    selected={};duplicates=[]
    def add(path,relative=None,large=False):
        path=resolve(path);origin=ROOT if inside(path,ROOT) else EVIDENCE
        relative=relative or path.relative_to(origin).as_posix()
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Invalid public relative path')
        if not large:require(path.stat().st_size<=64*1024**2,'Review unexpected large source separately: '+relative)
        if relative in selected:
            previous=selected[relative]
            require(sha(previous['path'])==sha(path),'Differing frozen copies of public path: '+relative)
            duplicates.append(relative);return
        info=path.stat()
        selected[relative]={'path':path,'origin':'project' if origin==ROOT else 'evidence_mirror',
            'source_path':path.relative_to(origin).as_posix(),'stat':(info.st_size,info.st_mtime_ns,info.st_ctime_ns),
            'large':large}
    for name in ('code','third_party','configs/methods','configs/models','configs/eval','configs/protocol','configs/p0',
                 'configs/matrix','configs/cache','configs/speed','configs/datasets/v3','configs/publication'):
        for path in files(ROOT/name) or []:add(path)
    for path in (ROOT/'configs').glob('*.json'):add(path)
    # Reports enter through fixed completed-evidence inventories. A concurrently
    # produced result report must not silently enter an earlier phase cutoff.
    for pattern in AUDIT_ROOT_GLOBS:
        for path in (ROOT/'audit').glob(pattern):add(path)
    for name in AUDIT_DIRS:
        for path in files(ROOT/'audit'/name) or []:add(path)
    for pattern in ('GITHUB_*DELIVERY*.json','*ANONYMOUS*VERIFICATION*.json'):
        for path in (ROOT/'audit/publication').glob(pattern):add(path)
    for base in (ROOT,EVIDENCE):
        for name in ('native_p0','qwen_bridge','p0'):
            for path in files(base/'results'/name,technical=True) or []:add(path)
        family=contract.get('family','').lower()
        for name in ('results/matrix','results/comparisons'):
            for path in files(base/name) or []:
                if family and family in path.as_posix().lower():add(path)
    for node in closure:
        for path in node['files']:add(path,large=True)
    for artifact in artifacts:
        if not inside(artifact['config'],ROOT):
            add(artifact['config'],f"configs/publication/{contract['phase_label']}/{artifact['config'].stem}.json")
        if artifact['scored']:
            for name in ('summary.json','scores.csv','scored_records.jsonl'):add(artifact['scored']/name,large=True)
    if not inside(matrix_path,ROOT):add(matrix_path,f"configs/publication/{contract['phase_label']}/matrix.json")
    if contract_path and not inside(contract_path,ROOT):add(contract_path,f"configs/publication/{contract['phase_label']}/source_contract.json")
    def relative(path):
        path=resolve(path)
        for rel,entry in selected.items():
            if entry['path']==path:return rel
        raise ValueError('Selected evidence path missing: '+str(path))
    artifact_refs=[{'cell':a['cell'],'raw_dir':relative(a['raw']/'manifest.json').rsplit('/',1)[0],
        'scored_dir':relative(a['scored']/'summary.json').rsplit('/',1)[0] if a['scored'] else None,
        'config':relative(a['config']),'kind':a['kind'],'expected_QA':a['expected_QA']} for a in artifacts]
    closure_refs=[{'raw_dir':relative(node['path']/'manifest.json').rsplit('/',1)[0],'kind':node['kind']} for node in closure]
    return selected,duplicates,artifact_refs,closure_refs,relative(matrix_path)


def expected_in_git(relative,size):
    if Path(relative).name in ('records.jsonl','scored_records.jsonl'):return False
    if relative.startswith('evidence/line_identities/'):return False
    if relative.startswith(('results/p0/','audit/')) and size>2*1024**2:return False
    return size<=16*1024**2


def write_public_file(relative,source,public,sanitizer):
    """Stream large JSONL; original/public/LF identities stay separate."""
    path=source['path'];target=public/relative;target.parent.mkdir(parents=True,exist_ok=True)
    original=hashlib.sha256();line_path=None;line_count=0;findings=set()
    if path.suffix.lower()=='.jsonl':
        line_path='evidence/line_identities/'+hashlib.sha256(relative.encode()).hexdigest()[:24]+'.jsonl'
        side=public/line_path;side.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as stream,target.open('xb') as out,side.open('xb') as lines:
            for number,raw in enumerate(stream,1):
                original.update(raw);require(raw.strip(),'Blank source JSONL row')
                obj=json.loads(raw);value=sanitizer.tree(obj)
                body=raw if value==obj else (json.dumps(value,ensure_ascii=False)+'\n').encode()
                findings.update(security_findings(relative,body));out.write(body)
                item={'line':number,'id':obj.get('id'),'source_sha256':hashlib.sha256(raw).hexdigest(),
                      'public_sha256':hashlib.sha256(body).hexdigest()}
                lines.write((json.dumps(item,separators=(',',':'))+'\n').encode());line_count+=1
    else:
        raw=path.read_bytes();original.update(raw);body,_=sanitizer.transform(relative,raw)
        findings.update(security_findings(relative,body));target.write_bytes(body)
    source_sha=original.hexdigest();public_sha=sha(target)
    require(sha(path)==source_sha,'Source changed while exporting: '+relative)
    info=path.stat();require((info.st_size,info.st_mtime_ns,info.st_ctime_ns)==source['stat'],'Source stat changed during export: '+relative)
    entry={'source_origin':source['origin'],'source_path':source['source_path'],'source_bytes':info.st_size,
        'source_sha256':source_sha,'public_bytes':target.stat().st_size,'public_sha256':public_sha,
        'transformation':'EXACT' if source_sha==public_sha else 'DERIVED_PATH_TEMPLATES',
        'git_lf':identity(target,path.suffix.lower()=='.npy'),'expected_in_git':expected_in_git(relative,target.stat().st_size)}
    if line_path:entry.update(line_identities_file=line_path,line_count=line_count)
    return entry,line_path,sorted(findings)


def generated_entry(path,expected_git=True):
    return {'source_origin':'generated_public_helper_or_identity','source_path':None,'source_bytes':None,'source_sha256':None,
        'public_bytes':path.stat().st_size,'public_sha256':sha(path),'transformation':'GENERATED_PUBLIC_HELPER_OR_IDENTITY',
        'git_lf':identity(path),'expected_in_git':expected_git}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--contract',type=Path);parser.add_argument('--matrix',type=Path);parser.add_argument('--phase-label')
    parser.add_argument('--aux-contracts',nargs='*',type=Path,default=[])
    parser.add_argument('--support-file-lists',nargs='*',type=Path,default=[])
    parser.add_argument('--fixture-only',action='store_true')
    parser.add_argument('--view-root',type=Path);parser.add_argument('--score-root',type=Path);parser.add_argument('--coverage-scope')
    parser.add_argument('--require-complete',action='store_true');parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args();require(re.fullmatch(r'[A-Za-z0-9_-]+',args.stage),'Plain new stage name required')
    contract,contract_path=contract_from_arguments(args)
    require(re.fullmatch(r'[A-Za-z0-9_.-]+',contract['phase_label']),'Plain phase-label slug required')
    phase,artifacts,matrix_path=validate_phase(contract)
    groups=[{'contract':contract,'contract_path':contract_path,'phase':phase,'artifacts':artifacts,'matrix_path':matrix_path}]
    for auxiliary_path in args.aux_contracts:
        auxiliary_path=resolve(auxiliary_path);auxiliary=load(auxiliary_path)
        require(auxiliary['schema']=='baseline_matrix.public_phase_contract.v2','Unknown auxiliary contract')
        extra_phase,extra_artifacts,extra_matrix=validate_phase(auxiliary)
        groups.append({'contract':auxiliary,'contract_path':auxiliary_path,'phase':extra_phase,
                       'artifacts':extra_artifacts,'matrix_path':extra_matrix})
    require(len({g['phase']['phase_label'] for g in groups})==len(groups),'Duplicate independent phase label')
    require(len({str(a['config']) for g in groups for a in g['artifacts']})==sum(len(g['artifacts']) for g in groups),
            'Duplicate artifact across phase groups; use one shared supporting phase')
    complete=all(g['phase']['status']=='PASS_DECLARED_MATRIX_COMPLETE' for g in groups)
    if args.require_complete and not complete:
        print(json.dumps({'status':'REJECTED_INCOMPLETE_DECLARED_PHASE_GROUP','phases':[g['phase'] for g in groups],
                          'github_mutation_performed':False}));raise SystemExit(3)
    if args.validation_only:
        print(json.dumps({'status':'PHASE_GROUPS_VALIDATED_NO_EXPORT','phase':phase,
            'auxiliary_phases':[g['phase'] for g in groups[1:]],'github_mutation_performed':False}));return
    selected={};duplicates=[];closure_by_path={}
    for group in groups:
        closure=source_closure(group['artifacts'])
        chosen,dups,refs,closed_refs,matrix_ref=collect(group['contract'],group['contract_path'],group['matrix_path'],group['artifacts'],closure)
        for relative,entry in chosen.items():
            if relative in selected:
                require(sha(selected[relative]['path'])==sha(entry['path']),'Conflicting evidence between independent phase groups: '+relative)
                duplicates.append(relative)
            else:selected[relative]=entry
        duplicates.extend(dups)
        group.update(artifact_refs=refs,closure_refs=closed_refs,matrix_ref=matrix_ref)
        for node in closure:closure_by_path[str(node['path'])]=node
    artifact_refs=groups[0]['artifact_refs'];matrix_ref=groups[0]['matrix_ref']
    closure_refs=list({ref['raw_dir']:ref for group in groups for ref in group['closure_refs']}.values())
    auxiliary_refs=[{'phase':group['phase'],'contract':group['contract'],
                     'phase_artifacts':group['artifact_refs'],'source_closure':group['closure_refs'],
                     'declared_matrix_file':group['matrix_ref']} for group in groups[1:]]
    support_groups=[]
    for list_path in args.support_file_lists:
        list_path=resolve(list_path);support=load(list_path)
        require(support['schema']=='baseline_matrix.public_support_files.v1','Unsupported support-evidence file list')
        require(support.get('completed_evidence_only') is True,'Supporting evidence must explicitly be complete')
        require(support.get('primary_quality_QA_added',0)==0,'Supporting controls cannot add primary-quality QA')
        require(support.get('quality_QA_added',0)==0,'Timing/support trials cannot add quality QA')
        additions=[{'path':str(list_path),'sha256':sha(list_path),'role':'support_inventory'}]+support['files']
        refs=[]
        for row in additions:
            path=resolve(row['path']);require(sha(path)==row['sha256'],'Supporting evidence changed before snapshot')
            require(path.suffix.lower() in TEXT_SUFFIXES|{'.log'} or path.name.upper().startswith(('LICENSE','COPYING','NOTICE')),
                    'No tensor/image/weight dump in support evidence')
            require(not set(path.parts)&BLOCK_DIRS and path.stat().st_size<=64*1024**2,'Unreviewed support file size or path')
            origin=ROOT if inside(path,ROOT) else EVIDENCE;relative=path.relative_to(origin).as_posix();stat=path.stat()
            if relative in selected:require(sha(selected[relative]['path'])==row['sha256'],'Conflicting support evidence path')
            else:selected[relative]={'path':path,'origin':'project' if origin==ROOT else 'evidence_mirror',
                'source_path':relative,'stat':(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns),'large':False}
            refs.append({'file':relative,'source_sha256':row['sha256'],'role':row.get('role','completed_support')})
        counts=support.get('counts',{key:value for key,value in support.items() if key.startswith('counts_') or
            key in ('primary_quality_QA_added','quality_QA_added','new_trials','previous_support_trials_excluded_from_new',
                    'source_representative_conditional_trials_unrun')})
        support_groups.append({'label':support['label'],'scope':support['scope'],
            'counts':counts,'not_part_of_primary_or_auxiliary_quality_QA':True,'files':refs,
            'operator_rerun_requires_nonpublic_capture_tensors':support.get('operator_rerun_requires_nonpublic_capture_tensors',False),
            'collection_performed_new_inference':support.get('collection_performed_new_inference',False)})
    target=(EVIDENCE/'public_staging'/args.stage).resolve();require(inside(target,EVIDENCE/'public_staging'),'Export must stay on named E evidence root')
    require(not target.exists(),'Use a new immutable stage name');target.mkdir(parents=True)
    public=io_path(target/'tree'/NAMESPACE);public.mkdir(parents=True);sanitizer=Sanitizer();entries={};unsafe=[]
    for count,(relative,source) in enumerate(sorted(selected.items()),1):
        entry,side,findings=write_public_file(relative,source,public,sanitizer);entries[relative]=entry
        if side:entries[side]=generated_entry(public/side,False)
        if findings:unsafe.append({'path':relative,'findings':findings})
        if count%100==0:print(json.dumps({'status':'EXPORTING_V3','processed_files':count}),flush=True)
    scope=phase['coverage_scope']
    readme=(f"# External baseline increment: {phase['phase_label']}\n\nDeclared scope: {scope}\n\n"
        f"Closed/scored cells: {phase['verified_scored_cells']}/{phase['expected_cells']}; questions: {phase['verified_scored_questions']}/{phase['expected_questions']}.\n\n"
        "The primary phase totals above exclude the independently declared auxiliary groups below. "
        "Auxiliary view counts do not imply additional unique GPU inference; provenance retains reused predictions. "
        "Reports are bound by explicit completed-evidence inventories; source/configuration files may also describe unrun targets. "
        "Completion is assessed only against these declared groups, not the whole external SOTA library or all P2/P3 research. "
        "A progress preview cannot be published by the reviewed publisher. Derived views retain original predictions and instrumented timing; "
        "they are not new GPU inference runs. All original donor runs and source/retrieval proofs are archived, including donors from earlier phases.\n\n"
        "PUBLIC_CONTENT_REFS.json separates original source, sanitized archive and LF-normalized Git identities. "
        "Original SHA fields remain source identities. Per-line maps are separate archived JSONL sidecars. "
        "Code/configuration/tables/small manifests are browsable in Git; full raw evidence is in the ZIP. "
        "Images, weights, tensor dumps, private routes and compiler caches are excluded.\n\n"
        "From the extracted namespace run `python -B code/verify_public_export.py --root . --recompute-scores`. "
        "This validates donor closure, exact prediction provenance, fixed-reference metrics, cluster CIs and declared coverage. "
        "Use exact archive bytes for source guards that distinguish original CRLF from Git LF. Runtime and method limitations remain explicit in reports.\n\n"+
        ''.join(f"- Auxiliary `{g['phase']['phase_label']}`: {g['phase']['expected_cells']} views / {g['phase']['expected_questions']} view questions. {g['phase']['coverage_scope']}\n" for g in groups[1:]))
    generated={'README_PUBLIC.md':readme.encode(),'.gitattributes':b'* text=auto eol=lf\n*.npy -text\n',
        'code/verify_public_export.py':(ROOT/'ops/verify_public_export_v3.py').read_bytes(),
        'evidence/DECLARED_PHASE.json':json_bytes(sanitizer.tree({'contract':contract,'phase':phase,'artifacts':artifact_refs,'closure':groups[0]['closure_refs']})),
        'evidence/AUXILIARY_PHASES.json':json_bytes(sanitizer.tree(auxiliary_refs))}
    generated['evidence/SUPPORTING_COMPLETED_EVIDENCE.json']=json_bytes(sanitizer.tree(support_groups))
    for relative,raw in generated.items():
        require(relative not in entries,'Generated path conflict');p=public/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        entries[relative]=generated_entry(p)
    for relative,source in selected.items():
        require(sha(source['path'])==entries[relative]['source_sha256'],
                'Original source changed after snapshot copy: '+relative)
    fixture=bool(args.fixture_only or any(g['contract'].get('fixture_only',False) for g in groups))
    publishable=complete and not unsafe and not fixture
    index={'schema':'baseline_matrix.public_content_refs.v3','namespace':NAMESPACE,'repository':'https://github.com/zhouz4884-coder/TrajKV',
        'release_tag':args.tag,'phase':sanitizer.tree(phase),'created_utc':datetime.now(timezone.utc).isoformat(),
        'publication_eligible':publishable,'fixture_only':fixture,
        'completion_scope':'declared_matrix_only','whole_external_library_completion_assessed':False,
        'phase_artifacts':artifact_refs,'source_closure':closure_refs,'declared_matrix_file':matrix_ref,'files':entries,
        'primary_source_closure':groups[0]['closure_refs'],
        'auxiliary_phases':sanitizer.tree(auxiliary_refs),'phase_group_totals_are_separate':True,
        'supporting_evidence_groups':sanitizer.tree(support_groups),
        'source_hash_fields_inside_derived_evidence':'Original immutable identities, resolved via file and per-line maps',
        'git_stage_plan':{'force_paths':[NAMESPACE+'/'+p for p,e in entries.items() if e['expected_in_git']],
            'plus_generated_index':NAMESPACE+'/PUBLIC_CONTENT_REFS.json','verify_actual_tree_and_blob_bodies':True},
        'publication_binding':'Publisher receipt binds fixed commit/tree/index blob and exact archive SHA; export is not upload'}
    (public/'PUBLIC_CONTENT_REFS.json').write_bytes(json_bytes(index))
    private={'status':'BLOCKED_UNSAFE_SELECTED_FILES' if unsafe else 'PASS_SAFE_EXPORT_V3','phase':phase,'unsafe':unsafe,
        'auxiliary_phases':[g['phase'] for g in groups[1:]],
        'duplicates_exact':duplicates,'source_paths_checked_within_project_or_E':True,'selected_files':len(selected),
        'normalization_counts':dict(sanitizer.counts),'github_mutation_performed':False}
    (target/'PRIVATE_EXPORT_AUDIT.json').write_bytes(json_bytes(private))
    if unsafe:print(json.dumps({'status':private['status'],'unsafe':unsafe}));raise SystemExit(2)
    archive=target/(args.stage+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zipped:
        for p in sorted(public.rglob('*')):
            if p.is_file():zipped.write(p,(Path(NAMESPACE)/p.relative_to(public)).as_posix())
    delivery={'schema':'baseline_matrix.public_delivery.v3','status':'PREPARED_NOT_PUBLISHED' if publishable else 'PREPARED_INCOMPLETE_PROGRESS_PREVIEW_NOT_PUBLISHABLE',
        'publication_eligible':publishable,'stage':args.stage,'namespace':NAMESPACE,'release_tag':args.tag,'phase':sanitizer.tree(phase),
        'completion_scope':'declared_matrix_only','whole_external_library_completion_assessed':False,
        'auxiliary_phases':sanitizer.tree([g['phase'] for g in groups[1:]]),'phase_group_totals_are_separate':True,
        'supporting_completed_evidence_groups':sanitizer.tree([{'label':g['label'],'scope':g['scope'],'counts':g['counts']} for g in support_groups]),
        'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
        'content_index':'tree/'+NAMESPACE+'/PUBLIC_CONTENT_REFS.json','content_index_sha256':sha(public/'PUBLIC_CONTENT_REFS.json'),
        'public_files':len(entries),'public_payload_bytes':sum(e['public_bytes'] for e in entries.values()),
        'expected_git_payload_files':sum(e['expected_in_git'] for e in entries.values()),
        'closed_views_or_raw_artifacts':len(artifact_refs),'auxiliary_closed_views_or_raw_artifacts':sum(len(g['artifact_refs']) for g in groups[1:]),
        'closure_original_runs':sum(n['kind']=='original_inference_run' for n in closure_refs),
        'github_mutation_performed':False}
    (target/'DELIVERY.json').write_bytes(json_bytes(delivery));print(json.dumps(delivery))


if __name__=='__main__':main()
