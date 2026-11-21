"""Cold verification of separately counted primary/auxiliary phase closures."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode=True


def require(value,message):
    if not value:raise ValueError(message)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def sha_stream(stream):
    h=hashlib.sha256()
    for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()
def sha(path):
    with Path(path).open('rb') as stream:return sha_stream(stream)
def relative(value):
    value=str(value).replace('\\','/')
    for marker in ('/data/registry/v3/','/data/refs/v3/'):
        if marker in value:return 'configs/datasets/v3/'+value.split(marker,1)[1]
    for marker in ('/results/','/configs/'):
        if marker in value:return marker.strip('/')+'/'+value.split(marker,1)[1]
    if value.startswith(('results/','configs/')):return value
    raise ValueError('Unresolvable public evidence path template')


def io_path(value):
    path=Path(value).resolve();prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(path)) if os.name=='nt' and not str(path).startswith(prefix) else path


def verify(root,archive=None,recompute_scores=False):
    root=io_path(root);index=load(root/'PUBLIC_CONTENT_REFS.json');entries=index['files']
    require(index['schema'] in ('baseline_matrix.public_content_refs.v2','baseline_matrix.public_content_refs.v3'),
            'Expected source/public identity schema')
    require(index['completion_scope']=='declared_matrix_only' and index['whole_external_library_completion_assessed'] is False,'Unsupported whole-library completion claim')
    total=0;line_count=0
    for name,entry in entries.items():
        path=root/name;require(path.resolve().is_relative_to(root),'Indexed path escapes public namespace')
        require(path.stat().st_size==entry['public_bytes'] and sha(path)==entry['public_sha256'],'Public file identity mismatch: '+name)
        lf=hashlib.sha256();blob=hashlib.sha1(f"blob {entry['git_lf']['bytes']}\0".encode());size=0
        with path.open('rb') as stream:
            pending=b''
            for piece in iter(lambda:stream.read(1024**2),b''):
                piece=pending+piece;pending=b''
                if path.suffix!='.npy' and piece.endswith(b'\r'):piece,pending=piece[:-1],b'\r'
                if path.suffix!='.npy':piece=piece.replace(b'\r\n',b'\n')
                lf.update(piece);blob.update(piece);size+=len(piece)
            if pending:lf.update(pending);blob.update(pending);size+=len(pending)
        require(size==entry['git_lf']['bytes'] and lf.hexdigest()==entry['git_lf']['sha256'] and blob.hexdigest()==entry['git_lf']['blob_oid_sha1'],'Git LF/blob identity mismatch: '+name)
        if entry['transformation']=='EXACT':require(entry['source_sha256']==entry['public_sha256'],'False exact-source assertion')
        if entry.get('line_identities_file'):
            side=entry['line_identities_file'];require(side in entries,'Missing line identity sidecar')
            count=0
            with path.open('rb') as data,(root/side).open(encoding='utf-8') as identities:
                for count,line in enumerate(data,1):
                    identity=json.loads(identities.readline());obj=json.loads(line)
                    require(identity['line']==count and identity['id']==obj.get('id') and identity['public_sha256']==hashlib.sha256(line).hexdigest(),'Line identity mismatch: '+name)
                require(not identities.readline(),'Extra line identities')
            require(count==entry['line_count'],'Line count mismatch');line_count+=count
        total+=entry['public_bytes']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==set(entries)|{'PUBLIC_CONTENT_REFS.json'},'Unindexed or missing public files')
    for group in index.get('supporting_evidence_groups',[]):
        require(group['not_part_of_primary_or_auxiliary_quality_QA'] is True,'Supporting controls mixed into quality QA counts')
        for row in group['files']:
            require(row['file'] in entries and entries[row['file']]['source_sha256']==row['source_sha256'],
                    'Supporting original evidence identity differs')
    if archive:
        prefix=index['namespace']+'/'
        with zipfile.ZipFile(archive) as zipped:
            require(set(zipped.namelist())=={prefix+p for p in entries}|{prefix+'PUBLIC_CONTENT_REFS.json'},'Archive member coverage mismatch')
            require(zipped.read(prefix+'PUBLIC_CONTENT_REFS.json')==(root/'PUBLIC_CONTENT_REFS.json').read_bytes(),'Archive content index mismatch')
            for name,entry in entries.items():
                with zipped.open(prefix+name) as stream:require(sha_stream(stream)==entry['public_sha256'],'Archive body mismatch: '+name)
    sys.path.insert(0,str(root/'code'))
    def public_module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
        sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
    # Use the verified public modules even when this verifier is called from
    # a process that already imported private workspace cache/metric modules.
    cache=public_module('_verified_public_prediction_cache_v2',root/'code/prediction_cache.py')
    previous_vendor=sys.modules.get('vendor_textvqa_metric')
    public_module('vendor_textvqa_metric',root/'code/vendor_textvqa_metric.py')
    metric=public_module('_verified_public_metrics_v2',root/'code/metrics.py');score_question=metric.score_question
    if previous_vendor is not None:sys.modules['vendor_textvqa_metric']=previous_vendor
    import numpy as np
    raw_meta={};raw_lines={};contracts={};visited=set();active=set()
    def entry(folder,name):
        path=folder+'/'+name;require(path in entries,'Evidence closure missing: '+path);return entries[path]
    def original_hash(folder,name):return entry(folder,name)['source_sha256']
    def original_graph(folder,name,expected):require(original_hash(folder,name)==expected,'Original SHA graph mismatch: '+folder+'/'+name)
    def source_file_reference(value):
        value=str(value).replace('\\','/')
        for alias,origin in [('${PROJECT_ROOT}/','project'),('${EVIDENCE_ROOT}/','evidence_mirror')]:
            if value.startswith(alias):
                matches=[p for p,e in entries.items() if e['source_origin']==origin and e['source_path']==value[len(alias):]]
                require(len(matches)==1,'Unresolved declared source file');return matches[0]
        if value in entries:return value
        candidate=relative(value)
        require(candidate in entries,'Declared source file missing');return candidate
    groups=[{'phase':index['phase'],'phase_artifacts':index['phase_artifacts'],
             'declared_matrix_file':index.get('declared_matrix_file'),
             'source_closure':index.get('primary_source_closure',index['source_closure'])}]+index.get('auxiliary_phases',[])
    require(len({g['phase']['phase_label'] for g in groups})==len(groups),'Duplicate independent phase labels')
    artifacts=[];declared={};artifact_group={}
    for group_number,group in enumerate(groups):
        phase=group['phase']
        require(phase['completion_scope']=='declared_matrix_only' and phase['whole_external_library_completion_assessed'] is False,
                'Unsupported auxiliary whole-library claim')
        matrix_name=group.get('declared_matrix_file')
        if matrix_name is None:
            candidates=[name for name,info in entries.items() if info['source_sha256']==phase['declared_matrix_source_sha256']]
            require(candidates,'Declared matrix original SHA not represented');matrix_name=candidates[0]
        require(entries[matrix_name]['source_sha256']==phase['declared_matrix_source_sha256'],'Wrong declared matrix source SHA')
        matrix=load(root/matrix_name);require(matrix['rows'],'Missing declared matrix rows')
        local={source_file_reference(row['config']):row for row in matrix['rows']}
        require(len(local)==len(matrix['rows'])==phase['expected_cells'] and
                sum(int(row['count']) for row in matrix['rows'])==phase['expected_questions'],'Declared independent matrix totals differ')
        require(not set(local)&set(declared),'Duplicate target across phases; shared reference views require their own support group')
        declared.update(local)
        require(len(group['phase_artifacts'])==len({a['config'] for a in group['phase_artifacts']}),'Duplicate phase artifact')
        require(all(a['config'] in local for a in group['phase_artifacts']),'Artifact lies outside its declared independent phase')
        for artifact in group['phase_artifacts']:
            artifacts.append(artifact);artifact_group[artifact['config']]=group_number
    def line_metadata(folder):
        path=folder+'/records.jsonl';side=entry(folder,'records.jsonl')['line_identities_file'];rows=[]
        with (root/path).open('rb') as stream,(root/side).open(encoding='utf-8') as identity:
            for offset,line in enumerate(stream,1):
                row=json.loads(line);proof=json.loads(identity.readline())
                rows.append({'id':row['id'],'status':row['status'],'source_line_sha256':proof['source_sha256'],
                    'public_line_sha256':proof['public_sha256'],'input_key_sha256':cache.input_key(row),
                    'execution_sha256':cache.execution_fingerprint(row)})
        return rows
    def check_proof(folder,name,required,status):
        proof=load(root/folder/name)
        require(proof['status']==status and proof['remote_hashes_computed_independently'] is True,'Missing independent original proof')
        values={p['file']:p for p in proof['files']};require(set(values)==set(required),'Proof file-set mismatch')
        for filename in required:
            original_graph(folder,filename,values[filename]['sha256'])
            require(entry(folder,filename)['source_bytes']==values[filename]['bytes'],'Original proof length mismatch')
    def public_contract(manifest,inventory,source_eval):
        identity=manifest['model_identity']
        return {'schema':'baseline_matrix.effective_prediction_contract.v1',
            'behavior':cache.normalize_behavior(manifest['config'],manifest['model_spec']),'model_spec':manifest['model_spec'],
            'model_files':sorted(({k:r[k] for k in ('file','bytes','sha256')} for r in identity['files']),key=lambda r:r['file']),
            'runtime':{k:manifest[k] for k in ('torch_version','transformers_version','cuda_version','python_version','hardware','runtime_executable')},
            'processor_id':manifest['processor_id'],'processor_version':manifest['processor_version'],
            'processor_file_hashes':manifest['processor_file_hashes'],'generation_native':manifest['generation_native'],
            'generation_overrides':manifest['generation_overrides'],'relevant_source_hashes':cache._dependencies(manifest,inventory,source_eval)}
    def walk(folder):
        if folder in visited:return
        require(folder not in active,'Cyclic public donor closure');active.add(folder)
        m=load(root/folder/'manifest.json');c=load(root/folder/'complete.json')
        original_graph(folder,'manifest.json',c['manifest_sha256']);original_graph(folder,'records.jsonl',c['records_jsonl_sha256'])
        require(c['coverage_complete'] is True and c['quality_scored'] is False and m['references_read'] is False,'Raw/view reference-free coverage contract differs')
        rows=line_metadata(folder);raw_lines[folder]=rows
        require([r['id'] for r in rows]==m['expected_ids'] and len(rows)==len({r['id'] for r in rows})==m['expected_QA']==c['expected_QA']==c['recorded_QA'],'Raw/view coverage/order mismatch')
        require(dict(Counter(r['status'] for r in rows))==c['counts'],'Raw/view status counts mismatch')
        if m['schema']=='baseline_matrix.derived_prediction_view.v1':
            require(m['inference_run'] is False and m['prediction_cache_used'] is True and c['derived_prediction_view'] is True and c['inference_run'] is False,'Derived view mislabeled as inference')
            original_graph(folder,'provenance.jsonl',c['provenance_jsonl_sha256'])
            donors={}
            for descriptor in m['source_runs']:
                donor=relative(descriptor['raw_dir']);walk(donor);donors[donor]=descriptor
                require(descriptor['original_inference_run'] is True,'Expected original GPU donor')
                for filename,digest in {**descriptor['raw_hashes'],**descriptor['source_proof_hashes']}.items():original_graph(donor,filename,digest)
                original_graph(donor,'retrieval_receipt.json',descriptor['retrieval_receipt_sha256'])
                original_graph(donor,'source_retrieval_receipt_v2.json',descriptor['source_retrieval_receipt_sha256'])
                require(descriptor['effective_contract_sha256']==m['effective_contract_sha256'],'Original effective-contract identity differs')
                require(contracts[donor]==m['effective_contract'],'Public normalized effective contract differs after path templating')
            roles=Counter()
            with (root/folder/'provenance.jsonl').open(encoding='utf-8') as stream:
                for position,row in enumerate(rows,1):
                    proof=json.loads(stream.readline());donor=relative(proof['source_run'])
                    require(donor in donors and proof['role'] in ('fresh','reused'),'Unknown donor/origin role')
                    number=proof['source_line_number'];require(1<=number<=len(raw_lines[donor]),'Invalid original source row number')
                    original=raw_lines[donor][number-1]
                    require(proof['id']==row['id']==original['id'] and row['source_line_sha256']==original['source_line_sha256']==proof['source_exact_line_sha256'],'Original row identity changed across derived view')
                    require(row['public_line_sha256']==original['public_line_sha256'],'Sanitized derived/source row bytes differ')
                    require(proof['source_raw_hashes']==donors[donor]['raw_hashes'],'Per-row donor SHA set differs')
                    require(proof['input_key_sha256']==row['input_key_sha256']==original['input_key_sha256'] and
                            proof['source_execution_sha256']==row['execution_sha256']==original['execution_sha256'],'Input/execution identity differs')
                    require(proof['effective_contract_sha256']==m['effective_contract_sha256'],'Per-row contract differs')
                    roles[proof['role']]+=1
                require(not stream.readline(),'Unexpected extra provenance')
            require(dict(roles)==m['QA_origin_counts']==c['QA_origin_counts'],'Fresh/reused accounting differs')
            contracts[folder]=m['effective_contract']
        else:
            require(m['schema']=='baseline_matrix.raw_inference.v1','Unknown original donor schema')
            check_proof(folder,'retrieval_receipt.json',('manifest.json','records.jsonl','complete.json'),'PASS_REMOTE_RAW_SHA256_EXACT')
            check_proof(folder,'source_retrieval_receipt_v2.json',('source_inventory.json','launch_receipt.json','source_evaluate.py'),'PASS_REMOTE_SOURCE_SHA256_EXACT')
            inventory=load(root/folder/'source_inventory.json');launch=load(root/folder/'launch_receipt.json')
            require(launch['source_inventory_sha256']==original_hash(folder,'source_inventory.json'),'Launch/inventory identity differs')
            require(launch['source_snapshot']==m['runtime_source_root'] and launch['runtime_python']==m['runtime_executable']==launch['argv_executed'][0],'Original runtime/snapshot binding differs')
            require(inventory['code/evaluate.py']==original_hash(folder,'source_evaluate.py'),'Original evaluate source differs')
            require(sha(root/folder/'source_evaluate.py')==inventory['code/evaluate.py'],'Producer evaluate source was transformed; explicit additional AST map required')
            contracts[folder]=public_contract(m,inventory,root/folder/'source_evaluate.py')
        raw_meta[folder]=(m,c);visited.add(folder);active.remove(folder)
    for artifact in artifacts:walk(artifact['raw_dir'])
    require(visited=={n['raw_dir'] for n in index['source_closure']},'Source closure is incomplete or contains unclaimed nodes')
    def reachable(folder,seen):
        if folder in seen:return
        seen.add(folder);manifest=raw_meta[folder][0]
        for descriptor in manifest.get('source_runs',[]):reachable(relative(descriptor['raw_dir']),seen)
    for group in groups:
        own=set()
        for artifact in group['phase_artifacts']:reachable(artifact['raw_dir'],own)
        require(own=={node['raw_dir'] for node in group['source_closure']},'Independent phase donor closure differs')
    scoring=[];complete_cells=Counter();complete_questions=Counter()
    source_lock=load(root/'configs/datasets/v3/source_lock.json')
    for artifact in artifacts:
        folder=artifact['raw_dir'];m,c=raw_meta[folder];cfg=load(root/artifact['config'])
        require(artifact['config'] in declared,'Artifact outside declared matrix')
        declaration=declared[artifact['config']]
        require(int(declaration['count'])==artifact['expected_QA'],'Artifact declared count differs')
        if declaration.get('model') is not None:require(cfg['model_id']==declaration['model'],'Declared model differs')
        require(cfg==m['config'],'Declared target/view configuration differs')
        inputs=relative(cfg['inputs']);refs=inputs.replace('_inputs.jsonl','_refs.jsonl')
        for path in (inputs,refs):
            lock=source_lock['files'][Path(path).name];require(entries[path]['source_sha256']==lock['sha256'] and entries[path]['source_bytes']==lock['bytes'],'Original input/reference lock differs')
        input_rows=[json.loads(line) for line in (root/inputs).read_text(encoding='utf-8').splitlines()]
        input_lookup={r['id']:r for r in input_rows};ref_lookup={r['id']:r for r in (json.loads(line) for line in (root/refs).read_text(encoding='utf-8').splitlines())}
        expected=cfg.get('ids') if cfg.get('ids') is not None else [r['id'] for r in input_rows[int(cfg.get('start',0)):int(cfg.get('start',0))+int(cfg.get('count',len(input_rows)))]]
        require(expected==m['expected_ids'] and len(expected)==artifact['expected_QA'],'Declared target cohort differs')
        for row in raw_lines[folder]:require(row['input_key_sha256']==cache.input_key(input_lookup[row['id']]),'Target frozen input differs from raw')
        if not artifact['scored_dir']:continue
        scored=artifact['scored_dir'];summary=load(root/scored/'summary.json')
        require(summary['expected_run_questions']==len(expected),'Scored expected coverage differs')
        require(summary['model_id']==cfg['model_id'] and summary['method']==cfg['method']['id'],'Scored model/method label differs')
        if declaration.get('dataset') is not None:require(summary['dataset']==declaration['dataset'],'Declared/scored dataset differs')
        for name,value in summary['integrity']['raw_files'].items():original_graph(folder,name,value)
        for name,value in summary['derived_file_hashes'].items():original_graph(scored,name,value)
        require(summary['integrity']['refs_sha256']==entries[refs]['source_sha256'] and summary['integrity']['inputs_sha256']==entries[inputs]['source_sha256'],'Scoring input/reference identity differs')
        if m.get('schema')=='baseline_matrix.derived_prediction_view.v1':
            proof=summary['integrity']['derived_source_provenance'];require(proof['inference_run'] is False and proof['origin_counts']==m['QA_origin_counts'],'Scoring provenance label differs')
        if recompute_scores:
            means=[];clusters=defaultdict(list);group_rows=defaultdict(list);failures=0
            with (root/folder/'records.jsonl').open(encoding='utf-8') as raw,(root/scored/'scored_records.jsonl').open(encoding='utf-8') as scores:
                for number,expected_id in enumerate(expected,1):
                    row=json.loads(raw.readline());value=json.loads(scores.readline());identity=raw_lines[folder][number-1]
                    require(row['id']==value['id']==expected_id and value['raw_status']==row['status'],'Scored/raw ID/status differs')
                    require(value['raw_jsonl_line_number']==number and value['raw_jsonl_exact_line_sha256']==identity['source_line_sha256'],'Scored original line proof differs')
                    if row['status']!='PASS':require(value['score'] is None,'Fake failure score');failures+=1;continue
                    for field in ('raw_prediction','output_token_ids','original_image_sha256','actual_visual_token_count','stored_visual_KV_bytes'):require(row.get(field)==value.get(field),'Scored/source field differs')
                    score=score_question(row['dataset'],row['raw_prediction'],ref_lookup[row['id']])
                    require(abs(score['score']-value['score'])<1e-12 and score['metric_id']==value['metric_id'] and score['reference_texts']==value['reference_texts'],'Score/reference does not recompute')
                    require(score['group']==value.get('group'),'Scored ChartQA group differs from frozen references')
                    means.append(score['score']);clusters[value['source_cluster_id']].append(score['score'])
                    if score['group'] in ('human','aug'):group_rows[score['group']].append((value['source_cluster_id'],score['score']))
                require(not raw.readline() and not scores.readline(),'Unexpected scored/raw rows')
            if failures:require(summary['primary_mean'] is None,'Failure retained a false full-cohort mean')
            else:
                require(abs(sum(means)/len(means)-summary['primary_mean'])<1e-12,'Mean does not recompute')
                ordered=sorted(clusters);sums=np.asarray([sum(clusters[k]) for k in ordered]);sizes=np.asarray([len(clusters[k]) for k in ordered])
                if len(ordered)>=2:
                    b=summary['bootstrap'];rng=np.random.default_rng(b['seed']);draws=rng.integers(0,len(ordered),size=(b['repetitions'],len(ordered)))
                    interval=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
                    require(np.allclose(interval,summary['primary_ci95'],rtol=0,atol=1e-12),'Cluster CI differs')
                for group,values in group_rows.items():
                    require(group in summary['groups'],'Missing ChartQA subgroup')
                    data=summary['groups'][group];require(data['expected_questions']==len(values),'ChartQA subgroup count differs')
                    require(abs(sum(v for _,v in values)/len(values)-data['primary']['mean'])<1e-12,'ChartQA subgroup mean differs')
                    cluster=defaultdict(list)
                    for key,value in values:cluster[key].append(value)
                    ordered=sorted(cluster)
                    if len(ordered)>=2:
                        sums=np.asarray([sum(cluster[k]) for k in ordered]);sizes=np.asarray([len(cluster[k]) for k in ordered])
                        b=summary['bootstrap'];rng=np.random.default_rng(b['seed']);draws=rng.integers(0,len(ordered),size=(b['repetitions'],len(ordered)))
                        interval=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
                        require(np.allclose(interval,data['primary']['ci95'],rtol=0,atol=1e-12),'ChartQA subgroup CI differs')
        passed=summary['status']=='PASS_SCORED_COMPLETE' and c['status']=='PASS_RAW_COMPLETE'
        group_number=artifact_group[artifact['config']]
        if passed:complete_cells[group_number]+=1;complete_questions[group_number]+=len(expected)
        scoring.append({'cell':artifact['cell'],'questions':len(expected),'mean':summary['primary_mean'],'passed':passed,
                        'phase_label':groups[group_number]['phase']['phase_label'],
                        'kind':artifact['kind'],'origin_counts':m.get('QA_origin_counts',{'fresh':len(expected)}),
                        'metrics_recomputed':bool(recompute_scores)})
    verified_groups=[];all_complete=True
    for group_number,group in enumerate(groups):
        phase=group['phase']
        require(phase['verified_scored_cells']==complete_cells[group_number] and phase['verified_scored_questions']==complete_questions[group_number],
                'Published independent phase progress differs from closed artifacts')
        is_complete=complete_cells[group_number]==phase['expected_cells'] and complete_questions[group_number]==phase['expected_questions'] and not phase['missing']
        require((phase['status']=='PASS_DECLARED_MATRIX_COMPLETE')==is_complete,'False independent phase completion')
        all_complete=all_complete and is_complete
        verified_groups.append({'phase_label':phase['phase_label'],'cells':complete_cells[group_number],
                                'view_questions':complete_questions[group_number],'primary':group_number==0})
    eligible=all_complete and not index.get('fixture_only',False)
    require(index['publication_eligible']==eligible,'Incomplete preview/fixture publication eligibility differs')
    phase=index['phase']
    return {'status':'PASS_PUBLIC_EXPORT_VERIFIED_V3','public_files':len(entries),'public_payload_bytes':total,
        'independent_phases_verified':verified_groups,'phase_group_totals_are_separate':True,
        'supporting_evidence_groups_verified':len(index.get('supporting_evidence_groups',[])),
        'jsonl_line_identities':line_count,'raw_complete_runs_verified':len(visited),
        'original_donors':sum(n['kind']=='original_inference_run' for n in index['source_closure']),
        'derived_views':sum(n['kind']=='derived_prediction_view' for n in index['source_closure']),
        'source_closure_verified':True,'score_recomputations':scoring,'archive_verified':bool(archive),
        'phase':phase,'publication_eligible':eligible,'fixture_only':bool(index.get('fixture_only',False)),
        'completion_scope':'declared_matrix_only',
        'whole_external_library_completion_assessed':False,'github_upload_verified':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path);parser.add_argument('--recompute-scores',action='store_true');parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();result=verify(args.root,args.archive,args.recompute_scores)
    if args.receipt:
        with args.receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
