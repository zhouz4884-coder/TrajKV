"""Read-only private validation and dependency closure for declared phases."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path('${EVIDENCE_ROOT}').resolve()
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code'))


def require(value,message):
    if not value:raise ValueError(message)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()
def inside(path,root):
    try:Path(path).resolve().relative_to(Path(root).resolve());return True
    except ValueError:return False
def resolve(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    require(inside(path,ROOT) or inside(path,EVIDENCE),'Evidence escapes approved project/E roots')
    return path.resolve()


def validate_scored(scored,validated):
    """Verify pre-existing scores against original source rows and locked refs."""
    from metrics import score_question
    summary=load(scored/'summary.json')
    require(summary['integrity']['raw_files']==validated['raw_hashes'],'Scored summary refers to other raw bytes')
    for name,digest in summary['derived_file_hashes'].items():require(sha(scored/name)==digest,'Derived score hash mismatch')
    require(summary['expected_run_questions']==len(validated['rows']),'Scored coverage differs')
    require(summary['integrity']['refs_sha256']==sha(validated['refs_path']),'Scoring reference lock differs')
    require(summary['integrity']['inputs_sha256']==sha(validated['input_path']),'Scoring input lock differs')
    means=[];clusters=defaultdict(list);group_rows=defaultdict(list);count=0;failures=0
    with (scored/'scored_records.jsonl').open(encoding='utf-8') as stream:
        for count,(raw,line_sha) in enumerate(zip(validated['rows'],validated['line_hashes']),1):
            text=stream.readline();require(text,'Missing scored row');row=json.loads(text)
            require(row['id']==raw['id'] and row['raw_status']==raw['status'],'Scored/raw identity differs')
            require(row['raw_jsonl_exact_line_sha256']==line_sha and row['raw_jsonl_line_number']==count,'Scored original row binding differs')
            if raw['status']!='PASS':
                require(row['score'] is None,'Failure assigned a fake score');failures+=1;continue
            for field in ('raw_prediction','output_token_ids','original_image_sha256','actual_visual_token_count','stored_visual_KV_bytes'):
                require(row.get(field)==raw.get(field),'Scored/raw field differs: '+field)
            result=score_question(raw['dataset'],raw['raw_prediction'],validated['refs'][raw['id']])
            require(abs(result['score']-row['score'])<1e-12 and result['metric_id']==row['metric_id'],'Score does not recompute')
            require(result['reference_texts']==row['reference_texts'],'Frozen reference texts differ')
            require(result['group']==row.get('group'),'Scored group differs from frozen reference')
            means.append(result['score']);clusters[row['source_cluster_id']].append(result['score'])
            if result['group'] in ('human','aug'):group_rows[result['group']].append((row['source_cluster_id'],result['score']))
        require(not stream.readline(),'Unexpected scored rows')
    require(count==summary['expected_run_questions'],'Unexpected scored count')
    if failures:
        require(summary['primary_mean'] is None,'Incomplete results reported as full quality')
    else:
        require(summary['status']=='PASS_SCORED_COMPLETE','Expected complete-score status')
        require(abs(sum(means)/len(means)-summary['primary_mean'])<1e-12,'Primary mean differs')
        ordered=sorted(clusters);sums=np.asarray([sum(clusters[k]) for k in ordered]);sizes=np.asarray([len(clusters[k]) for k in ordered])
        if len(ordered)>=2:
            bootstrap=summary['bootstrap'];rng=np.random.default_rng(bootstrap['seed'])
            samples=rng.integers(0,len(ordered),size=(bootstrap['repetitions'],len(ordered)))
            interval=np.quantile(sums[samples].sum(1)/sizes[samples].sum(1),[.025,.975])
            require(np.allclose(interval,summary['primary_ci95'],rtol=0,atol=1e-12),'Cluster CI differs')
        for group,values in group_rows.items():
            require(group in summary['groups'],'Missing required ChartQA subgroup')
            reported=summary['groups'][group];require(reported['expected_questions']==len(values),'Subgroup coverage differs')
            require(abs(sum(v for _,v in values)/len(values)-reported['primary']['mean'])<1e-12,'Subgroup mean differs')
            cluster=defaultdict(list)
            for key,value in values:cluster[key].append(value)
            ordered=sorted(cluster)
            if len(ordered)>=2:
                sums=np.asarray([sum(cluster[k]) for k in ordered]);sizes=np.asarray([len(cluster[k]) for k in ordered])
                rng=np.random.default_rng(summary['bootstrap']['seed']);draws=rng.integers(0,len(ordered),size=(summary['bootstrap']['repetitions'],len(ordered)))
                interval=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
                require(np.allclose(interval,reported['primary']['ci95'],rtol=0,atol=1e-12),'Subgroup cluster CI differs')
    return summary,failures


def contract_from_arguments(args):
    if args.contract:
        path=resolve(args.contract);contract=load(path)
        require(contract['schema']=='baseline_matrix.public_phase_contract.v2','Unknown phase contract')
        return contract,path
    require(args.matrix and args.phase_label and args.view_root and args.score_root,
            'Provide --contract or explicit --matrix --phase-label --view-root --score-root')
    matrix=resolve(args.matrix);plan=load(matrix)
    return {'schema':'baseline_matrix.public_phase_contract.v2','phase_label':args.phase_label,
        'family':str(plan.get('phase',args.phase_label)).lower(),'matrix':str(matrix),
        'coverage_scope':args.coverage_scope or plan.get('scope','Declared matrix only; no whole-library claim'),
        'view_root':str(resolve(args.view_root)),'score_root':str(resolve(args.score_root)),
        'completion_policy':'all_expected_QA_pass','completion_scope':'declared_matrix_only',
        'whole_library_completion_assessed':False},None


def validate_phase(contract):
    from score import validate_raw
    matrix_path=resolve(contract['matrix']);matrix=load(matrix_path)
    require(contract.get('completion_scope','declared_matrix_only')=='declared_matrix_only','Tool cannot certify a whole research library')
    require(contract.get('completion_policy','all_expected_QA_pass')=='all_expected_QA_pass','Use explicit per-QA pass completion policy')
    require(matrix.get('rows') and len({row['config'] for row in matrix['rows']})==len(matrix['rows']),'Empty/duplicate declared matrix')
    artifacts=[];observed=[];missing=[]
    for row in matrix['rows']:
        cfg_path=resolve(row['config']);cfg=load(cfg_path);stem=cfg_path.stem
        if row.get('model') is not None:require(cfg['model_id']==row['model'],'Matrix/model ID mismatch: '+stem)
        raw=resolve(row['view']) if row.get('view') else resolve(Path(contract['view_root'])/stem)
        scored=resolve(row['scored']) if row.get('scored') else resolve(Path(contract['score_root'])/stem)
        if not (raw/'complete.json').is_file():missing.append({'cell':stem,'missing':'closed_raw_or_view'});continue
        manifest=load(raw/'manifest.json')
        require(manifest['config']==cfg,'Declared target config differs from result view: '+stem)
        require(manifest['expected_QA']==int(row['count']),'Declared per-cell QA count differs: '+stem)
        input_name=Path(cfg['inputs'].replace('\\','/')).name
        refs=ROOT/'configs/datasets/v3'/input_name.replace('_inputs.jsonl','_refs.jsonl')
        validated=validate_raw(raw,refs)
        if row.get('dataset') is not None:require(validated['dataset']==row['dataset'],'Matrix/dataset mismatch: '+stem)
        require(len(validated['expected'])==int(row['count']),'Target coverage differs')
        kind='derived_prediction_view' if manifest.get('inference_run') is False else 'original_inference_run'
        artifact={'cell':stem,'config':cfg_path,'raw':raw,'scored':None,'kind':kind,'expected_QA':int(row['count']),
                  'raw_status':validated['complete']['status'],'origin_counts':manifest.get('QA_origin_counts',{'fresh':int(row['count'])})}
        artifacts.append(artifact)
        if not (scored/'summary.json').is_file():missing.append({'cell':stem,'missing':'complete_scored_summary'});continue
        summary,failures=validate_scored(scored,validated);artifact['scored']=scored
        require(summary['model_id']==cfg['model_id'] and summary['method']==cfg['method']['id'] and
                summary['dataset']==validated['dataset'],'Scoring model/method/dataset label differs')
        if failures or validated['complete']['status']!='PASS_RAW_COMPLETE':
            missing.append({'cell':stem,'missing':'all_expected_QA_pass','failed_QA':failures});continue
        observed.append({'cell':stem,'questions':int(row['count']),'kind':kind,'raw_status':validated['complete']['status'],
                         'score_status':summary['status'],'origin_counts':artifact['origin_counts']})
    phase={'status':'PASS_DECLARED_MATRIX_COMPLETE' if not missing else 'INCOMPLETE_DECLARED_MATRIX_PREVIEW',
        'phase_label':contract['phase_label'],'family':contract.get('family',str(matrix.get('phase','')).lower()),
        'coverage_scope':contract['coverage_scope'],'completion_scope':'declared_matrix_only',
        'whole_external_library_completion_assessed':False,'declared_matrix_source_sha256':sha(matrix_path),
        'expected_cells':len(matrix['rows']),'verified_scored_cells':len(observed),
        'expected_questions':sum(int(row['count']) for row in matrix['rows']),
        'verified_scored_questions':sum(row['questions'] for row in observed),
        'closed_artifact_cells':len(artifacts),'missing':missing,'observed':observed}
    return phase,artifacts,matrix_path


def source_closure(artifacts):
    """Collect every original donor plus proofs; no phase-name assumption."""
    directories={};active=set()
    def walk(path):
        path=resolve(path);key=str(path)
        if key in active:raise ValueError('Cyclic prediction provenance')
        if key in directories:return
        active.add(key);manifest=load(path/'manifest.json');complete=load(path/'complete.json')
        required=['manifest.json','records.jsonl','complete.json']
        if manifest.get('schema')=='baseline_matrix.derived_prediction_view.v1':
            required.append('provenance.jsonl')
            require(manifest['inference_run'] is False and complete['derived_prediction_view'] is True,'Derived view mislabeled')
            for descriptor in manifest['source_runs']:
                source=resolve(descriptor['raw_dir']);walk(source)
                for name,value in {**descriptor['raw_hashes'],**descriptor['source_proof_hashes']}.items():
                    require(sha(source/name)==value,'Source descriptor original SHA mismatch')
                require(sha(source/'retrieval_receipt.json')==descriptor['retrieval_receipt_sha256'],'Original retrieval proof changed')
                require(sha(source/'source_retrieval_receipt_v2.json')==descriptor['source_retrieval_receipt_sha256'],'Original source proof changed')
        else:
            required+=['retrieval_receipt.json','source_inventory.json','launch_receipt.json','source_evaluate.py','source_retrieval_receipt_v2.json']
            from prediction_cache import SourceRun
            SourceRun(path)  # Original-only artifacts need the same producer proof as donors.
        require(all((path/name).is_file() for name in required),'Incomplete original source evidence set')
        directories[key]={'path':path,'kind':'derived_prediction_view' if 'provenance.jsonl' in required else 'original_inference_run',
                          'files':[path/name for name in required]}
        active.remove(key)
    for artifact in artifacts:walk(artifact['raw'])
    return list(directories.values())
