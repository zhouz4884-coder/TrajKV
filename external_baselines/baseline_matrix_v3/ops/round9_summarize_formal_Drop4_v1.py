"""Same fixed500 Drop4 readiness/comparison; no partial candidate main scores."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];E=Path('${EVIDENCE_ROOT}')
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'code')]
import analyze_r7_fixed500_v2 as frozen

PLAN=E/'results/round9/formal_Drop4_draft_v1/PLAN.json'
REF=E/'results/r7_matrix500_v2/competitive_final_v1/source_index.json'
REF_SHA='8f74910273224f5bfbc57945cbc859a803325fad170fa1ede9e03149beccdd27'
COMMON=E/'results/r7_alignment/M1_four_domains_full500_baseline_reference_v1/INPUT_AND_COMMON_CONTRACTS500.json'
LABELS={'F':'FP16 same500','R7_K1':'R7 low','R7_K2':'R7 high','K2_g32R32':'KIVI source g32/R32','C2_bias32':'Calib common-FP16 numeric-v2 C2'}
DOMAINS=('textvqa','infovqa','docvqa','chartqa')

def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return frozen.sha(p)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in row.items()})

def load_refs():
    require(sha(REF)==REF_SHA,'Frozen same500 references changed')
    data=read(REF);require(data['status']=='PASS_CLOSED8_ALL_CURRENT_METHODS_MATCHED_FIXED500','Wrong reference phase')
    refs={};sources=[]
    for cell in data['cells']:
        if cell['method_key']not in LABELS:continue
        require(cell['status']=='COMPLETE_FIXED500'and cell['expected_QA']==500,'Only complete500 anchors')
        summary=frozen.score_metadata(cell['scored'],cell['score_summary_sha256'])
        require(summary['model_id']=='M1'and summary['dataset']==cell['dataset']and summary['primary_mean']==cell['primary_mean'],'Reference source identity differs')
        refs[cell['dataset'],cell['method_key']]=cell
        sources.append({'dataset':cell['dataset'],'point':cell['method_key'],'scored':cell['scored'],'score_summary_sha256':cell['score_summary_sha256'],
                        'raw_file_sha256':cell['raw_file_sha256'],'original_source_evidence_reused_not_new_GPU':True})
    require(len(refs)==20 and set(refs)=={(d,k)for d in DOMAINS for k in LABELS},'Exact5 references x4domains')
    require(sha(COMMON)==frozen.REF_PINS['INPUT_AND_COMMON_CONTRACTS500.json'],'Frozen full500 input/context fingerprints changed')
    return refs,read(COMMON),sources

def candidate(row,common):
    raw=Path(row['raw']);scored=Path(row['scored']);required=['manifest.json','records.jsonl','complete.json','retrieval_receipt.json','source_retrieval_receipt_round9_v2.json']
    missing=[x for x in required if not(raw/x).is_file()]
    if not(scored/'summary.json').is_file():missing.append('scored_summary')
    empty={'dataset':row['dataset'],'point':'BRPE_Drop4','status':'DRAFT_UNLAUNCHED_OR_NOT_CLOSED500','QA':500,'primary_mean':None,'CI95':None,'resources':None,'raw':str(raw),'scored':str(scored),'missing':missing}
    if missing:return empty
    summary=read(scored/'summary.json');complete=read(raw/'complete.json')
    if summary['status']!='PASS_SCORED_COMPLETE'or complete['status']!='PASS_RAW_COMPLETE':
        require(summary['primary_mean']is None,'Failed formal source primary must stay null');empty.update(status='CLOSED_WITH_FAILURES_NO_PRIMARY',missing=['all500_QA_PASS']);return empty
    import round9_public_contract_v3 as strict
    item=strict.validate_inference(raw);cfg=item['config'];m=item['manifest'];ds=row['dataset']
    require(cfg==read(ROOT/row['config'])and m['config_sha256']==row['config_sha256']and sha(raw/'source_job.json')==row['job_sha256'],'Actual formal source differs from prepared candidate identity')
    require(cfg['candidate']['mode']=='BRPE_Drop4'and cfg['candidate']['profile_sha256']=='7ec305a76cebf5ca9e358842e11e68fe6a77e6af9f2d83c39af723ff17eb5c22'and cfg['cohort_kind']=='formal500'and cfg['count']==cfg['input_count']==500 and cfg['start']==0,'Exact formal500 Drop4 scope')
    spec=next(x for x in common['rows']if x['dataset']==ds)
    require(m['inputs_sha256']==row['inputs_sha256']==spec['input_sha256']and m['expected_ids']==spec['selected_ids'],'Original same500 cohort/order required')
    context=common['common_contracts_by_dataset'][ds]['context']
    require(frozen.native_context(m)=={k:context[k]for k in frozen.native_context(m)}and m['model_spec']==context['model_spec'],'Actual checkpoint/processor/generation/runtime differs')
    summary=frozen.score_metadata(scored,sha(scored/'summary.json'));require(summary['expected_run_questions']==500 and summary['metric_ids']==[cfg['evaluation_contract']['primary_metric_id']],'Frozen primary score differs')
    scored_rows={}
    with(scored/'scored_records.jsonl').open(encoding='utf-8')as f:
        for line in f:
            v=json.loads(line);require(v['id']not in scored_rows,'Duplicate score row');scored_rows[v['id']]=v
    identities={};values=[];bits=[]
    with(raw/'records.jsonl').open('rb')as f:
        for number,line in enumerate(f,1):
            v=json.loads(line);score=scored_rows[v['id']]
            require(score['raw_jsonl_line_number']==number and score['raw_jsonl_exact_line_sha256']==hashlib.sha256(line).hexdigest(),'Candidate source/score exact line differs')
            actual=strict.verify_brpe_row(v,m['model_spec'],'BRPE_Drop4');values.append(actual);bits.append(v['actual_effective_visual_bits']);identities[v['id']]=frozen.input_identity(v)
    require(len(values)==500 and frozen.canonical(identities)==common['parent_full500_alignment'][ds]['per_ID_identity_fingerprint'],'Full500 image/prompt/tokens/visual positions/boundary differ')
    return {'dataset':ds,'point':'BRPE_Drop4','status':'CLOSED_COMPLETE500_SOURCE_SCORE_AND_PHYSICAL','QA':500,'primary_mean':summary['primary_mean'],'CI95':summary['primary_ci95'],
            'resources':{'mean_complete_visual_bytes':sum(values)/500,'mean_complete_bits':sum(bits)/500,'max_complete_bits':max(bits),
                         'strict_all_QA_1.3':all(160*v<=13*(28*2*4*r['actual_visual_token_count']*128*2)for v,r in zip(values,item['rows'])),
                         'strict_all_QA_1.8':all(80*v<=9*(28*2*4*r['actual_visual_token_count']*128*2)for v,r in zip(values,item['rows']))},
            'raw':str(raw),'scored':str(scored),'score_summary_sha256':sha(scored/'summary.json'),'missing':[],'SourceRun_admission':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=PLAN);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(a.output.resolve().is_relative_to(E.resolve())and not a.output.exists(),'Use new E result directory')
    plan=read(a.plan);require(plan['expected_cells']==4 and plan['expected_model_answers']==2000 and plan['not400']is True,'Not original4x500 draft')
    refs,common,sources=load_refs();candidates=[candidate(row,common)for row in plan['rows']];closed=sum(x['primary_mean']is not None for x in candidates)
    output=[];comparisons={}
    for ds in DOMAINS:
        for key,label in LABELS.items():
            cell=refs[ds,key];r=cell['resources'];output.append({'dataset':ds,'point':key,'method':label,'status':'COMPLETE_REFERENCE_FIXED500','QA':500,
                'quality_percent':100*cell['primary_mean'],'candidate_minus_same500_FP16_pp':100*(cell['primary_mean']-refs[ds,'F']['primary_mean']),
                'CI95_percent':[100*x for x in cell['CI95']],'mean_complete_MiB':r['mean_complete_visual_bytes']/2**20,'mean_complete_bits':r['mean_complete_bits'],'max_complete_bits':r['max_complete_bits'],
                'strict_all500_1p3':r['strict_all_QA_1.3'],'strict_all500_1p8':r['strict_all_QA_1.8']})
        point=next(x for x in candidates if x['dataset']==ds);r=point['resources']or{}
        output.append({'dataset':ds,'point':'BRPE_Drop4','method':'BRPE Drop4/profile7ec','status':point['status'],'QA':500,'quality_percent':None if point['primary_mean']is None else 100*point['primary_mean'],
            'candidate_minus_same500_FP16_pp':None if point['primary_mean']is None else 100*(point['primary_mean']-refs[ds,'F']['primary_mean']),
            'CI95_percent':None if point['CI95']is None else[100*x for x in point['CI95']],'mean_complete_MiB':None if not r else r['mean_complete_visual_bytes']/2**20,
            'mean_complete_bits':r.get('mean_complete_bits'),'max_complete_bits':r.get('max_complete_bits'),'strict_all500_1p3':r.get('strict_all_QA_1.3'),'strict_all500_1p8':r.get('strict_all_QA_1.8')})
        if point['primary_mean']is not None:
            from compare_runs import compare
            for key in LABELS:comparisons[ds+'_minus_'+key]=compare(Path(refs[ds,key]['scored']),Path(point['scored']))
    a.output.mkdir(parents=True)
    summary={'status':'PASS_ALL4_FORMAL500_CANDIDATE_CLOSED'if closed==4 else'PENDING_OR_PARTIAL_FORMAL4_NO_WHOLE_CANDIDATE_CLAIM','candidate_closed_cells':closed,'expected_cells':4,'expected_candidate_QA':2000,
             'candidates':candidates,'rows':output,'reference_sources':sources,'plan_sha256':sha(a.plan),'reference_index_sha256':REF_SHA,'whole_official_split_or_independent_confirmation_claim':False,
             'only_same500_metrics_no_cross_domain_average':True,'unadjusted_pairCI_not_equivalence_or_winner':True,'new_GPU_or_remote_or_Git_calls':0}
    write(a.output/'summary.json',summary);csvwrite(a.output/'same500_quality_complete_bytes.csv',output);write(a.output/'paired_comparisons.json',comparisons)
    write(a.output/'RECEIPT.json',{'status':summary['status'],'candidate_closed':closed,'source_sha256':sha(__file__),'files':[{'file':x.name,'sha256':sha(x)}for x in a.output.iterdir()if x.is_file()],'old_files_modified':False,'GPU_remote_Git':0})
    print(json.dumps({'status':summary['status'],'candidate_closed':closed,'reference_points':20,'candidate_pending_main_scores':4-closed}))

if __name__=='__main__':main()
