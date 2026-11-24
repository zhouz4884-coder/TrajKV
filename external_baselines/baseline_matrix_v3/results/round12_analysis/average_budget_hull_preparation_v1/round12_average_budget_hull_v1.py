"""Post-hoc endpoint randomization diagnostic, admitted only after BOTH500 close.

This is expected quality at a matched average visual-owner budget, not a new
GPU run, per-image-budget competitor, algorithm selector or preregistered test.
All inputs are existing CPU-readable artifacts; incomplete states are rejected
before opening reference packs, prediction records or output directories.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

DATA_ROOT = Path('${EVIDENCE_ROOT}')
DEFAULT_PACK = DATA_ROOT/'results/round12/fixed500_harvest_v1/reference_pack_v1/REFERENCE_PACK_v1.json'
DOMAINS = ('infovqa','docvqa')
METRIC_CONTRACT = 'matrix_v3_metrics_v1'
IDENTITY = ('dataset','dataset_revision','split','source_id','source_cluster_id','question_id',
    'question_sha256','original_image_sha256','model_id','model_commit','dtype','preprocessing',
    'full_prompt_hash','input_ids_sha256','native_position_ids_sha256','compression_boundary',
    'actual_visual_token_count','question_suffix_sha256')


def require(condition, message):
    if not condition:
        raise ValueError('R12_AVERAGE_BUDGET_HULL: '+message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open('rb') as f:
        for number,line in enumerate(f,1):
            if line.strip():
                yield number,line,json.loads(line)


def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def admit_complete_state(state):
    require(state.get('schema')=='trajkv.round12.fixed500_collection_state.v1'
        and state.get('status')=='COMPLETE_FIXED500_1000_NEW_ANSWERS'
        and state.get('result_status')=='COMPLETE_MATCHED1000',
        'requires the final collector state: status COMPLETE_FIXED500_1000_NEW_ANSWERS and result_status COMPLETE_MATCHED1000')
    require(state.get('completed_shard_questions')=={'infovqa':500,'docvqa':500}
        and state.get('new_raw_recorded_tasks')==1000
        and state.get('new_physical_terminal_answers_present')==1000
        and state.get('planned_new_physical_answers')==1000
        and state.get('running_rows_provisional')==0
        and state.get('errors')==[] and not state.get('report_error'),
        'both domains must have all500 successful closed/scored answers and zero collection errors/running rows')
    require(state.get('independent_holdout_claimed') is False,'wrong benchmark role')
    require(state['identity']['planned_new_tasks']==1000
        and state['identity']['exact500_per_domain_without_reselection'] is True
        and state['identity']['selected_method_name']=='TrajKV_R12_R_G8matched',
        'not the frozen exact500 comparison')
    for domain in DOMAINS:
        cells = [c for c in state['cells'] if c['domain']==domain]
        require(sum(c['expected_QA'] for c in cells)==500,'cell plans do not cover500 '+domain)
        require(all(c.get('state')=='COMPLETE'
            and c.get('collection_status')=='COMPLETE_RETRIEVED_OFFICIAL_SCORED'
            and c.get('score_status')=='PASS_SCORED_COMPLETE'
            and c.get('raw_counts')=={'PASS':c['expected_QA']} for c in cells),
            'one or more shards is not complete successful official score '+domain)
    require(all(c['domain'] in DOMAINS for c in state['cells']),'unexpected extra domain')


def load_run(raw_dir, score_dir, expected, *, raw_hashes=None, summary_sha=None):
    raw_dir,score_dir = Path(raw_dir),Path(score_dir)
    summary_path = score_dir/'summary.json'
    if summary_sha:
        require(sha(summary_path)==summary_sha,'historical score-summary SHA changed')
    score = read(summary_path)
    require(score.get('schema')=='baseline_matrix.offline_score.v1'
        and score.get('status')=='PASS_SCORED_COMPLETE'
        and score.get('metric_contract_version')==METRIC_CONTRACT
        and score.get('expected_run_questions')==expected
        and score['integrity']['status']=='PASS_ALL_SHA_AND_EXPECTED_ID_CHECKS'
        and score['integrity']['fixture_only'] is False,'unverified or incompatible score source')
    hashes = score['integrity']['raw_files']
    for name in ('manifest.json','records.jsonl','complete.json'):
        require(name in hashes and sha(raw_dir/name)==hashes[name],'raw triplet changed: '+name)
        if raw_hashes is not None:
            require(hashes[name]==raw_hashes[name],'state/pack raw anchor differs: '+name)
    require(sha(score_dir/'scored_records.jsonl')==score['derived_file_hashes']['scored_records.jsonl'],
        'official scored rows changed')
    scored = {r['id']:r for _,_,r in jsonl(score_dir/'scored_records.jsonl')}
    require(len(scored)==expected,'scored duplicate/coverage')
    result = {}
    for line_number,line,raw in jsonl(raw_dir/'records.jsonl'):
        ident = raw['id']; require(ident in scored and ident not in result,'raw ID coverage/duplicate')
        s = scored[ident]
        require(raw['status']=='PASS' and s['score_status']=='SCORED_PASS','non-PASS row')
        require(s['raw_jsonl_line_number']==line_number and s['raw_jsonl_exact_line_sha256']==hashlib.sha256(line).hexdigest(),
            'score not bound to exact raw line')
        require(s['raw_prediction']==raw['raw_prediction'] and s['output_token_ids']==raw['output_token_ids'],
            'scored prediction differs')
        byte_count,denominator = raw['complete_visual_owner_bytes'],raw['full_kv_byte_denominator']
        require(isinstance(byte_count,int) and not isinstance(byte_count,bool) and byte_count>0
            and isinstance(denominator,int) and denominator>0,'invalid complete physical bytes')
        bits = float(raw['actual_effective_visual_bits'])
        require(math.isfinite(bits) and math.isclose(bits,16*byte_count/denominator,rel_tol=1e-9),
            'reported bits do not equal actual complete owner / original full denominator')
        value = float(s['score']);require(math.isfinite(value) and 0<=value<=1,'invalid task score')
        result[ident]={'id':ident,'source_cluster_id':raw['source_cluster_id'],'score':value,
            'metric_id':s['metric_id'],'bytes':byte_count,'full_bytes':denominator,'bits':bits,
            'identity':{key:raw.get(key) for key in IDENTITY}}
    require(len(result)==expected,'raw count differs from expected complete source')
    return result,{'raw_dir':str(raw_dir),'scored_dir':str(score_dir),'raw_sha256':hashes,
        'score_summary_sha256':sha(summary_path),'scored_rows_sha256':sha(score_dir/'scored_records.jsonl'),
        'input_sha256':score['integrity']['inputs_sha256']}


def load_endpoint(reference):
    directory=Path(reference['scored_dir']);summary=read(directory/'summary.json')
    rows,proof=load_run(summary['integrity']['raw_dir'],directory,500,
        raw_hashes=reference['raw_hashes'],summary_sha=reference['summary_sha256'])
    require(proof['scored_rows_sha256']==reference['scored_rows_sha256']
        and proof['input_sha256']==reference['input_sha256'],'endpoint pack score/input binding changed')
    require(set(rows)==set(reference['rows']),'endpoint pack IDs differ')
    for ident,item in rows.items():
        packed=reference['rows'][ident]
        require(item['score']==packed['score'] and item['source_cluster_id']==packed['source_cluster_id']
            and item['metric_id']==packed['metric_id']
            and item['bytes']==packed['complete_visual_owner_bytes']
            and item['full_bytes']==packed['full_kv_byte_denominator'],'endpoint pack values differ from actual source')
    return rows,proof


def totals_point(totals):
    # Order: B_R, B_low, B_high, score_R, score_low, score_high, row count.
    br,bl,bh,sr,sl,sh,n=totals
    require(n>0 and bh>bl,'endpoint total bytes do not define an increasing segment')
    lam=(br-bl)/(bh-bl)
    if not 0<=lam<=1:
        return {'lambda':lam,'inside_convex_segment':False,'delta_pp':None,'mixed_score':None}
    expected=((1-lam)*sl+lam*sh)/n
    return {'lambda':lam,'inside_convex_segment':True,'delta_pp':100*(sr/n-expected),
        'mixed_score':expected,'R_score':sr/n,'low_score':sl/n,'high_score':sh/n,
        'R_total_visual_owner_bytes':br,'low_total_visual_owner_bytes':bl,
        'high_total_visual_owner_bytes':bh,'mixture_expected_total_visual_owner_bytes':(1-lam)*bl+lam*bh,
        'n':int(n)}


def percentile(values,q):
    values=sorted(values);at=q*(len(values)-1);lo=int(at);hi=min(lo+1,len(values)-1)
    return values[lo]+(at-lo)*(values[hi]-values[lo])


def diagnose_domain(domain,r,low,high,repetitions,seed):
    require(len(r)==500 and set(r)==set(low)==set(high),'not the same exact500 IDs '+domain)
    groups=defaultdict(lambda:[0]*7);row_table=[]
    for ident in sorted(r):
        a,l,h=r[ident],low[ident],high[ident]
        require(a['identity']==l['identity']==h['identity'],'model/input/position identity differs '+ident)
        require(a['full_bytes']==l['full_bytes']==h['full_bytes']
            and a['metric_id']==l['metric_id']==h['metric_id'],'denominator or metric differs '+ident)
        cluster=a['source_cluster_id'];values=[a['bytes'],l['bytes'],h['bytes'],a['score'],l['score'],h['score'],1]
        groups[cluster]=[v+x for v,x in zip(groups[cluster],values)]
        row_table.append({'dataset':domain,'id':ident,'source_cluster_id':cluster,
            'R_bytes':a['bytes'],'low_bytes':l['bytes'],'high_bytes':h['bytes'],
            'R_score':a['score'],'low_score':l['score'],'high_score':h['score'],
            'original_full_visual_KV_bytes':a['full_bytes']})
    totals=[sum(v[i] for v in groups.values()) for i in range(7)]
    point=totals_point(totals);aggregate=list(groups.values());rng=random.Random(seed)
    draws=[];outside=0
    if point['inside_convex_segment']:
        for _ in range(repetitions):
            sampled=[aggregate[rng.randrange(len(aggregate))] for _ in aggregate]
            boot=totals_point([sum(v[i] for v in sampled) for i in range(7)])
            if not boot['inside_convex_segment']:
                outside+=1
            else:
                draws.append((boot['lambda'],boot['delta_pp']))
    full=sum(v['full_bytes'] for v in r.values())
    ci_status='AVAILABLE_DESCRIPTIVE_PERCENTILE' if point['inside_convex_segment'] and outside==0 else 'UNAVAILABLE_NONCONVEX_RESAMPLES_OR_POINT'
    return {'dataset':domain,**point,'source_clusters':len(groups),
        'R_element_weighted_complete_bits':16*totals[0]/full,
        'low_element_weighted_complete_bits':16*totals[1]/full,
        'high_element_weighted_complete_bits':16*totals[2]/full,
        'R_mean_complete_bits':statistics.mean(v['bits'] for v in r.values()),
        'R_max_complete_bits':max(v['bits'] for v in r.values()),
        'mixture_possible_per_image_max_bits':max(v['bits'] for x in (low,high) for v in x.values()),
        'bootstrap':{'repetitions':repetitions,'seed':seed,'unit':'source_cluster_id',
            'lambda_recomputed_each_resample':True,'CI_status':ci_status,
            'draws_performed':len(draws)+outside,
            'outside_convex_segment_draws':outside,'draws_clipped_or_silently_discarded':False,
            'delta_ci95_pp':[percentile([v[1] for v in draws],q) for q in (.025,.975)] if ci_status.startswith('AVAILABLE') else None,
            'lambda_ci95':[percentile([v[0] for v in draws],q) for q in (.025,.975)] if ci_status.startswith('AVAILABLE') else None,
            'posthoc_descriptive_not_preregistered_primary':True,'multiplicity_adjusted':False}},row_table


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--state',required=True,type=Path)
    p.add_argument('--reference-pack',type=Path,default=DEFAULT_PACK)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261124)
    a=p.parse_args(argv)
    state=read(a.state)
    admit_complete_state(state) # Deliberately before ANY reference/raw read or output creation.
    require(a.bootstrap>=2000,'at least2000 descriptive source bootstrap draws required')
    require(a.output.resolve().is_relative_to(DATA_ROOT.resolve()) and not a.output.exists(),
        'new output directory must stay on the named E data disk')
    require(sha(a.reference_pack)==state['reference_pack_sha256'],'complete state/reference-pack SHA differs')
    pack=read(a.reference_pack);receipt=read(a.reference_pack.parent/'REFERENCE_PACK_RECEIPT_v1.json')
    require(pack['schema']=='trajkv.round12.fixed500_reference_pack.v1'
        and receipt['status']=='PASS_MATCHED_FIXED500_REFERENCE_PACK'
        and receipt['pack_sha256']==sha(a.reference_pack)
        and pack['metadata_manifest_sha256']==state['identity']['input_metadata_manifest_sha256']
        and receipt['metadata_manifest_sha256']==pack['metadata_manifest_sha256'],
        'reference pack is not the complete matched frozen benchmark')
    source_report=read(Path(state['report_dir'])/'summary.json')
    require(source_report['status']=='COMPLETE_MATCHED1000'
        and source_report['reference_pack_sha256']==sha(a.reference_pack)
        and source_report['input_metadata_manifest_sha256']==state['identity']['input_metadata_manifest_sha256'],
        'complete report/reference mismatch')
    candidate={d:{} for d in DOMAINS};proofs=[]
    for cell in state['cells']:
        anchors={n:v['sha256'] for n,v in cell['core_hashes'].items()}
        values,proof=load_run(cell['local_raw'],cell['scored_dir'],cell['expected_QA'],raw_hashes=anchors)
        require(proof['input_sha256']==pack['references'][cell['domain']]['R7low']['input_sha256']
            ==pack['references'][cell['domain']]['R7high']['input_sha256'],'candidate/endpoint input file SHA differs')
        target=candidate[cell['domain']]
        require(not set(values)&set(target),'duplicate candidate ID across closed shards')
        target.update(values);proofs.append({'kind':'existing_R_candidate','cell':cell['name'],**proof})
    results=[];all_rows=[]
    for domain in DOMAINS:
        refs={}
        for name in ('R7low','R7high'):
            refs[name],proof=load_endpoint(pack['references'][domain][name])
            proofs.append({'kind':'reused_historical_endpoint','dataset':domain,'method':name,**proof})
        result,table=diagnose_domain(domain,candidate[domain],refs['R7low'],refs['R7high'],a.bootstrap,a.seed)
        declared=[t for t in source_report['tables'] if t['dataset']==domain
            and t['method']==state['identity']['selected_method_name']]
        require(len(declared)==1 and declared[0]['complete_fixed500_coverage'] is True
            and math.isclose(declared[0]['primary_full500_mean'],statistics.mean(v['score'] for v in candidate[domain].values()),abs_tol=1e-12),
            'candidate recomputed full500 mean differs from completed report')
        results.append(result);all_rows.extend(table)
    report={'schema':'trajkv.round12.fixed500_average_budget_endpoint_hull.v1',
        'status':'PASS_POSTHOC_COMPLETE500_AVERAGE_BUDGET_DIAGNOSTIC',
        'new_inference':0,'new_GPU_calls':0,'existing_R_answers_analyzed':1000,'historical_endpoint_rows_analyzed':2000,
        'complete_state_sha256':sha(a.state),'reference_pack_sha256':sha(a.reference_pack),
        'source_report_sha256':sha(Path(state['report_dir'])/'summary.json'),'helper_sha256':sha(__file__),
        'metric_contract':METRIC_CONTRACT,'results':results,'source_proofs':proofs,
        'interpretation':[
            'Old low/high outputs are reused to compute expected quality of an independent Bernoulli choice before encoding; no randomized deployment was run.',
            'Lambda uses only actual complete visual-owner bytes, not answers. The expected average storage equals R total visual-owner bytes.',
            'This matches an average visual-owner budget, not a per-image cap, all-request peak, text-KV budget, transfer volume, encode time or latency.',
            'The mixture can use the old high per-image precision; R has a different per-image resource distribution. This is not a strict per-image dominance claim.',
            'Intervals are post-hoc descriptive source-cluster bootstrap with lambda recomputed each draw; not a new preregistered primary or independent confirmation.',
            'No candidate, threshold, data selection, method or running experiment is changed by this diagnostic.']}
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'AVERAGE_BUDGET_HULL_v1.json',report)
    with (a.output/'SOURCE_COST_SCORE_ROWS_v1.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_rows)
    lines=['# 完整500问：旧低/高独立随机混合的平均预算诊断','',
        '此表是复用已完成输出的期望质量分析，新增推理0；不是一次实际随机运行，也不是逐图相同上限的部署比较。',
        '', '| 域 | λ（选择高档概率） | R分数% | 混合期望分数% | R−混合pp | 描述性95% CI |', '|---|---:|---:|---:|---:|---|']
    for value in results:
        ci=value['bootstrap']['delta_ci95_pp']
        lines.append('| '+value['dataset']+' | '+f"{value['lambda']:.6f}"+' | '
            +(f"{100*value['R_score']:.6f} | {100*value['mixed_score']:.6f} | {value['delta_pp']:+.6f}" if value['inside_convex_segment'] else '不可定义 | 不可定义 | 不可定义')
            +' | '+(f'[{ci[0]:+.6f}, {ci[1]:+.6f}]' if ci is not None else '不可定义')+' |')
    lines.extend(['','λ由真实ΣB计算，每次source bootstrap重新估计；不裁剪区间外λ。仅匹配平均完整视觉owner费用；混合可能使用约1.8bit高档，R逐图上限不同。',
        '该分析为事后辅助，不是新增预注册主判据，不据此重新挑方法。完整成本、bootstrap信息与源SHA见JSON；逐源成本/分数见CSV。',''])
    with (a.output/'AVERAGE_BUDGET_HULL_v1.zh.md').open('x',encoding='utf-8',newline='\n') as f:
        f.write('\n'.join(lines))
    print(json.dumps({'status':report['status'],'results':results,'output':str(a.output)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
