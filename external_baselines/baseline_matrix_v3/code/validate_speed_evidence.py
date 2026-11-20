"""CPU independent token/context/owner/repeat audit of actual speed evidence.

No GT, GPU or ranking. CUDA-event stream intervals and CPU dispatch/arrival
metrics remain separate; host_quiet=false is preserved as a claim limitation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def audit(raw,output,expected_ledger=None):
    from prediction_cache import evaluate_semantics,normalize_behavior
    receipt=load(raw/'speed_retrieval_receipt_v1.json')
    require(receipt['status']=='PASS_REMOTE_SPEED_RAW_AND_SOURCE_SHA' and receipt['remote_hashes_computed_independently'],'Independent speed raw/source receipt required')
    for entry in receipt['files']:
        require(sha(raw/entry['file'])==entry['sha256'] and (raw/entry['file']).stat().st_size==entry['bytes'],'Fetched evidence differs: '+entry['file'])
    manifest,complete,plan,summary=(load(raw/name) for name in ('manifest.json','complete.json','plan.json','summary.json'))
    require(manifest['schema']=='baseline_matrix.speed_actual_run.v1' and complete['GPU_used'] is True,'Actual CUDA timing schema required')
    for name,key in [('manifest.json','manifest_sha256'),('records.jsonl','records_jsonl_sha256'),('summary.json','summary_sha256')]:
        require(sha(raw/name)==complete[key],'Raw speed closure mismatch: '+name)
    require(sha(raw/'plan.json')==manifest['plan_sha256'],'Plan SHA mismatch')
    require(sha(raw/'source_benchmark_speed.py')==plan['source_sha256'],'Executed benchmark source differs from plan')
    stage=load(ROOT/'configs/matrix/SPEED_stage_v2.json')
    require(plan['source_sha256']==stage['benchmark_source_sha256'],'Unreviewed benchmark source')
    inventory=load(raw/'source_inventory.json');launch=load(raw/'launch_receipt.json')
    require(launch['source_inventory_sha256']==sha(raw/'source_inventory.json'),'Source inventory launch mismatch')
    require(inventory['code/benchmark_speed.py']==sha(raw/'source_benchmark_speed.py'),'Benchmark snapshot mismatch')
    require(inventory['code/evaluate.py']==sha(raw/'source_evaluate.py'),'Core snapshot mismatch')
    for name,digest in manifest['source_hashes'].items():require(inventory[name.replace('\\','/')]==digest,'Manifest/source mismatch: '+name)
    require(sha(raw/'expected_predictions.json')==manifest['expected_predictions_sha256'],'Expected-model predictions SHA differs')
    expected=load(raw/'expected_predictions.json')
    require(expected['schema']=='baseline_matrix.expected_model_predictions.v1' and expected['references_read'] is False,'Model outputs, never GT, required')
    lookup={(row['method'],row['id']):row for row in expected['records']}
    require(len(lookup)==len(expected['records']),'Duplicate expected outputs')
    require(plan['references_read'] is False and manifest['references_read'] is False,'Timing used task refs')
    require(plan['job']['repeat']==3 and plan['job']['warmup_per_qa']==1,'Expected repeat3+warm1')
    require(plan['job']['verify_gpu_exclusive'] is True,'No source GPU-exclusivity guard')
    require(manifest['host_quiet_asserted']==plan['job']['host_quiet_asserted'],'Host-quiet scope changed')
    exclusive=manifest['initial_exclusive_GPU_check']
    require(all(pid==exclusive['own_PID'] for pid in exclusive['observed_compute_PIDs']),'GPU shared at initial check')
    require(str(exclusive['physical_GPU_index'])==str(manifest['runtime']['visible_GPU'])==str(launch['CUDA_VISIBLE_DEVICES']),'Physical GPU declarations differ')
    require(manifest['quality_context_validation']['status']=='PASS_QUALITY_RUNTIME_SOURCE_CHECKPOINT_CONTEXT','Producer quality context was not checked')
    model_files=sorted(({key:row[key] for key in ('file','bytes','sha256')} for row in manifest['model_identity']['files']),key=lambda row:row['file'])
    for cfg in plan['effective_configs']:
        method=cfg['method']['id'];ctx=expected['source_contracts_by_method'][method]
        require(normalize_behavior(cfg,plan['model_spec'])==ctx['behavior'],'Timing/quality behavior differs')
        require(ctx['model_files']==model_files and ctx['generation_native']==manifest['generation_native'],'Checkpoint/generation context differs')
        for expected_name,actual_name in [('torch_version','torch'),('transformers_version','transformers'),('cuda_version','cuda'),('hardware','hardware')]:
            require(ctx['runtime'][expected_name]==manifest['runtime'][actual_name],'Runtime differs: '+expected_name)
        require(ctx['runtime']['runtime_executable']==launch['runtime_python'],'Runtime executable differs')
        for name,digest in ctx['relevant_source_hashes'].items():
            actual=evaluate_semantics(raw/'source_evaluate.py',sha(raw/'source_evaluate.py')) if name=='code/evaluate.py' else inventory[name]
            require(actual==digest,'Quality/benchmark source differs: '+name)
    records=[json.loads(line) for line in (raw/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(records)==complete['recorded_trials']==complete['expected_trials']==len(plan['schedule']),'Trial coverage mismatch')
    require(dict(Counter(row['status'] for row in records))==complete['warmup_and_measure_status_counts'],'Trial counts differ')
    if expected_ledger:
        for key,actual in [('expected_trials',len(records)),('measured_trials',plan['measured_trials']),('warmup_trials',plan['warmup_trials'])]:
            require(expected_ledger[key]==actual,'Ledger count mismatch '+key)
        require(expected_ledger['host_quiet_asserted']==manifest['host_quiet_asserted'],'Ledger host claim differs')
    expected_keys=set();errors=[];contexts=defaultdict(set)
    for index,(row,item) in enumerate(zip(records,plan['schedule'])):
        for field in ('phase','repeat','id','method','config_index','method_order_within_QA'):
            require(row[field]==item[field],f'Interleaved schedule changed at {index}: {field}')
        key=(row['phase'],row['repeat'],row['id'],row['method'])
        require(key not in expected_keys,'Duplicate trial');expected_keys.add(key)
        require(row['reference_answers_read'] is False,'Reference flag in trial')
        if row['status']!='PASS':errors.append({'index':index,'status':row['status'],'reason':'producer trial failed'});continue
        target=lookup[(row['method'],row['id'])]
        required_forbid={'answers','labels','ground_truth','reference_answers','official_score','score'}
        require(not(set(target)&required_forbid),'Expected predictions contains task scoring/GT')
        valid=row['output_token_ids']==target['output_token_ids'] and row['output_tokens']==len(target['output_token_ids'])
        for field in ('original_image_sha256','input_ids_sha256','full_prompt_hash'):
            if field in target:valid &= row[field]==target[field]
        if target.get('stored_visual_KV_bytes') is not None:valid &= row['stored_visual_KV_bytes']==target['stored_visual_KV_bytes']
        valid &= row['quality_token_identity_status']=='EXACT_MATCH' and row['canonical_speed_eligible'] is True
        valid &= len(row['inter_token_ms'])==max(0,row['output_tokens']-1)
        valid &= sum(row['output_chunk_sizes'])==row['output_tokens'] and all(size==1 for size in row['output_chunk_sizes'])
        for key2 in ('wall_ms','ttft_ms','question_prefill_ms','encode_ms','peak_allocated_bytes','peak_reserved_bytes'):
            valid &= row.get(key2) is not None and math.isfinite(row[key2]) and row[key2]>=0
        require(row['owner_audit_in_timed_region'] is False,'Owner audit contaminated measured interval')
        require(row['owner_after_timing']['complete_visual_owner_bytes']==row['complete_visual_owner_bytes'],'Owner byte report differs')
        contexts[row['id']].add((row['original_image_sha256'],row['input_ids_sha256'],row['full_prompt_hash'],row['native_position_ids_sha256'],row['actual_visual_token_count']))
        if not valid:errors.append({'index':index,'status':row['status'],'reason':'independent token/context/owner/stream validation failed'})
    require(all(len(values)==1 for values in contexts.values()),'Methods used different per-QA image/prompt/native positions')
    methods=[];perQA=[]
    def describe(values):
        values=[value for value in values if value is not None]
        return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)} if values else {'n':0,'mean':None,'median':None,'min':None,'max':None}
    for method in plan['method_ids']:
        warm=[row for row in records if row['method']==method and row['phase']=='warmup']
        measured=[row for row in records if row['method']==method and row['phase']=='measure']
        coverage=Counter((row['phase'],row['repeat'],row['id']) for row in warm+measured)
        require(len(warm)==len(plan['expected_ids']) and len(measured)==3*len(plan['expected_ids']),'Per-method repeat coverage wrong')
        require(all(count==1 for count in coverage.values()),'Repeated/missing QA trial')
        good=not errors and all(row['status']=='PASS' for row in warm+measured)
        source_summary=summary['methods'][method]
        require(source_summary['expected_trials']==len(measured) and source_summary['recorded_trials']==len(measured),'Producer method count differs')
        item={'method':method,'measured_trials':len(measured),'warmup_trials':len(warm),
              'token_context_owner_repeat_validated':good,'status_counts':dict(Counter(row['status'] for row in measured)),
              'no_wall_only_ranking':True,'host_quiet_asserted':manifest['host_quiet_asserted']}
        passed=[row for row in measured if row['status']=='PASS']
        for key in ('wall_ms','ttft_ms','question_prefill_ms','question_prefill_host_dispatch_ms','output_tokens','encode_ms',
                    'generation_ms','generation_wall_ms','peak_allocated_bytes','peak_reserved_bytes','complete_visual_owner_bytes','actual_effective_visual_bits'):
            item[key]=describe([row.get(key) for row in passed])
        item['phase_GPU_stream_interval_ms']={phase:describe([row['phase_GPU_stream_interval_ms'][phase] for row in passed]) for phase in ('preprocess','prefill','encode','generation')}
        item['phase_host_dispatch_ms']={phase:describe([row['phase_host_dispatch_ms'][phase] for row in passed]) for phase in ('preprocess','prefill','encode','generation')}
        item['pooled_natural_ITL_ms']=describe([value for row in passed for value in row['inter_token_ms']])
        item['mean_request_natural_ITL_ms']=describe([row.get('mean_inter_token_ms') for row in passed])
        item['one_token_measured_outputs']=sum(row.get('output_tokens')==1 for row in passed)
        item['descriptive_successful_only']=not good
        methods.append(item)
        for identity in plan['expected_ids']:
            rows=[row for row in passed if row['id']==identity]
            if rows:perQA.append({'method':method,'id':identity,'repeats':len(rows),'output_tokens':rows[0]['output_tokens'],
                'TTFT_host_mean_ms':statistics.mean(row['ttft_ms'] for row in rows),
                'question_prefill_GPU_mean_ms':statistics.mean(row['question_prefill_ms'] for row in rows),
                'encode_GPU_stream_interval_mean_ms':statistics.mean(row['encode_ms'] for row in rows),
                'ITL_host_mean_ms':statistics.mean(value for row in rows for value in row['inter_token_ms']) if any(row['inter_token_ms'] for row in rows) else None,
                'complete_visual_owner_bytes':rows[0]['complete_visual_owner_bytes']})
    result={'schema':'baseline_matrix.independent_timing_audit.v1',
        'status':'PASS_TOKEN_CONTEXT_OWNER_REPEAT_TIMING_AUDIT' if not errors else 'CLOSED_TIMING_WITH_FAILED_OR_UNVERIFIED_TRIALS',
        'model':plan['model_spec']['model_id'],'dataset':records[0]['dataset'],'QA':len(plan['expected_ids']),
        'GPU_benchmark_actually_used':True,'CPU_validator_used_GPU':False,'quality_500_QA_added':0,
        'raw_trials':len(records),'warmup_trials':plan['warmup_trials'],'measured_trials':plan['measured_trials'],
        'host_quiet_asserted':manifest['host_quiet_asserted'],'exclusive_compute_GPU_check':exclusive,
        'claim_scope':'Token-validated single-GPU-compute-exclusive interleaved trials on shared host' if not manifest['host_quiet_asserted'] else 'Token-validated declared-host-quiet interleaved trials',
        'GPU_stream_intervals_are_not_pure_kernel_time':True,'do_not_rank_by_shorter_natural_answers':True,
        'no_cross_domain_pooling_or_global_speedup':True,'errors':errors,'methods':methods,'per_QA':perQA,
        'raw_hashes':{name:sha(raw/name) for name in ('manifest.json','records.jsonl','complete.json','summary.json')},
        'source_benchmark_sha256':sha(raw/'source_benchmark_speed.py'),'retrieval_receipt_sha256':sha(raw/'speed_retrieval_receipt_v1.json')}
    output.mkdir(parents=True,exist_ok=False)
    with (output/'audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    with (output/'per_QA.csv').open('x',encoding='utf-8',newline='') as stream:
        if perQA:
            writer=csv.DictWriter(stream,fieldnames=list(perQA[0]),lineterminator='\n');writer.writeheader();writer.writerows(perQA)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw_dir',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ledger-row',type=Path)
    args=parser.parse_args();result=audit(args.raw_dir,args.output,load(args.ledger_row) if args.ledger_row else None)
    print(json.dumps({key:result[key] for key in ('status','model','dataset','QA','raw_trials','host_quiet_asserted','quality_500_QA_added')}))


if __name__=='__main__':main()
