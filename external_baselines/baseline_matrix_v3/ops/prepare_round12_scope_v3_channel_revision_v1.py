"""Prepare a new unsealed R12 scope with one archive-only QA0 channel fix.

No existing scope/source/runtime is changed. No GPU, rescoring, seal, export,
Git or authentication. The full recorded diagnostic remains in the archive.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE=Path('${EVIDENCE_ROOT}/results/round12')
FULL_SHA='d99baa4311131db5988709eb20269c6248bd79ede6560f668c76dac4a36ecc35'
FULL_PUBLIC='evidence/QA0/bm_R12_value_offset_diag_gpu1_v1/VALUE_OFFSET_DIAGNOSTIC_v1.json'
LIMIT=16*1024**2


def require(value,message):
    if not value:raise ValueError('R12_CHANNEL_REVISION: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def stats(values):
    require(values and all(math.isfinite(v) for v in values),'invalid recorded QA0 scalar')
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
            'min':min(values),'max':max(values)}


def compact(report):
    scalars=('status','actual_CUDA','GT_or_answers_loaded','training_performed','candidate_or_f0_modified',
        'task_answer_events','generated_answer_tokens','native_reference_retained_for_QA0',
        'production_memory_or_speed_proof','task_quality_gain_claimed','model_parameter_count',
        'native_prefix_model_forwards','suffix_TF_model_forwards','completed_model_forwards',
        'GPU','torch_version','elapsed_seconds','frozen_variant_sha256')
    result={'schema':'trajkv.round12.value_offset_diagnostic_compact.v1',
        'derivation':'Recorded scalar summaries only; no model/scorer execution or task-quality recomputation',
        'full_original_source_sha256':FULL_SHA,'full_public_path':FULL_PUBLIC,
        'full_archive_member':'external_baselines/baseline_matrix_v3/'+FULL_PUBLIC,
        'full_evidence_delivery':'ARCHIVE_ONLY; fixed release content_ref is supplied by PUBLIC_CONTENT_REFS.json',
        'owner_trace_preservation':'Every original trace and scalar remains in the full archive member; none removed from that member',
        'new_GPU_calls':0,'new_task_answers':0,'new_quality_scoring_calls':0,
        **{k:report[k] for k in scalars if k in report},
        'model_identity':{k:report['model_identity'][k] for k in ('status','revision','source_lock_sha256')
                          if k in report['model_identity']},'sources':[]}
    branch_total=0
    for source in report['sources']:
        item={k:source[k] for k in ('id','dataset','original_image_sha256','status','role','visual_tokens',
              'diagnostic_peak_allocated_bytes_including_native_reference') if k in source}
        item['branches']={}
        for name,branch in source['branches'].items():
            branch_total+=1
            out={'first_logits':branch['first_logits'],
                 'native_reference':branch.get('native_reference',False)}
            for key in ('actual_packed_owner_bytes','native_reference_extra_memory_excluded_from_production_claim'):
                if key in branch:out[key]=branch[key]
            if 'selected_stages' in branch:
                out['selected_stage_count']=len(branch['selected_stages'])
            metrics=branch.get('value_layer_head_metrics',[])
            out['value_layer_head_count']=len(metrics)
            if metrics:
                scalar_fields=sorted({k for m in metrics for k,v in m.items()
                    if isinstance(v,(int,float)) and not isinstance(v,bool) and k not in ('layer','head','N')})
                out['unweighted_recorded_layer_head_scalar_statistics']={key:stats([float(m[key]) for m in metrics if key in m])
                    for key in scalar_fields}
                vector_fields=sorted({k for m in metrics for k,v in m.items()
                    if isinstance(v,list) and v and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v)})
                out['recorded_channel_array_statistics']={key:stats([float(v) for m in metrics for v in m.get(key,[])])
                    for key in vector_fields}
                out['statistics_scope']='Descriptive unweighted recorded QA0 scalar/array summaries, not task quality or production latency'
            item['branches'][name]=out
        result['sources'].append(item)
    result['source_count']=len(result['sources'])
    result['branch_count']=branch_total
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-scope',type=Path,default=BASE/'final_scope_v2/R12_SCOPE_UNSEALED.json')
    p.add_argument('--output',type=Path,default=BASE/'final_scope_v3')
    a=p.parse_args()
    require(a.source_scope.resolve().is_relative_to(BASE.resolve()),'source scope outside named data directory')
    require(a.output.resolve().is_relative_to(BASE.resolve()) and not a.output.exists(),'new data-disk output required')
    original=json.loads(a.source_scope.read_text(encoding='utf-8-sig'))
    require(original['stage_state']=='READY_FOR_ROOT_SEAL' and original.get('research_cutoff_complete') is True,
            'expected completed unsealed v2 source scope')
    scope=deepcopy(original)
    target=next(x for x in scope['files'] if x['public_path']==FULL_PUBLIC)
    require(target['sha256']==FULL_SHA and sha(target['source_path'])==FULL_SHA,'full original QA0 source changed')
    require(target.get('delivery_channel')=='BROWSER_AND_ARCHIVE','expected exactly the rejected original browser channel')
    original_target=deepcopy(target)
    target['delivery_channel']='ARCHIVE_ONLY'
    require('expected_in_git' not in target,'unexpected second delivery declaration; review rather than silently modifying it')
    a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads(Path(target['source_path']).read_text(encoding='utf-8-sig'))
    summary=compact(report)
    compact_path=a.output/'VALUE_OFFSET_DIAGNOSTIC_COMPACT_v1.json'
    write(compact_path,summary)
    note_path=a.output/'ROUND12_DELIVERY_CHANNEL_REVISION_v1.md'
    note=(
        '# Round12 delivery-channel revision\n\n'
        'The full VALUE_OFFSET_DIAGNOSTIC_v1.json exceeds the existing 16 MiB browser-file limit. '
        'Its original bytes, SHA and public member path remain unchanged and it is now explicitly archive-only. '
        'The entire diagnostic, including all owner traces, remains in the release ZIP; no evidence is deleted and the limit is not raised.\n\n'
        'VALUE_OFFSET_DIAGNOSTIC_COMPACT_v1.json is a browser-visible summary of the already recorded scalar counts, '
        'first-logit measurements and descriptive per-branch statistics. It binds the full original SHA '
        +FULL_SHA+'. No GPU execution, new answer, rescoring or scientific-result change occurs.\n\n'
        'Scope v2 and the failed first export remain preserved. This new scope is unsealed; Root must seal and export it into a new stage, '
        'then independently verify and publish it. Preparation is not upload.\n')
    with note_path.open('x',encoding='utf-8',newline='\n') as f:f.write(note)
    def add(path,public,role):
        require(public not in {x['public_path'] for x in scope['files']},'new public member collision')
        scope['files'].append({'source_path':str(path),'public_path':public,'sha256':sha(path),'role':role,
                               'delivery_channel':'BROWSER_AND_ARCHIVE'})
    add(compact_path,'evidence/QA0/bm_R12_value_offset_diag_gpu1_v1/VALUE_OFFSET_DIAGNOSTIC_COMPACT_v1.json','technical')
    add(note_path,'evidence/delivery/ROUND12_DELIVERY_CHANNEL_REVISION_v1.md','protocol')
    add(Path(__file__).resolve(),'ops/prepare_round12_scope_v3_channel_revision_v1.py','code')
    science_keys=('cells','views','comparisons','qa0','attempts','groups','expected_accounting','report_summary')
    require(all(scope.get(k)==original.get(k) for k in science_keys),'scientific scope changed')
    require(len(scope['files'])==len(original['files'])+3,'unexpected file change')
    old_inventory=deepcopy(scope['files'][:len(original['files'])])
    next(x for x in old_inventory if x['public_path']==FULL_PUBLIC)['delivery_channel']='BROWSER_AND_ARCHIVE'
    require(old_inventory==original['files'],'original source inventory changed beyond one channel')
    revision={'schema':'trajkv.round12.delivery_channel_revision.v1','status':'PREPARED_CHANNEL_ONLY_UNSEALED',
        'source_scope_sha256':sha(a.source_scope),'full_original_source_sha256':FULL_SHA,
        'changed_original_member':{'before':original_target,'after':target},
        'original_file_count':len(original['files']),'scientific_fields_unchanged':list(science_keys),
        'scientific_fields_canonical_sha256':canonical({k:original.get(k) for k in science_keys}),
        'new_GPU_calls':0,'new_task_answers':0,'new_scoring_calls':0,'original_source_bytes_modified':False,
        'other_existing_files_changed':False,'browser_limit_bytes':LIMIT,'browser_limit_raised':False,
        'full_evidence_deleted':False,'stage_sealed':False,'GitHub_mutation':False}
    revision_path=a.output/'DELIVERY_CHANNEL_REVISION_v1.json'
    write(revision_path,revision)
    add(revision_path,'evidence/delivery/DELIVERY_CHANNEL_REVISION_v1.json','protocol')
    browser=[x for x in scope['files'] if x.get('delivery_channel')=='BROWSER_AND_ARCHIVE'
             or (x.get('delivery_channel') is None and x.get('expected_in_git',True))]
    sizes=[{'public_path':x['public_path'],'source_bytes':Path(x['source_path']).stat().st_size,
            'original_source_sha256':x['sha256']} for x in browser]
    overs=[x for x in sizes if x['source_bytes']>LIMIT]
    require(not overs,'another browser source exceeds unchanged16MiB limit')
    audit={'schema':'trajkv.round12.browser_source_size_audit.v1','status':'PASS_ALL_SELECTED_BROWSER_SOURCES_WITHIN_16MIB',
        'browser_limit_bytes':LIMIT,'scope_browser_files_before_this_audit':len(browser),
        'source_files_over_limit':overs,'browser_sources_over_1MiB':[x for x in sizes if x['source_bytes']>1024**2],
        'largest_browser_source':max(sizes,key=lambda x:x['source_bytes']),
        'audited_source_sizes':sizes,'this_audit_source_checked_separately_after_write':True,
        'full_archive_only_QA0_source_bytes':Path(target['source_path']).stat().st_size,
        'full_archive_only_QA0_original_sha256':FULL_SHA,'raw_or_model_recomputation':False,
        'source_scope_sha256':sha(a.source_scope),'stage_sealed':False}
    audit_path=a.output/'BROWSER_SOURCE_SIZE_AUDIT_v1.json'
    write(audit_path,audit)
    require(audit_path.stat().st_size<=LIMIT,'size audit itself exceeds browser limit')
    add(audit_path,'evidence/delivery/BROWSER_SOURCE_SIZE_AUDIT_v1.json','technical')
    scope['stage_state']='READY_FOR_ROOT_SEAL';scope['root_seal']=None
    scope['delivery_revision']={'schema':revision['schema'],'source_scope_sha256':sha(a.source_scope),
        'reason':'One full QA0 trace exceeds existing browser limit; retain exact full member archive-only and add compact scalar view',
        'scientific_fields_canonical_sha256':revision['scientific_fields_canonical_sha256'],
        'new_stage_required':True,'stage_sealed':False}
    out=a.output/'R12_SCOPE_UNSEALED.json'
    write(out,scope)
    require(sha(target['source_path'])==FULL_SHA and sha(a.source_scope)==revision['source_scope_sha256'],
            'an original source changed during preparation')
    print(json.dumps({'status':'PREPARED_SCOPE_V3_CHANNEL_ONLY_NOT_SEALED','scope':str(out),
        'files':len(scope['files']),'QA0_compact_bytes':compact_path.stat().st_size,
        'QA0_source_count':summary['source_count'],'QA0_branch_count':summary['branch_count'],
        'browser_source_over_limit':len(overs),'largest_browser_source':audit['largest_browser_source'],
        'new_GPU_calls':0,'new_task_answers':0,'stage_sealed':False,'GitHub_mutation':False}))


if __name__=='__main__':main()
