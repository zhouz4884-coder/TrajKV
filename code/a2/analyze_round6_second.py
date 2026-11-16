"""Strict, CPU-only first-question scoring for real second-backbone packed runs.

References enter only here, after generation. This analyzer intentionally does
not reuse multi-turn eligibility assumptions or compare first-question scores
with the primary backbone's all-turn scores. Static identity transfer is not a
second-backbone task-calibrated method or a blind confirmation.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/analysis'))
from analyze_external import score_docvqa, PROTOCOLS
from analyze_cycle import cluster_bootstrap

L, H, D, NV = 32, 32, 128, 576


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local_path(value):
    normalized = str(value).replace('\\', '/')
    if '/TrajKV/' in normalized:
        return ROOT / normalized.split('/TrajKV/', 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-12)


def csv_write(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def static_order(control):
    if control == 'spread':
        order = [int(f'{i:010b}'[::-1], 2) for i in range(L*H)]
    elif control == 'random1212':
        order = list(range(L*H)); random.Random(1212).shuffle(order)
    else:
        raise ValueError('Unknown second-backbone static allocation control')
    return [(axis, index//H, index%H) for index in order for axis in ('k', 'v')]


def packed_plan(variant, factor):
    """Independent integer reconstruction of this specific named width/pair plan."""
    require(variant.get('v_group_size') == 'all', 'Second-backbone whole-image V metadata required')
    require(float(variant.get('budget_bits', 1.8)) <= 1.8, 'Requested codec cap exceeds 1.8bit')
    require(variant.get('depth_policy') in ('incumbent', 'v_pair'), 'Only frozen width/pair policies supported')
    require(not any(variant.get(key) is not None for key in
        ('selected_stages', 'res_stages_limit', 'res_units_limit', 'residual_units_limit', 'interaction_policy_path')),
        'Unexpected second-backbone allocation override')
    units = [tuple(x) for x in variant['selected_units']]
    require(units == static_order(variant['axis_control']), 'Static 32x32 ranking differs from its declared source-independent rule')
    full = L*H*NV*D*4
    base = L*H*NV*50 + L*H*D*2 + 4*L*H*D + 256*4*2 + 8*NV
    factors = int(factor['parameter_count'])*4
    lookup, ranking, policy, stage_bytes = L*2*H*2*4, len(units)*2, 5, 18*NV
    fixed = base + factors + lookup + ranking + policy
    allowed = math.floor(float(variant.get('budget_bits', 1.8))*full/16)
    affordable = (allowed-fixed)//stage_bytes
    old_fixed = base + factors + L*2*H*4 + 2*(2*L*H)
    incumbent_cap = min(len(units), max(0, (allowed-old_fixed)//stage_bytes))
    fixed_k = [u for u in units[:incumbent_cap] if u[0] == 'k']
    require(len(fixed_k) <= affordable, 'Insufficient budget for the declared fixed K subset')
    values = [u for u in units if u[0] == 'v']
    vbudget = min(affordable-len(fixed_k), len(values))
    stages = [(*u, 0) for u in fixed_k]
    if variant['depth_policy'] == 'incumbent':
        stages.extend((*u, 0) for u in values[:vbudget])
    else:
        for u in values[:vbudget//2]:
            stages.extend([(*u, 0), (*u, 1)])
        if vbudget%2:
            stages.append((*values[vbudget//2], 0))
    ids = [(int(axis == 'v')*L+layer)*H+head for axis, layer, head in units]
    packed_rank = struct.pack('<'+'H'*len(ids), *ids)
    descriptor = struct.pack('<Bf', 0, 0.) if variant['depth_policy']=='incumbent' else struct.pack('<Bf', 1, 1.)
    return dict(full=full, fixed=fixed, allowed=allowed, stages=stages, factors=factors,
        lookup=lookup, ranking=ranking, policy=policy, stage_bytes=stage_bytes,
        complete=fixed+len(stages)*stage_bytes, rank_sha256=hashlib.sha256(packed_rank).hexdigest(),
        policy_sha256=hashlib.sha256(descriptor).hexdigest())


def audit_factor(variant, external):
    path = local_path(variant['metadata_factor_path'])
    book_path = local_path(variant['direction_book_path'])
    require(path.is_file() and book_path.is_file(), 'Missing local second-model factor/book evidence')
    require(external.get(variant['metadata_factor_path']) == sha(path), 'Second-model identity factor hash mismatch')
    require(external.get(variant['direction_book_path']) == sha(book_path), 'Global encoder book hash mismatch')
    factor, book = read(path), read(book_path)
    require(factor.get('geometry') == dict(layers=L, kv_heads=H, head_dim=D), 'Factors are not new 32x32 geometry')
    require(factor.get('calibration_mode') == 'identity_no_task_data' and factor.get('cases') == 0
        and factor.get('images') == 0 and factor.get('labels_used') is False, 'Unexpected second-model task calibration')
    require(factor.get('value_group_size') == 'all' and factor.get('key_gain_mode') == 'none', 'Unexpected identity factor semantics')
    require(factor.get('parameter_count') == 3*L*H and factor.get('encoder_book_sha256') == sha(book_path), 'Identity factor/book/count mismatch')
    for name, expected in (('key_head_factors', 1.), ('value_head_factors', 1.), ('value_center_offsets', 0.)):
        array = factor[name]
        require(len(array)==L and all(len(row)==H and all(float(x)==expected for x in row) for row in array), 'Nonidentity or wrong-shape second-model factor array')
    require(len(book['centers'])==256 and all(len(row)==4 for row in book['centers']), 'Expected global 256x4 encoder book')
    return factor


def audit_record(record, variant, sample, plan=None):
    sid, method = sample['id'], variant['method']
    require(record['variant_config']==variant and record['method']==method, 'Record differs from manifest codec configuration')
    require(record['image_id']==sample['image_id'] and record.get('split')==sample.get('split'), 'Record/sample source identity mismatch')
    require(record.get('source_document_id')==sample.get('source_document_id'), 'Record/source-document mismatch')
    require(record.get('answers_visible_to_runner') is False and record.get('task_accuracy_eligible') is True, 'Record is not answer-free eligible inference')
    require(record['visual_tokens']==NV and record.get('padded_visual_tokens')==0, 'Real Nv576 without padding required')
    require(len(record['turns'])==1 and record['turns'][0]['turn']==1, 'Exactly first-question turn1 is required')
    turn = record['turns'][0]
    require(isinstance(turn['answer'],str) and isinstance(turn['token_ids'],list) and turn['token_ids'], 'Missing real free-answer tokens')
    require(record.get('protocol')=='first_question_image_boundary_free_running' if method!='native_hf'
        else record.get('protocol')=='native_hf_first_question_free_running', 'Unexpected answer-generation protocol')
    prompt = record['prompt_metadata']
    require(prompt['visual_tokens']==NV and prompt['padded_visual_tokens']==0 and prompt['turns']==1, 'Prompt does not match first-question Nv576 protocol')
    owner = record['owned_initial']
    full = L*H*NV*D*4
    require(int(owner['visual_fp16_denominator'])==full, 'Wrong actual L32/H32/N576 visual denominator')
    encoder = int(record['encoder_calibration_parameters_bytes'])
    actual = sum(int(owner.get(key,0)) for key in ('visual_base','visual_residual','selector_metadata','visual_position'))+encoder
    bits = 16*actual/full
    require(close(bits,record['actual_visual_bits_including_encoder']), 'Actual complete owner bit rate mismatch')
    require(int(owner.get('visual_position',0))==8*NV, 'Position owner missing or padded')
    require(record.get('read_ledger_exact') is True, 'Actual bank read ledger failed')
    if method=='round6_depth':
        require(record.get('question_state_scope')=='recomputed_under_codec'
            and record.get('initial_logits_shared_fp16') is False and record.get('first_question_suffix_replay_tokens',0)>0,
            'Question was not fully replayed after visual compression')
        require(owner.get('full_visual_retained') is False and int(owner.get('diagnostic_fp16_reference',0))==0
            and record.get('depth_residual_source_fp16_retained') is False and record.get('visual_fp16_materialization') is False,
            'Packed candidate retains source visual FP16 or materializes it')
        require(record.get('depth_residual_full_visual_support') is True and record.get('depth_residual_budget_verified') is True,
            'Packed codec support/budget receipt failed')
        stages = [tuple(x) for x in record['depth_residual_selected_stages']]
        require(stages==plan['stages'] and len(set(stages))==len(stages), 'Actual stages differ from exact named policy')
        require(actual==plan['complete']==int(record['complete_visual_owner_bytes'])==int(record['depth_residual_complete_bytes']), 'Actual packed complete owner differs from exact shape plan')
        require(actual<=plan['allowed'] and bits<=1.8+1e-12, 'Packed complete owner exceeds 1.8bit')
        require(int(owner['visual_residual'])==len(stages)*plan['stage_bytes'], 'Residual pool was padded or incompletely charged')
        require(int(owner['selector_metadata'])==plan['lookup'] and encoder==plan['factors']+plan['ranking']+plan['policy'], 'Lookup/ranking/policy/factor owner mismatch')
        for key, expected in [('depth_residual_fixed_complete_bytes',plan['fixed']),('depth_residual_allowed_bytes',plan['allowed']),
            ('depth_residual_stage_lookup_bytes',plan['lookup']),('depth_residual_encoder_ranking_bytes',plan['ranking']),
            ('depth_residual_encoder_policy_bytes',plan['policy']),('depth_residual_stage_bytes',plan['stage_bytes']),
            ('depth_residual_stages',len(stages)),('depth_residual_K_stages',sum(x[0]=='k' for x in stages)),
            ('depth_residual_V_stages',sum(x[0]=='v' for x in stages))]:
            require(int(record[key])==expected, 'Mismatch in '+key)
        require(close(bits,record['depth_residual_actual_budget_bits']), 'Depth metadata bit rate mismatch')
        require(record['depth_residual_encoder_ranking_sha256']==plan['rank_sha256']
            and record['depth_residual_encoder_policy_sha256']==plan['policy_sha256'], 'Actual compact ranking/policy constants differ from config')
    elif method=='full':
        require(int(owner['visual_base'])==full and int(owner['visual_residual'])==0 and encoder==0,
            'Full baseline is not the expected FP16 visual owner')
    else:
        raise ValueError('This strict packed analyzer expects Full plus width/pair only; native identity has its dedicated smoke analyzer')
    return dict(session_id=sid,variant=variant['name'],method=method,complete_bytes=actual,actual_bits=bits,
        padded_visual_tokens=0,visual_tokens=NV,stages=len(plan['stages']) if plan else 0,
        K_stages=sum(x[0]=='k' for x in plan['stages']) if plan else 0,
        V_stages=sum(x[0]=='v' for x in plan['stages']) if plan else 0,
        source_fp16_retained=method=='full',passed=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs','--run',nargs='+',required=True)
    p.add_argument('--configs','--config',nargs='+',type=Path,required=True)
    p.add_argument('--samples',type=Path,default=ROOT/'configs/a2/docvqa_next_dev64.json')
    p.add_argument('--references',type=Path,default=ROOT/'configs/data/docvqa_next_references.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap-repeats',type=int,default=8000)
    args=p.parse_args()
    require(len(args.runs)==len(args.configs), 'Supply one exact original config per run')
    supplied = {x['id']:x for x in read(args.samples)}
    require(len(supplied)==len(read(args.samples)), 'Duplicate sample IDs')
    references={}
    for line in args.references.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        if row['id'] in supplied:
            require(row['id'] not in references, 'Duplicate reference session')
            first=[x for x in row['references'] if int(x['turn'])==1]
            require(len(first)==1, 'Missing/duplicate turn1 reference')
            require(row.get('source_document_id')==supplied[row['id']].get('source_document_id'), 'Reference source identity mismatch')
            references[row['id']]=first[0]['answer']
    require(set(references)==set(supplied), 'Supplied sample set has missing turn1 references')
    variants,records,audits,receipts,selected_union={},[],[],[],set()
    seen=set()
    for run,config_path in zip(args.runs,args.configs):
        folder=Path(run)
        if not folder.is_absolute():
            folder=ROOT/folder if (ROOT/folder).is_dir() else ROOT/'results'/folder
        manifest,complete=read(folder/'manifest.json'),read(folder/'complete.json')
        config=read(config_path)
        require(manifest['schema']=='trajkv.round6.second-backbone-packed.v1' and complete['status']=='COMPLETE', 'Not a complete second packed run')
        require(sha(config_path)==manifest['config_sha256'] and config==manifest['config'], 'Exact source config SHA/object mismatch')
        geometry=manifest['geometry']
        require((geometry['layers'],geometry['kv_heads'],geometry['query_heads'],geometry['head_dim'])==(L,H,H,D), 'Wrong second-backbone actual projection geometry')
        require(manifest.get('answers_visible_to_runner') is False and manifest.get('model_weights_frozen') is True, 'Ineligible inference manifest')
        source_path=local_path(config['arguments']['samples'])
        require(source_path.is_file() and sha(source_path)==manifest['samples_sha256'], 'Cannot verify actual inference sample SHA')
        source=read(source_path)
        expected_rows=source[int(config.get('start',0)):int(config.get('start',0))+int(config['count'])]
        ids=[x['id'] for x in expected_rows]
        require(ids==manifest['selected_ids'] and len(ids)==int(config['count']) and len(set(ids))==len(ids), 'Selected ID/order/count does not match exact input selection')
        require(all(sid in supplied and supplied[sid]==row for sid,row in zip(ids,expected_rows)), 'Analysis samples differ from exact inference sources')
        require(int(config['arguments'].get('turns',1))==1 and int(config['expected_visual_tokens'])==NV, 'Wrong inference turn/token protocol')
        for source_name,digest in manifest['sources'].items():
            source_file=ROOT/source_name
            require(source_file.is_file() and sha(source_file)==digest, 'Runtime source SHA mismatch: '+source_name)
        run_variants={v['name']:v for v in config['variants']}
        require(len(run_variants)==len(config['variants']) and sum(v['method']=='full' for v in run_variants.values())==1,
            'Require unique variants and exactly one Full baseline')
        plans={}
        for name,variant in run_variants.items():
            require(name not in variants or variants[name]==variant, 'Variant configuration drift across shards')
            variants[name]=variant
            if variant['method']=='round6_depth':
                factor=audit_factor(variant,manifest['external_codec_inputs'])
                plans[name]=packed_plan(variant,factor)
            else: require(variant['method']=='full', 'Dedicated analyzer supports Full/packed width/pair')
        for control in {v.get('axis_control') for v in run_variants.values() if v['method']=='round6_depth'}:
            require({v['depth_policy'] for v in run_variants.values() if v.get('axis_control')==control}=={'incumbent','v_pair'}, 'Incomplete static width/pair configuration')
        run_seen=set();run_records=[]
        for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            record=json.loads(line);sid,name=record['session_id'],record['variant'];key=(sid,name)
            require(sid in ids and name in run_variants and key not in seen, 'Unexpected/duplicate second-model record')
            require(record.get('model')==config['arguments']['model'], 'Model identity differs from exact run config')
            require(len(record['turns'][0]['token_ids'])<=int(config['arguments']['max_tokens']), 'Generation exceeds configured token budget')
            audit=audit_record(record,run_variants[name],supplied[sid],plans.get(name))
            audits.append(audit);records.append(record);run_records.append(record);seen.add(key);run_seen.add(key)
        require(run_seen=={(sid,name) for sid in ids for name in run_variants}, 'Incomplete first-question x variant coverage')
        require(len(run_seen)==int(complete['records'])==int(complete['expected']), 'Completion receipt record count mismatch')
        require(complete.get('padded_visual_tokens')==0 and complete.get('compressed_variants_run') is True, 'Completion lacks real no-padding compressed run')
        checks={x['id']:x for x in complete['paired_owner_checks']}
        require(set(checks)==set(ids) and len(checks)==len(complete['paired_owner_checks']), 'Incomplete paired-owner receipt coverage')
        for sid in ids:
            rows=[x for x in run_records if x['session_id']==sid]
            require(len({x['prompt_metadata']['prompt_sha256'] for x in rows})==1, 'Methods did not receive the same first-question prompt')
            packed=[x for x in audits if x['session_id']==sid and x['method']=='round6_depth']
            signature={(x['complete_bytes'],x['stages'],x['K_stages'],x['V_stages']) for x in packed}
            require(len(signature)==1 and checks[sid]['matched_bytes_and_axis_stage_costs'] is True, 'Packed width/pair/static controls are not same-byte same-axis-cost')
            expected_signature=next(iter(signature))
            require(all(tuple(x)==expected_signature for x in checks[sid]['groups'].values()), 'Completion paired-owner vector differs from actual records')
        selected_union.update(ids)
        receipts.append(dict(run=str(folder),config_path=str(config_path),config_sha256=sha(config_path),
            manifest_sha256=sha(folder/'manifest.json'),records_sha256=sha(folder/'records.jsonl'),complete_sha256=sha(folder/'complete.json'),
            variants={name:canonical_sha(v) for name,v in run_variants.items()},selected_ids=ids,source_sha256=manifest['sources'],
            external_codec_sha256=manifest['external_codec_inputs']))
    require(seen=={(sid,name) for sid in selected_union for name in variants}, 'Shards do not cover one complete same-input variant set')
    by=defaultdict(dict);scored=[];costs=defaultdict(list)
    for record in records:
        sid,name=record['session_id'],record['variant'];sample=supplied[sid]
        cluster=str(sample.get('source_document_id') or sample['image_id'])
        score=score_docvqa(record['turns'][0]['answer'],references[sid])
        key=(cluster,sid,1)
        by[name][key]=float(score['score']);costs[name].append(record)
        scored.append(dict(session_id=sid,image_id=sample['image_id'],source_document_id=cluster,variant=name,turn=1,
            score=score['score'],exact=score['exact'],anls_strict_threshold_aux=score['anls_strict_threshold_aux'],
            prediction=record['turns'][0]['answer'],references=json.dumps(references[sid],ensure_ascii=False)))
    table=[]
    for name,variant in variants.items():
        rows=[x for x in scored if x['variant']==name]
        estimate=cluster_bootstrap(((x['source_document_id'],x['score']) for x in rows),repeats=args.bootstrap_repeats,seed=1212)
        owner=[x for x in audits if x['variant']==name]
        table.append(dict(variant=name,method=variant['method'],axis_control=variant.get('axis_control'),depth_policy=variant.get('depth_policy'),
            questions=len(rows),source_documents=len({x['source_document_id'] for x in rows}),
            anls=estimate['mean'],anls_percent=100*estimate['mean'],ci_low=estimate['ci_lower'],ci_high=estimate['ci_upper'],
            exact=float(np.mean([x['exact'] for x in rows])),actual_bits=owner[0]['actual_bits'],
            complete_visual_bytes=owner[0]['complete_bytes'],stages=owner[0]['stages'],K_stages=owner[0]['K_stages'],V_stages=owner[0]['V_stages'],
            eligible_packed_le_1_8=variant['method']=='round6_depth' and all(x['actual_bits']<=1.8+1e-12 and not x['source_fp16_retained'] for x in owner),
            session_seconds_median=float(np.median([x['session_ms']/1000 for x in costs[name]])),
            peak_allocated_max=max(x['peak_allocated'] for x in costs[name])))
    full_names=[name for name,v in variants.items() if v['method']=='full']
    require(len(full_names)==1, 'Global Full baseline name/config drift')
    full=full_names[0];pairs={(name,full) for name,v in variants.items() if v['method']=='round6_depth'}
    controls={v['axis_control'] for v in variants.values() if v['method']=='round6_depth'}
    for control in controls:
        names={v['depth_policy']:name for name,v in variants.items() if v.get('axis_control')==control}
        pairs.add((names['v_pair'],names['incumbent']))
    if {'spread','random1212'}<=controls:
        for policy in ('incumbent','v_pair'):
            left=next(name for name,v in variants.items() if v.get('axis_control')=='spread' and v.get('depth_policy')==policy)
            right=next(name for name,v in variants.items() if v.get('axis_control')=='random1212' and v.get('depth_policy')==policy)
            pairs.add((left,right))
    comparisons=[]
    for left,right in sorted(pairs):
        require(by[left].keys()==by[right].keys(), 'Paired ANLS does not use identical first questions')
        delta=[(key[0],by[left][key]-by[right][key]) for key in sorted(by[left])]
        estimate=cluster_bootstrap(delta,repeats=args.bootstrap_repeats,seed=1212)
        comparisons.append(dict(variant=left,baseline=right,questions=len(delta),source_documents=estimate['n_images'],
            delta_anls=estimate['mean'],delta_pp=100*estimate['mean'],ci_low=estimate['ci_lower'],ci_high=estimate['ci_upper'],
            ci_low_pp=None if estimate['ci_lower'] is None else 100*estimate['ci_lower'],
            ci_high_pp=None if estimate['ci_upper'] is None else 100*estimate['ci_upper'],
            wins=sum(x>0 for _,x in delta),ties=sum(x==0 for _,x in delta),losses=sum(x<0 for _,x in delta)))
    result=dict(schema='trajkv.round6.second-backbone-packed-analysis.v1',status='COMPLETE',
        analysis_role='development_unfitted_static_transfer',sessions=len(records),variants=len(variants),
        source_questions_per_variant=len(selected_union),turns_scored=[1],selected_ids=sorted(selected_union),
        metric=PROTOCOLS['docvqa'],confidence_interval='95% percentile source_document_id cluster bootstrap, question-weighted; paired differences use identical first-question keys',
        bootstrap_repeats=args.bootstrap_repeats,bootstrap_seed=1212,
        full_baseline=full,full_absolute_quality=next(x for x in table if x['variant']==full),
        source_geometry=dict(layers=L,kv_heads=H,head_dim=D,visual_tokens=NV,padded_visual_tokens=0),
        scope='actual packed second-backbone first-question E2E using new 32x32 identity factors and fixed source-independent static rankings; no second-model calibration; reused Docdev sources, not blind confirmation',
        comparison_limit='Only the same first-question subset is compared; do not equate this ANLS with primary-backbone multi-turn or 255-question scores. Runtime wall time includes native extraction and compressed suffix replay, not optimized deployment prefill.',
        audit_passed=True,samples_sha256=sha(args.samples),references_sha256=sha(args.references),
        analyzer_sha256=sha(__file__),scorer_sha256=sha(ROOT/'code/analysis/analyze_external.py'),
        bootstrap_source_sha256=sha(ROOT/'code/analysis/analyze_cycle.py'),runs=receipts,table=table,comparisons=comparisons)
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,rows in [('scores.csv',scored),('table.csv',table),('comparisons.csv',comparisons),('owner_audit.csv',audits)]:
        csv_write(args.output/filename,rows)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','questions_per_variant':len(selected_union),'table':table,'comparisons':comparisons}))


if __name__=='__main__':
    main()
