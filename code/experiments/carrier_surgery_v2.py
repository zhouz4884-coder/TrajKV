"""Nonzero, magnitude-matched K/V history surgery on a fixed transcript.

This is a continuous cache diagnostic, not a legal codec or online repair
algorithm. Carrier candidates use only previously generated answer tokens.
All future questions and answer tokens are identical across interventions.
V2 excludes the common pre-quantization prefix, matches actual reference/base
state-difference magnitude independently for K/V, and supports real 1-bit reads.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import torch

from bank import ProgressiveBank, install_legacy
from run_cycle import digest, features
from mechanism import (generate_canonical, teacher_turn, snapshot_difference,
                       logit_comparison, _read_cost, _snapshot_position, _sync)


STOP_WORDS = {'a','an','the','is','are','was','were','it','its','they','there',
              'that','this','these','those','of','to','in','on','at','and','or',
              'with','for','as','what','about','how','does','do','be'}


def _decode(tokenizer, ids):
    return tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)


def answer_carriers(tokenizer, turn, eos_ids, mode='both', maximum=1):
    """Map completed-answer lexical words back onto actual generated IDs.

    Prefix decode alignment is checked against the full string. A byte-piece
    sequence whose prefixes are not monotone is rejected instead of assigning
    guessed positions. Word identity is a candidate heuristic, not a semantic
    assertion that the selected span carries a correct visual fact.
    """
    ids=turn['token_ids']
    text=_decode(tokenizer,ids)
    prefixes=[_decode(tokenizer,ids[:i]) for i in range(len(ids)+1)]
    offsets=[]
    for prefix in prefixes:
        common=0
        while common<min(len(prefix),len(text)) and prefix[common]==text[common]:
            common+=1
        offsets.append(common)
    if any(b<a for a,b in zip(offsets,offsets[1:])):
        return [],dict(decoded=text,status='nonmonotone_prefix_decode',prefix_offsets=offsets)
    start=turn['logical_start']+len(turn['suffix_token_ids'])
    result=[]
    special=set(getattr(tokenizer,'all_special_ids',[]))|set(eos_ids)
    for match in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",text):
        word=match.group()
        if word.lower() in STOP_WORDS:
            continue
        token_indices=[i for i in range(len(ids))
                       if offsets[i]<match.end() and offsets[i+1]>match.start() and ids[i] not in special]
        if not token_indices or token_indices!=list(range(token_indices[0],token_indices[-1]+1)):
            continue
        choices=[]
        if mode in ('word','both'):
            choices.append(('word',token_indices))
        if mode in ('single','both'):
            choices.append(('single',[token_indices[0]]))
        for granularity,indices in choices:
            positions=[start+i for i in indices]
            existing=next((r for r in result if r['positions']==positions),None)
            if existing is not None:
                existing['granularities'].append(granularity)
                continue
            result.append(dict(candidate_word=word,character_span=[match.start(),match.end()],
                               answer_token_indices=indices,positions=positions,
                               token_ids=[ids[i] for i in indices],granularities=[granularity],
                               selected_using='completed current answer only; no future question or target'))
        if len({r['character_span'][0] for r in result})>=maximum:
            break
    return result,dict(decoded=text,status='aligned' if result else 'no_eligible_word',prefix_offsets=offsets)


def token_position_map(initial_ids, turns):
    result={i:int(token) for i,token in enumerate(initial_ids)}
    for turn in turns:
        position=turn['logical_start']
        sequence=turn['suffix_token_ids']+turn['token_ids']
        if turn['forced_chat_closure']:
            sequence=sequence+[turn['closure_token_id']]
        for token in sequence:
            if position in result:
                raise ValueError('Duplicate logical token position')
            result[position]=int(token)
            position+=1
        if position!=turn['logical_end']:
            raise ValueError('Transcript position accounting failed')
    return result


def _runs(positions, width):
    available=set(positions)
    return [list(range(p,p+width)) for p in sorted(available)
            if all(q in available for q in range(p,p+width))]


def position_delta_energies(base, reference, axis, selected_layers, minimum_position):
    """Actual FP16 reference/base delta energy per original text position."""
    if axis not in ('k','v','both'):
        raise ValueError('Unknown state axis')
    axes=(0,1) if axis=='both' else ((0,) if axis=='k' else (1,))
    positions=base[0]['values'][2]
    energy=torch.zeros(positions.numel(),dtype=torch.float64)
    for layer in selected_layers:
        b,r=base[layer],reference[layer]
        if not torch.equal(b['values'][2],positions) or not torch.equal(r['values'][2],positions):
            raise ValueError('Text positions differ across layers or history branches')
        for offset in axes:
            delta=r['values'][offset].float()-b['values'][offset].float()
            energy+=delta.square().sum((0,1,3),dtype=torch.float64)
    return {int(position):float(value) for position,value in zip(positions.tolist(),energy.tolist())
            if position>=minimum_position}


def matched_positions(tokenizer, turn, carrier, source_snapshot, reference_snapshot,
                      position_ids, seed, axis, selected_layers, minimum_position,
                      maximum_ratio=1.5, random_top_k=5):
    """Match nonzero counterfactual deltas, never a shared-prefix zero vector.

    Both controls first use current-turn question suffix positions, outside the
    generated answer. Random control can fall back to earlier post-quantization
    text. Every token in an eligible span has nonzero delta on the chosen axis.
    Failure to match is returned explicitly; no zero control is substituted.
    """
    width=len(carrier['positions'])
    specials=set(getattr(tokenizer,'all_special_ids',[]))
    energy=position_delta_energies(source_snapshot,reference_snapshot,axis,selected_layers,minimum_position)
    carrier_energy=sum(energy.get(p,0.) for p in carrier['positions'])
    carrier_norm=math.sqrt(carrier_energy)
    target_end=turn['logical_start']+len(turn['suffix_token_ids'])
    positive=[p for p,e in energy.items() if e>0 and position_ids[p] not in specials]
    prompt=[p for p in positive if turn['logical_start']<=p<target_end]
    forbidden=set(carrier['positions'])

    def describe(span, norm, pool, matched, candidate_count, reason=None):
        ratio=norm/carrier_norm if carrier_norm else None
        return dict(positions=span,token_ids=[position_ids[p] for p in span],
                    decoded=_decode(tokenizer,[position_ids[p] for p in span]),
                    actual_delta_l2=norm,carrier_delta_l2=carrier_norm,
                    delta_l2_ratio_to_carrier=ratio,matched=matched,
                    candidate_pool=pool,eligible_candidates=candidate_count,
                    every_position_delta_nonzero=all(energy.get(p,0)>0 for p in span),
                    minimum_position=minimum_position,unmatched_reason=reason)

    carrier_positive=all(energy.get(p,0)>0 for p in carrier['positions'])
    details={'carrier':describe(carrier['positions'],carrier_norm,'generated_current_answer',
                                carrier_positive and carrier_norm>0,1,
                                None if carrier_positive and carrier_norm>0 else 'carrier_has_zero_delta_position')}
    status=dict(axis=axis,span_length=width,carrier_delta_l2=carrier_norm,
                maximum_allowed_norm_ratio=maximum_ratio,
                nonzero_postquantization_positions=len(positive),
                shared_prefix_excluded_below=minimum_position,
                semantic_warning='location control is not certified fact-free; matching uses offline reference states')
    if not carrier_positive or carrier_norm<=0:
        for name in ('prompt_control','random_control'):
            details[name]=dict(matched=False,positions=[],unmatched_reason='zero_delta_carrier')
        status['status']='unmatched_zero_delta_carrier'
        return details,status

    def candidates(pool, excluded):
        values=[]
        for span in _runs([p for p in pool if p not in excluded],width):
            norm=math.sqrt(sum(energy[p] for p in span))
            ratio=norm/carrier_norm
            if norm>0:
                values.append((abs(math.log(ratio)),span,norm,1/maximum_ratio<=ratio<=maximum_ratio))
        return sorted(values,key=lambda row:(row[0],row[1]))

    prompt_candidates=candidates(prompt,forbidden)
    valid=[value for value in prompt_candidates if value[3]]
    if valid:
        _,span,norm,_=valid[0]
        details['prompt_control']=describe(span,norm,'same_turn_question_suffix',True,len(valid))
        forbidden.update(span)
    elif prompt_candidates:
        _,span,norm,_=prompt_candidates[0]
        details['prompt_control']=describe(span,norm,'same_turn_question_suffix',False,0,'norm_ratio_outside_tolerance')
    else:
        details['prompt_control']=dict(matched=False,positions=[],unmatched_reason='no_nonzero_same_turn_prompt_span')

    random_candidates=candidates(prompt,forbidden)
    random_valid=[value for value in random_candidates if value[3]]
    random_pool='same_turn_question_suffix_outside_answer'
    if not random_valid:
        # Avoid the whole current answer, not just the selected carrier.
        prior=[p for p in positive if p<target_end and p not in forbidden]
        random_candidates=candidates(prior,forbidden)
        random_valid=[value for value in random_candidates if value[3]]
        random_pool='postquantization_history_outside_current_answer'
    if random_valid:
        _,span,norm,_=random.Random(seed).choice(random_valid[:random_top_k])
        details['random_control']=describe(span,norm,random_pool,True,len(random_valid))
    elif random_candidates:
        _,span,norm,_=random_candidates[0]
        details['random_control']=describe(span,norm,random_pool,False,0,'norm_ratio_outside_tolerance')
    else:
        details['random_control']=dict(matched=False,positions=[],unmatched_reason='no_nonzero_outside_answer_span')
    status['status']='matched' if all(d['matched'] for d in details.values()) else 'partially_unmatched'
    status['match_successes']={name:detail['matched'] for name,detail in details.items()}
    return details,status


def surgical_snapshot(base, reference, positions, axis, alpha, selected_layers=None, delta_scale=1.):
    """Replace only selected stored K/V elements; preserve positions and lengths."""
    if axis not in ('k','v','both') or not 0<=alpha<=1 or not math.isfinite(delta_scale) or delta_scale<=0:
        raise ValueError('Expected axis k/v/both and interpolation alpha in [0,1]')
    if len(base)!=len(reference):
        raise ValueError('Layer count mismatch')
    selected_layers=set(range(len(base))) if selected_layers is None else set(selected_layers)
    if not selected_layers or min(selected_layers)<0 or max(selected_layers)>=len(base):
        raise ValueError('Invalid layer selection')
    if len(set(positions))!=len(positions) or not positions:
        raise ValueError('Positions must be unique and nonempty')
    result=[]
    total_bytes=0
    intervention_squared=reference_squared=0.
    largest=0.
    axes=(0,1) if axis=='both' else ((0,) if axis=='k' else (1,))
    for layer,(b,r) in enumerate(zip(base,reference)):
        if (b['length'],b['text_length'])!=(r['length'],r['text_length']):
            raise ValueError('History branches have different token counts')
        if not torch.equal(b['values'][2],r['values'][2]):
            raise ValueError('History branches have different positions')
        values=list(b['values'])
        if layer in selected_layers:
            ids=torch.tensor(positions,dtype=b['values'][2].dtype)
            indices=torch.searchsorted(b['values'][2],ids)
            if bool((indices>=b['text_length']).any()) or not torch.equal(b['values'][2][indices],ids):
                raise ValueError('Selected position is not a text-cache position')
            for offset in axes:
                original=b['values'][offset].index_select(2,indices)
                goal=r['values'][offset].index_select(2,indices)
                if alpha==0:
                    written=original.clone()
                elif alpha==1 and delta_scale==1:
                    written=goal.clone()
                else:
                    written=(original.float()+alpha*delta_scale*(goal.float()-original.float())).to(original.dtype)
                values[offset]=values[offset].clone()
                values[offset].index_copy_(2,indices,written)
                error=written.float()-original.float()
                intervention_squared+=float(error.square().sum(dtype=torch.float64))
                reference_squared+=float((goal.float()-original.float()).square().sum(dtype=torch.float64))
                largest=max(largest,float(error.abs().max()))
                total_bytes+=written.numel()*written.element_size()
        result.append(dict(length=b['length'],text_length=b['text_length'],values=values))
    # Hash the actual immutable position tensors on both sides; no input token
    # or original position is rewritten by this intervention.
    position_hash=lambda snapshots:hashlib.sha256(b''.join(s['values'][2].numpy().tobytes() for s in snapshots)).hexdigest()
    before_hash,after_hash=position_hash(base),position_hash(result)
    if before_hash!=after_hash:
        raise AssertionError('Surgery changed cache positions')
    return result,dict(axis=axis,alpha=alpha,delta_scale=delta_scale,
                       exact_reference_endpoint=alpha==1 and delta_scale==1,
                       positions=list(positions),selected_layers=sorted(selected_layers),
                       logical_value_write_bytes=total_bytes,
                       write_scope='selected cache K/V value bytes; CPU snapshot cloning and complete restore are diagnostic overhead',
                       position_sha256_before=before_hash,position_sha256_after=after_hash,
                       positions_unchanged=True,token_ids_modified=False,
                       intervention_l2=math.sqrt(intervention_squared),
                       full_ref_difference_at_selected_elements_l2=math.sqrt(reference_squared),
                       intervention_max_abs=largest,
                       interpolation_scope='continuous diagnostic, not legal bitwidth, natural history, or serving policy')


def future_probe(model, bank, source, turns, traces, write_turn, args, initial_logits,
                 eos, tag, mode='base', baseline_logits=None):
    state=source
    rows=[]
    logits_list=[]
    for number in range(write_turn+1,min(len(turns),write_turn+args.future_horizon)+1):
        pass_kind=f'{tag}_future{number}'
        logits,state=teacher_turn(model,bank,state,turns[number-1],number,mode,initial_logits,pass_kind)
        reference=traces[number-1] if baseline_logits is None else baseline_logits[number-write_turn-1]
        metrics=logit_comparison(logits,reference,turns[number-1]['token_ids'],eos)
        rows.append(dict(turn=number,horizon=number-write_turn,metrics=metrics,read_cost=_read_cost(bank,pass_kind)))
        logits_list.append(logits)
    return rows,logits_list,state


def _flatten_metrics(session, image, record):
    first=record['future'][0]['metrics']['per_token'][0]
    for future in record['future']:
        yield dict(intervention_id=record['intervention_id']+f':h{future["horizon"]}',
                   session_id=session,image_id=image,write_turn=record['write_turn'],
                   axis=record['axis'],alpha=record['alpha'],position_type=record['position_type'],
                   dose_kind=record['dose_kind'],base_bits=record['base_bits'],
                   location_pair_complete=record['location_pair_complete'],
                   matching_case_status=record['matching_case_status'],
                   delta_l2_ratio_to_carrier=record['selected']['delta_l2_ratio_to_carrier'],
                   actual_injection_l2=record['surgery']['intervention_l2'],
                   delta_scale=record['surgery']['delta_scale'],
                   carrier_id=record['carrier_id'],horizon=future['horizon'],
                   policy='matched_history_carrier_continuous_surgery',action=record['axis'],
                   delta_current=first['target_logprob_gain'],
                   delta_future=future['metrics']['content_logprob_gain_mean'],
                   current_metric='first_next_answer_canonical_token_logprob_gain_after_same_question_suffix',
                   future_metric='mean_canonical_content_token_logprob_gain',
                   current_effect_kl=first['kl_reference_to_actual'],
                   current_effect_logit_rms=first['rms_logit_difference'],
                   current_effect_argmax_changed=first['argmax_changed'],
                   future_content_tokens=future['metrics']['content_token_count'],
                   logical_value_write_bytes=record['surgery']['logical_value_write_bytes'],
                   read_cost=future['read_cost'],
                   positive_means='higher likelihood of fixed base-generated tokens; not measured task accuracy')


def run_session(model, processor, row, args, records_log, controls_log, scatter_log):
    from p180_multiturn_core import attention_bridge
    started=time.perf_counter()
    f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype)
       for k,v in features(processor,row,args.max_patches).items()}
    initial_ids=f['input_ids'][0].cpu().tolist()
    visual=torch.nonzero(f['input_ids'][0]==model.config.image_token_id).flatten()
    with attention_bridge(model,native=True):
        out=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
    initial_logits=out.logits[:,-1].detach().float().cpu().clone()
    cache=out.past_key_values
    del out,f
    method='asym1' if args.base_bits==1 else 'asym'
    bank=ProgressiveBank(model,cache,visual,method,args.legacy_scripts,args.kivi_source,keep_reference=True)
    image=row.get('source_image_id',row.get('image_id'))
    try:
        initial=bank.snapshot(cpu=True)
        turns,traces,original_states,eos=generate_canonical(model,processor,bank,row,args,initial_logits)
        position_ids=token_position_map(initial_ids,turns)
        transcript_hash=hashlib.sha256(json.dumps([t['suffix_token_ids']+t['token_ids']+
            ([t['closure_token_id']] if t['forced_chat_closure'] else []) for t in turns]).encode()).hexdigest()
        selected_turns=sorted(t for t in set(args.write_turns) if 1<=t<len(turns))
        if args.layer_group=='all':
            layers=list(range(len(bank.states)))
        else:
            group=int(args.layer_group)
            layers=list(range(group*len(bank.states)//4,(group+1)*len(bank.states)//4))
        ref_history=base_history=initial
        sources={}
        for number,turn in enumerate(turns,1):
            if number>max(selected_turns,default=0):
                break
            _,ref_history=teacher_turn(model,bank,ref_history,turn,number,'ref',initial_logits,'source_ref_history')
            base_logits,base_history=teacher_turn(model,bank,base_history,turn,number,'base',initial_logits,'source_base_history')
            if number in selected_turns:
                sources[number]=(base_history,ref_history)
                control=dict(session_id=row['id'],image_id=image,write_turn=number,
                             base_bits=args.base_bits,
                             control='source_base_reencoding_vs_original',
                             metrics=logit_comparison(base_logits,traces[number-1],turn['token_ids'],eos),
                             state=snapshot_difference(base_history,original_states[number-1],_snapshot_position(initial)))
                controls_log.write(json.dumps(control)+'\n');controls_log.flush()
        summaries=[]
        record_count=0
        for write_turn in selected_turns:
            base,reference=sources[write_turn]
            carriers,alignment=answer_carriers(processor.tokenizer,turns[write_turn-1],eos,args.carrier_mode,args.max_carriers)
            baseline,baseline_logits,_=future_probe(model,bank,base,turns,traces,write_turn,args,initial_logits,
                                                   eos,f't{write_turn}_original_base')
            future_controls={}
            for name,history,mode in (('base_repeat_null',base,'base'),
                                      ('complete_ref_history__base_future_read',reference,'base'),
                                      ('base_history__ref_future_read',base,'ref'),
                                      ('complete_ref_history__ref_future_read',reference,'ref')):
                values,_,_=future_probe(model,bank,history,turns,traces,write_turn,args,initial_logits,eos,
                                        f't{write_turn}_{name}',mode,baseline_logits)
                control=dict(session_id=row['id'],image_id=image,write_turn=write_turn,control=name,future=values,
                             base_bits=args.base_bits,
                             token_schedule_sha256=transcript_hash,reference_answers_visible_to_runner=False)
                future_controls[name]=values
                controls_log.write(json.dumps(control)+'\n');controls_log.flush()
            current_summary=dict(write_turn=write_turn,carrier_alignment=alignment,carriers=[],baseline=baseline,
                                 full_history_state_difference=snapshot_difference(base,reference,_snapshot_position(initial)))
            for carrier_index,carrier in enumerate(carriers):
                seed=int(hashlib.sha256(f'{row["id"]}:{write_turn}:{carrier_index}'.encode()).hexdigest()[:16],16)
                carrier_id=f't{write_turn}_carrier{carrier_index}'
                expected_bytes={}
                for axis in args.axes:
                    matches,match_status=matched_positions(processor.tokenizer,turns[write_turn-1],carrier,
                        base,reference,position_ids,seed,axis,layers,_snapshot_position(initial),
                        args.match_max_ratio,args.random_match_top_k)
                    current_summary['carriers'].append(dict(carrier=carrier,axis=axis,
                                                           matching=match_status,positions=matches))
                    matching_record=dict(session_id=row['id'],image_id=image,write_turn=write_turn,
                                         base_bits=args.base_bits,control='position_matching',
                                         carrier_id=carrier_id,axis=axis,matching=match_status,positions=matches)
                    controls_log.write(json.dumps(matching_record)+'\n');controls_log.flush()
                    print(json.dumps(dict(stage='position_matching_v2',session=row['id'],write_turn=write_turn,
                                          axis=axis,carrier_id=carrier_id,matching=match_status)),flush=True)
                    for position_type,details in matches.items():
                        if not details['matched']:
                            continue
                        dose_kinds=[('reference_delta',1.)]
                        if args.normalize_delta:
                            scale=match_status['carrier_delta_l2']/details['actual_delta_l2']
                            dose_kinds.append(('synthetic_norm_matched',scale))
                        for dose_kind,delta_scale in dose_kinds:
                            for alpha in args.alphas:
                                intervention_id=f'{row["id"]}:b{args.base_bits}:{carrier_id}:{position_type}:{axis}:{dose_kind}:alpha{alpha:g}'
                                edited,audit=surgical_snapshot(base,reference,details['positions'],axis,alpha,layers,delta_scale)
                                if axis in expected_bytes and expected_bytes[axis]!=audit['logical_value_write_bytes']:
                                    raise AssertionError('Matched position controls have different write budgets')
                                expected_bytes[axis]=audit['logical_value_write_bytes']
                                if alpha>0 and audit['intervention_l2']==0:
                                    # A nonzero real delta can round away at small alpha;
                                    # preserve this observation instead of faking a dose.
                                    audit['dose_rounded_to_zero']=True
                                else:
                                    audit['dose_rounded_to_zero']=False
                                expected_l2=alpha*match_status['carrier_delta_l2']
                                audit['injection_l2_ratio_to_alpha_carrier']=(audit['intervention_l2']/expected_l2 if expected_l2 else None)
                                future,_,_=future_probe(model,bank,edited,turns,traces,write_turn,args,initial_logits,
                                                        eos,intervention_id,'base',baseline_logits)
                                record=dict(schema='trajkv.carrier_surgery.v2',intervention_id=intervention_id,
                                            session_id=row['id'],image_id=image,split=row.get('split'),base_bits=args.base_bits,
                                            write_turn=write_turn,carrier_id=carrier_id,carrier=carrier,
                                            position_type=position_type,selected=details,axis=axis,alpha=alpha,
                                            location_pair_complete=match_status['status']=='matched',
                                            matching_case_status=match_status['status'],
                                            dose_kind=dose_kind,surgery=audit,future=future,
                                            token_schedule_sha256_before=transcript_hash,
                                            token_schedule_sha256_after=transcript_hash,
                                            future_read=f'base{args.base_bits} only; all visual tokens exposed',
                                            current_influence='first next-answer prediction after common next-question suffix',
                                            matching_scope='axis-specific, nonzero postquantization locations; empirical real-delta L2 matching',
                                            causal_scope='synthetic history-state intervention; not a natural history or legal encoding',
                                            reference_answers_visible_to_runner=False,paid_calls=0)
                                records_log.write(json.dumps(record)+'\n');records_log.flush()
                                for point in _flatten_metrics(row['id'],image,record):
                                    scatter_log.write(json.dumps(point)+'\n')
                                scatter_log.flush()
                                record_count+=1
                                print(json.dumps(dict(stage='carrier_surgery_v2',session=row['id'],write_turn=write_turn,
                                                      position_type=position_type,axis=axis,alpha=alpha,dose_kind=dose_kind,
                                                      norm_ratio=details['delta_l2_ratio_to_carrier'],
                                                      future_gains=[r['metrics']['content_logprob_gain_mean'] for r in future],
                                                      logical_write_bytes=audit['logical_value_write_bytes'])),flush=True)
                                del edited
            summaries.append(current_summary)
        _sync()
        return dict(schema='trajkv.carrier_surgery_session.v2',session_id=row['id'],image_id=image,base_bits=args.base_bits,
                    canonical_turns=turns,token_schedule_sha256=transcript_hash,source_turns=selected_turns,
                    cases=summaries,interventions=record_count,selected_layers=layers,
                    owned_diagnostic=bank.owned(),read_events=bank.event_rows(),
                    session_ms=1000*(time.perf_counter()-started),
                    carrier_definition='lexical word or token in previously generated answer; no correctness or factual-carrier annotation',
                    controls='nonzero postquantization positions, axis-specific delta-L2 matching within frozen ratio tolerance; same position count/layers/bytes',
                    null_policy='base repeat plus alpha0 replicates; empirical errors reported without fixed rejection threshold',
                    diagnostic_memory='full reference retained; CPU snapshot restore overhead is not a deployment efficiency result',
                    paid_calls=0,reference_answers_visible_to_runner=False)
    finally:
        bank.close()
        del bank,cache
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','samples','legacy-scripts','kivi-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--turns',type=int,default=6)
    parser.add_argument('--write-turns',nargs='+',type=int,default=[2,4])
    parser.add_argument('--future-horizon',type=int,default=2)
    parser.add_argument('--max-tokens',type=int,default=24)
    parser.add_argument('--max-patches',type=int,default=4)
    parser.add_argument('--carrier-mode',choices=['word','single','both'],default='both')
    parser.add_argument('--max-carriers',type=int,default=1)
    parser.add_argument('--axes',nargs='+',choices=['k','v','both'],default=['k','v'])
    parser.add_argument('--alphas',nargs='+',type=float,default=[0.,.25,.5,.75,1.])
    parser.add_argument('--layer-group',choices=['all','0','1','2','3'],default='all')
    parser.add_argument('--base-bits',type=int,choices=[1,2],default=1)
    parser.add_argument('--match-max-ratio',type=float,default=1.5,
                        help='Accepted real delta norms are [carrier/ratio, carrier*ratio]')
    parser.add_argument('--random-match-top-k',type=int,default=5,
                        help='Random matched control samples from this many closest eligible spans')
    parser.add_argument('--normalize-delta',action='store_true',
                        help='Additionally emit separately labelled synthetic norm-matched doses')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.count,args.turns,args.future_horizon,args.max_tokens,args.max_patches,args.max_carriers)<1:
        parser.error('Counts, token caps and horizons must be positive')
    if any(not 0<=a<=1 for a in args.alphas):
        parser.error('Continuous diagnostic doses must be in [0,1]')
    if any(turn<1 for turn in args.write_turns):
        parser.error('Write turns must be positive')
    if not math.isfinite(args.match_max_ratio) or args.match_max_ratio<1:
        parser.error('Match ratio must be finite and at least one')
    if args.random_match_top_k<1:
        parser.error('Random match top-k must be positive')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    install_legacy(args.legacy_scripts)
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads(args.samples.read_text(encoding='utf-8'))[args.start:args.start+args.count]
    files=[Path(__file__),Path(__file__).with_name('mechanism.py'),Path(__file__).with_name('bank.py'),args.samples]
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  selected_ids=[r['id'] for r in rows],torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                  sources={str(path):digest(path) for path in files},reference_answers_visible_to_runner=False,
                  intervention_kind='continuous diagnostic, no claim of deployable bitwidth',paid_calls=0)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,
                device_map='cuda:0',attn_implementation='sdpa').eval()
    seen=set()
    session_path=args.output/'sessions.jsonl'
    if args.resume and session_path.exists():
        seen={json.loads(line)['session_id'] for line in session_path.read_text(encoding='utf-8').splitlines() if line.strip()}
    count=0
    with torch.inference_mode(), \
         (args.output/'surgery_records.jsonl').open('a',encoding='utf-8') as records_log, \
         (args.output/'controls.jsonl').open('a',encoding='utf-8') as controls_log, \
         (args.output/'surgery_scatter.jsonl').open('a',encoding='utf-8') as scatter_log, \
         session_path.open('a',encoding='utf-8') as session_log:
        for row in rows:
            if row['id'] in seen:
                continue
            summary=run_session(model,processor,row,args,records_log,controls_log,scatter_log)
            session_log.write(json.dumps(summary)+'\n');session_log.flush();count+=1
    (args.output/'complete.json').write_text(json.dumps(dict(sessions_added=count,paid_calls=0)),encoding='utf-8')


if __name__=='__main__':
    main()
