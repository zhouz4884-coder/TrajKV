# Isolated image-write-only prefix-anchor protocol.
"""Real single-image, persistent-cache TrajKV generation and calibration.

Run on the authorized GPU host. Quality references are scored separately;
no reference answer is provided to the generation process.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import torch
from masked_bank import A2Bank as ProgressiveBank
from bank import install_legacy

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()

def features(processor,row,max_patches):
    from PIL import Image
    images=[]
    for name in row.get('remote_images',row['images']):
        with Image.open(name) as im:images.append(im.convert('RGB'))
    prompt=processor.apply_chat_template([{'role':'user','content':row['content']}],tokenize=False,add_generation_prompt=True)
    return processor(text=[prompt],images=images,return_tensors='pt',crop_to_patches=True,max_patches=max_patches)

def margins(logits):
    top=logits.float().topk(2,-1).values
    return float((top[...,0]-top[...,1]).item())

def consume(model,cache,ids,logical):
    out=model(input_ids=ids,past_key_values=cache,
              position_ids=torch.arange(logical,logical+ids.shape[-1],device=ids.device)[None],
              use_cache=True,return_dict=True,logits_to_keep=1)
    return out,logical+ids.shape[-1]

def layer_schedule(name,bank,phase,step,turn):
    n=len(bank.states)
    if name in ('always','asym'):return set(range(n))
    if name=='prefill':return set(range(n)) if phase=='text_prefill' else set()
    if name=='early':return set(range(n)) if phase=='text_prefill' or step<2 else set()
    if name=='late':return set(range(n)) if phase=='decode' and step>=2 else set()
    if name.startswith('layer_'):
        group=int(name.split('_')[1]);return set(range(group*n//4,(group+1)*n//4))
    if name=='periodic':return set(range(n)) if step%4==0 else set()
    if name=='random':return set(range(n)) if random.Random(12000+turn*100+step).random()<.25 else set()
    if name=='uniform':
        group=(turn+step)%4;return set(range(group*n//4,(group+1)*n//4))
    if name in ('table_current','table_future'):
        axis=name.split('_',1)[1];kind='text_prefill' if phase=='text_prefill' else ('early' if step<2 else 'late')
        table=bank.gain_table['tables'][axis]['phase_layer_group_ranking']
        ranked=table.get(kind,[])
        if not ranked:return set()
        group=int(ranked[0]);return set(range(group*n//4,(group+1)*n//4))
    if name.startswith('dose_'):return set(range(n*int(name.split('_')[1])//4))
    return set()

def run_session(model,processor,row,args,method,policy,forced=None,keep_reference=False):
    from p180_multiturn_core import attention_bridge,next_suffix
    from p184_working_set import attention
    import run_p175_modern_replication as oldbase
    eos=oldbase.stop_ids(model,processor);canonical=processor.tokenizer.eos_token_id
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();sync();started=time.perf_counter()
    from anchor_protocol import prepare_row,prefix_inputs,verify_fixed_prefix
    from anchor_bank import anchor_count
    original_settings=ProgressiveBank.settings
    row=prepare_row(row,original_settings)
    f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype)
       for k,v in features(processor,row,args.max_patches).items()}
    visual=torch.nonzero(f['input_ids'][0]==model.config.image_token_id).flatten()
    visual_end=int(visual[-1])+1;count=anchor_count(model,visual.numel(),original_settings)
    boundary_start=visual_end+count
    verify_fixed_prefix(processor,row,args.max_patches,features,f,boundary_start,original_settings)
    native_end=boundary_start if original_settings.get('anchor_mode')=='native' else visual_end
    native=prefix_inputs(f,native_end)
    with attention_bridge(model,native=True):out=model(**native,use_cache=True,return_dict=True,logits_to_keep=1)
    cache=out.past_key_values;logical=native_end;native_history=False
    suffix=f['input_ids'][:,boundary_start:].clone();anchor_ids=f['input_ids'][:,visual_end:boundary_start].clone()
    del out,native,f
    ProgressiveBank.settings=dict(original_settings,_anchor_count=count)
    bank=ProgressiveBank(model,cache,visual,method,args.legacy_scripts,args.kivi_source,keep_reference=False)
    if original_settings.get('anchor_mode')=='compressed' and count:
        bank.phase='anchor_write';bank.pass_kind='committed';bank.turn=1
        with attention(model):out,logical=consume(model,cache,anchor_ids,visual_end)
        del out
        bank.capture_anchors()
    boundary_suffix=suffix
    bank.phase='question_prefill';bank.pass_kind='committed';bank.turn=1
    with attention(model):out,logical=consume(model,cache,boundary_suffix,boundary_start)
    logits=out.logits[:,-1];initial_logits=logits.float().cpu();del out
    bank.gain_table=getattr(args,'gain_table_data',None)
    bank.read_budget_bytes=getattr(args,'read_budget_bytes',None)
    if policy.startswith('probe'):
        from self_probe import ProbeController
        parts=policy.split('_')
        layer=int(parts[1][1:]) if len(parts)>1 else getattr(args,'probe_layer',7)
        threshold=float(parts[2][1:]) if len(parts)>2 else getattr(args,'probe_threshold',.02)
        bank.probe_controller=ProbeController(min(layer,len(bank.states)-1),threshold,budget_bytes=None)
    bank.refine_layers=layer_schedule(policy,bank,'decode',0,1)
    owned_initial=bank.owned();turns=[];snapshots=[];probe_rows=[];previous=None
    write_rows=[]
    def extra_spent():
        return bank.extra_read_bytes
    def limited_layers(requested,replay=False,incoming=1):
        limit=getattr(args,'read_budget_bytes',None)
        if limit is None:return set(requested)
        remaining=limit-extra_spent()
        if replay:
            remaining-=sum(sum(t.numel()*t.element_size() for t in bank._visual_tensors(s))+
                 2*s.text_k.shape[0]*s.text_k.shape[1]*(s.text_length+incoming)*s.text_k.shape[-1]*s.text_k.element_size()
                 for s in bank.states)
        selected=set()
        for i in sorted(requested):
            from bank import unique_bytes
            cost=unique_bytes(bank.refinements[i] or [])
            if cost<=remaining:selected.add(i);remaining-=cost
        return selected
    try:
      for number,question in enumerate(row['questions'][:args.turns],1):
        bank.on_turn_begin(number,logical);bank.turn=number;bank.phase='text_prefill';bank.pass_kind='committed'
        bank.hide_visual=bool(getattr(args,'hide_visual_after_description',False) and number>1)
        bank.refine_layers=layer_schedule(policy,bank,'text_prefill',0,number)
        sync();turn_started=started if number==1 else time.perf_counter();emissions=[];refined_steps=0
        repair_start=bank.snapshot() if policy.startswith('repair') else None
        repair_logical=logical;suffix=None
        if number>1:
            suffix=next_suffix(processor,question).to(model.device)
            before=bank.lengths()
            bank.refine_layers=limited_layers(bank.refine_layers,incoming=suffix.shape[-1])
            with attention(model):out,logical=consume(model,cache,suffix,logical)
            logits=out.logits[:,-1];del out
            previous=(suffix,before,logical-suffix.shape[-1],'text_prefill')
        ids=[];token_margins=[];replays=0
        target=None if forced is None else forced[number-1]['token_ids']
        token_cap=(getattr(args,'description_tokens',96) if number==1 and not row.get('first_turn_scored',True) else args.max_tokens)
        for step in range(token_cap if target is None else len(target)):
            bank.phase='decode'
            margin=margins(logits);token_margins.append(margin)
            token=int(logits.argmax(-1).item())
            trigger=policy in ('margin','carrier_margin') and margin<args.margin and previous is not None
            word=processor.tokenizer.decode([token],skip_special_tokens=True).strip()
            carrier=bool(word) and any(ch.isalnum() for ch in word) and word.lower() not in {'the','a','an','is','it','of','and','to','in'}
            trigger=trigger and (policy!='carrier_margin' or carrier) and replays<args.max_replays
            if trigger:
                consumed,prior,prior_logical,consumed_phase=previous
                selected=limited_layers(set(range(len(bank.states))),replay=True,incoming=consumed.shape[-1])
                if selected:
                    bank.rollback(prior);bank.refine_layers=selected;bank.pass_kind='replay';bank.phase=consumed_phase
                    with attention(model):out,logical=consume(model,cache,consumed,prior_logical)
                    logits=out.logits[:,-1];del out
                    new=int(logits.argmax(-1).item());replays+=1
                    probe_rows.append(dict(turn=number,step=step,margin=margin,carrier=carrier,base_token=token,
                                           refined_token=new,changed=token!=new,consumed_start=prior_logical,
                                           consumed_end=logical,consumed_phase=consumed_phase,selected_next_position=logical))
                    token=new
            if target is not None:
                token=target[step]
                word=processor.tokenizer.decode([token],skip_special_tokens=True).strip()
                carrier=bool(word) and any(ch.isalnum() for ch in word) and word.lower() not in {'the','a','an','is','it','of','and','to','in'}
            ids.append(token)
            emissions.append(time.perf_counter())
            requested=layer_schedule(policy,bank,'decode',step,number)
            write_trigger=(policy=='margin_write' and margin<args.margin) or (policy=='carrier_write' and carrier and margin<args.margin) or (policy=='carrier_all' and carrier)
            if write_trigger and refined_steps<args.max_replays:requested=set(range(len(bank.states)))
            bank.refine_layers=limited_layers(requested);bank.pass_kind='committed';bank.phase='decode'
            if bank.refine_layers:
                refined_steps+=1
                write_rows.append(dict(turn=number,step=step,input_position=logical,input_token=token,
                    selected_from_margin=margin,lexical_content_heuristic=carrier,layers=sorted(bank.refine_layers),
                    feature_availability='online',selected_next_position=logical+1))
            incoming=torch.tensor([[token]],device=model.device);prior=bank.lengths();prior_logical=logical
            with attention(model):out,logical=consume(model,cache,incoming,logical)
            logits=out.logits[:,-1];del out
            previous=(incoming,prior,prior_logical,'decode')
            if target is None and token in eos:break
        closed=not ids or ids[-1]!=canonical
        if closed:
            bank.phase='decode';bank.pass_kind='committed'
            bank.refine_layers=limited_layers(layer_schedule(policy,bank,'decode',len(ids),number))
            incoming=torch.tensor([[canonical]],device=model.device);prior=bank.lengths();prior_logical=logical
            with attention(model):out,logical=consume(model,cache,incoming,logical)
            logits=out.logits[:,-1];del out
            previous=(incoming,prior,prior_logical,'decode')
        if repair_start is not None and (policy=='repair' or (policy=='repair_once' and number==2)):
            # Same generated text, more accurate state formation. No future query
            # or answer is consulted. This is a coarse, charged prototype.
            replay_ids=ids+([canonical] if closed else [])
            incoming=torch.tensor([replay_ids],device=model.device)
            if suffix is not None:incoming=torch.cat((suffix,incoming),-1)
            # Repair preserves the original suffix/token chunk schedule. Charge
            # every replayed base/text scan, rather than one concatenated call.
            chunks=([suffix.shape[-1]] if suffix is not None else [])+[1]*len(replay_ids)
            cap=getattr(args,'read_budget_bytes',None)
            remaining=float('inf') if cap is None else cap-extra_spent()
            from bank import unique_bytes
            if cap is not None:
                for s,start_state in zip(bank.states,repair_start):
                    text_count=start_state['text_length']
                    for chunk in chunks:
                        text_count+=chunk
                        remaining-=(0 if bank.hide_visual else unique_bytes(bank._visual_tensors(s)))
                        remaining-=2*s.text_k.shape[0]*s.text_k.shape[1]*text_count*s.text_k.shape[-1]*s.text_k.element_size()
            selected=set()
            for i in range(len(bank.states)):
                cost=0 if bank.hide_visual else unique_bytes(bank.refinements[i] or [])*len(chunks)
                if cost<=remaining:selected.add(i);remaining-=cost
            if selected:
                bank.restore(repair_start);bank.refine_layers=selected;bank.pass_kind='repair';bank.phase='state_repair'
                logical=repair_logical
                with attention(model):
                    if suffix is not None:
                        out,logical=consume(model,cache,suffix,logical);logits=out.logits[:,-1];del out
                    for value in replay_ids:
                        incoming=torch.tensor([[value]],device=model.device);prior=bank.lengths();prior_logical=logical
                        out,logical=consume(model,cache,incoming,logical);logits=out.logits[:,-1];del out
                previous=(incoming,prior,prior_logical,'state_repair')
        sync()
        turn=dict(turn=number,answer=processor.tokenizer.decode(ids,skip_special_tokens=True).strip(),
                  token_ids=ids,forced_chat_closure=closed,logical_end=logical,margin=token_margins,
                  replay_count=replays,engine_ms=1000*(time.perf_counter()-turn_started))
        turn.update(refined_write_steps=refined_steps,ttft_ms=1000*(emissions[0]-turn_started) if emissions else None,
                    mean_inter_token_ms=1000*sum(b-a for a,b in zip(emissions,emissions[1:]))/max(len(emissions)-1,1))
        bank.on_turn_end(number,logical)
        turns.append(turn)
        if args.save_states:snapshots.append(bank.snapshot(cpu=True))
        print(json.dumps(dict(session=row['id'],method=method,policy=policy,turn=number,
                              tokens=len(ids),answer=turn['answer'],ms=turn['engine_ms'])),flush=True)
      sync()
      result=dict(session_id=row['id'],image_id=row['source_image_id'],split=row['split'],method=method,
                  policy=policy,protocol='image_boundary_fixed_tokens' if forced else 'image_boundary_free_running',turns=turns,
                  dataset=row.get('dataset'),data_protocol=row.get('protocol'),
                  native_session_id=row.get('native_session_id',row['id']),
                  reference_turn_offset=row.get('reference_turn_offset',0),
                  first_turn_scored=row.get('first_turn_scored',True),
                  hide_visual_after_description=getattr(args,'hide_visual_after_description',False),
                  initial_logits_sha256=hashlib.sha256(initial_logits.numpy().tobytes()).hexdigest(),
                  visual_tokens=int(visual.numel()),owned_initial=owned_initial,owned_final=bank.owned(),
                  peak_allocated=torch.cuda.max_memory_allocated(),session_ms=1000*(time.perf_counter()-started),
                  read_events=bank.event_rows(),margin_probes=probe_rows,model=str(args.model),
                  state_write_events=write_rows,extra_logical_read_bytes=extra_spent(),
                  read_budget_bytes=getattr(args,'read_budget_bytes',None),
                  read_budget_ok=getattr(args,'read_budget_bytes',None) is None or extra_spent()<=args.read_budget_bytes,
                  intra_step_probes=bank.probe_controller.event_rows() if bank.probe_controller else [],
                  intra_step_probe_summary=bank.probe_controller.summary() if bank.probe_controller else None,
                  intra_step_probe_skipped=bank.probe_skipped,
                  read_byte_scope='logical tensor payload per fused call; not measured HBM traffic',
                  backend='historical_P202_fused_all_visual_support',paid_calls=0)
      raw_extra=sum(e.get('selection_bytes',0)+e['residual_bytes']+(e['base_visual_bytes']+e['text_bytes'] if e['pass_kind'] in ('replay','repair','probe') else 0)
                    for e in bank.event_rows())
      if raw_extra!=bank.extra_read_bytes:raise RuntimeError('Read ledger differs from actual fused-call records')
      result['read_ledger_exact']=True
      result['compression_point']='native image plus fixed prefix only; actual question unseen before packing'
      result['question_state_scope']='fixed-prefix anchors then actual question recomputed under codec'
      if native_history:result['protocol']='native_history_fixed_tokens' if forced else 'native_history_free_running'
      result['initial_logits_shared_fp16']=False
      result['first_question_suffix_replay_tokens']=int(boundary_suffix.shape[-1])
      result['prefill_work_scope']='native causal prefix extracted via full prefill, later native suffix discarded and charged replayed; no extraction-speed claim'
      result.update(bank.result_metadata())
      result['kernel_variant']='a2_all_support_masked_refinement'
      kernel=Path(__file__).with_name('fused_direction.py' if method=='anchor_direction' else 'fused_rates.py' if method=='anchor_affine' else 'fused_masked.py')
      result['kernel_source_sha256']=digest(kernel)
      return result,snapshots
    finally:bank.close();ProgressiveBank.settings=original_settings;del bank,cache;gc.collect()
