"""Native SDPA versus full fused-cache parity on the same generated tokens.

Two independent native visual prefills create separate caches. One retains
standard Transformers cache updates and SDPA dispatch; the other is converted
to the all-support FP16 fused bank. A single canonical token is committed to
both, at the same position and with the same chunking. No quantizer, selector,
reference answer, or arbitrary numerical rejection threshold is involved.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from bank import ProgressiveBank, install_legacy
from mechanism import logit_comparison, _sync
from run_cycle import consume, digest, features


def _tensor_summary(actual, reference):
    difference=actual.float()-reference.float()
    # Return all three scalar statistics in one device-to-host transfer.
    numbers=torch.stack((difference.square().sum(),reference.float().square().sum(),
                         difference.abs().max() if difference.numel() else difference.new_zeros(())))
    error,energy,largest=numbers.double().cpu().tolist()
    return dict(squared_error=error,squared_reference=energy,max_abs=largest,
                elements=difference.numel(),relative_l2=math.sqrt(error/max(energy,1e-300)),
                rms=math.sqrt(error/max(difference.numel(),1)))


def cache_difference(native_cache, bank, minimum_position=0, include_visual=False):
    """Compare original-position post-RoPE KV, excluding reserved capacity."""
    per_layer=[]
    aggregate={}
    if len(native_cache.layers)!=len(bank.states):
        raise ValueError('Native and fused cache layer counts differ')
    for layer,(native,state) in enumerate(zip(native_cache.layers,bank.states)):
        if native.keys is None or native.values is None:
            raise ValueError('Native cache unexpectedly lost its combined KV owner')
        if native.keys.shape[2]!=state.length or native.values.shape[2]!=state.length:
            raise ValueError('Native and fused logical cache lengths differ')
        if state.visual_pos.numel()+state.text_length!=state.length:
            raise ValueError('Fused visual/text position accounting is incomplete')
        positions=state.text_pos[:state.text_length]
        if bool((positions<0).any()) or bool((positions>=state.length).any()):
            raise ValueError('Fused text positions are outside native cache bounds')
        selected=positions>=minimum_position
        detail=dict(layer=layer,logical_length=state.length,
                    selected_text_positions=int(selected.sum()),text_positions_valid=True)
        for axis in ('k','v'):
            actual=getattr(state,'text_'+axis)[:,:,:state.text_length][:,:,selected]
            reference=(native.keys if axis=='k' else native.values).index_select(2,positions[selected])
            detail['text_'+axis]=_tensor_summary(actual,reference)
            if include_visual:
                if not hasattr(state,'visual_'+axis):
                    raise ValueError('Parity bank must retain unquantized full visual KV')
                actual_visual=getattr(state,'visual_'+axis)
                reference_visual=(native.keys if axis=='k' else native.values).index_select(2,state.visual_pos)
                detail['visual_'+axis]=_tensor_summary(actual_visual,reference_visual)
        for key,value in detail.items():
            if not isinstance(value,dict):
                continue
            total=aggregate.setdefault(key,dict(squared_error=0.,squared_reference=0.,max_abs=0.,elements=0))
            for name in ('squared_error','squared_reference','elements'):
                total[name]+=value[name]
            total['max_abs']=max(total['max_abs'],value['max_abs'])
        per_layer.append(detail)
    for total in aggregate.values():
        total['relative_l2']=math.sqrt(total['squared_error']/max(total['squared_reference'],1e-300))
        total['rms']=math.sqrt(total['squared_error']/max(total['elements'],1))
    return dict(aggregate=aggregate,per_layer=per_layer,minimum_position=minimum_position,
                visual_included=include_visual,
                scope='post-RoPE KV at identical original positions; active text rows only')


def _events(bank):
    return sum(value['calls'] for value in bank.events.values())


def _prediction(native_logits, fused_logits, generate_side, eos):
    chosen=native_logits if generate_side=='native' else fused_logits
    target=int(chosen.argmax(-1).item())
    value=logit_comparison([fused_logits.float().cpu()],[native_logits.float().cpu()],[target],eos)
    result=value['per_token'][0]
    result['target_token_source']=generate_side+'_greedy'
    result['next_token_committed']=False
    return result


def run_session(model, processor, row, args, event_log):
    from p180_multiturn_core import attention_bridge,next_suffix
    from p184_working_set import attention
    import run_p175_modern_replication as oldbase
    started=time.perf_counter()
    eos=set(oldbase.stop_ids(model,processor))
    canonical=int(processor.tokenizer.eos_token_id)
    f={key:value.to(model.device,dtype=model.dtype if value.is_floating_point() else value.dtype)
       for key,value in features(processor,row,args.max_patches).items()}
    visual=torch.nonzero(f['input_ids'][0]==model.config.image_token_id).flatten()
    initial_position=int(f['input_ids'].shape[-1])
    input_hash=hashlib.sha256(f['input_ids'].cpu().numpy().tobytes()).hexdigest()
    # Both initial prefills are independent standard SDPA computations. The
    # first remains native throughout; only the second is split into a bank.
    with attention_bridge(model,native=True):
        native_out=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
        fused_out=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
    native_cache=native_out.past_key_values
    fused_cache=fused_out.past_key_values
    if native_cache is fused_cache:
        raise ValueError('Independent prefills unexpectedly shared one cache object')
    native_logits=native_out.logits[:,-1].detach().clone()
    fused_logits=fused_out.logits[:,-1].detach().clone()
    del native_out,fused_out,f
    bank=ProgressiveBank(model,fused_cache,visual,'full',args.legacy_scripts,args.kivi_source,
                         keep_reference=False)
    try:
        native_ptrs={(str(t.device),t.untyped_storage().data_ptr())
                     for item in native_cache.layers for t in (item.keys,item.values)}
        fused_ptrs={(str(t.device),t.untyped_storage().data_ptr())
                    for state in bank.states for t in (state.visual_k,state.visual_v,state.text_k,state.text_v)}
        if native_ptrs&fused_ptrs:
            raise ValueError('Native and fused cache tensor owners alias')
        bank.refine_layers=set();bank.reference_layers=set()
        initial_state=cache_difference(native_cache,bank,include_visual=True)
        logical=initial_position
        rows=[]
        turns=[]
        native_total_calls=fused_total_calls=0
        current_prediction=_prediction(native_logits,fused_logits,args.generate_side,eos)
        initial_record=dict(session_id=row['id'],stage='initial_native_prefill_repeat',turn=1,
                            logical_position=logical,prediction=current_prediction,
                            state=initial_state,
                            native_branch='standard SDPA and unmodified native DynamicCache',
                            fused_branch='second independent native prefill, before first fused append')
        rows.append(initial_record)
        # Records are emitted after a subsequent token commits, so the
        # next_token_committed flag cannot be silently stale in the JSONL.
        pending=initial_record
        call_number=0

        def flush_pending():
            nonlocal pending
            if pending is not None:
                event_log.write(json.dumps(pending)+'\n');event_log.flush()
                pending=None

        def paired_consume(ids,phase,turn,step,committed_by):
            nonlocal native_logits,fused_logits,logical,native_total_calls,fused_total_calls
            nonlocal current_prediction,pending,call_number
            start_position=logical
            if phase=='decode' and committed_by=='greedy':
                current_prediction['next_token_committed']=True
                current_prediction['committed_token_id']=int(ids[0,0])
            flush_pending()
            before_events=_events(bank)
            outputs={}
            order=['native','fused']
            if args.order=='fused_first' or (args.order=='alternate' and call_number%2):
                order.reverse()
            for side in order:
                if side=='native':
                    count_before=_events(bank)
                    with attention_bridge(model,native=True):
                        out,new_logical=consume(model,native_cache,ids,start_position)
                    if out.past_key_values is not native_cache:
                        raise ValueError('Native forward replaced its cache object')
                    if _events(bank)!=count_before:
                        raise AssertionError('Native SDPA accidentally invoked the fused bank')
                    native_total_calls+=1
                else:
                    bank.turn=turn;bank.phase=phase;bank.pass_kind='full_fused_parity'
                    with attention(model):
                        out,new_logical=consume(model,fused_cache,ids,start_position)
                    if out.past_key_values is not fused_cache:
                        raise ValueError('Fused forward replaced its cache object')
                    fused_total_calls+=1
                outputs[side]=out.logits[:,-1].detach().clone()
                if new_logical!=start_position+ids.shape[-1]:
                    raise ValueError('A branch changed token/position accounting')
                del out
            if _events(bank)<=before_events:
                raise AssertionError('Full fused branch never used the fused attention hook')
            logical=start_position+ids.shape[-1]
            native_logits,fused_logits=outputs['native'],outputs['fused']
            current_prediction=_prediction(native_logits,fused_logits,args.generate_side,eos)
            item=dict(session_id=row['id'],stage=phase,turn=turn,step=step,
                      input_token_ids=ids[0].cpu().tolist(),input_start_position=start_position,
                      logical_position=logical,branch_execution_order=order,
                      input_source=committed_by,prediction=current_prediction,
                      state=cache_difference(native_cache,bank),
                      newly_written_state=cache_difference(native_cache,bank,minimum_position=start_position),
                      native_fused_hook_calls=0,
                      fused_hook_calls=_events(bank)-before_events,
                      note='same canonical input IDs and positions; branch outputs may disagree')
            rows.append(item);pending=item;call_number+=1

        for number,question in enumerate(row['questions'][:args.turns],1):
            if number>1:
                paired_consume(next_suffix(processor,question).to(model.device),'text_prefill',number,-1,'frozen_question_suffix')
            answer=[]
            for step in range(args.max_tokens):
                source=native_logits if args.generate_side=='native' else fused_logits
                token=int(source.argmax(-1).item())
                answer.append(token)
                paired_consume(torch.tensor([[token]],device=model.device),'decode',number,step,'greedy')
                if token in eos:
                    break
            closed=not answer or answer[-1]!=canonical
            if closed:
                paired_consume(torch.tensor([[canonical]],device=model.device),'decode',number,
                               len(answer),'forced_chat_closure')
            value=dict(turn=number,token_ids=answer,
                       answer=processor.tokenizer.decode(answer,skip_special_tokens=True).strip(),
                       forced_chat_closure=closed,logical_end=logical)
            turns.append(value)
            print(json.dumps(dict(stage='native_parity',session=row['id'],turn=number,
                                  answer=value['answer'],events=len(rows),
                                  max_abs_logit_difference=max(r['prediction']['max_abs_logit_difference'] for r in rows))),flush=True)
        flush_pending()
        final_state=cache_difference(native_cache,bank,include_visual=True)
        _sync()
        prediction_rows=[r['prediction'] for r in rows]
        return dict(schema='trajkv.native_parity.v1',session_id=row['id'],
                    image_id=row.get('source_image_id',row.get('image_id')),split=row.get('split'),
                    initial_input_ids_sha256=input_hash,canonical_generation_side=args.generate_side,
                    turns=turns,events=len(rows),native_calls=native_total_calls,fused_calls=fused_total_calls,
                    empirical_logit_max_distribution=[r['max_abs_logit_difference'] for r in prediction_rows],
                    empirical_kl_distribution=[r['kl_reference_to_actual'] for r in prediction_rows],
                    max_abs_logit_difference=max(r['max_abs_logit_difference'] for r in prediction_rows),
                    argmax_disagreements=sum(r['argmax_changed'] for r in prediction_rows),
                    committed_prediction_argmax_disagreements=sum(r['argmax_changed'] for r in prediction_rows if r['next_token_committed']),
                    initial_state=initial_state,final_state=final_state,
                    branch_cache_objects_distinct=True,branch_cache_storages_distinct=True,
                    native_branch_fused_hook_calls=0,full_fused_read_events=bank.event_rows(),
                    full_fused_owned=bank.owned(),model_dtype=str(model.dtype),
                    numerical_policy='empirical distribution only; no fixed 1e-3 rejection threshold',
                    parity_scope='native SDPA versus original-position all-support full FP16 fused bank; fixed shared trajectory',
                    interpretation='instrument calibration; parity alone does not establish quantized quality or serving efficiency',
                    timing_scope='instrumented alternating branch calls; not a latency benchmark',
                    session_ms=1000*(time.perf_counter()-started),paid_calls=0,
                    reference_answers_visible_to_runner=False)
    finally:
        bank.close()
        del bank,native_cache,fused_cache
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','samples','legacy-scripts','kivi-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--turns',type=int,default=3)
    parser.add_argument('--max-tokens',type=int,default=24)
    parser.add_argument('--max-patches',type=int,default=4)
    parser.add_argument('--generate-side',choices=['native','fused'],default='native')
    parser.add_argument('--order',choices=['native_first','fused_first','alternate'],default='alternate')
    parser.add_argument('--dtype',choices=['float16','bfloat16'],default='float16')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.count,args.turns,args.max_tokens,args.max_patches)<1:
        parser.error('Counts, turns and token/patch caps must be positive')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    install_legacy(args.legacy_scripts)
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads(args.samples.read_text(encoding='utf-8'))[args.start:args.start+args.count]
    paths=[Path(__file__),Path(__file__).with_name('mechanism.py'),Path(__file__).with_name('bank.py'),
           Path(__file__).with_name('run_cycle.py'),args.samples]
    native_attention=ALL_ATTENTION_FUNCTIONS['sdpa']
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  selected_ids=[r['id'] for r in rows],torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                  sources={str(path):digest(path) for path in paths},
                  native_sdpa_function_module=getattr(native_attention,'__module__',None),
                  native_sdpa_function_name=getattr(native_attention,'__qualname__',None),
                  sdpa_enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                                    memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                    math=torch.backends.cuda.math_sdp_enabled()),
                  reference_answers_visible_to_runner=False,paid_calls=0)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,
                dtype=getattr(torch,args.dtype),device_map='cuda:0',attn_implementation='sdpa').eval()
    summary_path=args.output/'sessions.jsonl'
    seen=set()
    if args.resume and summary_path.exists():
        seen={json.loads(line)['session_id'] for line in summary_path.read_text(encoding='utf-8').splitlines() if line.strip()}
    count=0
    with torch.inference_mode(),(args.output/'records.jsonl').open('a',encoding='utf-8') as event_log, \
         summary_path.open('a',encoding='utf-8') as session_log:
        for row in rows:
            if row['id'] in seen:
                continue
            summary=run_session(model,processor,row,args,event_log)
            session_log.write(json.dumps(summary)+'\n');session_log.flush();count+=1
    (args.output/'complete.json').write_text(json.dumps(dict(sessions_added=count,paid_calls=0)),encoding='utf-8')


if __name__=='__main__':
    main()
