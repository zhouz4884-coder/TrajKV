"""Independent M1 early/late visual-compression boundary diagnostic.

A is the primary image-prefix->compress->question-suffix protocol. B first
pays for native complete-question attention and native y1, keeps native text
KV, then compresses visual KV and continues from y1. B is a separate late,
query-known control and is never a same-information quantized main-table row.
"""
from __future__ import annotations
from copy import deepcopy
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
DATA_ROOT=Path(os.environ.get('BASELINE_DATA_ROOT',str(ROOT)))
sys.path.insert(0,str(ROOT/'code'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_greedy_scope(config):
    if int(getattr(config,'num_beams',1) or 1)!=1:
        raise ValueError('Boundary prototype supports native beam-one greedy only')
    for name in ('stop_strings','exponential_decay_length_penalty','forced_decoder_ids','watermarking_config'):
        if getattr(config,name,None) is not None:raise ValueError('Unsupported active generation setting: '+name)
    for name in ('token_healing','is_assistant'):
        if getattr(config,name,False):raise ValueError('Unsupported active generation setting: '+name)
    if getattr(config,'max_time',None) is not None:raise ValueError('max_time would reset between stages')


def continuation_config(original):
    """Keep full history; account for the native token in relative processors."""
    check_greedy_scope(original);config=deepcopy(original)
    if getattr(config,'min_new_tokens',None) is not None:
        config.min_new_tokens=max(0,int(config.min_new_tokens)-1)
    # First-generation-only suppression was already applied to native y1.
    config.begin_suppress_tokens=None
    return config


def eos_ids(config):
    value=getattr(config,'eos_token_id',None)
    if value is None:return set()
    if isinstance(value,int):return {value}
    return set(map(int,value))


def native_full_first(model,batch,max_new_tokens):
    """One native HF step with the ORIGINAL total cap and logits processors.

    A stopping criterion ends after y1; max_new_tokens is not set to1 because
    doing so could incorrectly activate a forced-EOS-at-max-length processor.
    """
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    check_greedy_scope(model.generation_config)
    if max_new_tokens<1:raise ValueError('Positive complete answer cap required')
    length=batch.full_length
    class StopAfterOne(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return torch.full((input_ids.shape[0],),input_ids.shape[-1]>=length+1,
                              dtype=torch.bool,device=input_ids.device)
    native_events=[]
    def before(module,unused,kwargs):
        ids=kwargs.get('input_ids');past=kwargs.get('past_key_values')
        native_events.append({'input_tokens':ids.shape[-1] if ids is not None else None,
                              'cache_tokens_before':int(past.get_seq_length()) if past is not None else 0})
    handle=model.register_forward_pre_hook(before,with_kwargs=True)
    try:
        result=model.generate(**batch.inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
            return_dict_in_generate=True,output_scores=True,output_logits=True,
            stopping_criteria=StoppingCriteriaList([StopAfterOne()]))
    finally:handle.remove()
    sequence=result.sequences
    if sequence.shape[-1]!=length+1:raise RuntimeError('Native first-step stopping contract violated')
    cache=result.past_key_values
    if cache is None or int(cache.get_seq_length())!=length:
        raise RuntimeError('Native cache must contain the prompt and exclude unprocessed native y1')
    first=int(sequence[0,-1]);score=result.scores[0]
    if int(score[0].argmax())!=first:raise RuntimeError('Native y1 is not processed-logit greedy argmax')
    raw=getattr(result,'logits',None)
    if raw and not bool(torch.isfinite(raw[0]).all()):raise RuntimeError('Nonfinite native first-token raw logits')
    info={'native_first_token_id':first,'first_token_source':'complete_prompt_native_attention',
          'processed_native_argmax_id':int(score[0].argmax()),
          'raw_native_argmax_id':int(raw[0][0].argmax()) if raw else None,
          'native_prompt_cache_length':int(cache.get_seq_length()),'original_total_answer_cap':max_new_tokens,
          'native_prefill_forward_events':native_events,
          'temporary_native_logits_released_before_compression':True}
    # Only the cache and scalar first-token audit survive. No native logits are
    # supplied to A or to any compressed continuation.
    del score,raw,sequence,result
    return cache,first,info


def continue_after_native_first(model,batch,cache,first,max_new_tokens,reader_context=None):
    """Generate y2.. using cache(P) and complete history P+y1; no y1 replay."""
    import torch
    if int(cache.get_seq_length())!=batch.full_length:
        raise RuntimeError('B cache must contain native full prompt and exclude y1')
    if first in eos_ids(model.generation_config) or max_new_tokens==1:
        return [first],{'continuation_forward_needed':False,'compressed_generation_forward_calls_expected':0,
                        'native_first_is_EOS':first in eos_ids(model.generation_config)}
    first_tensor=batch.ids.new_tensor([[first]])
    history=torch.cat((batch.ids,first_tensor),-1)
    mask=torch.cat((batch.mask,batch.mask.new_ones((batch.mask.shape[0],1))),-1)
    config=continuation_config(model.generation_config)
    context=reader_context if reader_context is not None else nullcontext()
    with context:
        sequence=model.generate(input_ids=history,attention_mask=mask,past_key_values=cache,
            generation_config=config,max_new_tokens=max_new_tokens-1,do_sample=False,use_cache=True)
    ids=sequence[0,batch.full_length:].detach().cpu().tolist()
    if not ids or ids[0]!=first or len(ids)>max_new_tokens:
        raise RuntimeError('B answer cap or native-first-token preservation violated')
    return ids,{'continuation_forward_needed':True,'native_first_is_EOS':False,
                'history_length_for_continuation':history.shape[-1],
                'cache_length_before_continuation':batch.full_length,
                'continuation_new_token_cap':max_new_tokens-1,
                'relative_min_new_tokens_remaining':getattr(config,'min_new_tokens',None)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--qa-per-dataset',type=int,default=1,choices=(1,4))
    parser.add_argument('--max-new-tokens',type=int,default=64)
    args=parser.parse_args();job=json.loads(args.job.read_text(encoding='utf-8'))
    assert not (set(job)&{'refs','references','answers','labels','scores'})
    args.output.mkdir(parents=True,exist_ok=False)
    report={'schema':'baseline_matrix.independent_execution_boundary_diagnostic.v1','status':'RUNNING',
        'references_read':False,'quality_scored':False,'primary_benchmark_changed':False,
        'same_information_condition':False,'B_first_token_native':True,'B_question_text_KV_native':True,
        'diagnostic_native_cache_prototypes_retained':True,'memory_or_speed_benchmark':False,
        'author_model_patch_reproduced_bitwise':False,'common_M1_FP16_numeric_ports':True,
        'cases':[],'started_unix':time.time()}
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from dataset_registry import load_inputs
        from model_registry import get_model_spec
        from evaluate import model_identity
        from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_generate_cached,clone_m1_prefix_cache
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context
        from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasAdapter
        from ownership import cache_storage
        torch.set_grad_enabled(False);torch.manual_seed(0);torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
        spec=get_model_spec('M1');assert job.get('model_id','M1')=='M1'
        report['model_identity']=model_identity(job,spec)
        processor=AutoProcessor.from_pretrained(job['model_path'],local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(job['model_path'],dtype=torch.float16,
             device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
        assert all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False)
        check_greedy_scope(model.generation_config)
        if not eos_ids(model.generation_config):raise ValueError('M1 prototype requires declared generation EOS IDs')
        paths=job.get('diagnostic_inputs',[str(DATA_ROOT/'data/registry/v3/textvqa_p1_inputs.jsonl'),
                                           str(DATA_ROOT/'data/registry/v3/infovqa_p1_inputs.jsonl')])
        rows=[row for path in paths for row in load_inputs(path)[:args.qa_per_dataset]]
        assert len(rows)==2*args.qa_per_dataset and len({row['id'] for row in rows})==len(rows)
        report.update(expected_QA=len(rows),expected_arms_per_QA=6,max_new_tokens=args.max_new_tokens,
            generation_native=model.generation_config.to_dict(),hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
            input_file_hashes={path:sha(path) for path in paths})
        suffix=job.get('question_suffix','\nAnswer the question using a single word or phrase.')
        for row in rows:
            case={'id':row['id'],'dataset':row['dataset'],'original_image_sha256':row['original_image_sha256'],'arms':[]}
            image_prefix=full_prompt=batch=None
            try:
                assert sha(row['image_path'])==row['original_image_sha256']
                with Image.open(row['image_path']) as image:rgb=image.convert('RGB')
                batch=prepare_m1_inputs(processor,rgb,row['question']+suffix,image_token_id=spec['image_token_id'],
                      max_patches=int(job.get('preprocessing',{}).get('max_patches',8)))
                case.update(visual_tokens=int(batch.visual_positions.numel()),image_compression_boundary=batch.boundary,
                            full_prompt_tokens=batch.full_length,question_suffix_tokens=batch.suffix_length,
                            full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest())
                output=m1_prefix_forward(model,batch);image_prefix=output.past_key_values;del output
                full_prompt,y1,native_first=native_full_first(model,batch,args.max_new_tokens)
                case['B_paid_native_prefill']=native_first
                case['visual_prefix_FP16_A_vs_B']=[{'layer':i,'K_max_abs':float((left.keys[:,:,:batch.boundary].float()-right.keys[:,:,:batch.boundary].float()).abs().max()),
                    'V_max_abs':float((left.values[:,:,:batch.boundary].float()-right.values[:,:,:batch.boundary].float()).abs().max())}
                    for i,(left,right) in enumerate(zip(image_prefix.layers,full_prompt.layers))]
                for condition,prototype in [('A_image_prefix_before_question',image_prefix),('B_native_question_and_first_then_compress',full_prompt)]:
                    for label,bits in [('F',None),('C1_bias32',1),('C2_bias32',2)]:
                        arm={'condition':condition,'method':label,'status':'RUNNING','first_token_native':bits is None or condition.startswith('B'),
                             'information_condition':'query_unknown' if condition.startswith('A') else 'query_known_late_control',
                             'not_primary_quantized_benchmark_row':condition.startswith('B')}
                        cache=owners=None
                        try:
                            cache=clone_m1_prefix_cache(prototype)
                            if bits is not None:
                                adapter=CalibQuantFP32BiasAdapter(bits,'upstream_triton')
                                owners=[adapter.encode(layer.keys.index_select(-2,batch.visual_positions),layer.values.index_select(-2,batch.visual_positions)) for layer in cache.layers]
                                cache=HybridVisualCache(cache,batch.visual_positions,owners,reader_backend='upstream_triton')
                            arm['owner_after_encoding_before_generation']=cache_storage(cache,batch.visual_positions)
                            events=[]
                            finite_events=[]
                            def observe(module,unused,kwargs):
                                ids=kwargs.get('input_ids');past=kwargs.get('past_key_values')
                                events.append({'input_tokens':ids.shape[-1] if ids is not None else None,
                                               'cache_tokens_before':int(past.get_seq_length()) if past is not None else 0})
                            handle=model.register_forward_pre_hook(observe,with_kwargs=True)
                            def finite_check(module,unused,value):
                                good=bool(torch.isfinite(value.logits).all());finite_events.append(good)
                                if not good:raise RuntimeError('NONFINITE_LOGITS during boundary-controlled generation')
                            finite_handle=model.register_forward_hook(finite_check)
                            try:
                                context=attention_context(model,cache=cache) if bits is not None else nullcontext()
                                if condition.startswith('A'):
                                    with context:sequence=m1_generate_cached(model,batch,cache,max_new_tokens=args.max_new_tokens)
                                    ids=sequence[0,batch.full_length:].cpu().tolist();del sequence
                                    control={'own_compressed_suffix_and_first':bits is not None,'native_first_token_reused':False}
                                    if not events or events[0]['input_tokens']!=batch.suffix_length:
                                        raise RuntimeError('A did not recompute the complete question suffix')
                                else:
                                    ids,control=continue_after_native_first(model,batch,cache,y1,args.max_new_tokens,context)
                                    if control['continuation_forward_needed'] and (not events or events[0]['input_tokens']!=1):
                                        raise RuntimeError('B did not continue with the single unprocessed native y1')
                            finally:handle.remove();finite_handle.remove()
                            assert ids and len(ids)<=args.max_new_tokens
                            arm.update(status='PASS',output_token_ids=ids,raw_prediction=processor.decode(ids,skip_special_tokens=True),
                                generation_control=control,model_forward_events=events,
                                all_generation_raw_logits_finite=all(finite_events),
                                answer_native_token_count=len(ids) if bits is None else (1 if condition.startswith('B') else 0),
                                answer_generated_from_compressed_state_token_count=(len(ids)-1 if condition.startswith('B') else len(ids)) if bits is not None else 0)
                        except Exception as exc:arm.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
                        finally:
                            case['arms'].append(arm);cache=owners=None;gc.collect();torch.cuda.empty_cache()
                        print(json.dumps({'id':row['id'],'condition':condition,'method':label,'status':arm['status']}),flush=True)
                case['status']='PASS' if all(arm['status']=='PASS' for arm in case['arms']) else 'COMPLETE_WITH_FAILURES'
            except Exception as exc:case.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
            finally:
                report['cases'].append(case)
                with (args.output/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(case,ensure_ascii=False)+'\n')
                image_prefix=full_prompt=batch=None;gc.collect();torch.cuda.empty_cache()
        report['status']='BOUNDARY_DIAGNOSTIC_RAW_COMPLETE' if all(case['status']=='PASS' for case in report['cases']) else 'BOUNDARY_DIAGNOSTIC_COMPLETE_WITH_FAILURES'
        report['source_sha256']={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in
            [Path(__file__),ROOT/'code/m1_protocol.py',ROOT/'code/cache_bridge.py',ROOT/'code/attention_bridge.py',
             ROOT/'code/methods/calibquant_fp32_bias_adapter.py',ROOT/'third_party/calibquant/patch_internvl.py']}
    except Exception as exc:report.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report['completed_unix']=time.time()
        with (args.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),'quality_scored':False}),flush=True)


if __name__=='__main__':main()
