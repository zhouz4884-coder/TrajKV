"""Real M1 same-prefix TurboQuant diagnostic; never reads task references.

Native KV prototypes and native queries are intentionally retained for causal
controls. The resulting peak/time is NOT a compression or speed benchmark.
Each arm recomputes the entire question suffix and its own first answer token.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('BASELINE_DATA_ROOT', str(ROOT)))
sys.path.insert(0, str(ROOT / 'code'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qa-per-dataset',type=int,default=2)
    parser.add_argument('--max-new-tokens',type=int,default=16)
    args=parser.parse_args();job=json.loads(args.job.read_text(encoding='utf-8'))
    assert args.qa_per_dataset>0 and args.max_new_tokens>0
    assert not (set(job)&{'refs','references','answers','labels','scores'})
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':'REAL_M1_SAME_PREFIX_MULTI_ARM_DIAGNOSTIC_NO_REFERENCE_ANSWERS',
        'references_read':False,'quality_scored':False,'new_task_quality_score':False,
        'storage_or_speed_benchmark':False,'native_prototypes_retained_for_diagnostic':True,
        'max_new_tokens':args.max_new_tokens,'qa_per_dataset':args.qa_per_dataset,
        'started_unix':time.time(),'cases':[]}
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from dataset_registry import load_inputs
        from model_registry import get_model_spec
        from evaluate import model_identity
        from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_forward_suffix,m1_generate_cached,clone_m1_prefix_cache
        from attention_bridge import attention_context,language_attention_modules
        from cache_bridge import HybridVisualCache
        from ownership import cache_storage
        from methods.turboquant_adapter import TurboQuantAdapter
        from methods.turboquant_buffered_adapter import TurboQuantBufferedAdapter
        torch.set_grad_enabled(False);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
        torch.cuda.set_device(0)
        spec=get_model_spec('M1');assert job.get('model_id','M1')=='M1'
        report['model_identity']=model_identity(job,spec)
        processor=AutoProcessor.from_pretrained(job['model_path'],local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(job['model_path'],dtype=torch.float16,
                  device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
        assert all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False)
        _,modules=language_attention_modules(model)
        selected_layers=[0,7,14,21,27]
        arms=[('F',None,None),('T0_packed',0,2),('T0_torch_dequant',0,2),
              ('T128_packed',128,2),('T128_torch_dequant',128,2),('T3V4_R128_packed',128,4)]
        paths=job.get('diagnostic_inputs',[str(DATA_ROOT/'data/registry/v3/textvqa_p1_inputs.jsonl'),
                                               str(DATA_ROOT/'data/registry/v3/infovqa_p1_inputs.jsonl')])
        inputs=[]
        for path in paths:inputs.extend(load_inputs(path)[:args.qa_per_dataset])
        assert inputs and len({row['id'] for row in inputs})==len(inputs)
        report.update(expected_QA=len(inputs),expected_arms_per_QA=len(arms),
            input_files=[{'file':p,'sha256':sha(p)} for p in paths],model_revision=spec['revision'],
            hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
            selected_native_dump_layers=selected_layers,
            input_question_suffix=job.get('question_suffix','\nAnswer the question using a single word or phrase.'))
        native_dump=args.output/'native_QKV';native_dump.mkdir()
        def stats(x):
            x=x.float().flatten()
            return {'mean':float(x.mean()),'min':float(x.min()),'median':float(x.median()),'max':float(x.max())}
        def difference(value,reference):
            delta=value.float()-reference.float()
            return {'max_abs':float(delta.abs().max()),'mean_abs':float(delta.abs().mean()),
                    'relative_L2':float(delta.norm()/reference.float().norm().clamp_min(1e-20))}
        def qk(q,k,scale):
            b,hq,nq,d=q.shape;hkv=k.shape[1]
            return ((q.float().reshape(b,hkv,hq//hkv,nq,d)@k.float().unsqueeze(2).transpose(-1,-2)).reshape(b,hq,nq,-1))*scale
        def mechanism(owner,state,backend):
            q=state['q'].to('cuda');k=state['visual_k'].to('cuda');v=state['visual_v'].to('cuda')
            tk=state['text_keys'].to('cuda');tv=state['text_values'].to('cuda')
            vp=state['visual_positions'].to('cuda');tp=state['text_positions'].to('cuda');qp=state['query_positions'].to('cuda')
            scale=state['scaling'];kh,vh=owner.materialize(torch.float32)
            native_logits=qk(q,k,scale);dequant_logits=qk(q,kh,scale)
            centered=native_logits-native_logits.mean(-1,keepdim=True)
            approx_centered=dequant_logits-dequant_logits.mean(-1,keepdim=True)
            noise=dequant_logits-native_logits
            b,hq,nq,d=q.shape;hkv=k.shape[1]
            mask=torch.cat((vp,tp))[None,:]<=qp[:,None]
            def dense(vis_logits,vis_values):
                logits=torch.cat((vis_logits,qk(q,tk,scale)),-1).masked_fill(~mask[None,None],-torch.inf)
                p=torch.softmax(logits,-1)
                values=torch.cat((vis_values.float(),tv.float()),-2)
                out=(p.reshape(b,hkv,hq//hkv,nq,-1)@values.unsqueeze(2)).reshape(b,hq,nq,d)
                return out,p
            reference,p=dense(native_logits,v)
            k_only,pk=dense(dequant_logits,v);v_only,_=dense(native_logits,vh);both,pb=dense(dequant_logits,vh)
            actual=owner.attend(q,exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,scale=scale,backend=backend)
            cosine=torch.nn.functional.cosine_similarity
            mean_k=k.float().mean(-2,keepdim=True)
            result={'oracle_scope':'same native FP16 q/K/V; FP32 dense attention instrument, no whole-model quality inference',
                'K_cosine':stats(cosine(k.float(),kh.float(),dim=-1)),
                'K_norm_ratio':stats(kh.float().norm(dim=-1)/k.float().norm(dim=-1).clamp_min(1e-20)),
                'K_error':difference(kh,k),'V_cosine':stats(cosine(v.float(),vh.float(),dim=-1)),'V_error':difference(vh,v),
                'K_mean_vector_energy_fraction':float(mean_k.square().sum()*k.shape[-2]/k.float().square().sum()),
                'visual_logit_correlation':stats(cosine(centered,approx_centered,dim=-1)),
                'centered_logit_noise_to_signal':stats((noise-noise.mean(-1,keepdim=True)).norm(dim=-1)/centered.norm(dim=-1).clamp_min(1e-20)),
                'attention_error':{'K_only':difference(k_only,reference),'V_only':difference(v_only,reference),
                                   'both_dense':difference(both,reference),'actual_reader_vs_decoded_dense':difference(actual,both)},
                'visual_probability_mass':{'native':float(p[...,:k.shape[-2]].sum(-1).mean()),
                                           'quantized':float(pb[...,:k.shape[-2]].sum(-1).mean())}}
            return result
        for index,row in enumerate(inputs):
            case={'id':row['id'],'dataset':row['dataset'],'source_cluster_id':row['source_cluster_id'],
                  'original_image_sha256':row['original_image_sha256'],'arms':[],'native_QKV_dump':[]}
            prototype=native_cache=batch=prefix_output=capture=None
            try:
                assert sha(row['image_path'])==row['original_image_sha256']
                with Image.open(row['image_path']) as image:rgb=image.convert('RGB')
                batch=prepare_m1_inputs(processor,rgb,row['question']+report['input_question_suffix'],
                        image_token_id=spec['image_token_id'],max_patches=int(job.get('preprocessing',{}).get('max_patches',8)))
                case.update(visual_tokens=int(batch.visual_positions.numel()),compression_boundary=batch.boundary,
                            question_suffix_tokens=batch.suffix_length,input_ids_sha256=hashlib.sha256(batch.ids.cpu().numpy().tobytes()).hexdigest())
                assert batch.boundary==int(batch.visual_positions[-1])+1
                prefix_output=m1_prefix_forward(model,batch);prototype=prefix_output.past_key_values;del prefix_output
                native_cache=clone_m1_prefix_cache(prototype)
                with attention_context(model,capture_queries=True,window_size=batch.suffix_length) as capture:
                    native_output=m1_forward_suffix(model,batch,native_cache,logits_to_keep=batch.suffix_length)
                assert torch.isfinite(native_output.logits).all()
                native_logits=native_output.logits.detach().cpu();del native_output
                native_states={}
                full_positions=torch.arange(batch.full_length,device=batch.ids.device)
                text_mask=torch.ones(batch.full_length,dtype=torch.bool,device=batch.ids.device);text_mask[batch.visual_positions]=False
                for layer in selected_layers:
                    state={'q':capture['queries_by_layer'][layer].detach().cpu(),
                        'visual_k':prototype.layers[layer].keys.index_select(-2,batch.visual_positions).detach().cpu(),
                        'visual_v':prototype.layers[layer].values.index_select(-2,batch.visual_positions).detach().cpu(),
                        'text_keys':native_cache.layers[layer].keys[:,:,text_mask].detach().cpu(),
                        'text_values':native_cache.layers[layer].values[:,:,text_mask].detach().cpu(),
                        'visual_positions':batch.visual_positions.detach().cpu(),
                        'text_positions':full_positions[text_mask].detach().cpu(),
                        'query_positions':torch.arange(batch.boundary,batch.full_length),
                        'scaling':getattr(modules[layer],'scaling',128**-.5),'layer_idx':layer,'id':row['id'],
                        'original_image_sha256':row['original_image_sha256'],'scope':'NATIVE_FP16_PREFIX_AND_SUFFIX_Q_KV_NOT_COMPRESSED_BRANCH'}
                    native_states[layer]=state
                    path=native_dump/f'{index:03d}_layer{layer}.pt';torch.save(state,path)
                    case['native_QKV_dump'].append({'layer':layer,'file':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                                                   'download_policy':'remote data disk only; never upload original derived feature tensors'})
                capture=None;native_cache=None;gc.collect()
                for name,window,vbits in arms:
                    branch={'name':name,'status':'RUNNING'};cache=None;owners=None;old_read=None;context=None;start=time.perf_counter()
                    try:
                        cache=clone_m1_prefix_cache(prototype)
                        backend=None
                        if name!='F':
                            owners=[]
                            for layer,original in enumerate(cache.layers):
                                ctor=TurboQuantAdapter if window==0 else TurboQuantBufferedAdapter
                                kwargs={'value_bits':vbits,'seed':42+6*layer}
                                if window:kwargs['recent_window']=window
                                adapter=ctor(**kwargs)
                                owners.append(adapter.encode(original.keys.index_select(-2,batch.visual_positions),
                                    original.values.index_select(-2,batch.visual_positions),positions=batch.visual_positions,layer_idx=layer))
                            backend='torch_dequant' if 'torch_dequant' in name else 'upstream_triton_score'
                            cache=HybridVisualCache(cache,batch.visual_positions,owners,reader_backend=backend)
                            branch['native_same_q_single_layer_mechanism']={str(layer):mechanism(cache.layers[layer].visual_owner,native_states[layer],backend)
                                                                         for layer in selected_layers}
                            branch['persistent_method_cache_owner']=cache_storage(cache,batch.visual_positions)
                            old_read=cache.read;branch_q=[]
                            def observed_read(layer,q,*positional,**keyword):
                                if layer in selected_layers and q.shape[-2]==batch.suffix_length:
                                    branch_q.append({'layer':layer,'difference_to_native_suffix_q':difference(q,native_states[layer]['q'].to(q.device))})
                                return old_read(layer,q,*positional,**keyword)
                            cache.read=observed_read
                        context=attention_context(model,cache=cache) if name!='F' else None
                        if context:
                            with context:output=m1_forward_suffix(model,batch,cache,logits_to_keep=batch.suffix_length)
                        else:output=m1_forward_suffix(model,batch,cache,logits_to_keep=batch.suffix_length)
                        assert torch.isfinite(output.logits).all()
                        branch['suffix_logits_vs_native']=difference(output.logits.cpu(),native_logits)
                        branch['suffix_last_token_argmax_id']=int(output.logits[0,-1].argmax())
                        branch['native_suffix_last_token_argmax_id']=int(native_logits[0,-1].argmax())
                        if name!='F':branch['actual_suffix_q_drift']=branch_q;cache.read=old_read
                        del output;cache.crop(batch.boundary)
                        finite_calls=[]
                        def finite_hook(module,unused,result):
                            finite=bool(torch.isfinite(result.logits).all());finite_calls.append(finite)
                            if not finite:raise RuntimeError('NONFINITE_LOGITS in own compressed free generation')
                        handle=model.register_forward_hook(finite_hook)
                        try:
                            if name!='F':
                                with attention_context(model,cache=cache):generated=m1_generate_cached(model,batch,cache,max_new_tokens=args.max_new_tokens)
                            else:generated=m1_generate_cached(model,batch,cache,max_new_tokens=args.max_new_tokens)
                        finally:handle.remove()
                        ids=generated[0,batch.full_length:].cpu().tolist();assert ids
                        branch.update(status='PASS',raw_prediction=processor.decode(ids,skip_special_tokens=True),output_token_ids=ids,
                                      all_freegen_outer_logits_finite=bool(finite_calls) and all(finite_calls),actual_reader_backend=backend)
                        del generated
                    except Exception as exc:branch.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
                    finally:
                        torch.cuda.synchronize();branch['diagnostic_wall_ms']=(time.perf_counter()-start)*1000
                        branch['timing_or_peak_is_benchmark']=False;case['arms'].append(branch)
                        cache=owners=old_read=context=None;gc.collect();torch.cuda.empty_cache()
                    print(json.dumps({'id':row['id'],'arm':name,'status':branch['status']}),flush=True)
                case['status']='PASS' if all(arm['status']=='PASS' for arm in case['arms']) else 'COMPLETE_WITH_FAILURES'
            except Exception as exc:case.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
            finally:
                report['cases'].append(case)
                with (args.output/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(case,ensure_ascii=False)+'\n')
                prototype=native_cache=batch=capture=None;gc.collect();torch.cuda.empty_cache()
        report['status']='DIAGNOSTIC_RAW_COMPLETE' if all(case['status']=='PASS' for case in report['cases']) else 'DIAGNOSTIC_COMPLETE_WITH_FAILURES'
        report['source_sha256']={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in
            [Path(__file__),ROOT/'code/methods/turboquant_adapter.py',ROOT/'code/methods/turboquant_buffered_adapter.py',
             ROOT/'code/m1_protocol.py',ROOT/'code/cache_bridge.py',ROOT/'code/attention_bridge.py',ROOT/'code/evaluate.py']}
    except Exception as exc:report.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report['completed_unix']=time.time()
        with (args.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),'quality_scored':False}),flush=True)


if __name__=='__main__':main()
