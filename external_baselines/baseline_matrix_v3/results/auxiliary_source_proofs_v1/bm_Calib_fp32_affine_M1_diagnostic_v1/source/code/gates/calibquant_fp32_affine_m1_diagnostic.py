"""Two-source real-M1 precision-port diagnostic. Root schedules CUDA.

Native prototypes and code-reference temporaries are intentional instruments,
not storage/latency benchmark residents. Every arm computes its own full suffix
and free first token. No answers are loaded or scored.
"""
import argparse
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-new-tokens',type=int,default=16)
    args=parser.parse_args();job=json.loads(args.job.read_text(encoding='utf-8'))
    assert not (set(job)&{'refs','references','answers','labels','scores'})
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':'REAL_M1_TWO_SOURCE_PRECISION_PORT_DIAGNOSTIC',
            'references_read':False,'quality_scored':False,'benchmark_memory_or_speed':False,
            'native_prototypes_retained':True,'cases':[],'started_unix':time.time()}
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from dataset_registry import load_inputs
        from model_registry import get_model_spec
        from evaluate import model_identity
        from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_forward_suffix,m1_generate_cached,clone_m1_prefix_cache
        from attention_bridge import attention_context
        from cache_bridge import HybridVisualCache
        from ownership import cache_storage
        from methods.calibquant_adapter import repeat_kv
        from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasAdapter
        from methods.calibquant_fp32_affine_adapter import CalibQuantFP32AffineAdapter
        torch.set_grad_enabled(False);torch.manual_seed(0);torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
        spec=get_model_spec('M1');assert job.get('model_id','M1')=='M1'
        report['model_identity']=model_identity(job,spec)
        processor=AutoProcessor.from_pretrained(job['model_path'],local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(job['model_path'],dtype=torch.float16,
             device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
        assert all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False)
        inputs=job.get('diagnostic_inputs',[str(DATA_ROOT/'data/registry/v3/textvqa_p1_inputs.jsonl'),
                                            str(DATA_ROOT/'data/registry/v3/infovqa_p1_inputs.jsonl')])
        rows=[load_inputs(path)[0] for path in inputs]
        assert len(rows)==2 and len({row['id'] for row in rows})==2
        selected=[0,14,27]
        suffix=job.get('question_suffix','\nAnswer the question using a single word or phrase.')
        report.update(expected_QA=2,expected_arms_per_QA=5,max_new_tokens=args.max_new_tokens,
            model_revision=spec['revision'],hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
            new_port_manifest=CalibQuantFP32AffineAdapter().manifest(),input_hashes={path:sha(path) for path in inputs})
        def difference(actual,expected):
            delta=actual.double()-expected.double()
            return {'max_abs':float(delta.abs().max()),'relative_L2':float(delta.norm()/expected.double().norm().clamp_min(1e-20))}
        def coded_math(owner,q):
            g=q.shape[1]//owner.k_packed.shape[1]
            raw=owner._raw_qk(q)
            ks,kz=repeat_kv(owner.k_scale,g).double(),repeat_kv(owner.k_zero,g).double()
            ref=raw.double()/128**.5-(ks*kz*q.double()).sum(-1,keepdim=True)/128**.5
            lo,hi=ref.amin(-1,keepdim=True),ref.amax(-1,keepdim=True)
            ref=((ref-lo)/(hi-lo+1e-8))*(hi-3-lo)+lo
            actual=owner.read(q)
            assert actual.dtype==torch.float32 and torch.isfinite(actual).all()
            p=torch.softmax(actual,-1).half()
            raw_v=owner._raw_weighted_codes(p)
            vs,vz=repeat_kv(owner.v_scale,g).double(),repeat_kv(owner.v_zero,g).double()
            ideal_v=raw_v.double()*vs-(vs*vz)*p.double().sum(-1,keepdim=True)
            output=owner.weighted_value(p)
            assert output.dtype==torch.float16 and torch.isfinite(output).all()
            return {'scope':'FP64 coded affine/calibration conditioned on identical actual hard-FP16 original-kernel boundary; native teacher q; visual-only normalized weights for WV operator control',
                    'read_dtype':str(actual.dtype),'weighted_return_dtype':str(output.dtype),
                    'logits_vs_boundary_conditioned_FP64':difference(actual,ref),
                    'weighted_vs_boundary_conditioned_FP64':difference(output,ideal_v),
                    'weighted_vs_boundary_conditioned_FP64_Half':difference(output,ideal_v.half())}
        for row in rows:
            case={'id':row['id'],'dataset':row['dataset'],'original_image_sha256':row['original_image_sha256'],'arms':[]}
            prototype=capture=batch=None
            try:
                assert sha(row['image_path'])==row['original_image_sha256']
                with Image.open(row['image_path']) as image:rgb=image.convert('RGB')
                batch=prepare_m1_inputs(processor,rgb,row['question']+suffix,image_token_id=spec['image_token_id'],
                      max_patches=int(job.get('preprocessing',{}).get('max_patches',8)))
                case.update(visual_tokens=int(batch.visual_positions.numel()),compression_boundary=batch.boundary,
                            suffix_tokens=batch.suffix_length,full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest())
                prefix=m1_prefix_forward(model,batch);prototype=prefix.past_key_values;del prefix
                teacher=clone_m1_prefix_cache(prototype)
                with attention_context(model,capture_queries=True,window_size=batch.suffix_length) as capture:
                    native=m1_forward_suffix(model,batch,teacher,logits_to_keep=batch.suffix_length)
                assert torch.isfinite(native.logits).all()
                native_logits=native.logits.detach().cpu();del native,teacher
                native_q={i:capture['queries_by_layer'][i] for i in selected};capture=None
                for name,bits,variant in [('F',None,None),('C1_bias32',1,'bias'),('C1_affine32',1,'affine'),
                                          ('C2_bias32',2,'bias'),('C2_affine32',2,'affine')]:
                    arm={'name':name,'status':'RUNNING'};cache=owners=None;start=time.perf_counter()
                    try:
                        cache=clone_m1_prefix_cache(prototype)
                        if bits is not None:
                            adapter=(CalibQuantFP32AffineAdapter(bits,'upstream_triton') if variant=='affine' else CalibQuantFP32BiasAdapter(bits,'upstream_triton'))
                            owners=[];equal=True
                            for i,layer in enumerate(cache.layers):
                                k=layer.keys.index_select(-2,batch.visual_positions);v=layer.values.index_select(-2,batch.visual_positions)
                                owner=adapter.encode(k,v);owners.append(owner)
                                if variant=='affine':
                                    old=CalibQuantFP32BiasAdapter(bits,'upstream_triton').encode(k,v)
                                    equal &= all(torch.equal(tensor,owner.owner_tensors()[key]) for key,tensor in old.owner_tensors().items())
                                    equal &= old.byte_report()==owner.byte_report()
                            assert equal
                            cache=HybridVisualCache(cache,batch.visual_positions,owners,reader_backend='upstream_triton')
                            arm['persistent_owner']=cache_storage(cache,batch.visual_positions)
                            arm['encoder_and_owner_equal_old_port']=equal
                            if variant=='affine':
                                arm['coded_math_native_q']={str(i):coded_math(cache.layers[i].visual_owner,native_q[i]) for i in selected}
                            with attention_context(model,cache=cache):output=m1_forward_suffix(model,batch,cache,logits_to_keep=batch.suffix_length)
                        else:output=m1_forward_suffix(model,batch,cache,logits_to_keep=batch.suffix_length)
                        assert torch.isfinite(output.logits).all()
                        arm['suffix_logits_vs_native']=difference(output.logits.cpu(),native_logits)
                        arm['suffix_last_argmax_id']=int(output.logits[0,-1].argmax())
                        del output;cache.crop(batch.boundary)
                        finite=[]
                        def check(module,unused,value):
                            good=bool(torch.isfinite(value.logits).all());finite.append(good)
                            if not good:raise RuntimeError('NONFINITE_LOGITS in own free generation')
                        handle=model.register_forward_hook(check)
                        try:
                            if bits is not None:
                                with attention_context(model,cache=cache):generated=m1_generate_cached(model,batch,cache,max_new_tokens=args.max_new_tokens)
                            else:generated=m1_generate_cached(model,batch,cache,max_new_tokens=args.max_new_tokens)
                        finally:handle.remove()
                        ids=generated[0,batch.full_length:].cpu().tolist();assert ids
                        arm.update(status='PASS',output_token_ids=ids,raw_prediction=processor.decode(ids,skip_special_tokens=True),
                                   all_freegen_logits_finite=bool(finite) and all(finite))
                        del generated
                    except Exception as exc:arm.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
                    finally:
                        torch.cuda.synchronize();arm['instrumented_diagnostic_wall_ms']=(time.perf_counter()-start)*1000
                        case['arms'].append(arm);cache=owners=None;gc.collect();torch.cuda.empty_cache()
                    print(json.dumps({'id':row['id'],'arm':name,'status':arm['status']}),flush=True)
                case['status']='PASS' if all(arm['status']=='PASS' for arm in case['arms']) else 'COMPLETE_WITH_FAILURES'
            except Exception as exc:case.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
            finally:
                report['cases'].append(case)
                with (args.output/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(case,ensure_ascii=False)+'\n')
                prototype=capture=batch=None;gc.collect();torch.cuda.empty_cache()
        report['status']='DIAGNOSTIC_RAW_COMPLETE' if all(case['status']=='PASS' for case in report['cases']) else 'DIAGNOSTIC_COMPLETE_WITH_FAILURES'
        report['source_sha256']={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in
             [Path(__file__),ROOT/'code/methods/calibquant_fp32_affine_adapter.py',ROOT/'code/methods/calibquant_fp32_bias_adapter.py',
              ROOT/'code/cache_bridge.py',ROOT/'code/attention_bridge.py',ROOT/'code/m1_protocol.py']}
    except Exception as exc:report.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report['completed_unix']=time.time()
        with (args.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),'quality_scored':False}),flush=True)


if __name__=='__main__':main()
