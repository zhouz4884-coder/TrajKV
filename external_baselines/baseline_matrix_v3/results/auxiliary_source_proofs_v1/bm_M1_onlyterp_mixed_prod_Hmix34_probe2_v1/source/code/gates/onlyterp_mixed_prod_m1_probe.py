"""Real M1 diagnostic driver for the new source-IP port; no registry mutation.

Each arm replays the complete question suffix and generates its own first
token. Native prototypes are retained for controls, so this is not a process
memory/speed benchmark or a cacheable formal raw-evaluation producer.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024**2),b''):h.update(data)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    assert not(set(cfg)&{'refs','references','answers','labels','scores'})
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','schema':'baseline_matrix.real_source_ip_diagnostic.v1','references_read':False,
        'quality_scored':False,'formal_prediction_cache_producer':False,'storage_or_speed_benchmark':False,
        'native_prototypes_retained':True,'config_sha256':sha(args.config),'config':cfg,'cases':[],
        'started_unix':time.time(),'producer_source_sha256':sha(Path(__file__))}
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from dataset_registry import load_inputs
        from model_registry import get_model_spec
        from evaluate import model_identity  # read-only identity check; no ALLOWLIST modification
        from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_forward_suffix,m1_generate_cached,clone_m1_prefix_cache
        from cache_bridge import HybridVisualCache
        from attention_bridge import attention_context
        from ownership import cache_storage
        from methods.turboquant_onlyterp_mixed_prod_adapter import OnlyTerpMixedProdAdapter,READER
        torch.set_grad_enabled(False);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
        torch.cuda.set_device(0);spec=get_model_spec('M1');assert cfg['model_id']=='M1'
        report['model_identity']=model_identity(cfg,spec)
        processor=AutoProcessor.from_pretrained(cfg['model_path'],local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(cfg['model_path'],dtype=torch.float16,
            device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
        assert all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False)
        rows=load_inputs(cfg['inputs'])[:int(cfg.get('count',2))]
        assert len(rows)==int(cfg.get('count',2))
        report.update(expected_QA=len(rows),hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
            CUDA=torch.version.cuda,model_revision=spec['revision'],input_file_sha256=sha(cfg['inputs']))
        max_tokens=int(cfg.get('max_new_tokens',64));suffix=cfg.get('question_suffix','\nAnswer the question using a single word or phrase.')
        modes=cfg.get('value_read_modes',['source_cache_mse','full_prod_decode'])
        assert set(modes)<= {'source_cache_mse','full_prod_decode'} and modes
        for row in rows:
            case={'id':row['id'],'dataset':row['dataset'],'original_image_sha256':row['original_image_sha256'],'arms':[]}
            try:
                assert sha(row['image_path'])==row['original_image_sha256']
                with Image.open(row['image_path']) as image:
                    batch=prepare_m1_inputs(processor,image.convert('RGB'),row['question']+suffix,
                        image_token_id=spec['image_token_id'],max_patches=int(cfg.get('preprocessing',{}).get('max_patches',8)))
                native=m1_prefix_forward(model,batch).past_key_values;vp=batch.visual_positions
                assert len(native.layers)==spec['layers']
                adapter=OnlyTerpMixedProdAdapter(**cfg.get('constructor_args',{}))
                owners=[adapter.encode(layer.keys.index_select(-2,vp),layer.values.index_select(-2,vp),positions=vp,layer_idx=i)
                        for i,layer in enumerate(native.layers)]
                case.update(actual_visual_token_count=vp.numel(),prefix_length=batch.boundary,full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest(),
                    adapter_manifest=adapter.manifest(),own_first_token_for_every_arm=True)
                def generate(cache,context):
                    observed={'calls':0,'all_logits_finite':True}
                    def hook(module,inputs,output):
                        if getattr(output,'logits',None) is not None:
                            observed['calls']+=1
                            if not torch.isfinite(output.logits).all():observed['all_logits_finite']=False;raise RuntimeError('Nonfinite generated logits')
                    handle=model.register_forward_hook(hook)
                    try:
                        with context:ids=m1_generate_cached(model,batch,cache,max_new_tokens=max_tokens)
                    finally:handle.remove()
                    tokens=ids[0,batch.full_length:].detach().cpu().tolist();assert tokens
                    return {'raw_prediction':processor.decode(tokens,skip_special_tokens=True),'output_token_ids':tokens,**observed}
                # Separate clones for suffix diagnostics and generation: no
                # question KV or first token from the reference enters an arm.
                f_logits=m1_forward_suffix(model,batch,clone_m1_prefix_cache(native)).logits
                native_last=f_logits[:,-1].detach().float();assert torch.isfinite(native_last).all()
                from contextlib import nullcontext
                f=generate(clone_m1_prefix_cache(native),nullcontext())
                case['arms'].append({'name':'F_native_split','status':'PASS',**f})
                for mode in modes:
                    record={'name':'OnlyTerp_mixed_prod_'+mode,'status':'RUNNING','V_read_mode':mode}
                    try:
                        selected=[replace(owner,value_read_mode=mode) for owner in owners]
                        first=HybridVisualCache(clone_m1_prefix_cache(native),vp,selected,reader_backend=READER)
                        with attention_context(model,cache=first):logits=m1_forward_suffix(model,batch,first).logits
                        last=logits[:,-1].detach().float();assert torch.isfinite(last).all()
                        record['last_suffix_logit_vs_native']={'max_abs':(last-native_last).abs().max().item(),
                            'relative_L2':((last-native_last).norm()/native_last.norm().clamp_min(1e-12)).item(),
                            'argmax_id':last.argmax(-1).item(),'native_argmax_id':native_last.argmax(-1).item()}
                        first=None;logits=None;gc.collect()
                        fresh=HybridVisualCache(clone_m1_prefix_cache(native),vp,selected,reader_backend=READER)
                        accounting=cache_storage(fresh,vp)
                        denominator=spec['layers']*2*spec['kv_heads']*vp.numel()*spec['head_dim']*2
                        record.update(complete_visual_owner_bytes=accounting['complete_visual_owner_bytes'],
                            actual_effective_visual_bits=16*accounting['complete_visual_owner_bytes']/denominator,
                            encoder_constants_bytes=accounting['encoder_constants_bytes'],positions_bytes=accounting['positions_bytes'],
                            coder_metadata_bytes=accounting['coder_metadata_bytes'],
                            process_peak_not_a_compression_benchmark=True)
                        record.update(generate(fresh,attention_context(model,cache=fresh)));record['status']='PASS'
                        fresh=None;selected=None
                    except Exception as exc:
                        record.update(status='FAILED_DIAGNOSTIC_ARM',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
                    case['arms'].append(record)
                case['status']='PASS' if all(a['status']=='PASS' for a in case['arms']) else 'COMPLETE_WITH_ARM_FAILURES'
                owners=None;native=None;batch=None;f_logits=None;gc.collect();torch.cuda.empty_cache()
            except Exception as exc:
                case.update(status='FAILED_DIAGNOSTIC_QA',error=str(exc),traceback=traceback.format_exc())
            report['cases'].append(case)
            with (args.output/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(case,ensure_ascii=False)+'\n')
            print(json.dumps({'id':row['id'],'status':case['status'],'completed':len(report['cases'])}),flush=True)
        report['status']='DIAGNOSTIC_RAW_COMPLETE' if all(c['status']=='PASS' for c in report['cases']) else 'DIAGNOSTIC_COMPLETE_WITH_FAILURES'
    except Exception as exc:
        report.update(status='FAILED_DIAGNOSTIC',error=str(exc),traceback=traceback.format_exc())
    report['finished_unix']=time.time()
    with (args.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}),flush=True)


if __name__=='__main__':main()
