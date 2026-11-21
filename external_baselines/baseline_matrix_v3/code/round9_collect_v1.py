"""Real-model Round9 calibration collection; diagnostic sources stay on data disk.

One native visual prefill per image. A cloned FP16 prefix generates the short
teacher continuation; the actual packed R7 student replays that token history.
Local samples retain true visual tensors for calibration only. This collector
makes no serving-residency or free-generation-quality claim.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def collect(config,output):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from round9_inputs_v1 import load_round9_inputs
    from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_generate_cached,m1_forward_suffix,clone_m1_prefix_cache
    from attention_bridge import attention_context,language_attention_modules
    from cache_bridge import HybridVisualCache
    from r7_frozen_pilot_v2 import frozen_variant,share_original_visual_position_map,SUFFIX
    from r7_legacy_bridge_v2 import build_r7_owners
    from round9_query_capture_v1 import native_suffix_queries
    from model_registry import get_model_spec
    from evaluate import model_identity
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
    assert config.get('purpose')=='calibration' and config['count']==32
    assert not set(config)&{'answers','references','reference_path','labels','scores'}
    assert sha(config['inputs'])==config['inputs_sha256']
    rows=load_round9_inputs(config['inputs'],expected_sha256=config['inputs_sha256'])
    assert len(rows)==32 and len({r['original_image_sha256'] for r in rows})==32
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_grad_enabled(False);torch.manual_seed(9)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity=model_identity(config,get_model_spec('M1'))
    processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(config['model_path'],dtype=torch.float16,
        device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
    model.requires_grad_(False)
    _,modules=language_attention_modules(model);assert len(modules)==28
    variant=frozen_variant(config)
    factors=json.loads(Path(variant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
    snapshot={k:sha(variant[k]) for k in ('direction_book_path','metadata_factor_path','round7_policy_path')}
    snapshot['collector_source_sha256']=sha(__file__)
    if config.get('student_theta_path'):
        assert sha(config['student_theta_path'])==config['student_theta_sha256']
        snapshot['student_theta_sha256']=config['student_theta_sha256']
        snapshot['student_value_mode']='V-RE'
    write_new(output/'manifest.json',{'schema':'trajkv.round9.calibration_collection.v1','config':config,
        'student_snapshot':snapshot,'expected_sources':32,'references_read':False,
        'training_only_source_tensors':True,'model_identity':identity,'actual_GPU':torch.cuda.get_device_name(),
        'free_generation_quality_claim':False,'created_unix':time.time()})
    completed=[]
    for number,row in enumerate(rows):
        begin=time.perf_counter()
        assert sha(row['image_path'])==row['original_image_sha256']
        with Image.open(row['image_path']) as im:image=im.convert('RGB')
        batch=prepare_m1_inputs(processor,image,row['question']+SUFFIX,max_patches=8)
        native=m1_prefix_forward(model,batch).past_key_values
        vp=batch.visual_positions
        source_layers=[{'visual_k':layer.keys.index_select(2,vp).detach().cpu().contiguous(),
                        'visual_v':layer.values.index_select(2,vp).detach().cpu().contiguous()} for layer in native.layers]
        teacher_cache=clone_m1_prefix_cache(native)
        with native_suffix_queries(model,vp) as native_trace:
            teacher_ids=m1_generate_cached(model,batch,teacher_cache,max_new_tokens=3)[0,batch.full_length:].detach().cpu().tolist()
        teacher_cache=None;gc.collect()
        kwargs=dict(legacy_a2=config['legacy_a2'],legacy_experiments=config['legacy_experiments'],
            legacy_scripts=config['legacy_scripts'],kivi_source=config['kivi_source'],
            expected_asset_sha256=config['expected_asset_sha256'])
        if config.get('student_theta_path'):
            from round9_value_bank_v1 import build_round9_value_owners
            owners,metadata=build_round9_value_owners(native,vp,variant,mode='V-RE',
                theta_path=config['student_theta_path'],theta_sha256=config['student_theta_sha256'],**kwargs)
        else:owners,metadata=build_r7_owners(native,vp,variant,**kwargs)
        from fast_wht import wht
        from quantizers import binary_centroids
        for index,owner in enumerate(owners):
            layer=source_layers[index]
            assert factors['key_gain_mode']=='centered', 'Explicit pre-factor mean needed for another K mode'
            layer['visual_k_rotated']=wht(layer['visual_k'].to('cuda').float()-owner.state.direction_mean.float()).cpu()
            layer['r7_fields']={name:value.detach().cpu().contiguous().clone()
                for name,value in vars(owner.state).items() if isinstance(value,torch.Tensor) and
                name not in ('visual_pos','text_k','text_v','text_pos')}
            layer['r7_fields'].setdefault('depth_residual_code',torch.empty((0,int(vp.numel()),4),dtype=torch.int32))
            layer['r7_fields'].setdefault('depth_residual_alpha',torch.empty((0,int(vp.numel())),dtype=torch.float16))
            raw=binary_centroids(layer['visual_v'].to('cuda').transpose(2,3).contiguous(),group_size=int(vp.numel()))
            assert torch.equal(raw[0],owner.state.v_code)
            layer.update(raw_v_code=raw[0].cpu(),raw_v_scale=raw[1].cpu(),raw_v_min=raw[2].cpu())
        book=owners[0].shared['book'].detach().cpu().clone()
        student=HybridVisualCache(native,vp,owners,consume_native=True)
        share_original_visual_position_map(student)
        events=[];phase={'name':'suffix_last','step':0}
        for index,owner in enumerate(owners):
            original=owner.attend
            def wrapper(q,*,_index=index,_original=original,**kwargs):
                q_last=q[...,-1:,:].contiguous()
                native_event=native_trace['queries_by_layer'][_index][phase['step']]
                assert torch.equal(native_event['query_positions'],kwargs['query_positions'][-1:].cpu())
                events.append({'schema':'trajkv.round9.local_sample.v1','purpose':'calibration',
                    'source_id':row['source_id'],'id':row['id'],'dataset':row['dataset'],
                    'image_sha256':row['original_image_sha256'],'layer_idx':_index,
                    'phase':phase['name'],'step':phase['step'],'student_snapshot':snapshot,
                    'q':q_last.detach().cpu().clone(),'q_rotated':wht(q_last,half=True).detach().cpu().clone(),
                    'native_q':native_event['q'],
                    'native_q_rotated':wht(native_event['q'].to('cuda'),half=True).cpu(),
                    'native_exact_keys':native_event['exact_keys'],'native_exact_values':native_event['exact_values'],
                    'native_exact_positions':native_event['exact_positions'],
                    'native_query_positions':native_event['query_positions'],
                    'exact_keys':kwargs['exact_keys'].detach().cpu().clone(),
                    'exact_values':kwargs['exact_values'].detach().cpu().clone(),
                    'exact_positions':kwargs['exact_positions'].detach().cpu().clone(),
                    'query_positions':kwargs['query_positions'][-1:].detach().cpu().clone(),
                    'scaling':128**-.5})
                return _original(q,**kwargs)
            owner.attend=wrapper
        with attention_context(model,cache=student):
            result=m1_forward_suffix(model,batch,student)
            eos=model.generation_config.eos_token_id
            stop=set(eos if isinstance(eos,list) else [eos])
            for step,token in enumerate(teacher_ids[:2],1):
                if token in stop:break
                phase.update(name='teacher_forced_decode',step=step)
                pos=batch.full_length+step-1
                result=model(input_ids=torch.tensor([[token]],device='cuda'),
                    attention_mask=torch.ones((1,pos+1),dtype=batch.mask.dtype,device='cuda'),
                    position_ids=torch.tensor([[pos]],device='cuda'),past_key_values=student,
                    use_cache=True,return_dict=True,logits_to_keep=1)
        assert {e['layer_idx'] for e in events}==set(range(28))
        case={'schema':'trajkv.round9.calibration_case.v1','source':row,'visual_positions':vp.cpu(),
              'teacher_generated_token_ids':teacher_ids,'layers':source_layers,'events':events,
              'book':book,'student_snapshot':snapshot,'original_factors':factors,
              'selected_stages':metadata['selected_stages'],'all_28_layers_captured':True}
        path=output/'shards'/f'{number:03d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(case,path)
        item={'source_id':row['source_id'],'id':row['id'],'file':str(path),'sha256':sha(path),
              'bytes':path.stat().st_size,'events':len(events),'visual_tokens':int(vp.numel()),
              'wall_s':time.perf_counter()-begin}
        completed.append(item)
        write_new(output/'receipts'/f'{number:03d}.json',item)
        print(json.dumps(dict(phase='collected',completed=len(completed),expected=32,**item)),flush=True)
        case=events=owners=student=native=result=source_layers=batch=image=raw=book=owner=layer=native_trace=None
        gc.collect();torch.cuda.empty_cache()
    wos=output/'wo_private.pt'
    torch.save({i:m.o_proj.weight.detach().cpu().clone() for i,m in enumerate(modules)},wos)
    complete={'status':'PASS_CALIBRATION_32_REAL_MODEL_SOURCES','expected_sources':32,'sources':completed,
              'wo_file':str(wos),'wo_sha256':sha(wos),'private_model_weights_must_not_be_published':True,
              'actual_GPU':True,'all_28_layers_captured':True,'references_read':False,
              'calibration_only_no_quality_score':True,'student_snapshot':snapshot}
    write_new(output/'complete.json',complete)
    return complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    print(json.dumps(collect(config,a.output)),flush=True)


if __name__=='__main__':main()
