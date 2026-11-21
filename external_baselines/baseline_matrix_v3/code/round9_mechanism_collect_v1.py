"""Round9 mechanism-only same-state collection; no fitting or task quality.

One native visual prefill per image. A cloned FP16 prefix generates the short
teacher continuation; the actual packed R7 student replays that token history.
Local samples retain true visual tensors for mechanism diagnosis only. This collector
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


CAPTURE_LAYERS=[0,6,13,20,27]
FORK_PARENT_SOURCE_SHA256='2d709806122c8127a766747ef5d29821fb22f389238f8448f0dfd25c363a61df'

def validate_config(config):
    assert config.get('purpose')=='mechanism_only' and config.get('count')==32
    assert config.get('capture_layers')==CAPTURE_LAYERS
    assert config.get('model_id')=='M1' and config.get('preprocessing',{}).get('max_patches')==8
    assert config.get('variant_source',{}).get('name')=='r7_joint_k1_gt_cond'
    assert not set(config)&{'answers','references','reference_path','labels','scores'}
    return config

def collect(config,output,config_path=None):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from round9_inputs_v1 import load_round9_inputs
    from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_generate_cached,m1_forward_suffix,clone_m1_prefix_cache
    from attention_bridge import attention_context,language_attention_modules
    from cache_bridge import HybridVisualCache
    from r7_frozen_pilot_v2 import frozen_variant,share_original_visual_position_map,SUFFIX
    from r7_legacy_bridge_v2 import build_r7_owners
    from model_registry import get_model_spec
    from evaluate import model_identity
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
    validate_config(config)
    assert not set(config)&{'answers','references','reference_path','labels','scores'}
    assert sha(config['inputs'])==config['inputs_sha256']
    rows=load_round9_inputs(config['inputs'],expected_sha256=config['inputs_sha256'])
    assert len(rows)==32 and len({r['original_image_sha256'] for r in rows})==32
    assert sum(r['dataset']=='infovqa' for r in rows)==16 and sum(r['dataset']=='textvqa' for r in rows)==16
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
    locked_assets={variant[k]:snapshot[k] for k in ('direction_book_path','metadata_factor_path','round7_policy_path')}
    locked_assets[config['variant_source']['path']]=config['variant_source']['sha256']
    if config.get('student_theta_path'):
        assert sha(config['student_theta_path'])==config['student_theta_sha256']
        snapshot['student_theta_sha256']=config['student_theta_sha256']
        snapshot['student_value_mode']='V-RE'
        locked_assets[config['student_theta_path']]=config['student_theta_sha256']
    source_files={str(ROOT/'code'/name):sha(ROOT/'code'/name) for name in ('round9_mechanism_collect_v1.py','round9_inputs_v1.py','m1_protocol.py','attention_bridge.py','cache_bridge.py','r7_frozen_pilot_v2.py','r7_legacy_bridge_v1.py','r7_legacy_bridge_v2.py','ownership.py','model_registry.py','evaluate.py')}
    input_sha=sha(config['inputs']);legacy_sources={}
    config_sha=sha(config_path) if config_path else None
    if config.get('student_theta_path'):source_files[str(ROOT/'code/round9_value_bank_v1.py')]=sha(ROOT/'code/round9_value_bank_v1.py')
    write_new(output/'manifest.json',{'schema':'trajkv.round9.mechanism_collection.v1','purpose':'mechanism_only','config':config,
        'config_sha256':config_sha,'input_sha256':input_sha,'source_sha256':source_files,
        'fork_parent_source_sha256':FORK_PARENT_SOURCE_SHA256,'capture_layers':CAPTURE_LAYERS,
        'teacher_generation_cap':3,'student_teacher_forced_decode_cap':2,
        'student_snapshot':snapshot,'expected_sources':32,'references_read':False,
        'mechanism_only_source_tensors':True,'training_eligible':False,'model_identity':identity,'actual_GPU':torch.cuda.get_device_name(),
        'free_generation_quality_claim':False,'created_unix':time.time()})
    completed=[]
    for number,row in enumerate(rows):
        begin=time.perf_counter()
        assert sha(row['image_path'])==row['original_image_sha256']
        with Image.open(row['image_path']) as im:image=im.convert('RGB')
        batch=prepare_m1_inputs(processor,image,row['question']+SUFFIX,max_patches=8)
        native=m1_prefix_forward(model,batch).past_key_values
        vp=batch.visual_positions
        source_layers={i:{'visual_k':native.layers[i].keys.index_select(2,vp).detach().cpu().contiguous(),
                          'visual_v':native.layers[i].values.index_select(2,vp).detach().cpu().contiguous()} for i in CAPTURE_LAYERS}
        teacher_cache=clone_m1_prefix_cache(native)
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
        for index in CAPTURE_LAYERS:
            owner=owners[index];layer=source_layers[index]
            assert factors['key_gain_mode']=='centered', 'Explicit pre-factor mean needed for another K mode'
            layer['visual_k_rotated']=wht(layer['visual_k'].to('cuda').float()-owner.state.direction_mean.float()).cpu()
            layer['r7_fields']={name:value.detach().cpu().contiguous().clone()
                for name,value in vars(owner.state).items() if isinstance(value,torch.Tensor) and
                name not in ('visual_pos','text_k','text_v','text_pos')}
            layer['r7_fields'].setdefault('depth_residual_code',torch.empty((0,int(vp.numel()),4),dtype=torch.int32))
            layer['r7_fields'].setdefault('depth_residual_alpha',torch.empty((0,int(vp.numel())),dtype=torch.float16))
        book=owners[0].shared['book'].detach().cpu().clone()
        student=HybridVisualCache(native,vp,owners,consume_native=True)
        share_original_visual_position_map(student)
        events=[];phase={'name':'suffix_last','step':0}
        for index in CAPTURE_LAYERS:
            owner=owners[index];original=owner.attend
            def wrapper(q,*,_index=index,_original=original,**kwargs):
                actual=_original(q,**kwargs)
                assert bool(torch.isfinite(actual).all())
                q_last=q[...,-1:,:].contiguous()
                events.append({'schema':'trajkv.round9.mechanism_sample.v1','purpose':'mechanism_only',
                    'source_id':row['source_id'],'id':row['id'],'dataset':row['dataset'],
                    'image_sha256':row['original_image_sha256'],'layer_idx':_index,
                    'phase':phase['name'],'step':phase['step'],'student_snapshot':snapshot,
                    'q':q_last.detach().cpu().clone(),'q_rotated':wht(q,half=True)[...,-1:,:].detach().cpu().contiguous().clone(),
                    'exact_keys':kwargs['exact_keys'].detach().cpu().clone(),
                    'exact_values':kwargs['exact_values'].detach().cpu().clone(),
                    'exact_positions':kwargs['exact_positions'].detach().cpu().clone(),
                    'query_positions':kwargs['query_positions'][-1:].detach().cpu().clone(),
                    'scaling':128**-.5,'packed_output_heads':actual[...,-1:,:].detach().cpu().contiguous().clone(),
                    'packed_input_query_tokens':int(q.shape[-2]),'capture_slice':'last actual query/output',
                    'returned_model_value_unchanged':True})
                return actual
            owner.attend=wrapper
        with attention_context(model,cache=student):
            result=m1_forward_suffix(model,batch,student)
            assert bool(torch.isfinite(result.logits).all())
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
                assert bool(torch.isfinite(result.logits).all())
        assert {e['layer_idx'] for e in events}==set(CAPTURE_LAYERS)
        for path,info in metadata['legacy_sources'].items():
            assert path not in legacy_sources or legacy_sources[path]==info['sha256']
            legacy_sources[path]=info['sha256']
        case={'schema':'trajkv.round9.mechanism_case.v1','purpose':'mechanism_only','training_eligible':False,'source':row,'visual_positions':vp.cpu(),
              'teacher_generated_token_ids':teacher_ids,'layers':source_layers,'events':events,
              'book':book,'student_snapshot':snapshot,'original_factors':factors,
              'selected_stages':metadata['selected_stages'],'capture_layers':CAPTURE_LAYERS,
              'input_contract':{'input_ids':batch.ids.cpu(),'position_ids':batch.positions.cpu(),
                  'boundary':batch.boundary,'image_sha256':row['original_image_sha256']},
              'free_generation_quality_eligible':False,'local_mechanism_only':True}
        path=output/'shards'/f'{number:03d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(case,path)
        item={'source_id':row['source_id'],'id':row['id'],'file':str(path),'sha256':sha(path),
              'bytes':path.stat().st_size,'events':len(events),'visual_tokens':int(vp.numel()),
              'wall_s':time.perf_counter()-begin}
        completed.append(item)
        write_new(output/'receipts'/f'{number:03d}.json',item)
        print(json.dumps(dict(phase='collected',completed=len(completed),expected=32,**item)),flush=True)
        case=events=owners=student=native=result=source_layers=batch=image=raw=book=owner=layer=None
        gc.collect();torch.cuda.empty_cache()
    wos=output/'wo_private.pt'
    torch.save({i:modules[i].o_proj.weight.detach().cpu().clone() for i in CAPTURE_LAYERS},wos)
    assert all(sha(path)==value for path,value in source_files.items())
    assert sha(config['inputs'])==input_sha
    assert all(sha(path)==value for path,value in locked_assets.items())
    assert all(sha(path)==value for path,value in legacy_sources.items())
    if config_path:assert sha(config_path)==config_sha
    complete={'status':'PASS_MECHANISM_32_REAL_MODEL_SOURCES','purpose':'mechanism_only','capture_layers':CAPTURE_LAYERS,'training_eligible':False,'expected_sources':32,'sources':completed,
              'wo_file':str(wos),'wo_sha256':sha(wos),'private_model_weights_must_not_be_published':True,
              'actual_GPU':True,'all_capture_layers_present':True,'manifest_sha256':sha(output/'manifest.json'),'source_sha256':source_files,'executed_legacy_source_sha256':legacy_sources,'references_read':False,
              'mechanism_only_no_quality_score':True,'task_quality_eligible':False,'student_snapshot':snapshot}
    write_new(output/'complete.json',complete)
    return complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    print(json.dumps(collect(config,a.output,a.config)),flush=True)


if __name__=='__main__':main()
