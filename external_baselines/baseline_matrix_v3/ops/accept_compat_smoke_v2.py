"""CPU acceptance of real own-model compatibility raw/source evidence.

V1 and its original controls remain unchanged. This version checks the fixed
input identities and recorded original-position/GQA/storage geometry. It is
not a task-quality threshold or a new GPU/numerical-equivalence experiment.
"""
import argparse
import json
import math
from pathlib import Path,PurePosixPath
import re
import sys
from remote_control import ROOT
sys.path.insert(0,str(ROOT/'code'));sys.dont_write_bytecode=True
from prediction_cache import SourceRun,selected_inputs,sha,normalize_behavior,input_identity,input_key,require
from model_registry import get_model_spec

DTYPE_BYTES={'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.float64':8,
             'torch.int64':8,'torch.int32':4,'torch.int16':2,'torch.int8':1,'torch.uint8':1,'torch.bool':1}


def storage_table(report):
    tensors=report['tensors'];require(bool(tensors),'Missing physical storage table')
    for name,value in tensors.items():
        require(value['dtype'] in DTYPE_BYTES,'Unknown recorded dtype: '+value['dtype'])
        logical=math.prod(value['shape'])*DTYPE_BYTES[value['dtype']]
        require(logical==value['logical_bytes'] and value['storage_bytes']>=logical,'Storage capacity/logical-shape mismatch: '+name)
        require(value['charged_bytes'] in (0,value['storage_bytes']),'Invalid unique-storage charge: '+name)
        if value['shared_with'] is not None:
            require(value['charged_bytes']==0 and value['shared_with'] in tensors,'Unbound shared-storage name: '+name)
            first=tensors[value['shared_with']]
            require(first['device']==value['device'] and first['storage_bytes']==value['storage_bytes'],'Shared allocation capacity/device mismatch')
    require(sum(v['charged_bytes'] for v in tensors.values())==report['total_bytes'],'Physical charges do not sum to allocation total')
    return tensors


def validate_record(row,expected,spec):
    require(input_identity(row)==input_identity(expected) and input_key(row)==input_key(expected),'Locked question/image/source input identity differs')
    require(row['question_unknown_prefix_verified'] is True and row['original_full_prefix_released'] is True,'Prefix information/ownership protocol not verified')
    calls=row['outer_forward_calls'];require(calls and all(c['raw_logits_finite'] is True for c in calls),'Nonfinite or missing outer-forward evidence')
    require(any(c['phase']=='prefix' for c in calls) and any(c['phase']=='generation' for c in calls),'Prefix or compressed-suffix generation evidence missing')
    ids=row['input_token_ids'];ninput=row['input_tokens'];vp=row['visual_positions'];nv=row['actual_visual_token_count'];boundary=row['compression_boundary']
    require(isinstance(ids,list) and len(ids)==ninput and all(type(i) is int for i in ids),'Input token list/count mismatch')
    require(isinstance(vp,list) and vp and all(type(i) is int for i in vp) and vp==sorted(set(vp)),'Visual positions not original sorted unique indices')
    require(len(vp)==nv and all(0<=i<ninput for i in vp),'Visual positions/count/range mismatch')
    require(vp==[i for i,token in enumerate(ids) if token==spec['image_token_id']],'Visual indices do not match native image placeholders')
    require(boundary==vp[-1]+1 and 0<boundary<ninput and row['question_suffix_tokens']==ninput-boundary,'Native compression boundary or full suffix length differs')
    shape=[1,ninput] if spec['model_id']=='M1' else [3,1,ninput]
    require(row['native_position_ids_shape']==shape,'Native position axes/length incompatible with this model')
    require(re.fullmatch(r'[0-9a-f]{64}',row['native_position_ids_sha256']) is not None,'Native position fingerprint missing')
    layers,hkv,dim=spec['layers'],spec['kv_heads'],spec['head_dim']
    require(spec['query_heads']%hkv==0,'Model has nonintegral GQA')
    denominator=layers*2*hkv*nv*dim*2
    require(row['full_kv_byte_denominator']==denominator,'Visual FP16 denominator differs from locked model geometry')
    native=row['native_prefix_owner'];native_tensors=storage_table(native)
    kv_names={f'layers.{i}.{kind}' for i in range(layers) for kind in ('K','V')}
    require({k for k in native_tensors if k.startswith('layers.')}==kv_names,'Native prefix KV layer/count mismatch')
    for name in kv_names:
        value=native_tensors[name]
        require(value['shape']==[1,hkv,boundary,dim] and value['dtype']=='torch.float16' and value['device'].startswith('cuda:'),'Native KV GQA/head/position/dtype geometry mismatch: '+name)
        require(value['storage_bytes']==value['logical_bytes']==value['charged_bytes'],'Native K/V allocation aliases or hides backing capacity')
    require(native['stored_visual_payload_bytes']==denominator,'Native visual byte denominator evidence differs')
    owner=row['owner_before_suffix'];tensors=storage_table(owner)
    require(len(row['method_owner_manifests'])==layers,'Method visual-owner layer count differs')
    textlen=boundary-nv;textbytes=layers*2*hkv*textlen*dim*2
    require(owner['text_kv_bytes']==row['text_kv_bytes']==textbytes,'Exact text byte count differs from native prefix')
    expected_text_names={f'layers.{i}.text_{kind}' for i in range(layers) for kind in ('keys','values')}
    require({k for k in tensors if k.endswith(('.text_keys','.text_values'))}==expected_text_names,'Exact text layer count differs')
    for name in expected_text_names:
        value=tensors[name]
        require(value['shape']==[1,hkv,textlen,dim] and value['dtype']=='torch.float16' and value['device'].startswith('cuda:'),'Exact text KV lost native GQA/position geometry')
    for i in range(layers):
        prefix=f'layers.{i}.visual.';visual={k:v for k,v in tensors.items() if k.startswith(prefix)}
        require(visual,'Missing visual owner layer')
        pairs=[('k_code','v_code'),('k_packed','v_packed')]
        found=False
        for k,v in pairs:
            if prefix+k in tensors and prefix+v in tensors:
                found=True
                for leaf in (k,v):
                    value=tensors[prefix+leaf]
                    require(value['shape'][:2]==[1,hkv] and value['device'].startswith('cuda:'),'Packed K/V head geometry/device differs')
        require(found,'No physical packed K/V payload pair in owner')
    if spec['model_id'] in ('M2','M3'):
        delta=tensors.get('common.rope_deltas')
        require(delta is not None and delta['shape']==[1,1] and delta['dtype']=='torch.int64' and delta['device'].startswith('cuda:') and delta['logical_bytes']==8,'Native Qwen RoPE delta not retained/accounted')
    complete=sum(v['charged_bytes'] for k,v in tensors.items() if '.visual.' in k or k=='original_visual_positions' or k.startswith('common.'))
    require(complete==owner['complete_visual_owner_bytes']==owner['stored_visual_KV_bytes']==row['complete_visual_owner_bytes']==row['stored_visual_KV_bytes']>0,'Complete visual owner does not match actual unique physical charges')
    require(owner['total_bytes']==complete+owner['text_kv_bytes']+owner['text_positions_bytes']+owner['other_persistent_metadata_bytes'],'Owner total/component balance differs')
    bits=16*complete/denominator
    require(math.isfinite(bits) and math.isclose(bits,row['actual_effective_visual_bits'],rel_tol=0,abs_tol=1e-10),'Complete effective bits inconsistent')
    require(row['peak_allocated_bytes']>0 and row['peak_reserved_bytes']>=row['peak_allocated_bytes'],'Invalid actual CUDA peak accounting')
    require(row['output_token_ids'] and len(row['output_token_ids'])<=64,'Invalid generated output token coverage')
    return {'id':row['id'],'PASS':True,'actual_visual_tokens':nv,'actual_effective_visual_bits':bits,
        'complete_visual_owner_bytes':complete,'native_layers':layers,'native_Hq':spec['query_heads'],'native_Hkv':hkv,'native_D':dim,
        'native_position_ids_shape':shape,'compression_boundary':boundary,'full_question_suffix_tokens':ninput-boundary,
        'finite_outer_forward_calls':len(calls),'peak_allocated_bytes':row['peak_allocated_bytes'],
        'peak_reserved_bytes':row['peak_reserved_bytes'],'output_tokens':len(row['output_token_ids']),'quality_threshold_applied':False}


def accept(raw_dir,target_config):
    source=SourceRun(raw_dir);target=json.loads(Path(target_config).read_text(encoding='utf-8'));spec=get_model_spec(target['model_id']);config=source.manifest['config']
    require(source.manifest['model_spec']==spec,'Actual source model geometry/revision differs from target model registry')
    require(normalize_behavior(config,spec)==normalize_behavior(target,spec),'Full source/target inference behavior differs')
    for key in ('model_path','model_source_lock'):require(config.get(key)==target.get(key),'Target checkpoint location/source lock differs: '+key)
    require(source.complete['status']=='PASS_RAW_COMPLETE' and source.complete['expected_QA']==source.complete['recorded_QA']==2 and source.complete['counts']=={'PASS':2},'Actual two-QA smoke is incomplete or contains failure')
    inputname=PurePosixPath(target['inputs'].replace('\\','/')).name
    require(inputname=='textvqa_p2_inputs.jsonl','This compatibility gate expects fixed first two standard TextVQA P2 inputs')
    expected=selected_inputs(target,ROOT/'configs/datasets/v3'/inputname)[:2]
    require([e['id'] for e in source.entries]==[r['id'] for r in expected],'Fixed first two question IDs differ')
    require(config['max_new_tokens']==target['max_new_tokens']==64,'Declared generation limit must remain64')
    reports=[validate_record(json.loads(source.line(entry)),row,spec) for entry,row in zip(source.entries,expected)]
    return {'schema':'baseline_matrix.actual_model_compatibility_gate.v2','status':'PASS_ACTUAL_MODEL_2QA_SOURCE_FINITE_OWNER_INPUT_GATE',
        'model':spec['model_id'],'method':config['method'],'source_run':source.source_descriptor(),
        'target_config':str(target_config),'target_config_sha256':sha(target_config),'gate_source_sha256':sha(Path(__file__)),
        'reports':reports,'actual_GPU_QA':2,'no_reference_answers_read':True,'new_GPU_experiment_by_gate':False,
        'quality_threshold_or_numerical_equivalence_claim':False,
        'positions_limit':'Actual shape/indices/boundary and recorded SHA verified; exact native 3-axis values rely on locked audited protocol and prior native bridge controls, not newly recomputed by this CPU gate'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw_dir',type=Path);parser.add_argument('--target-config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=accept(args.raw_dir,args.target_config);body=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():require(args.output.read_bytes()==body,'Frozen gate receipt differs')
    else:
        with args.output.open('xb') as stream:stream.write(body)
    print(json.dumps({'status':value['status'],'model':value['model'],'actual_QA':2,'bits':[r['actual_effective_visual_bits'] for r in value['reports']]}))
