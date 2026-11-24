"""Round12 v2 input-sealed low-probe -> forced information action -> packed M1 answer.

Forced actions build the development counterfactual table, not an online gate.
Probe tokens, on-site re-prefill/encoding and actual peak bytes remain distinct.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc,hashlib,inspect,json,os,platform,sys,time,traceback,weakref
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import torch
from PIL import Image
from attention_bridge import attention_context,language_attention_modules
from cache_bridge import HybridVisualCache
from evaluate import GenerationObserver,observe_generation,model_identity,tensor_sha
from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_generate_cached
from model_registry import get_model_spec
from ownership import model_storage,cache_storage,assert_no_full_prefix_alias
from r7_legacy_bridge_v2 import build_r7_owners
from round9_runner_v1 import PROFILES,SUFFIX,PRIMARY_METRIC_IDS,FORBIDDEN,frozen_variant,share_original_visual_position_map
from round12_action_probe_v1 import run_low_probe
from round12_multiimage_v1 import prepare_multiimage_inputs,select_roi
from round12_inputs_v1 import load_round12_inputs

SCHEMA='trajkv.round12_action.raw.v2'
RUNTIME_FILES=('round12_action_runner_v2.py','round12_action_probe_v1.py','round12_multiimage_v1.py',
 'round12_inputs_v1.py','m1_protocol.py','cache_bridge.py','attention_bridge.py','ownership.py','evaluate.py',
 'model_registry.py','dataset_registry.py','round9_runner_v1.py','round9_inputs_v1.py',
 'r7_legacy_bridge_v1.py','r7_legacy_bridge_v2.py')

def require(value,message):
 if not value:raise ValueError('ROUND12_ACTION: '+message)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def canonical(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_new(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def validate_config(c):
 require(c.get('schema')==SCHEMA,'explicit Round12 action schema required')
 require(c.get('action_mode') in ('P','Z','Z-random'),'P/Z/Z-random required')
 require(c.get('model_id')=='M1' and c.get('seed',0)==0,'M1 seed0 required')
 require(c.get('max_new_tokens')==64 and c.get('question_suffix')==SUFFIX,'native suffix/cap64 required')
 require(c.get('preprocessing',{}).get('max_patches')==8,'original cap8 required')
 require(not(set(c)&FORBIDDEN),'no inference labels')
 require(c['variant_source']['name']=='r7_joint_k1_gt_cond','original cond43 low required')
 require(c['expected_asset_sha256']==PROFILES['r7_joint_k1_gt_cond']['assets'],'low assets differ')
 require(c['method']['family']=='trajkv_round12_action_research','action method family required')
 require(c['method']['information_condition']=='query_unknown_prefix_then_query_conditional_action','information condition differs')
 require(c.get('position_owner_layout')=='single_shared_visual_position_map','shared original visual positions required')
 require(c.get('input_role')=='round12_development' and c.get('cohort_kind')=='development','explicit reused development cohort required')
 require(c.get('evaluation_contract',{}).get('version')=='matrix_v3_metrics_v1','frozen matrix-v3 scoring contract required')
 require(sha(c['input_manifest_path'])==c['input_manifest_sha256'],'input manifest hash differs')
 require(type(c.get('start')) is int and type(c.get('count')) is int and c['start']>=0 and c['count']>0,'explicit slice required')
 return c

def variants_from_config(c):
 low=frozen_variant(c);h=deepcopy(c)
 h['variant_source']['name']='r7_budget_k2_cond';h['variant']=c['high_variant']
 h['expected_asset_sha256']=PROFILES['r7_budget_k2_cond']['assets']
 return low,frozen_variant(h)

def pack_native_prefix(model,batch,variant,c,observer=None):
 if observer is not None:observer.phase='prefix'
 begin=time.perf_counter();out=m1_prefix_forward(model,batch);torch.cuda.synchronize()
 native=out.past_key_values;out=None;prefill_ms=(time.perf_counter()-begin)*1000
 require(native.get_seq_length()==batch.boundary and len(native.layers)==28,'native prefix differs')
 vp=batch.visual_positions;report=cache_storage(native,vp)
 refs=[weakref.ref(t) for layer in native.layers for t in (layer.keys,layer.values)]
 old={(str(t.device),t.untyped_storage().data_ptr()) for layer in native.layers for t in (layer.keys,layer.values)}
 begin=time.perf_counter()
 owners,meta=build_r7_owners(native,vp,variant,legacy_a2=c['legacy_a2'],legacy_experiments=c['legacy_experiments'],
  legacy_scripts=c['legacy_scripts'],kivi_source=c['kivi_source'],expected_asset_sha256=PROFILES[variant['name']]['assets'])
 cache=HybridVisualCache(native,vp,owners,consume_native=True);owners=native=None
 sharing=share_original_visual_position_map(cache);gc.collect();torch.cuda.synchronize()
 require(not any(r() is not None for r in refs),'original FP16 prefix retained')
 assert_no_full_prefix_alias(cache,old)
 return cache,dict(prefill_ms=prefill_ms,encode_ms=(time.perf_counter()-begin)*1000,
  native_prefix_owner=report,position_owner_layout=sharing,r7_bridge=meta,
  original_full_prefix_released=True,original_FP16_visual_alias_retained=False)

def input_evidence(b):
 return dict(full_prompt=b.prompt,full_prompt_hash=hashlib.sha256(b.prompt.encode()).hexdigest(),
  input_tokens=b.full_length,input_token_ids=b.ids.detach().cpu().reshape(-1).tolist(),input_ids_sha256=tensor_sha(b.ids),
  question_suffix_sha256=tensor_sha(b.ids[:,b.boundary:]),compression_boundary=b.boundary,question_suffix_tokens=b.suffix_length,
  visual_positions=b.visual_positions.detach().cpu().tolist(),actual_visual_token_count=int(b.visual_positions.numel()),
  native_position_ids_shape=list(b.positions.shape),native_position_ids_sha256=tensor_sha(b.positions),
  native_position_ids=b.positions.detach().cpu().reshape(-1).tolist())

def verify_low_input_binding(processor,image,batch,row,c):
 manifest=json.loads(Path(c['input_manifest_path']).read_text(encoding='utf-8-sig'))
 bound=manifest['native_bindings'][row['id']]
 evidence=input_evidence(batch)
 fields=('full_prompt_hash','input_ids_sha256','native_position_ids_sha256','compression_boundary','actual_visual_token_count')
 for field in fields:require(evidence[field]==bound[field],'native low input binding differs: '+field)
 require(bound['original_image_sha256']==row['original_image_sha256'] and bound['raw_question_sha256']==row['question_sha256'],'source image/question binding differs')
 require(bound['model_commit']==get_model_spec('M1')['revision'] and bound['dtype']=='float16','native model/dtype binding differs')
 alternate_text='Describe the objects.' if row['question']=='What is visible in the image?' else 'What is visible in the image?'
 alternate=prepare_m1_inputs(processor,image,alternate_text+SUFFIX,image_token_id=get_model_spec('M1')['image_token_id'])
 require(alternate.boundary==batch.boundary and torch.equal(alternate.ids[:,:batch.boundary],batch.ids[:,:batch.boundary])
  and torch.equal(alternate.visual_positions,batch.visual_positions) and torch.equal(alternate.positions[:,:batch.boundary],batch.positions[:,:batch.boundary]),'original visual prefix depends on future question')
 for key in set(batch.inputs)|set(alternate.inputs):
  if key in ('input_ids','attention_mask','position_ids','token_type_ids'):continue
  require(key in batch.inputs and key in alternate.inputs,'future question changes native visual input keys')
  left,right=batch.inputs[key],alternate.inputs[key]
  require(torch.equal(left,right) if isinstance(left,torch.Tensor) else left==right,'future question changes native visual data')
 alternate=left=right=None
 return dict(reference_input_manifest_sha256=c['input_manifest_sha256'],native_input_identity_checked_fields=list(fields),
  same_model_and_source_binding=True,alternate_question_prefix_ids_positions_pixels_bit_exact=True,
  references_or_scores_read=False)


def release_pixel_inputs(b):
 b.inputs={'input_ids':b.ids,'attention_mask':b.mask};b.prefix_inputs={}
 b.positions=b.positions.detach().cpu();b.visual_positions=b.visual_positions.detach().cpu()

def run_action_row(model,processor,row,c,output_dir):
 spec=get_model_spec('M1');low,high=variants_from_config(c);mode=c['action_mode']
 started=time.perf_counter();torch.cuda.reset_peak_memory_stats();cache=batch=image=observer=None
 r={**row,'model_id':'M1','model_commit':spec['revision'],'model_arch':spec['architecture'],'dtype':'float16',
  'method':c['method']['id'],'method_contract':c['method'],'preprocessing':c['preprocessing'],
  'status':'RUNNING','official_score':None,'score_status':'NOT_READ_OR_COMPUTED_BY_RUNNER','action_mode':mode,
  'output_token_ids':None,'raw_prediction':None,'forced_development_counterfactual_action':True,
  'online_selector_deployment_claimed':False}
 try:
  require(sha(row['image_path'])==row['original_image_sha256'],'original image SHA differs')
  with Image.open(row['image_path']) as source:image=source.convert('RGB')
  r['original_dimensions']=list(image.size);rendered=row['question']+SUFFIX
  batch=prepare_m1_inputs(processor,image,rendered,image_token_id=spec['image_token_id'])
  original=input_evidence(batch);r['original_low_input']=original
  r['initial_native_input_binding']=verify_low_input_binding(processor,image,batch,row,c)
  observer=GenerationObserver(model,started,batch)
  with observe_generation(model,observer):
   cache,meta=pack_native_prefix(model,batch,low,c,observer);r['low_stage']=meta
   r['low_owner_before_probe']=cache_storage(cache,cache.visual_positions)
   release_pixel_inputs(batch);observer.phase='probe'
   probe=run_low_probe(model,processor,batch,cache,row['question'])
   require(observer.stream_ids==probe['generated_probe_tokens'],'probe stream differs')
   low_calls=observer.calls
  r['probe']=probe;r['initial_question_unknown_prefix_verified']=True;r['low_probe_text_KV_rolled_back_to_prefix']=True
  seed=hashlib.sha256(('TrajKV-R12-random-ROI-v1|'+row['id']).encode()).hexdigest()
  roi=select_roi(probe['spatial_weights'],image.size,original['actual_visual_token_count']//256,
   random_seed=seed if mode=='Z-random' else None)
  r['roi']=roi
  cache=batch=observer=None;gc.collect();torch.cuda.empty_cache();action_start=time.perf_counter()
  if mode=='P':
   batch=prepare_m1_inputs(processor,image,rendered,image_token_id=spec['image_token_id']);variant=high
   require(input_evidence(batch)==original,'P changed original input IDs/pixels contract')
   r.update(action_prefix_information_condition='same_query_unknown_image_prefix_rebuilt',question_unknown_prefix_verified=True,
    action_implementation='onsite_native_prefill_encode_high_replace_low')
  else:
   crop=image.crop(tuple(roi['bbox_xyxy']))
   batch,multi=prepare_multiimage_inputs(processor,image,crop,rendered,image_token_id=spec['image_token_id'])
   crop=None;variant=low;r['native_two_view_protocol']=multi
   r.update(action_prefix_information_condition='query_conditioned_ROI_then_native_complete_two_view_prefix',
    question_unknown_prefix_verified=False,action_implementation='complete_native_two_view_prefix_all_visual_packed_low')
  r.update(input_evidence(batch));r.update(raw_question=row['question'],rendered_question=rendered,question_suffix=SUFFIX,
   raw_question_sha256=hashlib.sha256(row['question'].encode()).hexdigest(),rendered_question_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
   chat_template_hash=canonical(processor.chat_template),host_cached_upgrade_bank_bytes_read=0,
   physical_host_to_device_bytes_measured=False)
  observer=GenerationObserver(model,started,batch)
  with observe_generation(model,observer):
   cache,meta=pack_native_prefix(model,batch,variant,c,observer);r.update(meta)
   r['action_prefill_encode_wall_ms']=(time.perf_counter()-action_start)*1000
   r['owner_before_suffix']=cache_storage(cache,cache.visual_positions);own=r['owner_before_suffix']
   for field in ('stored_visual_payload_bytes','coder_metadata_bytes','positions_bytes','encoder_constants_bytes','text_positions_bytes','text_kv_bytes'):r[field]=own[field]
   complete=int(own['complete_visual_owner_bytes']);denominator=28*2*4*int(batch.visual_positions.numel())*128*2
   r.update(stored_visual_KV_bytes=complete,complete_visual_owner_bytes=complete,full_kv_byte_denominator=denominator,
    actual_effective_visual_bits=16*complete/denominator,
    per_request_strict_budget={'1.3':160*complete<=13*denominator,'1.8':80*complete<=9*denominator})
   _answer(model,processor,batch,cache,row,c,output_dir,r,observer)
  r['outer_forward_calls']=[dict(stage='low_probe',**x) for x in low_calls]+[dict(stage='action',**x) for x in observer.calls]
  r['ttft_ms']=(observer.token_times[0]-started)*1000
  r['inter_token_ms']=[(b-a)*1000 for a,b in zip(observer.token_times,observer.token_times[1:])]
  r['question_prefill_ms']=next((x['wall_ms'] for x in observer.calls if x['phase']=='generation'),None)
  r['reference_scope']='initial_low_prefix_then_forced_information_action_and_full_question_suffix';r['status']='PASS'
 except Exception as exc:
  r.update(status='OOM_AT_DECLARED_INPUT' if 'out of memory' in str(exc).lower() else 'FAILED_RUN',
   error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),output_token_ids=None,raw_prediction=None)
 finally:
  r.update(wall_ms=(time.perf_counter()-started)*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
   peak_reserved_bytes=torch.cuda.max_memory_reserved(),end_allocated_bytes=torch.cuda.memory_allocated(),
   timing_scope='actual low probe plus onsite re-prefill/re-encode and answer; instrumented quality prototype')
  cache=batch=image=observer=None;gc.collect();torch.cuda.empty_cache()
 return r

def _answer(model,processor,batch,cache,row,c,output_dir,r,observer):
 first=[];events=[];expected_ids=tensor_sha(batch.ids[:,batch.boundary:]);expected_pos=tensor_sha(batch.positions[:,batch.boundary:])
 def before(module,args,kwargs):
  if events:return
  ids=kwargs.get('input_ids',args[0] if args else None);past=kwargs.get('past_key_values');positions=kwargs.get('position_ids')
  require(ids is not None and tensor_sha(ids)==expected_ids and past.get_seq_length()==batch.boundary,'first action forward is not full original suffix')
  require(positions is None or tensor_sha(positions)==expected_pos,'action positions changed')
  events.append(dict(input_tokens=int(ids.shape[-1]),input_ids_sha256=tensor_sha(ids),cache_tokens_before=int(past.get_seq_length()),
   native_first_logits_injected=False,position_ids_sha256=tensor_sha(positions) if positions is not None else None))
 def after(module,args,out):
  if not first:first.append(out.logits[0,-1].detach().cpu().contiguous().clone())
 release_pixel_inputs(batch);gc.collect();observer.phase='generation'
 pre=model.register_forward_pre_hook(before,with_kwargs=True);post=model.register_forward_hook(after);begin=time.perf_counter()
 r['actual_terminal_generation_invoked']=True
 try:
  with attention_context(model,cache=cache):generated=m1_generate_cached(model,batch,cache,max_new_tokens=64)
 finally:pre.remove();post.remove()
 torch.cuda.synchronize();r['final_generation_wall_ms']=(time.perf_counter()-begin)*1000
 require(torch.equal(generated[:,:batch.full_length],batch.ids),'action changed native prompt history')
 tokens=generated[0,batch.full_length:].detach().cpu().tolist()
 r['actual_terminal_generated_token_count']=len(tokens)
 r['terminal_generation_completed_before_audit']=True
 require(tokens and len(tokens)<=64 and observer.stream_ids==tokens and first,'native final token/stream evidence differs')
 logits=first[0];require(bool(torch.isfinite(logits).all()),'nonfinite final logits')
 lp=Path(output_dir)/'first_logits'/(hashlib.sha256(row['id'].encode()).hexdigest()[:24]+'.pt')
 lp.parent.mkdir(exist_ok=True);require(not lp.exists(),'evidence would overwrite')
 torch.save({'id':row['id'],'raw_last_position_logits':logits,'scope':'actual action suffix before HF logits processors'},lp)
 top=torch.topk(logits.float(),5)
 r.update(first_generation_call=events[0],first_logits=dict(file=str(lp),sha256=sha(lp),bytes=lp.stat().st_size,
  dtype=str(logits.dtype),vocabulary_size=logits.numel(),finite=True,argmax_id=int(logits.argmax()),
  top5_ids=top.indices.tolist(),top5_logits=top.values.tolist()),output_token_ids=tokens,output_tokens=len(tokens),
  raw_prediction=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
  raw_decode_with_special_tokens=processor.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False))
 r['owner_after_generation']=cache_storage(cache,cache.visual_positions)
 require(r['owner_after_generation']['complete_visual_owner_bytes']==r['complete_visual_owner_bytes'],'visual owner changed in generation')

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=validate_config(json.loads(a.config.read_text(encoding='utf-8-sig')))
 all_rows=load_round12_inputs(c['inputs'],c['input_manifest_path'],c['inputs_sha256'])
 require(len(all_rows)==c['input_count'],'input count differs')
 rows=all_rows[c['start']:c['start']+c['count']]
 require(len(rows)==c['count'] and len({x['id'] for x in rows})==len(rows),'slice incomplete/duplicate')
 require(len({r['dataset'] for r in rows})==1,'one domain per run')
 require(c['evaluation_contract']['primary_metric_id']==PRIMARY_METRIC_IDS[rows[0]['dataset']],'frozen metric differs')
 variants_from_config(c)
 sources={'code/'+name:sha(ROOT/'code'/name) for name in RUNTIME_FILES}
 require(c.get('expected_source_hashes')==sources,'complete configured runtime graph differs')
 a.output.mkdir(parents=True,exist_ok=False)
 manifest=dict(schema='baseline_matrix.raw_inference.v1',runner_schema=SCHEMA,config=c,config_sha256=sha(a.config),
  inputs_sha256=c['inputs_sha256'],expected_ids=[r['id'] for r in rows],expected_QA=len(rows),references_read=False,
  prediction_cache_used=False,source_hashes=sources,runtime_source_root=str(ROOT),runtime_executable=sys.executable,
  started_unix=time.time(),phase_is_confirmatory=False,forced_actions_for_counterfactual_development=True,
  online_selector_deployment_claimed=False)
 records=[];error=None;written=False
 try:
  import transformers
  from transformers import AutoProcessor,AutoModelForImageTextToText
  torch.set_grad_enabled(False);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6),'actual CUDA3090 SM86 required');torch.cuda.set_device(0)
  spec=get_model_spec('M1');identity=model_identity(c,spec)
  processor=AutoProcessor.from_pretrained(c['model_path'],local_files_only=True)
  model=AutoModelForImageTextToText.from_pretrained(c['model_path'],dtype=torch.float16,device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
  model.requires_grad_(False)
  require(all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False),'original full FP16 weights required')
  require(len(language_attention_modules(model)[1])==28,'whole M1 geometry differs')
  manifest.update(model_spec=spec,model_identity=identity,model_residency=model_storage(model),model_parameter_count=sum(p.numel() for p in model.parameters()),
   torch_version=torch.__version__,transformers_version=transformers.__version__,CUDA_version=torch.version.cuda,
   hardware=torch.cuda.get_device_name(),python_version=platform.python_version(),generation_native=model.generation_config.to_dict(),
   runtime_library_source_sha256={str(Path(inspect.getfile(t))):sha(inspect.getfile(t))
    for t in (type(model),type(processor),type(processor.image_processor),type(processor.tokenizer))})
  write_new(a.output/'manifest.json',manifest);written=True
  for row in rows:
   r=run_action_row(model,processor,row,c,a.output);records.append(r)
   with (a.output/'records.jsonl').open('a',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   write_new(a.output/'records'/f'{len(records)-1:06d}.json',r)
   print(json.dumps(dict(id=row['id'],status=r['status'],completed=len(records),expected=len(rows))),flush=True)
  require(all(sha(ROOT/path)==value for path,value in sources.items()),'runtime source changed')
  require(sha(c['inputs'])==c['inputs_sha256'],'input file changed')
  require(sha(c['input_manifest_path'])==c['input_manifest_sha256'],'input manifest changed')
  require(sha(a.config)==manifest['config_sha256'],'config changed during run')
 except Exception as exc:error=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
 if not written:manifest['initialization_error']=error;write_new(a.output/'manifest.json',manifest)
 done={r['id'] for r in records}
 for row in rows:
  if row['id'] in done:continue
  r={**row,'model_id':'M1','method':c['method']['id'],'status':'FAILED_RUN','output_token_ids':None,'raw_prediction':None,'official_score':None,'error':error}
  records.append(r)
  with (a.output/'records.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 counts={}
 for r in records:counts[r['status']]=counts.get(r['status'],0)+1
 complete=dict(status='PASS_RAW_COMPLETE' if error is None and counts.get('PASS')==len(rows) else 'COMPLETE_WITH_FAILURES',
  expected_QA=len(rows),recorded_QA=len(records),counts=counts,failed_QA_must_remain_in_primary_denominator=True,
  quality_scored=False,load_error=error,manifest_sha256=sha(a.output/'manifest.json'),records_jsonl_sha256=sha(a.output/'records.jsonl'),
  completed_unix=time.time(),actual_new_terminal_answers=sum(bool(r.get('terminal_generation_completed_before_audit')) for r in records),
  actual_successful_terminal_answers=sum(r.get('status')=='PASS' for r in records),
  actual_probe_tokens=sum(r.get('probe',{}).get('actual_probe_token_count',0) for r in records),
  forced_development_counterfactual_actions=True,online_selector_deployment_claimed=False,frozen_score_py_compatible_raw_schema=True)
 write_new(a.output/'complete.json',complete);print(json.dumps(complete),flush=True)
 return 0 if complete['status']=='PASS_RAW_COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
