"""Real Qwen FP16 logical-cache/native-generation gate; no task scores."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor
from PIL import Image
from attention_bridge import attention_context
from cache_bridge import HybridVisualCache
from model_registry import get_model_spec,qwen_language_model
from qwen_protocol import (prepare_qwen_inputs,qwen_generate_native,qwen_native_full_forward,
    qwen_prefix_forward,qwen_forward_suffix,qwen_generate_cached)

p=argparse.ArgumentParser()
p.add_argument('--model-job',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--indices',nargs='+',type=int,default=[0,5])
p.add_argument('--max-new-tokens',type=int,default=16)
a=p.parse_args()
assert not a.output.exists(),'Preserve existing gate evidence'
job=json.loads(a.model_job.read_text());spec=get_model_spec(job['model_id'])
report={'status':'RUNNING','scope':'REAL_QWEN_FP16_BRIDGE_NATIVE_GENERATION_NO_TASK_SCORE',
    'model_id':job['model_id'],'model_commit':spec['revision'],'dtype':'float16',
    'started_unix':time.time(),'argv':sys.argv,'python':sys.executable,'cases':[],
    'max_new_tokens':a.max_new_tokens,'source_hashes':{}}
for name in ['qwen_protocol.py','cache_bridge.py','attention_bridge.py','model_registry.py']:
 f=Path(__file__).resolve().parents[1]/name
 report['source_hashes'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
def normalized_positions(pos):
 if pos.ndim==2:return pos[None].expand(3,-1,-1)
 if pos.shape[0]==4:return pos[1:]
 if pos.shape[0]==1:return pos.expand(3,-1,-1)
 return pos
try:
 assert json.loads(Path(job['download_integrity_report']).read_text())['status']=='PASS'
 torch.set_grad_enabled(False);torch.manual_seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 processor=AutoProcessor.from_pretrained(job['model_path'],local_files_only=True,
   min_pixels=job['min_pixels'],max_pixels=job['max_pixels'])
 model=AutoModelForImageTextToText.from_pretrained(job['model_path'],dtype=torch.float16,
   device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
 report['hardware']=torch.cuda.get_device_name(0)
 images=json.loads(Path(job['image_manifest']).read_text())
 snapshots=[]
 def capture(module,args,kwargs):
  pos=kwargs.get('position_ids')
  deep=kwargs.get('deepstack_visual_embeds')
  snapshots.append({'position_ids':pos.detach().cpu() if pos is not None else None,
    'deepstack_shapes':[list(x.shape) for x in deep] if deep is not None else []})
 handle=qwen_language_model(model).register_forward_pre_hook(capture,with_kwargs=True)
 for index in a.indices:
  source=images[index];image=Image.open(source['image_path']).convert('RGB')
  batch=prepare_qwen_inputs(processor,image,source['question'],model=model,device='cuda:0')
  torch.cuda.reset_peak_memory_stats();snapshots.clear()
  native=qwen_generate_native(model,batch,max_new_tokens=a.max_new_tokens)
  native_ids=native[0,batch.full_length:].cpu().tolist()
  native_deepstack=snapshots[0]['deepstack_shapes']
  del native
  full=qwen_native_full_forward(model,batch)
  full_logits=full.logits[:,-1].float().cpu();del full
  snapshots.clear()
  with attention_context(model,capture_queries=True,window_size=64) as capture_state:
   prefix=qwen_prefix_forward(model,batch)
  prefix_snapshot=snapshots[0]
  if spec['deepstack_visual_indexes']:
   assert len(native_deepstack)==len(prefix_snapshot['deepstack_shapes'])==3,'native DeepStack lost at prefix'
  assert len(capture_state['queries_by_layer'])==spec['layers']
  assert all(n==batch.boundary for n in capture_state['captured_tokens_by_layer'].values())
  cache=HybridVisualCache(prefix.past_key_values,batch.visual_positions,consume_native=True)
  captured_shapes={str(k):list(v.shape) for k,v in capture_state['queries_by_layer'].items()}
  del capture_state,prefix
  gc.collect();torch.cuda.empty_cache();snapshots.clear()
  with attention_context(model,cache=cache):
   suffix=qwen_forward_suffix(model,batch,cache)
  max_error=float((suffix.logits[:,-1].float().cpu()-full_logits).abs().max())
  suffix_position=snapshots[0]['position_ids']
  assert torch.equal(normalized_positions(suffix_position),batch.positions[...,batch.boundary:].cpu())
  assert not snapshots[0]['deepstack_shapes'],'question suffix must not reinsert image features'
  del suffix,full_logits
  cache.crop(batch.boundary);snapshots.clear()
  with attention_context(model,cache=cache):
   generated=qwen_generate_cached(model,batch,cache,max_new_tokens=a.max_new_tokens)
  ids=generated[0,batch.full_length:].cpu().tolist()
  first_position=snapshots[0]['position_ids']
  assert torch.equal(normalized_positions(first_position),batch.positions[...,batch.boundary:].cpu()),'cached HF preparation changed suffix mRoPE'
  position_steps=[]
  for step,snap in enumerate(snapshots[1:]):
   expected=batch.positions[...,-1:]+step+1
   actual=normalized_positions(snap['position_ids'])
   assert torch.equal(actual,expected.cpu()),'cached decode did not use native text-position continuation'
   assert not snap['deepstack_shapes']
   position_steps.append(actual.tolist())
  row={'source_id':source['source_id'],'index':index,'original_dimensions':list(image.size),
   'visual_tokens':int(batch.visual_positions.numel()),'boundary':batch.boundary,
   'full_length':batch.full_length,'suffix_length':batch.suffix_length,
   'native_output_token_ids':native_ids,'bridge_output_token_ids':ids,
   'native_raw_prediction':processor.decode(native_ids,skip_special_tokens=True),
   'bridge_raw_prediction':processor.decode(ids,skip_special_tokens=True),
   'all_generated_tokens_equal':native_ids==ids,'suffix_logit_max_abs_error':max_error,
   'native_positions':batch.positions.cpu().tolist(),'native_rope_deltas':batch.rope_deltas.cpu().tolist(),
   'native_deepstack_shapes':native_deepstack,'prefix_deepstack_shapes':prefix_snapshot['deepstack_shapes'],
   'suffix_native_position_ids':normalized_positions(suffix_position).tolist(),
   'cached_generation_suffix_positions':normalized_positions(first_position).tolist(),
   'cached_decode_positions':position_steps,'captured_question_free_query_shapes':captured_shapes,
   'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
   'status':'PASS' if native_ids==ids else 'FAILED_NATIVE_TOKEN_EQUIVALENCE'}
  report['cases'].append(row)
  print(json.dumps({k:row[k] for k in ['index','visual_tokens','status','suffix_logit_max_abs_error']}),flush=True)
  del generated,cache,batch,image,snapshots[:]
  gc.collect();torch.cuda.empty_cache()
 handle.remove()
 assert all(x['all_generated_tokens_equal'] for x in report['cases']),'FP16 bridge differs from native greedy generation'
 report.update(status='PASS',completed_unix=time.time())
except Exception:
 report.update(status='FAILED_RUN',traceback=traceback.format_exc(),completed_unix=time.time())
 raise
finally:
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2)
