"""CPU-only native processor readiness; never model/hardware/task evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_registry import get_model_spec

p=argparse.ArgumentParser()
p.add_argument('--jobs',nargs='+',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
from PIL import Image
from transformers import AutoConfig,AutoProcessor

rows=[]
for path in a.jobs:
 job=json.loads(path.read_text());spec=get_model_spec(job['model_id'])
 config=AutoConfig.from_pretrained(job['model_path'],local_files_only=True)
 text=config.text_config
 assert (text.num_hidden_layers,text.num_key_value_heads,getattr(text,'head_dim',text.hidden_size//text.num_attention_heads))==(spec['layers'],spec['kv_heads'],spec['head_dim'])
 processor=AutoProcessor.from_pretrained(job['model_path'],local_files_only=True,
  min_pixels=job['min_pixels'],max_pixels=job['max_pixels'])
 source=json.loads(Path(job['image_manifest']).read_text())[0]
 image=Image.open(source['image_path']).convert('RGB')
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':source['question']}]}]
 prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 inputs=processor(text=[prompt],images=[image],return_tensors='pt')
 assert 'mm_token_type_ids' in inputs,'native Transformers requires processor modality IDs'
 count=int((inputs['input_ids']==spec['image_token_id']).sum())
 modality=int((inputs['mm_token_type_ids']==1).sum())
 grid=inputs['image_grid_thw'][0].tolist()
 expected=grid[0]*grid[1]*grid[2]//4
 patch=14 if job['model_id']=='M2' else 16
 pixels=grid[1]*grid[2]*patch**2
 assert count==modality==expected and pixels<=job['max_pixels']
 rows.append({'model_id':job['model_id'],'status':'PASS_PROCESSOR_CONTRACT_ONLY',
  'GPU_used':False,'model_weights_loaded':False,'task_scores':None,
  'input_shapes':{k:list(v.shape) for k,v in inputs.items()},
  'visual_token_count':count,'image_grid_thw':grid,'processed_pixels':pixels,
  'image_processor_config':processor.image_processor.to_dict(),
  'full_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
  'chat_template_type':type(processor.chat_template).__name__})
result={'status':'PASS','scope':'CPU_PROCESSOR_CONTRACT_ONLY_NO_MODEL_HARDWARE_OR_QUALITY','records':rows}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'status':result['status'],'scope':result['scope'],'models':[x['model_id'] for x in rows]}))
