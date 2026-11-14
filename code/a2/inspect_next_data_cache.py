"""Shallow, read-only cache/model API inspection for the next research phase."""
import json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,PYTHON
def main():
 code='''import json
from pathlib import Path
from transformers import AutoConfig,AutoProcessor
disk=Path('/media/user/5bc0dc8d-c938-4b2a-af1b-94174fc511a9');out={}
path=disk/'models/LLaVA-1.5-7B-HF';c=AutoConfig.from_pretrained(path,local_files_only=True);t=c.get_text_config()
out['llava_resolved']={k:getattr(t,k,None) for k in ['num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size']}
p=AutoProcessor.from_pretrained(path,local_files_only=True);out['processor_type']=type(p).__name__;out['image_processor_type']=type(p.image_processor).__name__;out['image_size']=getattr(p.image_processor,'size',None)
out['cache_entries']=[]
for path in [disk/'datasets/hf_cache/hub',disk/'datasets/modelscope-cache/datasets']:
 if path.is_dir():
  entries=[x for x in sorted(path.iterdir()) if any(z in x.name.lower() for z in ['doc','text','vqa','dataset_repo'])]
  out['cache_entries'].append({'path':str(path),'names':[x.name for x in entries[:50]]})
  for x in entries[:5]:
   if x.is_dir() and 'dataset_repo' in x.name.lower():out['cache_entries'].append({'path':str(x),'names':[z.name for z in sorted(x.iterdir())[:50]]})
print(json.dumps(out,default=str))
'''
 c=connect()
 try:
  _,o,e=c.exec_command(shlex.join([PYTHON,'-c',code]),timeout=60);value=o.read().decode();err=e.read().decode()
  if o.channel.recv_exit_status():raise RuntimeError(err)
 finally:c.close()
 out=json.loads(value);(ROOT/'audit/a2/next_data_cache_metadata.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
