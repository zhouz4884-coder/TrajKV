"""Read-only model/config and dataset-cache metadata, no GPU model loading."""
import json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,PYTHON
def main():
 code='''import json
from pathlib import Path
disk=Path('/media/user/5bc0dc8d-c938-4b2a-af1b-94174fc511a9');out={'models':[],'caches':[]}
for name in ['LLaVA-1.5-7B-HF','llava-v1.6-vicuna-7b-hf','InternVL2_5-8B']:
 p=disk/'models'/name/'config.json'
 if not p.exists():continue
 c=json.loads(p.read_text());t=c.get('text_config',c.get('llm_config',{}))
 out['models'].append({'name':name,'model_type':c.get('model_type'),'architecture':c.get('architectures'),'image_token_id':c.get('image_token_id'),'image_token_index':c.get('image_token_index'),
 'text':{k:t.get(k) for k in ['model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size','torch_dtype','dtype']}})
for p in [disk/'datasets/hf_cache',disk/'datasets/modelscope-cache',disk/'datasets/sources']:
 r={'path':str(p),'exists':p.exists()}
 if p.is_dir():r['entries']=[x.name for x in sorted(p.iterdir())[:80]]
 out['caches'].append(r)
print(json.dumps(out))
'''
 c=connect()
 try:
  _,o,e=c.exec_command(shlex.join([PYTHON,'-c',code]),timeout=30);value=o.read().decode();err=e.read().decode()
  if o.channel.recv_exit_status():raise RuntimeError(err)
 finally:c.close()
 out=json.loads(value);(ROOT/'audit/a2/next_model_metadata.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
