"""One read-only SSH metadata inventory; no GPU/model import or remote writes."""
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))

REMOTE_CODE = r'''
import ast, hashlib, importlib.metadata, importlib.util, json, struct, sys
from pathlib import Path
disk=Path('/media/user/5bc0dc8d-c938-4b2a-af1b-94174fc511a9')
models=disk/'models'
result={'read_only':True,'GPU_used':False,'weight_payload_read':False,'models_root':str(models),'python':sys.executable,'packages':{},'models':[],'legacy_sources':{}}
for package in ['transformers','torch','safetensors','accelerate']:
 try: result['packages'][package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError: result['packages'][package]=None
for root in sorted(models.iterdir()):
 if not root.is_dir(): continue
 config=root/'config.json'
 item={'name':root.name,'path':str(root),'config_exists':config.is_file()}
 if not config.is_file():
  item['direct_entries']=[p.name for p in sorted(root.iterdir())[:30]]
  result['models'].append(item);continue
 raw=config.read_bytes();c=json.loads(raw)
 text=c.get('text_config') or c.get('llm_config') or c
 item.update(config_sha256=hashlib.sha256(raw).hexdigest(),model_type=c.get('model_type'),architectures=c.get('architectures'),image_token_id=c.get('image_token_id'),image_token_index=c.get('image_token_index'),vision_feature_select_strategy=c.get('vision_feature_select_strategy'),text={key:text.get(key) for key in ['model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size','intermediate_size','head_dim','torch_dtype','dtype']},vision={key:c.get('vision_config',{}).get(key) for key in ['model_type','image_size','patch_size','hidden_size','num_hidden_layers']})
 for name in ['processor_config.json','preprocessor_config.json','tokenizer_config.json','chat_template.jinja']:
  path=root/name
  if not path.is_file():continue
  data=path.read_bytes();entry={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  if path.suffix=='.json':
   value=json.loads(data)
   for key in ['processor_class','image_processor_type','tokenizer_class','chat_template','patch_size','vision_feature_select_strategy','num_additional_image_tokens']:
    if key in value:entry[key]=value[key]
  elif name=='chat_template.jinja':entry['chat_template']=data.decode()
  item[name]=entry
 indexes=[];expected={}
 for index in sorted(root.glob('*.index.json')):
  data=json.loads(index.read_bytes())
  if 'weight_map' not in data:continue
  indexes.append({'file':index.name,'sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'tensor_count':len(data['weight_map']),'total_size_metadata':data.get('metadata',{}).get('total_size')})
  for tensor,file in data['weight_map'].items():expected.setdefault(file,set()).add(tensor)
 if not expected:
  expected={path.name:set() for pattern in ['*.safetensors','pytorch_model*.bin'] for path in root.glob(pattern)}
 shards=[];complete=True
 for file,tensors in sorted(expected.items()):
  path=root/file;row={'file':file,'exists':path.is_file()}
  if not path.is_file():complete=False;shards.append(row);continue
  row['bytes']=path.stat().st_size
  if path.suffix=='.safetensors':
   try:
    with path.open('rb') as stream:
     first=stream.read(8)
     if len(first)!=8:raise ValueError('truncated header length')
     size=struct.unpack('<Q',first)[0]
     if size>64*1024*1024 or size+8>row['bytes']:raise ValueError('invalid header length')
     header=json.loads(stream.read(size))
    keys=set(header)-{'__metadata__'};missing=tensors-keys
    offsets=[value['data_offsets'] for key,value in header.items() if key!='__metadata__']
    payload=row['bytes']-8-size
    ordered=sorted(offsets)
    valid=not missing and bool(offsets) and all(0<=a<=b<=payload for a,b in offsets) and max(b for a,b in offsets)==payload and all(ordered[i][1]==ordered[i+1][0] for i in range(len(ordered)-1)) and ordered[0][0]==0
    row.update(header_bytes=size,header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),tensor_count=len(keys),index_expected_tensors=len(tensors),missing_index_tensors=len(missing),tensor_offsets_cover_payload=valid,weight_payload_bytes=payload)
    complete=complete and valid
   except Exception as error:
    row['header_error']=type(error).__name__+': '+str(error);complete=False
  else:row['integrity_scope']='existence/size only; pickle weights not deserialized'
  shards.append(row)
 item.update(indexes=indexes,shards=shards,shard_count=len(shards),total_weight_file_bytes=sum(row.get('bytes',0) for row in shards),all_index_shards_present_and_safetensors_headers_valid=complete and bool(shards),integrity_scope='index mapping plus full shard inventory, safetensors headers/tensor offsets and exact file lengths; payload bytes not hashed')
 result['models'].append(item)
legacy=disk/'VLAtest/VisContract_P14_2/scripts'
for filename,names in {'p180_multiturn_core.py':['attention_bridge'],'p184_working_set.py':['attention'],'run_p175_modern_replication.py':['stop_ids']}.items():
 path=legacy/filename
 if not path.is_file():continue
 source=path.read_text();tree=ast.parse(source)
 result['legacy_sources'][filename]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'definitions':{node.name:ast.get_source_segment(source,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names}}
spec=importlib.util.find_spec('transformers')
if spec and spec.origin:
 result['transformers_source_root']=str(Path(spec.origin).parent)
print(json.dumps(result))
'''


def main():
    from remote import connect, PYTHON
    target = ROOT / 'audit/a2/round6_second_model_inventory_v1.json'
    if target.exists():
        raise FileExistsError(target)
    client = connect()
    try:
        _, output, error = client.exec_command(shlex.join([PYTHON, '-c', REMOTE_CODE]), timeout=60)
        payload = output.read().decode()
        diagnostic = error.read().decode()
        if output.channel.recv_exit_status():
            raise RuntimeError('Metadata inventory failed: ' + diagnostic[-1000:])
    finally:
        client.close()
    data = json.loads(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
    print(json.dumps({'inventory':str(target),'packages':data['packages'],'models':[
        {key:item.get(key) for key in ('name','model_type','text','shard_count','total_weight_file_bytes','all_index_shards_present_and_safetensors_headers_valid')}
        for item in data['models']]}))


if __name__ == '__main__':
    main()
