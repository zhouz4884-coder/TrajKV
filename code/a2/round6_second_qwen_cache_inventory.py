"""Bounded read-only data-disk Qwen2/2.5-VL 3B/7B cache metadata survey."""
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))

REMOTE_CODE = r'''
import hashlib,json,re,struct
from pathlib import Path
disk=Path('/media/user/5bc0dc8d-c938-4b2a-af1b-94174fc511a9').resolve()
pattern=re.compile(r'qwen2(?:[._-]5)?[-_]?vl[-_]?(?:3b|7b)(?:\b|[-_])',re.I)
out={'read_only':True,'GPU_used':False,'downloads':False,'weight_payload_read':False,'search_scope':'explicit data-disk cache roots and VLAtest one-level models/checkpoints; no home/system scan','roots':[],'candidates':[]}
candidates=set()
def safe(path):
 return path.resolve().is_relative_to(disk)
def listing(path,kind):
 row={'path':str(path),'kind':kind,'exists':path.is_dir()}
 if not row['exists']:out['roots'].append(row);return []
 if not safe(path):raise ValueError('Search root escaped data disk')
 entries=sorted(path.iterdir());row['direct_entries']=len(entries)
 row['matching_names']=[p.name for p in entries if pattern.search(p.name)]
 out['roots'].append(row)
 for p in entries:
  if p.is_dir() and pattern.search(p.name):candidates.add(p)
 return entries
hub=disk/'datasets/hf_cache/hub'
listing(hub,'HF Hub direct model repositories')
ms=disk/'datasets/modelscope-cache'
base=listing(ms,'ModelScope bounded root')
for path in [ms/'models',ms/'hub',ms/'Qwen',ms/'qwen',ms/'models/Qwen',ms/'models/qwen',ms/'hub/Qwen',ms/'hub/qwen']:
 listing(path,'known ModelScope Qwen namespace')
vl=disk/'VLAtest'
projects=listing(vl,'VLAtest direct children; only models/checkpoints descendants examined')
for directory in [disk/'models',disk/'checkpoints',vl/'models',vl/'checkpoints']:
 listing(directory,'nearby explicit models/checkpoints root')
for project in projects:
 if not project.is_dir() or project.name in ['models','checkpoints']:continue
 for name in ['models','checkpoints']:
  path=project/name
  if path.is_dir():listing(path,'one-level project models/checkpoints')
expanded=set()
for path in candidates:
 snapshots=path/'snapshots'
 if snapshots.is_dir():
  expanded.update(p for p in snapshots.iterdir() if p.is_dir())
 else:expanded.add(path)
for root in sorted(expanded):
 if not safe(root):raise ValueError('Candidate escaped data disk')
 row={'path':str(root),'config_exists':(root/'config.json').is_file()}
 if not row['config_exists']:
  row['status']='NO_LOCAL_CONFIG';row['entries']=[p.name for p in root.iterdir()][:30];out['candidates'].append(row);continue
 config=json.loads((root/'config.json').read_bytes());q=config.get('quantization_config')
 row.update(config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),model_type=config.get('model_type'),architectures=config.get('architectures'),dtype=config.get('dtype',config.get('torch_dtype')),quantization_config=q)
 expected={};indexes=[]
 for path in root.glob('*.index.json'):
  if not path.is_file():continue
  index=json.loads(path.read_bytes())
  if 'weight_map' not in index:continue
  indexes.append({'file':path.name,'tensor_count':len(index['weight_map']),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  for tensor,file in index['weight_map'].items():expected.setdefault(file,set()).add(tensor)
 if not expected:expected={p.name:set() for p in root.glob('*.safetensors')}
 shards=[];complete=bool(expected);quantized_tensors=False;dtypes=set()
 for file,tensors in sorted(expected.items()):
  path=root/file;item={'file':file,'exists':path.is_file()}
  if not path.is_file():complete=False;shards.append(item);continue
  if not safe(path):raise ValueError('Weight symlink escaped data disk')
  item['bytes']=path.stat().st_size
  if path.suffix!='.safetensors':
   complete=False;item['status']='NON_SAFETENSORS_NOT_DESERIALIZED';shards.append(item);continue
  try:
   with path.open('rb') as stream:
    size=struct.unpack('<Q',stream.read(8))[0]
    if size>64*1024*1024 or size+8>item['bytes']:raise ValueError('header length')
    header=json.loads(stream.read(size))
   keys=set(header)-{'__metadata__'}
   offsets=sorted(v['data_offsets'] for k,v in header.items() if k!='__metadata__')
   dt={v['dtype'] for k,v in header.items() if k!='__metadata__'};dtypes.update(dt)
   quantized_tensors=quantized_tensors or any(re.search(r'(?:^|\.)(?:qweight|qzeros|weight_packed)(?:\.|$)',key) for key in keys)
   payload=item['bytes']-8-size
   valid=bool(offsets) and not(tensors-keys) and offsets[0][0]==0 and offsets[-1][1]==payload and all(0<=a<=b<=payload for a,b in offsets) and all(offsets[i][1]==offsets[i+1][0] for i in range(len(offsets)-1))
   item.update(tensor_count=len(keys),header_bytes=size,dtypes=sorted(dt),header_offsets_cover_payload=valid,missing_index_tensors=len(tensors-keys));complete=complete and valid
  except Exception as error:
   item['error']=type(error).__name__+': '+str(error);complete=False
  shards.append(item)
 row.update(indexes=indexes,shards=shards,total_weight_file_bytes=sum(x.get('bytes',0) for x in shards),all_shards_present_and_headers_valid=complete,header_dtypes=sorted(dtypes),quantized_tensors_detected=quantized_tensors,
 full_precision_candidate=complete and not q and not quantized_tensors and bool(dtypes & {'F16','BF16'}))
 out['candidates'].append(row)
out['complete_full_precision_candidates']=[r['path'] for r in out['candidates'] if r.get('full_precision_candidate')]
out['integrity_scope']='index+shard inventory and safetensors header offsets/file lengths; weight payload bytes not hashed'
print(json.dumps(out))
'''


def main():
    from remote import connect, PYTHON
    path = ROOT / 'audit/a2/round6_second_qwen_cache_inventory_v1.json'
    if path.exists():
        raise FileExistsError(path)
    client = connect()
    try:
        _, output, error = client.exec_command(shlex.join([PYTHON, '-c', REMOTE_CODE]), timeout=60)
        payload = output.read().decode()
        diagnostic = error.read().decode()
        if output.channel.recv_exit_status():
            raise RuntimeError('Read-only metadata survey failed: ' + diagnostic[-1000:])
    finally:
        client.close()
    data = json.loads(payload)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
    print(json.dumps(dict(roots=len(data['roots']), candidates=len(data['candidates']),
        full_precision_candidates=data['complete_full_precision_candidates'],
        candidates_summary=[{key:item.get(key) for key in ('path','model_type','dtype','total_weight_file_bytes',
            'all_shards_present_and_headers_valid','full_precision_candidate')} for item in data['candidates']])))


if __name__ == '__main__':
    main()
