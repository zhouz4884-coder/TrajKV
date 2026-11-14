"""Legitimate public validation shards for the next evaluation expansion.

No model inference, no authentication token, no current held-out outcomes.
Pin the repository revision and verify available LFS SHA256 values.
"""
import argparse,hashlib,json,os
from pathlib import Path
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 os.environ['HF_HUB_OFFLINE']='0';os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
 from huggingface_hub import HfApi,hf_hub_download
 repo='lmms-lab-encoder/DocVQA';info=HfApi(token=False).dataset_info(repo,files_metadata=True,timeout=30)
 files=[]
 for x in info.siblings:
  if not (x.rfilename.startswith('DocVQA/validation-') and x.rfilename.endswith('.parquet')):continue
  lfs=getattr(x,'lfs',None);expected=getattr(lfs,'sha256',None) if lfs is not None else None
  if isinstance(lfs,dict):expected=lfs.get('sha256')
  files.append({'name':x.rfilename,'bytes':x.size,'expected_sha256':expected})
 if len(files)!=6:raise RuntimeError('Unexpected validation shard layout')
 manifest={'repo':repo,'revision':info.sha,'public_token_used':False,'dataset_card_license':(getattr(info,'card_data',None) or {}).get('license') if isinstance(getattr(info,'card_data',None),dict) else str(getattr(info,'card_data',None)),
  'purpose':'next-phase independent expansion; current64/128 protocols unchanged','files':files,'metadata_only':a.metadata_only}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'revision':info.sha,'shards':len(files),'bytes':sum(x['bytes'] or 0 for x in files)}),flush=True)
 if not a.metadata_only:
  for item in files:
   path=Path(hf_hub_download(repo,item['name'],repo_type='dataset',revision=info.sha,token=False,local_dir=a.output/'official'))
   observed=digest(path)
   if item['expected_sha256'] and item['expected_sha256']!=observed:raise RuntimeError('Validation shard identity mismatch')
   item.update(local=str(path),sha256=observed);(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'downloaded':item['name'],'sha256':observed}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','metadata_only':a.metadata_only,'shards':len(files)}))
if __name__=='__main__':main()
