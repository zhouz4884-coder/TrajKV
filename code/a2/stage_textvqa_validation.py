"""Pinned public TextVQA validation bytes; no model or answer-driven selection."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPO='lmms-lab-encoder/textvqa';REVISION='9c0699cd19768ac5ab97568f6b3cbac4c0062884'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 folder=ROOT/'data/textvqa_validation_pinned';folder.mkdir(exist_ok=True,parents=True)
 marker=folder/'complete.json'
 if marker.exists():raise RuntimeError('Pinned validation already complete')
 os.environ.update(HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
 from huggingface_hub import HfApi,hf_hub_download
 info=HfApi(token=False).dataset_info(REPO,revision=REVISION,files_metadata=True,timeout=30)
 files=[]
 for entry in info.siblings:
  if entry.rfilename.startswith('data/validation-') and entry.rfilename.endswith('.parquet'):
   files.append({'name':entry.rfilename,'bytes':entry.size,'expected_sha256':entry.lfs.sha256})
 assert len(files)==3 and info.sha==REVISION
 manifest={'repo':REPO,'revision':REVISION,'authentication_token_used':False,'files':files,
  'scope':'public official validation export, for source-disjoint transfer and standard benchmark evaluation; membership selected separately before new method outputs'}
 mp=folder/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));print(json.dumps({'revision':REVISION,'files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)
 for entry in files:
  path=Path(hf_hub_download(REPO,entry['name'],repo_type='dataset',revision=REVISION,token=False,local_dir=folder))
  observed=digest(path);assert observed==entry['expected_sha256']
  entry.update(local=str(path),sha256=observed);mp.write_text(json.dumps(manifest,indent=2));print(json.dumps({'downloaded':entry['name'],'bytes':entry['bytes'],'sha256':observed}),flush=True)
 marker.write_text(json.dumps({'status':'PASS','shards':len(files),'bytes':sum(x['bytes'] for x in files)}))

if __name__=='__main__':main()
