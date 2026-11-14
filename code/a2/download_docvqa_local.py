"""Pinned public validation-only archive download; no model or credentials."""
import concurrent.futures,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 metadata=json.loads((ROOT/'audit/a2/docvqa_public_local_metadata.json').read_text());folder=(ROOT/'data/docvqa_validation_pinned').resolve();folder.mkdir(exist_ok=True,parents=True)
 assert folder.is_relative_to(ROOT.resolve())
 def fetch(item):
  dest=(folder/Path(item['rfilename']).name).resolve();part=dest.with_suffix(dest.suffix+'.part');assert dest.is_relative_to(folder) and part.resolve().is_relative_to(folder)
  expected=item['lfs']['sha256'];url=f"https://huggingface.co/datasets/{metadata['repo']}/resolve/{metadata['revision']}/{item['rfilename']}?download=true"
  if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:return {'file':dest.name,'sha256':expected,'bytes':dest.stat().st_size,'reused':True}
  for attempt in range(3):
   try:
    h=hashlib.sha256();count=0;req=urllib.request.Request(url+f'&cache_version={int(time.time())}',headers={'User-Agent':'TrajKV-public-research-data/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response,part.open('wb') as out:
     while True:
      chunk=response.read(4*1024*1024)
      if not chunk:break
      out.write(chunk);h.update(chunk);count+=len(chunk)
    if count!=item['size'] or h.hexdigest()!=expected:raise ValueError('Public shard checksum mismatch')
    part.replace(dest);result={'file':dest.name,'sha256':expected,'bytes':count,'reused':False};print(json.dumps(result),flush=True);return result
   except Exception as exc:
    if attempt==2:return {'file':dest.name,'error_type':type(exc).__name__,'http_status':getattr(exc,'code',None)}
    time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,metadata['files']))
 output={'status':'COMPLETE' if all('sha256' in r for r in results) else 'INCOMPLETE','repo':metadata['repo'],'revision':metadata['revision'],'files':results,
  'purpose':'next-phase independent data expansion; no current held-out outcomes used','token_used':False}
 (ROOT/'audit/a2/docvqa_validation_download.json').write_text(json.dumps(output,indent=2));print(json.dumps({'status':output['status'],'verified_shards':sum('sha256' in r for r in results)}))
if __name__=='__main__':main()
