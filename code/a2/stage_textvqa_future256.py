"""Reserve fresh balanced transfer evidence before learned-book development."""
import argparse,collections,hashlib,io,json,shlex,sys
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,mkdirs,REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--upload',action='store_true');a=p.parse_args()
 marker=ROOT/'audit/a2/textvqa_future256_membership.json'
 if marker.exists():raise RuntimeError('Future transfer set is already frozen')
 source=ROOT.parent/'benchmark_samples/textvqa_samples.jsonl';legacy=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
 used_ids={str(row['image_id']) for row in legacy}
 used={sha(ROOT.parent/'benchmark_samples/textvqa_images'/row['image_path'].replace('\\','/').split('/')[-1]) for row in legacy}
 folder=ROOT/'data/textvqa_validation_pinned';assert json.loads((folder/'complete.json').read_text())['status']=='PASS'
 official=json.loads((folder/'manifest.json').read_text());shards=[Path(x['local']) for x in official['files']]
 rows=[]
 for shard in shards:rows.extend(pq.read_table(shard,columns=['image_id','question_id','question','answers']).to_pylist())
 assert len(rows)==5000;all_ids={str(row['image_id']) for row in rows};pools=collections.defaultdict(list);seen=set()
 for index,row in enumerate(sorted(rows,key=lambda x:int(x['question_id']))):
  source_id=str(row['image_id'])
  if source_id in used_ids or source_id in seen:continue
  seen.add(source_id)
  if len(row['answers'])!=10:continue
  majority=collections.Counter(x.strip().lower() for x in row['answers']).most_common(1)[0][0]
  stratum='binary' if majority in ('yes','no') else 'numeric' if any(ch.isdigit() for ch in majority) else 'text'
  key=hashlib.sha256(('trajkv-textvqa-future-v1:'+source_id).encode()).hexdigest()
  pools[stratum].append((key,index,row))
 assert len(pools['numeric'])>=128 and len(pools['text'])>=128
 selected=sorted([(s,*row) for s in ['numeric','text'] for row in sorted(pools[s],key=lambda x:x[0])[:128]],key=lambda x:x[1])
 wanted={str(entry[-1]['image_id']) for entry in selected};paths={};image_dir=ROOT/'data/textvqa_future256/images';image_dir.mkdir(exist_ok=True,parents=True)
 for shard in shards:
  for batch in pq.ParquetFile(shard).iter_batches(batch_size=64,columns=['image_id','image']):
   for item in batch.to_pylist():
    source_id=str(item['image_id'])
    if source_id not in wanted or source_id in paths:continue
    raw=item['image']['bytes'];digest=hashlib.sha256(raw).hexdigest();assert digest not in used
    with Image.open(io.BytesIO(raw)) as im:fmt=im.format;im.verify()
    path=image_dir/(digest+{'PNG':'.png','JPEG':'.jpg','TIFF':'.tif','WEBP':'.webp'}[fmt]);path.write_bytes(raw);assert sha(path)==digest;paths[source_id]=(path,digest)
 assert len(paths)==256 and len({entry[1] for entry in paths.values()})==256
 samples=[];references=[];images=[]
 for stratum,key,index,row in selected:
  path,digest=paths[str(row['image_id'])]
  sid='textvqa_future_'+digest[:16];question=row['question']+'\nAnswer the question using a single word or phrase.'
  remote=REMOTE_ROOT+'/data/textvqa_future256/images/'+digest+path.suffix
  samples.append({'schema':'trajkv.session.v1','dataset':'textvqa','id':sid,'session_id':sid,'image_id':'sha256:'+digest,'source_image_id':'sha256:'+digest,
   'hashes':[digest],'images':[str(path)],'remote_images':[remote],'questions':[question],'turns':1,
   'content':[{'type':'image'},{'type':'text','text':question}],'protocol':'single_image_single_question','source_row_index':index,'source_original_image_id':row['image_id'],'source_question_id':row['question_id'],
   'split':'textvqa_future256_balanced','historical_use':'earlier whole-project exposure not exhaustively audited'})
  references.append({'id':sid,'image_id':'sha256:'+digest,'answers':row['answers'],'answer_stratum':stratum})
  images.append({'local':str(path),'remote':remote,'sha256':digest,'bytes':path.stat().st_size})
 sp=ROOT/'configs/a2/textvqa_future256.json';rp=ROOT/'configs/data/textvqa_future256_references.json'
 for path,data in [(sp,samples),(rp,references)]:
  if path.exists():raise RuntimeError('Do not overwrite reserved input')
  path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
 audit={'created_at_utc':datetime.now(timezone.utc).isoformat(),'images':256,'strata':{'numeric':128,'text':128},'one_question_per_image':True,
  'selection':'exclude all legacy source-image IDs and legacy image-content hashes; first128 stable source-ID hashes per digit-containing/nonbinary-text majority stratum; lowest questionId per image; no model predictions used',
  'excluded_legacy_source_ids':len(used_ids),'legacy_ids_matched_official_validation':len(used_ids&all_ids),'excluded_legacy_content_hashes':len(used),
  'source_sha256':sha(source),'official_revision':official['revision'],'official_repo':official['repo'],'official_rows':len(rows),'official_unique_images':len(all_ids),'official_manifest_sha256':sha(folder/'manifest.json'),
  'samples_sha256':sha(sp),'references_sha256':sha(rp),
  'pool_sizes':{key:len(value) for key,value in pools.items()},'images_manifest':images,'future_model_outcomes_seen':False,
  'scope':'balanced within-subset transfer comparison, not official full-distribution accuracy; codebook candidate selection only on DocVQA development'}
 marker.write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print(json.dumps({'membership':'FROZEN','images':256,'strata':audit['strata'],'image_bytes':sum(x['bytes'] for x in images)}),flush=True)
 if a.upload:
  client=connect();sftp=client.open_sftp()
  try:
   checks={}
   for image in images:
    mkdirs(sftp,image['remote'].rsplit('/',1)[0]);sftp.put(image['local'],image['remote']);checks[image['remote']]=image['sha256']
   sftp.put(str(sp),REMOTE_ROOT+'/configs/a2/'+sp.name);checks[REMOTE_ROOT+'/configs/a2/'+sp.name]=sha(sp)
   script="import hashlib,json;from pathlib import Path;e=%r;m=[p for p,h in e.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];print(json.dumps({'files':len(e),'mismatches':len(m)}));assert not m"%checks
   _,out,err=client.exec_command(shlex.join([PYTHON,'-c',script]),timeout=60);report=json.loads(out.read());error=err.read().decode()
   if out.channel.recv_exit_status():raise RuntimeError(error[-400:])
  finally:sftp.close();client.close()
  receipt={'status':'PASS',**report,'answers_uploaded':False};(ROOT/'audit/a2/textvqa_future256_transport.json').write_text(json.dumps(receipt,indent=2))
  print(json.dumps(receipt))

if __name__=='__main__':main()
