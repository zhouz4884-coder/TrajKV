"""Complete pinned official validation, preserving all5000 QA pairs and bytes."""
import argparse,collections,hashlib,io,json,shlex,sys
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,mkdirs,REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--upload',action='store_true');args=parser.parse_args()
 marker=ROOT/'audit/a2/textvqa_standard5000_membership.json'
 if marker.exists():raise RuntimeError('Complete validation membership already frozen')
 folder=ROOT/'data/textvqa_validation_pinned';assert json.loads((folder/'complete.json').read_text())['status']=='PASS'
 official=json.loads((folder/'manifest.json').read_text());shards=[Path(x['local']) for x in official['files']]
 rows=[]
 for shard in shards:rows.extend(pq.read_table(shard,columns=['image_id','question_id','question','answers']).to_pylist())
 assert len(rows)==5000 and len({r['question_id'] for r in rows})==5000 and all(len(r['answers'])==10 for r in rows)
 paths={};images=[];formats=collections.Counter();images_dir=ROOT/'data/textvqa_standard5000/images';images_dir.mkdir(exist_ok=True,parents=True)
 for shard in shards:
  for batch in pq.ParquetFile(shard).iter_batches(batch_size=64,columns=['image_id','image']):
   for row in batch.to_pylist():
    original=str(row['image_id'])
    if original in paths:continue
    raw=row['image']['bytes'];digest=hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:fmt=image.format;image.verify()
    suffix={'JPEG':'.jpg','PNG':'.png','TIFF':'.tif','WEBP':'.webp'}.get(fmt)
    if suffix is None:suffix=min([extension for extension,format_name in Image.registered_extensions().items() if format_name==fmt],key=lambda x:(len(x),x))
    path=images_dir/(digest+suffix);formats[fmt]+=1
    if not path.exists():path.write_bytes(raw)
    assert sha(path)==digest;remote=REMOTE_ROOT+'/'+path.relative_to(ROOT).as_posix();paths[original]=(path,remote,digest)
    images.append({'local':str(path),'remote':remote,'sha256':digest,'bytes':path.stat().st_size,'source_image_id':original})
 samples=[];references=[];strata=collections.Counter()
 for row in sorted(rows,key=lambda x:int(x['question_id'])):
  path,remote,digest=paths[str(row['image_id'])];sid='textvqa_standard_'+str(row['question_id']);question=row['question']+'\nAnswer the question using a single word or phrase.'
  majority=collections.Counter(x.strip().lower() for x in row['answers']).most_common(1)[0][0]
  stratum='binary' if majority in ['yes','no'] else 'numeric' if any(ch.isdigit() for ch in majority) else 'text';strata[stratum]+=1
  samples.append({'schema':'trajkv.session.v1','dataset':'textvqa','id':sid,'session_id':sid,'image_id':'sha256:'+digest,'source_image_id':'sha256:'+digest,
   'source_original_image_id':row['image_id'],'source_question_id':row['question_id'],'hashes':[digest],'images':[str(path)],'remote_images':[remote],
   'questions':[question],'turns':1,'content':[{'type':'image'},{'type':'text','text':question}],'protocol':'single_image_single_question','split':'complete_official_validation5000'})
  references.append({'id':sid,'image_id':'sha256:'+digest,'source_question_id':row['question_id'],'answers':row['answers'],'answer_stratum':stratum})
 sp=ROOT/'configs/a2/textvqa_standard5000.json';rp=ROOT/'configs/data/textvqa_standard5000_references.json'
 for path,data in [(sp,samples),(rp,references)]:
  if path.exists():raise RuntimeError('Do not overwrite complete validation input')
  path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
 unique_images={x['remote']:x for x in images};audit={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'questions':5000,'source_image_ids':len(paths),
  'unique_image_contents':len(unique_images),'strata_by_majority_answer':dict(strata),'image_formats':dict(formats),'image_loading':'preserve original bytes; common model loader takes PIL first frame and converts RGB, including MPO',
  'official_repo':official['repo'],'official_revision':official['revision'],
  'official_manifest_sha256':sha(folder/'manifest.json'),'samples_sha256':sha(sp),'references_sha256':sha(rp),'images_manifest':list(unique_images.values()),
  'scope':'all official5000 validation QA, single-image independent questions with fixed prompt/cap8; no OCR tokens supplied; includes previously observed legacy and balanced subsets, not a wholly unseen confirmation',
  'selection':'all5000 QA, no output-based filtering; quantizer methods frozen using DocVQA development, never selected using these TextVQA scores'}
 marker.write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps({key:audit[key] for key in ['questions','source_image_ids','unique_image_contents','strata_by_majority_answer']}),flush=True)
 if args.upload:
  client=connect();sftp=client.open_sftp()
  try:
   checks={}
   for index,item in enumerate(unique_images.values()):
    mkdirs(sftp,item['remote'].rsplit('/',1)[0]);sftp.put(item['local'],item['remote']);checks[item['remote']]=item['sha256']
    if (index+1)%250==0:print(json.dumps({'uploaded_images':index+1,'total':len(unique_images)}),flush=True)
   sftp.put(str(sp),REMOTE_ROOT+'/configs/a2/'+sp.name);checks[REMOTE_ROOT+'/configs/a2/'+sp.name]=sha(sp)
   # A file avoids the OS single-argument length limit for thousands of paths.
   checkfile=ROOT/'audit/a2/textvqa_standard5000_transport_inputs.json';checkfile.write_text(json.dumps(checks),encoding='utf-8')
   remote_check=REMOTE_ROOT+'/configs/data/textvqa_standard5000_transport_inputs.json';mkdirs(sftp,remote_check.rsplit('/',1)[0]);sftp.put(str(checkfile),remote_check)
   script="import hashlib,json;from pathlib import Path;e=json.loads(Path(%r).read_text());bad=[p for p,h in e.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];print(json.dumps({'files':len(e),'mismatches':len(bad)}));assert not bad"%remote_check
   _,out,err=client.exec_command(shlex.join([PYTHON,'-c',script]),timeout=120);report=json.loads(out.read());error=err.read().decode()
   if out.channel.recv_exit_status():raise RuntimeError(error[-400:])
  finally:sftp.close();client.close()
  receipt={'status':'PASS',**report,'answers_uploaded':False};(ROOT/'audit/a2/textvqa_standard5000_transport.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
