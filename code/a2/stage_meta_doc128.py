"""Reserve new source documents for later methods, outside running confirm256."""
import argparse,hashlib,io,json,shlex,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,mkdirs,REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stable(text):return hashlib.sha256(('trajkv-next-docvqa-v1:'+text).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--upload',action='store_true');a=p.parse_args();marker=ROOT/'audit/a2/meta_doc128_membership.json'
 if marker.exists():raise RuntimeError('Future membership already frozen')
 previous=json.loads((ROOT/'audit/a2/next_docvqa_source_split_v1.json').read_text(encoding='utf-8'))
 used=set(previous['excluded_source_documents']);used_hashes=set()
 for relative in previous['manifests'].values():
  for row in json.loads((ROOT/relative).read_text(encoding='utf-8')):used.add(row['source_document_id']);used_hashes.update(row['hashes'])
 for row in json.loads((ROOT/'configs/a2/docvqa_future128.json').read_text(encoding='utf-8')):
  used.add(row['source_document_id']);used_hashes.update(row['hashes'])
 shards=sorted((ROOT/'data/docvqa_validation_pinned').glob('*.parquet'));rows=[]
 for path in shards:rows.extend(pq.read_table(path,columns=['questionId','question','question_types','docId','ucsf_document_id','answers']).to_pylist())
 docs=defaultdict(lambda:defaultdict(list))
 for row in rows:
  if row['ucsf_document_id'] not in used:docs[row['ucsf_document_id']][row['docId']].append(row)
 chosen=sorted(docs,key=stable)[:128];assert len(chosen)==128;selected={}
 for name in chosen:
  page=min(docs[name],key=lambda page:stable(name+':page:'+str(page)))
  selected[page]={'name':name,'rows':sorted(docs[name][page],key=lambda r:int(r['questionId']))[:6]}
 images_dir=ROOT/'data/docvqa_meta128/images';images_dir.mkdir(exist_ok=True,parents=True)
 for shard in shards:
  for row in pq.read_table(shard,columns=['docId','image']).to_pylist():
   page=row['docId']
   if page not in selected or 'path' in selected[page]:continue
   raw=row['image']['bytes'];digest=hashlib.sha256(raw).hexdigest();assert digest not in used_hashes
   with Image.open(io.BytesIO(raw)) as im:fmt=im.format;im.verify()
   suffix={'PNG':'.png','JPEG':'.jpg','TIFF':'.tif'}[fmt];path=images_dir/(digest+suffix)
   if not path.exists():path.write_bytes(raw)
   assert sha(path)==digest;selected[page].update(path=path,sha256=digest)
 samples=[];refs=[];images=[]
 for name in chosen:
  page,entry=next((p,e) for p,e in selected.items() if e['name']==name);sid='docvqa_meta_'+str(page);digest=entry['sha256'];path=entry['path']
  questions=[r['question']+'\nAnswer the question using a single word or phrase.' for r in entry['rows']];remote=REMOTE_ROOT+'/'+path.relative_to(ROOT).as_posix()
  samples.append({'schema':'trajkv.session.v1','dataset':'docvqa_grouped','id':sid,'session_id':sid,'image_id':'sha256:'+digest,'source_image_id':'sha256:'+digest,
   'source_document_id':name,'source_page_id':page,'images':[str(path)],'remote_images':[remote],'hashes':[digest],'questions':questions,'turns':len(questions),
   'content':[{'type':'image'},{'type':'text','text':questions[0]}],'protocol':'derived_same_document_independent_qa_grouped','split':'meta_unseen_source128',
   'history_dependence':'not_assumed_requires_qualification','source_question_ids':[r['questionId'] for r in entry['rows']]})
  refs.append({'id':sid,'source_document_id':name,'references':[{'turn':i+1,'answer':r['answers'],'question_id':r['questionId'],'question_types':r['question_types']} for i,r in enumerate(entry['rows'])]})
  images.append({'local':str(path),'remote':remote,'sha256':digest,'bytes':path.stat().st_size})
 assert len({r['image_id'] for r in samples})==128
 sp=ROOT/'configs/a2/docvqa_meta128.json';rp=ROOT/'configs/data/docvqa_meta128_references.jsonl'
 for path in (sp,rp):
  if path.exists():raise RuntimeError('Future input already exists')
 sp.write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding='utf-8');rp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs),encoding='utf-8')
 report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'images':128,'source_documents':128,'questions':sum(r['turns'] for r in samples),
  'selection':'exclude legacy212 and all32cal+64dev+256confirm+previous future128 sources; hash sort remaining source IDs, one page by independent hash, first6numeric questionIds',
  'previous_membership_sha256':sha(ROOT/'audit/a2/next_docvqa_source_split_v1.json'),'samples_sha256':sha(sp),'references_sha256':sha(rp),
  'excluded_previous_future128_sha256':sha(ROOT/'audit/a2/future_doc128_membership.json'),'source_documents_excluded':len(used),'remaining_before_selection':len(docs),'images_manifest':images,
  'scope':'future functional-metadata methods; excludes both prior confirmation cohorts; no model outcomes on these reserved source documents',
  'source_revision':previous['source_revision'],'source_sha256':sha(Path(__file__))}
 marker.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'membership':'FROZEN','source_documents':128,'questions':report['questions'],'image_bytes':sum(r['bytes'] for r in images)}),flush=True)
 if a.upload:
  client=connect();sftp=client.open_sftp()
  try:
   checks={}
   for item in images:mkdirs(sftp,item['remote'].rsplit('/',1)[0]);sftp.put(item['local'],item['remote']);checks[item['remote']]=item['sha256']
   remote=REMOTE_ROOT+'/configs/a2/'+sp.name;sftp.put(str(sp),remote);checks[remote]=sha(sp)
   script="import json,hashlib;from pathlib import Path;e=%r;bad=[p for p,h in e.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];print(json.dumps({'files':len(e),'mismatches':len(bad)}));assert not bad"%checks
   _,out,err=client.exec_command(shlex.join([PYTHON,'-c',script]),timeout=60);result=json.loads(out.read());error=err.read().decode();status=out.channel.recv_exit_status()
   if status:raise RuntimeError(error[-400:])
  finally:sftp.close();client.close()
  (ROOT/'audit/a2/meta_doc128_transport.json').write_text(json.dumps({'status':'PASS',**result,'answers_uploaded':False},indent=2));print(json.dumps({'transport':'PASS',**result,'answers_uploaded':False}))
if __name__=='__main__':main()
