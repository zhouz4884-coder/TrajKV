"""Prepare source-document-disjoint next-phase data from pinned public shards.

No model outcomes enter selection. Original encoded images are copied exactly;
answers live in a separate local scoring file and are not uploaded by this tool.
"""
import argparse, hashlib, io, json, shlex, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from remote import connect, mkdirs, REMOTE_ROOT, PYTHON

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_file(path):return sha_bytes(path.read_bytes())
def stable(text):return sha_bytes(('trajkv-next-docvqa-v1:'+text).encode())
def dump(path,data):
 path.parent.mkdir(exist_ok=True,parents=True)
 payload=json.dumps(data,indent=2,ensure_ascii=False).encode('utf-8')
 if path.exists() and path.read_bytes()!=payload:raise RuntimeError('Refusing to overwrite different frozen input: '+str(path))
 path.write_bytes(payload)

def prepare():
 frozen=ROOT/'audit/a2/next_docvqa_source_split_v1.json'
 if frozen.exists():
  result=json.loads(frozen.read_text(encoding='utf-8'))
  for relative,expected in result['input_sha256'].items():
   if sha_file(ROOT/relative)!=expected:raise RuntimeError('Staged input changed: '+relative)
  return result
 archive=json.loads((ROOT/'audit/a2/docvqa_validation_download.json').read_text(encoding='utf-8'))
 assert archive['status']=='COMPLETE'
 shards=sorted((ROOT/'data/docvqa_validation_pinned').glob('*.parquet'))
 expected={r['file']:r['sha256'] for r in archive['files']}
 columns=['questionId','question','question_types','docId','ucsf_document_id','ucsf_document_page_no','answers']
 rows=[]
 for path in shards:
  if sha_file(path)!=expected[path.name]:raise RuntimeError('Public shard mismatch')
  rows.extend(pq.read_table(path,columns=columns).to_pylist())
 old_path=Path('/path/to/benchmark_samples/docvqa_samples.jsonl')
 old=[json.loads(line) for line in old_path.read_text(encoding='utf-8').splitlines() if line.strip()]
 assert len(old)==1000
 for a,b in zip(old,rows):
  if a['question']!=b['question'] or a['answer'] not in b['answers']:raise RuntimeError('Legacy question order not established')
 excluded={r['ucsf_document_id'] for r in rows[:len(old)]}
 docs=defaultdict(lambda:defaultdict(list))
 for row in rows:
  if row['ucsf_document_id'] not in excluded:docs[row['ucsf_document_id']][row['docId']].append(row)
 ordered=sorted(docs,key=stable)
 allocation={'cal32':ordered[:32],'dev64':ordered[32:96],'confirm256':ordered[96:352]}
 assert len(ordered)>=352 and sum(map(len,allocation.values()))==352
 selected={}
 for split,names in allocation.items():
  for name in names:
   page=min(docs[name],key=lambda page:stable(name+':page:'+str(page)))
   selected[page]={'source_doc':name,'split':split,'rows':sorted(docs[name][page],key=lambda r:int(r['questionId']))[:6]}
 image_dir=ROOT/'data/docvqa_next/images';image_dir.mkdir(exist_ok=True,parents=True)
 for path in shards:
  for item in pq.read_table(path,columns=['docId','image']).to_pylist():
   page=item['docId']
   if page not in selected or 'image_path' in selected[page]:continue
   raw=item['image']['bytes'];digest=sha_bytes(raw)
   with Image.open(io.BytesIO(raw)) as im:fmt=im.format;size=list(im.size);im.verify()
   suffix={'PNG':'.png','JPEG':'.jpg','TIFF':'.tif'}[fmt]
   dest=image_dir/(digest+suffix)
   if dest.exists() and sha_file(dest)!=digest:raise RuntimeError('Existing image hash mismatch')
   if not dest.exists():dest.write_bytes(raw)
   selected[page].update(image_path=dest,sha256=digest,image_size=size)
 assert all('image_path' in r for r in selected.values())
 manifests={};references=[];inputs={};counts={};images=[]
 prompt='Answer the question using a single word or phrase.'
 for split,names in allocation.items():
  sessions=[]
  for name in names:
   page,entry=next((p,e) for p,e in selected.items() if e['source_doc']==name)
   questions=[r['question']+'\n'+prompt for r in entry['rows']]
   sid='docvqa_next_'+str(page);digest=entry['sha256'];local=entry['image_path'];relative=local.relative_to(ROOT).as_posix()
   sessions.append({'schema':'trajkv.session.v1','dataset':'docvqa_grouped','id':sid,'session_id':sid,
    'image_id':'sha256:'+digest,'source_image_id':'sha256:'+digest,'source_document_id':name,'source_page_id':page,
    'images':[str(local)],'remote_images':[REMOTE_ROOT+'/'+relative],'hashes':[digest],'questions':questions,'turns':len(questions),
    'content':[{'type':'image'},{'type':'text','text':questions[0]}],
    'protocol':'derived_same_document_independent_qa_grouped','history_dependence':'not_assumed_requires_qualification',
    'split':'nextphase_source_disjoint_'+split,'answer_format_protocol':'internvl_official_short_phrase_prompt_v1',
    'historical_result_reused':False,'source_question_ids':[r['questionId'] for r in entry['rows']]})
   references.append({'id':sid,'split':split,'source_document_id':name,'references':[{'turn':i+1,'question_id':r['questionId'],
    'answer':r['answers'],'question_types':r['question_types']} for i,r in enumerate(entry['rows'])]})
   inputs[relative]=digest;images.append({'relative':relative,'sha256':digest,'bytes':local.stat().st_size,'size':entry['image_size']})
  path=ROOT/'configs/a2'/('docvqa_next_'+split+'.json');dump(path,sessions);manifests[split]=path.relative_to(ROOT).as_posix();inputs[manifests[split]]=sha_file(path)
  counts[split]={'source_documents':len(sessions),'pages':len(sessions),'questions':sum(r['turns'] for r in sessions)}
 ref_path=ROOT/'configs/data/docvqa_next_references.jsonl'
 ref_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in references),encoding='utf-8');inputs[ref_path.relative_to(ROOT).as_posix()]=sha_file(ref_path)
 result={'schema':'trajkv.next.source-split.v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
  'source_repo':archive['repo'],'source_revision':archive['revision'],'shards_sha256':expected,
  'legacy1000_sha256':sha_file(old_path),'legacy_order_matches':1000,'excluded_source_documents':sorted(excluded),
  'selection_rule':'exclude every UCSF source document in legacy1000; stable SHA256 sort source-document IDs; first32 calibration, next64 development, next256 confirmation; one page per source selected by independent ID hash; ascending numeric questionId, max6',
  'remaining_source_documents':len(docs),'counts':counts,'manifests':manifests,'images':images,'input_sha256':inputs,
  'confirmation_status':'no nextphase model outcomes exist at membership freeze; method freeze occurs after development',
  'image_scope':'original public shard encoded bytes; no JPEG reencoding; differs from legacy reencoded files',
  'exposure_scope':'source-document-disjoint from verified legacy1000; earlier unrelated project data use not exhaustively audited',
  'source_sha256':sha_file(Path(__file__))}
 dump(frozen,result);return result

def upload(result):
 client=connect();receipts=[]
 try:
  sftp=client.open_sftp()
  try:
   paths=[x['relative'] for x in result['images']]+list(result['manifests'].values())
   for relative in paths:
    dest=REMOTE_ROOT+'/'+relative;mkdirs(sftp,str(Path(dest).parent).replace('\\','/'))
    sftp.put(str(ROOT/relative),dest);receipts.append({'relative':relative,'sha256':result['input_sha256'][relative]})
  finally:sftp.close()
  checks={x['relative']:x['sha256'] for x in receipts}
  code="import hashlib,json;from pathlib import Path;r=Path(%r);e=%r;m=[p for p,h in e.items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h];print(json.dumps({'checked':len(e),'mismatches':len(m)}));assert not m"%(REMOTE_ROOT,checks)
  _,out,err=client.exec_command(shlex.join([PYTHON,'-c',code]),timeout=60)
  text=out.read().decode();error=err.read().decode();status=out.channel.recv_exit_status()
  if status:raise RuntimeError('Remote staged-data verification failed: '+error[-500:])
  report={'status':'PASS','files':receipts,'remote_verification':json.loads(text),'answers_uploaded':False}
  (ROOT/'audit/a2/next_docvqa_data_transport.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 finally:client.close()
 print(json.dumps({'transport':'PASS','files':len(receipts),'answers_uploaded':False}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--upload',action='store_true');args=p.parse_args();result=prepare()
 print(json.dumps({'membership':'FROZEN','counts':result['counts'],'excluded_source_documents':len(result['excluded_source_documents']),'image_bytes':sum(r['bytes'] for r in result['images'])}),flush=True)
 if args.upload:upload(result)
if __name__=='__main__':main()
