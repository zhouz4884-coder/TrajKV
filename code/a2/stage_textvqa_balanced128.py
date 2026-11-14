"""Fresh content-stratified external128, selected without any model outcomes."""
import argparse,collections,hashlib,json,shlex,sys
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,mkdirs,REMOTE_ROOT,PYTHON

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--upload',action='store_true');a=p.parse_args();audit_path=ROOT/'audit/a2/textvqa_balanced128_membership.json'
 if audit_path.exists():raise RuntimeError('Balanced external membership already frozen')
 source=ROOT.parent/'benchmark_samples/textvqa_samples.jsonl';rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
 used={h for name in ('textvqa_dev32','textvqa_eval128') for r in json.loads((ROOT/f'configs/a2/{name}.json').read_text(encoding='utf-8')) for h in r['hashes']}
 pools=collections.defaultdict(list);seen=set()
 for index,row in enumerate(rows):
  path=ROOT.parent/'benchmark_samples/textvqa_images'/row['image_path'].replace('\\','/').split('/')[-1]
  h=sha(path)
  if h in used or h in seen:continue
  seen.add(h)
  if len(row['answers'])!=10:continue
  with Image.open(path) as im:im.verify()
  majority=collections.Counter(x.strip().lower() for x in row['answers']).most_common(1)[0][0]
  stratum='binary' if majority in ('yes','no') else 'numeric' if any(ch.isdigit() for ch in majority) else 'text'
  pools[stratum].append((h,index,path,row))
 assert len(pools['numeric'])>=64 and len(pools['text'])>=64
 selected=[(s,*r) for s in ('numeric','text') for r in sorted(pools[s],key=lambda x:x[0])[:64]]
 selected.sort(key=lambda x:x[1]);samples=[];refs=[];images=[]
 for stratum,h,index,path,row in selected:
  sid='textvqa_balanced_'+h[:16];question=row['question']+'\nAnswer the question using a single word or phrase.'
  remote=REMOTE_ROOT+'/data/textvqa_next/images/'+h+path.suffix
  samples.append({'schema':'trajkv.session.v1','dataset':'textvqa','id':sid,'session_id':sid,'source_image_id':'sha256:'+h,'image_id':'sha256:'+h,
   'hashes':[h],'images':[str(path)],'remote_images':[remote],'questions':[question],'turns':1,
   'content':[{'type':'image'},{'type':'text','text':question}],'protocol':'single_image_single_question',
   'source_row_index':index,'split':'textvqa_balanced128_nextphase','historical_use':'earlier whole-project exposure not exhaustively audited'})
  refs.append({'id':sid,'image_id':'sha256:'+h,'answers':row['answers'],'answer_stratum':stratum})
  images.append({'local':str(path),'remote':remote,'sha256':h,'bytes':path.stat().st_size})
 sp=ROOT/'configs/a2/textvqa_balanced128.json';rp=ROOT/'configs/data/textvqa_balanced128_references.json'
 for path,data in [(sp,samples),(rp,refs)]:
  if path.exists():raise RuntimeError('Do not overwrite external input')
  path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
 audit={'created_at_utc':datetime.now(timezone.utc).isoformat(),'images':128,'one_question_per_image':True,'source_sha256':sha(source),
  'selection':'exclude all160 current-cycle TextVQA images; first64 content hashes in numeric-majority-answer stratum and first64 in nonbinary-text stratum; first source QA per image; no predictions or compression losses used',
  'pool_sizes':{k:len(v) for k,v in pools.items()},'strata':{'numeric':64,'text':64},'excluded_current_cycle_images':len(used),
  'samples_sha256':sha(sp),'references_sha256':sha(rp),'images_manifest':images,'model_outcomes_seen':False,
  'reporting_scope':'prospective content-balanced subset, not an estimate of the overall official TextVQA distribution; compare methods within this subset',
  'method_freeze':'methods selected only on DocVQA development; no inference on balanced128 before method freeze'}
 audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps({'membership':'FROZEN','images':128,'strata':audit['strata'],'image_bytes':sum(x['bytes'] for x in images)}),flush=True)
 if a.upload:
  client=connect();sftp=client.open_sftp()
  try:
   checks={}
   for image in images:
    mkdirs(sftp,image['remote'].rsplit('/',1)[0]);sftp.put(image['local'],image['remote']);checks[image['remote']]=image['sha256']
   sftp.put(str(sp),REMOTE_ROOT+'/configs/a2/'+sp.name);checks[REMOTE_ROOT+'/configs/a2/'+sp.name]=sha(sp)
   script="import json,hashlib;from pathlib import Path;e=%r;m=[p for p,h in e.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];print(json.dumps({'files':len(e),'mismatches':len(m)}));assert not m"%checks
   _,out,err=client.exec_command(shlex.join([PYTHON,'-c',script]),timeout=60);report=json.loads(out.read());error=err.read().decode();status=out.channel.recv_exit_status()
   if status:raise RuntimeError(error[-400:])
  finally:sftp.close();client.close()
  (ROOT/'audit/a2/textvqa_balanced128_transport.json').write_text(json.dumps({'status':'PASS',**report,'answers_uploaded':False},indent=2))
  print(json.dumps({'transport':'PASS',**report,'answers_uploaded':False}))
if __name__=='__main__':main()
