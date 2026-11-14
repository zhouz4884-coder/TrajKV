"""Prospective external OCR split, independent of current method outcomes."""
import hashlib,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
def main():
 source=ROOT.parent/'benchmark_samples/textvqa_samples.jsonl';rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
 doc_hashes=set()
 for f in (ROOT/'configs/data').glob('docvqa*runner.json'):
  for r in json.loads(f.read_text(encoding='utf-8')):doc_hashes.update(r.get('hashes',[]))
 remote='/path/to/storage/VLAtest/gsm8k_server_bundle/benchmark_samples/textvqa_images'
 eligible={};excluded=[]
 for n,row in enumerate(rows):
  name=row['image_path'].replace('\\','/').split('/')[-1];path=ROOT.parent/'benchmark_samples/textvqa_images'/name
  if not path.exists():excluded.append({'row':n,'reason':'missing_image'});continue
  if len(row['answers'])!=10:excluded.append({'row':n,'reason':'not10_annotations'});continue
  try:
   with Image.open(path) as im:im.verify()
  except Exception:excluded.append({'row':n,'reason':'invalid_image'});continue
  h=hashlib.sha256(path.read_bytes()).hexdigest()
  if h in doc_hashes:excluded.append({'row':n,'reason':'overlap_doc_image'});continue
  if h in eligible:continue
  question=row['question']+'\nAnswer the question using a single word or phrase.';sid='textvqa_'+h[:16]
  sample={'schema':'trajkv.session.v1','dataset':'textvqa','id':sid,'session_id':sid,'source_image_id':'sha256:'+h,'image_id':'sha256:'+h,'hashes':[h],
   'images':[str(path)],'remote_images':[remote+'/'+name],'questions':[question],'turns':1,'content':[{'type':'image'},{'type':'text','text':question}],
   'protocol':'single_image_single_question','source_row_index':n,'source_original_image_id':row.get('image_id'),'historical_use':'not_exhaustively_audited'}
  eligible[h]=(sample,{'id':sid,'image_id':'sha256:'+h,'answers':row['answers']})
 ordered=[eligible[k] for k in sorted(eligible)]
 if len(ordered)<160:raise RuntimeError('Insufficient independent external images')
 outputs={}
 for split,start,count in [('dev32',0,32),('eval128',32,128)]:
  samples=[];refs=[]
  for s,r in ordered[start:start+count]:samples.append(dict(s,split='textvqa_'+split+'_prospective_20261113'));refs.append(r)
  for suffix,data in [('',samples),('_refs',refs)]:
   f=ROOT/'configs/a2'/f'textvqa_{split}{suffix}.json'
   if f.exists():raise RuntimeError('External manifest already exists; do not overwrite')
   f.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');outputs[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 result={'schema':'trajkv.external-ocr.split.v1','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'source_rows':len(rows),'eligible_unique_images':len(ordered),'excluded':excluded,'selection':'first32 and next128 eligible content hashes, first source QA per image; no model outcomes used',
  'split_hashes':outputs,'inference_started':False,'metric':'official TextVQA normalized10-annotator leave-one-out soft accuracy; ANLS only secondary if requested',
  'history_boundary':'current-cycle prospective external split; prior whole-project exposure not exhaustively audited'}
 (ROOT/'audit/a2/textvqa_prospective_split.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('source_rows','eligible_unique_images','split_hashes','inference_started')}))
if __name__=='__main__':main()
