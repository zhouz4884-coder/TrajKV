"""Content-addressed external images; preserve existing historical files."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,upload,REMOTE_ROOT
def main():
 files=[ROOT/'configs/a2'/f'textvqa_{s}.json' for s in ('dev32','eval128')];rows=[];images={};old_hashes={}
 for f in files:
  old_hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest();xs=json.loads(f.read_text(encoding='utf-8'));rows.append((f,xs))
  for r in xs:
   local=Path(r['images'][0]);assert hashlib.sha256(local.read_bytes()).hexdigest()==r['hashes'][0]
   r['remote_images']=[REMOTE_ROOT+'/data/textvqa_prospective/images/'+r['hashes'][0]+local.suffix]
   images[r['remote_images'][0]]=local
 c=connect()
 try:
  receipts=[upload(c,p,remote) for remote,p in images.items()]
  for f,xs in rows:f.write_text(json.dumps(xs,indent=2,ensure_ascii=False),encoding='utf-8')
 finally:c.close()
 audit=json.loads((ROOT/'audit/a2/textvqa_prospective_split.json').read_text());audit['path_revision_before_inference']={'old_manifest_hashes':old_hashes,
  'reason':'historical remote image filenames use a different subset/order; fresh content-addressed image copies, membership unchanged',
  'new_manifest_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'images':len(receipts)}
 for f in files:audit['split_hashes'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 (ROOT/'audit/a2/textvqa_prospective_split.json').write_text(json.dumps(audit,indent=2));(ROOT/'audit/a2/textvqa_uploads.json').write_text(json.dumps(receipts,indent=2))
 print(json.dumps({'uploaded_images':len(receipts),'historical_files_overwritten':False,'membership_changed':False}))
if __name__=='__main__':main()
