"""Check the prospective TextVQA image bytes before any GPU evaluation."""
import hashlib,json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,PYTHON
def main():
 rows=sum([json.loads((ROOT/'configs/a2'/f'textvqa_{s}.json').read_text(encoding='utf-8')) for s in ('dev32','eval128')],[])
 expected={r['remote_images'][0]:r['hashes'][0] for r in rows}
 code='''import json,sys,hashlib
from pathlib import Path
rows=json.loads(sys.argv[1]);out=[]
for name,h in rows.items():
 p=Path(name);v=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 out.append({'path':name,'matches':h==v,'sha256':v})
print(json.dumps({'status':'PASS' if all(r['matches'] for r in out) else 'FAIL','images':out}))
'''
 c=connect()
 try:
  _,o,e=c.exec_command(shlex.join([PYTHON,'-c',code,json.dumps(expected)]),timeout=60);out=json.loads(o.read().decode());err=e.read().decode()
  if o.channel.recv_exit_status():raise RuntimeError(err)
 finally:c.close()
 (ROOT/'audit/a2/textvqa_asset_identity.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':out['status'],'images':len(out['images'])}))
 if out['status']!='PASS':raise RuntimeError('External images unavailable or changed')
if __name__=='__main__':main()
