"""Read-only full model/image identity against frozen project hashes."""
import hashlib,json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,PYTHON
def main():
 expected=json.loads((ROOT/'audit/model_identity.json').read_text())
 rows=sum([json.loads((ROOT/'configs/a2'/(name+'.json')).read_text(encoding='utf-8')) for name in ('dev8','confirm64')],[])
 images={r['remote_images'][0]:r['hashes'][0] for r in rows}
 payload={'root':expected['model_root'],'files':{r['name']:r['sha256'] for r in expected['files']},'images':images}
 script="""import json,hashlib,platform,subprocess,importlib.metadata as md
from pathlib import Path
x=json.loads(__import__('sys').argv[1]);out={'model':[],'images':[],'runtime':{k:md.version(k) for k in ['torch','transformers','triton','safetensors','numpy','Pillow']}}
def check(path,expected):
 p=Path(path);a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 z=p.stat();v=h.hexdigest();return {'path':str(p),'bytes':a.st_size,'sha256':v,'matches':v==expected,'stable':(a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)}
for n,h in x['files'].items():out['model'].append(check(str(Path(x['root'])/n),h))
for n,h in x['images'].items():out['images'].append(check(n,h))
out['status']='PASS' if all(z['matches'] and z['stable'] for k in ['model','images'] for z in out[k]) else 'FAIL'
print(json.dumps(out))"""
 c=connect()
 try:
  _,o,e=c.exec_command(shlex.join([PYTHON,'-c',script,json.dumps(payload)]),timeout=120)
  result=json.loads(o.read().decode());err=e.read().decode();rc=o.channel.recv_exit_status()
  if err:print(err,file=sys.stderr)
  if rc:raise RuntimeError('Remote identity verification failed')
 finally:c.close()
 (ROOT/'audit/a2/assets_identity.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'status':result['status'],'model_files':len(result['model']),'images':len(result['images']),'runtime':result['runtime']}))
 if result['status']!='PASS':raise RuntimeError('Assets differ from frozen hashes')
if __name__=='__main__':main()
