"""Authorized transport and compact authoritative run status, no credentials printed."""
import argparse, hashlib, json, shlex, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,upload,REMOTE_ROOT,PYTHON
def execute(c,code):
 _,o,e=c.exec_command(shlex.join([PYTHON,'-c',code]),timeout=45)
 text=o.read().decode();err=e.read().decode();status=o.channel.recv_exit_status()
 if text:print(text,end='')
 if err:print(err,file=sys.stderr)
 if status:raise RuntimeError(f'remote status {status}')
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['upload','preflight','status']);p.add_argument('names',nargs='*');a=p.parse_args();c=connect()
 try:
  if a.action=='upload':
   files=[(ROOT/name).resolve() for name in a.names] if a.names else list((ROOT/'code/a2').glob('*.py'))+list((ROOT/'configs/a2').glob('*.json'))
   if not all(f.is_file() and f.is_relative_to(ROOT) for f in files):raise ValueError('Upload input must be an existing workspace file')
   receipts=[upload(c,f,REMOTE_ROOT+'/'+f.relative_to(ROOT).as_posix()) for f in files]
   checks={r['remote']:r['sha256'] for r in receipts}
   execute(c,"import json,hashlib;from pathlib import Path;e=%r;bad=[p for p,h in e.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not bad,bad;print(json.dumps({'verified_uploads':len(e)}))"%checks)
   folder=ROOT/'audit/a2';folder.mkdir(exist_ok=True);(folder/'upload_latest.json').write_text(json.dumps(receipts,indent=2));print(json.dumps({'uploaded':len(files)}))
  elif a.action=='preflight':
   code="""import json,subprocess,platform,torch,os,shutil,socket
r={'host':socket.gethostname(),'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'disk':shutil.disk_usage('/media/user/5bc0dc8d-c938-4b2a-af1b-94174fc511a9')._asdict(),'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv']).decode(),'processes':subprocess.check_output(['ps','-eo','pid,args']).decode()}
r['processes']=[x for x in r['processes'].splitlines() if 'python' in x and ('/code/' in x or 'VisContract' in x) and '-c ' not in x]
print(json.dumps(r))"""
   execute(c,code)
  else:
   execute(c,"""import json,os
from pathlib import Path
root=Path(%r)
for name in %r:
 p=root/'results'/name;r={'name':name,'exists':p.exists()}
 for f in ['complete.json','manifest.json']:
  if (p/f).exists():
   try:r[f]=json.loads((p/f).read_text()) if f=='complete.json' else True
   except Exception:r[f]='INVALID'
 if (p/'records.jsonl').exists():
  lines=(p/'records.jsonl').read_text().splitlines();r['records']=len(lines)
  if lines:
   z=json.loads(lines[-1]);r['last']={k:z.get(k) for k in ['session_id','variant','method','max_patches']}
 log=root/'logs'/(name+'.log')
 if log.exists():r['log_tail']=log.read_text(errors='replace').splitlines()[-4:]
 launch=root/'jobs'/(name+'.launch.json')
 if launch.exists():
  pid=json.loads(launch.read_text())['pid'];r['pid']=pid
  try:os.kill(pid,0);r['pid_exists']=True
  except ProcessLookupError:r['pid_exists']=False
 print(json.dumps(r))"""%(REMOTE_ROOT,a.names))
 finally:c.close()
if __name__=='__main__':main()
