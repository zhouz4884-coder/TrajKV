"""Upload a command manifest and launch a detached, PID-recorded remote worker."""
import argparse,json,shlex,sys
from pathlib import Path
from remote import connect,mkdirs,REMOTE_ROOT,PYTHON

def main():
 p=argparse.ArgumentParser();p.add_argument('job',type=Path);a=p.parse_args()
 job=json.loads(a.job.read_text());name=job['name']
 if not name.replace('_','').replace('-','').isalnum():raise ValueError('Simple job name required')
 c=connect();s=c.open_sftp()
 try:
  mkdirs(s,REMOTE_ROOT+'/jobs');mkdirs(s,REMOTE_ROOT+'/logs')
  remote=REMOTE_ROOT+'/jobs/'+name+'.json'
  with s.file(remote,'w') as f:f.write(json.dumps(job))
  code="""import json,os,subprocess,shutil,hashlib
from pathlib import Path
p=Path(__import__('sys').argv[1]);j=json.loads(p.read_text());e=os.environ.copy();e.update(j.get('env',{}))
log=p.parents[1]/'logs'/(j['name']+'.log')
snapshot=p.parent/(j['name']+'.source');snapshot.mkdir(exist_ok=False)
hashes={}
for source in (p.parents[1]/'code').rglob('*.py'):
 if '__pycache__' in source.parts:continue
 relative=source.relative_to(p.parents[1]);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(source,target);hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
(snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2))
ports=p.parents[1]/'third_party/reference_ports'
if ports.exists():
 for source in ports.glob('*.py'):
  target=snapshot/'third_party/reference_ports'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
for rel in ['vllm/model_executor/layers/quantization/kvarn/sinkhorn.py','vllm/v1/attention/ops/kvarn_store.py','vllm/v1/attention/ops/kvarn_decode.py']:
 source=p.parents[1]/'third_party/KVarN'/rel
 if source.exists():
  target=snapshot/'third_party/KVarN'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
argv=list(j['argv'])
for index,value in enumerate(argv):
 if value.startswith(str(p.parents[1]/'code')+'/') and value.endswith('.py'):
  argv[index]=str(snapshot/Path(value).relative_to(p.parents[1]))
with log.open('ab',buffering=0) as out,open(os.devnull,'rb') as inp:
 w=subprocess.Popen(argv,cwd=j.get('cwd',str(p.parents[1])),env=e,stdin=inp,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
receipt={'pid':w.pid,'name':j['name'],'log':str(log),'job':str(p),'source_snapshot':str(snapshot),'immutable_code_execution':True,'argv_executed':argv}
p.with_suffix('.launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt),flush=True)
"""
  _,out,err=c.exec_command(shlex.join([PYTHON,'-c',code,remote]),timeout=25)
  response=out.read().decode()
  if response.strip():
   receipt=json.loads(response)
   print(json.dumps({k:receipt.get(k) for k in ('pid','name','log','source_snapshot','immutable_code_execution')}))
  error=err.read().decode()
  if error:print(error,file=sys.stderr)
 finally:s.close();c.close()

if __name__=='__main__':main()
