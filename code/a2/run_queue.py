"""Finite experiment dependency queue; each child exits before the next starts."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(exist_ok=False,parents=True)
 (a.output/'manifest.json').write_text(json.dumps(c,indent=2))
 if c.get('wait_complete'):
  path=Path(c['wait_complete']);launch=Path(c['wait_launch'])
  pid=json.loads(launch.read_text())['pid']
  while not path.exists():
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('Dependency exited without completion: '+str(path))
   time.sleep(10)
  receipt=json.loads(path.read_text())
  if receipt.get('status') not in ('COMPLETE','PASS'):raise RuntimeError('Dependency not successful')
 for task in c['tasks']:
  argv=[sys.executable,'-u',str(Path(__file__).with_name(task['entrypoint'])),*task['arguments'],'--output',str(a.output/task['name'])]
  print(json.dumps({'starting':task['name'],'argv':argv}),flush=True)
  subprocess.run(argv,check=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','tasks':len(c['tasks'])}))
if __name__=='__main__':main()
