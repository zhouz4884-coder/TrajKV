"""Run one finite child only after all declared experiment dependencies pass."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 cfg=json.loads(args.config.read_text());print(json.dumps({'waiting_dependencies':len(cfg['dependencies'])}),flush=True)
 start=time.monotonic()
 while True:
  done=True
  for dependency in cfg['dependencies']:
   receipt=Path(dependency['complete']);launch=Path(dependency['launch'])
   if receipt.exists():
    if json.loads(receipt.read_text()).get('status') not in ('COMPLETE','PASS'):raise RuntimeError('Dependency receipt not successful')
    continue
   done=False;pid=json.loads(launch.read_text())['pid']
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('Dependency exited without success: '+str(receipt))
  if done:break
  if time.monotonic()-start>cfg.get('timeout_seconds',21600):raise TimeoutError('Finite dependency wait expired')
  time.sleep(10)
 settle=float(cfg.get('settle_seconds',0));assert 0<=settle<=60
 if settle:time.sleep(settle)
 argv=[sys.executable,'-u',str(Path(__file__).with_name(cfg['entrypoint'])),*cfg['arguments'],'--output',str(args.output)]
 print(json.dumps({'starting_child':cfg['entrypoint']}),flush=True);subprocess.run(argv,check=True)
if __name__=='__main__':main()
