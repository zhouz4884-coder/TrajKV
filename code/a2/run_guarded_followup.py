"""One finite follow-up before an idle owned timing waiter; always release it."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();c=json.loads(a.config.read_text())
 timing=json.loads(Path(c['timing_launch']).read_text());pid=timing['pid'];proc=Path('/proc')/str(pid)
 command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert timing['source_snapshot'] in command and 'run_after_dependencies.py' in command
 assert not (proc/'task'/str(pid)/'children').read_text().strip() and not Path(c['timing_complete']).exists()
 a.output.mkdir(parents=True,exist_ok=False);guard={'status':'WAITING_FOR_QUALITY','timing_pid':pid,'started_unix':time.time(),
  'scope':'only our idle timing dependency waiter held; no active measurements stopped; original timing configs/methods unchanged'}
 def interrupted(signum,frame):raise RuntimeError('Follow-up interrupted; releasing timing waiter')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 os.kill(pid,signal.SIGSTOP)
 try:
  (a.output/'guard.json').write_text(json.dumps(guard,indent=2));started=time.monotonic()
  while True:
   done=True
   for dep in c['dependencies']:
    path=Path(dep['complete'])
    if path.exists():assert json.loads(path.read_text())['status']=='COMPLETE';continue
    done=False;other=json.loads(Path(dep['launch']).read_text())['pid'];os.kill(other,0)
   if done:break
   if time.monotonic()-started>21600:raise TimeoutError('Finite quality wait exceeded')
   time.sleep(10)
  time.sleep(3);guard['status']='RUNNING_BOUNDED_DIAGNOSTIC';(a.output/'guard.json').write_text(json.dumps(guard,indent=2))
  argv=[sys.executable,'-u',str(Path(__file__).with_name('run_variants.py')),'--config',c['quality_config'],'--output',str(a.output/'quality')]
  subprocess.run(argv,check=True);receipt=json.loads((a.output/'quality/complete.json').read_text());assert receipt['status']=='COMPLETE'
  (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','scope':'fixedK2 V-precision diagnostic only; no new method selection or frozen benchmark modification','records':receipt['records']}))
 finally:
  try:os.kill(pid,signal.SIGCONT);guard['status']='RELEASED'
  except ProcessLookupError:guard['status']='TIMING_PROCESS_GONE'
  guard['finished_unix']=time.time();(a.output/'guard.json').write_text(json.dumps(guard,indent=2))

if __name__=='__main__':main()
