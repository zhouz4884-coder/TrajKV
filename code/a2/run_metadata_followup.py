"""Finite two-card research extension; release the owned upstream wait chain."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text())
 parent=json.loads(Path(c['held_parent_launch']).read_text());held=parent['pid'];proc=Path('/proc')/str(held)
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert parent['source_snapshot'] in cmd and 'run_guarded_followup.py' in cmd
 prior=json.loads(Path(c['held_parent_guard']).read_text());assert prior['status']=='WAITING_FOR_QUALITY' and not (proc/'task'/str(held)/'children').read_text().strip()
 a.output.mkdir(exist_ok=False,parents=True);guard={'status':'WAITING_FOR_QUALITY','held_parent_pid':held,'timing_pid':prior['timing_pid'],'started_unix':time.time(),
  'scope':'owned idle follow-up waiter held, which holds original idle timing waiter; no active GPU worker stopped; finally releases chain'}
 active=[];logs=[]
 def interrupted(signum,frame):raise RuntimeError('Metadata extension interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 os.kill(held,signal.SIGSTOP)
 try:
  time.sleep(.1);assert not (proc/'task'/str(held)/'children').read_text().strip()
  (a.output/'guard.json').write_text(json.dumps(guard,indent=2));started=time.monotonic()
  while True:
   done=True
   for dep in c['dependencies']:
    path=Path(dep['complete'])
    if path.exists():assert json.loads(path.read_text())['status']=='COMPLETE';continue
    done=False;os.kill(json.loads(Path(dep['launch']).read_text())['pid'],0)
   if done:break
   if time.monotonic()-started>21600:raise TimeoutError('Metadata finite quality wait expired')
   time.sleep(10)
  time.sleep(3)
  for phase in c['phases']:
   guard.update(status='RUNNING_'+phase['name'].upper());(a.output/'guard.json').write_text(json.dumps(guard,indent=2));active=[];logs=[]
   print(json.dumps({'phase':phase['name'],'workers':2}),flush=True)
   for shard,task in enumerate(phase['workers']):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard));folder=a.output/task['name'];log=(a.output/(task['name']+'.log')).open('w');logs.append(log)
    argv=[sys.executable,'-u',str(Path(__file__).with_name(task['entrypoint'])),*task['arguments'],'--output',str(folder)]
    active.append(subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
   for child in active:
    if child.wait()!=0:raise RuntimeError('Metadata worker failed; see phase logs')
   for log in logs:log.close()
   for task in phase['workers']:assert json.loads((a.output/task['name']/'complete.json').read_text())['status'] in ['COMPLETE','PASS']
   active=[];logs=[]
  (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','phases':len(c['phases']),'scope':'new offline metadata calibration and open development only; original frozen confirmation/benchmark unchanged'}))
 finally:
  for child in active:
   if child.poll() is None:
    assert os.getpgid(child.pid)==child.pid
    child_cmd=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(Path(__file__).parent) in child_cmd
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  for log in logs:log.close()
  try:os.kill(held,signal.SIGCONT);guard['status']='RELEASED'
  except ProcessLookupError:guard['status']='HELD_PARENT_GONE'
  guard['finished_unix']=time.time();(a.output/'guard.json').write_text(json.dumps(guard,indent=2))

if __name__=='__main__':main()
