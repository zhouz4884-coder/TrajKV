"""Compact authoritative receipts for the active finite GPU queues."""
import json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,REMOTE_ROOT,PYTHON
def main():
 c=connect()
 try:
  code='''import json,os
from pathlib import Path
root=Path(%r)
for name in %r:
 p=root/'results'/name;r={'run':name,'complete':False}
 if (p/'complete.json').exists():r['receipt']=json.loads((p/'complete.json').read_text());r['complete']=r['receipt'].get('status') in ('COMPLETE','PASS')
 child=[]
 for f in p.rglob('records.jsonl'):
  n=sum(1 for _ in f.open());child.append({'task':f.parent.name,'records':n})
 for f in p.rglob('proxies.jsonl'):
  n=sum(1 for _ in f.open());child.append({'task':f.parent.name,'proxies':n})
 r['tasks']=child
 if (p/'guard.json').exists():
  guard=json.loads((p/'guard.json').read_text());r['guard']={key:guard.get(key) for key in ['status','timing_pid','held_parent_pid'] if key in guard}
 nested=[]
 for logfile in p.glob('*.log'):
  lines=logfile.read_text(errors='replace').splitlines();entry={'worker':logfile.stem}
  for line in lines:
   if line.startswith('{'):
    try:value=json.loads(line)
    except json.JSONDecodeError:continue
    if 'starting' in value:entry['task']=value['starting'];entry.pop('step',None)
    if 'step' in value:entry['step']={key:value.get(key) for key in ['step','kl_before_update','gradient_norm','seconds']}
  if any('Traceback' in line or 'Error:' in line for line in lines[-10:]):entry['error_tail']=lines[-4:]
  nested.append(entry)
 if nested:r['nested_workers']=nested
 launch=root/'jobs'/(name+'.launch.json')
 if launch.exists():
  pid=json.loads(launch.read_text())['pid']
  try:os.kill(pid,0);r['pid_exists']=True
  except ProcessLookupError:r['pid_exists']=False
 log=root/'logs'/(name+'.log')
 if log.exists():
  lines=log.read_text(errors='replace').splitlines()
  steps=[]
  for line in lines:
   if line.startswith('{'):
    try:entry=json.loads(line)
    except json.JSONDecodeError:continue
    if 'starting' in entry:r['current_task']=entry['starting'];steps=[]
    if 'step' in entry and 'gradient_norm' in entry:steps.append(entry)
  if steps:r['calibration_progress']={'completed_steps':len(steps),'last':steps[-1]}
  if not r['complete']:r['error_tail']=lines[-3:] if any('Traceback' in x or 'Error:' in x for x in lines[-8:]) else []
 print(json.dumps(r))
'''%(REMOTE_ROOT,sys.argv[1:])
  _,out,err=c.exec_command(shlex.join([PYTHON,'-c',code]),timeout=60);text=out.read().decode();error=err.read().decode()
  print(text,end='')
  if out.channel.recv_exit_status():raise RuntimeError(error)
 finally:c.close()
if __name__=='__main__':main()
