"""Cancel only a named dependency-waiting queue, before it has any child."""
import argparse,json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,REMOTE_ROOT,PYTHON
def main():
 p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
 code='''import json,os,signal,sys
from pathlib import Path
root=Path(sys.argv[1]);name=sys.argv[2];launch=json.loads((root/'jobs'/(name+'.launch.json')).read_text());pid=launch['pid']
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\\x00',' ')
if name not in cmd or 'run_queue.py' not in cmd:raise RuntimeError('Process identity changed')
children=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
if children:raise RuntimeError('Queue has a running child; do not cancel with this helper')
if any((root/'results'/name).rglob('records.jsonl')):raise RuntimeError('Inference task has started')
os.kill(pid,signal.SIGTERM)
receipt={'status':'CANCELLED_BEFORE_CHILD','reason':'external image identity repair, no inference ran','pid':pid}
(root/'results'/name/'cancelled.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
 c=connect()
 try:
  _,o,e=c.exec_command(shlex.join([PYTHON,'-c',code,REMOTE_ROOT,a.name]),timeout=30);out=o.read().decode();err=e.read().decode()
  print(out,end='')
  if o.channel.recv_exit_status():raise RuntimeError(err)
 finally:c.close()
if __name__=='__main__':main()
