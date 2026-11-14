"""Use an idle card for one small model probe, then release the timing guard."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
 timing=json.loads(Path(cfg['timing_launch']).read_text());pid=timing['pid'];command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert timing['source_snapshot'] in command and 'run_after_dependencies.py' in command
 children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().strip()
 if children or Path(cfg['timing_complete']).exists():raise RuntimeError('Timing has started; do not insert a model probe')
 os.kill(pid,signal.SIGSTOP);started=time.time();a.output.mkdir(exist_ok=False,parents=True)
 guard={'timing_pid':pid,'status':'GUARDED_BEFORE_MEASUREMENTS','started_unix':started,'scope':'only our idle dependency waiter paused, no active timing child'}
 (a.output/'guard.json').write_text(json.dumps(guard,indent=2))
 try:
  subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('calibrate_direction_books.py')),'--config',cfg['probe_config'],'--output',str(a.output/'gradient')],check=True)
  (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','scope':'4case2step full-model feasibility only, not task accuracy'}))
 finally:
  os.kill(pid,signal.SIGCONT);guard.update(status='RELEASED',finished_unix=time.time());(a.output/'guard.json').write_text(json.dumps(guard,indent=2))
if __name__=='__main__':main()
