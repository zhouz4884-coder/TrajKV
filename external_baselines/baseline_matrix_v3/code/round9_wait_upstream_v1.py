"""Wait for an unchanged original500 lane, then run a new ordinary GPU lane.

CPU-only prerequisite wait. It never interrupts/restarts an upstream process,
edits its plan/state, or acquires its GPU lease before it has finished.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream-name',required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['root'])
    upstream=root/'results/queues'/a.upstream_name
    previous=None
    while True:
        state=json.loads((upstream/'state.json').read_text())
        status=state['status']
        if status!=previous:
            print(json.dumps({'phase':'WAIT_UNCHANGED_ORIGINAL500','upstream':a.upstream_name,'status':status}),flush=True)
            previous=status
        if status!='RUNNING':
            assert status=='PASS_QUEUE_COMPLETE','Original500 lane failed; do not silently skip it'
            complete=json.loads((upstream/'complete.json').read_text())
            assert complete['status']==status and all(x['status']=='COMPLETE' for x in complete['lanes'].values())
            break
        time.sleep(10)
    source=Path(__file__).resolve().parent/'queue_supervisor.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['supervisor_source_sha256']
    # The supervisor independently takes the same flock and fails if another
    # owner exists. Waiting here does not bypass or replace that lease.
    for attempt in range(6):
        receipt=root/'jobs'/('bm_'+a.upstream_name+'.launch.json')
        if not receipt.exists():break
        value=json.loads(receipt.read_text());pidpath=Path('/proc')/str(value['pid'])
        if not pidpath.exists():break
        actual=[x.decode() for x in (pidpath/'cmdline').read_bytes().split(b'\0') if x]
        if actual!=value['argv_executed']:break
        time.sleep(2)
    print(json.dumps({'phase':'START_NEW_LANE_AFTER_ORIGINAL500','output':str(a.output)}),flush=True)
    result=subprocess.run([sys.executable,'-B',str(source),'--plan',str(a.plan),'--output',str(a.output)])
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
