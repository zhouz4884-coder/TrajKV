"""Stdlib CPU admission wrapper; then unchanged613d queue supervisor.

This is control code only, never an inference producer/cache numerical dependency.
No worker is killed/retried. A draft without a real pinned all-six receipt refuses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SUPERVISOR_SHA='613dfdf18d75bbe9e4ccabcd2521d80c0b8b2f380a70c8525b46bd8b1e10ecf1'


def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def actual_gone(receipt):
    p=Path('/proc')/str(receipt['pid'])
    if not p.exists():return True
    state=(p/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
    if state=='Z':return True
    actual=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
    require(actual==receipt['argv_executed'],'Receipt PID reused or process identity unknown; do not schedule')
    return False


def admit(plan):
    root=Path(plan['root']);a=plan['numeric_v2_admission']
    require(a['schema']=='baseline_matrix.numeric_v2_queue_admission.v1','Wrong controller admission')
    require(a['all_P0_receipt_sha256'] is not None,'Unbound CPU draft is not executable')
    require(sha(root/a['P2_scope'])==a['P2_scope_sha256'],'Reviewed P2-114 declaration changed')
    for relative,digest in a['source_sha256'].items():
        require(sha(root/relative)==digest,'Current reviewed dependency changed: '+relative)
    path=root/a['all_P0_receipt']
    if not path.exists():return False,'WAIT_PINNED_ALL6_RECEIPT'
    require(sha(path)==a['all_P0_receipt_sha256'],'Pinned actual P0 receipt changed')
    receipt=load(path)
    require(receipt['status']==a['all_P0_receipt_status'] and receipt['accepted_cells']==6 and receipt['accepted_arm_QA']==16 and not receipt['waiting'],'P0 not actually all accepted')
    require(receipt['gate_source_sha256']==a['P0_acceptor_sha256'],'Unknown P0 source gate')
    reports={(r['model'],r['bits']):r for r in receipt['reports']}
    require(len(reports)==6 and len(a['smokes'])==6,'Wrong actual P0 coverage')
    for row in a['smokes']:
        report=reports[(row['model'],row['bits'])]
        require(report['target_config_sha256']==row['config_sha256'] and report['actual_GPU_QA']==row['count'],'Wrong own model/bit P0 evidence')
        output=Path(row['remote_output']);require(str(output).startswith(str(root)+'/results/p0_numeric_v2/'),'Wrong P0 output root')
        m=load(output/'manifest.json');c=load(output/'complete.json')
        require(c['status']=='PASS_RAW_COMPLETE' and c['expected_QA']==c['recorded_QA']==row['count'] and c['counts']=={'PASS':row['count']},'Smoke did not truly pass')
        require(m['config']==load(root/row['config']) and sha(root/row['config'])==row['config_sha256'],'Smoke input/method configuration changed')
        for name,digest in report['source_run']['raw_hashes'].items():require(sha(output/name)==digest,'Actual P0 raw changed: '+name)
        snapshot=Path(m['runtime_source_root']);require(str(snapshot).startswith(str(root)+'/jobs/') and snapshot.name.endswith('.source'),'Wrong original source snapshot')
        launch=load(root/'jobs'/(snapshot.name[:-7]+'.launch.json'))
        require(sha(root/'jobs'/(snapshot.name[:-7]+'.launch.json'))==report['source_run']['source_proof_hashes']['launch_receipt.json'],'Actual launch proof changed')
        require(sha(snapshot/'sha256.json')==launch['source_inventory_sha256']==report['source_run']['source_proof_hashes']['source_inventory.json'],'Actual source inventory changed')
        if not actual_gone(launch):return False,'WAIT_REAL_P0_PROCESS_EXIT'
    gate=load(a['operator_gate_complete'])
    require(gate['status']==a['operator_gate_status'] and gate['device']=='cuda' and len(gate['checks'])==4 and all(r['status']=='PASS' for r in gate['checks']),'Operator CUDA gate not passed')
    # No P0 marker race may start a second model on an occupied GPU. The only
    # permitted preexisting compute process is the followed original GPU0 job.
    followed=load(root/'jobs'/(a['follow_existing']['name']+'.launch.json'))
    require(followed['pid']==a['follow_existing']['expected_pid_at_preparation'],'Followed original PID changed; never relaunch here')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],universal_newlines=True)
    ids={v[1].strip():v[0].strip() for v in (line.split(',') for line in devices.splitlines() if line.strip())}
    computes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
    for line in computes.splitlines():
        if not line.strip():continue
        uuid,pid=[v.strip() for v in line.split(',')]
        require(ids.get(uuid)=='0' and int(pid)==followed['pid'],'Unexpected active GPU process; do not start controller')
        require(not actual_gone(followed),'NVIDIA/PID observation inconsistent')
    return True,'PASS_ACTUAL_P0_AND_GPU_ADMISSION'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=load(args.plan)
    source=Path(__file__).resolve().parent/'queue_supervisor.py'
    require(sha(source)==SUPERVISOR_SHA,'Frozen supervisor613d differs')
    while True:
        ready,status=admit(plan);print(json.dumps({'status':status,'no_GPU_worker_started_by_wrapper':True}),flush=True)
        if ready:break
        time.sleep(10)
    # No alternate scheduling implementation. The frozen supervisor consumes
    # the exact same already-reviewed plan, lane nodes and active descriptor.
    import queue_supervisor
    require(sha(Path(queue_supervisor.__file__))==SUPERVISOR_SHA,'Wrong supervisor import')
    queue_supervisor.main()


if __name__=='__main__':main()
