"""Read completed GPU calibration artifacts with independent remote byte anchors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from remote_control import REMOTE_ROOT, connect

INSPECT = r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);name=sys.argv[2];full=sys.argv[3]=='full'
out=root/'results/round10_v1'/name;cp=out/'complete.json'
if not cp.exists():
 print(json.dumps({'status':'WAIT_CALIBRATION_NOT_CLOSED'}));raise SystemExit(0)
def h(p):
 q=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):q.update(b)
 return q.hexdigest()
closed=json.loads(cp.read_text());assert closed['actual_GPU'] is True and closed['references_read'] is False
assert closed['status'] in ('PASS_LS48_AND_ACTUAL_MODEL_SELECT16','PASS_TECHNICAL_ONE_SOURCE_LAMBDA0_W1_ACTUAL_MODEL')
job=root/'jobs'/('bm_R10_'+name+'.json');launch=root/'jobs'/('bm_R10_'+name+'.launch.json')
j=json.loads(job.read_text());l=json.loads(launch.read_text());snapshot=Path(l['source_snapshot'])
assert l['job_sha256']==h(job) and snapshot==root/'jobs'/('bm_R10_'+name+'.source')
argv=list(j['argv'])
for i,v in enumerate(argv):
 if isinstance(v,str) and v.startswith(str(root)+'/code/') and v.endswith('.py'):
  argv[i]=str(snapshot/Path(v).relative_to(root))
assert l['argv_executed']==argv and argv[argv.index('--output')+1]==str(out)
config=Path(argv[argv.index('--config')+1]);cfg=json.loads(config.read_text())
manifest=json.loads((out/'manifest.json').read_text())
# This calibration schema embeds the complete config; it does not declare
# the inference runner's separate config_sha256 field.
assert manifest['config']==dict(cfg,calibration_config_sha256=h(config))
assert h(snapshot/'sha256.json')==l['source_inventory_sha256']
inventory=json.loads((snapshot/'sha256.json').read_text())
for rel,digest in closed['source_hashes'].items():
 assert rel.startswith('code/') and '..' not in Path(rel).parts and inventory[rel]==digest==h(snapshot/rel)
assert cfg['cal_fit_inputs'].endswith('cal_fit48_inputs.jsonl') and cfg['cal_select_inputs'].endswith('cal_select16_inputs.jsonl')
assert h(Path(cfg['cal_fit_inputs']))==cfg['cal_fit_sha256'] and h(Path(cfg['cal_select_inputs']))==cfg['cal_select_sha256']
items=[]
def add(label,p):
 assert p.is_file()
 items.append({'file':label,'remote_file':str(p),'bytes':p.stat().st_size,'sha256':h(p)})
for n in ['manifest.json','complete.json','value_theta_selected.json','window_selected.json','fit_collection_complete.json','value_theta_LS_proposal.json']:
 if (out/n).exists():add(n,out/n)
if closed['status']=='PASS_LS48_AND_ACTUAL_MODEL_SELECT16':
 assert closed['fit_sources']==48 and closed['select_sources']==16
 assert h(out/'value_theta_selected.json')==closed['theta_sha256']
 assert h(out/'window_selected.json')==closed['window_sha256']
 if full:
  assert h(out/'records.jsonl')==closed['records_sha256'];add('records.jsonl',out/'records.jsonl')
for label,p in [('source_config.json',config),('source_job.json',job),('launch_receipt.json',launch),('source_inventory.json',snapshot/'sha256.json')]:add(label,p)
if full:
 for rel in closed['source_hashes']:add('source_files/'+rel,snapshot/rel)
print(json.dumps({'status':'PASS_REMOTE_CALIBRATION_SOURCE_AND_RAW_SHA','scope':'calibration_not_task_quality',
 'remote_hashes_computed_independently':True,
 'files':items,'source_snapshot':str(snapshot),'closed_status':closed['status'],'launch_name':l['name']}))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full',action='store_true')
    a=p.parse_args()
    assert re.fullmatch(r'cal(?:tech|64)_v[0-9]+',a.run)
    c=connect()
    def inspect():
        _,out,err=c.exec_command(shlex.join(['/usr/bin/python3','-B','-c',INSPECT,REMOTE_ROOT,a.run,'full' if a.full else 'minimal']),timeout=60)
        b,e=out.read(),err.read()
        assert out.channel.recv_exit_status()==0,e.decode(errors='replace')[-2000:]
        return json.loads(b)
    try:
        proof=inspect()
        if proof['status'].startswith('WAIT'):
            print(json.dumps(proof));return
        a.output.mkdir(parents=True,exist_ok=True)
        s=c.open_sftp()
        try:
            for item in proof['files']:
                rel=PurePosixPath(item['file']);assert not rel.is_absolute() and '..' not in rel.parts
                target=a.output.joinpath(*rel.parts)
                if target.exists():
                    assert target.stat().st_size==item['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'];continue
                with s.open(item['remote_file'],'rb') as f:b=f.read()
                assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(b)
        finally:s.close()
        assert inspect()==proof,'Remote closed artifacts changed during retrieval'
        receipt=a.output/('remote_calibration_retrieval_full_v1.json' if a.full else 'remote_calibration_retrieval_minimal_v1.json')
        b=(json.dumps(proof,indent=2)+'\n').encode()
        if receipt.exists():assert receipt.read_bytes()==b
        else:
            with receipt.open('xb') as f:f.write(b)
        print(json.dumps({'status':proof['status'],'closed_status':proof['closed_status'],'files':len(proof['files']),'output':str(a.output)}))
    finally:c.close()


if __name__=='__main__':main()
