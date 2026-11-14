"""Read-only, shallow inventory for a future independent validation pool."""
import json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import connect,PYTHON
def main():
 code='''import json
from pathlib import Path
disk=Path('/media/user/5bc0dc8d-c938-4b2a-af1b-94174fc511a9')
paths=[disk/'models',disk/'datasets',Path('/home/RESEARCHER/.cache/huggingface/datasets'),disk/'VLAtest/gsm8k_server_bundle/benchmark_samples']
for p in paths:
 r={'path':str(p),'exists':p.exists()}
 if p.is_dir():r['entries']=[{'name':x.name,'directory':x.is_dir(),'bytes':x.stat().st_size if x.is_file() else None} for x in sorted(p.iterdir())[:60]]
 print(json.dumps(r))
'''
 c=connect()
 try:
  _,out,err=c.exec_command(shlex.join([PYTHON,'-c',code]),timeout=30);value=out.read().decode();error=err.read().decode()
  if out.channel.recv_exit_status():raise RuntimeError(error)
  rows=[json.loads(s) for s in value.splitlines()];dest=ROOT/'audit/a2/next_asset_inventory.json';dest.write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
 finally:c.close()
if __name__=='__main__':main()
