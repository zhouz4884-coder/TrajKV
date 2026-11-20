"""Clone the frozen reference environment without modifying it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--conda',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and not a.target.exists(), 'Preserve existing runtime/run'
    assert str(a.source.resolve()).startswith('/media/') and str(a.target.resolve()).startswith('/media/')
    a.output.mkdir(parents=True);relative=['lib/python3.10/site-packages/torch/__init__.py',
        'lib/python3.10/site-packages/transformers/__init__.py','lib/python3.10/site-packages/triton/__init__.py']
    before={name:digest(a.source/name) for name in relative}
    env=dict(os.environ);base=a.target.parents[1]/'cache'
    env['CONDA_PKGS_DIRS']=str(base/'conda_pkgs')+os.pathsep+str(a.conda.parent.parent/'pkgs')
    env['CONDA_ENVS_PATH']=str(a.target.parent);env['CONDA_NO_PLUGINS']='true'
    (base/'conda_pkgs').mkdir(parents=True,exist_ok=True);started=time.time()
    command=[str(a.conda),'create','--yes','--offline','--clone',str(a.source),'--prefix',str(a.target)]
    with (a.output/'clone.log').open('xb') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        (a.output/'failure.json').write_text(json.dumps(dict(status='FAILED_RUNTIME_CLONE',returncode=result.returncode,preserve_target=True)))
        raise RuntimeError('Conda clone failed; preserved partial target and log; no changes to source')
    after={name:digest(a.source/name) for name in relative};copied={name:digest(a.target/name) for name in relative}
    assert before==after==copied,'Reference or cloned core package differs'
    probe="import sys,json,importlib.metadata as m;print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,packages={k:m.version(k) for k in ['torch','transformers','triton','huggingface-hub','accelerate']})))"
    check=subprocess.run([str(a.target/'bin/python'),'-B','-c',probe],env=env,capture_output=True,text=True)
    assert check.returncode==0,check.stderr[-1500:]
    identity=json.loads(check.stdout);assert Path(identity['prefix']).resolve()==a.target.resolve()
    receipt=dict(status='COMPLETE',source=str(a.source),target=str(a.target),seconds=time.time()-started,
        core_source_before=before,core_source_after=after,cloned_core=copied,identity=identity,
        GPU_used=False,old_environment_modified=False)
    (a.output/'complete.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
