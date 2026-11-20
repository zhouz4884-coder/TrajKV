"""Independent venv overlay with explicitly shared, frozen base dependencies.

No base packages are installed, removed, edited or silently upgraded. The
overlay is a declared runtime type, not a claim of a complete Conda clone.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


PROBE="""import json,sys,hashlib,importlib.metadata as m,importlib.util as u
names=['torch','transformers','triton','numpy','huggingface_hub','accelerate','PIL','pyarrow']
packages={}
for name in names:
 spec=u.find_spec(name)
 if spec and spec.origin:
  p=__import__('pathlib').Path(spec.origin);packages[name]=dict(origin=str(p.resolve()),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for name,dist in [('torch','torch'),('transformers','transformers'),('triton','triton'),('numpy','numpy'),('huggingface_hub','huggingface-hub'),('accelerate','accelerate'),('PIL','pillow'),('pyarrow','pyarrow')]:
 if name in packages:packages[name]['version']=m.version(dist)
print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,python=sys.executable,packages=packages)))"""


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert str(a.target.resolve()).startswith('/media/') and not a.target.exists() and not a.output.exists()
    a.output.mkdir(parents=True)
    def probe(exe):
        result=subprocess.run([str(exe),'-B','-c',PROBE],capture_output=True,text=True,env=os.environ.copy())
        if result.returncode:raise RuntimeError(result.stderr[-1200:])
        return json.loads(result.stdout)
    before=probe(sys.executable)
    venv.EnvBuilder(system_site_packages=True,with_pip=False,symlinks=True).create(a.target)
    after=probe(sys.executable);overlay=probe(a.target/'bin/python')
    assert before==after and overlay['packages']==before['packages']
    assert Path(overlay['prefix']).resolve()==a.target.resolve()
    assert Path(overlay['base_prefix']).resolve()==Path(before['prefix']).resolve()
    report=dict(status='COMPLETE',runtime_type='isolated_venv_overlay_with_readonly_frozen_base',
        before=before,base_after=after,overlay=overlay,base_dependencies_changed=False,
        project_packages_must_install_only_inside_overlay=True,
        scope='All benchmark workers use this exact overlay. Shared dependency origins and SHA are explicit; no full Conda clone claim.',GPU_used=False)
    (a.output/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',runtime_type=report['runtime_type'],worker_python=overlay['python'],packages={k:v['version'] for k,v in overlay['packages'].items()})),flush=True)


if __name__=='__main__':main()
